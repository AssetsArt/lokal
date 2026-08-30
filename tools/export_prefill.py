# export_prefill.py — export the model's prefill half as a Core ML graph so it can run
# on the Apple Neural Engine.
#
# Why a Python file exists in a Rust project: the ANE has no direct programming API —
# the only road in is Core ML, and the model-authoring toolchain (coremltools) is
# Python-only. The graph is built here once; the Rust side (src/ane.rs) just loads
# and invokes the compiled .mlmodelc.
#
# The exported graph = the prefill half of Model::forward (model.rs) for S tokens at once:
#   ids [1,S] → embedding → Block × N (rmsnorm → full causal attention → SwiGLU)
#   → returns K,V for every layer and position (no lm_head — the Rust side lets Metal
#   compute the final token's logits, which is cheaper than shipping a vocab-sized
#   matmul through Core ML)
#
# Deliberate constraint: fixed shapes (prompts are zero-padded up to the next
# available size). The ANE wants static graphs, and the causal mask guarantees pad
# positions cannot affect the K,V of the real positions before them. Each shape in
# --shapes becomes its own prefill-<S>.mlmodelc; the Rust side picks the smallest
# one that fits the prompt.
#
# Why separate files rather than one enumerated-shape model: tried and measured —
# ct.EnumeratedShapes on this graph makes ANECCompile fail (the runtime silently
# falls back) and the compiled model OOMs a 16 GB machine at load time. Fixed
# shapes compile clean and load light; the cost is one weight copy on disk per
# shape, which is why the default ladder is short.
#
# Usage:
#   uv run --python 3.12 --with torch --with coremltools --with safetensors \
#       tools/export_prefill.py <model-dir> --shapes 512,2048

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate_half(x, h):  # same convention as HF Llama and rope() in model.rs
    # h = head_dim//2 is passed as a Python int — reading it from x.shape during
    # tracing would become a tensor op (floor_divide) that coremltools can't convert.
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class PrefillNet(torch.nn.Module):
    """The prefill half of a Llama-family transformer — weights registered as raw
    buffers rather than nn.Linear, so each op maps 1:1 onto a line of model.rs.

    Every op is shape-polymorphic in S (no Python ints derived from tensor shapes,
    reshapes use -1, RoPE tables are gathered by position, the causal mask is built
    from positions) so one traced graph serves every enumerated sequence length."""

    def __init__(self, cfg, weights, max_seq):
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg["num_hidden_layers"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.hd = cfg["hidden_size"] // self.n_heads
        self.eps = cfg["rms_norm_eps"]
        for name, t in weights.items():
            self.register_buffer(name.replace(".", "_"), t.float(), persistent=False)

        # Precomputed RoPE cos/sin tables (positions 0..max_seq-1); the forward pass
        # gathers the first S rows by position id, so the table itself stays constant.
        theta = cfg.get("rope_theta", 10000.0)
        inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        ang = torch.outer(torch.arange(max_seq).float(), inv)  # [max_seq, hd/2]
        emb = torch.cat((ang, ang), dim=-1)                    # [max_seq, hd] (HF-style)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)

    def w(self, layer, name):
        return getattr(self, f"model_layers_{layer}_{name}".replace(".", "_"))

    def forward(self, ids):  # ids: int32 [1, S] for any enumerated S
        hd = self.hd
        # Positions 0..S-1 derived from the input itself (shape-polymorphic).
        pos = torch.cumsum(torch.ones_like(ids, dtype=torch.int32), dim=1) - 1  # [1,S]
        cos = F.embedding(pos.squeeze(0).to(torch.long), self.rope_cos)  # [S, hd]
        sin = F.embedding(pos.squeeze(0).to(torch.long), self.rope_sin)
        # Causal mask from positions: -3e4 (safely below any fp16 score) where j > i.
        causal = (pos.unsqueeze(1) > pos.unsqueeze(2)).float() * -30000.0  # [1,S,S]
        causal = causal.unsqueeze(1)  # [1,1,S,S], broadcast over heads

        x = F.embedding(ids.to(torch.long), self.model_embed_tokens_weight)  # [1,S,H]
        k_out, v_out = [], []
        for li in range(self.n_layers):
            xn = rmsnorm(x, self.w(li, "input_layernorm.weight"), self.eps)
            q = F.linear(xn, self.w(li, "self_attn.q_proj.weight"),
                         self.opt_bias(li, "q_proj"))
            k = F.linear(xn, self.w(li, "self_attn.k_proj.weight"),
                         self.opt_bias(li, "k_proj"))
            v = F.linear(xn, self.w(li, "self_attn.v_proj.weight"),
                         self.opt_bias(li, "v_proj"))
            q = q.view(1, -1, self.n_heads, hd).transpose(1, 2)  # [1,heads,S,hd]
            k = k.view(1, -1, self.n_kv, hd).transpose(1, 2)
            v = v.view(1, -1, self.n_kv, hd).transpose(1, 2)

            q = q * cos + rotate_half(q, hd // 2) * sin
            k = k * cos + rotate_half(k, hd // 2) * sin

            # Capture K,V (post-RoPE, pre-GQA-expansion) — laid out [S, kv*hd] to match
            # the Rust cache layout exactly.
            k_out.append(k.transpose(1, 2).reshape(-1, self.n_kv * hd))
            v_out.append(v.transpose(1, 2).reshape(-1, self.n_kv * hd))

            # GQA: expand kv heads to match the query heads, then full causal attention.
            # (expand+reshape instead of repeat_interleave — coremltools converts it cleanly.)
            rep = self.n_heads // self.n_kv
            kf = k.unsqueeze(2).expand(1, self.n_kv, rep, -1, hd).reshape(1, self.n_heads, -1, hd)
            vf = v.unsqueeze(2).expand(1, self.n_kv, rep, -1, hd).reshape(1, self.n_heads, -1, hd)
            att = (q @ kf.transpose(-1, -2)) * (hd ** -0.5) + causal
            o = (F.softmax(att, dim=-1) @ vf).transpose(1, 2).reshape(1, -1, self.n_heads * hd)
            x = x + F.linear(o, self.w(li, "self_attn.o_proj.weight"))

            xn = rmsnorm(x, self.w(li, "post_attention_layernorm.weight"), self.eps)
            gate = F.linear(xn, self.w(li, "mlp.gate_proj.weight"))
            up = F.linear(xn, self.w(li, "mlp.up_proj.weight"))
            x = x + F.linear(F.silu(gate) * up, self.w(li, "mlp.down_proj.weight"))

        return torch.stack(k_out), torch.stack(v_out)  # [layers, S, kv*hd]

    def opt_bias(self, layer, name):  # Qwen2 has q/k/v biases; Llama does not
        attr = f"model_layers_{layer}_self_attn_{name}_bias"
        return getattr(self, attr) if hasattr(self, attr) else None


class WindowedNet(torch.nn.Module):
    """One prefill chunk of S tokens attending to up to P past positions, fed in as
    K/V inputs (validity-masked, zero-padded). This is what lets long prompts run
    through the ANE in chunks: the Rust side accumulates each chunk's K/V and feeds
    it back as the next chunk's past. All shapes are static.

    Measured (2026-08-29, Qwen2.5-0.5B-Instruct, natural text): the fp16 path holds
    its numeric envelope through P + S = 8,192 — max |Δ| vs torch f32 was 0.54/0.41
    (K/V) at 8,192 vs 0.50/0.34 for the shipped 6,144 graph. The earlier "breaks
    hard at 8,192" note predated the host-side cos/sin fix and was the misdiagnosed
    in-graph fp16 position bug, not a width limit. The practical bound is the
    first-load ANECompilerService cost, which every machine pays once per graph:
    99 s at 6,144 → 250 s at 8,192 → 21+ min (unfinished, killed) at 16,384.
    """

    def __init__(self, cfg, weights, s, p, n_front=None):
        super().__init__()
        self.s = s
        self.p = p
        self.n_layers = cfg["num_hidden_layers"]
        # Layer-split variant: with n_front set, the graph runs only layers
        # 0..n_front and additionally returns the hidden state, so the remaining
        # layers can continue on another device (see src/ane.rs, split prefill).
        # k_past/v_past then carry only the front layers. n_front=None keeps the
        # full-model graph byte-for-byte as it was.
        self.n_front = self.n_layers if n_front is None else n_front
        self.emit_x = self.n_front < self.n_layers
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.hd = cfg["hidden_size"] // self.n_heads
        self.eps = cfg["rms_norm_eps"]
        for name, t in weights.items():
            self.register_buffer(name.replace(".", "_"), t.float(), persistent=False)
        # Within-block causality never depends on the absolute position — a static mask.
        causal = torch.full((s, s), -30000.0).triu(1)
        self.register_buffer("causal", causal.view(1, 1, s, s), persistent=False)

    def w(self, layer, name):
        return getattr(self, f"model_layers_{layer}_{name}".replace(".", "_"))

    def opt_bias(self, layer, name):
        attr = f"model_layers_{layer}_self_attn_{name}_bias"
        return getattr(self, attr) if hasattr(self, attr) else None

    def forward(self, ids, cos, sin, k_past=None, v_past=None, past_valid=None):
        # ids [1,S] i32 · cos/sin [S,hd] (host-computed for this chunk's absolute
        # positions — computing positions in-graph is fatal: the fp16 pipeline can't
        # represent integers above 2048 exactly, which corrupts RoPE from position
        # 2048 on) · k/v_past [L,P,kv·hd] · past_valid [1,P] (1=real)
        S, P, hd, kv = self.s, self.p, self.hd, self.n_kv
        # Mask over [past | current]: padded past is dead, current is causal-in-block.
        # P=0 is the first-chunk rung: no past inputs at all, and no past attention
        # paid — the mask is just the in-block causal one.
        if P > 0:
            past_mask = (1.0 - past_valid) * -30000.0                 # [1,P]
            full_mask = torch.cat(
                (past_mask.view(1, 1, 1, P).expand(1, 1, S, P), self.causal), dim=3
            )  # [1,1,S,P+S]
        else:
            full_mask = self.causal

        x = F.embedding(ids.to(torch.long), self.model_embed_tokens_weight)
        k_out, v_out = [], []
        rep = self.n_heads // kv
        for li in range(self.n_front):
            xn = rmsnorm(x, self.w(li, "input_layernorm.weight"), self.eps)
            q = F.linear(xn, self.w(li, "self_attn.q_proj.weight"), self.opt_bias(li, "q_proj"))
            k = F.linear(xn, self.w(li, "self_attn.k_proj.weight"), self.opt_bias(li, "k_proj"))
            v = F.linear(xn, self.w(li, "self_attn.v_proj.weight"), self.opt_bias(li, "v_proj"))
            q = q.view(1, S, self.n_heads, hd).transpose(1, 2)
            k = k.view(1, S, kv, hd).transpose(1, 2)
            v = v.view(1, S, kv, hd).transpose(1, 2)
            q = q * cos + rotate_half(q, hd // 2) * sin
            k = k * cos + rotate_half(k, hd // 2) * sin
            k_out.append(k.transpose(1, 2).reshape(S, kv * hd))
            v_out.append(v.transpose(1, 2).reshape(S, kv * hd))

            if P > 0:
                kp = k_past[li].view(1, P, kv, hd).transpose(1, 2)  # past is already post-RoPE
                vp = v_past[li].view(1, P, kv, hd).transpose(1, 2)
                k_all = torch.cat((kp, k), dim=2)
                v_all = torch.cat((vp, v), dim=2)
            else:
                k_all, v_all = k, v
            kf = k_all.unsqueeze(2).expand(1, kv, rep, P + S, hd).reshape(1, self.n_heads, P + S, hd)
            vf = v_all.unsqueeze(2).expand(1, kv, rep, P + S, hd).reshape(1, self.n_heads, P + S, hd)
            # Segmented attention with an online-softmax merge (flash-attention at
            # graph level): Core ML's fp16 path degrades hard once one softmax spans
            # more than ~2,048 real positions, so every softmax here stays ≤ SEG wide
            # and the segments merge exactly via log-sum-exp.
            SEG = 2048
            scale = hd ** -0.5
            outs, ms, ls = [], [], []
            for s0 in range(0, P + S, SEG):
                s1 = min(s0 + SEG, P + S)
                att = (q @ kf[:, :, s0:s1].transpose(-1, -2)) * scale + full_mask[:, :, :, s0:s1]
                m = att.amax(dim=-1, keepdim=True)  # [1,H,S,1]
                e = torch.exp(att - m)
                outs.append(e @ vf[:, :, s0:s1])
                ms.append(m)
                ls.append(e.sum(dim=-1, keepdim=True))
            big = ms[0]
            for m in ms[1:]:
                big = torch.maximum(big, m)
            num = sum(torch.exp(m - big) * o for m, o in zip(ms, outs))
            den = sum(torch.exp(m - big) * l for m, l in zip(ms, ls))
            o = (num / den).transpose(1, 2).reshape(1, S, -1)
            x = x + F.linear(o, self.w(li, "self_attn.o_proj.weight"))

            xn = rmsnorm(x, self.w(li, "post_attention_layernorm.weight"), self.eps)
            gate = F.linear(xn, self.w(li, "mlp.gate_proj.weight"))
            up = F.linear(xn, self.w(li, "mlp.up_proj.weight"))
            x = x + F.linear(F.silu(gate) * up, self.w(li, "mlp.down_proj.weight"))
        if self.emit_x:
            return torch.stack(k_out), torch.stack(v_out), x
        return torch.stack(k_out), torch.stack(v_out)


def export_front(args, cfg, weights, spec):
    """Export the layer-split FRONT half: a windowed chunk graph that runs layers
    0..A and returns both their K/V and the hidden state, so a second device can
    finish layers A..L for the same chunk. This is what makes GPU and ANE work on
    the prompt at the same time — the ANE runs chunk c's front half while Metal
    runs chunk c-1's back half (src/ane.rs, LOKAL_SPLIT_PREFILL).

    Export and first-load cost are first-class numbers here, not footnotes: every
    new graph shape is a fresh ANECCompilerService pass on each machine, and that
    pass is silent and slow (99 s at width 6,144, 250 s at 8,192). Both are timed
    and printed."""
    import time

    import coremltools as ct

    parts = [int(v) for v in spec.split("x")]
    fs, fp = parts[0], parts[1]
    L = cfg["num_hidden_layers"]
    A = parts[2] if len(parts) > 2 else (args.front_layers or L // 2)
    if not 0 < A < L:
        raise SystemExit(f"--front-layers must be in 1..{L - 1} (got {A})")
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    kvd = cfg["num_key_value_heads"] * hd
    H = cfg["hidden_size"]

    t_export = time.time()
    fnet = WindowedNet(cfg, weights, fs, fp, n_front=A).eval()
    print(f"tracing front graph (S={fs}, P={fp}, layers 0..{A} of {L}) ...", flush=True)
    ex = (
        torch.zeros(1, fs, dtype=torch.int32),
        torch.zeros(fs, hd),
        torch.zeros(fs, hd),
    ) + ((
        torch.zeros(A, fp, kvd),
        torch.zeros(A, fp, kvd),
        torch.zeros(1, fp),
    ) if fp > 0 else ())
    with torch.no_grad():
        traced = torch.jit.trace(fnet, ex)
    print("converting ...", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="ids", shape=(1, fs), dtype=np.int32),
            ct.TensorType(name="cos", shape=(fs, hd), dtype=np.float16),
            ct.TensorType(name="sin", shape=(fs, hd), dtype=np.float16),
        ] + ([
            ct.TensorType(name="k_past", shape=(A, fp, kvd), dtype=np.float16),
            ct.TensorType(name="v_past", shape=(A, fp, kvd), dtype=np.float16),
            ct.TensorType(name="past_valid", shape=(1, fp), dtype=np.float16),
        ] if fp > 0 else []),
        # K/V come out as fp16, the dtype the KV cache stores: the graph already
        # computes in fp16, so upcasting here only to round back down on the Rust
        # side costs a full extra pass over the data on the critical thread.
        # x_out stays f32 — it is written straight into Metal's f32 activations.
        outputs=[ct.TensorType(name="k_cache", dtype=np.float16),
                 ct.TensorType(name="v_cache", dtype=np.float16),
                 ct.TensorType(name="x_out", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    del traced
    gc.collect()

    # Two-chunk sanity on natural text, same method as the windowed graph: chunk 1
    # through torch to build the past, then chunk 2 Core ML (fp16) vs torch (f32).
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    sent = ("The river was beautiful that morning and everyone stopped to look "
            "at it for a while before walking on toward the harbor. ")
    text_ids = np.array(tok.encode(sent * 200).ids[: 2 * fs], dtype=np.int32)
    theta = cfg.get("rope_theta", 10000.0)

    def rope_tables(pos0, n):
        inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
        ang = np.outer(np.arange(pos0, pos0 + n, dtype=np.float64), inv)
        emb = np.concatenate([ang, ang], axis=-1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    c1, s1 = rope_tables(0, fs)
    c2, s2 = rope_tables(fs, fs)
    if fp > 0:
        with torch.no_grad():
            z = torch.zeros(A, fp, kvd)
            k1, v1, _ = fnet(torch.from_numpy(text_ids[:fs][None, :]),
                             torch.from_numpy(c1), torch.from_numpy(s1),
                             z, z, torch.zeros(1, fp))
            kp, vp = torch.zeros(A, fp, kvd), torch.zeros(A, fp, kvd)
            kp[:, :fs], vp[:, :fs] = k1, v1
            valid = torch.zeros(1, fp)
            valid[0, :fs] = 1.0
            k2, v2, x2 = fnet(torch.from_numpy(text_ids[fs:][None, :]),
                              torch.from_numpy(c2), torch.from_numpy(s2), kp, vp, valid)
        got = mlmodel.predict({
            "ids": text_ids[fs:][None, :],
            "cos": c2.astype(np.float16),
            "sin": s2.astype(np.float16),
            "k_past": kp.numpy().astype(np.float16),
            "v_past": vp.numpy().astype(np.float16),
            "past_valid": valid.numpy().astype(np.float16),
        })
    else:
        # P=0: a single first chunk, no past to build — compare chunk 1 directly.
        with torch.no_grad():
            k2, v2, x2 = fnet(torch.from_numpy(text_ids[:fs][None, :]),
                              torch.from_numpy(c1), torch.from_numpy(s1))
        got = mlmodel.predict({
            "ids": text_ids[:fs][None, :],
            "cos": c1.astype(np.float16),
            "sin": s1.astype(np.float16),
        })
    dk = np.abs(got["k_cache"] - k2.numpy()).max()
    dv = np.abs(got["v_cache"] - v2.numpy()).max()
    dx = np.abs(got["x_out"] - x2.numpy()).max()
    xr = float(np.abs(x2.numpy()).max())
    print(f"front S={fs} P={fp} layers 0..{A}: max abs diff Core ML vs torch fp32: "
          f"K {dk:.4f}, V {dv:.4f}, x {dx:.4f} (|x|max {xr:.2f})", flush=True)
    del fnet, got, k2, v2, x2
    gc.collect()

    dest = args.model_dir / f"prefill-f{A}-{fs}w{fp}.mlmodelc"
    with tempfile.TemporaryDirectory() as tmp:
        mlmodel.save(str(Path(tmp) / f"prefill-f{A}-{fs}w{fp}.mlpackage"))
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(mlmodel.get_compiled_model_path(), dest)
    del mlmodel
    gc.collect()
    export_s = time.time() - t_export
    print(f"saved: {dest}", flush=True)
    print(f"EXPORT COST front-{A}-{fs}w{fp}: {export_s:.1f}s "
          f"(trace+convert+check+save; the per-machine ANECompilerService pass is "
          f"separate and shows up on the first Rust load)", flush=True)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--shapes", default="512,2048",
                    help="sequence lengths to export, comma-separated (one graph each), 'none' to skip")
    ap.add_argument("--window", default="1024x7168",
                    help="windowed graph as SxP (chunk x past), 'none' to skip")
    ap.add_argument("--front", default="none",
                    help="layer-split front-half graphs as SxP[xA][,...] (chunk x past x front "
                         "layers, A defaulting to --front-layers). Runs layers 0..A and also "
                         "returns the hidden state so another device finishes the rest (split "
                         "prefill). A is per chunk width on purpose: where the GPU half is the "
                         "bottleneck the ANE should carry more layers, and that balance point "
                         "moves with the chunk size")
    ap.add_argument("--front-layers", type=int, default=0,
                    help="how many leading layers the --front graph runs (default: half)")
    args = ap.parse_args()
    shapes = [] if args.shapes == "none" else sorted(int(s) for s in args.shapes.split(","))

    import coremltools as ct

    cfg = json.loads((args.model_dir / "config.json").read_text())
    weights = load_file(args.model_dir / "model.safetensors")
    net = PrefillNet(cfg, weights, shapes[-1]).eval() if shapes else None

    done = []
    for seq in shapes:
        print(f"tracing graph (S={seq}) ...", flush=True)
        traced = torch.jit.trace(net, torch.zeros(1, seq, dtype=torch.int32))

        print("converting to Core ML (fp16, requesting CPU_AND_NE — no GPU, it stays free for decoding) ...", flush=True)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="ids", shape=(1, seq), dtype=np.int32)],
            outputs=[ct.TensorType(name="k_cache", dtype=np.float32),
                     ct.TensorType(name="v_cache", dtype=np.float32)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.macOS15,
            convert_to="mlprogram",
        )
        del traced
        gc.collect()

        # Sanity check: Core ML (fp16) vs torch (fp32) on random ids. The real gate
        # is the end-to-end greedy comparison against the CPU backend.
        ids = np.random.randint(1, cfg["vocab_size"], size=(1, seq), dtype=np.int32)
        got = mlmodel.predict({"ids": ids})
        with torch.no_grad():
            want_k, want_v = net(torch.from_numpy(ids))
        dk = np.abs(got["k_cache"] - want_k.numpy()).max()
        dv = np.abs(got["v_cache"] - want_v.numpy()).max()
        print(f"S={seq}: max abs diff Core ML vs torch fp32: K {dk:.4f}, V {dv:.4f}", flush=True)
        del got, want_k, want_v
        gc.collect()

        # Keep only the precompiled .mlmodelc next to the model (the Rust side loads
        # it directly); the intermediate .mlpackage goes to a temp dir so it doesn't
        # clutter the shared Hugging Face cache snapshot.
        with tempfile.TemporaryDirectory() as tmp:
            mlmodel.save(str(Path(tmp) / f"prefill-{seq}.mlpackage"))
            dest = args.model_dir / f"prefill-{seq}.mlmodelc"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(mlmodel.get_compiled_model_path(), dest)
        del mlmodel
        gc.collect()
        done.append(dest)
        print(f"saved: {dest}", flush=True)

    del net
    gc.collect()

    front_widths = set()
    if args.front != "none":
        for spec in args.front.split(","):
            front_widths.add(int(spec.split("x")[0]))
            done.append(export_front(args, cfg, weights, spec))

    if args.window != "none":
        ws, wp = (int(v) for v in args.window.split("x"))
        L = cfg["num_hidden_layers"]
        kvd = cfg["num_key_value_heads"] * (cfg["hidden_size"] // cfg["num_attention_heads"])
        wnet = WindowedNet(cfg, weights, ws, wp).eval()
        del weights
        gc.collect()

        hd = cfg["hidden_size"] // cfg["num_attention_heads"]
        print(f"tracing windowed graph (S={ws}, P={wp}) ...", flush=True)
        ex = (
            torch.zeros(1, ws, dtype=torch.int32),
            torch.zeros(ws, hd),
            torch.zeros(ws, hd),
            torch.zeros(L, wp, kvd),
            torch.zeros(L, wp, kvd),
            torch.zeros(1, wp),
        )
        with torch.no_grad():
            traced = torch.jit.trace(wnet, ex)
        print("converting ...", flush=True)
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="ids", shape=(1, ws), dtype=np.int32),
                ct.TensorType(name="cos", shape=(ws, hd), dtype=np.float16),
                ct.TensorType(name="sin", shape=(ws, hd), dtype=np.float16),
                ct.TensorType(name="k_past", shape=(L, wp, kvd), dtype=np.float16),
                ct.TensorType(name="v_past", shape=(L, wp, kvd), dtype=np.float16),
                ct.TensorType(name="past_valid", shape=(1, wp), dtype=np.float16),
            ],
            outputs=[ct.TensorType(name="k_cache", dtype=np.float32),
                     ct.TensorType(name="v_cache", dtype=np.float32)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.macOS15,
            convert_to="mlprogram",
        )
        del traced
        gc.collect()

        # Two-chunk sanity on natural text (random ids overstate fp16 drift):
        # chunk 1 through torch, then chunk 2 with that past, Core ML vs torch.
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
        sent = ("The river was beautiful that morning and everyone stopped to look "
                "at it for a while before walking on toward the harbor. ")
        text_ids = np.array(tok.encode(sent * 200).ids[: 2 * ws], dtype=np.int32)
        theta = cfg.get("rope_theta", 10000.0)

        def rope_tables(pos0, n):
            inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
            ang = np.outer(np.arange(pos0, pos0 + n, dtype=np.float64), inv)
            emb = np.concatenate([ang, ang], axis=-1)
            return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

        c1, s1 = rope_tables(0, ws)
        c2, s2 = rope_tables(ws, ws)
        with torch.no_grad():
            z = torch.zeros(L, wp, kvd)
            k1, v1 = wnet(torch.from_numpy(text_ids[:ws][None, :]),
                          torch.from_numpy(c1), torch.from_numpy(s1),
                          z, z, torch.zeros(1, wp))
            kp = torch.zeros(L, wp, kvd)
            vp = torch.zeros(L, wp, kvd)
            kp[:, :ws] = k1
            vp[:, :ws] = v1
            valid = torch.zeros(1, wp)
            valid[0, :ws] = 1.0
            k2, v2 = wnet(torch.from_numpy(text_ids[ws:][None, :]),
                          torch.from_numpy(c2), torch.from_numpy(s2), kp, vp, valid)
        got = mlmodel.predict({
            "ids": text_ids[ws:][None, :],
            "cos": c2.astype(np.float16),
            "sin": s2.astype(np.float16),
            "k_past": kp.numpy().astype(np.float16),
            "v_past": vp.numpy().astype(np.float16),
            "past_valid": valid.numpy().astype(np.float16),
        })
        dk = np.abs(got["k_cache"] - k2.numpy()).max()
        dv = np.abs(got["v_cache"] - v2.numpy()).max()
        print(f"windowed S={ws} P={wp}: max abs diff Core ML vs torch fp32: K {dk:.4f}, V {dv:.4f}", flush=True)
        del wnet, got, k1, v1, k2, v2, kp, vp
        gc.collect()

        with tempfile.TemporaryDirectory() as tmp:
            mlmodel.save(str(Path(tmp) / f"prefill-{ws}w{wp}.mlpackage"))
            dest = args.model_dir / f"prefill-{ws}w{wp}.mlmodelc"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(mlmodel.get_compiled_model_path(), dest)
        del mlmodel
        gc.collect()
        done.append(dest)
        print(f"saved: {dest}", flush=True)

    # Remove stale graphs from older exports — but only of the kinds this run
    # actually rebuilt: a plain-only run must not delete the windowed graph and
    # a --shapes none windowed run must not delete the plain graphs (that
    # cross-delete once wiped a model's whole graph set).
    for old in args.model_dir.glob("prefill-*.mlmodelc"):
        spec = old.stem.removeprefix("prefill-")
        if spec.startswith("f") and "-" in spec:   # front-half split graph
            # Scoped to the chunk width: rebuilding the 128-wide family must not
            # delete the 256-wide one (they are independent ladders, and they may
            # legitimately carry different front-layer counts).
            width = spec.split("-", 1)[1].split("w")[0]
            rebuilt_kind = width.isdigit() and int(width) in front_widths
        elif "w" in spec:                          # windowed graph
            rebuilt_kind = args.window != "none"
        else:                                      # plain fixed-shape graph
            rebuilt_kind = bool(shapes)
        if rebuilt_kind and old not in done:
            shutil.rmtree(old)
    print(f"done: {len(done)} graph(s)")


if __name__ == "__main__":
    main()
