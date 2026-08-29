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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--shapes", default="512,2048",
                    help="sequence lengths to export, comma-separated (one graph each)")
    args = ap.parse_args()
    shapes = sorted(int(s) for s in args.shapes.split(","))

    import coremltools as ct

    cfg = json.loads((args.model_dir / "config.json").read_text())
    weights = load_file(args.model_dir / "model.safetensors")
    net = PrefillNet(cfg, weights, shapes[-1]).eval()
    del weights  # the net holds its own f32 copies; keep peak memory down
    gc.collect()

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

    # Remove graphs from older exports with sizes not in this run.
    for old in args.model_dir.glob("prefill-*.mlmodelc"):
        if old not in done:
            shutil.rmtree(old)
    print(f"done: {len(done)} graph(s), shapes {shapes}")


if __name__ == "__main__":
    main()
