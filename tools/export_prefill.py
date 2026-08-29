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
# Deliberate constraint: a fixed shape S=512 (prompts are zero-padded at the tail).
# The ANE strongly prefers static graphs, and the causal mask guarantees pad positions
# cannot affect the K,V of the real positions before them.
#
# Usage:
#   uv run --python 3.12 --with torch --with coremltools --with safetensors \
#       tools/export_prefill.py ~/.cache/lokal/HuggingFaceTB--SmolLM2-135M --seq 512

import argparse
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
    buffers rather than nn.Linear, so each op maps 1:1 onto a line of model.rs."""

    def __init__(self, cfg, weights, seq):
        super().__init__()
        self.cfg = cfg
        self.seq = seq
        self.n_layers = cfg["num_hidden_layers"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.hd = cfg["hidden_size"] // self.n_heads
        self.eps = cfg["rms_norm_eps"]
        for name, t in weights.items():
            self.register_buffer(name.replace(".", "_"), t.float(), persistent=False)

        # Precomputed RoPE cos/sin tables (positions 0..S-1) — constants in the graph.
        theta = cfg.get("rope_theta", 10000.0)
        inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        ang = torch.outer(torch.arange(seq).float(), inv)  # [S, hd/2]
        emb = torch.cat((ang, ang), dim=-1)                # [S, hd] (both halves, HF-style)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)
        # Causal mask: position i sees only 0..i (this is what makes tail padding safe).
        mask = torch.full((seq, seq), float("-inf")).triu(1)
        self.register_buffer("causal", mask.view(1, 1, seq, seq), persistent=False)

    def w(self, layer, name):
        return getattr(self, f"model_layers_{layer}_{name}".replace(".", "_"))

    def forward(self, ids):  # ids: int32 [1, S]
        S, hd = self.seq, self.hd
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
            q = q.view(1, S, self.n_heads, hd).transpose(1, 2)  # [1,heads,S,hd]
            k = k.view(1, S, self.n_kv, hd).transpose(1, 2)
            v = v.view(1, S, self.n_kv, hd).transpose(1, 2)

            q = q * self.rope_cos + rotate_half(q, hd // 2) * self.rope_sin
            k = k * self.rope_cos + rotate_half(k, hd // 2) * self.rope_sin

            # Capture K,V (post-RoPE, pre-GQA-expansion) — laid out [S, kv*hd] to match
            # the Rust cache layout exactly.
            k_out.append(k.transpose(1, 2).reshape(S, self.n_kv * hd))
            v_out.append(v.transpose(1, 2).reshape(S, self.n_kv * hd))

            # GQA: expand kv heads to match the query heads, then full causal attention.
            # (expand+reshape instead of repeat_interleave — coremltools converts it cleanly.)
            rep = self.n_heads // self.n_kv
            kf = k.unsqueeze(2).expand(1, self.n_kv, rep, S, hd).reshape(1, self.n_heads, S, hd)
            vf = v.unsqueeze(2).expand(1, self.n_kv, rep, S, hd).reshape(1, self.n_heads, S, hd)
            att = (q @ kf.transpose(-1, -2)) * (hd ** -0.5) + self.causal
            o = (F.softmax(att, dim=-1) @ vf).transpose(1, 2).reshape(1, S, -1)
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
    ap.add_argument("--seq", type=int, default=512)
    args = ap.parse_args()

    import coremltools as ct

    cfg = json.loads((args.model_dir / "config.json").read_text())
    weights = load_file(args.model_dir / "model.safetensors")
    net = PrefillNet(cfg, weights, args.seq).eval()

    print(f"tracing graph (S={args.seq}) ...")
    example = torch.zeros(1, args.seq, dtype=torch.int32)
    traced = torch.jit.trace(net, example)

    print("converting to Core ML (fp16, requesting CPU_AND_NE — no GPU, it stays free for decoding) ...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="ids", shape=(1, args.seq), dtype=np.int32)],
        outputs=[ct.TensorType(name="k_cache", dtype=np.float32),
                 ct.TensorType(name="v_cache", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )

    # Sanity check: Core ML (fp16) vs torch (fp32) on random ids.
    ids = np.random.randint(1, cfg["vocab_size"], size=(1, args.seq), dtype=np.int32)
    got = mlmodel.predict({"ids": ids})
    want_k, want_v = net(torch.from_numpy(ids))
    dk = np.abs(got["k_cache"] - want_k.numpy()).max()
    dv = np.abs(got["v_cache"] - want_v.numpy()).max()
    print(f"max abs diff, Core ML vs torch fp32: K {dk:.4f}, V {dv:.4f} (fp16 accumulation drift; the real gate is the end-to-end greedy comparison)")

    # Keep only the precompiled .mlmodelc next to the model (the Rust side loads it
    # directly); the intermediate .mlpackage goes to a temp dir so it doesn't clutter
    # the shared Hugging Face cache snapshot.
    with tempfile.TemporaryDirectory() as tmp:
        mlmodel.save(str(Path(tmp) / f"prefill-{args.seq}.mlpackage"))
        dest = args.model_dir / f"prefill-{args.seq}.mlmodelc"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(mlmodel.get_compiled_model_path(), dest)
    print(f"done: {dest}")


if __name__ == "__main__":
    main()
