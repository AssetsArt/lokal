# lokal — Design

This document explains how lokal works and why it is shaped the way it is.
File and function names are given instead of line numbers so the references
stay valid as the code moves.

## Goals

- **Local-first**: a single binary that pulls a model from the Hugging Face Hub
  once and then runs fully offline. The HTTP server is a mode, not the product.
- **Whole stack in one repo**: no ML framework. Reading this codebase top to
  bottom should explain how LLM inference actually works — from safetensors
  bytes to sampled tokens.
- **Every accelerator on the machine**: on Apple Silicon that means the CPU,
  the Metal GPU, and the Neural Engine, all behind one trait and one CLI flag.
- **Verifiable correctness**: independent backend implementations that can be
  diffed against each other deterministically (see Testing below).

Non-goals for now: training, multi-GPU, exotic architectures (only the
Llama family), and squeezing out the last 2x of kernel performance at the cost
of readability.

## Bird's-eye view

```
prompt ──tokenizer──→ token ids ──[session.prefill]──→ logits of last position
   ┌──────────────────────────────────────────────────────────┘
   └→ sampler → next token ──[session.forward]──→ logits → sampler → ... → EOS
                                                        (autoregressive loop)

session.forward/prefill dispatch to the selected backend:
  -b cpu   → model.rs        (reference implementation, plain Rust)
  -b metal → gpu/metal.rs    (Apple GPU, MSL kernels)
  -b ane   → ane.rs          (Neural Engine prefill + Metal decode)
```

## Startup

`main::run` assembles three ingredients and hands them to a backend:

1. `hub::resolve_model` — resolves config.json, tokenizer.json, and the
   safetensors file(s) through the standard Hugging Face cache
   (`~/.cache/huggingface/hub`, via the `hf-hub` crate), so models are shared
   with every other HF-ecosystem tool on the machine. Falls back to
   cache-only resolution when offline.
2. `config::ModelConfig::load` — hyperparameters, field names matching the JSON.
3. `model::Model::load` — `weights::load` maps every tensor to f32 in RAM and
   wires them into typed structs (`Linear`, `Block`), converting bf16/f16 and
   handling sharded checkpoints and tied embeddings.
4. `engine::create` — wraps the `Model` in the chosen backend. The CPU engine
   uses it directly; Metal converts to f16 and uploads; the ANE engine wraps a
   Metal engine plus a compiled Core ML graph.

## The transformer (reference implementation)

`model::Model::forward` processes **one token at one position** and returns
logits over the vocabulary. It is the contract every backend reimplements:

```
embedding lookup
for each Block:
    x' = rmsnorm(x); q,k,v = projections(x'); rope(q), rope(k)
    append k,v to the KV cache at this position
    att = softmax(q·K / √d) · V        (per head; GQA maps q heads → kv heads)
    x  += o_proj(att)                   (residual)
    x' = rmsnorm(x)
    x  += down(silu(gate(x')) * up(x')) (SwiGLU MLP, residual)
final rmsnorm → lm_head → logits
```

Two facts make the whole system simple:

- **Prefill and decode are the same function.** Prefill feeds known prompt
  tokens; decode feeds the token just sampled. One forward implementation
  serves both.
- **The KV cache is the only cross-step state.** The model itself is pure;
  everything the model "remembers" lives in the cache. This is what makes the
  Engine/Session split (below) natural.

## The backend seam: Engine and Session

Defined in `engine.rs`:

- **Engine** — loaded weights, read-only, `Send + Sync`, shared across threads.
  One per process.
- **Session** — one generation run's mutable state (the KV cache). Created per
  request, used from a single thread. Provides:
  - `forward(token, pos)` — required; one token, one position.
  - `prefill(ids)` — provided; the default loops over `forward`, and backends
    with batch support override it. This is also the plug-in point for
    alternative prefill hardware — the ANE backend overrides exactly this.

The generation loop (`generate::generate`) knows nothing about backends. The
HTTP server (`server.rs`) shares one `Arc<dyn Engine>` and builds a session per
request — concurrent requests need no locks because nothing mutable is shared.

Adding a backend = implement the two traits, add one arm to `engine::create`,
gate with `#[cfg]`, add a target-specific dependency. The checklist and the
invariants worth keeping are in `src/gpu/mod.rs`.

## Metal backend

`gpu/metal.rs` + `gpu/kernels.metal`. Three decisions carry all of its
performance:

1. **Weights are f16 and resident.** Decode is memory-bandwidth-bound: every
   generated token must stream every weight byte once. Halving the bytes
   roughly doubles decode speed; uploading once at engine creation makes the
   per-token traffic weights-only. Activations and the KV cache stay f32, and
   all accumulation is f32.
2. **One command buffer per step.** A CPU↔GPU sync costs ~100 µs; a forward
   pass is ~450 dispatches. All of them are encoded into a single serial
   command buffer, submitted once, waited on once. Token ids go in, logits come
   out; the KV cache never leaves the GPU. On M-series unified memory, even
   those crossings are plain memcpys.
3. **matvec for decode, tiled matmul for prefill.** Every kernel takes an
   `n_rows` parameter; decode is simply `n_rows = 1`. For prefill,
   `enc_linear` switches to a threadgroup-tiled matmul (8 tokens × 32 outputs
   × 32-wide k-slices staged in on-chip memory), so one read of W serves a
   whole chunk of tokens instead of a single one. Prompts are processed in
   chunks of 128; later chunks attend to earlier ones through the cache, and
   causality is just the per-row loop bound `0..=pos0+row`.

The kernels are deliberately annotated against their CPU twins — `matvec` ↔
`math::matvec`, `attention` ↔ `model::attention`, and so on.

## ANE backend

`ane.rs` + `tools/export_prefill.py`. The Neural Engine is only reachable
through Core ML, whose authoring toolchain is Python — so the design accepts a
one-time Python export step and keeps the runtime pure Rust + Core ML.

- **What runs on the ANE: prefill only.** The ANE wants static graphs; decode's
  ever-growing cache fits it poorly, but prefill is a fixed-shape,
  compute-bound batch job — exactly its diet. The export script rebuilds the
  prefill half of the transformer (through `embedding` → blocks → per-layer
  K,V; no lm_head) at a fixed S=512, fp16, requesting `CPU_AND_NE`.
- **Padding is safe by construction.** Prompts shorter than 512 are zero-padded
  at the tail; the causal mask guarantees pad positions cannot influence the
  K,V of real positions before them, and the padded rows are simply not copied
  out.
- **The hybrid handoff.** `AneSession::prefill` sends `ids[..n-1]` (up to 512)
  through Core ML, memcpys the resulting K,V into the Metal session's cache
  (`MetalSession::write_kv`), then lets Metal handle any overflow plus the
  final prompt token (`MetalSession::prefill_from`) — the last token produces
  the logits, which is why the ANE graph can omit lm_head entirely. Decoding
  is pure Metal.
- **Placement is verified, not assumed.** Core ML decides where a graph runs;
  we check with the MLComputePlan API (1,733 ops on the NeuralEngine device,
  6 on the CPU for SmolLM2-135M) and `powermetrics --samplers ane_power`.

## Numerics and correctness

- Reference path: f32 everywhere.
- Metal: f16 weights, f32 activations/accumulation.
- ANE: fp16 graph end to end (Core ML converts), K/V returned as f32.

The correctness instrument is the **golden greedy test**: at `--temperature 0`
the system is fully deterministic, and the three backends are independent
implementations of the same contract. Running the same prompt on cpu, metal,
and ane must produce token-identical output — in practice it does, including
across prefill chunk boundaries and Qwen2's bias path. Divergence caused by
fp16 rounding at a near-tie in the logits is theoretically possible; divergence
into gibberish is a bug, and the fragility of a 30-layer transformer makes
gibberish loud.

Unit tests cover the leaf math (`math.rs`, `sampler.rs`) with hand-checkable
values and run offline.

## Performance snapshot (M1 Pro, SmolLM2-135M, greedy)

| workload | cpu | metal | ane hybrid |
|---|---|---|---|
| decode | ~49 tok/s | ~160–190 tok/s | = metal |
| prefill, 669-token prompt | ~33 tok/s | ~740 tok/s | ~1,780 tok/s |
| ANE-only portion (512 tokens) | — | — | ~0.07 s |

## Future work

Quantization (int8/int4) is the highest-leverage next step — it shrinks the
bandwidth bill that dominates decode. On Metal, `simdgroup_matrix` MMA would
lift the matmul toward MLX-class throughput. On the ANE, enumerated shapes
would remove the fixed-512 padding cost, and Core ML stateful models (MLState)
are the speculative path to ANE decoding. Server mode wants an
OpenAI-compatible API (`/v1/chat/completions`), SSE streaming, and continuous
batching. CUDA/Vulkan backends can reuse the Engine/Session seam and the Metal
backend's invariants as a template.
