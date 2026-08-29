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

`gpu/metal.rs` + `gpu/kernels.metal`. Four decisions carry all of its
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

4. **Decode has its own fused path.** A single decode step is hundreds of tiny
   launches, so launch count and bandwidth efficiency dominate. When
   `n_rows == 1` the layer loop switches to fused kernels — q/k/v in one
   dispatch (k,v written straight into their cache slot), the whole SwiGLU
   inner step in one, residual adds folded into o/down matvecs, RoPE for q+k
   in one — 9 dispatches per layer instead of 15. Attention becomes
   flash-decoding: cached positions split into 128-wide windows, one
   threadgroup per (head, window) computes a partial softmax-weighted sum
   (scores never touch device memory), and a tiny second kernel merges the
   windows with the online-softmax rule — the position loop that used to be
   serial per head now spreads across the whole GPU. All dot products load
   `half4`/`float4` vectors, which is what pushes a bandwidth-bound kernel
   toward peak. Prefill keeps the plain tiled-matmul path.

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
  K,V; no lm_head) at fixed shapes (S=512 and S=2048 by default), fp16,
  requesting `CPU_AND_NE`.
- **A ladder of fixed graphs, not one dynamic one.** Enumerated shapes were
  tried and measured: on this graph `ct.EnumeratedShapes` makes ANECCompile
  fail (silent CPU fallback) and the compiled model OOMs a 16 GB machine at
  load. Separate fixed graphs compile clean; the cost is one weight copy on
  disk per size. Routing (`ane::pick_graph`): prompts under 64 tokens skip
  the ANE entirely (the GPU is faster than the smallest padded graph); use
  the smallest graph that fits, but only step up to a bigger graph when the
  prompt fills at least half of it — ANE time grows superlinearly with S
  (attention is S²: 46 ms at 512, 510 ms at 2048), so below half-full it is
  cheaper to fill the smaller graph and let Metal take the overflow.
- **Padding is safe by construction.** Prompts shorter than the chosen graph
  are zero-padded at the tail; the causal mask guarantees pad positions
  cannot influence the K,V of real positions before them, and the padded
  rows are simply not copied out.
- **The hybrid handoff.** `AneSession::prefill` sends `ids[..n-1]` (up to the
  chosen graph's length) through Core ML, memcpys the resulting K,V into the
  Metal session's cache (`MetalSession::write_kv`), then lets Metal handle
  any overflow plus the final prompt token (`MetalSession::prefill_from`) —
  the last token produces the logits, which is why the ANE graphs can omit
  lm_head entirely. Decoding is pure Metal.
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
| decode (short context) | ~49 tok/s | ~267 tok/s | = metal |
| decode (~500 positions) | — | ~237 tok/s | = metal |
| prefill, 676-token prompt | ~33 tok/s | ~740 tok/s | ~1,700 tok/s |
| prefill, 1,223-token prompt | — | ~515 tok/s | ~2,200 tok/s |
| ANE-only graph run | — | — | 0.05 s (S=512) / 0.51 s (S=2048) |

End to end — 676-token prompt, 200 tokens generated:

| backend | TTFT | decode | total |
|---|---|---|---|
| cpu | 18.4 s | ~27 tok/s | ~26 s |
| metal | 0.91 s | ~228 tok/s | ~1.8 s |
| ane+metal | 0.40 s | ~231 tok/s | ~1.3 s |

Two readings worth taking away. First, the hybrid's decode speed is exactly
Metal's — the ANE changes time-to-first-token only, and at chat latencies
0.91 s → 0.40 s is very perceptible. Second, decode still slows as context
grows (attention's cost is linear in cached positions), but flash-decoding
flattened the slide dramatically: before it, the same run decoded at
~76 tok/s; the remaining per-position cost is mostly the f32 KV cache reads.

## Where the time goes

"matvec dominates" is a statement about the CPU backend, not about the system.
Each device has its own cost structure:

- **cpu** — matvec is 90%+ of everything; one token is one full pass over the
  weights through the memory hierarchy.
- **metal** — the same arithmetic becomes ~270 fused dispatches (decode) or
  ~450 plain ones (prefill) encoded into one command buffer per step. The
  decode floor is streaming the f16 weights (~1.4 ms/token for 135M at full
  bandwidth); the fixed costs are launch overhead and the single sync per
  step; attention grows linearly with context but is spread across the GPU
  by flash-decoding; and prefill moves the cost into the tiled matmul, where
  *reuse* — not raw FLOPs — is the lever that matters.
- **ane** — execution is a black box scheduled by Core ML. What stays visible
  from outside is the fixed-shape graph execution (~70 ms for 512 positions)
  and the Core ML ↔ Metal handoff (one K/V memcpy per prefill).

The consequence: the open optimization frontier is no longer the matmul — it
is decode state management and scheduling work across devices.

## Speculative decoding

`generate::speculative`, active with `--draft <model>` at temperature 0. A small
same-tokenizer model proposes a block of tokens; the target verifies the whole
block in one batched pass (`Session::forward_batch` — one Metal submission with
lm_head on every row). Greedy acceptance compares argmax against argmax, so the
output is token-identical to running the target alone — verified: a target
drafting for itself accepts 100% of proposals.

Design notes:

- **Position-addressed KV caches make rollback free.** Both backends write K,V
  by explicit position, so a rejected token's rows are simply overwritten on
  the next round — `draft_pos = draft_pos.min(seq.len())` is the entire
  unwind logic.
- **The block size adapts** (grow on full accept, shrink on none) between 1
  and 7, keeping the verify batch inside the Metal logits buffer.
- **When it pays — the honest arithmetic.** A round costs ~γ draft steps plus
  one verify (≈ one target step); it emits `accepted + 1` tokens. With the
  pairs that fit in 16 GB today the draft is too expensive relative to its
  target (Qwen2.5-0.5B is ~37% of a 1.5B step; acceptance measured 46%) —
  break-even, not a win. The technique needs a target ~6x+ the draft, i.e.
  3B+ targets with a 0.5B draft. The blocker is the loader (weights expand to
  f32 in RAM, so 3B needs ~12 GB just to load); f16 loading is the unlock and
  is on the roadmap. The machinery itself is correct, exact, and free when
  `--draft` is not passed.

## Serve-mode concurrency

Because an Engine is immutable and every Session owns its own state,
concurrent requests need no locks — and on the hybrid backend they overlap
across *devices*: while the GPU decodes one request, the Neural Engine
prefills the next. Nothing schedules this explicitly; requests live on
separate blocking threads, and both MLModel prediction and Metal command
submission are documented thread-safe. Measured (M1 Pro, SmolLM2-135M,
451-token prompts, 100 tokens generated, 4 concurrent requests):

| | metal | ane hybrid |
|---|---|---|
| single-request prefill | ~0.56 s | ~0.06 s |
| ANE prefill time under load | — | 0.05–0.18 s (≈ idle) |
| aggregate throughput | 126 gen-tok/s | 264 gen-tok/s |

Aggregate decode saturates at a few concurrent generations; past that, extra
concurrency only splits the same tokens/sec across more requests and holds
more KV caches in RAM. So serve admits `--max-concurrent` requests (default
4) into generation and queues the rest FIFO (a tokio semaphore) — under a
16-request burst, admitted requests keep their full share instead of
everyone slowing to a crawl. Raising the ceiling itself needs continuous
batching (Future work).

### Why not split tensors across ANE + GPU?

The measured reason finer-grained parallelism (Megatron-style tensor split)
does not pay on this hardware:

- Core ML cannot exchange tensors mid-graph, so the finest possible split
  unit is one whole-graph invocation. Measured round-trips: 0.07 ms floor,
  0.65 ms for one MLP layer at S=512. A tensor split needs two joins per
  layer — ~60 invocations per step, ~39 ms per 512-token prefill chunk,
  more than the 15–23 ms a perfect ANE/GPU split could save (the ANE alone
  already prefills 512 tokens in 50–80 ms, in a single invocation).
- Decode gains nothing at any overhead: it is bandwidth-bound, and both
  devices draw from the same unified-memory bandwidth. Splitting the weights
  adds FLOPs, not bytes/s.

The general rule on Apple Silicon: parallelism pays at coarse granularity —
per request, per phase (prefill/decode) — and loses at fine granularity,
where every ANE↔GPU boundary costs a graph invocation.

## Future work

Roughly ordered by leverage:

1. **f16 KV cache** — the last piece of the decode-speed work (the
   flash-decoding attention, fused decode kernels, and vectorized loads
   landed already — ~76 → ~237 tok/s at ~500 positions). Halving the KV
   bytes flattens the remaining per-position cost; it touches both attention
   paths, the RoPE cache writes, and the ANE handoff.
2. **Quantization (int8/int4)** — shrinks the bandwidth bill that sets the
   decode floor on every device, and admits 1B–8B models on modest RAM.
   Needs its own quality methodology: quantized greedy output no longer
   matches the f32 reference token-for-token.
3. **Continuous batching in serve mode** — the only way past the ~124
   gen-tok/s aggregate ceiling: batch concurrent decode steps into one
   matmul so a single read of the weights serves every active request. The
   kernels already take `n_rows`, and the admission queue already exists.
4. **ANE decode via Core ML stateful models (MLState)** — keep the KV cache
   inside the Core ML graph across invocations. The measured ~0.65 ms
   invocation cost says the overhead is tolerable once per-step compute is
   large enough (i.e. bigger models); the open questions are whether ANE
   placement survives the state ops.
5. The rest: an OpenAI-compatible API (`/v1/chat/completions`) and SSE
   streaming in serve mode; `simdgroup_matrix` MMA on Metal; flash-style
   attention for the *prefill* path (its weighted-V loop is still serial per
   thread, which is what caps very-long-prompt Metal prefill); a hybrid
   scheduler that picks the backend automatically; CUDA/Vulkan backends on
   the same Engine/Session seam.
