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
  -b hybrid→ ane.rs          (Neural Engine prefill — or an ANE+GPU split — plus Metal decode)
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

1. **Weights and the KV cache are f16 and resident.** Decode is
   memory-bandwidth-bound: every generated token must stream every weight
   byte once. Halving the bytes roughly doubles decode speed; uploading once
   at engine creation makes the per-token traffic weights-only. The KV cache
   is f16 too — half the memory per session and half the attention bandwidth
   at long context. Activations stay f32, and all accumulation is f32.
2. **One command buffer per step.** A CPU↔GPU sync costs ~100 µs; a forward
   pass is ~450 dispatches. All of them are encoded into a single serial
   command buffer, submitted once, waited on once. Token ids go in, logits come
   out; the KV cache never leaves the GPU. On M-series unified memory, even
   those crossings are plain memcpys.
3. **matvec for decode, simdgroup-matrix matmul for prefill.** Every kernel
   takes an `n_rows` parameter; decode is simply `n_rows = 1`. For prefill,
   `enc_linear` switches to a threadgroup-tiled matmul (8 tokens × 32 outputs
   × 32-wide k-slices staged in on-chip memory), so one read of W serves a
   whole chunk of tokens instead of a single one — and the multiply itself
   runs on the GPU's matrix hardware (`simdgroup_multiply_accumulate` on 8×8
   blocks, staged as f32 so accumulation precision is unchanged). Prompts are
   processed in chunks of 128; later chunks attend to earlier ones through
   the cache, and causality is just the per-row loop bound `0..=pos0+row`.

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

## Hybrid backend (ANE + GPU)

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
- **Windowed chunking past the largest graph.** Longer prompts run in
  S=1024 chunks through a windowed graph (`prefill-1024w5120.mlmodelc`): each
  chunk attends to up to P=5120 rows of accumulated past K/V fed in as
  validity-masked inputs, so everything up to 6,144 positions stays on the
  ANE, and Metal takes only the tail beyond that. Two hard-won rules are
  baked into the export: (1) the graph must never derive positions
  internally — the fp16 pipeline cannot represent integers above 2,048, which
  silently corrupts RoPE from that position on (this presented as a "cliff"
  that was first misread as a softmax-width limit), so RoPE cos/sin arrive
  as host-computed inputs; (2) every softmax stays ≤ 2,048 positions wide,
  with segments merged by an exact online-softmax. Measured: a 6,086-token
  prompt prefills in 3.3 s vs 14.1 s on Metal alone (4.3x), token-identical
  to the CPU reference.
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
- Metal: f16 weights and KV cache, f32 activations/accumulation.
- ANE: fp16 graph end to end (Core ML converts), K/V converted to the f16
  cache during the handoff.

The correctness instrument is the **golden greedy test**: at `--temperature 0`
the system is fully deterministic, and the three backends are independent
implementations of the same contract. Running the same prompt on cpu and metal
must produce token-identical output — any flip between them is a bug. The fp16
ane path matches them in practice too, including across prefill chunk
boundaries and Qwen2's bias path, but on long prompts a greedy near-tie can
resolve differently (measured, rare, both continuations sensible); its hard
gate is the numeric envelope vs the f32 reference — flat across positions, no
NaN — where error *growing* with position is the bug signature. Divergence
into gibberish is a bug, and the fragility of a 30-layer transformer makes
gibberish loud.

Unit tests cover the leaf math (`math.rs`, `sampler.rs`) with hand-checkable
values and run offline.

## Performance snapshot (M1 Pro, SmolLM2-135M, greedy)

| workload | cpu | metal | hybrid |
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

## Serve-mode concurrency: continuous batching

serve (on the GPU backends) runs one scheduler thread (`batch.rs`) that owns a
`Batcher`: a pooled KV cache plus a batched decode step. Requests queue on a
channel; a free slot admits the next request (its prompt prefills into the
slot — on the hybrid backend that runs on the ANE), and every loop iteration
advances ALL active requests by one token in a single GPU submission. Decode
is bandwidth-bound on the weights, so one read of the weights serving four
requests is nearly four times the aggregate. Outputs are bit-for-bit the
same as running each request alone — verified with concurrent identical
prompts, distinct prompts, and requests joining mid-flight.

**KV layout: static slots, deliberately not PagedAttention.** The pool is
`[slot][max_seq][kv_dim]` per layer. PagedAttention exists because discrete
VRAM fragments; on unified memory, macOS commits 16 KB pages lazily, so an
untouched slot tail costs virtual address space, not RAM — reserving full
slots is effectively free, and the kernels keep fully linear, coalesced
`half4` access with no block-table indirection. Block tables would buy prompt
caching and 16+ concurrency, which this project doesn't need yet.

Measured (M1 Pro, SmolLM2-135M-Instruct, 451-token prompts, 128 generated,
4 concurrent):

| | metal | hybrid |
|---|---|---|
| single-request prefill | ~0.49 s | ~0.06 s |
| aggregate throughput | 168 gen-tok/s | **365 gen-tok/s** |

The gap between the two columns is prefill: admission runs a whole prompt's
prefill before decode steps resume, and on metal that stalls the batch for
~0.5 s per join, while the ANE does it in ~0.06 s on another device — the
hybrid design's whole thesis in one number. (Interleaving prefill chunks with
decode steps would close the metal gap — Future work.) The cpu backend keeps
the earlier per-request path behind a FIFO semaphore.

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

### Why not MLState decode on the ANE?

Probed 2026-08-29 (coremltools 9.0, torch 2.13.0, macOS 15 target, M1 Pro).
The idea — keep the KV cache inside a stateful Core ML graph (`ct.StateType`)
so ANE decode stops paying the host↔ANE KV round-trip — dies on three
independently measured ceilings, any one of which is sufficient:

- **NE placement caps at ~6 MB of total state per graph.** A 1-layer decode
  step with KV state places on the NE up to S=7168 (5.5 MB of state) and
  falls 100% to GPU at S=8192 (6.3 MB), in every variant tried. Splitting
  the state into more, smaller buffers does not evade it — 2×4096 and 4×2048
  (same 6.3 MB total) also go 100% GPU while a single 3.1 MB pair stays NE.
  The bound tracks total state bytes, not per-buffer size.
- **Multi-layer stateful graphs leave the NE, then stop loading at all.**
  2/4/8-layer stateful graphs convert and load but place 100% on GPU; at
  ≥16 layers `ANECCompile()` fails outright (execution-plan error -14), even
  with state shrunk to 2048 ctx. The real SmolLM2-135M (30 layers, 60 state
  buffers) fails to load.
- **The per-layer-model escape hatch loses on the invocation floor.** A chain
  of 30 single-layer stateful models is NE-placeable, but the measured
  ~0.74 ms predict floor × 30 layers ≈ 22 ms/token ≈ 45 tok/s — below the
  Metal decode it would have to beat (122 tok/s at 7k ctx on SmolLM2).

Even ignoring all three, NE-placeable context in the stateful variants caps
at ~7k — below the ≥10k regime this engine targets.

Worth keeping for a toolchain revisit: state ops per se DO survive NE
placement (the answer to the old open question) — in fact the state *update*
op is what pulls a graph onto the NE; a read-only stateful graph places on
CPU. `state.copy_(expr)` does not convert; mask read-modify-write
(`state.mul_(1-m); state.add_(k*m)`) with a host-fed one-hot mask does, and
sidesteps the fp16-2048 integer cliff entirely — verified position-flat
error across 2047/2048/2049/4095/4096 over 6101 sequential predicts against
torch f32. Segmented ≤2048-wide softmax over the state (the windowed-prefill
pattern) stays NE-placed. If Apple's toolchain moves: re-probe the
layer-count load wall first, then the 8k+ placement matrix, then end-to-end.

## Future work

Roughly ordered by leverage:

1. **f16 model loading** — weights currently expand to f32 in RAM during
   loading, so a 3B checkpoint needs ~12 GB before the GPU copy exists.
   Loading bf16 → f16 directly (for the GPU backends) halves that and admits
   3B+ targets on 16 GB — which is also what makes speculative decoding pay
   off (see the Speculative decoding section).
2. **Quantization (int8/int4)** — shrinks the bandwidth bill that sets the
   decode floor on every device, and admits 1B–8B models on modest RAM.
   Needs its own quality methodology: quantized greedy output no longer
   matches the f32 reference token-for-token.
3. **Chunked-prefill scheduling** — admission currently prefills a whole
   prompt before decode steps resume, stalling the batch ~0.5 s per join on
   the metal backend (the ANE hybrid hides this on another device).
   Interleaving PREFILL_CHUNK-sized pieces with decode steps caps the stall
   at one chunk.
4. **Prefill: what is left is attention.** Prefill went 1,461 -> ~9,900 tok/s
   on the GPU (and ~11,900 hybrid) during 2026-08-30, through a flash
   attention kernel with no scores scratch, matmuls on Metal 4 tensor ops
   (`mpp::tensor_ops::matmul2d`, the path llama.cpp's Metal backend takes on
   macOS 26 — hand-tiled simdgroup MMA plateaus near ~700 GFLOPS here no
   matter the tiling), k/v projections onto the same, a concurrent prefill
   encoder, and FA tiles retuned for long K/V loops (96 query rows x 32
   positions: row reuse is the axis that pays once the loop is thousands of
   positions long, while wider position tiles blow past the 16 KB of shared
   memory that keeps two threadgroups per core).

   A phase profile against llama.cpp on Qwen2.5-0.5B then settled where the
   remaining gap lives, and it is not where a reader would guess:

   - **The GEMMs are not it.** Per layer per token they are *more*
     FLOP-efficient than SmolLM2's despite 4.26x the parameters.
   - **Attention is it.** Per head per layer, 62 us against SmolLM2's 43 at
     identical tile shape and head_dim — a 1.46x premium that tracks the
     GQA 14:2 ratio, i.e. the prefill kernel re-reads each kv head once per
     query head, the same redundancy the decode kernel already eliminated.
     At 8k, attention is 52% of prefill and the *entire* gap: 51.7 ns/tok^2
     against llama.cpp's 37.9, while non-attention time matches theirs.
   - Two per-chunk overheads were closed once the profile named them (the
     RoPE pair fused, the q-bias folded into its matmul): +8-9% at 500-2k.
   - Two costs worth knowing when reading a benchmark: a 1-token tail chunk
     costs 8-24 ms (decode walk plus lm_head), and every process pays
     ~250 ms of shader compilation, which a server amortizes and a
     one-shot CLI run does not.

   **Measured and rejected**, so nobody spends the day again:

   - Porting the flash kernel's QK^T and P.V to tensor ops costs 39% at 2k —
     the online softmax between the two matmuls breaks the
     cooperative-tensor dataflow.
   - The GQA redundancy that the decode kernel eliminates is NOT the prefill
     gap, though the per-head-layer numbers make it look like it. Two
     independent falsifications: variants that remove it moved SmolLM2 (9:3)
     *more* than Qwen (14:2), the opposite of what the hypothesis predicts,
     and llama.cpp's own flash-attention carries the same per-query-head
     redundancy — one threadgroup per query head, leaning on L2 to
     deduplicate the K/V reads.
   - Six structurally different attention kernels — the incumbent staged
     96-row tiling, direct-load variants at 96/32/16 rows, and a faithful
     reimplementation of llama.cpp's own kernel shape — land within ±3% of
     each other at 8k. The architecture axis is exhausted.
   - What the remaining ~16% actually is: with llama.cpp's flash attention
     switched off, our kernel beats their fallback by 16%; switched on, they
     lead by about the same. Their edge survives replicating their kernel's
     shape, so it lives below it — fully template-specialized per-head-dim
     kernels compiled offline into a metallib, against our single kernel
     compiled from source at startup. That is a toolchain difference, not a
     tiling one, and closing it means changing how lokal ships shaders.

   The 8,192-position ANE window remains the long-prompt head start; wider
   windows stay priced out by the superlinear first-load ANE compile (99 s
   at 6,144, 250 s at 8,192, 21+ min at 16,384).
5. The rest: an OpenAI-compatible API (`/v1/chat/completions`) and SSE
   streaming in serve mode; a hybrid scheduler that picks the backend
   automatically; CUDA/Vulkan backends on the same Engine/Session seam.
