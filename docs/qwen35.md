<!-- Vendored from the qwen35-research lane (artifact ea99643a), with the
     research's two inferences replaced by facts from the 2026-08-31
     metadata dump of the human's UD files. -->

# qwen35 (Qwen3.5 hybrid) — what it actually requires of lokal

Source: llama.cpp read-only clone (build 9960 era, cloned 2026-08-31), files
`src/models/qwen35.cpp` (graph + loader), `src/models/delta-net-base.cpp`
(deltanet math), `src/llama-hparams.cpp` (state formulas), `src/llama-arch.cpp`
(keys/names), `ggml/src/ggml-metal/ggml-metal-device.m` (Metal op support).
Everything below is CONFIRMED from source unless marked **inferred**.

## 0. Executive summary

qwen35 is a hybrid: per block it is EITHER a full-attention block (with two
twists: joint Q+gate projection and multi-section MRoPE) OR a gated-deltanet
linear-attention block (conv1d + delta-rule recurrent state), plus one optional
MTP block that standard generation SKIPS ENTIRELY. The recurrent state is small
(~150 MB total for the 27B — REAL dims, measured off the human's files
2026-08-31) while KV exists on only 16 of the 64 trunk blocks — the
architecture is memory-cheap for us, compute-new. llama.cpp ships
a FUSED Metal kernel for the whole delta step (`GGML_OP_GATED_DELTA_NET`,
ggml-metal-device.m:1743), plus an unfused op-by-op fallback — both are
readable references. Nothing here is windowed; lowmem/metal both can host it.

## 1. Block schedule and the 64-vs-65 discrepancy

- `n_layer() = n_layer_all − n_layer_nextn` (llama-hparams.cpp:322-324).
- MTP blocks are appended AFTER the trunk and "loaded as extra decoder blocks
  but not executed in the main pass" (qwen35.cpp, graph loop comment).
- Recurrency map: explicit `%s.attention.recurrent_layers` array if present,
  else `(i+1) % full_attention_interval != 0` with interval default 4
  (qwen35.cpp load_arch_hparams; key at llama-arch.cpp:243).
- SETTLED off the real files (metadata dump, 2026-08-31): NO explicit
  recurrent_layers array; `full_attention_interval = 4` → 16 trunk attention
  layers, and the census's 17th `attn_q` belongs to the MTP block itself.
  64-vs-65: Q3_K_XL has `nextn_predict_layers = 1` / block_count 65, IQ1_M has
  nextn 0 / block 64 — trunk is 64 in both. Loader rule: interval, with the
  array as an optional override.

## 2. Full-attention blocks (17) — op inventory (build_layer_attn)

Not vanilla qwen3. Order confirmed from qwen35.cpp:
1. `wq` projects to `(head_dim*2) * n_head` — INTERLEAVED per head:
   [q(256) | gate(256)] × 24 heads. Q = strided view (stride 2*hd), gate = the
   other half, materialized via cont.
2. Per-head RMSNorm on q and k (same as qwen3; k after reshape to heads).
3. **MRoPE**: `ggml_rope_multi` with `rope_sections[4]` from
   `%s.rope.dimension_sections` — NOT plain rope. hd=256, n_rot from metadata.
4. Standard causal GQA attention (24q/4kv, scale 1/√256 unless
   `f_attention_scale`).
5. **Output gating**: `attn_out * sigmoid(gate)` BEFORE `wo`.
6. `wo`: [n_head*256, 5120].
KV per layer = 2 · ctx · (4·256) · f16 — only 17 layers have KV AT ALL.
FFN: standard SwiGLU (`build_layer_ffn`), no MoE (asserted).
Residual structure: pre-norm attn + POST-attn norm before FFN (attn_post_norm
— a separate weight, not our post_attention_layernorm semantics; same math
slot as qwen3's though: norm between residual add and FFN).

## 3. Linear blocks (48) — op inventory (build_layer_attn_linear + delta-net-base)

Projections (per block tensors: attn_qkv, attn_gate, ssm_beta, ssm_alpha,
ssm_dt.bias, ssm_a, ssm_conv1d, ssm_norm, ssm_out):
- `qkv_mixed = wqkv(x)` → [2·key_dim + value_dim] where key_dim =
  d_state·n_group, value_dim = d_state·dt_rank (head_v_dim == d_state).
- `z = wqkv_gate(x)` [value_dim] — the output gate.
- `beta = sigmoid(ssm_beta(x))` [n_v_heads]; `alpha = ssm_alpha(x)`;
  `g = −exp(A_log)·softplus(alpha + dt_bias)` i.e.
  `ggml_softplus(alpha + ssm_dt) * ssm_a` (ssm_a stores −A_log.exp(),
  name LLM_TENSOR_SSM_A_NOSCAN).
- **Conv path**: concat(prev conv state [d_conv−1, C], qkv_mixedᵀ) →
  `ggml_ssm_conv` (depthwise, kernel [d_conv, C], C = d_inner + 2·n_group·d_state)
  → SiLU → split into q,k,v views; q,k get `ggml_l2_norm` per head; q,k
  repeat-broadcast k-heads→v-heads when unfused.
- **Delta rule** (three interchangeable forms in delta-net-base.cpp):
  - decode (n=1) `build_delta_net_autoregressive` — ~14 elementwise/reduce ops:
    s *= exp(g); sk = Σ_row(s∘k); d = (v − skᵀ)·β; s += k⊗d; o = Σ_row(s∘q).
  - prefill `build_delta_net_chunking` — chunked parallel form (CS=64):
    cumsum, tri masks, exp, `ggml_solve_tri` (unit-lower triangular solve!),
    batched 4-D mul_mats, then a PER-CHUNK sequential loop updating s.
  - fused `ggml_gated_delta_net` — single op, both shapes; **Metal kernel
    exists upstream** (supported when d_state % 32 == 0,
    ggml-metal-device.m:1743) — the reference for our kernel lane.
- **Output**: gated RMSNorm `rmsnorm(o; ssm_norm) * silu(z)` per head_v_dim,
  then `ssm_out` [value_dim, 5120]. No RoPE anywhere in linear blocks.

## 4. State model (formulas CONFIRMED, llama-hparams.cpp:183-233)

Per linear layer, per sequence:
- conv state `n_embd_r = (d_conv−1) · (d_inner + 2·n_group·d_state)` elements.
- delta state `n_embd_s = d_state · d_inner` elements
  (= d_state² · dt_rank; graph views it as [S, S, H_v]).
**Real dims for the 27B** (metadata dump off the human's files, 2026-08-31):
d_conv 4, d_state 128, n_group 16, dt_rank 48 → d_inner 6144;
conv = 3·(6144 + 2·16·128) = 30,720 el (120 KB f32); delta = 128·6144 =
786,432 el (3 MB f32). 48 layers: **~5.9 MB conv + ~151 MB delta ≈ 150 MB per
sequence (f32)** —
vs KV on the 17 attention layers: 17 · ctx · 1024 · 2 B · 2 ≈ 71 MB @1k ctx,
2.3 GB @32k. The recurrent state is CONSTANT in context length — the whole
point of the hybrid. State is read-modify-write per forward pass; llama.cpp
keeps optional rollback snapshots (n_rs_seq) — we need exactly ONE live state
per sequence for v1, no snapshots (greedy CLI/serve has no rollback).
lowmem arithmetic: +105 MB fixed to the budget's fixed pool (like activations),
NOT per-token; KV window math unchanged but applies to 17 layers only.

## 5. MTP verdict: SKIP, cleanly

Confirmed three ways: loader marks all nextn.* and the MTP block's weights
`TENSOR_SKIP` unless `ml.load_mtp` (an explicit opt-in); the main graph never
executes MTP layers; a separate `LLM_GRAPH_TYPE_DECODER_MTP` graph exists only
for draft-style decoding. For lokal v1: ignore every `nextn.*` tensor and any
block index ≥ trunk count, subtract MTP from block_count via
`nextn_predict_layers`. Zero quality cost for standard generation (it is a
draft head). Bonus fact: MTP-ONLY GGUFs exist (loader tolerates absent trunk)
— refuse those with a clear message.

## 6. head_dim 256 implications for our stack

- FLASH_HEAD_DIM=64 → our flash prefill never fires; hd=256 > DEC_TG=128 →
  our FUSED DECODE path never fires either: attention on the 17 layers runs
  the fallback prefill kernel for BOTH phases as the code stands. That kernel
  is now determinism-hardened (scores in barriers, metal-qwen3 lane) but it
  was measured ~tie with flash only at hd=64 — for 17 of 65 layers at 27B
  scale this is acceptable v1 and a later FC-specialization target.
- gqa_decode_dims threadgroup math: q_s = group·hd·4B = 6·256·4 = 6 KB — fits;
  our scratch audit from metal-qwen3 (size by q_dim = 24·256 = 6144 > hidden
  5120!) applies VERBATIM: every buffer sized by hidden must be re-audited —
  the Dims plumbing already carries true q_dim/kv_dim, so this is config, not
  surgery.
- MRoPE needs a new (small) kernel or a section-aware variant of ours: rope
  with per-section base positions — pure elementwise, no state.

## 7. Lane split proposal (Detoro rules)

- **A. qwen35-loader** (metal.rs untouched): parse qwen35 arch keys incl.
  recurrent_layers/interval/nextn; keep GGUF-NATIVE tensor names for this arch
  (no HF twin exists — extending hf_name would invent names; propose a
  per-arch name passthrough in the loader instead); MTP skip; census gate
  against all three UD files (the one command from §1). Boundary:
  src/lowmem/gguf.rs, manifest seam additions if any (through you). SMALL-MEDIUM.
- **B. deltanet-cpu-ref**: pure-Rust reference of conv step + decode-form
  delta rule + gated norm (the dequant_row_ref doctrine: the numerics oracle
  lane C tests bit-for-bit against). Feeds a unit-vector test set hand-derived
  from the formulas in §3. Boundary: a new reference module + tests. SMALL,
  and it unblocks C's gates before C exists.
- **C. qwen35-metal-kernels**: conv1d, the decode-form fused step (start
  autoregressive: ~14 ops, one token — this is the shape our serial decode
  encoder loves), MRoPE, attn out-gate. Prefill v1 = run the decode form
  token-by-token on GPU (correct, slow-ish); chunked/fused kernel as a
  follow-up with ggml's Metal `gated_delta_net` kernel as the reference.
  Boundary: kernels.metal + metal.rs. LARGE — the real work.
- **D. qwen35-session-state**: state buffers beside KV (17-layer KV + 48-layer
  conv/delta states), session reset semantics, serve per-request states,
  lowmem budget line. Boundary: metal.rs session + lowmem plumbing. MEDIUM.
A→B→(C∥D) ordering; C's gates: vs B bit-for-bit (decode form), vs llama.cpp
greedy 5/5 on a small qwen35 checkpoint (0.8B/2B exist per the type table —
qwen35.cpp:29-33), determinism at 4+ runs, and the scratch-audit gate from §6.

Note: `rope.dimension_sections = [11, 11, 10, 0]` on the real files (MRoPE
splits the 256-dim rotation into 11/11/10 frequency sections; last section 0).

## 8. Verification commands (run against the human's real files)

    LOKAL_GGUF_INFO=1 lokal -b metal -m <file>          # tensor census
    (dump keys) qwen35.attention.recurrent_layers, .full_attention_interval,
    .nextn_predict_layers, .block_count, .ssm_{conv_kernel,inner_size,
    state_size,time_step_rank,group_count}, .rope.dimension_sections
