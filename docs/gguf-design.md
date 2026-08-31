# GGUF Support Design — lokal

Adapted for lokal from the human's GGUF v2 spec (2026-08-31; verbatim original:
workspace artifact 04a69bce / scratchpad spec-gguf-v2.md). This file is the
canonical version: every section is mapped onto lokal's real modules, backends,
and shipped work. Landed by lane gguf-unify; amended in-repo from here on.

## Core Principle (the human's words, binding)

> **อย่าทำ GGUF backend เป็นอีก model implementation** —
> ให้ GGUF เป็นแค่ input format ที่แปลงเข้าสู่ `ModelConfig` เดียวกับ SafeTensors
> แล้วให้ CPU / Metal / Hybrid ใช้ graph เดียวกัน

GGUF is a container/storage format, not an architecture. What lokal supports is:
GGUF metadata, tensor layout, quantization types, model architectures, attention
mechanisms, position encodings, FFN/activation, normalization, and per-layer
state semantics (KV cache AND recurrent state — lokal already runs a hybrid
recurrent model, which the generic spec did not cover).

## Target architecture (lokal modules)

```text
   model.safetensors            model.gguf
        │                           │
   weights.rs                src/gguf/container.rs   (v2/v3 parse, mmap)
   (SafeTensors mmap)        src/gguf/dequant.rs     (GgmlType, 17 types, IQ grids)
        │                    src/gguf/arch.rs        (metadata → ModelConfig)
        │                    src/gguf/tokenizer.rs   (BPE from metadata)
        └───────────┬───────────────┘
                    │
              ModelConfig (src/config.rs — ONE type for both loaders)
              TensorStore seam (L3: trait over both mmap sources)
                    │
                Model graph (forward paths)
                    │
      ┌─────────┬───┴─────┬──────────────┐
     cpu      metal     lowmem         hybrid
   (oracle) (gpu/metal.rs) (budgeted   (ANE+Metal prefill,
                            streaming)  Metal decode)
```

- Backend set is lokal's real one: `-b cpu | metal | lowmem | hybrid`. ANE is a
  role inside hybrid (prefill), not a standalone serving backend — see §ANE.
- Loader dispatch is already by extension + magic; no special flag. The hub
  resolver accepts `owner/repo:TAG` for GGUF downloads (c7b5563).

## Current state (main @887a0ec) — what is already true

Shipped, do not re-implement:
- GGUF container/metadata parse; 17 quant types incl. K-quants and IQ grids (676f172).
- Quantized execution directly on Metal — no cpu-dequant-then-copy on the metal
  path (c7b5563). The lowmem backend dequantizes CPU-side BY DESIGN: it is the
  memory-budgeted streaming backend; that is its mechanism, not a violation.
- qwen35 (Qwen3.5 gated-deltanet hybrid) executes on `-b lowmem` (d7baf25):
  joint Q+gate projection split, partial RoPE (rot_dim = 2*sum(sections)), KV
  cache only on the 6 attention layers, MTP tensors skipped.
- Split prefill (default), window-mode (sliding-window semantics), graph cache.
- Identity-gate practice = spec §19 (see §Validation). Bench collectors =
  spec §20 groundwork (benchmarks/collect-gguf-rows.sh, collect-metal-quant.sh).

Gaps (the roadmap, in order): arch abstractions (L3); GGUF through hybrid
(L4); MoE (L5) and MLA (L6) once target models are named. L1 (structure move
out of src/lowmem, with the backend×format matrix) and L2 (deltanet on metal)
have landed.

## Backend × format matrix

Every cell below was established by RUNNING it (L1-B1 survey, main @887a0ec +
the L1 lane). Rule: every cell either runs, or refuses with a one-line
mechanism-named reason asserted by a unit test (`gguf_backend_refusal` for the
GGUF rows, the config.rs arch refusal for safetensors). No silent backend
fallback, ever. Since L2 (metal-deltanet), qwen35 GGUF runs on `-b metal`
as well as `-b lowmem`.

| backend | safetensors dense | safetensors Qwen3/Qwen3.5 | GGUF dense | GGUF qwen35 (deltanet) |
|---|---|---|---|---|
| `cpu` | runs | refused by name | runs — `load_f32` expansion (fits-in-RAM guarded) | refused: no gated-deltanet path |
| `metal` | runs | refused by name | runs — direct quant execution, no expansion | **runs** (L2) |
| `lowmem` | runs | refused by name | runs — budgeted streaming, CPU-side dequant by design | **runs** (d7baf25) |
| `hybrid` (alias `ane`) | runs | refused by name | refused: ANE prefill graphs are exported from safetensors (L4 opens this) | refused: no gated-deltanet path |

The safetensors Qwen3-family refusal is config.rs's arch check (explicit
head_dim + qk-norm break the Llama walk; the GGUF path is the wired one). The
GGUF refusal column is one pure function the CLI and the tests share, so the
printed reason and the tested reason cannot drift.

## Attention & per-layer state

Mechanisms lokal must express (spec §6): MHA, GQA, MQA (GQA with kv_heads=1),
sliding window (shipped as window-mode), MLA (deferred, L6), and — beyond the
generic spec — gated deltanet (shipped, qwen35). GQA-aware kernels exist on the
metal path; the flash prefill/decode kernels are the reuse targets.

State seam (LANDED, L3): per-layer state is explicitly two-kind —
`LayerStateKind::{Kv, Recurrent}` with `state_schedule()` as the one deriving
function (gpu/metal.rs), adopted at session construction; recurrent state is
`DeltaNetStates` (conv window + delta state). A
hybrid-recurrent model schedules both kinds across its layer list (qwen35:
KV on 6 attention layers, interval-4 schedule from gguf/arch.rs metadata).

## Position encoding

RoPE parameters are already generic (`RopeParams`, incl. partial rotation via
`rot_dim`) — resolved from metadata, never hard-coded per model. MRoPE
text-broadcast is proven equal to plain RoPE (negative-controlled); no sectioned
kernel exists or is planned. New encodings (ALiBi/NoPE) enter as `RopeParams`-
level variants only when a target model needs them.

## FFN / activation / normalization

Today: SwiGLU with a fused Metal kernel — the reuse target for every compatible
model. LANDED (L3): `Activation::SwiGLU` and `NormType::RmsNormPre` on
ModelConfig (config.rs), resolved from checkpoint metadata (`hidden_act`;
unknown names refuse by name) and matched exhaustively at construction, before
any layer loop. The rule stands: an enum variant is added when a target model
actually needs it; until then the seam stays single-variant. Do not assume Llama-style RMSNorm in new code — read it
from ModelConfig.

## MoE

`FeedForward::{Dense(DenseFfn), MoE}` LANDED (L3, model.rs): Dense is the only
constructor; the MoE arm is an unreachable shell carrying no invented router
layout, so L5 changes shape (a constructor + an arm), not seams.
Real MoE (router, top-k experts, combine) is L5 and BLOCKED on the human naming
a target model — expert layouts in GGUF vary by family and we do not build
against a hypothetical checkpoint.

## Quantization

Storage dtype → quant decode → compute backend stay separated. The 17 supported
GGML types (F32/F16/BF16, Q8_0..Q4_0, K-quants, IQ types incl. IQ1) decode via
one reference (`dequant_row_ref`, gguf/dequant.rs) that both the CPU oracle and
the Metal kernels are gated against. On metal: dequant/matmul on-GPU. On lowmem:
CPU-side within the memory budget (by design). Mixed-type files (e.g. Q4_K_M
carrying Q5_0 tensors) are normal — per-tensor type, never per-file assumptions.

## Metal backend

Reuse the shipped kernel set — tensor-op matmul, FlashAttention prefill, flash
decode, GQA-aware attention, F16 KV cache, fused QKV, fused SwiGLU — for every
compatible model; no new execution paths without necessity. The deltanet block
kernels (ssm_conv_decode, delta_decode_step, delta_gates) exist for the lowmem
oracle; their metal-side execution landed in lane L2 (metal-deltanet), gated
byte-identical against `-b lowmem` on Qwen3.5-2B.

## ANE / Hybrid

ANE participates ONLY as hybrid's prefill engine, and only for graphs that
compile to Core ML. Hard limits are measured and recorded (MLState spike,
artifact 905296f7): stateful ANE decode is a NO-GO (~6MB state ceiling,
16-layer load wall, 0.74ms dispatch floor). Therefore, exactly as spec §13–14
says: prefill → ANE+Metal where compatible, else Metal; decode → Metal, always.
GGUF models flow through hybrid in L4 by reusing the existing split-prefill
path; GGUF support must not harden any hybrid dependency — hybrid remains one
backend among four.

## Validation (the gate doctrine)

Every model, every lane: CPU reference → Metal → Hybrid (where supported),
greedy (`-t 0`), plus an EXTERNAL oracle — llama.cpp greedy agreement on the
same GGUF (identity gates prove sameness, not rightness; the external reference
proves rightness). Three-tier numerics doctrine for recurrent paths: bit-exact
where achievable, measured-stateless and measured-conditioning tiers otherwise,
with negative controls. Quantized models: token-level agreement expected at the
quant level actually stored; numeric tolerance follows quantization.

**A cell that runs has been shown to execute, not shown to be correct; only a
cross-engine or cross-format comparison shows the second.** (Mellow, from the
rope-mirror regression: the metal×GGUF cell was surveyed as "runs" at the same
SHA that shipped it producing fluent, wrong output. Running a cell proves it
executes. The same trap caught the gguf-loader q/k permute, which passed
cpu==metal by being consistently wrong on both sides.)

**When two engines disagree, also diff each engine against ITSELF: prefill
against decode.** (Mellow, from metal-deltanet. metal≠lowmem on qwen35 came
down to one line — prefill applied qk-norm to the raw joint Q+gate buffer that
the split had already consumed, so attention ran on an un-normalized Q, while
metal's own decode path already did it correctly. Reading metal-prefill against
metal-decode against lowmem-prefill found in minutes what a long run of
cross-engine feature-toggling had only narrowed. Corollary, learned the
expensive way on the same lane: a probe's EXCLUSION is a measurement and can be
wrong. "Toggling qk-norm changes neither hash" was recorded as settled and sent
the hunt to the wrong half of the layer for most of a session. Before trusting
an exclusion, confirm the feature it excludes is even live on that
checkpoint — qwen35 does carry attn_q_norm/attn_k_norm.)

## Benchmarks

Per GGUF model: model, quant, params, context, prefill tok/s, decode tok/s,
peak RAM, Metal memory. Rungs: 500 / 2K / 4K / 8K / 16K; long-context adds
32K / 64K. Collection via the agent-free collectors (benchmarks/); tok/s rows
measured on a non-memory-quiet box are REQUIREMENT statements, not results —
final rows wait for a memory-quiet window (recorded protocol).

## Structure & naming (the standing improvement)

GGUF code lives in src/gguf/ (container/dequant/arch/tokenizer + testutil),
never inside a backend. The CPU oracle is src/deltanet_ref.rs. Naming rule
(DESIGN.md-bound): arch names appear ONLY where behavior keys on the arch
string; mechanisms carry mechanism names (DeltaNet*, never Qwen35* for shared
machinery, never Hybrid* for non-backend code). LowMemSource stays in lowmem —
constructor-side and backend-specific.

## Implementation order (spec order → lokal lanes)

| spec step | status |
|---|---|
| 1 GGUFLoader | done |
| 2 metadata → ModelConfig | done (gguf/arch.rs); ONE shared type verified in L1-B3 |
| 3 architecture detection | done (arch string + refusal-by-name honesty) |
| 4 tensor abstraction | done (L3: `TensorStore` in weights.rs — eager map + LowMemSource behind one trait, `Model::from_store` consumes it) |
| 5 F16 GGUF | done |
| 6 GQA/MHA mapping | done |
| 7 RoPE parameters | done (partial rope incl.) |
| 8 Q8..Q4 (+IQ) | done — 17 types |
| 9 Metal quantized matmul | done — dense and the deltanet block (L2) |
| 10 more dense variants (Gemma/Phi/…) | after L3, per human-named targets |
| 11 MoE | L5, blocked on target model |
| 12 MLA / advanced | L6, blocked on target model |

End state: `lokal -m model.gguf -b cpu|metal|lowmem|hybrid` — backends never
know which container the model came from; users never need to know RoPE/GQA/
SwiGLU/Q4_K are involved. All of it resolves from metadata + ModelConfig.
