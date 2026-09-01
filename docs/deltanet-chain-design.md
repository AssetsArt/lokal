# Batching and fusing the gated-deltanet chain — design memo

Lane `deltanet-chain-design`, design-only. Base `main @398feb6`. No engine code
is changed by this document; §7 is written to be lifted into an implementation
plan when the metal files free up.

**The one-line result.** The chain's cost is not spread over six kernels and it
is not launch overhead. It is *one* kernel, `delta_decode_step`, at 96% of the
chain, running at 9.6–14.7 GB/s on a box whose matvecs reach 65–70. The
smallest fix — transposing the delta state so adjacent threads read adjacent
addresses — is measured at **2.10× (dev dims) / 3.65× (27B dims), bit-identical**,
and it does not touch the summation order the qwen35 arc froze.

## 0. Provenance — what was read and what was measured

Everything cited from llama.cpp is from the read-only checkout at
`~/.unsloth/llama.cpp`. It carries **no git metadata**, so it is pinned by
content hash rather than by a commit; files are dated 2026-08-09. Line numbers
below are that checkout's, and a reader who finds them shifted should re-hash
before assuming the claim moved:

| file | sha256 |
|---|---|
| `src/models/delta-net-base.cpp` | `5e3b318953f854b3825dd7cfe4b483ca7935cd5f6680438a3ece4be9b4d5428f` |
| `ggml/src/ggml-metal/ggml-metal.metal` | `5d577d20a699016d108b83d517fe49c716c635e4dee46a0538157542a8789130` |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | `e4eaf6dc3a528089f943d42249e64aeab4ddd1ce9bbfce7f77ce8b4ebb3566da` |
| `ggml/src/ggml-metal/ggml-metal-device.m` | `7f1b54baf7d789fc90debc2ff8a1301ca770c3ba92e254e3eab3c1e4946cac9c` |

`docs/qwen35.md` (this repo) is the architecture canon and its citations come
from a *different* clone — "build 9960 era, cloned 2026-08-31". Where this memo
and that one both cite a line number, they may disagree by a few lines; the
facts agreed on re-reading.

Measurements are from a standalone benchmark that compiles **this repo's own**
`src/gpu/kernels.metal` at `398feb6`, so it times the shipped kernels rather
than a copy. Its whole-chain total reproduces the in-engine attribution
(39.8 µs/dispatch, lane `metal-perf-attribution`) to within 1.5%, which is the
check that says the harness measures the right thing. Raw logs live in the lane
scratchpad as `dnbench.log` and `coalesce.log`. Machine state: quiet and
memory-quiet per `protocol:gpu-bench` (12 swapins / 19,238 decompressions across
the first batch; 111.8% foreign CPU and 147 MB free for the second).

Dimensions used throughout, read directly out of the GGUF headers, not assumed:

| | dev (0.8B **and** 2B) | 27B |
|---|---|---|
| `ssm.state_size` S | 128 | 128 |
| `ssm.group_count` H_k | 16 | 16 |
| `ssm.time_step_rank` H_v | 16 | 48 |
| `ssm.inner_size` d_inner | 2048 | 6144 |
| `ssm.conv_kernel` d_conv | 4 | 4 |
| key_dim = S·H_k | 2048 | 2048 |
| C = 2·key_dim + d_inner | 6144 | 10240 |
| linear layers | 18 | 48 |
| delta state / layer | 262,144 el (1.05 MB) | 786,432 el (3.15 MB) |

**The 0.8B and the 2B have byte-identical deltanet dims** — only
`embedding_length` and `feed_forward_length` differ. Any argument of the form
"it is invariant across those two models, therefore it does not depend on size"
is void, and one such argument was retracted before this memo was written.

## 1. What the chain actually costs

Each row is the median of three passes, N dispatches back-to-back **inside one
encoder**, so no encoder-boundary cost is included. The floor row is the same
kernel at the same dims with its work turned down to nothing — `delta_gates`
over one head, one threadgroup in which 255 of 256 threads return immediately.
That is a direct measurement of launch cost, not an inference from invariance.

| kernel | dev dims | 27B dims |
|---|---|---|
| **launch floor** (1-head dispatch) | **2.12–2.24 µs** | **2.13–2.14 µs** |
| `delta_gates` | 2.23 µs | 2.20–2.23 µs |
| `l2norm_rows` (×2/token) | 2.36–2.44 µs | 2.43–2.55 µs |
| `gated_output_norm` | 2.95–3.02 µs | 2.88–3.13 µs |
| `ssm_conv_decode` | 5.03–5.97 µs | 6.55–6.64 µs |
| **`delta_decode_step`** | **223.8–227.1 µs** | **426.9–483.3 µs** |
| whole chain (6 dispatches/layer) | 4.23 ms | 8.01–9.09 ms |
| chain per-dispatch *average* | 39.2 µs | 74–84 µs |
| launch floor × 108 dispatches | 0.23 ms = **5.4%** | 0.23 ms = **2.5–2.9%** |

Three things follow, and the first two overturn the lane split this memo was
commissioned to detail.

**(a) The 39.8 µs/dispatch from the attribution lane is an average over kernels
that differ by a factor of a hundred.** It was never a per-dispatch fixed cost.
Five of the six kernels cost 2.2–6.0 µs against a 2.1–2.2 µs floor; they are
essentially free and essentially all launch.

**(b) Fusing six kernels into one — task B2 as planned — is worth at most 4.5%
of the chain.** Five dispatches per layer per token, at ~2.2 µs of floor each,
is 11 µs against a 39.2 µs × 6 = 235 µs layer-token. This is a measured
negative result and §7 records it as one so that it is not re-proposed in a
month. It does not become worthwhile after the transpose either: it grows to
~9% of a smaller chain, which is still not a lane.

**(c) The chain is one kernel.** `delta_decode_step` is 96% of it. Everything
below is about that kernel.

## 2. Why that kernel is slow, and the fix that keeps the constraint

`delta_decode_step` (`src/gpu/kernels.metal:3601`) gives thread *(h, j)* the
state column `state + h·S·S + j·S` and walks `st[i]` for i = 0…S−1. Within a
thread that walk is contiguous. Across threads it is not: neighbours *j* and
*j+1* are S·4 = **512 bytes apart**, so at each load instruction the 32 lanes of
a SIMD group request 32 distinct cache lines and consume four bytes of each.

Measured useful bandwidth (counting one read and one write per element; the
kernel actually does two of each):

* dev dims — 37.7 MB in 4.09 ms = **9.6 GB/s**
* 27B dims — 113.2 MB in 7.69 ms = **14.7 GB/s**

The same box sustains 65–70 GB/s on every quant matvec (lane
`metal-perf-attribution`) and peaks near 200.

### 2.1 The layout is the defect, not the parallelisation

The standing constraint from the qwen35 arc (workspace memory `ef397862`) is:

> Delta decode-form split one-thread-per-(v-head, out-column) preserves exact
> summation order; splitting the contraction index forfeits bit-equality.

It constrains **which thread sums which terms**. It says nothing about **which
address a lane reads**. Storing the state transposed — element (i, j) of head h
at `h·S·S + i·S + j` instead of `h·S·S + j·S + i` — makes adjacent threads read
adjacent addresses while each thread still walks i in the reference's order. The
arithmetic is untouched; only the address expression changes, from `st[i]` to
`st[i·S]` with `st = state + h·S·S + j`.

**Measured, not argued.** Two kernels, same math, same order, same thread count,
same dims, in the same run; the experiment asserts bit-exactness *before* it
reports any timing, because a faster kernel that moved one bit would be
worthless here:

| | shipped layout | transposed | speedup | bit-exactness |
|---|---|---|---|---|
| dev (H_v 16, 2048 threads) | 218.8 µs / 9.6 GB/s | 102.6–104.9 µs / 20.0–20.4 GB/s | **2.08–2.13×** | 0 output, 0 state mismatches |
| 27B dims (H_v 48, 6144 threads) | 427.5–427.9 µs / 14.7 GB/s | 116.6–117.2 µs / 53.7–54.0 GB/s | **3.65–3.67×** | 0 output, 0 state mismatches |

At 27B dims the transposed kernel reaches 54 GB/s — the rate this box
demonstrably sustains elsewhere. The thread count is identical in both columns,
which is what rules out the competing explanation that the kernel is simply
short of threads.

### 2.2 What upstream does, and the half of it we must not copy

`kernel_gated_delta_net_impl` (`ggml-metal.metal:2649`; a second definition of
the same name at :2792 sits inside the `#else` of an `#if 1` and is dead code,
marked upstream "a simplified version... no performance improvement") stores the
state
transposed for exactly this reason — its own comment at 2678–2680 reads "state
is stored transposed: M[i20][is] = S[is][i20], so row i20 is contiguous". It
then **also** splits the contraction across SIMD lanes and reduces with
`simd_sum` (2730 and 2743). That second half is what the standing
constraint forbids, and upstream needs it for a reason that does not apply to
us: it keeps the state in *registers* across its token loop, and a full column
of S = 128 floats does not fit in one thread's registers, so it must be spread
over 32 lanes × NSG ∈ {1,2,4}.

The op is gated on `has_simdgroup_reduction && op->src[2]->ne[0] % 32 == 0`
(`ggml-metal-device.m:1384-1385`); S = 128 passes. Dispatch and pipeline
selection are at `ggml-metal-ops.cpp:1813` and `:1829`.

**So: take the layout, leave the reduction.** We get coalescing without giving
up bit-equality, and we need no renegotiation of `ef397862`.

## 3. The prefill problem, and three candidate forms

`enc_delta_block` (`src/gpu/metal.rs:2419`) loops `for t in 0..n` and issues the
six kernels per token, so a 512-token prefill chunk pays 512 decode steps. The
attribution lane measured the consequence: 1.03× the decode cost per token,
i.e. no batching benefit at all, 30% of 2B prefill and 13.8% of 27B prefill.

llama.cpp chooses its form at `delta-net-base.cpp:425-448`: `n_seq_tokens == 1`
→ `build_delta_net_autoregressive` (:289), otherwise
`build_delta_net_chunking` (:16); either is replaced by
`build_delta_net_fused` (:373) when the corresponding `cparams.fused_gdn_*`
flag is set.

### 3.0 The recurrence, dimension-checked against the reference

Every shape below is one the reference *asserts*, not one this memo asserts.
`deltanet_ref::delta_decode_step` (`src/deltanet_ref.rs:135`) checks each of them
on entry, so a later form can be checked against the code rather than against
prose:

| tensor | shape | reference's own check |
|---|---|---|
| `state` | S·S·H_v | `assert_eq!(state.len(), dims.delta_state_elems())` |
| `q`, `k` | S·H_k | `assert_eq!(q.len(), s_dim * hk)` (and `k`) |
| `v` | S·H_v | `assert_eq!(v.len(), s_dim * hv)` |
| `g`, `beta` | H_v | `assert_eq!(g.len(), hv)` (and `beta`) |
| `out` | S·H_v | `vec![0f32; s_dim * hv]` |
| `group` | H_v / H_k | `let group = hv / hk;` — 3 on the 27B, 1 on the dev models |

and `delta_state_elems() = d_state · d_state · n_v_heads` = S²·H_v = d_state·d_inner
(`deltanet_ref.rs:40-42`), which is the identity that lets §0's table quote
786,432 elements for the 27B and 262,144 for the dev models.

Per V head h, with K head ⌊h/group⌋, for output column j:

```
q̂    = q[h/group] · S^(-1/2)                    [S]
s    ← s · exp(g[h])                            [S,S]
sk_j = Σ_i s[i,j] · k[h/group][i]               scalar — contraction over i
d_j  = (v[h][j] − sk_j) · β[h]                  scalar
s[i,j] += k[h/group][i] · d_j                   [S], column j only
o_j  = Σ_i s[i,j] · q̂[i]                        scalar
```

**i is the CONTRACTION index; j is the OUTPUT index, and the whole memo turns on
keeping them apart.** The standing constraint freezes the summation over i. The
transpose in §2.1 changes only where (i, j) lives in memory. P2 below changes
only where it lives *between tokens*. None of the three touches the order of the
sums, which is why §6 can put all of them in tier 1.

The other four stages, for completeness: `conv_step` (`deltanet_ref.rs:91`,
state [C][d_conv−1] oldest-first, SiLU inside, window rolls *after* the dot),
`split_qkv` (:116, conv output cut [key_dim | key_dim | d_inner] then l2-norm per
q/k head), `delta_gate` (:82, `g = a·softplus(alpha + dt_bias)`; β is sigmoided
by the caller), and `gated_output_norm` (:187, per-head RMSNorm then `· silu(z)`).

### P1 — today: the decode form, per token
6 dispatches per layer per token. Dispatch counts in §5.

### P2 — persistent state, one dispatch per layer per chunk
This is the shape of upstream's *fused* op, and the important thing about it is
that **it is not the chunked algorithm**: the Metal kernel runs the ordinary
autoregressive recurrence with a `for t = 0; t < args.ne22; t++` loop *inside*
the kernel (`ggml-metal.metal:2708`), keeping the state live across tokens
instead of round-tripping it to device memory. Same arithmetic as P1, same order,
1/(2n) of the state traffic.

For us the state cannot live in registers (§2.2), but it can live in
**threadgroup memory**. The device reports 32,768 B of threadgroup memory and
1024 threads per threadgroup (measured on the dev M1 Pro). One column is S = 128
floats = 512 B, so a threadgroup can hold 64 columns; 32 columns (16 KB) leaves
room for occupancy. Columns are independent — column j is read and written only
by iteration j of the reference's j loop — so a tile of columns is
self-contained and needs no cross-thread communication at all. With H_v·S
columns total that is 6144/32 = 192 threadgroups on the 27B, which is ample
parallelism.

Bit-exactness survives: each thread performs the reference's operations on its
own column in the reference's order, for each token in sequence.

**P2 is the recommendation for prefill.** It is a straight-line kernel, it
inherits the transpose's coalescing for its one load and one store, and it is
bit-exact by the same argument as §2.1.

### P3 — the chunked delta rule (studied, not recommended for v1)
`build_delta_net_chunking` (:16) with `CS = 64` for a non-KDA model (:61 —
`kda` is false for us because `g->ne[0] == 1`, one scalar per V head, asserted
at :31). Per (head, chunk) it computes, with our names:

1. `g_cs` = cumulative sum of g along the chunk, [CS] (:89)
2. `decay_mask[i][j] = exp(tril_diag(g_cs[j] − g_cs[i]))`, [CS,CS] (:133-137)
3. `kb = (k·k_bᵀ) ⊙ decay_mask`, `kq = (k·qᵀ) ⊙ decay_mask`, [CS,CS] (:139, :143)
4. `attn = (I + tril(kb, −1))⁻¹` via `ggml_solve_tri` on a unit-lower system
   (:152-167) — the UT transform
5. `v_new_basis = attnᵀ · v_bᵀ` (:171), `k_cd = attn · (k_bᵀ ⊙ exp(g_cs))` (:183)
6. a **sequential loop over chunks** (:235-282) carrying the state:
   `v' = k_cdᵀ·s`; `v_new = v − v'`; `o = q_g_exp·s + kq·v_new`;
   `s = s·exp(g_last) + kgᵀ·v_new`

Dimensionally this is consistent with our layout: `s` is [S, S, H_v] (:229
reshapes to `S_v, S_v, 1, H_v*n_seqs`), which is `DeltaNetLayout::delta_elems =
d_state·d_inner = S·S·H_v` and matches `deltanet_ref::delta_decode_step`'s
`assert_eq!(state.len(), dims.delta_state_elems())`.

Why not v1: (i) it needs a triangular solve and `ggml_tri`/`ggml_cumsum`
equivalents we do not have; (ii) it does ~2.8× more arithmetic than the
recurrence, traded for parallelism we do not currently lack — the delta step
runs at 19 GFLOP/s of a ~10 TFLOP/s device, so we are three orders of magnitude
from being arithmetic-bound and P2 recovers the traffic without new numerics;
(iii) it is a genuinely different algorithm, so it lands in tier 3 of the
numerics doctrine (§6) where P2 lands in tier 1. Recorded here so the next
person does not have to re-read 280 lines of ggml graph code to reach the same
conclusion.

## 4. The conv, gates and norms batch trivially

They are 4% of the chain, so this is about dispatch structure, not time — but
P2 is pointless if the other five kernels still run per token.

* `ssm_conv_decode` (`kernels.metal:3475`) is a depthwise causal conv with
  d_conv = 4. Over a chunk it is a plain causal convolution with the incoming
  `(d_conv−1)`-wide window prepended: every output position depends only on the
  input, never on another output. **Fully parallel over the chunk — 2 dispatches**
  (read + roll: every early token READS the window the roll WRITES, and one
  dispatch cannot do both without a cross-threadgroup race — corrected at
  landing, lane deltanet-prefill-batching).
  The rolling-state write becomes a single store of the chunk's last
  `d_conv−1` columns. llama.cpp does the same thing with one `ggml_ssm_conv`
  over the concatenated state and inputs (`delta-net-base.cpp:449-470`
  builds that concatenation).
* `delta_gates` (:3717) is per token per head, elementwise. **1 dispatch.**
* `l2norm_rows` (:3541) is per (token, k-head) row. **2 dispatches.**
* `gated_output_norm` (:3662) is per (token, v-head) row. **1 dispatch.**

One caution that costs correctness if missed: the conv rolls a **read-modify-write**
state, and three v-heads share one k-head's q/k channels (group = H_v/H_k = 3 on
the 27B). Any fusion that lets several threadgroups roll the same channels races.
Keeping the conv as its own dispatch, as above, avoids the question entirely —
which is a second reason not to chase B2.

## 5. Dispatch arithmetic

Per linear layer, per prefill chunk of `n` tokens: **P1 = 6n**, **P2 = 6**
(conv 1 + gates 1 + l2norm 2 + delta 1 + output norm 1).

27B, 2198-token prompt (5 chunks of ≤512), 48 linear layers:

| | dispatches | measured / projected |
|---|---|---|
| P1 (today) | 2198 × 48 × 6 = **633,024** | 17.55 s measured (Studio), 13.8% of prefill |
| P2 | 5 × 48 × 7 = **1,680** | launch cost ≈ 1,680 × ~2.2 µs ≈ **3.7 ms** |

(7 per layer per chunk as landed — gates 1, conv 2, l2norm 2, delta 1, gated
norm 1; the original 6 assumed a single conv dispatch, see §4.)

The launch cost stops mattering; what remains is the delta step's own work and
its state traffic, and P2 cuts that traffic from 4 accesses per element per
token to 2 per element per **chunk** — a factor of ~1000 at n = 512.

Decode is unchanged in shape (n = 1, so P1 ≡ P2) and 288 dispatches/token on the
27B stays 288. Its win is the transpose alone:

| | today | with the transpose |
|---|---|---|
| 2B decode, delta chain | 4.30 ms/token | ≈ **2.0 ms/token** (2.10× on 96% of it) |
| 2B decode step total | 25.1 ms → 38.0 tok/s | ≈ 22.8 ms → **≈ 41.8 tok/s** (+10%) |
| 27B decode, delta chain (Studio) | 8.46 ms/token | ≈ **2.4 ms/token** (3.65× measured at these dims) |

The 27B row is a cross-box projection: the ratio is measured at 27B *dims* on
the dev box, the baseline is measured on the Studio. It is the shakiest number
here and is marked as such; the Studio rerun after the lane lands is what
confirms it. Today that saves ~1.2% of a 498 ms IQ1_M step — but the delta share
grows as `quant-matvec-rework` shrinks everything around it, which is precisely
why it is worth doing and precisely why it must land *after* that lane.

## 6. Numerics gates, decided here — `protocol:gate-scripts` rule 14 tiers

The doctrine: tier 1 bit-exact for pure add/multiply with fma contraction off;
tier 2 measured ULP bounds for stateless transcendentals; tier 3
measured-conditioning bounds for recurrent kernels, each carrying a
**constructed bit-exact half**. Reference is `src/deltanet_ref.rs`; the existing
oracle is `src/gpu/metal.rs::deltanet_kernel_oracle`.

**T1 — the transpose. Tier 1, and against a stronger reference than usual.**
The change is address arithmetic only, so the gate is not "close to the
reference" but "bit-identical to the *current kernel*", which is a strictly
stronger claim than the oracle it already passes. Both halves:
* *bit-exact half (the whole thing)*: for randomised state, q, k, v, g, β at
  both dim sets, the transposed kernel's output **and** its resulting state must
  match the shipped kernel's bit-for-bit under the layout permutation. Already
  demonstrated in the lane benchmark — 0 output and 0 state mismatches at
  S=128/H_v=16 and S=128/H_v=48 — and it lifts into the oracle directly.
* *negative control*: `deltanet_kernel_oracle` already carries a transposed
  control, put there because "reading it transposed gives plausible finite
  numbers and wrong answers" (`kernels.metal:3598-3600`). After this change that
  control must be **re-pointed, not deleted** — it now has to fire on the *old*
  orientation. A lane that deletes it has removed the only guard against doing
  the transpose half-way.
* `#pragma clang fp contract(off)` stays exactly where it is. Metal contracts
  `a*b+c` into an fma and fast-math being off does not stop it (workspace memory
  `c35de604`); the pragma is what makes tier 1 achievable at all.

**T2 — the persistent-state prefill kernel (P2). Tier 1 for the recurrence,
tier 2 for what it inherits.**
* *bit-exact half*: with the token loop run for n = 1, P2 must be bit-identical
  to the decode kernel — same thread mapping, same column, same order. This is
  the constructed half, and it is free: it is just the decode case of the same
  kernel.
* *the real gate*: for n > 1, bit-identical to n applications of
  `deltanet_ref::delta_decode_step`, because P2 changes only *where the state
  lives between tokens*, never the arithmetic or its order. Tier 1 is the right
  bar and anything less would be hiding a defect.
* *the gate that catches the actual risk*: threadgroup-memory staging is where a
  tile boundary bug lives. The prompt must exceed one tile **and** one prefill
  chunk — n > 512 with H_v·S/32 threadgroups — for the same reason rule 10 exists
  (a 5-token gate passed a buffer-overflow bug) and rule 14's split/merge lesson
  (a short prompt leaves the interesting path unexecuted).
* *transcendentals*: `exp(g)` per token per head is unchanged from the current
  kernel, so it inherits the existing tier-2 bound rather than needing a new one.

**T3 — the chunked form (P3), if it is ever built. Tier 3, and its half named
in advance.**
* Not comparable bit-for-bit: `solve_tri` and the cumsum reassociate the
  recurrence by construction.
* *constructed bit-exact half*: **g = 0** makes every `exp(g_cs)` exactly 1.0 and
  the decay mask exactly 1, collapsing the chunked form to a plain triangular
  accumulation that must match the reference bit-for-bit. A second, sharper one:
  **β = 0** makes `d = 0`, so the state never updates and `o = q·s` for the
  entering state — a pure matmul with a closed form.
* *bounded half*: against measured conditioning, never a fixed tolerance — the
  reference's own drift under a single 2⁻²⁴ perturbation of g, times a factor
  derived from the fact that heads never mix (`delta_decode_step` indexes head h
  only). The existing `delta_decode_bounded_once_the_decay_is_live`
  (`metal.rs:4564`) is the pattern to copy, including the perturbation.
* Dual reference, per rule 13: one reference verifies the port, two verify the
  comprehension. P3 would need `deltanet_ref` *and* a chunked CPU form, because
  the ten lowbit negative controls were all caught by shim-vs-seam disagreement
  and none by GPU-vs-seam.

## 7. The implementation lanes, ready to lift

Ruled by Detoro 2026-09-01: **the transpose lands strictly after
`quant-matvec-rework` merges, in its own small lane, never in parallel and never
inside Mellow's lane** — his structural before/after gate only proves anything
while his lane stays pure order-preserving load changes. Ranking endorsed:
transpose → B1 → B2-as-measured-dead.

### Lane 1 — `deltanet-state-transpose` (small)
*Boundary*: `src/gpu/kernels.metal`, `src/gpu/metal.rs`. **`src/lowmem/` is NOT
in it and needs no edit** — verified at 398feb6: `lowmem/forward.rs:579-580`
binds `st.delta` at offset 0 and dispatches the shared
`e.pipes.delta_decode_step` without ever touching the buffer's interior;
`forward.rs:225` builds states through the one shared
`gpu::metal::DeltaNetStates::new`, which zero-initialises (`metal.rs:616-628`),
and `reset()` zeroes too (`metal.rs:631-639`) — zero is layout-invariant;
`lowmem/mod.rs:121-124` sizes from `conv_state_elems + delta_state_elems`, which
a transpose does not move. lowmem therefore executes the transposed kernel over
transposed buffers and is correct unchanged.
*Changes*: two address expressions in `delta_decode_step` and its layout comment
(`kernels.metal:3598-3600`); the `DeltaNetStates` layout comment
(`metal.rs:488`); and `deltanet_kernel_oracle`'s `gpu_delta`/`delta_inputs`
(`metal.rs:4408-4530`), which seed state from a reference-layout vector and
compare the state back, so both sides transpose and the transposed negative
control is re-pointed.
*Gates*: the T1 pair above; the existing deltanet oracle green; 16/16 identity
cells byte-identical (the change is bit-exact, so any moved cell is a defect);
`benchmarks/decode-speed.sh` before/after on the 2B as the throughput evidence.
*Expected*: 2.10× (dev) / 3.65× (27B dims) on `delta_decode_step`; ≈ +10% on 2B
decode tok/s; larger on the 27B after `quant-matvec-rework`.

### Lane 2 — `deltanet-prefill-batching` (B1, the big one)
*Boundary*: `src/gpu/kernels.metal`, `src/gpu/metal.rs`, `src/lowmem/forward.rs`,
`src/lowmem/mod.rs` (two lines per new kernel: the Pipes field and its pipe()
call — the struct lives in mod.rs, not forward.rs; omission found at claim,
challenge befab0c5)
(lowmem's own `for t in 0..n` loop is the twin and must move with it, unlike
Lane 1).
*Shape*: P2 — batch conv/gates/l2norm/output-norm over the chunk (§4), and one
persistent-state delta kernel per layer per chunk with a 32-column tile in
threadgroup memory (§3).
*Gates*: the T2 set, including the n>512 prompt.
*Expected*: 633,024 → 1,680 dispatches on the 27B prompt; delta's 30% of 2B
prefill and 13.8% of 27B prefill largely recovered.

### Lane 3 — none. B2 is measured dead.
Fusing the six kernels into one is worth ≤4.5% of the chain today (§1b) and ~9%
of the smaller chain after Lane 1, against a race hazard in the shared conv
state (§4). **Do not cut it.** This paragraph exists so the idea is not
re-proposed from the dispatch count alone — 288 dispatches per token is a real
number and a misleading one.

### Risk ledger
1. **The half-done transpose.** Changing the kernel without the oracle, or the
   oracle without the kernel, produces plausible finite wrong numbers. The
   re-pointed negative control is the guard; a lane that touches one side and
   not the other must fail it.
2. **The C/D seam.** `DeltaNetLayout`/`DeltaNetStates` carry "changes go through
   the lead, never pairwise" in their own comment. Lane 1 changes that layout's
   documented meaning even though it changes no field, so it is the lead's to
   sequence — which is why the ordering above is a ruling and not a preference.
3. **Textual collision.** Lanes 1 and 2 both live in files that
   `quant-matvec-rework` claims. The two changes do not overlap semantically at
   all — `lm_dot_run_*` versus `delta_decode_step` — but a blind interleave in
   one file is how a peer's work gets swept (three stage-filter incidents on this
   ledger already).
4. **The cross-box projection.** The 27B decode row in §5 multiplies a dev-box
   ratio by a Studio baseline. Confirm it with a Studio rerun of
   `benchmarks/metal-attribution.sh` after Lane 1; do not quote it as measured
   before then.
5. **Threadgroup-memory tiling (Lane 2).** 32 columns × 128 floats = 16 KB
   against a 32 KB limit, measured on the dev M1 Pro. A device reporting less, or
   an S larger than 128, breaks the tile arithmetic — read the limit at runtime
   and refuse rather than silently mis-size, the way the scratch audit does.
