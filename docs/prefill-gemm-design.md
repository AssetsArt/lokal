# Why the quant GEMM barely beats the matvec, and the fix — design memo

Lane `prefill-gemm-design`, design-only. Base `main @73410a1`. No engine code is
changed by this document; §7 is written to be lifted into an implementation plan.

**The one-line result.** `matmul_pg` and llama.cpp's `mul_mm` have the *same*
tile, the *same* thread-to-row map and the *same* simdgroup outer product — the
comment above `matmul` says as much, and re-reading both confirms it. They
differ in exactly one thing: our staging loop calls `lm_dequant` **once per
weight element**, sixteen times per thread per K-slice, each call re-deriving
the superblock base, `d`, `dmin` and the 6-bit scale/min pair; upstream calls
its `dequantize_q4_K` **once per thread per K-slice** and gets sixteen values
out of one scale decode. A thread's sixteen elements are provably one
superblock, one scale group, one nibble half and sixteen *consecutive* bytes, so
that decode is loop-invariant — hoisting it is a pure code motion that leaves
the per-element expression, and therefore the bits, untouched.

## 0. Provenance — what was read and what was measured

Repo files are pinned at the lane base, `main @73410a1`:

| file | sha256 |
|---|---|
| `src/gpu/kernels.metal` | `199669f3f551579aa2623b38e3e35c92ca8fba46e7a0700b9e436782318680d8` |
| `src/gpu/metal.rs` | `d44cf2776f04a0d8a6b3ec7592b89d18c1f255c055bfe15e09fc3cec960bd402` |
| `src/lowmem/forward.rs` | `18aff5c68fb92e15431dadb0267f8d2ee1653984b1b39d6f73c7212b7523a35c` |
| `src/lowmem/mod.rs` | `dbea89d424ca26b80965d9c73e3df0020233857bb7893b39844eaf6e90677ddd` |

Everything cited from llama.cpp is from the read-only checkout at
`~/.unsloth/llama.cpp`, which carries **no git metadata**; it is pinned by
content hash, exactly as `docs/deltanet-chain-design.md` §0 does. The file is
dated 2026-08-09 and the repo's own IQ tables were vendored from this tree at
"llama.cpp b9960" (`src/gpu/kernels.metal:113`).

| file | sha256 |
|---|---|
| `ggml/src/ggml-metal/ggml-metal.metal` | `5d577d20a699016d108b83d517fe49c716c635e4dee46a0538157542a8789130` |

That hash is byte-identical to the one `docs/deltanet-chain-design.md` §0
records, so the two memos cite the same bytes and their line numbers are
directly comparable.

Model dimensions are read out of the GGUF header, not assumed
(`Qwen3.5-2B-Q4_K_M.gguf`, unsloth snapshot `f6d5376b`):

| key | value |
|---|---|
| `qwen35.block_count` | 24 |
| `qwen35.embedding_length` | 2048 |
| `qwen35.feed_forward_length` | 6144 |
| `qwen35.attention.key_length` / `value_length` | 256 |
| vocab | 248,320 |
| `ffn_gate.weight`, `ffn_up.weight` | `[2048, 6144]`, Q4_K |
| `ffn_down.weight` | `[6144, 2048]`, Q6_K on some layers, Q4_K on others |

The dispatch mix is the engine's own, from the attribution ledger: per
512-token prefill chunk the MLP issues 72 `matmul_pg` dispatches, 60 of them
Q4_K and 12 Q6_K — 24 layers x 3 projections, exactly.

### Reproducing this memo

The harness and the raw logs are in the lane scratchpad, `mmbench/`
(`main.rs` + `variants.metal`, `mmbench-run1.log`, `mmbench-run2.log`) and
`attr-now.err` for the in-engine pass. **They are deliberately not in the repo:
this lane's boundary is `docs/prefill-gemm-design.md` alone.** Promoting the
harness into `benchmarks/` is Lane 1's job and is written into its boundary in
§7, because `protocol:gate-scripts` rule 16 is explicit that an instrument which
decides a merge belongs in the repo beside its baselines, and a scratchpad-only
harness is one session teardown from unrunnable.

Every citation in this memo is machine-checked by `check-citations.py` in the
same scratchpad: it re-hashes the §0 files, asserts every `path:line` reference
is inside its file, and asserts that 44 load-bearing lines still *contain* what
the memo says they contain — so a reader who finds a line number shifted is
looking at a moved line, not a wrong claim. It passes at `73410a1`.

## 1. What `matmul_pg` costs

### 1.1 The machine state every row below carries

One quiet window, 2026-09-01 14:12-14:26Z, dev box M1 Pro, `main @73410a1`.
Entry: three consecutive `benchmarks/quiet.py` samples at 137.7 / 118.4 / 120.8%
foreign CPU against the 150% gate; exit 127.7%. Memory-quiet across the whole
window: **+77,914 decompressions and +2,895 swapins**, against
`protocol:gpu-bench`'s void thresholds of 200,000 and 20,000. The window was
taken only after a `duetexpertd` daemon that had been pinning ~99% for ten
minutes finished — two verdicts seconds apart disagreed while it was winding
down, so "QUIET once" was not accepted as quiet.

The `lokal` binary was verified current rather than assumed: `git status` clean
and `cargo build --release` a 0.58 s no-op, which is `protocol:gate-scripts`
rule 12 (a stale binary "differs from the reference" exactly as convincingly as
a wrong one).

### 1.2 In-engine, and the harness that reproduces it

Measurements come from `mmbench`, a standalone harness that compiles **this
repo's own** `src/gpu/kernels.metal` through the same precise, `fast_math`-off
options the engine builds its quant pipelines with (`src/gpu/metal.rs:1703-1707`),
specialized with function constant 25 = 4 (Q4_K), and dispatches `matmul_pg`
with exactly the grid `enc_qmm` uses (`src/gpu/metal.rs:2644-2647`). It times
the shipped kernel, not a copy.

In-engine, one 512-token prefill chunk of the 2B Q4_K_M in this window:

| phase | ms | dispatches |
|---|---|---|
| `mm:mlp:q4_k` | 2609.3 | 60 |
| `mm:dn:q5_k` | 1245.6 | 36 |
| `mm:mlp:q6_k` | 476.7 | 12 |
| `delta` | 374.6 | 55,296 |
| `mm:dn:q4_k` | 296.8 | 18 |
| `mm:attn:q4_k` | 280.8 | 20 |
| everything else | 87.7 | 214 |
| **GPU total** | **5371.5** | |

**Weight multiplies are 91.8% of prefill GPU time** (4930.9 of 5371.5 ms) now
that the deltanet chain batches; `mm:mlp:q4_k` alone is 48.6%. This is the arc's
biggest remaining gap and it is one kernel.

The harness predicts that row at 2729.6 ms (48 x 46.20 + 12 x 42.67) against
2609.3 ms measured in-engine — **agreement to 4.6%**. That is looser than the
1.5% the deltanet microbench reached, and the difference is the window itself:
this window's prefill ran at 89.0 tok/s against the 94.1 recorded in a quieter
one, so engine and harness are both reading a few percent slow together. The
harness is validated to 4.6%, and no claim below rests on a tighter figure.

### 1.3 Where the time goes

Two independent runs, five reps each, median; `x` and `y` as the engine binds
them; twelve distinct 7.1 MB weight buffers cycled over the dispatch count so no
tensor sits in the 24 MB system cache the way a single reused buffer would.

| variant | ms/dispatch (gate/up) | ms/dispatch (down) | vs V0 | bits |
|---|---|---|---|---|
| **V0** `matmul_pg` as shipped | 46.20 / 49.53 | 42.67 / 43.78 | 1.00x | — |
| **V1** no weight read at all | 5.30 / 5.36 | 5.67 / 5.55 | 7.5-9.3x | differs (probe) |
| **V2** raw nibble, no scale decode | 18.98 / 19.06 | 19.22 / 19.27 | 2.2-2.6x | differs (probe) |
| **V3** hoisted scale decode | 9.16 / 9.02 | 9.47 / 8.23 | **4.2-5.5x** | **bit-identical** |
| **V4** V3 + `uchar4` nibble loads | 9.21 / 9.06 | 9.40 / 9.36 | 4.7-5.5x | **bit-identical** |

Three things fall out, and the third was a surprise:

1. **The weight path is ~89% of the kernel.** V1 removes only the weight read
   and keeps every barrier, the X staging, the outer product and the store; it
   runs 8x faster. Whatever else `matmul_pg` does is 11% of it.
2. **The cost is the per-element *call*, not the arithmetic inside it.** V2
   deletes the entire scale decode — no `d`, no `dmin`, no `lm_scale_min_k4` —
   and still lands at 19 ms, four times slower than V3, which does *all* of that
   arithmetic but once per sixteen elements instead of sixteen times. What V2
   keeps and V3 removes is the per-element re-derivation of the block base and
   the address. That is the cost.
3. **Widening the loads buys nothing.** V4 is V3 with the sixteen scalar nibble
   reads replaced by four `uchar4` loads, and it is within noise of V3 on both
   shapes. Once the call count drops, the byte loads were never the constraint —
   so **C2 is dropped from the recommendation** (§7), taking its buffer-alignment
   precondition with it. A whole class of risk disappears because the measurement
   was taken before the design was written.

V0's own numbers move 7% between runs while V3's barely move, which is itself
consistent: V0 issues sixteen times the memory operations and is correspondingly
more sensitive to a contended box. Where the two runs disagree, §7 plans against
the **conservative** figure.

An A/A control rules out the obvious way a table like this goes wrong. Mellow
lost most of an afternoon to an A/B probe that reported 1.18x and 1.63x between
two pipelines built from *identical* code, because whichever arm ran first ate
the warm-up, and raised it here before this number could be cited. The harness
now warms every pipeline before timing any, and runs the shipped kernel **again
as the last arm**. First against last: **0.97x** on `ffn_gate/up` (41.88 vs
43.07 ms/dispatch) and **1.02x** on `ffn_down` (40.57 vs 39.79) — within 3%, in
opposite directions, so noise rather than arm order, and roughly 150x smaller
than the effect being claimed. (That control run was taken on a *non-quiet* box,
so only its ratio is used; its absolute numbers appear nowhere in this memo.)
Across all three runs V3 lands at 5.04 / 5.49 / 5.08 on `ffn_gate/up` and
4.51 / 5.32 / 4.19 on `ffn_down`, which is where the **4.2-5.5x** range comes
from — and §7 plans against the 4.2.

### 1.4 The per-element decode rate is flat across quant formats

Dividing each in-engine phase by the number of dequant calls it makes
(`out_dim x in_dim x ceil(n_rows/32)`, summed over that phase's tensors):

| phase | dequant calls | ns per call |
|---|---|---|
| `mm:mlp:q6_k` | 2.416 G | 0.197 |
| `mm:mlp:q4_k` | 12.080 G | 0.216 |
| `mm:dn:q4_k` | 1.208 G | 0.246 |
| `mm:dn:q5_k` | 4.831 G | 0.258 |

Flat within ±14% across three formats whose unpack cost differs a great deal —
Q6_K reads two packed arrays plus a signed scale, Q5_K adds a whole high-bit
array on top of Q4_K's nibble. The harness lands independently at 0.212-0.246
ns/call on the same kernel. **The GEMM's cost tracks the number of decode calls,
not the work inside a decode**, which is why removing fifteen of every sixteen
calls buys 5x while changing no arithmetic.

This is the same wall the matvec lane hit from the other side — that lane found
its ceiling was the loads a lane can keep in flight rather than the bytes moved.
Credit to Mellow for asking the question in this form; it turns two lanes'
separate findings into one statement about `lm_dequant`'s per-element interface.
(`mm:dn:q8_0` is excluded: its tensors are `[2048, 16]`, so at 0.507 ns/call it
is measuring launch overhead, not decode.)

## 2. The defect is the dequant granularity, not the tiling

### 2.1 The two kernels are the same kernel

`matmul_pg` (`src/gpu/kernels.metal:1832`) is `matmul` (`:1706`) plus a
`y_stride`, and the comment above `matmul` (`:1687-1693`) already says whose
layout it is: *"the layout llama.cpp's mul_mm kernel proved out on this
hardware."* Re-reading upstream confirms the claim is exact, not aspirational.
Side by side, `kernels.metal` against `ggml-metal.metal:10173`:

| | `matmul_pg` | `kernel_mul_mm` |
|---|---|---|
| outputs per tile | `MM_TN` 64 (`:1683`) | `NR0` 64 (`:10186`) |
| tokens per tile | `MM_TM` 32 (`:1682`) | `NR1` 32 (`:10187`) |
| K slice staged | `MM_TK` 32 (`:1684`) | `NK` 32 (`:10189`) |
| threads | 128 = 4 simdgroups (`:1685`) | 128 = 4 simdgroups |
| W row per thread | `w_row = tid / 2` (`:1865`) | `lr0 = tiitg / NL0`, `NL0 = 2` (`:10190`, `:10202`) |
| K strip per thread | `w_strip = tid % 2` (`:1866`) | `il0 = tiitg % NL0` (`:10205`) |
| accumulators | `mc[8]`, `ma[4]`, `mb[2]` (`:1857-1859`) | `mc[8]`, `ma[4]`, `mb[2]` (`:10225-10228`) |
| inner product | `mb[i/4] x ma[i%4]`, 4 K-blocks (`:1909-1911`) | identical (`:10331-10333`) |
| staging index | `sa[64*ib + 8*(i%8) + w_row%8]` (`:1886`) | `sa + 64*ib + 8*ly + lx` (`:10273`) |

Same tile, same thread-to-row map, same barrier count, same outer product, same
threadgroup layout. Whatever is costing us, it is **not** the tiling, and a
redesign that reshapes the tile is redesigning the part that already matches a
reference implementation.

### 2.2 The one material difference

Upstream stages a K-slice like this (`ggml-metal.metal:10254-10274`):

```
S0_4x4 temp_a;
dequantize_func(x, il, temp_a);      // ONE call -> 16 values
...
FOR_UNROLL (short i = 0; i < 16; i++) { *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4]; }
```

and `dequantize_q4_K` (`:736-753`) decodes `d`, `dmin` and the scale/min pair
**once**, then fills sixteen registers from sixteen consecutive `qs` bytes.

We stage the same slice like this (`src/gpu/kernels.metal:1872-1887`):

```
for (uint i = 0; i < 16; i++) {
    uint gk = k0 + w_strip * 16 + i;
    ...
    v = (LM_W_QTYPE >= 2)
        ? (half)lm_dequant((device const uchar *)w + (ulong)go * lm_row_bytes(p.in_dim), gk)
        : w[...];
    sa[64 * ib + 8 * (i % 8) + w_row % 8] = v;
}
```

`lm_dequant` takes `(row_pointer, column)` and has no memory between calls, so
`lm_dequant_q4_K` (`:882-897`) re-does, on **every one of the sixteen
iterations**: the superblock offset `(col >> 8) * 144`, two `lm_f16_at` loads
for `d` and `dmin`, and `lm_scale_min_k4` (`:101-109`), which is two or three
more byte loads plus the 6-bit unpack. Sixteen elements therefore cost sixteen
scale decodes where upstream pays one.

That is a per-element interface problem, not a tuning problem: the signature
`lm_dequant(row, col)` **cannot** express "give me the next sixteen", so no
amount of tile tuning removes the redundancy. It is the same shape that makes
the decode matvec slow — the matvec lane found its ceiling was the loads a lane
can keep in flight rather than the bytes moved — seen from the GEMM side, and
§1 tests it directly by measuring the per-element decode rate.

### 2.3 Why the hoist is provably legal (and therefore bit-exact)

A thread's strip is `gk = k0 + 16*w_strip + i` for `i` in `[0,16)`. `k0` steps
by `MM_TK` = 32 and `w_strip` is 0 or 1, so `gk0 = k0 + 16*w_strip` is **a
multiple of 16**. Sixteen consecutive indices starting at a multiple of 16
cannot cross a 32, 64 or 256 boundary. Therefore, over one strip:

* `gk >> 8` — the Q4_K superblock — is **constant** (144-byte block, `:883`);
* `(gk & 255) >> 5` — the argument `lm_scale_min_k4` takes — is **constant**
  (`:889`), so `sc` and `mn` are constant, and so are `d`, `dmin`;
* `(gk & 255) & 32` — the nibble half — is **constant** (`:895`);
* `(gk & 255) & 31` runs 0..15 or 16..31, so the sixteen `qs` bytes are
  **sixteen consecutive bytes**, 16-byte aligned (`:894`: the offsets 144, 32
  and `(ib & 31)` in `{0,16}` are all multiples of 16).

So `d1 = d * sc` and `m1 = dmin * mn` are loop-invariant and the sixteen nibbles
are one contiguous 16-byte run. Hoisting the first out of the loop and widening
the second to vector loads changes **no arithmetic**: the per-element expression
stays `d1 * (float)q - m1`, evaluated in the same order, in a library compiled
with `fast_math` disabled (`src/gpu/metal.rs:1703-1704`,
`src/lowmem/mod.rs:858-862`), so there is no contraction to differ over. The
result is bit-identical by construction, which is what makes §6's gate a
**bit-exact** gate rather than a tolerance.

The 16-alignment holds only while `in_dim % 32 == 0`. Every tensor in play
satisfies it (`in_dim` is 2048 or 6144, both multiples of 256), but the kernel
is generic, so the fast path must be guarded and a scalar tail kept — §7 makes
that a named gate, not an assumption.

## 3. Candidate forms

All four keep the tile, the thread map and the outer product exactly as §2.1
found them. They differ only in how a thread gets its sixteen weights.

### C1 — hoist the scale decode (recommended for v1)

One strip, one superblock lookup, one `d`/`dmin` pair, one `lm_scale_min_k4`,
then sixteen nibble reads:

```
uint gk0 = k0 + w_strip * 16;                        // multiple of 16, §2.3
device const uchar *b = wrow + (gk0 >> 8) * 144;
float d = lm_f16_at(b), dmin = lm_f16_at(b + 2);
uint ib0 = gk0 & 255, sc, mn;
lm_scale_min_k4(ib0 >> 5, b + 4, sc, mn);
float d1 = d * (float)sc, m1 = dmin * (float)mn;
device const uchar *qs = b + 16 + (ib0 >> 6) * 32 + (ib0 & 31);
bool hi = (ib0 & 32) != 0;
for (uint i = 0; i < 16; i++) {
    uint q = hi ? (qs[i] >> 4) : (qs[i] & 0xF);
    vals[i] = (half)(d1 * (float)q - m1);            // UNCHANGED expression
}
```

Scale decodes per strip: 16 -> 1. Device loads for metadata per strip: ~64 -> 4.
**Bit-exact by construction** (§2.3). No new kernel, no new pipeline object, no
host-side change at all — the entire diff is inside `matmul_pg`'s staging loop.

### C2 — C1 plus vector nibble loads

The strip's sixteen `qs` bytes are contiguous and 16-byte aligned (§2.3), so
they can arrive as four `uchar4` loads instead of sixteen scalar ones. Also
bit-exact; adds an alignment precondition on the bound buffer offset, which is
a real constraint because `enc_qmm` binds `w` at `l.w_off`
(`src/gpu/metal.rs:2639`) and lowmem binds a pool page at offset 0
(`src/lowmem/forward.rs:394`). §7 makes the alignment a gate, not a hope.

### C3 — raise `MM_TM` from 32 to 64

Dequant work per token halves, because each staged weight tile serves 64 tokens
instead of 32. Also bit-exact: widening the token tile changes *which tokens
share a staged weight*, never the order of the K accumulation for any single
output. The costs are real and must be measured, not assumed:

* threadgroup memory goes from 8 KB (`sa` 4 KB + `sb` 2 KB + bias 2 KB,
  `src/gpu/kernels.metal:1846-1850`) to ~10 KB, which drops resident
  threadgroups per core from 4 to 3 — the comment at `:1720-1721` (on `matmul`, the twin `matmul_pg` is kept in
  sync with) says the 8 KB figure is deliberate;
* accumulators go from `mc[8]` to `mc[16]`, doubling that register class.

So C3 trades occupancy for dequant work and is only worth it if §1 shows
dequant dominating. It is a **prediction**, not a measurement, until a lane
builds it.

### C4 — a block-shaped dequant helper family

The general form of C1: add `lm_dequant16_*` beside each `lm_dequant_*`, taking
a row pointer and an aligned starting column and filling sixteen values. This is
structurally what upstream's `dequantize_q4_K` is, and it is the form that
extends past Q4_K to Q5_K, Q6_K and Q8_0 without writing the hoist out four
times.

C4 carries a trap that C1 does not, and §6 makes it a blocking gate: the oracle
kernel `lm_dequant_oracle` (`src/gpu/kernels.metal:1374-1386`) is what proves
our dequant math matches `dequant_row_ref` bit-for-bit, and it calls
`lm_dequant`. If the GEMM switches to `lm_dequant16_*` while the oracle keeps
calling `lm_dequant`, **the oracle stops covering the code prefill actually
runs**, silently. Either the block helper is defined in terms of the scalar one,
or the oracle must be extended to drive both. C1 has no such problem, which is
one more reason it is the v1 recommendation.

## 4. What upstream does, and the half we must not copy

Studied, cited, not copied. `dequantize_q4_K`
(`ggml-metal.metal:736-753`) is the right *shape* — one scale decode, sixteen
values, into a thread register tile — and C1/C4 adopt that shape.

What we must **not** adopt is its arithmetic. Upstream factors the scale as
`d = il < 2 ? xb->d : xb->d / 16.h` with a nibble mask
(`:743`, `:748`), where ours multiplies `d * sc` and subtracts `dmin * mn`
directly (`src/gpu/kernels.metal:891-896`). The two agree mathematically and
will disagree in the last ulp on exactly the values a quantizer produces. Our
`lm_dequant_*` are bound bit-for-bit to `dequant_row_ref` through the oracle
gate, and `protocol:gate-scripts` rule 13 requires two independent references
for any dequant-math lane. Importing upstream's factoring would break that
binding for a speedup that C1 already gets without touching a single arithmetic
operation.

The distinction matters because it is the difference between a code-motion lane
(bit-exact, gate is byte equality) and a numerics lane (needs the full
dual-reference treatment). §7 keeps the recommended lane firmly in the first
category.

## 5. Dispatch and traffic arithmetic

Per `ffn_gate` dispatch (in_dim 2048, out_dim 6144, n_rows 512), all figures
derived from the kernel's own constants, not measured:

| quantity | value | where it comes from |
|---|---|---|
| threadgroups | 1536 | `ceil(6144/64) x ceil(512/32)` (`src/gpu/metal.rs:2645`) |
| K slices per threadgroup | 64 | `in_dim / MM_TK` = 2048/32 |
| W dequants per threadgroup per slice | 2048 | 128 threads x 16 (`src/gpu/kernels.metal:1872`) |
| **dequant calls per dispatch** | **201,326,592** | `out_dim x in_dim x ceil(n_rows/32)` |
| useful MACs per dispatch | 6,442,450,944 | `in_dim x out_dim x n_rows` |
| MACs per dequant call | **32** | `= MM_TM`, by construction |
| weight bytes | 7.08 MB | 6144 rows x 8 superblocks x 144 B |
| weight re-reads per dispatch | 16 | `n_rows / MM_TM` |

`ffn_down` (6144 -> 2048) has the *same* 201,326,592 dequant calls — the product
`out_dim x in_dim` is symmetric — which is why §1.3's two shapes agree so
closely, and why the fix's leverage does not depend on which projection it is.

**The bandwidth story is dead, and the earlier arithmetic for killing it was
itself wrong.** A prior note computed the MLP Q4_K weights as ~425 MB read once
per chunk, ~2.1 GB over a 2198-token prompt, and concluded 0.2 GB/s. The "read
once" is not right: the kernel re-reads each weight tile once per 32-token tile,
so per chunk it demands 425 MB x 16 = **6.8 GB**, and ~34 GB over the whole
prompt. At the measured 2609 ms per chunk that is 2.6 GB/s, not 0.2. The
conclusion is unchanged and in fact firmer — 2.6 GB/s on a 200 GB/s part is
nowhere near a bandwidth wall, and §1.3's V1 (which does the identical MMA work
with no weight traffic at all) settles it directly. But the number itself is
corrected here rather than repeated.

### The 27B projection — CROSS-BOX, not measured

The 27B prefill figure of ~18 tok/s against llama.cpp's 137-140 pp2048 comes
from the Studio table (`ref:llamacpp-27b-studio`), a **different machine** from
the M1 Pro every measurement above was taken on. Nothing in this memo licenses a
27B speedup claim. What the arithmetic does support, and only as a shape:

* 27B is 48 layers against 24, `d_inner` 6144 against 2048, and its MLP tensors
  are correspondingly larger, so the dequant-call count per chunk grows with the
  parameter count while the MACs-per-call ratio stays pinned at `MM_TM` = 32;
* the defect is therefore *scale-invariant* — it is a property of the staging
  loop, not of a dimension — so there is no size at which it stops applying.

That is an argument about the mechanism, not a predicted number. It is also
exactly the shape of reasoning this arc has already had to retract once: an
invariance is only evidence if the thing held constant actually varied. Here the
claim is deliberately weaker — that a per-element loop cost cannot be outgrown —
and the 27B number stays unmeasured until someone runs it on the Studio.

## 6. Numerics gates, decided here — `protocol:gate-scripts` rule 14 tiers

Rule 14 is three-tier, and this lane is unusual in landing squarely in tier 1.
The GEMM is pure multiply-add; the quant pipelines compile with `fast_math`
disabled in **both** engines (`src/gpu/metal.rs:1703-1707`,
`src/lowmem/mod.rs:858-862`), so there is no contraction to differ over; and
every recommended candidate preserves the per-element expression and the K
accumulation order exactly (§2.3). There is therefore **no tolerance anywhere in
this lane**. A candidate either produces the same bytes or it is wrong.

| # | gate | tier | what it pins |
|---|---|---|---|
| G1 | whole-buffer byte equality, new kernel vs shipped, at the real MLP shapes | **bit-exact** | that the code motion moved no arithmetic |
| G2 | one-hot column sweep (below) | **bit-exact** | index mapping, independently of accumulation |
| G3 | `lm_dequant_oracle` vs `dequant_row_ref` | **bit-exact** | that the *production* dequant path is still the verified one |
| G4 | edge tiles: `out_dim % 64`, `n_rows % 32`, `in_dim % 32` all non-zero | **bit-exact** | the guarded fast path and its scalar tail |
| G5 | 2198-token prompt, metal == lowmem, 18 windows | **bit-exact** | both engines, past one prefill chunk |

### G2 — the one-hot gate, per the Annaka rule

The Annaka rule's constructive half: an index-remapping change earns a one-hot
bit-exact gate. Drive the **real** kernel with an `x` that is `1.0` at exactly
one column `c` and `0` elsewhere. Then every accumulator carries exactly one
non-zero term and `y[t, o]` must equal `dequant(W[o, c]) + bias[o]` exactly, so
the index mapping is pinned independently of the accumulation that would
otherwise hide a swapped pair.

`c` must visit every structural class the Q4_K addressing distinguishes, because
a hoist that is right for one class and wrong for another is exactly the bug
this gate exists to catch:

`c` in `{0, 15, 16, 31, 32, 47, 48, 63, 255, 256}` — both nibble halves
(`ib & 32`), both 32-element scale groups within a 64-element group
(`ib >> 5`), both 64-element groups within a superblock (`ib >> 6`), the last
column of a superblock and the first of the next (`col >> 8` stepping).

This is not optional politeness. A one-hot gate is what would have caught the
quarter-index mapping error that once produced fluent garbage and had a speed
number published off it.

### G3 — the trap C4 sets, stated as a blocking gate

`lm_dequant_oracle` (`src/gpu/kernels.metal:1374-1386`) is what makes our dequant
math verifiable: it dequantizes whole rows **through the same inline functions
the matvec and matmul paths use**, so the byte comparison against
`dequant_row_ref` covers production math. Its comment says exactly that.

If an implementation adds `lm_dequant16_*` helpers and points `matmul_pg` at
them while the oracle keeps calling `lm_dequant`, **the oracle silently stops
covering what prefill runs**. The gate would stay green while the code it
certifies is no longer the code that executes. So:

> Any lane that introduces a block-shaped dequant helper MUST either define it
> in terms of the scalar `lm_dequant_*` (so the oracle still covers it
> transitively), or extend `lm_dequant_oracle` to drive both and gate both.
> This is a merge blocker, not a follow-up.

C1 does not trip this, because it hoists inline inside `matmul_pg` without
introducing a second dequant entry point — one more reason it is the v1
recommendation.

### Two references, not one

`protocol:gate-scripts` rule 13 requires two independent references for any
dequant or kernel-math lane, because ten lowbit negative controls were caught by
shim-vs-seam disagreement and none by GPU-vs-seam. Here the two are: G1/G2
(GPU new vs GPU shipped, byte equality) and G3 (GPU vs the CPU
`dequant_row_ref`). G1 alone would pass a change that is self-consistently wrong
in both kernels; G3 alone would pass a change that decodes correctly but stages
into the wrong `sa` slot. Both are required.

### What the gates do *not* cover

Rule 8's lesson applies: none of these gates asserts that output text changes,
because none of these candidates may change it. The correct assertion is byte
equality of logits and of `y`, and G5 carries the same-position control.

## 7. The implementation lanes, ready to lift

### Lane 1 — `prefill-gemm-hoist` (C1; the whole prize, and small)

Hoist the per-strip scale decode in `matmul_pg`'s staging loop for the K-quant
arms, starting with Q4_K and extending to Q5_K and Q6_K by the same argument
(§2.3 holds for all three: their scale groups are also 32 elements wide).
**C2 is explicitly out of scope** — §1.3 measured it at zero gain, and dropping
it removes the buffer-alignment precondition entirely.

Measured leverage, conservative end of the three runs: 4.2x on `matmul_pg`'s Q4_K
arm. Against this window's in-engine rows, fixing the Q4_K arms alone addresses
3186.9 ms of a 5371.5 ms prefill chunk (59.3%); extending to Q5_K and Q6_K
reaches 4921 ms (91.6%).

**Boundary.** `src/gpu/kernels.metal` only, plus the in-repo instrument
(rule 16/17, below). This was verified against the code rather than assumed,
which is the `befab0c5` lesson: `matmul_pg` is looked up **by name** at four
construction sites — `src/gpu/metal.rs:1736` and `:1752`,
`src/lowmem/mod.rs:878` and `:934` — and selected at
`src/lowmem/mod.rs:1189-1194` and `src/gpu/metal.rs:3177-3300`. Editing the
kernel's *body* changes nothing at any of them, so **no new pipeline object is
created and no host file needs to be in the boundary.** That is a property of
C1 specifically; see Lane 2 for what changes when it is not true.

Per rule 16 and 17, the measuring instrument must land in the repo with the lane,
not stay in a scratchpad: promote `mmbench` to `benchmarks/mm-variants/` with its
baseline capture, and name that path in the lane's `--boundary` **at task
create** — a boundary that omits the instrument gets it silently dropped by
`conclave stage commit` (rule 11), and the READY gates then certify a commit that
cannot reproduce its own numbers.

**Gates.** G1-G5 from §6, every one bit-exact. Plus a before/after
`benchmarks/metal-attribution.sh` prefill pass in one quiet window under
`protocol:gpu-bench`, with the identity gates green **before** any throughput
number is quoted — the Annaka rule is first-order here.

### Lane 2 — `prefill-gemm-tile64` (C3; only if Lane 1's residual justifies it)

After Lane 1, V3 sits at 9.0-9.5 ms/dispatch against V1's 5.3-5.7 ms MMA floor,
so roughly 40% of the remaining time is still weight staging. C3 halves that by
serving 64 tokens per staged tile instead of 32. Bit-exact for the reason in §3.

Do not start this before Lane 1 lands and is re-measured: C3 trades occupancy
(threadgroup memory 8 KB -> ~10 KB, 4 resident threadgroups -> 3) for dequant
work, and that trade is only worth making against a *measured* residual, not
against today's V0.

**Boundary if C3 needs a new kernel name** (e.g. a `MM_TM`-64 variant selected
by `n_rows`), it is materially larger than Lane 1's and must list every site:

* `src/gpu/kernels.metal` — the kernel;
* `src/gpu/metal.rs` — `QuantPipes` field (`:409`), both `qpipe` construction
  sites (`:1736`, `:1752`), and `enc_qmm` (`:2621`);
* `src/lowmem/mod.rs` — **two** struct declarations (`:205` for the direct set,
  `:256` for `QuantPipes`), **two** construction sites (`:878`, `:934`), and the
  selector `matmul_pipe` (`:1189-1194`);
* `src/lowmem/forward.rs` — `enc_matmul_paged` (`:374`).

Six declaration/construction sites across two engines. Missing any one of them
is the `befab0c5` failure exactly: the lane cannot dispatch its own kernel
without an out-of-boundary edit, discovered after the claim.

### Risk ledger

| # | risk | why it is real | mitigation |
|---|---|---|---|
| R1 | The hoist is wrong for a `k0` alignment the tests never visit | The invariance in §2.3 needs `in_dim % 32 == 0`; every shipping tensor satisfies it, so a bug here is invisible to every current model | Guard the fast path and keep the scalar tail; G4 exercises a non-multiple `in_dim` deliberately |
| R2 | A block helper orphans the oracle | G3; the oracle would stay green while covering dead code | Merge blocker in §6; C1 avoids it structurally |
| R3 | Metal binds buffers by index at runtime | Changing a shared kernel's signature and missing a caller gives no compile error, no crash, no NaN — the kernel reads whatever was at that index. This arc hit it three times in one lane | C1 changes no signature at all. If a lane ever does, the tests are the only real guard — a grep sweep misses pipelines bound through a local |
| R4 | `matmul` and `matmul_pg` drift apart | `:1704-1705` says a fix in one belongs in both; they are the same algorithm and `matmul` serves the f16 path | Apply the hoist to the quant arms only — `matmul` has no dequant to hoist — and state that in the commit so the twin comment stays true |
| R5 | 27B numbers get quoted from this memo | §5's projection is CROSS-BOX and explicitly not a prediction | Marked at every occurrence; the Studio measurement is a separate lane |
| R6 | A throughput number escapes before identity gates | The arc's own history: fluent garbage with a speed number published off it | Annaka rule, first-order: gates green before any number leaves the lane |

