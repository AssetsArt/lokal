# One x copy per threadgroup — probe memo

Lane `matvec-x-staging-probe`, design/probe only. Base `main @be3a5ba`.
One question, thresholds fixed before the harness ran.

**The one-line result.** Dead, and not marginally: staging `x` into threadgroup
memory is **1.6-1.9x slower** than the shipped kernel, across three tile and
geometry variants and both shapes. All three staged arms are **bit-identical** to
production — a pre-registered pass condition — so this is a performance result,
not a broken probe. The control that explains it: doubling the consumers per
staged copy buys almost nothing, so the `x` device reads were never expensive.
What the bounding arm removes is not traffic but **load instructions**, which
staging replaces roughly one-for-one and then charges two barriers for.

## 0. Provenance

| file | sha256 |
|---|---|
| `src/gpu/kernels.metal` | `e4ab553379be13d06eb3f1e687f19a387f69c7045e0ffa98c58daf66a0996560` |
| `src/gpu/metal.rs` | `eaa04ba74bd0ebf179a27ebec26831c34cd7ead2529b19958047f07cfd6b47ae` |

Harness `xsbench` (lane scratchpad) compiles **this repo's own**
`src/gpu/kernels.metal` through the precise, `fast_math`-off options the engine
uses for quant pipelines, function constant 25 = 4 (Q4_K), 26 (`LM_MV_ACCUM`) = 1.
**B0 is the shipped `matvec`** (`:1587`), dispatched with
`dispatch_simdgroup_rows`' geometry (`src/gpu/metal.rs:1634-1638`).

**Machine state.** Clean window, 15:30-15:36Z: **+104 decompressions, +0
swapins**, A/A control **0.2%** and **1.5%**, exit 85.9% foreign CPU. An earlier
window was **void on both counts** — A/A 78.8% apart and +258,423 decompressions
against the 200,000 threshold — and was **re-run, not caveated**. Its direction
matched (staged arms 0.33-0.63x), which is recorded only because a void run that
disagreed would have mattered.

**Both pre-registered validation checks pass**: B0 at 77.8 / 68.3 GB/s against
the previous lane's clean-window 79.9 / 67.7, and B1 at 125.1 / 128.0 against
125.3 / 127.2 — inside the stated 10% bands, so the harness is measuring the same
kernels as `docs/matvec-call-overhead.md`.

## 1. The arms

| arm | threadgroup mem | gate/up | down | vs B0 | bits |
|---|---|---|---|---|---|
| **B0** `matvec`, shipped | 0 B | 4.37 ms / 77.8 GB/s | 1.24 ms / 68.3 | 1.00x | — |
| **B1** x deleted *(bounding)* | 0 B | 2.72 / 125.1 | 0.66 / 128.0 | 1.61x / 1.87x | probe |
| **B2** staged, TILE 1024, 4 rows | 4096 B | 7.66 / 44.4 | 2.07 / 41.1 | **0.57x / 0.60x** | **bit-identical** |
| **B3** staged, TILE 2048, 4 rows | 8192 B | 7.87 / 43.2 | 2.36 / 36.1 | **0.55x / 0.53x** | **bit-identical** |
| **B4** staged, TILE 1024, 8 rows | 4096 B | 7.28 / 46.6 | 1.97 / 43.1 | **0.60x / 0.63x** | **bit-identical** |
| **B0** again *(A/A)* | 0 B | 4.38 / 77.6 | 1.26 / 67.3 | 1.00x / 0.99x | — |

Best staged arm **0.60x / 0.63x** — the pre-registered **DEAD** band is <1.10x.
Headroom captured against B1: **-0.66 / -0.42**. Staging does not fall short of
the bound; it moves away from it.

## 2. Why, and the control that settles it

**B4 is the control that makes this more than a null.** It doubles the consumers
per staged copy — eight row-simdgroups share one `x` tile instead of four — and
buys 0.60x against B2's 0.57x. If the cost being attacked were the `x` *copy*,
halving the copies per row would have moved that number materially. It did not.

So the device-side `x` reads were never expensive, and B1's +61% / +87% does not
come from deleting traffic. It comes from deleting **32 load instructions per
run** from the inner loop. Staging replaces a device load with a threadgroup load
roughly one-for-one, leaves the instruction count where it was, and then adds:

* **two `threadgroup_barrier`s per tile** against production's zero — on a kernel
  whose four simdgroups otherwise run completely independently and hide each
  other's memory latency. A barrier makes the whole threadgroup wait for its
  slowest simdgroup, four times per row-group at in_dim 2048;
* **a staging write** that production never pays;
* **4-8 KB of threadgroup memory** on a kernel that used none, capping residency
  at 8 or 4 threadgroups per core on a 32 KB part.

The `x` reads it replaces were already L1 hits — `x` is 8.2 KB at in_dim 2048.
Trading a cache hit for a threadgroup read, plus barriers, is a bad trade, and
the measurement says so by a factor of 1.6-1.9.

This is the **third outcome named in the pre-registration**, in its own words:
"the traffic is already being served from cache and the cost is the load
instructions themselves, which staging does not remove." It is confirmed rather
than merely not-refuted.

## 3. Two constraints found by writing the probe, not by reasoning about it

Both are worth carrying forward, because either would have silently wrecked an
implementation:

1. **No early return is possible once a barrier is in the loop.** The shipped
   kernel opens `if (row >= p.out_dim) return;` (`:1599`). With a
   `threadgroup_barrier` in the K loop that deadlocks — returned threads never
   arrive at it. The guard has to become a predicate carried to the accumulate and
   the store, which is a structural change to a kernel every quant type shares.
2. **Tile size has a hard floor of 1024 elements**, derived: a tile holds
   `TILE/32` runs against 32 lanes, so anything smaller idles lanes every pass.
   The tile size is therefore not a free tuning knob, and the smallest useful tile
   already costs 4 KB of threadgroup memory.

## 4. What is now settled, and the next probe

**No implementation lane.** The candidate this lane existed to test is dead.

The decode matvec's dead list is now five mechanisms, measured, none to be
re-proposed:

| # | mechanism | result | where |
|---|---|---|---|
| 1 | traffic and occupancy shapes (five) | null | Mellow |
| 2 | weights staged in threadgroup memory | 3% slower | Mellow |
| 3 | inner loads widened to `int4`/`float4` | null, measured twice | Mellow; A5 |
| 4 | four independent accumulators | null | Mellow |
| 5 | per-call decode overhead removed | 7-11%, not the 2x | `matvec-call-overhead.md` |
| 6 | lanes cooperating on one run | 4.4x slower | `matvec-call-overhead.md` |
| 7 | **x staged in threadgroup memory** | **1.6-1.9x slower** | **this lane** |

And `probe_shape` is not a ceiling for this kernel (`matvec-call-overhead.md` §4),
so the "2x available" that framed all of this was never a headroom claim.

**The next probe follows from B4's failure rather than from a fresh idea.**
Every mechanism above changes how `x` is *fetched*. None changes how *often* it
is fetched per unit of arithmetic. The decode matvec reads the whole of `x` once
per output row, so at in 2048 / out 6144 it issues 6144 full passes over `x`; the
prefill GEMM avoids exactly this by giving each staged weight 32 token-consumers
(`MM_TM`, `docs/prefill-gemm-design.md` §5).

The decode analogue is to give each *loaded x value* several **output rows**: one
simdgroup accumulates R rows at once, loading `x` once and reusing it R times.
That divides x-side load instructions per row by R — attacking the quantity B1
says is worth +61-87% — and it does so with:

* **no threadgroup memory and no barriers**, so none of §2's costs apply;
* **no change to any row's accumulation order**, so it should be bit-exact, the
  same property that made the GEMM hoist a tier-1 gate;
* R extra accumulator registers per lane, which is the whole cost.

At R = 4 that is roughly a 37% cut in inner-loop load instructions, against a
bound of +61-87%. It is the one shape that reduces instruction count per unit of
work rather than relocating or widening the loads, and nothing on the dead list
touches it.

### Risk ledger

| # | risk | mitigation |
|---|---|---|
| R1 | "staging is slow" generalised to all threadgroup sharing | It is not: the GEMM shares a staged weight tile through threadgroup memory and wins, because 32 consumers amortise it. B4 shows 8 consumers is not enough here |
| R2 | The void window's numbers get quoted | Its A/A failed at 78.8% and it exceeded the decompression threshold; every number in §1 is from the clean run |
| R3 | The R-rows probe is assumed bit-exact | Stated as *should be* and owed a byte-equality check as a pass condition, exactly as this lane required of its staged arms |
| R4 | R-rows is read as obviously right because it is last | It is untested. Three of the seven dead mechanisms above were also obviously right |
