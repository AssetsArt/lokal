# x loaded once, reused across R output rows — probe memo

Lane `matvec-multirow-probe`, design/probe only. Base `main @7412a7f`.
The idea the previous lane's control pointed at. It does not work.

**The one-line result.** The implementation line fails decisively — no arm, in
any run, on either shape, reached 1.25x. The probe line is marginal at best. The
informative part is **R=2**: it costs *no* occupancy at all and cuts x-loads 25%,
and is still **slower** (0.78-0.92x). Multi-row does not remove load scatter, it
**moves it from the activation path to the weight path**, and that is a worse
trade than the one it was meant to fix.

**And a measurement failure worth stating first.** I could not obtain a valid A/A
window on `ffn_gate/up` in four attempts. The verdict below does not depend on
the precision I failed to get — but the reader should know I did not get it.

## 0. Provenance and the honest measurement status

`src/gpu/kernels.metal` @ `e4ab553379be13d06eb3f1e687f19a387f69c7045e0ffa98c58daf66a0996560`.
Harness `mrbench` (lane scratchpad) compiles this repo's own kernels through the
precise `fast_math`-off options, function constant 25 = 4 (Q4_K). **C0 is the
shipped `matvec`** (`:1587`).

| run | arms | A/A gate/up | A/A down | memory |
|---|---|---|---|---|
| 1, sequential | 6 | **21.2% FAIL** | 5.3% FAIL | +1,959 decomp |
| 2, interleaved | 6 | **8.1% FAIL** | 9.3% FAIL | +147 |
| 3, interleaved + rotated | 6 | **11.7% FAIL** | 7.4% FAIL | +3 |
| 4, focused 3-arm | 3 | **24.6% FAIL** | **3.0% PASS** | +5 |

Memory was clean in every run — this is not the decompression failure of the
previous lane. Two harness fixes were made *because* of the failures, not
around them: arms interleaved rep-by-rep so a burst hits all arms equally, then
the order **rotated** so every arm visits every position. Neither fixed it. Raw
per-rep dumps say why: the production arm is genuinely slow in 4 of 6 reps
(`6.29 6.74 4.45 6.28 4.43 7.30` ms) while the *identical* control arm is stable
(`4.45 4.35 4.35 5.45 4.34 4.44`) — isolated slow reps a 6-sample median cannot
absorb, on a box whose desktop is live.

**I stopped at four runs rather than re-running until one passed.** Taking the
first window that clears an A/A after several that did not is selection bias, and
it is the exact failure the pre-registration exists to prevent. `ffn_down` held a
valid A/A twice (4.7%, 3.0%) and those two windows carry the load below.

## 1. The arms

`ffn_gate/up` then `ffn_down`, all four runs, as multiples of C0:

| arm | maxThreads/tg | gate/up | down |
|---|---|---|---|
| **C0** shipped | 1024 | 1.00x | 1.00x |
| **C1** x deleted *(bounding)* | 1024 | 1.63-2.05x | 1.83-1.88x |
| **C2** R=2 | **1024** | 0.78 / 0.84 / 0.92 | 0.78 / 0.83 / 0.83 |
| **C3** R=4 | 640 | 0.80 / 0.85 / 0.92 / 1.17 | **0.99 / 1.08 / 1.08 / 1.12** |
| **C4** R=8 | 448 | 0.50 / 0.60 / 0.62 | 0.58 / 0.62 / 0.62 |

Every multi-row arm is **bit-identical** to production in every run — the
pre-registered pass condition held, so these are performance results.

In the two windows with a **valid** A/A (both `ffn_down`), C3 read 1.08x and
1.12x: astride the 1.10x probe line, nowhere near the 1.25x implementation line.

## 2. Why it fails, and the arm that shows it

**R=2 is the result that matters.** It costs *nothing* in occupancy — the
pipeline still reports 1024 max threads per threadgroup, identical to production
— and it does cut x-side loads by 25%. It is nonetheless slower, consistently,
in all six measurements across both shapes.

So the failure is not register pressure. Register pressure is real and shows up
cleanly at higher R (1024 → 640 at R=4 → 448 at R=8, and C4's 0.5-0.6x tracks
it), but it cannot explain R=2.

What multi-row actually does to the inner loop: production walks one row's `qs[j]`
for j = 0..31 — **one sequential stream**. Multi-row walks `qs[i][j]` for i < R at
addresses one full row-stride apart (1152 B for Q4_K at in_dim 2048), so each `j`
iteration now touches **R cache lines instead of one**. The x-side scatter this
lane set out to remove has simply reappeared on the weight side, multiplied by R.

That also explains the shape dependence: `ffn_down` (in 6144, out 2048) has 3x
the runs per row, so the x-load saving is proportionally larger against the same
per-row weight scatter — and it is the only shape where C3 ever exceeds 1.0x.

**The existence proof agreed with this all along, and said so in advance.**
`ggml-metal-impl.h` defines `N_R0_Q4_K 2`, `N_R0_Q6_K 2`, `N_R0_Q8_0 2` and
`N_R0_Q5_K 1` — a heavily tuned reference chose R=2 for three types and *no*
multi-row for Q5_K. That was recorded on the ledger before measuring, together
with the guard that a large win at R=4 or R=8 should make me suspect my harness
rather than celebrate. The guard was not needed; the numbers went the other way.

Note also that upstream holds a whole 32-element slice of x in registers
(`float yl[16]`/`yh[16]`, `ggml-metal.metal:8415-8416`) and loops rows inside,
where this probe loops j outer to keep only one x value live. The j-outer form
carries R far more cheaply in registers — and it still loses, which makes the
weight-scatter explanation stronger, not weaker: the cheaper form does not help
because registers were never the binding constraint at R=2.

## 3. Verdict and what it closes

**No implementation lane.** The probe line is marginal on one shape and negative
on the other; the implementation line is not approached by any arm.

The decode matvec's dead list is now **eight** mechanisms:

| # | mechanism | result |
|---|---|---|
| 1-4 | traffic/occupancy shapes, weights staged, `int4`/`float4` widening, four accumulators | null or slower (Mellow; A5) |
| 5 | per-call decode overhead removed | 7-11%, not the 2x |
| 6 | lanes cooperating on one run | 4.4x slower |
| 7 | x staged in threadgroup memory | 1.6-1.9x slower |
| 8 | **x reused across R output rows** | **0.78-1.12x, best case marginal** |

Taken together with `docs/matvec-call-overhead.md` §4 — `probe_shape` is not a
ceiling for this kernel, its rate moving 33% with cache footprint while the
matvec's does not move at all — the honest reading is that **the decode matvec is
close to what this kernel shape can do on this hardware**, and the "2x available"
that motivated eight mechanisms was an artifact of comparing against a
one-operand probe.

What remains genuinely unexplained is narrow: C1 still says deleting the x reads
is worth +63-88%, and no mechanism has captured any of it, because every one
tried so far either relocates those loads (staging), widens them (A5), or trades
them for an equal-or-worse scatter elsewhere (this lane). That is a statement
about the loads being irreducible in this decomposition, not about there being an
easy win left.

**A concrete precondition for anyone who revisits this**: get a box that can hold
a 5% A/A over a six-arm run. Four of my eight windows across two lanes were void,
and that rate makes fine distinctions unaffordable regardless of which mechanism
is being tested.

### Risk ledger

| # | risk | mitigation |
|---|---|---|
| R1 | The negative gets read as precise | Four of four gate/up windows failed A/A; §0 leads with that, and the verdict rests on direction across runs, not on any run's precision |
| R2 | "Multi-row is dead" generalised beyond Q4_K decode | Q4_K only, this decomposition only; upstream ships R=2 profitably in a *different* inner shape |
| R3 | R=2's null read as register pressure | R=2 has *no* occupancy cost (1024 threads, same as production) — that is precisely why it is the informative arm |
| R4 | The dead list read as "matvec is finished" | It is a statement about eight tested mechanisms, not a proof of optimality |
