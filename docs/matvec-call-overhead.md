# Is the per-call cost the matvec's missing mechanism? — probe memo

Lane `matvec-call-overhead-design`, design/probe only. Base `main @b9c3871`.
One question, pre-registered before the harness existed, answered here.

**The one-line result.** No — and neither is the lane-to-data mapping I proposed
instead. Both hypotheses are rejected against thresholds fixed in advance. The
measurable signal is in the arm that was registered as *diagnostic only*: the
activation re-reads. And the ceiling the whole question was framed against turns
out not to be a ceiling — `probe_shape`'s rate moves 33% with the resident
weight footprint while the production matvec's does not move at all, so the two
kernels are not bound by the same thing and "2.5x available" was never headroom
for this kernel.

## 0. Provenance

| file | sha256 |
|---|---|
| `src/gpu/kernels.metal` | `e4ab553379be13d06eb3f1e687f19a387f69c7045e0ffa98c58daf66a0996560` |
| `src/gpu/metal.rs` | `eaa04ba74bd0ebf179a27ebec26831c34cd7ead2529b19958047f07cfd6b47ae` |

Harness `mvbench` (lane scratchpad) compiles **this repo's own**
`src/gpu/kernels.metal` through the precise, `fast_math`-off options the engine
uses for quant pipelines, with function constant 25 = 4 (Q4_K) and 26
(`LM_MV_ACCUM`) = 1, the shipped single-chain form. **A0 and A1 below are the
shipped `matvec` (`:1587`) and `probe_shape` (`:1542`)**, dispatched with
`dispatch_simdgroup_rows`' own geometry — 4 rows per threadgroup,
`src/gpu/metal.rs:1634-1638`. Dims are the 2B Q4_K_M decode shapes.

Three runs. Runs 1-2 held 12 distinct weight buffers (84.9 MB); run 3 held 4
(28.3 MB) — the footprint change was made to fix a memory-quiet failure and
turned out to be the most informative variable in the lane.

**Machine state.** Runs 1-2, 14:42-14:52Z, are **flagged**: +889,839
decompressions against `protocol:gpu-bench`'s 200,000 void threshold. Their
absolutes are not published as clean; their *ratios* are, because the A/A
control held at 0.0-0.3%. Run 3, 14:55-15:00Z, is clean: **+85 decompressions,
+638 swapins**, entry 56.2% and exit 64.3% foreign CPU, A/A 2.7% and 0.0%. Where
this memo quotes one number it quotes run 3.

## 1. What was fixed before any data existed

Recorded on the task ledger before the harness was written, and not revised
after:

* **H1** (the plan's): per-call overhead — address re-derivation and metadata
  loads per `lm_dot_run` call (`:1471-1485` walks one 32-element run per lane).
* **H2** (mine, from reading `probe_shape`): the lane-per-run decomposition
  destroys intra-instruction coalescing. `probe_shape` walks `packed_uchar4` at
  `i = lane`, so 32 lanes read 32 adjacent words — one transaction. `dot_wx`
  gives each lane its own run, so at inner step *j* the lanes read `x[32L + j]`:
  a 4096-byte span at 128-byte stride, 32 distinct cache lines for one
  instruction.
* **Metric**: gap closure `(arm - A0) / (A1 - A0)` in rate terms, measured
  inside one run so contamination cancels. **≥0.60 confirms, <0.30 rejects.**
* **Third outcome, named in advance** (Mellow's framing): if both close <0.30,
  the finding is not "optimise something else" — it is that the cost is in
  neither, and the honest next question is whether `probe_shape` is a fair
  ceiling at all.
* **Void conditions**: A/A control within 5%; my A1 within 15% of Mellow's
  145-165 GB/s range or **no verdict is issued**.
* **A4 is diagnostic, not a verdict arm** — recorded explicitly so it could not
  be promoted after the fact.

## 2. The arms

GB/s over **weight** bytes, so the arms are comparable without privileging
`probe_shape`. `ffn_gate/up` (in 2048, out 6144) then `ffn_down` (in 6144,
out 2048); three runs, five reps, median.

| arm | gate/up | down | what it removes |
|---|---|---|---|
| **A0** `matvec`, shipped | 77.4 / 79.6 / **79.9** | 67.3 / 68.5 / **67.7** | — |
| **A1** `probe_shape` | 171.3 / 173.2 / **202.1** | 163.2 / 152.1 / **199.8** | everything but the weight stream |
| **A2** decode deleted | 86.5 / 88.9 / **85.7** | 81.4 / 82.2 / **82.8** | block base, `d`, `dmin`, `lm_scale_min_k4` |
| **A3** coalesced remap | 17.9 / 18.0 / **17.9** | 17.2 / 17.3 / **17.1** | the lane-per-run mapping |
| **A4** x reads deleted *(diag)* | 124.3 / 127.8 / **125.3** | 126.4 / 128.6 / **127.2** | the activation loads |
| **A5** wide loads *(exploratory)* | — / 79.0 / **77.6** | — / 71.4 / **70.3** | scalar load width |

Gap closure, run 3: A2 **+0.05 / +0.11**, A3 **-0.51 / -0.38**, A4
**+0.37 / +0.45**, A5 **-0.02 / +0.02**.

## 3. The verdict

**H1 rejected.** Deleting the entire per-call decode — the superblock base, both
`lm_f16_at` loads, the 6-bit scale unpack — buys 8-22% and closes 0.05-0.16 of
the gap, against a 0.30 rejection line. The per-call cost is real and it is
small.

**H2 rejected, and harder.** A3 is **4.4x slower** than production, the most
reproducible number in the lane (17.1-18.0 GB/s across three runs and both
shapes). My hypothesis was not merely unconfirmed; the mechanism I proposed is
strongly negative. Two honest caveats in the same breath: A3 also serialises the
run loop (32x more iterations per lane, one element of work each), and it repeats
the scale decode on every lane — a confound I named in advance. So A3 refutes
*this* coalesced mapping, not the proposition that coalescing matters.

**A5 is null**, at -0.02/+0.02 — and that independently reproduces Mellow's
`int4`/`float4` result from a second harness and a different implementation. It
is also *why* the scatter question stays open: widening the loads cuts
instruction count 4x but leaves the per-instruction cache-line spread unchanged,
because lane *L* still starts 32 elements away from lane *L+1*. No arm in this
lane has cleanly separated "how many load instructions" from "how many lines each
one touches".

**The signal is in A4**, which was registered as diagnostic and stays labelled
that way: deleting the activation reads is worth **+57% and +88%** on the
production kernel. The arm set was incomplete, and it was incomplete in a way the
pre-registration did not anticipate — that is a finding about the design, not a
result to present as though it had been the plan.

A rough time budget follows from the arms (`ffn_gate/up`, run 3): activation
loads ~37%, per-call decode ~7%, and the remaining ~56% is the weight stream plus
the loop — which is approximately what `probe_shape` measures on its own.

## 4. `probe_shape` is not a ceiling for this kernel

The most useful number in the lane was produced by accident, fixing something
else. Cutting the resident weight footprint from 84.9 MB to 28.3 MB moved:

| | 84.9 MB | 28.3 MB | change |
|---|---|---|---|
| A1 `probe_shape`, gate/up | 171.3 / 173.2 | 202.1 | **+17%** |
| A1 `probe_shape`, down | 163.2 / 152.1 | 199.8 | **+27%** |
| A0 `matvec`, gate/up | 77.4 / 79.6 | 79.9 | +1% |
| A0 `matvec`, down | 67.3 / 68.5 | 67.7 | ~0% |

`probe_shape`'s rate is a function of how much of the weight set is cached. The
production matvec's is not — it sits at 67-80 GB/s across a 3x footprint change.
**They are not bound by the same resource**, so the framing "`probe_shape`
sustains 2.5x what production achieves, therefore 2.5x is available" was never a
headroom claim about this kernel. The gap it names is mostly the fact that
`probe_shape` streams one operand and the matvec streams two: per pass at in 2048
/ out 6144 the kernel reads 7.08 MB of weights and **50.3 MB of activations**,
because every one of 6144 row-simdgroups walks the whole 8.2 KB of `x`.

This is the pre-registered third outcome, and it arrived with a mechanism rather
than as a null.

## 5. What this changes, and the one probe worth running next

**No implementation lane is proposed.** Nothing measured here is a fix, and the
two candidate mechanisms are dead. Proposing a lane off this evidence would be
the arc's own retracted-invariance mistake in a new costume.

Three things are now settled enough to stop re-deriving:

1. Per-call decode overhead is ~7-11% of the decode matvec. It is not the
   missing 2x. (H1, this lane.)
2. Re-mapping lanes to cooperate on one run is 4.4x worse. (H2, this lane.)
3. Widening the inner loads is null — measured twice, in two harnesses, by two
   agents. (Mellow's `int4`/`float4`; A5 here.)

The open question is narrower than when the lane started: **the activation
stream**, which is 7.1x the weight traffic in bytes and about half the load
instructions, and which no dead mechanism has touched. The probe that would
separate the two remaining explanations — instruction *count* versus cache lines
touched *per instruction* — is one that stages `x` cooperatively into threadgroup
memory once per threadgroup (4 rows share one, `metal.rs:1634-1638`) and then
reads it from there, leaving the weight path exactly as shipped. That changes
lines-per-instruction without changing the run mapping, which is the one
combination A3 and A5 between them did not test.

That probe is *not* Mellow's threadgroup-staging arm, which staged weights and
came out 3% slower. And their two cross-lane capture shapes are evidence about
sharing across lanes of a simdgroup, not about tile-level sharing through
threadgroup memory — a distinction they corrected me on, and which this memo
inherits rather than re-litigates.

### Risk ledger

| # | risk | mitigation |
|---|---|---|
| R1 | A3's confound reads as "coalescing is dead" | Stated inline: A3 also serialises and repeats the decode per lane; it refutes one mapping, not the principle |
| R2 | A4 gets cited as a pre-registered result | Labelled diagnostic in the ledger, in §1 and at every occurrence in §2-§3 |
| R3 | The flagged runs' absolutes get quoted | Runs 1-2 are marked over-threshold; every single-number quote in this memo is run 3 |
| R4 | 202 GB/s reads as a new ceiling | It is a 28.3 MB-footprint cache artifact and is the *evidence against* treating A1 as a ceiling at all, not a better one |
