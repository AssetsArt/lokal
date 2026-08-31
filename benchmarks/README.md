# Benchmarks

Cross-engine comparison of local inference servers on Apple Silicon, and the
harness that produced it. Numbers here are honest measurements, including the
ones lokal loses — they exist to steer the roadmap, not to advertise.

## GGUF on `-b lowmem` — two rows held, and how to collect them

`-b lowmem` runs quantized GGUF checkpoints, and on a 16 GB machine the
capability is the result worth reporting:

| checkpoint | file | result |
|---|---|---|
| Qwen3-32B Q4_K_M | 19.76 GB | opens and answers correctly on a 16 GB box |
| Qwen2.5-14B Q4_K_M | 8.99 GB | pool-resident, answers correctly |
| Qwen3-0.6B Q4_K_M | 0.46 GB | 5/5 prompts byte-identical to llama.cpp, 48 greedy tokens |
| Qwen2.5-0.5B Q4_K_M | 0.40 GB | ~120 tok/s decode (resident, measured clean) |

**The 14B and 32B decode rows are deliberately not here.** Both were measured
in a quieted window and both were discarded: with 8.1 GB of weights on a 16 GB
box, macOS compresses the pool rather than swapping it, every decode pass
touches all of it, and the measured ~1.8 GB/s decompression rate accounted for
the throughput almost exactly. Such a row measures the compressor, not the
engine, and would not reproduce for a reader. What reproduces is the
requirement: **14B Q4 needs ~10 GB of genuinely free RAM to decode at resident
speed** (8.1 GB pool + 0.5 GB KV + 0.3 GB activations + overhead).

To collect the held rows on an actually-idle machine — a fresh boot, or after
any agent crew exits — run:

```
zsh benchmarks/collect-gguf-rows.sh
```

(models default to `~/.cache/gguf/`; override with
`M14=/path/to/14b.gguf M32=/path/to/32b.gguf`)

It gates itself: CPU-quiet before and after, `vm_stat` swapins and
decompressions sampled around each run (>20k swapins or >200k decompressions
fails the row), one run per model, and an untimed `vmmap` footprint pass kept
separate because `vmmap` suspends its target and would corrupt a timed number.
If it refuses, the machine was not idle enough — the refusal is the result, not
an obstacle to work around.

### Checking a quantized type against llama.cpp

`agree-vs-llamacpp.py` compares lokal's greedy output against llama.cpp on the
**same GGUF file** — the only comparison that isolates our dequantization from
everything else. Start `llama-server` on the file, then:

```
python3 benchmarks/agree-vs-llamacpp.py <model.gguf>
```

Two rules are baked in because both were learned the hard way. It talks to
`llama-server`'s `/completion` with `cache_prompt: false`, never `llama-cli` —
`llama-cli` applies the GGUF's chat template to Instruct models even under
`-no-cnv`, and `-st` is a conversation flag that re-enables it, so its text
reads as total disagreement at token 1 while the kernels are fine. And one
prompt is deliberately long: every prompt here was once under twenty tokens,
and a real scratch-buffer sizing bug passed the check 5/5 because nothing
reached the oversized region. Short prompts test the math, long prompts test
the sizes.

Agreement, not identity, is the bar — summation order differs, so greedy
decoding eventually elects a different token. An EARLY divergence is the signal
that means a real bug.

## Setup

- Hardware: Apple M1 Pro, 16 GB RAM, macOS 26.5
- Model: SmolLM2-135M-Instruct, full precision on every engine
  (lokal: f16 weights on GPU · llama.cpp: GGUF F16 · oMLX: MLX bf16 from
  `mlx-community/SmolLM2-135M-Instruct`)
- Workload: ~500-token prompt through each server's HTTP API, greedy
  (`temperature 0`), `max_tokens 128`
- Method: `bench_engines.py` — median of 5 runs per single-request metric,
  median of 3 runs for the 4-concurrent metric, one warmup request first.
  That whole procedure is one *pass*; the table reports the best of 3 passes
  per engine, applied identically to every engine (best-of-N for us and
  median-for-them would be a rigged table).
  Every request starts with a unique nonce so server-side prompt caches
  cannot answer from cache (llama.cpp otherwise reuses the prefix KV and
  reports an 8 ms "prefill"). Decode tok/s is marginal:
  `(tokens_128 − 1) / (wall_128 − wall_1)`, which cancels prompt processing
  out of the number.
- Machine hygiene: the harness refuses to record a row unless the machine is
  verified quiet (per-process cpu-time deltas over a 2 s window plus load
  average — not `ps %CPU`, which is a decaying lifetime average), and every
  row in `results.jsonl` carries the machine state it was measured under.
  All five configurations below were measured in the same session.

## Results (2026-08-30)

Best of 3 passes per engine — the same statistic for every row, on a machine
left cool between passes. The spread column is there because it is the honest
check on the headline: a number nobody can reproduce is worse than a slower
one that everybody can.

| engine | version | prefill tok/s | prefill spread | decode | 4x concurrent aggregate |
|---|---|---|---|---|---|
| lokal `-b hybrid` | main | **11,986** | 11,898–11,986 | 238 tok/s | **373 tok/s** |
| llama.cpp | b9960 | 9,880 | 9,727–9,880 | 232 tok/s | 363 tok/s |
| lokal `-b metal` | main | 9,900 | 9,742–9,900 | 252 tok/s | 364 tok/s |
| oMLX | 0.6.3 | 4,586 | 4,525–4,586 | **261 tok/s** | 336 tok/s |

All four engines were measured in one session with the machine idle
between passes, and every pass is a row in `results.jsonl` with its machine
state attached. An earlier session on a machine warm from hours of
benchmarking read 15–20% lower across the board and reordered the top two —
if you are reproducing these, let the machine cool first.

vLLM Metal 0.1.0 measured 253 tok/s decode / 217 aggregate on 2026-08-29
(with some completions stopped early); its venv was not rebuilt for this
pass, so it stays out of the same-session table. SGLang has no macOS
support.

## Reading

- **Prefill**: lokal's GPU-only path (9,900 tok/s) runs level with llama.cpp
  (9,880) — that path went 1,461 → 8,271 → ~9,500 in a day, first from a
  flash-attention rewrite plus Metal 4 tensor-ops matmuls (the same
  mechanism llama.cpp's Metal backend uses), then from moving the k/v
  projections onto tensor ops and letting independent prefill dispatches
  run concurrently. The lead, though, comes from not being GPU-only:
  `-b hybrid` pipelines each prompt across both engines — the front layers
  of a chunk on the Neural Engine while the GPU works the back layers of
  the previous chunk — for 11,986 tok/s, 21% past llama.cpp, and it holds
  that lead across the whole 500–2,000 band (next section). Those results
  compound: the GPU work is one of the pipeline's two stages, so the
  same-day GPU gain lifted the hybrid number from 11,080 to 11,896 without
  anyone touching the ANE side.
- **Decode**: oMLX leads single-stream (261); lokal (231–237) and llama.cpp
  (232) are tied behind it. Prefill work does not touch the decode path.
- **Concurrency**: continuous batching — one weight read serving every
  active request — puts lokal's hybrid on top (380 vs llama.cpp's 366 and
  oMLX's 335). A joining request is prefilled across both engines instead
  of stalling the batch that is already decoding.

## The 500–2,000 curve

A single ~500-token point can hide a curve that sags between rungs, so the
band real prompts live in was measured end to end (2026-08-30, same
session, each lokal/llama.cpp pair back to back, one
`bench_engines.py --prompt-tokens` row per point — median of 5 requests —
with the machine state recorded on every row; llama.cpp was re-measured at
582 and 794 and agreed with its first pass within noise). lokal is
`-b hybrid` with the ladder `run.sh export-hybrid` ships, no env vars:

| prompt tokens (lokal · llama.cpp) | lokal `-b hybrid` | llama.cpp | edge |
|---|---|---|---|
| 496 · 517 | **11,715** | 9,731 | +20% |
| 582 · 603 | **11,355** | 10,536–10,706 | +6–8% |
| 699 · 720 | **12,375** | 10,726 | +15% |
| 794 · 815 | **11,173** | 10,722–10,792 | +4% |
| 982 · 1,003 | **12,397** | 10,832 | +14% |
| 1,253 · 1,274 | **10,829** | 10,310 | +5% |
| 1,530 · 1,551 | **10,569** | 9,675 | +9% |
| 1,929 · 1,950 | **9,824** | 9,420 | +4% |

(The token counts differ per engine because each tokenizes the same
character slice itself.) Holding the whole band took four ladder changes,
each measured against a version without it: the stride guard counts real
chunks (`div_ceil` — the old `len >= 3*s` test pushed 513–767-token
prompts onto a stride whose ladder reaches only 512 tokens), a 256×512
rung closes the mid-band rung gap, the 256-wide family's split point moved
from 15 front layers to 13 (15, 12 and 11 all measured slower — the two
engines re-balance as rungs widen), and a P=0 rung lets the first chunk —
which has no past — skip paying the 256-position past attention every
other rung computes and masks. Handing short leftover chunks to the GPU
instead was measured slower every time it was tried.

## Long context

The table above is a ~500-token prompt, which is not the length real
documents arrive at. The harness measures long prompts too:
`bench_engines.py --prompt-tokens N` builds natural-text prompts from a
pinned public-domain corpus (auto-downloaded, sha256-checked), and
`run_longctx.py` drives the full engine × size matrix one server at a time,
with `summarize_longctx.py` building the tables.

Long prompts are where routing assumptions go stale: a ~7.7k-token Qwen
prompt through the old fixed-width windowed graph took 8.0 s (961 tok/s)
while plain `-b metal` took 4.1 s — the graph charges its full 8,192-position
attention on every chunk, which was right when Metal prefilled at 1,461
tok/s and inverted when Metal got its 7x. The 2026-08-30 routing rework
retires that graph from default routing (it cannot win at any length on
this hardware; `LOKAL_WINDOWED_PREFILL=1` revives it for A/B) and gives
Qwen its own split ladder — stride 512, 9 of 24 front layers (8 and 10
both measured slower), rungs {0, 2048, 4096, 7168}.

Measured same-session HTTP rows, Qwen2.5-0.5B-Instruct (single pass per
point, machine state on every row; llama.cpp needed `--ctx 34816` — its
`-c` is divided across `--parallel` slots, and the harness's shape-derived
default starves a long prompt to a 400):

| prompt tokens | `-b hybrid` | `-b metal` | llama.cpp | vs metal |
|---|---|---|---|---|
| ~494 | 4,063 | 3,854 | **4,630** | +5% |
| ~2,067 | **4,392** | 3,161 | 4,323 | +37% |
| ~4,039 | **3,825** | 2,593 | 3,766 | +48% |
| ~7,692 | 2,836 | 1,831 | **2,960** | +55% |

The split now also edges llama.cpp in the 2k–4k band; at 7.7k it lands 4%
short of llama.cpp on a machine warm from a day of benchmarking (the
integrator's cool-machine spot read 2,989 vs 2,966 before this lane's
numbers were re-taken — treat the GOAL as within thermal noise, not won),
and short Qwen prompts remain llama.cpp's (4,630 vs our plain-head 4,063 —
the ladder does not engage below three chunks). SmolLM2's ladder gained
long-band rungs {3072, 4608, 7168} for the same reason — at ~7.2k it used
to drain to Metal after 2.3k tokens and win by only 1%; it now pipelines
the whole prompt (+5%, byte-identical output to `-b metal`). The split is
never slower than Metal at any measured length on either model — that
property, not any single number, is what the routing guard enforces.

A cost fact that survives from the earlier pass: each Core ML graph
carries a full weight copy (~0.9 GB per graph on Qwen), so a hybrid
process peaks around 4 GB there — the exit-137 sweep death on a 16 GB
machine was the OS reclaiming memory with another engine's server
resident, and retiring the windowed graph from default loading removes
the biggest avoidable slice of that footprint.

## Long context: Qwen2.5-0.5B at 8k

The 500–2,000 curve above is SmolLM2. Long prompts need a model with the
context for them, so the 8k point is Qwen2.5-0.5B-Instruct (~7.7k-token
corpus prompt, HTTP transport, cool machine, median of 5 requests):

| engine | prefill | tok/s |
|---|---|---|
| llama.cpp b9960 | 2.61 s | **2,958** |
| lokal `-b metal` | 3.17 s | 2,429 |

`-b metal` was at 1,884 tok/s here this morning; retuning the flash
attention tiles for long K/V loops (96 query rows × 32 positions — row reuse
is what pays once the loop is thousands of positions long, while wider
position tiles blow past the 16 KB of shared memory that keeps two
threadgroups per core) and widening the GEMM row tile brought it to 2,429.
That fixed the *shape* of the curve: lokal now falls off at llama.cpp's own
rate from 500 to 7.7k tokens (−36% against −35%). What remains is a
length-independent gap of ~15% on Qwen, present even at 500 tokens where
attention is a rounding error — a model-level cost (Qwen's q/k/v biases,
its narrow 128-wide kv projections, 14:2 GQA) that tile tuning cannot
reach, and the next thing to profile.

The `-b hybrid` row at 8k is deliberately absent: Qwen's split ladder is
still being tuned, and a number measured against a ladder that is about to
change would be obsolete before it was published.

## Reproduction

One engine, one row — the server is started and stopped for you:

```bash
python3 benchmarks/bench_engines.py --engine lokal-hybrid
python3 benchmarks/bench_engines.py --engine llamacpp
python3 benchmarks/bench_engines.py --engine omlx --out benchmarks/results.jsonl

# lokal-hybrid splits by default once ./run.sh export-hybrid has run;
# the ladder-less hybrid row, for A/B against it:
LOKAL_SPLIT_PREFILL=0 python3 benchmarks/bench_engines.py --engine lokal-hybrid
```

`--engine` takes any key from `engines.py`: `lokal-metal`, `lokal-hybrid`,
`lokal-metal-cli`, `lokal-hybrid-cli`, `llamacpp`, `omlx` (the old
`lokal-ane*` spellings still resolve). Both drivers take the same
`--model smol|qwen` with the same default, so a single row and a sweep
compare like with like; every result line records which model produced it.
SmolLM2 stops at 8k tokens, so long sizes need `--model qwen` — the sweep
says so and stops rather than handing you a column of failures. Add
`--prompt-tokens N` for a long prompt on a single row. Sweeping sizes or
engines is the same registry through the other driver:

```bash
python3 benchmarks/run_longctx.py --engines lokal-hybrid-cli,llamacpp,omlx \
    --sizes 2000,6000,10000,16000,24000,32000
```

To measure a server you are running yourself, skip `--engine` and describe it:

```bash
python3 benchmarks/bench_engines.py --api openai --url http://127.0.0.1:8081/v1 \
    --model-name <name> --runs 5 --concurrency 4
```

llama.cpp needs a GGUF and oMLX needs an MLX copy of the model; the paths are
at the top of `engines.py` and a missing one is reported by name.

Raw per-run output with machine-state stamps: [results.jsonl](results.jsonl).

## The metal-quant Studio acceptance

`collect-metal-quant.sh` is the one-command gate for quantized GGUF on
`-b metal` with full attention — written for the 32 GB M2 Ultra where the 27B
Q4_K_M should sit resident:

```bash
MODEL="owner/repo:Q4_K_M" ./benchmarks/collect-metal-quant.sh
```

It resolves the tag, refuses a non-quiet machine, reports prefill+decode
tok/s from a timed run, proves two greedy runs byte-identical, and samples
`phys_footprint` in a separate UNTIMED run (vmmap suspends its target) —
asserting the footprint tracks the quant file size, not an f32 expansion.
Paste the whole output back.
