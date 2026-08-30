# Benchmarks

Cross-engine comparison of local inference servers on Apple Silicon, and the
harness that produced it. Numbers here are honest measurements, including the
ones lokal loses — they exist to steer the roadmap, not to advertise.

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
| lokal `-b hybrid` | main | **11,080** | 10,879–11,080 | 239 tok/s | **382 tok/s** |
| llama.cpp | b9960 | 9,641 | 9,511–9,641 | 232 tok/s | 364 tok/s |
| lokal `-b metal` | main | 8,271 | 8,251–8,271 | 238 tok/s | 359 tok/s |
| oMLX | 0.6.3 | 4,544 | 4,493–4,544 | **262 tok/s** | 335 tok/s |

The hybrid row implies the split-prefill ladder is exported
(`./run.sh export-ane-split`, once per model) — with it present, splitting
is simply what `-b hybrid` does. Measured without the ladder in the same
session (`LOKAL_SPLIT_PREFILL=0` reproduces this state): prefill 8,738
(spread 8,576–8,738), decode 237, aggregate 367.

All five configurations were measured in one session with the machine idle
between passes, and every pass is a row in `results.jsonl` with its machine
state attached. An earlier session on a machine warm from hours of
benchmarking read 15–20% lower across the board and reordered the top two —
if you are reproducing these, let the machine cool first.

vLLM Metal 0.1.0 measured 253 tok/s decode / 217 aggregate on 2026-08-29
(with some completions stopped early); its venv was not rebuilt for this
pass, so it stays out of the same-session table. SGLang has no macOS
support.

## Reading

- **Prefill**: lokal's single-device paths (8,271 metal, 8,738 ladder-less
  hybrid) sit below llama.cpp's 9,641 — its Metal backend is still the better
  GPU-only prefill, and lokal-metal's number is itself the 2026-08-30 rewrite
  that took this path from 1,461 tok/s using the same Metal 4 tensor-ops
  matmul mechanism llama.cpp uses. What wins the column is **split prefill**:
  the front layers of each chunk run on the Neural Engine while the GPU works
  the back layers of the previous chunk, so one prompt keeps both engines
  busy — 11,080 tok/s, 15% past llama.cpp, and the one result here a
  GPU-only engine cannot copy. It is what `-b hybrid` does whenever the
  ladder from `./run.sh export-ane-split` is present (~150 MB of disk per
  front graph); `LOKAL_SPLIT_PREFILL=0` turns it off for A/B runs.
- **Decode**: oMLX leads single-stream in every pass (262); lokal (236–239)
  and llama.cpp (232) are close behind and effectively tied with each other.
  Split prefill does not touch the decode path, so its decode differences
  from the ladder-less hybrid are noise.
- **Concurrency**: continuous batching — one weight read serving every
  active request — puts every lokal configuration at or above llama.cpp
  (359–382 vs 364), with oMLX last (335). Split prefill adds the most here
  for the same reason it wins single-stream: a joining request is prefilled
  across both engines instead of stalling the batch that is decoding.

## Long context

The table above is a ~500-token prompt, which is not the length real
documents arrive at. The harness measures long prompts too:
`bench_engines.py --prompt-tokens N` builds natural-text prompts from a
pinned public-domain corpus (auto-downloaded, sha256-checked), and
`run_longctx.py` drives the full engine × size matrix one server at a time,
with `summarize_longctx.py` building the tables.

The long-context matrix is pending re-measurement: the lokal-only baseline
taken on 2026-08-29/30 (2k–24k, in git history) predates the flash-prefill
rewrite and split prefill, so those rows are obsolete as a comparison.
Facts that survive from that pass: the windowed ANE graph covers the first
8,192 positions of a long prompt (Metal takes the tail), and a machine pays
a one-time ~250 s Apple ANE-compiler cost on the first load of that graph
(cached afterwards).

## Reproduction

One engine, one row — the server is started and stopped for you:

```bash
python3 benchmarks/bench_engines.py --engine lokal-ane
python3 benchmarks/bench_engines.py --engine llamacpp
python3 benchmarks/bench_engines.py --engine omlx --out benchmarks/results.jsonl

# lokal-ane splits by default once ./run.sh export-ane-split has run;
# the ladder-less hybrid row, for A/B against it:
LOKAL_SPLIT_PREFILL=0 python3 benchmarks/bench_engines.py --engine lokal-ane
```

`--engine` takes any key from `engines.py`: `lokal-metal`, `lokal-ane`,
`lokal-metal-cli`, `lokal-ane-cli`, `llamacpp`, `omlx`. Add `--model qwen` to
run the same engine on Qwen2.5-0.5B-Instruct instead of the default
SmolLM2-135M-Instruct, and `--prompt-tokens N` for a long prompt. Sweeping
sizes or engines is the same registry through the other driver:

```bash
python3 benchmarks/run_longctx.py --engines lokal-ane-cli,llamacpp,omlx \
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
