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
| lokal `-b hybrid` | main | **11,896** | 11,608–11,896 | 237 tok/s | **380 tok/s** |
| lokal `-b metal` | main | 9,920 | 9,490–9,920 | 231 tok/s | 362 tok/s |
| llama.cpp | b9960 | 9,803 | 9,687–9,803 | 232 tok/s | 366 tok/s |
| oMLX | 0.6.3 | 4,623 | 4,528–4,623 | **261 tok/s** | 335 tok/s |

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

- **Prefill**: lokal's GPU-only path (9,920 tok/s) now edges past llama.cpp
  (9,803) — that path went 1,461 → 8,271 → 9,920 in a day, first from a
  flash-attention rewrite plus Metal 4 tensor-ops matmuls (the same
  mechanism llama.cpp's Metal backend uses), then from moving the k/v
  projections onto tensor ops and letting independent prefill dispatches
  run concurrently. The lead, though, comes from not being GPU-only:
  `-b hybrid` pipelines each prompt across both engines — the front layers
  of a chunk on the Neural Engine while the GPU works the back layers of
  the previous chunk — for 11,896 tok/s, 21% past llama.cpp. Those two
  results compound: the GPU work is one of the pipeline's two stages, so
  the same-day GPU gain lifted the hybrid number from 11,080 to 11,896.
- **Decode**: oMLX leads single-stream (261); lokal (231–237) and llama.cpp
  (232) are tied behind it. Prefill work does not touch the decode path.
- **Concurrency**: continuous batching — one weight read serving every
  active request — puts lokal's hybrid on top (380 vs llama.cpp's 366 and
  oMLX's 335). A joining request is prefilled across both engines instead
  of stalling the batch that is already decoding.

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
python3 benchmarks/bench_engines.py --engine lokal-hybrid
python3 benchmarks/bench_engines.py --engine llamacpp
python3 benchmarks/bench_engines.py --engine omlx --out benchmarks/results.jsonl

# lokal-hybrid splits by default once ./run.sh export-ane-split has run;
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
