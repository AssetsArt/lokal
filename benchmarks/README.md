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
  Every request starts with a unique nonce so server-side prompt caches
  cannot answer from cache (llama.cpp otherwise reuses the prefix KV and
  reports an 8 ms "prefill"). Decode tok/s is marginal:
  `(tokens_128 − 1) / (wall_128 − wall_1)`, which cancels prompt processing
  out of the number.
- Machine hygiene: the harness refuses to record a row unless the machine is
  verified quiet (per-process cpu-time deltas over a 2 s window plus load
  average — not `ps %CPU`, which is a decaying lifetime average), and every
  row in `results.jsonl` carries the machine state it was measured under.
  All four engines below were measured back-to-back in one quiet session.

## Results (2026-08-30, after the flash-prefill rewrite)

| engine | version | prefill (~500 tok) | prefill tok/s | decode | 4x concurrent aggregate |
|---|---|---|---|---|---|
| lokal `-b ane` | main | **0.06 s** | 8,517 | 232 tok/s | **341 tok/s** |
| lokal `-b metal` | main | 0.06 s | 7,863 | 232 tok/s | 338 tok/s |
| llama.cpp | b9960 | **0.06 s** | **8,921** | 176 tok/s | 325 tok/s |
| oMLX | 0.6.3 | 0.12 s | 4,332 | **255 tok/s** | 324 tok/s |

vLLM Metal 0.1.0 measured 253 tok/s decode / 217 aggregate on 2026-08-29
(with some completions stopped early); its venv was not rebuilt for this
pass, so it stays out of the same-session table. SGLang has no macOS
support.

Aggregate throughput moves ±10% run to run; treat single-digit percentage
gaps in that column as a tie.

## Reading

- **Prefill**: llama.cpp, lokal-ane and lokal-metal are now within ~12% of
  each other at ~500 tokens. lokal-metal's 7,863 tok/s is the 2026-08-30
  flash-attention + Metal 4 tensor-ops rewrite (from 1,461 the same
  morning) — the same matmul mechanism llama.cpp's Metal backend uses. The
  ane number is the Neural Engine doing the same work off-GPU.
- **Decode**: oMLX leads single-stream (255), lokal holds 232 on both
  backends, llama.cpp trails at 176 (its own number moved 215 → 176 across
  our two measurement days with the same binary; server flags differ, so
  read that spread as configuration sensitivity, not a ranking).
- **Concurrency**: continuous batching (one weight read serves every active
  request) puts both lokal backends at the top of the aggregate column. The
  historic gap between them is gone: metal's admission stall was prompt
  prefill on the decode GPU, and the flash rewrite shrank it ~8x (216 → 338
  aggregate in one day). The ane hybrid still adds off-GPU prefill on top —
  its time-to-first-token stays the best under load.

## Long context

The table above is a ~500-token prompt, which is not the length real
documents arrive at. The harness measures long prompts too:
`bench_engines.py --prompt-tokens N` builds natural-text prompts from a
pinned public-domain corpus (auto-downloaded, sha256-checked), and
`run_longctx.py` drives the full engine × size matrix one server at a time,
with `summarize_longctx.py` building the tables.

The long-context matrix is pending re-measurement: the lokal-only baseline
taken on 2026-08-29/30 (2k–24k, in git history) predates the flash-prefill
rewrite that multiplied the Metal tail's rate 4–6x, so those rows are
obsolete as a comparison. Facts that survive from that pass: the ANE
windowed graph covers the first 8,192 positions of a long prompt (Metal
takes the tail), and a machine pays a one-time ~250 s Apple ANE-compiler
cost on the first load of that graph (cached afterwards).

## Reproduction

```bash
# classic table, one engine at a time (server lifecycle included):
python3 benchmarks/run_longctx.py --engines lokal-ane-http --sizes 500 --runs 5

# or against an already-running server:
python3 benchmarks/bench_engines.py --api openai --url http://127.0.0.1:8081/v1 \
    --model <name> --runs 5 --concurrency 4

# long-context matrix (when re-measuring):
python3 benchmarks/run_longctx.py --engines lokal-ane-cli,llamacpp,omlx \
    --sizes 2000,6000,10000,16000,24000,32000
```

Raw per-run output with machine-state stamps: [results.jsonl](results.jsonl).
