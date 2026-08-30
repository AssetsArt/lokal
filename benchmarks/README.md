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

## Results (2026-08-30, after the flash-prefill rewrite and split prefill)

| engine | version | prefill (~500 tok) | prefill tok/s | decode | 4x concurrent aggregate |
|---|---|---|---|---|---|
| lokal `-b hybrid` + split | main | **0.05 s** | **9,920** | 232 tok/s | **343 tok/s** |
| lokal `-b hybrid` | main | 0.06 s | 8,517 | 232 tok/s | 341 tok/s |
| lokal `-b metal` | main | 0.06 s | 7,863 | 232 tok/s | 338 tok/s |
| llama.cpp | b9960 | 0.06 s | 8,749 | 205 tok/s | 326 tok/s |
| oMLX | 0.6.3 | 0.12 s | 4,332 | **255 tok/s** | 324 tok/s |

vLLM Metal 0.1.0 measured 253 tok/s decode / 217 aggregate on 2026-08-29
(with some completions stopped early); its venv was not rebuilt for this
pass, so it stays out of the same-session table. SGLang has no macOS
support.

Aggregate throughput moves ±10% run to run; treat single-digit percentage
gaps in that column as a tie.

## Reading

- **Prefill**: the three single-device numbers sit within ~12% of each other
  — lokal-metal's 7,863 tok/s is the 2026-08-30 flash-attention + Metal 4
  tensor-ops rewrite (from 1,461 the same morning), the same matmul
  mechanism llama.cpp's Metal backend uses, and lokal-ane is the Neural
  Engine doing that work off-GPU. The top row is what neither competitor
  can do: **split prefill** runs the front layers of each chunk on the ANE
  while the GPU works the back layers of the previous chunk, so one prompt
  uses both engines at once — 9,920 tok/s, measured back-to-back against
  llama.cpp's 8,749 in the same session. It is opt-in
  (`LOKAL_SPLIT_PREFILL=1`) because it needs its own exported graph ladder;
  see the ANE setup notes in the root README.
- **Decode**: oMLX leads single-stream (255) and lokal holds 232 on every
  backend — split prefill does not touch decode. llama.cpp's own decode
  number moved 176 → 205 between two runs of the same binary an hour apart,
  which is the honest size of run-to-run spread in this column.
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

One engine, one row — the server is started and stopped for you:

```bash
python3 benchmarks/bench_engines.py --engine lokal-ane
python3 benchmarks/bench_engines.py --engine llamacpp
python3 benchmarks/bench_engines.py --engine omlx --out benchmarks/results.jsonl
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
