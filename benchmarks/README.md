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
  All five configurations below were measured back-to-back in one session.

## Results (2026-08-30)

| engine | version | prefill (~500 tok) | prefill tok/s | decode | 4x concurrent aggregate |
|---|---|---|---|---|---|
| lokal `-b hybrid` + split | main | **0.06 s** | 8,784 | 226 tok/s | **350 tok/s** |
| lokal `-b hybrid` | main | 0.06 s | 8,466 | 218 tok/s | 339 tok/s |
| lokal `-b metal` | main | 0.07 s | 7,604 | 205 tok/s | 321 tok/s |
| llama.cpp | b9960 | 0.06 s | **8,822** | 180 tok/s | 323 tok/s |
| oMLX | 0.6.3 | 0.12 s | 4,354 | **255 tok/s** | 326 tok/s |

**Run-to-run spread is large enough to change the prefill ranking**, so it
gets stated rather than hidden. A repeat pass minutes later, same binaries
and same harness: split prefill 10,405 tok/s, llama.cpp 8,984. Across three
sessions split measured 8,784 / 9,920 / 10,405 and llama.cpp 8,749 / 8,822 /
8,984 — llama.cpp is the steadier of the two, split is faster in most runs
and ties in its worst one. Read the prefill column as *lokal at parity with
llama.cpp, leading when the machine cooperates*, not as a settled win.
Decode drifted too (lokal 194–232, llama.cpp 171–205 across the same runs);
oMLX's decode lead is the one single-stream result that held in every pass.

vLLM Metal 0.1.0 measured 253 tok/s decode / 217 aggregate on 2026-08-29
(with some completions stopped early); its venv was not rebuilt for this
pass, so it stays out of the same-session table. SGLang has no macOS
support.

## Reading

- **Prefill**: all three lokal configurations and llama.cpp land in the same
  band; oMLX is half their rate. lokal-metal's 7,604 tok/s is the
  2026-08-30 flash-attention + Metal 4 tensor-ops rewrite (from 1,461 that
  morning) — the same matmul mechanism llama.cpp's Metal backend uses.
  `-b hybrid` moves prompt processing to the Neural Engine, and **split
  prefill** goes further: the front layers of each chunk run on the ANE
  while the GPU works the back layers of the previous chunk, so one prompt
  keeps both engines busy. That is the one thing no GPU-only engine here
  can copy, and it is why the top row exists. It is opt-in
  (`LOKAL_SPLIT_PREFILL=1`, graphs from `./run.sh export-ane-split`) because
  the ladder of front graphs costs ~150 MB of disk each.
- **Decode**: oMLX leads single-stream in every pass (255). lokal sits
  behind it and ahead of llama.cpp; split prefill does not touch the decode
  path, so its decode differences from plain `-b hybrid` are noise.
- **Concurrency**: continuous batching — one weight read serving every
  active request — puts lokal at the top of the aggregate column in every
  configuration (321–350 vs 323 and 326), and the hybrid's off-GPU prefill
  adds to it: newly arrived requests do not stall the batch that is already
  decoding.

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

# the split-prefill row (needs ./run.sh export-ane-split once per model):
LOKAL_SPLIT_PREFILL=1 python3 benchmarks/bench_engines.py --engine lokal-ane
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
