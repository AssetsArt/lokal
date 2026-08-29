# Benchmarks

Cross-engine comparison of local inference servers on Apple Silicon, and the
harness that produced it. Numbers here are honest measurements, including the
ones lokal loses — they exist to steer the roadmap, not to advertise.

## Setup

- Hardware: Apple M1 Pro, 16 GB RAM, macOS 26.5
- Model: SmolLM2-135M-Instruct, full precision on every engine
  (lokal: f16 weights on GPU · llama.cpp: GGUF F16 · oMLX / vLLM Metal:
  MLX bf16 from `mlx-community/SmolLM2-135M-Instruct`)
- Workload: ~460-token prompt through each server's chat API, greedy
  (`temperature 0`), `max_tokens 128`
- Method: `bench_engines.py` — median of 5 runs per single-request metric,
  median of 3 runs for the 4-concurrent metric, one warmup request first.
  Every request starts with a unique nonce so server-side prompt caches
  cannot answer from cache (llama.cpp otherwise reuses the prefix KV and
  reports an 8 ms "prefill")
- decode tok/s is marginal: `(tokens_128 − 1) / (wall_128 − wall_1)`, which
  cancels prompt processing out of the number

## Results (2026-08-29)

| engine | version | prefill (~460 tok) | decode | 4x concurrent aggregate |
|---|---|---|---|---|
| lokal `-b metal` | 05dbba1 | 0.57 s | 76 tok/s | 78 tok/s |
| lokal `-b ane` | 05dbba1 | **0.08 s** | 75 tok/s | 115 tok/s |
| llama.cpp | b9960 | **0.06 s** | 215 tok/s | **355 tok/s** |
| oMLX | 0.6.3rc3 | 0.12 s | **262 tok/s** | 332 tok/s |
| vLLM Metal | 0.1.0 | 0.08 s | 253 tok/s ¹ | 217 tok/s ¹ |
| SGLang | — | n/a: no macOS support | | |

¹ vLLM Metal stopped some completions early (123 of 128 single, 369 of 512
concurrent tokens generated); its tok/s are computed from the tokens it
actually produced.

Raw per-run output: [results.jsonl](results.jsonl).

## Reading

- **Prefill**: lokal's ANE path is in the leaders' club (0.08 s), and it is
  the only engine here doing prompt processing off-GPU — under concurrent
  load the GPU keeps decoding while prefill runs elsewhere.
- **Decode**: the mature engines are ~3x faster (215–262 vs 76 tok/s) on a
  short context. That gap is fused/hand-tuned kernels and per-step overhead,
  not memory bandwidth — see DESIGN.md Future work items 1 (attention
  split-position, f16 KV) and the fixed-cost notes in "Where the time goes".
- **Concurrency**: llama.cpp and oMLX batch concurrent decodes into shared
  forward passes (continuous batching); lokal interleaves whole single-token
  steps, so its aggregate is capped by single-stream decode. Same roadmap
  item 3.

## Reproduce

Each engine serves the same model; run the harness against it:

```bash
# lokal (this repo)
cargo run --release -- serve -b ane -m HuggingFaceTB/SmolLM2-135M-Instruct --port 8080
python3 benchmarks/bench_engines.py --api lokal --url http://127.0.0.1:8080 \
    --name "lokal (ane)" --out benchmarks/results.jsonl

# llama.cpp (brew install llama.cpp)
llama-server -hf unsloth/SmolLM2-135M-Instruct-GGUF:F16 --port 8081 -ngl 99 -c 4096 --parallel 4
python3 benchmarks/bench_engines.py --api openai --url http://127.0.0.1:8081/v1 \
    --model smollm2 --name "llama.cpp (metal)" --out benchmarks/results.jsonl

# oMLX (brew tap jundot/omlx https://github.com/jundot/omlx && brew install jundot/omlx/omlx)
# model dir contains mlx-community/SmolLM2-135M-Instruct (a symlink into the HF cache works)
omlx serve --model-dir <models> --port 8082 --api-key bench
python3 benchmarks/bench_engines.py --api openai --url http://127.0.0.1:8082/v1 \
    --model SmolLM2-135M-Instruct --bearer bench --name "oMLX (mlx)" --out benchmarks/results.jsonl

# vLLM Metal (pip install vllm-metal, python 3.12 arm64)
vllm-metal --model mlx-community/SmolLM2-135M-Instruct --port 8083
python3 benchmarks/bench_engines.py --api openai --url http://127.0.0.1:8083/v1 \
    --model mlx-community/SmolLM2-135M-Instruct --name "vllm-metal (mlx)" --out benchmarks/results.jsonl
```
