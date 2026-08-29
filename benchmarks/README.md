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
| lokal `-b metal` | main | 0.49 s | 240 tok/s | 168 tok/s |
| lokal `-b ane` | main | **0.06 s** | 232 tok/s | **365 tok/s** |
| llama.cpp | b9960 | **0.06 s** | 215 tok/s | 355 tok/s |
| oMLX | 0.6.3rc3 | 0.12 s | **262 tok/s** | 332 tok/s |
| vLLM Metal | 0.1.0 | 0.08 s | 253 tok/s ¹ | 217 tok/s ¹ |
| SGLang | — | n/a: no macOS support | | |

Before the decode-optimization pass (commit 05dbba1: pre flash-decoding
attention, unfused kernels, scalar loads) lokal decoded at 76 tok/s and
aggregated 78 (metal) / 115 (ane) — those runs are kept in results.jsonl.

¹ vLLM Metal stopped some completions early (123 of 128 single, 369 of 512
concurrent tokens generated); its tok/s are computed from the tokens it
actually produced.

Raw per-run output: [results.jsonl](results.jsonl).

## Reading

- **Prefill**: lokal's ANE path ties llama.cpp for the lead (0.06 s), and it
  is the only engine here doing prompt processing off-GPU — under concurrent
  load the GPU keeps decoding while prefill runs elsewhere.
- **Decode**: after the flash-decoding + kernel-fusion pass, lokal sits with
  the leaders (241 vs llama.cpp's 215 and oMLX's 262) — the pass is
  described in DESIGN.md, Metal backend decision 4.
- **Concurrency**: with continuous batching (one weight read serves every
  active request) plus ANE prefill off-GPU, lokal's hybrid leads the field
  (365 vs llama.cpp's 355 and oMLX's 332). The metal-only number is
  admission-bound: each join prefills ~0.5 s on the same GPU — chunked
  prefill scheduling is the known fix.

## Long-context baseline (2026-08-29)

The table above is a ~460-token prompt, which is not the length anyone
actually runs at. This section is the same comparison at 2k-32k on
Qwen2.5-0.5B-Instruct (32k context window) instead of SmolLM2, because the
question that steers the roadmap is what prefill and decode cost at real input
sizes.

- Prompt: natural prose sliced out of Project Gutenberg ebook 2701 (*Moby
  Dick*, public domain), pinned by sha256 and fetched on first use into
  `benchmarks/.cache/` rather than committed. Every engine gets the same text
  and is asked for its own token count, so tok/s is normalized by what that
  engine tokenized instead of by a shared guess. The unique-nonce prefix from
  the short-prompt method is kept, so no server answers from a prefix cache.
- Method: greedy, `max_tokens 128`, median of 3 runs per cell.
- Machine: one M1 Pro shared by three agents working in parallel, so every row
  records the machine state it was measured under (`machine_before` /
  `machine_after` in results.jsonl) and the harness refuses to write a row
  taken while something else is busy. An earlier pass of this table was thrown
  away for exactly that reason: contention cost it 8% of its prefill and 12-17%
  of its decode, which is small enough to read as noise and large enough to
  reverse a ranking. Those rows are kept in results.jsonl tagged
  `longctx-baseline-CONTAMINATED`.

**Prefill (wall / tokens per second)**

| prompt | lokal `-b ane` (cli) |
|---|---|
| 2k | 1.79 s / 1155 tok/s |
| 6k | 4.35 s / 1393 tok/s |
| 10k | 34.72 s / 294 tok/s |
| 16k | 102.80 s / 158 tok/s |

**Decode (marginal tokens per second)**

| prompt | lokal `-b ane` (cli) |
|---|---|
| 2k | 103 tok/s |
| 6k | 78 tok/s |
| 10k | 63 tok/s |
| 16k | 49 tok/s |

**Prompt length as each engine tokenized it**

| prompt | lokal `-b ane` (cli) |
|---|---|
| 2k | 2067 |
| 6k | 6058 |
| 10k | 10204 |
| 16k | 16272 |

### Where lokal's prefill time goes

| prompt | lokal `-b ane` (cli) |
|---|---|
| 2k | ANE 2066 tok / 1.67 s + Metal 1 tok / 0.12 s |
| 6k | ANE 6057 tok / 4.22 s + Metal 1 tok / 0.13 s |
| 10k | ANE 6144 tok / 4.21 s + Metal 4060 tok / 30.51 s |
| 16k | ANE 6144 tok / 4.21 s + Metal 10128 tok / 98.59 s |

The ANE is a flat cost and the Metal tail is a compounding one. lokal's only
windowed graph for this model is `prefill-1024w5120`, so `prefill_chunked` in
ane.rs covers `s + p` = 6144 positions and Metal prefills every token past
that. The ANE spends 4.21 s on those 6144 positions at a 10k prompt and the
identical 4.21 s at 16k — same window, same work. The tail does not merely
grow, it gets slower per token as it grows (133 tok/s over 4060 tokens, 103
tok/s over 10128), because each tail token attends to everything before it.
That is the whole story of the prefill column: 1393 tok/s at 6k, where the
prompt still fits the window, against 158 tok/s at 16k, where two thirds of it
does not.

### The 8192-token serve gap

lokal's HTTP server cannot be measured past 8192 tokens at all. Continuous
batching pools each slot at `POOL_SEQ_CAP` (batch.rs), and a longer prompt is
refused outright:

```
$ curl -s localhost:8080/generate -d '{"prompt": "<10k tokens>", ...}'
{"error":"prompt (10182 tokens) is empty or exceeds the 8192-token budget"}
```

So every lokal row above 8k here is measured through the CLI path
(`bench_engines.py --api cli`), which parses the prefill/decode split the
binary already prints. Raising that cap with chunked-prefill admission is a
known phase-2 item; this table documents the gap rather than benchmarking
around it.

### Reproducing the long-context table

`run_longctx.py` owns the server lifecycle — 16 GB does not hold three
inference servers and a 32k KV cache at once, and an idle engine still skews
the one being timed, so it starts one, sweeps every prompt size against it,
kills its process group, and moves on.

```bash
# the cross-engine sweep (lokal goes through its cli above 8k, see above)
python3 benchmarks/run_longctx.py --engines lokal-ane-cli,llamacpp,omlx \
    --sizes 2000,6000,10000,16000,24000,32000 --ctx 34816 --runs 3

# 4-way concurrency at 10k; llama.cpp splits one KV allocation across slots
python3 benchmarks/run_longctx.py --engines llamacpp,omlx --sizes 10000 \
    --ctx 45056 --parallel 4 --concurrency 4 --tag longctx-conc

# then rebuild the tables above
python3 benchmarks/summarize_longctx.py --tag longctx-baseline
```

The engines need their own weights: llama.cpp wants
`Qwen/Qwen2.5-0.5B-Instruct-GGUF:fp16` and oMLX wants
`mlx-community/Qwen2.5-0.5B-Instruct-bf16` unpacked under `~/.omlx/models/`.
lokal's ANE backend needs the exported Core ML graphs beside the model in the
HF cache (`prefill-512`, `prefill-2048`, `prefill-1024w5120`).

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
