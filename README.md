# lokal

**Run LLMs on your own machine — fast and simple.**

One small binary. Point it at a Hugging Face model and lokal downloads it,
caches it, and runs it locally — on the CPU, the Metal GPU, or the Apple
Neural Engine, switched with a single flag. A built-in HTTP server is there
when you want to serve instead of chat.

## Quickstart

```bash
./run.sh                              # SmolLM2-135M on the CPU
./run.sh gpu "Once upon a time"       # Metal GPU — ~4x faster decode
./run.sh chat "Why is the sky blue?"  # Q&A with an Instruct model
./run.sh serve                        # HTTP server on port 8080
```

Or drive the binary directly:

```bash
cargo run --release -- -b metal -p "Once upon a time" -n 200
cargo run --release -- -m Qwen/Qwen2.5-0.5B-Instruct --chat -p "hello" -b metal
cargo run --release -- serve -b metal --port 8080
```

Models live in the standard Hugging Face cache (`~/.cache/huggingface/hub`,
same as transformers, candle, LM Studio-style tools) — anything you've already
downloaded on this machine is reused as-is, and anything lokal downloads is
shared back. First run of the default model fetches ~270 MB; after that
everything is offline. Gated models work automatically if you're logged in
with `hf auth login`. No config files, no daemon.

## Performance

Measured 2026-08-30 on an M1 Pro, 16 GB, `-t 0`, on a verified-quiet machine
(SmolLM2-135M-Instruct, ~500-token prompt, unless noted):

| workload | cpu | metal | hybrid |
|---|---|---|---|
| prefill | ~33 tok/s | 9,491 tok/s | **11,986 tok/s** ¹ |
| decode | ~49 tok/s | 231 tok/s | 237 tok/s |
| prefill (Qwen2.5-0.5B) | ~5 tok/s | 3,802 tok/s | 4,660 tok/s ² |
| decode (Qwen2.5-0.5B) | ~27 tok/s | 119 tok/s | 120 tok/s |

¹ the hybrid row implies the split-prefill ladder is exported
(`./run.sh export-ane-split`, once per model) — with it present, `-b hybrid`
splits by default; without it hybrid prefill runs the plain ANE path. Best of 3 passes, as
in [benchmarks/](benchmarks/) — a warm laptop reads 15–20% lower.
² Qwen has the plain and windowed ANE graphs exported but not the split
ladder, so its hybrid row is ANE prefill without the two-device pipeline —
the ladder is per model (`./run.sh export-ane-split Qwen/Qwen2.5-0.5B-Instruct`).
Qwen is 3.7x the parameters of SmolLM2, which is most of the gap between
the two blocks.

Prefill took a 4–6x jump (2026-08-30) from a flash-attention kernel plus
Metal 4 tensor-ops matmuls — the same mechanism llama.cpp's Metal backend
uses — and then went past every engine on this machine by using two of them
at once: **split prefill** puts the front layers of each prompt chunk on the
Neural Engine while the GPU works the back layers of the previous chunk, so
one prompt keeps both busy — 11,986 tok/s against llama.cpp's 9,880 in the
same session, and the lead holds from 500 to 2,000 tokens (the curve is in
[benchmarks/](benchmarks/)). The gains compound: the GPU half is one of the
pipeline's stages, so the day's Metal work — which brought GPU-only prefill
level with llama.cpp, 9,491 vs 9,880 — lifted the hybrid number with it.
And the lead is a curve, not a point: measured across the 500–2,000-token
band, hybrid beats llama.cpp at every length — from +4% at the tightest
points to +20% — with the ladder `export-ane-split` ships; the full table
is in [benchmarks/](benchmarks/).

Decode speed comes from f16 weights resident on the GPU, flash-decoding
attention, and a GQA-aware kernel that
reads each cached KV byte once per q-head group — which is also what keeps
decode from collapsing at long context (Qwen: 119 tok/s at 500 ctx, 53 at 32k).

Serving is where the hybrid pays off most. Concurrent requests decode as one
batch — a single read of the weights serves every active request — while the
Neural Engine prefills newly arrived requests off-GPU:

| 4 concurrent requests (~500-token prompts) | metal | hybrid |
|---|---|---|
| single-request prefill | 0.05 s | **0.04 s** |
| aggregate throughput | 352 tok/s | **373 tok/s** |

How lokal compares against other engines on the same machine and model, with
reproduction steps, lives in [benchmarks/](benchmarks/).

## Supported models

Any Llama-architecture checkpoint with safetensors weights should work; these
are tested:

```
HuggingFaceTB/SmolLM2-135M              # default — small and fast
HuggingFaceTB/SmolLM2-135M-Instruct     # pair with --chat
HuggingFaceTB/SmolLM2-360M-Instruct
Qwen/Qwen2.5-0.5B-Instruct
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Architectures: `LlamaForCausalLM`, `Qwen2ForCausalLM`, `MistralForCausalLM`.

## Backends

Pick with `-b cpu | metal | hybrid`. cpu and metal are verified to produce
identical greedy output; hybrid matches them in practice too, though on very
long prompts fp16 rounding can pick a different — equally sensible — token
at a near-tie.

### Hybrid setup (optional, once per model)

```bash
./run.sh export-ane        # builds the plain prefill graphs (512 and 2048 tokens) — a few minutes
./run.sh export-ane-long   # optional: adds the long-context windowed graph (slow; see note below)
./run.sh export-ane-split  # optional: adds the split-prefill ladder — the fastest prefill (see below)
./run.sh hybrid "Once upon a time"
```

lokal picks the graph that fits the prompt: tiny prompts (< 64 tokens) skip
the ANE — the GPU is faster there — and, when the windowed graph is present,
long prompts run in chunks through it, carrying the accumulated context and
keeping everything up to 8k tokens on the ANE (beyond that, Metal takes the
tail seamlessly). That is what makes long prompts fast: a 6,086-token prompt
prefills in 3.3 s vs 14.1 s on Metal alone. Without the windowed graph the
ANE still serves the first 2,048 tokens and Metal the rest.

With the split ladder exported, `-b hybrid` runs prefill as a two-device
pipeline by default — the front layers of each prompt chunk on the Neural
Engine while the GPU works the back layers of the previous one; that is the
11,986 tok/s row above, and no flag is needed. `LOKAL_SPLIT_PREFILL=0`
disables it for A/B runs. The ladder costs ~150 MB of disk per front graph
(each rung carries its own weight copy), and a prompt the ladder cannot
serve falls back to the plain path automatically.

The windowed graph is the expensive one — minutes to export, and the first
load on each machine spends a few more minutes in Apple's ANE compiler
(one-time; cached after that, with a notice printed while it runs).

The export step needs [uv](https://docs.astral.sh/uv/) and runs offline.
Placement is verified, not assumed: inspecting the compiled graph with the
MLComputePlan API shows 1,733 ops on the NeuralEngine device and 6 on the CPU,
and you can watch it live with `sudo powermetrics --samplers ane_power`.

## HTTP server

```bash
lokal serve -b metal --port 8080
curl http://127.0.0.1:8080/generate -d '{"prompt": "Once upon a time", "max_tokens": 100}'
```

Single endpoint, JSON in and out: `prompt` (required), `max_tokens`,
`temperature`, `top_p`, `seed`, `chat`. The reply includes token counts and
tokens/sec. On the GPU backends, up to `--max-concurrent` requests (default 4)
decode as one continuous batch and the rest queue FIFO; outputs are identical
to running each request alone.

## CLI reference

```
lokal [options]           one-shot generation
lokal serve [options]     HTTP server (POST /generate)
lokal path [-m <model>]   download if needed, then print the model's local directory

-m, --model <repo|dir>   Hugging Face repo or local directory
    --draft <repo|dir>   smaller same-tokenizer model: speculative decoding (greedy)
-b, --backend <name>     cpu | metal | hybrid                      [cpu]
-p, --prompt <text>      prompt text
-n, --max-tokens <N>     generation budget                      [200]
-t, --temperature <T>    0 = greedy/deterministic               [0.7]
    --top-p <P>          nucleus sampling threshold             [0.9]
    --seed <N>           reproducible sampling
    --chat               ChatML template for -Instruct models
    --port <N>           serve-mode port                        [8080]
    --max-concurrent <N> serve mode: parallel generations       [4]
```

## Development

```bash
cargo build --release
cargo test
```

The main correctness gate is deterministic cross-backend comparison: with
`--temperature 0`, the same prompt must produce token-identical output on
cpu and metal; the fp16 hybrid backend is held to a numeric envelope vs the f32
reference and may differ at rare greedy near-ties on long prompts.
Architecture notes, backend internals, and the guide for adding new backends
live in [DESIGN.md](DESIGN.md).

## Roadmap

- [x] Fast decode on Metal — flash-decoding attention, fused kernels, vectorized loads
- [x] Speculative decoding (`--draft`) — exact greedy verification, adaptive block size
- [ ] f16 model loading — halve load RAM so 3B+ targets fit (what makes `--draft` pay off)
- [ ] Lossless speculative sampling (temperature > 0)
- [x] f16 KV cache — half the memory per session, flatter long-context decode
- [ ] OpenAI-compatible API in serve mode (`/v1/chat/completions`, works with existing clients)
- [ ] Streaming responses (SSE)
- [ ] Quantized weights (int8/int4) for larger models on modest RAM
- [x] Continuous batching — one weight read serves every active request
- [x] `simdgroup_matrix` (MMA) matmul on Metal
- [x] Flash-attention prefill + Metal 4 tensor-ops matmul — 3.9x prefill at ~500 tok, 6x at 2k
- [ ] CUDA (NVIDIA) and Vulkan (AMD/portable) backends
- [x] Multi-size ANE prefill graphs (512/2048) with automatic routing per prompt length
- [x] Windowed ANE prefill — long prompts chunk through the ANE with fed-back KV (8k coverage)
- [x] MLState decode spike — measured no-go on this toolchain (three hard ceilings; see DESIGN.md)
- [ ] Hybrid scheduler — route each phase to the best device automatically, no `--backend` flag needed
- [ ] Repetition penalty and top-k sampling

## License

MIT — see [LICENSE](LICENSE).
