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

Measured on an M1 Pro with `-t 0` (SmolLM2-135M unless noted):

| workload | cpu | metal | ane (hybrid) |
|---|---|---|---|
| decode | ~49 tok/s | ~230–270 tok/s | = metal |
| decode (Qwen2.5-0.5B) | ~27 tok/s | ~130 tok/s | — |
| prefill (676-token prompt) | ~33 tok/s | ~890 tok/s | **~1,900 tok/s** |

Decode speed comes from keeping f16 weights resident on the GPU. Prefill speed
comes from batching the prompt into tiled matrix-matrix work — and the `ane`
backend pushes prompt processing onto the Neural Engine, which is faster
still, draws far less power, and leaves the GPU free to decode for other
requests in serve mode.

What that means end to end (676-token prompt, 200 tokens generated):

| backend | first token after | decode | total |
|---|---|---|---|
| cpu | 18.4 s | ~27 tok/s | ~26 s |
| metal | 0.76 s | ~213 tok/s | ~1.7 s |
| ane | **0.35 s** | ~232 tok/s | **~1.2 s** |

The `ane` backend doesn't change decode speed — it cuts time-to-first-token,
which is the number you actually feel in a chat.

Serving is where the hybrid pays off most. Concurrent requests decode as one
batch — a single read of the weights serves every active request — while the
Neural Engine prefills newly arrived requests off-GPU:

| 4 concurrent requests (451-token prompts) | metal | ane (hybrid) |
|---|---|---|
| single-request prefill | 0.49 s | **0.06 s** |
| aggregate throughput | 168 tok/s | **365 tok/s** |

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

Pick with `-b cpu | metal | ane`. All three are verified to produce identical
greedy output, so switching backends never changes what the model says — only
how fast it says it.

### ANE setup (optional, once per model)

```bash
./run.sh export-ane        # builds the plain prefill graphs (512 and 2048 tokens) — a few minutes
./run.sh export-ane-long   # optional: adds the long-context windowed graph (slow; see note below)
./run.sh ane "Once upon a time"
```

lokal picks the graph that fits the prompt: tiny prompts (< 64 tokens) skip
the ANE — the GPU is faster there — and, when the windowed graph is present,
long prompts run in chunks through it, carrying the accumulated context and
keeping everything up to 8k tokens on the ANE (beyond that, Metal takes the
tail seamlessly). That is what makes long prompts fast: a 6,086-token prompt
prefills in 3.3 s vs 14.1 s on Metal alone. Without the windowed graph the
ANE still serves the first 2,048 tokens and Metal the rest.

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
-b, --backend <name>     cpu | metal | ane                      [cpu]
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
cpu, metal, and ane. Architecture notes, backend internals, and the guide for
adding new backends live in [DESIGN.md](DESIGN.md).

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
- [ ] CUDA (NVIDIA) and Vulkan (AMD/portable) backends
- [x] Multi-size ANE prefill graphs (512/2048) with automatic routing per prompt length
- [x] Windowed ANE prefill — long prompts chunk through the ANE with fed-back KV (~6k coverage)
- [ ] ANE decode via Core ML stateful models (MLState)
- [ ] Hybrid scheduler — route each phase to the best device automatically, no `--backend` flag needed
- [ ] Repetition penalty and top-k sampling

## License

MIT — see [LICENSE](LICENSE).
