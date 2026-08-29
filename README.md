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
| decode | ~49 tok/s | ~160–190 tok/s | = metal |
| decode (Qwen2.5-0.5B) | ~27 tok/s | ~106 tok/s | — |
| prefill (669-token prompt) | ~33 tok/s | ~740 tok/s | **~1,780 tok/s** |

Decode speed comes from keeping f16 weights resident on the GPU. Prefill speed
comes from batching the prompt into tiled matrix-matrix work — and the `ane`
backend pushes prompt processing onto the Neural Engine, which is faster
still, draws far less power, and leaves the GPU free to decode for other
requests in serve mode.

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
./run.sh export-ane        # builds prefill-512.mlmodelc next to the cached model
./run.sh ane "Once upon a time"
```

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
`temperature`, `top_p`, `seed`, `chat`. Requests run concurrently; the reply
includes token counts and tokens/sec.

## CLI reference

```
lokal [options]           one-shot generation
lokal serve [options]     HTTP server (POST /generate)
lokal path [-m <model>]   download if needed, then print the model's local directory

-m, --model <repo|dir>   Hugging Face repo or local directory
-b, --backend <name>     cpu | metal | ane                      [cpu]
-p, --prompt <text>      prompt text
-n, --max-tokens <N>     generation budget                      [200]
-t, --temperature <T>    0 = greedy/deterministic               [0.7]
    --top-p <P>          nucleus sampling threshold             [0.9]
    --seed <N>           reproducible sampling
    --chat               ChatML template for -Instruct models
    --port <N>           serve-mode port                        [8080]
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

- [ ] OpenAI-compatible API in serve mode (`/v1/chat/completions`, works with existing clients)
- [ ] Streaming responses (SSE)
- [ ] Quantized weights (int8/int4) for larger models on modest RAM
- [ ] `simdgroup_matrix` (MMA) matmul on Metal
- [ ] CUDA (NVIDIA) and Vulkan (AMD/portable) backends
- [ ] Enumerated-shape ANE graphs so short prompts skip the 512 padding
- [ ] Repetition penalty and top-k sampling

## License

MIT — see [LICENSE](LICENSE).
