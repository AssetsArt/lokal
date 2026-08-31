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
./run.sh export-hybrid                # once per model — Core ML graphs, ~900 MB, minutes
./run.sh hybrid "Once upon a time"    # Neural Engine + GPU together (needs export-hybrid above)
./run.sh chat "Why is the sky blue?"  # Q&A with an Instruct model
./run.sh serve                        # HTTP server on port 8080
```

The hybrid graphs are built once and land in `~/.cache/lokal/coreml/<model>/`;
every later run reuses them, and `export-hybrid` skips whatever is already there.

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
| prefill | ~33 tok/s | 9,900 tok/s | **11,986 tok/s** ¹ |
| decode | ~49 tok/s | 231 tok/s | 237 tok/s |
| prefill (Qwen2.5-0.5B) | ~5 tok/s | 3,802 tok/s | 4,660 tok/s ² |
| prefill @~7.7k (Qwen2.5-0.5B) | — | 1,831 tok/s | **2,836 tok/s** ² ³ |
| decode (Qwen2.5-0.5B) | ~27 tok/s | 119 tok/s | 120 tok/s |

¹ the hybrid row implies the split-prefill ladder is exported
(`./run.sh export-hybrid`, once per model) — with it present, `-b hybrid`
splits by default; without it hybrid prefill runs the plain ANE path. Best of 3 passes, as
in [benchmarks/](benchmarks/) — a warm laptop reads 15–20% lower.
³ the Qwen rows were measured just before the RoPE-pair fusion and q-bias
fold landed, which lift Qwen prefill ~8–9% at short lengths — they are a
floor, not a ceiling.
² the ladder is per model (`./run.sh export-hybrid
Qwen/Qwen2.5-0.5B-Instruct` — a different stride and split point than
SmolLM2's, both measured). With it, Qwen's hybrid prefill beats its Metal
path at every length — +34% at 2k, +48% at 4k, +55% at 7.7k (same-session
HTTP pairs; the 7.7k row above) — where the old fixed-width windowed
routing was 2x *slower* than Metal there. Qwen is 3.7x the parameters of
SmolLM2, which is most of the gap between the two blocks.

Prefill took a 4–6x jump (2026-08-30) from a flash-attention kernel plus
Metal 4 tensor-ops matmuls — the same mechanism llama.cpp's Metal backend
uses — and then went past every engine on this machine by using two of them
at once: **split prefill** puts the front layers of each prompt chunk on the
Neural Engine while the GPU works the back layers of the previous chunk, so
one prompt keeps both busy — 11,986 tok/s against llama.cpp's 9,880 in the
same session, and the lead holds from 500 to 2,000 tokens (the curve is in
[benchmarks/](benchmarks/)). The gains compound: the GPU half is one of the
pipeline's stages, so the day's Metal work — which brought GPU-only prefill
level with llama.cpp, 9,900 vs 9,880 — lifted the hybrid number with it.
And the lead is a curve, not a point: measured across the 500–2,000-token
band, hybrid beats llama.cpp at every length — from +4% at the tightest
points to +20% — with the ladder `export-hybrid` ships; the full table
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

### GGUF checkpoints

lokal also runs the ecosystem's pre-quantized GGUF files directly — point `-m`
at a `.gguf` path, or at a single file inside a Hub repo:

```bash
lokal -b metal -m ~/models/qwen2.5-0.5b-instruct-q8_0.gguf -p "hello"
lokal -b metal -m Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf -p "hello"
```

You can also name a quant by tag — `owner/repo:Q4_K_M` picks the right file
out of a GGUF repo (ambiguity and misses list what IS there), and a bare
`-GGUF` repo defaults to Q4_K_M.

Config and tokenizer come out of the file itself (nothing else to download);
tensor types F32, F16, Q8_0, Q4_0, Q5_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and the
i-quant family (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS,
IQ4_NL) are supported, which covers the mixed "UD" low-bit files whose tensors
span a dozen types at once. Anything else is refused by name — in ONE pass,
listing every unsupported type in the file with counts, so a mixed checkpoint
does not teach you its requirements one re-download at a time. On `-b metal` the weights STAY in their
quantized encoding on the GPU and dequantize on read — the memory cost is the
file size, which is what lets a ~16 GB 27B Q4_K_M run resident with full
attention on a 32 GB machine (`benchmarks/collect-metal-quant.sh` is the
one-command acceptance for exactly that). On `-b cpu` the weights are
dequantized to f32 up front, so the honest memory cost there is the EXPANDED
size — a 4 GB Q4 file is ~28 GB of f32, refused with both numbers named.
`-b lowmem` keeps quantized weights paged under a fixed budget with windowed
attention.

**On the 1- and 2-bit types: they work, and they cost quality.** IQ1-class
output is heavily degraded — that is the nature of ~1.5 bits per weight, not a
defect in this implementation, and no amount of correct dequantization buys it
back. They are here because they make otherwise-impossible models fit, and
because fewer bytes per weight means fewer bytes moved per token. If you are
choosing a quantization rather than living with one you already have,
**Q4_K_M is the floor worth defaulting to**; drop below it when the model does
not otherwise fit, not to go faster. Every type is verified bit-for-bit against
ggml's reference dequantization, so what you get is exactly what llama.cpp
would compute from the same file — including its quality loss.

The hybrid backend cannot run GGUF (its ANE graphs are exported
from safetensors). GGUF architectures: `llama`, `qwen2`, and `qwen3`
(per-head q/k norm runs on cpu, metal, and lowmem — hybrid's ANE graphs
cannot compute it), byte-level BPE tokenizers.

## Backends

Pick with `-b cpu | metal | hybrid | lowmem`. cpu and metal are verified to
produce identical greedy output; hybrid matches them in practice too, though on
very long prompts fp16 rounding can pick a different — equally sensible — token
at a near-tie.

`-b lowmem` is disk-backed paged inference, optimized for models **larger than
available RAM**: weights stay mmapped on disk and stream through a fixed pool
page by page, the KV cache is a bounded sliding window (`--context-window`,
default 2048, plus a few pinned "attention sink" tokens), and `--memory-budget`
(default 4096 MB) caps the whole working set — the budget split prints at load.
Prefill cost stays flat out to 32k instead of growing quadratically. The trade
is honest: a model that fits its budget runs near metal speed; one that doesn't
is disk-bound (a 29 GB model on a 16 GB machine decodes at SSD speed — fractions
of a token per second, with ~100 tok/s prefill), and content older than the
window is genuinely forgotten. See DESIGN.md for the physics.

### Hybrid setup (optional, once per model)

```bash
./run.sh export-hybrid     # plain prefill graphs + the split-prefill ladder — minutes, ~900 MB
./run.sh hybrid "Once upon a time"
```

The graphs land in a lokal-owned directory — `~/.cache/lokal/coreml/<model>/`
(`XDG_CACHE_HOME` respected, `LOKAL_GRAPH_DIR` relocates it wholesale) — not in
the Hugging Face cache, so `hf cache delete` and upstream revision bumps leave
them alone. Graphs from older lokal versions that still sit next to the weights
are moved over automatically on the next hybrid run. Each directory carries a
`graphs.json` naming the model and snapshot revision it was built from: when
the model moves on, lokal refuses the stale graphs (and says so) rather than
run old weights; re-running `export-hybrid` rebuilds only what is missing or
stale — `-f` forces a full rebuild.

lokal routes each prompt by what is measured fastest today: tiny prompts
(< 64 tokens) skip the ANE — the GPU wins there — short prompts run through
a padded plain graph while that still beats Metal on the same rows, and
everything longer goes through the split-prefill ladder when it is
exported, or straight to Metal when it is not. A wrong choice is never
kept: the hybrid backend is measured to be at least as fast as `-b metal`
at every length, on both test models.

With the split ladder exported, `-b hybrid` runs prefill as a two-device
pipeline by default — the front layers of each prompt chunk on the Neural
Engine while the GPU works the back layers of the previous one; that is the
11,986 tok/s row above, and no flag is needed. `LOKAL_SPLIT_PREFILL=0`
disables it for A/B runs. The ladder costs ~150 MB of disk per front graph
for SmolLM2 and ~500 MB for Qwen (each rung carries its own weight copy,
including the model's full embedding table), and a prompt the ladder cannot
serve falls back to the plain path automatically. On a 16 GB machine, note
that a hybrid process peaks around 4 GB on Qwen — running it beside other
resident inference servers is what tips the OS into killing one of them.

The old fixed-width windowed graph (built only by the legacy
`./run.sh export-ane-long` alias) is retired
from the default routing: it charges its full 8,192-position attention on
every chunk, which was the right call when Metal prefill ran at 1,461 tok/s
and is ~2x slower than Metal today. If it is on disk it stays idle (a
notice says so); `LOKAL_WINDOWED_PREFILL=1` loads it for A/B comparison.

### Sliding-window mode (optional)

`--context-window W [--attention-sink N]` opts `-b metal` and `-b hybrid` into
the flat-cost attention the lowmem backend runs by construction: each query
attends its last W positions plus N pinned "sink" tokens, and KV memory becomes
a fixed ring — O(window), flat in context length (Qwen 0.5B at 32k context:
~400 MB of full KV vs ~35 MB of ring at W=2048). The trade is real and the
mode is DEFAULT OFF: these models were trained full-causal, and each layer only
attends its own window — though beyond-window context still influences output
through depth (each position summarizes ITS window at the layer below, so the
effective receptive field is roughly layers × window, the same mechanism that
lets Mistral's sliding window carry long documents). Without the flags,
behavior is bit-for-bit unchanged. On `-b hybrid`, the ANE's graphs compute full causal
attention, which below position W is EXACTLY the window+sink result — so the
ANE serves those positions and windowed Metal takes everything past them, no
graph re-export, no approximation. Speed-wise the mode is a
long-context play, not a general win: at W=2048 on Qwen 0.5B the windowed
prefill curve is flat (2,669 / 2,495 / 2,716 tok/s at 2k/4k/8k) where full
attention starts higher and falls (3,164 / 3,149 / 2,402) — the lines cross
around 8k, and past it the gap only widens while full-causal KV keeps
growing. In serve mode a window means per-request sessions (the batcher pool
keeps the full-causal layout), `--draft` is refused, and RoPE positions past
the trained length degrade quality — same caveat as lowmem.

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
-b, --backend <name>     cpu | metal | hybrid | lowmem             [cpu]
-p, --prompt <text>      prompt text
-n, --max-tokens <N>     generation budget                      [200]
    --memory-budget <MB> lowmem: working-set budget            [4096]
    --context-window <N> lowmem: attention window, tokens      [2048]
    --attention-sink <N> lowmem: pinned initial tokens, 0=off     [4]
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
