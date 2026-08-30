#!/usr/bin/env bash
# Convenience launcher for lokal — list commands with: ./run.sh help
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release --quiet
BIN=target/release/lokal

cmd="${1:-demo}"
shift || true

case "$cmd" in
  demo) # ./run.sh  or  ./run.sh demo "prompt text"
    exec "$BIN" -p "${1:-Once upon a time}"
    ;;
  gpu) # ./run.sh gpu "prompt text"  — same as demo but on the Metal GPU
    exec "$BIN" -b metal -p "${1:-Once upon a time}"
    ;;
  hybrid | ane) # ./run.sh hybrid "prompt text"  — Neural Engine + GPU together
    exec "$BIN" -b hybrid -p "${1:-Once upon a time}"
    ;;
  export-ane) # ./run.sh export-ane [repo or dir]  — build the plain prefill graphs (fast, once per model)
    MODEL="${1:-HuggingFaceTB/SmolLM2-135M}"
    if [ -d "$MODEL" ]; then DIR="$MODEL"; else DIR="$("$BIN" path -m "$MODEL")"; fi
    exec uv run --python 3.12 --with torch --with coremltools --with safetensors --with numpy --with tokenizers \
      tools/export_prefill.py "$DIR" --shapes 512,2048 --window none
    ;;
  export-ane-split) # ./run.sh export-ane-split [repo or dir]  — add the split-prefill ladder (slow)
    MODEL="${1:-HuggingFaceTB/SmolLM2-135M}"
    if [ -d "$MODEL" ]; then DIR="$MODEL"; else DIR="$("$BIN" path -m "$MODEL")"; fi
    # The ladder is per model, and so is its split point: where the GPU half
    # binds, the ANE should carry more layers, and that balance moves with the
    # chunk width AND the model shape. Every spec below is exactly what its
    # model's published numbers were measured with (2026-08-30); an A sweep on
    # each side of every value measured slower. Do not add rungs casually: the
    # widest rung of a stride sets how far the ANE reaches (ane_total = s +
    # p_max in ane.rs), so a new one silently changes which part of a prompt
    # runs where. P=0 rungs let the first chunk skip the phantom past attention.
    case "$MODEL" in
      *[Qq]wen*)
        # 24 layers x 896 hidden: stride 512, 9 front layers (8 and 10 both
        # measured slower at 7.7k). ~500 MB per rung — the 272 MB embedding
        # table rides along in every one.
        SPEC=512x0x9,512x2048x9,512x4096x9,512x7168x9
        N=four ;;
      *)
        # SmolLM2 (30 layers x 576 hidden) and the default for untested models:
        # stride 128 @ 20 front layers short, stride 256 @ 13 beyond, long-band
        # rungs so an 8k prompt still pipelines instead of draining to Metal.
        SPEC=128x128x20,128x384x20,256x0x13,256x256x13,256x512x13,256x768x13,256x1280x13,256x2048x13,256x3072x13,256x4608x13,256x7168x13
        N=eleven ;;
    esac
    echo "note: $N front-half graphs (every rung carries its own weight copy), plus one" >&2
    echo "      ANE compile per rung on this machine's first load. Once exported," >&2
    echo "      -b hybrid uses the ladder automatically." >&2
    exec uv run --python 3.12 --with torch --with coremltools --with safetensors --with numpy --with tokenizers \
      tools/export_prefill.py "$DIR" --shapes none --window none --front "$SPEC"
    ;;
  export-ane-long) # ./run.sh export-ane-long [repo or dir]  — add the long-context windowed graph (slow)
    MODEL="${1:-HuggingFaceTB/SmolLM2-135M}"
    if [ -d "$MODEL" ]; then DIR="$MODEL"; else DIR="$("$BIN" path -m "$MODEL")"; fi
    echo "note: the windowed graph takes minutes to convert, and its first load on each" >&2
    echo "      machine spends ~4 min more in the ANE compiler (one-time, then cached)." >&2
    exec uv run --python 3.12 --with torch --with coremltools --with safetensors --with numpy --with tokenizers \
      tools/export_prefill.py "$DIR" --shapes none --window 1024x7168
    ;;
  chat) # ./run.sh chat "question"  — Q&A against an Instruct model with the chat template
    exec "$BIN" -m HuggingFaceTB/SmolLM2-135M-Instruct --chat \
      -p "${1:?usage:  ./run.sh chat \"Why is the sky blue?\"}"
    ;;
  serve) # ./run.sh serve [port]  — start the HTTP server
    exec "$BIN" serve --port "${1:-8080}"
    ;;
  test) # ./run.sh test  — unit tests (offline)
    exec cargo test --release
    ;;
  help | -h | --help)
    echo "usage: ./run.sh [command]"
    echo "  demo [text]     generate a continuation of the prompt (default)"
    echo "  gpu [text]      same as demo, on the Metal GPU (~4x cpu)"
    echo "  hybrid [text]   Neural Engine + GPU together, decode on Metal (needs export-ane once)"
    echo "  export-ane      build the Core ML prefill graphs (once per model)"
    echo "  export-ane-long add the long-context windowed graph (slow)"
    echo "  export-ane-split add the split-prefill ladder that -b hybrid runs on (slow)"
    echo "  chat <question> Q&A against an Instruct model"
    echo "  serve [port]    start the HTTP server (default 8080)"
    echo "  test            run unit tests"
    echo "  <anything else> passed straight to the binary, e.g.  ./run.sh -m Qwen/Qwen2.5-0.5B-Instruct --chat -p \"hi\""
    ;;
  *) # unknown command → treat it as raw binary arguments and pass everything through
    exec "$BIN" "$cmd" "$@"
    ;;
esac
