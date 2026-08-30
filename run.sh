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
    echo "note: eight front-half graphs, ~15-20 s to export and ~150 MB on disk each" >&2
    echo "      (every rung carries its own weight copy), plus one ANE compile per" >&2
    echo "      rung on this machine's first load. Enable with LOKAL_SPLIT_PREFILL=1." >&2
    # The ladder measured on 2026-08-30: stride 128 with 20 front layers for short
    # prompts, stride 256 with 15 for longer ones — the split point moves with the
    # chunk width because it is the GPU half that runs out of work first.
    exec uv run --python 3.12 --with torch --with coremltools --with safetensors --with numpy --with tokenizers \
      tools/export_prefill.py "$DIR" --shapes none --window none \
      --front 128x128x20,128x384x20,128x896x20,128x1920x20,256x256x15,256x768x15,256x1280x15,256x2048x15
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
    echo "  export-ane-split add the split-prefill ladder, for LOKAL_SPLIT_PREFILL=1 (slow)"
    echo "  chat <question> Q&A against an Instruct model"
    echo "  serve [port]    start the HTTP server (default 8080)"
    echo "  test            run unit tests"
    echo "  <anything else> passed straight to the binary, e.g.  ./run.sh -m Qwen/Qwen2.5-0.5B-Instruct --chat -p \"hi\""
    ;;
  *) # unknown command → treat it as raw binary arguments and pass everything through
    exec "$BIN" "$cmd" "$@"
    ;;
esac
