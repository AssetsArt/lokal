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
  ane) # ./run.sh ane "prompt text"  — prefill on the Neural Engine + decode on Metal
    exec "$BIN" -b ane -p "${1:-Once upon a time}"
    ;;
  export-ane) # ./run.sh export-ane [repo or dir]  — build the prefill graphs (once per model)
    MODEL="${1:-HuggingFaceTB/SmolLM2-135M}"
    if [ -d "$MODEL" ]; then DIR="$MODEL"; else DIR="$("$BIN" path -m "$MODEL")"; fi
    exec uv run --python 3.12 --with torch --with coremltools --with safetensors --with numpy \
      tools/export_prefill.py "$DIR" --shapes 512,2048
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
    echo "  ane [text]      prefill on the Neural Engine + decode on Metal (needs export-ane once)"
    echo "  export-ane      build the Core ML file for the ane backend (once per model)"
    echo "  chat <question> Q&A against an Instruct model"
    echo "  serve [port]    start the HTTP server (default 8080)"
    echo "  test            run unit tests"
    echo "  <anything else> passed straight to the binary, e.g.  ./run.sh -m Qwen/Qwen2.5-0.5B-Instruct --chat -p \"hi\""
    ;;
  *) # unknown command → treat it as raw binary arguments and pass everything through
    exec "$BIN" "$cmd" "$@"
    ;;
esac
