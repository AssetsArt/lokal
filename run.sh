#!/usr/bin/env bash
# Convenience launcher for lokal — list commands with: ./run.sh help
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release --quiet
BIN=target/release/lokal

cmd="${1:-demo}"
shift || true

# Shared by the export commands: model spec and -f may appear in any order.
# Resolves DIR (the weights) and GDIR (the lokal-owned graph directory,
# outside the HF cache — `lokal path --graphs` is the one path rule).
parse_export_args() {
  MODEL="" FORCE=""
  for a in "$@"; do
    case "$a" in
      -f | --force) FORCE=1 ;;
      *) MODEL="$a" ;;
    esac
  done
  MODEL="${MODEL:-HuggingFaceTB/SmolLM2-135M}"
  if [ -d "$MODEL" ]; then DIR="$MODEL"; else DIR="$("$BIN" path -m "$MODEL")"; fi
  GDIR="$("$BIN" path -m "$MODEL" --graphs)"
}

# The split-prefill ladder spec for a model (see export-ane-split for why the
# values are what they are).
ladder_spec() {
  case "$1" in
    *[Qq]wen*)
      # 24 layers x 896 hidden: stride 512, 9 front layers (8 and 10 both
      # measured slower at 7.7k). ~500 MB per rung — the 272 MB embedding
      # table rides along in every one.
      echo 512x0x9,512x2048x9,512x4096x9,512x7168x9 ;;
    *)
      # SmolLM2 (30 layers x 576 hidden) and the default for untested models:
      # stride 128 @ 20 front layers short, stride 256 @ 13 beyond, long-band
      # rungs so an 8k prompt still pipelines instead of draining to Metal.
      echo 128x128x20,128x384x20,256x0x13,256x256x13,256x512x13,256x768x13,256x1280x13,256x2048x13,256x3072x13,256x4608x13,256x7168x13 ;;
  esac
}

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
  export-hybrid) # ./run.sh export-hybrid [repo or dir] [-f]  — everything -b hybrid runs on
    # One command, no sub-modes: the plain prefill graphs AND the split-prefill
    # ladder, in that order. First run costs minutes and ~900 MB; graphs already
    # built for this snapshot revision are skipped, so re-running resumes.
    parse_export_args "$@"
    SPEC="$(ladder_spec "$MODEL")"
    echo "note: building the plain graphs plus the split ladder (every ladder rung carries" >&2
    echo "      its own weight copy — minutes and ~900 MB on a first run, skips on a re-run)," >&2
    echo "      into $GDIR" >&2
    exec uv run --python 3.12 --with torch --with coremltools --with safetensors --with numpy --with tokenizers \
      tools/export_prefill.py "$DIR" --shapes 512,2048 --window none --front "$SPEC" \
      --out "$GDIR" --model "$MODEL" ${FORCE:+-f}
    ;;
  export-ane) # legacy alias: the plain prefill graphs only (what export-ane always built)
    parse_export_args "$@"
    exec uv run --python 3.12 --with torch --with coremltools --with safetensors --with numpy --with tokenizers \
      tools/export_prefill.py "$DIR" --shapes 512,2048 --window none \
      --out "$GDIR" --model "$MODEL" ${FORCE:+-f}
    ;;
  export-ane-split) # legacy alias: the split-prefill ladder only
    parse_export_args "$@"
    # The ladder is per model, and so is its split point: where the GPU half
    # binds, the ANE should carry more layers, and that balance moves with the
    # chunk width AND the model shape. Every spec in ladder_spec is exactly what
    # its model's published numbers were measured with (2026-08-30); an A sweep
    # on each side of every value measured slower. Do not add rungs casually:
    # the widest rung of a stride sets how far the ANE reaches (ane_total = s +
    # p_max in ane.rs), so a new one silently changes which part of a prompt
    # runs where. P=0 rungs let the first chunk skip the phantom past attention.
    SPEC="$(ladder_spec "$MODEL")"
    case "$MODEL" in *[Qq]wen*) N=four ;; *) N=eleven ;; esac
    echo "note: $N front-half graphs (every rung carries its own weight copy), plus one" >&2
    echo "      ANE compile per rung on this machine's first load. Once exported," >&2
    echo "      -b hybrid uses the ladder automatically." >&2
    exec uv run --python 3.12 --with torch --with coremltools --with safetensors --with numpy --with tokenizers \
      tools/export_prefill.py "$DIR" --shapes none --window none --front "$SPEC" \
      --out "$GDIR" --model "$MODEL" ${FORCE:+-f}
    ;;
  export-ane-long) # legacy alias: the long-context windowed graph only (off the default route)
    parse_export_args "$@"
    echo "note: the windowed graph takes minutes to convert, and its first load on each" >&2
    echo "      machine spends ~4 min more in the ANE compiler (one-time, then cached)." >&2
    exec uv run --python 3.12 --with torch --with coremltools --with safetensors --with numpy --with tokenizers \
      tools/export_prefill.py "$DIR" --shapes none --window 1024x7168 \
      --out "$GDIR" --model "$MODEL" ${FORCE:+-f}
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
    echo "  hybrid [text]   Neural Engine + GPU together, decode on Metal (needs export-hybrid once)"
    echo "  export-hybrid   build everything -b hybrid runs on (once per model — minutes, ~900 MB)"
    echo "  chat <question> Q&A against an Instruct model"
    echo "  serve [port]    start the HTTP server (default 8080)"
    echo "  test            run unit tests"
    echo "  <anything else> passed straight to the binary, e.g.  ./run.sh -m Qwen/Qwen2.5-0.5B-Instruct --chat -p \"hi\""
    ;;
  *) # unknown command → treat it as raw binary arguments and pass everything through
    exec "$BIN" "$cmd" "$@"
    ;;
esac
