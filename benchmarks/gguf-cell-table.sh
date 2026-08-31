#!/bin/zsh
# The backend×format 16-cell identity table — the gate instrument behind the
# matrix in docs/gguf-design.md. Captures every cell greedily and (optionally)
# cmp's each one byte-for-byte against a baseline directory.
#
# Usage: ./gguf-cell-table.sh <lokal-binary> <out-dir> [baseline-dir]
#   1) captures all 16 cells into <out-dir> (greedy -t 0; stdout only — the
#      perf/banner lines go to stderr and are dropped; NO other post-processing)
#   2) if [baseline-dir] is given, cmp's every cell byte-for-byte against it.
# Committed baseline: benchmarks/cell-baselines/ (captured on main @4279c03,
# after rope-mirror-fix re-baselined the three gg-*-metal cells; the two
# q35-metal-* cells were added by lane metal-deltanet, which made that cell
# runnable at all, and their baselines are byte-equal to the lowmem cells they
# mirror — that equality IS the lane's gate, so a DIFFERS on either of them
# means metal and lowmem have drifted apart again).
#
# Provenance: written by Tiësto in lane gguf-unify; committed after the
# scratchpad-only copy was nearly lost. If you ever reconstruct or modify this
# harness, VALIDATE it against a known-good baseline before trusting a verdict.
# Two pins that have already produced spurious CHANGED cells when guessed:
#   - smol-* cells use SmolLM2-135M BASE, not -Instruct;
#   - q35-thailand and q35-metal-thailand use -n 12 (every other cell -n 24).
# A cell the change-under-test cannot reach (e.g. smol-cpu for a metal.rs
# change) that reads CHANGED indicts the harness, not the code.
# Baselines are valid only for the model files pinned below; regenerate the
# whole directory if any of them changes.
set -u
BIN=${1:?lokal binary}; OUT=${2:?out dir}; REF=${3:-}
mkdir -p "$OUT"
Q8=$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/*/qwen2.5-0.5b-instruct-q8_0.gguf | head -1)
Q35=$(ls ~/.cache/huggingface/hub/models--unsloth--Qwen3.5-2B-GGUF/snapshots/f6d5376*/Qwen3.5-2B-Q4_K_M.gguf | head -1)
F16=~/.cache/gguf/qwen2.5-0.5b-instruct-f16-from-7ae55760.gguf
Q3=~/.cache/gguf/Qwen3-0.6B-Q4_K_M.gguf
run() { # name model backend [prompt] [n]
  perl -e 'alarm 300; exec @ARGV' "$BIN" -m "$2" -b "$3" -t 0 \
    -p "${4:-The old lighthouse keeper}" -n "${5:-24}" 2>/dev/null > "$OUT/$1.txt" \
    || echo "RUN FAILED: $1"
}
run smol-cpu     HuggingFaceTB/SmolLM2-135M        cpu
run smol-metal   HuggingFaceTB/SmolLM2-135M        metal
run smol-hybrid  HuggingFaceTB/SmolLM2-135M        hybrid
run q25-cpu      Qwen/Qwen2.5-0.5B-Instruct        cpu
run q25-metal    Qwen/Qwen2.5-0.5B-Instruct        metal
run q25-hybrid   Qwen/Qwen2.5-0.5B-Instruct        hybrid
run gg-f16-metal  "$F16" metal
run gg-f16-lowmem "$F16" lowmem
run gg-q3-metal   "$Q3"  metal
run gg-q3-lowmem  "$Q3"  lowmem
run gg-q8-metal   "$Q8"  metal
run gg-q8-lowmem  "$Q8"  lowmem
run q35-lowmem    "$Q35" lowmem
run q35-thailand  "$Q35" lowmem "The capital of Thailand is" 12
# qwen35 on metal (lane metal-deltanet). Held byte-equal to the two lowmem
# cells above, not merely to its own past self.
run q35-metal          "$Q35" metal
run q35-metal-thailand "$Q35" metal "The capital of Thailand is" 12
if [[ -n "$REF" ]]; then
  DIFF=0
  for f in "$REF"/*.txt; do
    n=$(basename "$f")
    cmp -s "$f" "$OUT/$n" || { echo "DIFFERS: $n"; DIFF=1; }
  done
  (( DIFF == 0 )) && echo "ALL 16 CELLS BYTE-IDENTICAL vs $REF" || exit 1
fi
