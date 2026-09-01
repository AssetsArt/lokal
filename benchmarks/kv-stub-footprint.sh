#!/bin/zsh
# benchmarks/kv-stub-footprint.sh — how much GPU memory ONE session allocates.
#
# The instrument behind the kv-recurrent-stub-metal claim: a recurrent
# (gated-deltanet) layer carries conv + delta state and never touches a KV
# cache, so allocating cap × kv_dim there is RAM nobody reads. This measures
# the allocation, not the residency.
#
# Usage: ./kv-stub-footprint.sh <lokal-binary> <model> <backend> <max-tokens>
# Prints, on stdout, three structured lines a caller can diff:
#   ioaccel-virtual-bytes: N   ← Metal buffer address space (THE number)
#   footprint-peak-bytes:  N   ← physical footprint peak (noisy; see below)
#   samples:               N
#
# WHY VIRTUAL IS THE HEADLINE, and dirty/footprint is not:
# macOS commits pages lazily (the same fact src/engine.rs relies on for pooled
# KV slots), so a cache the model never reads may cost little physical RAM
# until it is touched — a resident-only metric under-reports an allocation win
# and can even read backwards on a swapping box (memory: a swapping box
# measures the workload, not the engine). vmmap's "IOAccelerator (graphics)"
# VIRTUAL column is the driver's own count of what the process asked the GPU
# for, which is exactly the quantity this lane changes.
#
# There is NO committed baseline capture: a footprint number is specific to
# this box, this model file and this -n. The baseline is the PEER RUN — run
# this twice on the same box, once per binary, and diff. Comparing against a
# number captured elsewhere is meaningless and must not be done.
#
# Sampling follows protocol:gate-scripts rule 9: poll-max in a while-alive
# loop, never a fixed sleep offset, so a faster run cannot make the gate read
# "unavailable". Allocation happens at session construction (after the weights
# load), so the poll must outlive model load — hence MAXSEC, not one shot.
set -u
BIN=${1:?lokal binary}; MODEL=${2:?model}; BACKEND=${3:?backend}; NTOK=${4:?max tokens}
MAXSEC=${MAXSEC:-90}
[[ -x "$BIN" ]] || { echo "FAIL: binary not executable: $BIN"; exit 2; }
command -v vmmap >/dev/null || { echo "FAIL: vmmap not on PATH"; exit 2; }

# to bytes: vmmap prints 112K / 378.4M / 1.2G
tob() { print -r -- "$1" | awk '
  /K$/{sub(/K$/,"");printf "%.0f\n",$0*1024;next}
  /M$/{sub(/M$/,"");printf "%.0f\n",$0*1048576;next}
  /G$/{sub(/G$/,"");printf "%.0f\n",$0*1073741824;next}
  {printf "%.0f\n",$0}'; }

"$BIN" -m "$MODEL" -b "$BACKEND" -t 0 -p "The capital of Thailand is" -n "$NTOK" \
  >/dev/null 2>/dev/null &
PID=$!
BEST_V=0 BEST_F=0 N=0 I=0
while kill -0 $PID 2>/dev/null; do
  I=$((I+1)); (( I > MAXSEC * 2 )) && break
  SUM=$(vmmap --summary $PID 2>/dev/null)
  if [[ -n "$SUM" ]]; then
    V=$(print -r -- "$SUM" | awk -F'  +' '/^IOAccelerator \(graphics\)/{print $2}' | head -1)
    F=$(print -r -- "$SUM" | awk '/^Physical footprint \(peak\)/{print $NF}' | head -1)
    if [[ -n "${V:-}" ]]; then
      VB=$(tob "$V"); (( VB > BEST_V )) && BEST_V=$VB
      N=$((N+1))
    fi
    if [[ -n "${F:-}" ]]; then
      FB=$(tob "$F"); (( FB > BEST_F )) && BEST_F=$FB
    fi
  fi
  sleep 0.5
done
kill $PID 2>/dev/null; wait $PID 2>/dev/null

# Rule 2: assert the measurement RAN before reporting a number. Zero usable
# samples means vmmap never saw the process (died early, wrong pid, SIP) — a
# loud failure, never a silent 0.
if (( N == 0 )); then
  echo "FAIL: no vmmap sample carried an 'IOAccelerator (graphics)' row —"
  echo "      the run died before allocating, or vmmap could not attach to $PID"
  exit 3
fi
echo "ioaccel-virtual-bytes: $BEST_V"
echo "footprint-peak-bytes: $BEST_F"
echo "samples: $N"
