#!/bin/zsh
# benchmarks/decode-speed.sh — decode throughput of one lokal binary, in tok/s.
#
# The instrument behind the quant-decode-hd256 claim. It exists because five
# merged lanes of identity gates shipped an ~80x decode slowdown invisibly:
# every gate compared BYTES and none compared TIME. Identity proves sameness,
# never usability, so an arc that changes an execution path needs at least one
# throughput gate on the target shape.
#
# Usage: ./decode-speed.sh <lokal-binary> <model> <backend> <ntok> [reps]
# Prints:
#   decode-tok-s: <median>     <- the number
#   decode-runs: r1 r2 r3      <- every sample, so a fluke is visible
#   prefill-tok-s: <median>    <- context, not the claim
#   quiet-swapins / quiet-decompressions / loadavg   <- machine-state evidence
#
# NO COMMITTED BASELINE: tok/s is specific to this box, this model file and this
# backend. The baseline is the PEER RUN — run this once per binary on the same
# box in the same session and compare (the kv-stub-footprint.sh precedent).
#
# PARSING, and the trap it avoids: lokal prints ONE stderr line carrying TWO
# "(N tok/s)" clauses —
#   prefill 5 tokens in 0.93s (5.4 tok/s) | generated 12 tokens in 0.31s (38.2 tok/s)
# an unanchored match takes the PREFILL number and silently reports the wrong
# quantity. Both parses below anchor on their own clause keyword.
#
# protocol:gpu-bench: decode is bandwidth-bound, so a swapping or compressing
# box measures the workload, not the engine. vm_stat deltas are sampled around
# the whole run and the row FAILS past the protocol's thresholds rather than
# publishing a number the machine invalidated.
set -u
BIN=${1:?lokal binary}; MODEL=${2:?model}; BACKEND=${3:?backend}; NTOK=${4:?ntok}; REPS=${5:-3}
[[ -x "$BIN" ]] || { echo "FAIL: binary not executable: $BIN"; exit 2; }
PROMPT=${PROMPT:-"The capital of Thailand is"}

vmstat_n() { vm_stat | awk -F: -v k="$1" '$1 ~ k {gsub(/[ .]/,"",$2); print $2}'; }
SWAP0=$(vmstat_n "Swapins"); DECO0=$(vmstat_n "Decompressions")

DEC=() PRE=()
for r in $(seq 1 "$REPS"); do
  ERR=$(perl -e 'alarm 900; exec @ARGV' "$BIN" -m "$MODEL" -b "$BACKEND" -t 0 \
        -p "$PROMPT" -n "$NTOK" 2>&1 >/dev/null)
  RC=$?
  # Rule 2: prove the command RAN before parsing its output.
  (( RC == 127 )) && { echo "FAIL: $BIN not found / not executable (rc 127)"; exit 2; }
  (( RC == 0 )) || { echo "FAIL: run $r exited rc=$RC"; print -r -- "$ERR" | tail -3; exit 3; }
  [[ -n "$ERR" ]] || { echo "FAIL: run $r produced no stderr — nothing to parse"; exit 3; }
  # Anchored on their own clause; see PARSING above.
  d=$(print -r -- "$ERR" | grep -oE 'generated [0-9]+ tokens in [0-9.]+s \([0-9.]+ tok/s\)' \
        | grep -oE '\([0-9.]+ tok/s\)' | grep -oE '[0-9.]+' | head -1)
  p=$(print -r -- "$ERR" | grep -oE 'prefill [0-9]+ tokens in [0-9.]+s \([0-9.]+ tok/s\)' \
        | grep -oE '\([0-9.]+ tok/s\)' | grep -oE '[0-9.]+' | head -1)
  [[ -n "$d" ]] || { echo "FAIL: run $r — no 'generated N tokens ... (N tok/s)' clause in stderr"; exit 4; }
  DEC+=("$d"); PRE+=("${p:-0}")
done
SWAP1=$(vmstat_n "Swapins"); DECO1=$(vmstat_n "Decompressions")
DSWAP=$(( SWAP1 - SWAP0 )); DDECO=$(( DECO1 - DECO0 ))

med() { print -l -- "$@" | sort -g | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
echo "decode-tok-s: $(med "${DEC[@]}")"
echo "decode-runs: ${DEC[*]}"
echo "prefill-tok-s: $(med "${PRE[@]}")"
echo "quiet-swapins: $DSWAP"
echo "quiet-decompressions: $DDECO"
echo "loadavg: $(sysctl -n vm.loadavg)"
# protocol:gpu-bench thresholds — a thrashing box measures the workload.
(( DSWAP > 20000 )) && { echo "FAIL: $DSWAP swapins during the run — the box was swapping, this row is void"; exit 5; }
(( DDECO > 200000 )) && { echo "FAIL: $DDECO decompressions during the run — compressor-bound, this row is void"; exit 5; }
echo "MACHINE STATE OK"
