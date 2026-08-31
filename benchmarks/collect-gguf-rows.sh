#!/bin/zsh
# The two quiet-machine numbers: 14B Q4 fully resident, and the 27B-class
# acceptance that cannot fit and must stream.
#
# Speed runs take NO vmmap sample — vmmap SUSPENDS its target, so sampling a
# timed run corrupts the very number it is meant to certify. Footprint is a
# separate, explicitly untimed run.
set -u
WT=$(cd "$(dirname "$0")/.." && pwd)
B=$WT/target/release/lokal
S=$(cd "$(dirname "$0")" && pwd)
M14=${M14:-$HOME/.cache/gguf/Qwen2.5-14B-Instruct-Q4_K_M.gguf}
M32=${M32:-$HOME/.cache/gguf/Qwen3-32B-Q4_K_M.gguf}
BUD=12000
P="The three most important ideas in operating system design are"
[ -x "$B" ] || { echo "GATE ERROR: binary missing"; exit 2; }
for m in "$M14" "$M32"; do
  [ -s "$m" ] || { echo "GATE ERROR: model missing at $m"; echo "  override with M14=/path M32=/path $0"; exit 2; }
done

echo "===== QUIET GATE (before) ====="
python3 $S/quiet.py || { echo "ABORT: machine not quiet before the runs"; exit 1; }

run() { # run <label> <model> <n>
  echo "\n===== $1 ====="
  LOKAL_LOWMEM_STATS=1 "$B" -b lowmem -m "$2" --memory-budget $BUD -t 0 -n "$3" -p "$P" \
      >"$S/big_$1.out" 2>"$S/big_$1.err"
  local rc=$?
  [ $rc -eq 0 ] || { echo "FAIL: $1 exited rc=$rc"; tail -3 "$S/big_$1.err"; return 1; }
  grep -E "budget [0-9]+ MB|disk-bound|lowmem: stats|tok/s" "$S/big_$1.err" | sed 's/^/  /'
  echo "  first line: $(head -c 110 "$S/big_$1.out")"
}

run 14B_resident "$M14" 32 || exit 1
run 32B_acceptance "$M32" 24 || exit 1

# Untimed footprint sample. vmmap suspends the target, so this run's tok/s is
# meaningless BY CONSTRUCTION and is not reported.
echo "\n===== 14B footprint (UNTIMED — vmmap suspends the target) ====="
LOKAL_LOWMEM_STATS=1 "$B" -b lowmem -m "$M14" --memory-budget $BUD -t 0 -n 24 -p "$P" \
    >"$S/big_fp.out" 2>"$S/big_fp.err" &
pid=$!; max=0; n=0
while kill -0 $pid 2>/dev/null; do
  f=$(vmmap -summary $pid 2>/dev/null | awk '/Physical footprint:/ {print $3; exit}')
  case "$f" in
    *G) v=$(echo "$f" | sed 's/G//' | awk '{printf "%d", $1*1024}');;
    *M) v=$(echo "$f" | sed 's/M//' | awk '{printf "%d", $1}');;
    *) v=0;;
  esac
  [ "$v" -gt "$max" ] && max=$v; n=$((n+1)); sleep 2
done
wait $pid
LIM=$(( BUD + BUD / 10 ))
echo "  peak phys_footprint ${max} MB over $n samples vs budget+10% = ${LIM} MB"
[ "$max" -ge 500 ] && [ "$n" -ge 5 ] || { echo "  FAIL: sampler never saw the process"; exit 1; }
[ "$max" -le "$LIM" ] && echo "  FOOTPRINT PASS" || { echo "  FOOTPRINT FAIL"; exit 1; }

echo "\n===== QUIET GATE (after) ====="
python3 $S/quiet.py || { echo "WARNING: machine was NOT quiet after — numbers above are suspect"; exit 1; }
echo "\nK4 COMPLETE"
