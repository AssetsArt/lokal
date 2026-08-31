#!/bin/zsh
# The Studio acceptance for quantized GGUF on -b metal (one command, run on the
# 32 GB M2 Ultra): loads the model RESIDENT with full attention, proves the
# footprint is quant-sized (no f32 expansion), and reports prefill+decode tok/s
# behind the same quiet gate as the other collectors.
#
#   MODEL="unsloth/Qwen3-27B-GGUF:Q4_K_M" ./benchmarks/collect-metal-quant.sh
#   MODEL=/path/to/model-Q4_K_M.gguf      ./benchmarks/collect-metal-quant.sh
#
# Speed runs take NO vmmap sample — vmmap SUSPENDS its target; footprint is a
# separate, explicitly untimed run (same rule as collect-gguf-rows.sh).
set -u
WT=$(cd "$(dirname "$0")/.." && pwd)
B=$WT/target/release/lokal
S=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:?usage: MODEL=owner/repo:TAG (or /path/to/file.gguf) $0}
N=${N:-64}
P=${P:-"The three most important ideas in operating system design are"}
[ -x "$B" ] || { echo "GATE ERROR: build first (rustup run stable cargo build --release)"; exit 2; }

echo "===== RESOLVE ====="
FILE=$("$B" path -m "$MODEL") || { echo "GATE ERROR: could not resolve $MODEL"; exit 2; }
SZ=$(stat -f %z "$FILE")
echo "  file: $FILE ($((SZ / 1048576)) MB)"

echo "===== QUIET GATE (before) ====="
python3 $S/quiet.py || { echo "ABORT: machine not quiet"; exit 1; }

echo "===== TIMED RUN ====="
"$B" -b metal -m "$FILE" -t 0 -n "$N" -p "$P" >"$S/mq_run.out" 2>"$S/mq_run.err"
rc=$?
[ $rc -eq 0 ] || { echo "FAIL: exited rc=$rc"; tail -3 "$S/mq_run.err"; exit 1; }
grep -E "Metal quant:|prefill .* tok/s" "$S/mq_run.err" | sed 's/^/  /'
echo "  first output: $(head -c 120 "$S/mq_run.out")"

echo "===== DETERMINISM ====="
"$B" -b metal -m "$FILE" -t 0 -n "$N" -p "$P" >"$S/mq_run2.out" 2>/dev/null
cmp -s "$S/mq_run.out" "$S/mq_run2.out" && echo "  two greedy runs byte-identical" \
  || { echo "FAIL: nondeterministic"; exit 1; }

echo "===== FOOTPRINT (UNTIMED — vmmap suspends the target) ====="
"$B" -b metal -m "$FILE" -t 0 -n 16 -p "$P" >/dev/null 2>"$S/mq_fp.err" &
PID=$!
sleep 20
FP=$(vmmap "$PID" 2>/dev/null | awk '/Physical footprint:/ {print $3; exit}')
kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "  phys_footprint: ${FP:-unavailable} (file is $((SZ / 1048576)) MB)"
# The whole point: footprint tracks the QUANT bytes, not a 4x-6x f32 blowup.
python3 - "$FP" "$SZ" <<'PY'
import sys
fp, sz = sys.argv[1], int(sys.argv[2])
mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
if not fp or fp[-1] not in mult:
    print("  note: footprint sample unavailable — rerun; do not report without it")
    sys.exit(1)
bytes_fp = float(fp[:-1]) * mult[fp[-1]]
lim = sz * 1.9 + (2 << 30)
ok = bytes_fp < lim
print(f"  {'OK' if ok else 'FAIL'}: footprint {bytes_fp/2**30:.1f} GB vs limit {lim/2**30:.1f} GB (quant-resident bound)")
sys.exit(0 if ok else 1)
PY
rc=$?
[ $rc -eq 0 ] || exit 1

echo "===== QUIET GATE (after) ====="
python3 $S/quiet.py || echo "  note: machine got busy during the run — treat the tok/s as a floor"
echo "\nDONE — paste this whole output back."
