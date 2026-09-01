#!/bin/zsh
# benchmarks/metal-attribution.sh — where a decode step and a prefill chunk of
# the metal GGUF-quant path actually spend their GPU time.
#
# The instrument behind lane metal-perf-attribution. It exists because the
# metal-parity arc has a 2.5x-6.3x decode gap and an 8.4x prefill gap against
# llama.cpp on the SAME files, and "deltanet dispatch count" / "per-element
# i-quant dequant" were suspects, not findings. This turns them into numbers.
#
# Usage: ./metal-attribution.sh <lokal-binary> <model.gguf> [out-dir] [ntok]
#   ntok defaults to 8 (decode seq 0 is dropped, so >= 4 gives >= 3 steady
#   tokens; rule 9 — never time the first token, it carries pipeline warmup).
#
# Prints, per section: a machine-state block (protocol:gpu-bench — a swapping
# or compressing box measures the workload, not the engine), throughput both
# untimed and timed so the timing mode's own overhead is visible rather than
# assumed, and the phase table.
#
# NO COMMITTED BASELINE: every number here is specific to this box, this file
# and this binary. The baseline is the PEER RUN (kv-stub-footprint.sh /
# decode-speed.sh precedent) — run it twice and compare, which is also the
# reproducibility gate this lane owes.
#
# HOW THE NUMBERS ARE PRODUCED, and what each is worth:
#   LOKAL_GPU_TIMING=total  one encoder per step, two timestamps. Structurally
#                           identical to the untimed path, so gpu_sum here is
#                           the TRUE GPU time of a step.
#   LOKAL_GPU_TIMING=1      one encoder per contiguous same-phase run. This is
#                           the attribution, and it costs: every boundary adds
#                           a fixed slice inside the measured window
#                           (calib_inside_ns) and a fixed gap outside it
#                           (calib_gap_ns), both measured in-process and
#                           printed in the `gputime hdr` line. The table below
#                           carries BOTH the raw ms and the ms with that
#                           per-encoder constant subtracted; believe the
#                           corrected column, and check it against the isolate
#                           runs (LOKAL_GPU_TIMING=<phase>) when a phase
#                           decides a lane.
#
# PARSING (protocol:gate-scripts rule 3): every number is read out of the
# binary's own structured `gputime` key=value lines, never out of free text.
# Rule 2: the run is proven to have happened before anything is parsed.
set -u
BIN=${1:?lokal binary}; MODEL=${2:?model gguf}
OUT=${3:-$(mktemp -d)}; NTOK=${4:-8}
[[ -x "$BIN" ]] || { echo "FAIL: binary not executable: $BIN"; exit 2; }
[[ -r "$MODEL" ]] || { echo "FAIL: model not readable: $MODEL"; exit 2; }
(( NTOK >= 4 )) || { echo "FAIL: ntok must be >= 4 (seq 0 is dropped)"; exit 2; }
mkdir -p "$OUT" || { echo "FAIL: cannot create $OUT"; exit 2; }

DECODE_PROMPT=${DECODE_PROMPT:-"The capital of Thailand is"}

# The prefill prompt. Generated, not vendored: a data file next to this script
# is one more thing to keep in sync across two machines, and prefill cost is a
# function of TOKEN COUNT and shapes only — never of what the tokens say. The
# length is asserted below against the binary's own "prefill N tokens" line, so
# a tokenizer that disagrees fails the run instead of quietly measuring pp900.
PREFILL_FILE="$OUT/prompt-prefill.txt"
if [[ -n "${PREFILL_PROMPT_FILE:-}" ]]; then
  [[ -r "$PREFILL_PROMPT_FILE" ]] || { echo "FAIL: PREFILL_PROMPT_FILE unreadable"; exit 2; }
  cp "$PREFILL_PROMPT_FILE" "$PREFILL_FILE"
else
  : > "$PREFILL_FILE"
  for i in $(seq 1 128); do
    print -r -- "Section $i. The engine encodes one command buffer per step and submits it once; the kernels read quantized weights, dequantize them on the fly, and accumulate into f32 activations that never leave the device." >> "$PREFILL_FILE"
  done
fi
PREFILL_MIN=${PREFILL_MIN:-1800}
PREFILL_MAX=${PREFILL_MAX:-2600}

vmstat_n() { vm_stat | awk -F: -v k="$1" '$1 ~ k {gsub(/[ .]/,"",$2); print $2}'; }
SWAP0=$(vmstat_n "Swapins"); DECO0=$(vmstat_n "Decompressions")

# run <tag> <timing-mode|off> <prompt-mode: short|long>
run() {
  local tag=$1 mode=$2 pmode=$3
  local -a args
  args=(-m "$MODEL" -b metal -t 0 -n "$NTOK")
  if [[ $pmode == long ]]; then
    args+=(-p "$(cat "$PREFILL_FILE")")
  else
    args+=(-p "$DECODE_PROMPT")
  fi
  if [[ $mode == off ]]; then
    perl -e 'alarm 1800; exec @ARGV' "$BIN" "${args[@]}" 2>"$OUT/$tag.err" >"$OUT/$tag.txt"
  else
    LOKAL_GPU_TIMING=$mode perl -e 'alarm 1800; exec @ARGV' "$BIN" "${args[@]}" \
      2>"$OUT/$tag.err" >"$OUT/$tag.txt"
  fi
  local rc=$?
  (( rc == 127 )) && { echo "FAIL: $BIN not found / not executable (rc 127)"; exit 2; }
  (( rc == 0 )) || { echo "FAIL: $tag exited rc=$rc"; tail -3 "$OUT/$tag.err"; exit 3; }
  [[ -s "$OUT/$tag.err" ]] || { echo "FAIL: $tag produced no stderr — nothing to parse"; exit 3; }
  if [[ $mode != off ]]; then
    grep -q "^gputime hdr" "$OUT/$tag.err" \
      || { echo "FAIL: $tag ran with LOKAL_GPU_TIMING=$mode but emitted no 'gputime hdr' line"; exit 4; }
    grep -q "^gputime step" "$OUT/$tag.err" \
      || { echo "FAIL: $tag emitted a header but no 'gputime step' line"; exit 4; }
  fi
}

# tok/s out of lokal's own stats line, anchored on its own clause: the line
# carries TWO "(N tok/s)" numbers and an unanchored match takes the wrong one.
toks() { # <file> <prefill|generated>
  grep -oE "$2 [0-9]+ tokens in [0-9.]+s \([0-9.]+ tok/s\)" "$1" \
    | grep -oE '\([0-9.]+ tok/s\)' | grep -oE '[0-9.]+' | head -1
}
prompt_tokens() {
  grep -oE 'prefill [0-9]+ tokens' "$1" | grep -oE '[0-9]+' | head -1
}
hdr_val() { grep -m1 "^gputime hdr" "$1" | grep -oE "$2=[-0-9.]+" | cut -d= -f2; }

# ---- the table. One phase per row, medianed over the steady-state steps of
# ---- one kind, with the per-encoder calibration subtracted in `corr_ms`.
table() { # <err file> <kind> <drop-first-seq: 0|1>
  awk -v kind="$2" -v drop="$3" '
    /^gputime hdr/ {
      for (i = 1; i <= NF; i++) if ($i ~ /^calib_inside_ns=/) { split($i, a, "="); inside = a[2] }
      next
    }
    /^gputime (step|phase)/ {
      delete f
      for (i = 3; i <= NF; i++) { split($i, a, "="); f[a[1]] = a[2] }
      if (f["kind"] != kind) next
      if (drop && f["seq"] + 0 == 0) next
      if ($2 == "step") {
        nstep++
        wall += f["wall_ns"]; enc += f["enc"]; gsum += f["gpu_sum_ns"]
        gspan += f["gpu_span_ns"]; cenc += f["cpu_encode_ns"]
        cpre += f["cpu_pre_ns"]; cpost += f["cpu_post_ns"]
        bad += f["bad"]; empty += f["empty"]; over += f["overflow"]
      } else {
        p = f["name"]
        ns[p] += f["ns"]; pe[p] += f["enc"]; pd[p] += f["disp"]; seen[p] = 1
      }
      next
    }
    END {
      if (nstep == 0) { print "  (no steps)"; exit }
      printf "  steps=%d  bad_enc=%d  empty_enc=%d  overflow=%d\n", nstep, bad, empty, over
      printf "  %-22s %9s %8s %9s %9s %9s\n", "phase", "ms/step", "%gpu", "corr_ms", "enc/step", "disp/step"
      n = 0
      for (p in seen) { n++; key[n] = p }
      # sort by ns desc
      for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++)
        if (ns[key[j]] > ns[key[i]]) { t = key[i]; key[i] = key[j]; key[j] = t }
      for (i = 1; i <= n; i++) {
        p = key[i]
        corr = (ns[p] - pe[p] * inside) / nstep / 1e6
        if (corr < 0) corr = 0
        printf "  %-22s %9.3f %7.1f%% %9.3f %9.1f %9.1f\n",
          p, ns[p]/nstep/1e6, 100*ns[p]/gsum, corr, pe[p]/nstep, pd[p]/nstep
      }
      printf "  %-22s %9.3f %7.1f%%\n", "TOTAL gpu_sum", gsum/nstep/1e6, 100
      printf "  %-22s %9.3f  (%d encoder boundaries/step x calib_gap)\n",
        "gpu inter-encoder gap", (gspan-gsum)/nstep/1e6, enc/nstep
      printf "  %-22s %9.3f\n", "cpu embed (pre)", cpre/nstep/1e6
      printf "  %-22s %9.3f\n", "cpu encode", cenc/nstep/1e6
      printf "  %-22s %9.3f\n", "cpu logits (post)", cpost/nstep/1e6
      printf "  %-22s %9.3f\n", "WALL", wall/nstep/1e6
    }' "$1"
}

echo "== metal-attribution =="
echo "binary: $BIN"
echo "model: $MODEL"
echo "out: $OUT"
echo "ntok: $NTOK"
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo
echo "== A. decode, short prompt =="
run dec-off   off   short
run dec-total total short
run dec-full  1     short
D_OFF=$(toks "$OUT/dec-off.err" generated)
D_TOT=$(toks "$OUT/dec-total.err" generated)
D_FUL=$(toks "$OUT/dec-full.err" generated)
[[ -n "$D_OFF" && -n "$D_TOT" && -n "$D_FUL" ]] \
  || { echo "FAIL: missing 'generated N tokens ... (N tok/s)' clause in a decode run"; exit 4; }
echo "decode-tok-s untimed:            $D_OFF"
echo "decode-tok-s LOKAL_GPU_TIMING=total: $D_TOT"
echo "decode-tok-s LOKAL_GPU_TIMING=1:     $D_FUL"
echo "timing-overhead: $(awk -v a="$D_OFF" -v b="$D_TOT" -v c="$D_FUL" \
  'BEGIN{printf "total-mode %.1f%%, full-split %.1f%% (tok/s lost vs untimed)", 100*(a-b)/a, 100*(a-c)/a}')"
echo "calib_inside_ns: $(hdr_val "$OUT/dec-full.err" calib_inside_ns)  calib_gap_ns: $(hdr_val "$OUT/dec-full.err" calib_gap_ns)"
echo "-- true per-step GPU time (total mode, one encoder, no split distortion):"
table "$OUT/dec-total.err" decode 1
echo "-- attribution (full split):"
table "$OUT/dec-full.err" decode 1

echo
echo "== B. prefill, long prompt =="
run pre-off   off   long
PT=$(prompt_tokens "$OUT/pre-off.err")
[[ -n "$PT" ]] || { echo "FAIL: no 'prefill N tokens' line — cannot verify prompt length"; exit 4; }
(( PT >= PREFILL_MIN && PT <= PREFILL_MAX )) \
  || { echo "FAIL: prompt tokenized to $PT, outside [$PREFILL_MIN,$PREFILL_MAX] — this is not a p2000 run"; exit 4; }
run pre-total total long
run pre-full  1     long
P_OFF=$(toks "$OUT/pre-off.err" prefill)
P_TOT=$(toks "$OUT/pre-total.err" prefill)
P_FUL=$(toks "$OUT/pre-full.err" prefill)
[[ -n "$P_OFF" && -n "$P_TOT" && -n "$P_FUL" ]] \
  || { echo "FAIL: missing 'prefill N tokens ... (N tok/s)' clause in a prefill run"; exit 4; }
echo "prompt-tokens: $PT"
echo "prefill-tok-s untimed:               $P_OFF"
echo "prefill-tok-s LOKAL_GPU_TIMING=total: $P_TOT"
echo "prefill-tok-s LOKAL_GPU_TIMING=1:     $P_FUL"
echo "timing-overhead: $(awk -v a="$P_OFF" -v b="$P_TOT" -v c="$P_FUL" \
  'BEGIN{printf "total-mode %.1f%%, full-split %.1f%% (tok/s lost vs untimed)", 100*(a-b)/a, 100*(a-c)/a}')"
echo "-- true per-chunk GPU time (total mode; every chunk, warmup included):"
table "$OUT/pre-total.err" prefill 0
echo "-- attribution (full split; every chunk):"
table "$OUT/pre-full.err" prefill 0
echo "-- decode attribution at LONG context (same run, pos ~$PT — decode cost is"
echo "   context-dependent, so this row is not the short-prompt one):"
table "$OUT/pre-full.err" decode 1

SWAP1=$(vmstat_n "Swapins"); DECO1=$(vmstat_n "Decompressions")
DSWAP=$(( SWAP1 - SWAP0 )); DDECO=$(( DECO1 - DECO0 ))
echo
echo "== machine state =="
echo "quiet-swapins: $DSWAP"
echo "quiet-decompressions: $DDECO"
echo "loadavg: $(sysctl -n vm.loadavg)"
(( DSWAP > 20000 )) && { echo "FAIL: $DSWAP swapins during the run — the box was swapping, this table is void"; exit 5; }
(( DDECO > 200000 )) && { echo "FAIL: $DDECO decompressions during the run — compressor-bound, this table is void"; exit 5; }
echo "MACHINE STATE OK"
