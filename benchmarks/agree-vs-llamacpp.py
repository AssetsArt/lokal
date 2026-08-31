"""Greedy agreement vs llama.cpp on the SAME file.

Compares against llama-server's /completion (RAW), never llama-cli: llama-cli
applies the GGUF's chat template to Instruct models, so its text answers a
different question and reads as a total disagreement at token 1.
Agreement, not identity, is the bar — different kernel order — so divergence
positions are reported rather than asserted away.
"""
import json, subprocess, sys, urllib.request
from pathlib import Path

PORT, N = 18099, 48
BIN = str(Path(__file__).resolve().parent.parent / "target/release/lokal")
MODEL = sys.argv[1]
# A LONG prompt is not optional. Every prompt here used to be under twenty
# tokens, and a real qwen3 scratch-buffer overflow (q/att/xh sized by
# hidden_size instead of n_heads*head_dim) passed this gate 5/5 because no
# prompt was long enough to reach the oversized region. Short prompts test the
# math; long prompts test the sizes.
LONG = " ".join(
    ["The quiet library at the edge of the harbor kept a ledger of every ship "
     "that never returned, and the keeper read it aloud each winter morning to "
     "nobody in particular."] * 12
) + " Summarised in one sentence, this passage says that"

PROMPTS = [
    "The three most important ideas in operating system design are",
    "The capital of Thailand is",
    "In 1969, the first humans landed on the moon. The mission was called",
    "def fibonacci(n):",
    "Water boils at a temperature of",
    LONG,
]

def llama(p):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps({"prompt": p, "n_predict": N, "temperature": 0,
                         "top_k": 1, "cache_prompt": False, "seed": 1}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["content"]

def lokal(p):
    out = subprocess.run([BIN, "-b", "lowmem", "-m", MODEL, "-t", "0", "-n", str(N), "-p", p],
                         capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        print(f"GATE ERROR: lokal exited {out.returncode}: {out.stderr[-300:]}"); sys.exit(2)
    t = out.stdout
    return t[len(p):] if t.startswith(p) else t

agree = tot = 0
flips = []
for p in PROMPTS:
    a, b = lokal(p), llama(p)
    n = min(len(a), len(b))
    if n == 0:
        print(f"GATE ERROR: empty completion for {p!r}"); sys.exit(2)
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    agree += i; tot += n
    mark = "identical" if i >= n else f"diverges at char {i}: {a[i:i+30]!r} vs {b[i:i+30]!r}"
    if i < n:
        flips.append((p, i, n))
    label = p[:44] if len(p) <= 44 else p[:28] + f"... ({len(p)} chars)"
    print(f"  [{i}/{n}] {label!r}: {mark}")

pct = 100.0 * agree / tot
ident = len(PROMPTS) - len(flips)
first = min([i for _, i, _ in flips], default=10**9)
print(f"K3 agreement {pct:.1f}% of {tot} compared chars; {ident}/{len(PROMPTS)} prompts identical "
      f"over {N} tokens; earliest divergence at char {first if flips else '-'}")

# The bar is agreement, not identity: lokal and llama.cpp sum in different
# orders, and under greedy decoding one ulp eventually elects a different
# token. What must NOT happen is an EARLY divergence — the first token differing
# on identical weights is a systematic dequant error, not rounding — or a
# collapse in the aggregate. Both thresholds are about telling those apart.
ok = first >= 16 and pct >= 60.0 and ident >= len(PROMPTS) // 2
print("K3 PASS" if ok else "K3 FAIL")
sys.exit(0 if ok else 1)
