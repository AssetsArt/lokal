# bench_engines.py — compare lokal against other local inference servers.
#
# Measures three things against an already-running server, greedy (temperature 0):
#   prefill   — wall time of the prompt with max_tokens=1 (TTFT proxy)
#   decode    — (tokens_long - 1) / (wall_long - wall_prefill), i.e. the marginal
#               cost per generated token, which cancels out prompt processing
#   Nx concurrent — aggregate generated tokens/sec with N simultaneous requests
#
# Two prompt workloads:
#   default             the original ~460-token synthetic prompt, kept so the
#                       short-prompt table in README.md stays reproducible
#   --prompt-tokens N   natural prose sliced out of a pinned public-domain book
#                       to roughly N tokens — the long-context workload
#
# Every engine gets the SAME text; engines tokenize it slightly differently, so
# each one is asked for its OWN token count and prefill tok/s is normalized by
# that count. Completion lengths are reported too, so an early EOS is visible
# instead of silently skewing the numbers.
#
# Each request starts with a unique nonce so server-side prompt caches (llama.cpp
# reuses the KV of a repeated prefix, for example) cannot answer from cache — the
# benchmark measures compute, not caching.
#
# Usage — one engine, server started and stopped for you (engines.py knows the
# flags; --model picks the family, default smol):
#   python3 benchmarks/bench_engines.py --engine lokal-hybrid
#   python3 benchmarks/bench_engines.py --engine llamacpp --model qwen --prompt-tokens 2000
#   python3 benchmarks/bench_engines.py --engine lokal-hybrid-cli --model qwen --prompt-tokens 16000
#
# Or point it at a server you are running yourself:
#   python3 benchmarks/bench_engines.py --api lokal  --url http://127.0.0.1:8080 --name "lokal (metal)"
#   python3 benchmarks/bench_engines.py --api openai --url http://127.0.0.1:8081/v1 \
#       --model-name SmolLM2-135M-Instruct --name "llama.cpp" --out benchmarks/results.jsonl
#
# --api lokal  → POST {url}/generate            (lokal's native endpoint)
# --api openai → POST {url}/chat/completions    (llama.cpp, oMLX, vLLM, ...)
# --api cli    → run the lokal binary once per request. serve mode caps a pooled
#                slot at 8192 tokens (src/batch.rs POOL_SEQ_CAP), so prompts past
#                that can only be measured through the CLI path.
# --out        → append the JSON result line to this file

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# The shared engine registry lives beside this file; make it importable no
# matter which directory the benchmark is started from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engines  # noqa: E402

SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the river runs "
    "quietly under the old stone bridge near the edge of town. "
)
# The instruction that follows the prompt body. It has to keep a 0.5B model
# talking for the full max_tokens: "retell everything above" makes Qwen2.5-0.5B
# stop after ~12 tokens, which would leave the decode rate measured over almost
# no work. This phrasing generated the full 128 every time.
TASK = ("Continue the passage above in the same style, writing a long new chapter "
        "of at least four hundred words. Do not stop early.")
# The original ~460-token workload, kept verbatim so the short-prompt table in
# README.md stays reproducible.
PROMPT = SENTENCE * 18 + "Retell everything above as one long detailed story. Do not stop."
MAX_TOKENS = 128
RUNS = 5       # per single-request metric; the median is reported
CONC_RUNS = 3  # per concurrency metric

# Long-prompt corpus: Moby Dick from Project Gutenberg — public domain, plain
# UTF-8, ~1.2 MB (≈290k tokens), far more than the largest prompt we build. It
# is downloaded on first use into benchmarks/.cache/ (gitignored — the repo does
# not carry a megabyte of novel) and pinned by sha256 so a silent upstream edit
# turns into a loud failure instead of a moved baseline.
CORPUS_URL = "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"
CORPUS_FILE = "pg2701.txt"
CORPUS_SHA256 = "9a6844ac0703853720010787c7b6c70b0020f1ab1862dcd74452fa46474d1215"
CORPUS_LABEL = "gutenberg-2701 (Moby Dick)"
# Skip the Gutenberg header and the table of contents; chapter 1 opens here.
CORPUS_START = "Call me Ishmael"
# Measured on this corpus with Qwen2.5's BPE: 43685 characters of the
# hard-wrapped prose tokenize to 10743 tokens, i.e. 4.07 characters each. Used
# only to turn --prompt-tokens into a character slice; the token counts that get
# reported always come from the engines themselves, so a drifting ratio makes a
# size label slightly off, never a measurement wrong.
CHARS_PER_TOKEN = 4.07

# "prefill 12345 tokens in 3.21s (…) | generated 128 tokens in 0.53s (…)"
CLI_STATS = re.compile(
    r"prefill (\d+) tokens in ([\d.]+)s.*?generated (\d+) tokens in ([\d.]+)s", re.S
)
# "  ANE prefill: 6144 tokens (6 chunks, windowed S=1024 P=5120) in 1.98s"
CLI_ANE = re.compile(r"ANE prefill: (\d+) tokens \((\d+) chunks.*?\) in ([\d.]+)s")


def corpus_text():
    """The prose body of the pinned book, downloaded once and sha256-verified."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", CORPUS_FILE)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading corpus {CORPUS_URL} -> {path}")
        with urllib.request.urlopen(CORPUS_URL, timeout=120) as r:
            blob = r.read()
        with open(path, "wb") as f:
            f.write(blob)
    else:
        with open(path, "rb") as f:
            blob = f.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != CORPUS_SHA256:
        raise SystemExit(
            f"corpus sha256 mismatch: expected {CORPUS_SHA256}, got {digest}.\n"
            f"Upstream changed the file — delete {path}, re-check the text, and "
            f"update CORPUS_SHA256 before trusting any number measured with it."
        )
    text = blob.decode("utf-8")
    return text[text.index(CORPUS_START):]


def build_prompt(args):
    """The fixed prompt body — identical for every engine in a run."""
    if args.prompt_tokens <= 0:
        return PROMPT
    chars = args.prompt_chars or round(args.prompt_tokens * CHARS_PER_TOKEN)
    text = corpus_text()
    if chars > len(text):
        raise SystemExit(f"corpus holds {len(text)} chars, need {chars}")
    # End on a whole word so the last token is not a truncated fragment.
    body = text[:chars].rsplit(" ", 1)[0]
    return f"{body}\n\n{TASK}"


def call(args, max_tokens):
    """One request. Returns a dict; keys the transport cannot fill stay None."""
    prompt = f"(session {time.time_ns()}) {args.prompt_body}"
    if args.api == "cli":
        return call_cli(args, prompt, max_tokens)
    if args.api == "lokal":
        body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "chat": True}
        url = f"{args.url}/generate"
    else:
        body = {
            "model": args.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        url = f"{args.url}/chat/completions"
    headers = {"content-type": "application/json"}
    if args.bearer:
        headers["authorization"] = f"Bearer {args.bearer}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        out = json.load(r)
    wall = time.time() - t0
    if args.api == "lokal":
        gen, ptok = out["generated_tokens"], out.get("prompt_tokens")
    else:
        usage = out.get("usage") or {}
        gen, ptok = usage["completion_tokens"], usage.get("prompt_tokens")
    return {"wall": wall, "gen": gen, "prompt_tokens": ptok, "prefill_s": None, "decode_s": None}


def call_cli(args, prompt, max_tokens):
    """One run of the lokal binary. Its stderr carries the engine's own timings,
    which is the only honest prefill number here — process wall would include
    model load and Core ML graph compilation."""
    cmd = [args.bin, "-b", args.backend, "-m", args.model_name, "--chat",
           "-t", "0", "-n", str(max_tokens), "-p", prompt]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    wall = time.time() - t0
    if p.returncode != 0:
        raise SystemExit(f"lokal exited {p.returncode}:\n{p.stderr[-2000:]}")
    m = CLI_STATS.search(p.stderr)
    if not m:
        raise SystemExit(f"could not parse lokal's stats line from:\n{p.stderr[-2000:]}")
    ptok, prefill_s, gen, decode_s = int(m[1]), float(m[2]), int(m[3]), float(m[4])
    rec = {"wall": wall, "gen": gen, "prompt_tokens": ptok,
           "prefill_s": prefill_s, "decode_s": decode_s}
    a = CLI_ANE.search(p.stderr)
    if a:
        rec["ane"] = {"tokens": int(a[1]), "chunks": int(a[2]), "secs": float(a[3])}
    return rec


def median_run(args, max_tokens, runs):
    """`runs` requests; returns the per-field medians plus the last ANE line."""
    recs = [call(args, max_tokens) for _ in range(runs)]
    ane = next((r["ane"] for r in reversed(recs) if r.get("ane")), None)

    def med(key):
        vals = [r[key] for r in recs if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    return {k: med(k) for k in ("wall", "gen", "prompt_tokens", "prefill_s", "decode_s")} | {"ane": ane}


def measure(args):
    """Prefill seconds and decode tok/s, by whichever route the transport allows.

    Over HTTP the server reports no split, so prefill is the wall of a
    max_tokens=1 request and decode is the marginal rate against a second,
    longer request. The CLI prints its own prefill/decode split, so one run per
    sample suffices — which matters: a 32k Metal prefill costs minutes, and
    halving the passes halves the whole long-context sweep."""
    if args.api == "cli":
        long = median_run(args, args.max_tokens, args.runs)
        return (long, long["prefill_s"], "engine-internal",
                long["gen"] / max(long["decode_s"], 1e-9), "engine-internal")
    short = median_run(args, 1, args.runs)
    long = median_run(args, args.max_tokens, args.runs)
    # Both requests pay the same prompt processing, so subtracting the walls
    # leaves only generation.
    decode_tps = (long["gen"] - 1) / max(long["wall"] - short["wall"], 1e-9)
    return long, short["wall"], "http-wall(max_tokens=1)", decode_tps, "marginal-wall"


def count_tokens(args, observed):
    """How many tokens the engine saw. Prefer what it reported; then llama.cpp's
    /tokenize; then a labelled character estimate."""
    if observed is not None:
        return int(observed), "engine-reported"
    body = json.dumps({"content": args.prompt_body}).encode()
    req = urllib.request.Request(
        args.url.removesuffix("/v1") + "/tokenize", data=body,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return len(json.load(r)["tokens"]), "/tokenize"
    except Exception:
        return round(len(args.prompt_body) / 4), "chars/4 estimate"


def _cpu_seconds():
    """pid -> (ppid, accumulated cpu seconds, command)."""
    out = subprocess.run(["ps", "-Ao", "pid=,ppid=,time=,comm="],
                         capture_output=True, text=True).stdout
    snap = {}
    for line in out.splitlines():
        f = line.split(None, 3)
        if len(f) != 4:
            continue
        parts = [float(x) for x in f[2].split(":")]
        secs = 0.0
        for x in parts:
            secs = secs * 60 + x
        snap[int(f[0])] = (int(f[1]), secs, f[3])
    return snap


def machine_state(own, ambient=(), window=2.0):
    """Load averages plus the heaviest processes that are not ours.

    Every number in this file is bandwidth- or core-bound, so a second agent
    compiling a Core ML graph on the same laptop moves it as far as a second
    agent on the GPU would. Sampling this into each row is what lets a reader
    tell a measurement from a coincidence.

    CPU share is computed from the DELTA in accumulated cpu-time across a short
    window, not from `ps -o pcpu`: that column is an average over the process's
    whole lifetime, so a service that burned a core for ten minutes and then
    went idle still reports ~100%, and a guard built on it cries wolf forever.

    `ambient` names pids that are accepted background load: they are reported
    separately and stamped into the row, but they do not gate it. That is for a
    steady consumer nobody can remove — an orphaned root-owned compile, say —
    where blocking forever costs more than disclosing the perturbation.
    """
    a = _cpu_seconds()
    time.sleep(window)
    b = _cpu_seconds()

    def ours(pid):  # walk up to init; our own children must not count as noise
        for _ in range(64):
            if pid in own:
                return True
            pid = b.get(pid, (0,))[0]
            if pid <= 1:
                return False
        return False

    busy, amb = [], []
    for pid, (_, secs, comm) in b.items():
        was = a.get(pid)
        pct = (secs - was[1]) / window * 100 if was else 0.0
        if pct < 20 or ours(pid):
            continue
        (amb if pid in ambient else busy).append((pct, f"{comm.rsplit('/', 1)[-1][:40]}[{pid}]"))
    busy.sort(reverse=True)
    load = os.getloadavg()
    return {"load1": round(load[0], 2), "load5": round(load[1], 2),
            "foreign_cpu": round(sum(p for p, _ in busy), 1),
            "top": [[round(p, 1), c] for p, c in busy[:4]],
            "ambient_cpu": round(sum(p for p, _ in amb), 1),
            "ambient": [[round(p, 1), c] for p, c in amb]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="",
                    help=f"start and stop a known server for me: {', '.join(sorted(engines.ENGINES))}")
    ap.add_argument("--model", default=engines.DEFAULT_MODEL,
                    help=f"model family for --engine: {', '.join(engines.MODELS)}")
    ap.add_argument("--api", choices=["lokal", "openai", "cli"],
                    help="talk to a server I am running myself (instead of --engine)")
    ap.add_argument("--url", default="", help="server base url (lokal/openai apis)")
    ap.add_argument("--model-name", default="",
                    help="model name for the openai api, repo/dir for the cli")
    ap.add_argument("--name", default="", help="label to print")
    ap.add_argument("--out", default="", help="append the JSON result line to this file")
    ap.add_argument("--bearer", default="", help="Authorization: Bearer token, if the server wants one")
    ap.add_argument("--bin", default="./target/release/lokal", help="lokal binary for --api cli")
    ap.add_argument("--backend", default="ane", help="lokal backend for --api cli")
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="build the prompt from the corpus at roughly this many tokens (0 = the short synthetic prompt)")
    ap.add_argument("--prompt-chars", type=int, default=0, help="override the character slice directly")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--conc-runs", type=int, default=CONC_RUNS)
    ap.add_argument("--warmup", type=int, default=1,
                    help="untimed requests before measuring (0 to skip — a 32k cli pass is minutes)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-concurrent", action="store_true", help="skip the concurrency metric")
    ap.add_argument("--timeout", type=int, default=1800, help="per-request timeout, seconds")
    ap.add_argument("--tag", default="", help="free-form label recorded in the JSON line")
    ap.add_argument("--max-load", type=float, default=4.0,
                    help="refuse to measure when the 1-minute load average is above this")
    ap.add_argument("--ambient-pid", default="",
                    help="comma-separated pids of accepted background load: reported "
                         "and stamped into the row, but not gated on")
    ap.add_argument("--max-foreign-cpu", type=float, default=150.0,
                    help="refuse to measure when other processes are burning more "
                         "than this much cpu (percent of one core, summed)")
    ap.add_argument("--ctx", type=int, default=0,
                    help="--engine only: server KV context (default: prompt + slack)")
    ap.add_argument("--logdir", default="/tmp", help="--engine only: where the server log goes")
    args = ap.parse_args()
    if bool(args.engine) == bool(args.api):
        raise SystemExit("pass either --engine <name> (managed server) or --api <kind> (your own)")

    server = None
    if args.engine:
        ctx = args.ctx or max(8192, round((args.prompt_tokens or 512) * 1.2) + 1024)
        cmd, ready, flags = engines.resolve(args.engine, args.model, ctx, args.concurrency or 1)
        for k, v in flags.items():
            setattr(args, k.replace("-", "_"), v)
        # The lokal server was started with -m, so its client flags carry no
        # model name; record which checkpoint it was anyway.
        args.model_name = args.model_name or engines.MODELS[args.model]["hf"]
        args.name = args.name or args.engine
        server = engines.start(cmd, ready, os.path.join(args.logdir, f"{args.engine}.server.log"))

    try:
        run(args)
    finally:
        engines.stop(server)


def run(args):
    name = args.name or args.api
    args.prompt_body = build_prompt(args)
    conc = 0 if (args.no_concurrent or args.api == "cli") else args.concurrency

    own = {os.getpid()}
    ambient = {int(x) for x in args.ambient_pid.split(",") if x.strip()}
    before = machine_state(own, ambient)
    # Two gates, because neither alone is enough: load average is the honest
    # sustained-quiet signal but lags a minute behind reality, and instantaneous
    # foreign cpu catches a compute job that just started. A desktop at rest
    # still burns ~60-80% on WindowServer and a browser, so the cpu gate sits
    # well above that and only fires on something doing real work.
    if before["load1"] - before["ambient_cpu"] / 100 > args.max_load \
            or before["foreign_cpu"] > args.max_foreign_cpu:
        raise SystemExit(
            f"machine is busy: load1 {before['load1']} (max {args.max_load}), "
            f"foreign cpu {before['foreign_cpu']}% (max {args.max_foreign_cpu}); "
            f"busiest {before['top']}. Refusing to measure — a contended row is "
            f"worse than a missing one."
        )

    for _ in range(args.warmup):
        call(args, args.max_tokens)  # first request pays caches, JIT, model load

    long, prefill_s, prefill_method, decode_tps, decode_method = measure(args)
    long_gen = int(long["gen"])
    ptok, ptok_src = count_tokens(args, long["prompt_tokens"])
    prefill_tps = ptok / max(prefill_s, 1e-9)

    agg_tps, conc_gen = None, 0
    if conc:
        agg_runs = []
        for _ in range(args.conc_runs):
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=conc) as ex:
                results = list(ex.map(lambda _: call(args, args.max_tokens), range(conc)))
            conc_gen = sum(r["gen"] for r in results)
            agg_runs.append(conc_gen / (time.time() - t0))
        agg_tps = statistics.median(agg_runs)

    conc_txt = f" | {conc}x concurrent {agg_tps:6.1f} tok/s ({conc_gen} gen/run)" if conc else ""
    print(
        f"{name:26s} {ptok:6d} tok prompt | prefill {prefill_s:7.2f}s ({prefill_tps:7.1f} tok/s)"
        f" | decode {decode_tps:6.1f} tok/s ({long_gen} gen){conc_txt}"
    )
    line = json.dumps({
        "name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": args.tag,
        "api": args.api,
        "model": args.model,
        "model_name": args.model_name,
        "prompt_target_tokens": args.prompt_tokens,
        "prompt_chars": len(args.prompt_body),
        "prompt_tokens": ptok,
        "prompt_tokens_source": ptok_src,
        "corpus": CORPUS_LABEL if args.prompt_tokens > 0 else "synthetic-460",
        "prefill_s": round(prefill_s, 3),
        "prefill_tps": round(prefill_tps, 1),
        "prefill_method": prefill_method,
        "decode_tps": round(decode_tps, 1),
        "decode_method": decode_method,
        "single_gen_tokens": long_gen,
        "ane_prefill": long["ane"],
        "concurrency": conc,
        "concurrent4_agg_tps": round(agg_tps, 1) if agg_tps else None,
        "concurrent4_gen_tokens": conc_gen,
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "warmup": args.warmup,
        "machine_before": before,
        "machine_after": machine_state(own, ambient),
        "conc_runs": args.conc_runs if conc else 0,
    })
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
