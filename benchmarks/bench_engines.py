# bench_engines.py — compare lokal against other local inference servers.
#
# Measures three things against an already-running server, greedy (temperature 0):
#   prefill   — wall time of a ~450-token prompt with max_tokens=1 (TTFT proxy)
#   decode    — (tokens_long - 1) / (wall_long - wall_prefill), i.e. the marginal
#               cost per generated token, which cancels out prompt processing
#   4x concurrent — aggregate generated tokens/sec with 4 simultaneous requests
#
# Every engine applies its own chat template, so prompt token counts differ by a
# few tokens across engines; the workload is identical in substance. Completion
# lengths are reported so an early EOS is visible instead of skewing the numbers.
#
# Each request starts with a unique nonce so server-side prompt caches (llama.cpp
# reuses the KV of a repeated prefix, for example) cannot answer from cache — the
# benchmark measures compute, not caching.
#
# Usage (server must already be running):
#   python3 benchmarks/bench_engines.py --api lokal  --url http://127.0.0.1:8080 --name "lokal (metal)"
#   python3 benchmarks/bench_engines.py --api openai --url http://127.0.0.1:8081/v1 \
#       --model SmolLM2-135M-Instruct --name "llama.cpp" --out benchmarks/results.jsonl
#
# --api lokal  → POST {url}/generate            (lokal's native endpoint)
# --api openai → POST {url}/chat/completions    (llama.cpp, oMLX, vLLM, ...)
# --out        → append the JSON result line to this file

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the river runs "
    "quietly under the old stone bridge near the edge of town. "
)
PROMPT = SENTENCE * 18 + "Retell everything above as one long detailed story. Do not stop."
MAX_TOKENS = 128
RUNS = 5       # per single-request metric; the median is reported
CONC_RUNS = 3  # per concurrency metric


def call(args, max_tokens):
    prompt = f"(session {time.time_ns()}) {PROMPT}"
    if args.api == "lokal":
        body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "chat": True}
        url = f"{args.url}/generate"
    else:
        body = {
            "model": args.model,
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
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    wall = time.time() - t0
    if args.api == "lokal":
        gen = out["generated_tokens"]
    else:
        gen = out["usage"]["completion_tokens"]
    return wall, gen


def median_run(args, max_tokens):
    walls, gens = [], []
    for _ in range(RUNS):
        wall, gen = call(args, max_tokens)
        walls.append(wall)
        gens.append(gen)
    return statistics.median(walls), statistics.median(gens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", choices=["lokal", "openai"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="", help="model name for the openai api")
    ap.add_argument("--name", default="", help="label to print")
    ap.add_argument("--out", default="", help="append the JSON result line to this file")
    ap.add_argument("--bearer", default="", help="Authorization: Bearer token, if the server wants one")
    args = ap.parse_args()
    name = args.name or args.api

    call(args, MAX_TOKENS)  # warmup: first request pays caches, JIT, model load

    prefill_wall, _ = median_run(args, 1)
    long_wall, long_gen = median_run(args, MAX_TOKENS)
    decode_tps = (long_gen - 1) / max(long_wall - prefill_wall, 1e-9)

    agg_runs, conc_gen = [], 0
    for _ in range(CONC_RUNS):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(lambda _: call(args, MAX_TOKENS), range(4)))
        conc_gen = sum(g for _, g in results)
        agg_runs.append(conc_gen / (time.time() - t0))
    agg_tps = statistics.median(agg_runs)

    print(
        f"{name:24s} prefill {prefill_wall:6.2f}s | decode {decode_tps:6.1f} tok/s"
        f" ({long_gen} gen) | 4x concurrent {agg_tps:6.1f} tok/s ({conc_gen} gen/run)"
    )
    line = json.dumps({
        "name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prefill_s": round(prefill_wall, 3),
        "decode_tps": round(decode_tps, 1),
        "single_gen_tokens": long_gen,
        "concurrent4_agg_tps": round(agg_tps, 1),
        "concurrent4_gen_tokens": conc_gen,
        "runs": RUNS,
        "conc_runs": CONC_RUNS,
    })
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
