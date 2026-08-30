# run_longctx.py — drive the long-context matrix, one engine server at a time.
#
# 16 GB of unified memory does not hold three inference servers and a 32k KV
# cache at once, and a second engine idling on the GPU skews the one being
# timed. So this script owns the lifecycle: start a server, poll it until it
# answers, run bench_engines.py against it for every prompt size, kill it, move
# to the next engine. lokal above 8192 tokens is the exception — serve mode caps
# a pooled slot there (src/batch.rs POOL_SEQ_CAP), so those sizes go through the
# CLI path instead, which needs no server at all.
#
#   python3 benchmarks/run_longctx.py --engines lokal-hybrid-cli,llamacpp,omlx \
#       --sizes 2000,6000,10000,16000,24000,32000 --ctx 34816
#
# Engine names, server flags and model packagings all live in engines.py, which
# bench_engines.py --engine uses too — one row and a whole sweep can never end up
# talking to differently-configured servers.
#
# Every run appends one JSON line to --out; nothing is aggregated here. Build the
# table from results.jsonl afterwards with summarize_longctx.py.

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engines  # noqa: E402

BENCH = os.path.join(HERE, "bench_engines.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", required=True,
                    help="comma-separated engine keys from engines.py")
    ap.add_argument("--model", default="qwen",
                    help="model family (engines.py MODELS): the long-context matrix is qwen")
    ap.add_argument("--sizes", required=True, help="comma-separated --prompt-tokens values")
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    ap.add_argument("--tag", default="longctx-baseline")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=34816, help="server KV context, tokens")
    ap.add_argument("--parallel", type=int, default=1, help="server slots")
    ap.add_argument("--concurrency", type=int, default=0, help="0 = single-stream only")
    ap.add_argument("--conc-runs", type=int, default=3)
    ap.add_argument("--logdir", default="/tmp")
    ap.add_argument("--ambient-pid", default="",
                    help="passed through to bench_engines.py: accepted background load")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    for key in args.engines.split(","):
        cmd, ready, _ = engines.resolve(key, args.model, args.ctx, args.parallel)
        print(f"\n=== {key}", flush=True)
        proc = None
        try:
            proc = engines.start(cmd, ready, os.path.join(args.logdir, f"{key}.server.log"))
            for size in sizes:
                cmd = [sys.executable, BENCH, *engines.bench_flags(key, args.model),
                       "--prompt-tokens", str(size), "--name", f"{key} @{size}",
                       "--out", args.out, "--tag", args.tag,
                       "--runs", str(args.runs), "--warmup", str(args.warmup)]
                if args.ambient_pid:
                    cmd += ["--ambient-pid", args.ambient_pid]
                if args.concurrency:
                    cmd += ["--concurrency", str(args.concurrency),
                            "--conc-runs", str(args.conc_runs)]
                else:
                    cmd += ["--no-concurrent"]
                print(f"  -> {key} @{size}", flush=True)
                t0 = time.time()
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    # One size failing (a context limit, an OOM) must not cost
                    # the whole sweep — record the refusal and keep going.
                    print(f"  FAILED @{size} after {time.time()-t0:.0f}s:\n"
                          f"{r.stdout[-800:]}{r.stderr[-1500:]}", flush=True)
                    with open(args.out, "a") as f:
                        f.write(json.dumps({
                            "name": f"{key} @{size}", "tag": args.tag,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "prompt_target_tokens": size, "unsupported": True,
                            "concurrency": args.concurrency,
                            "error": (r.stderr or r.stdout)[-400:],
                        }) + "\n")
                    continue
                print(r.stdout.strip().splitlines()[0], flush=True)
        finally:
            engines.stop(proc)


if __name__ == "__main__":
    main()
