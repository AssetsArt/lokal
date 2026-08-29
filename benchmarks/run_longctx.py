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
#   python3 benchmarks/run_longctx.py --engines lokal-ane-cli,llamacpp,omlx \
#       --sizes 2000,6000,10000,16000,24000,32000 --ctx 34816
#
# Every run appends one JSON line to --out; nothing is aggregated here. Build the
# table from results.jsonl afterwards with summarize_longctx.py.

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BENCH = os.path.join(HERE, "bench_engines.py")
LOKAL = os.path.join(REPO, "target/release/lokal")

QWEN_HF = "Qwen/Qwen2.5-0.5B-Instruct"
GGUF = os.path.expanduser("~/.cache/gguf/qwen2.5-0.5b-instruct-fp16.gguf")
# oMLX names a model after its directory, not the repo it came from.
OMLX_MODEL = "Qwen2.5-0.5B-Instruct-bf16"

# Each engine: how to start its server ("server": None for the CLI path), how to
# tell it is up, and the bench_engines.py flags that talk to it. {ctx} and
# {par} are filled from --ctx / --parallel: llama.cpp splits one KV allocation
# across its parallel slots, so a 4-way run at 10k needs 4x the context a
# single-stream run at 10k does.
ENGINES = {
    "lokal-ane-cli": {
        "server": None,
        "bench": ["--api", "cli", "--bin", LOKAL, "--model", QWEN_HF, "--backend", "ane"],
    },
    "lokal-metal-cli": {
        "server": None,
        "bench": ["--api", "cli", "--bin", LOKAL, "--model", QWEN_HF, "--backend", "metal"],
    },
    "lokal-ane-http": {
        "server": [LOKAL, "serve", "-b", "ane", "-m", QWEN_HF,
                   "--port", "8080", "--max-concurrent", "{par}"],
        "ready": ("POST", "http://127.0.0.1:8080/generate",
                  {"prompt": "hi", "max_tokens": 1, "temperature": 0}),
        "bench": ["--api", "lokal", "--url", "http://127.0.0.1:8080"],
    },
    "lokal-metal-http": {
        "server": [LOKAL, "serve", "-b", "metal", "-m", QWEN_HF,
                   "--port", "8080", "--max-concurrent", "{par}"],
        "ready": ("POST", "http://127.0.0.1:8080/generate",
                  {"prompt": "hi", "max_tokens": 1, "temperature": 0}),
        "bench": ["--api", "lokal", "--url", "http://127.0.0.1:8080"],
    },
    "llamacpp": {
        "server": ["llama-server", "-m", GGUF, "--port", "8081", "-ngl", "99",
                   "-c", "{ctx}", "--parallel", "{par}", "--no-webui"],
        "ready": ("GET", "http://127.0.0.1:8081/health", None),
        "bench": ["--api", "openai", "--url", "http://127.0.0.1:8081/v1",
                  "--model", "qwen2.5-0.5b-instruct"],
    },
    "omlx": {
        "server": ["omlx", "serve", "--model-dir", os.path.expanduser("~/.omlx/models"),
                   "--port", "8082", "--api-key", "bench", "--log-level", "warning",
                   "--max-concurrent-requests", "{par}"],
        "ready": ("GET", "http://127.0.0.1:8082/v1/models", None),
        "bench": ["--api", "openai", "--url", "http://127.0.0.1:8082/v1",
                  "--model", OMLX_MODEL, "--bearer", "bench"],
    },
}


def probe(ready):
    method, url, payload = ready
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"authorization": "Bearer bench", "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status == 200


def wait_ready(spec, proc, timeout):
    """Poll until the server answers, or it dies, or we run out of patience."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early with code {proc.returncode}")
        try:
            if probe(spec["ready"]):
                return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"server never became ready at {spec['ready'][1]}")


def start(spec, log_path, ctx, par):
    if not spec["server"]:
        return None
    cmd = [a.format(ctx=ctx, par=par) for a in spec["server"]]
    print(f"  starting: {shlex.join(cmd)}", flush=True)
    log = open(log_path, "w")
    # New session: llama-server and omlx both spawn children, and killing the
    # whole process group is the only way to be sure nothing is left on the GPU.
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    wait_ready(spec, proc, timeout=600)
    print("  server ready", flush=True)
    return proc


def stop(proc):
    if proc is None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    for _ in range(50):
        if proc.poll() is not None:
            break
        time.sleep(0.2)
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    time.sleep(3)  # let the GPU allocations actually go back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", required=True, help="comma-separated keys from ENGINES")
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
        spec = ENGINES[key]
        print(f"\n=== {key}", flush=True)
        proc = None
        try:
            proc = start(spec, os.path.join(args.logdir, f"{key}.server.log"),
                         args.ctx, args.parallel)
            for size in sizes:
                cmd = [sys.executable, BENCH, *spec["bench"],
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
            stop(proc)


if __name__ == "__main__":
    main()
