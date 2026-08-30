# engines.py — who the benchmark can talk to, and how to bring them up.
#
# One registry shared by both drivers, because an engine's server flags and its
# client flags have to agree and keeping two copies of that pairing is how a
# benchmark ends up measuring a differently-configured server than it reports.
#
#   bench_engines.py --engine lokal-hybrid   # one row, server managed for you
#   run_longctx.py --engines llamacpp ...    # a whole sweep, same registry
#
# A spec has three parts: how to START the server (None for the CLI path, which
# has no server), how to know it is READY, and the client flags that TALK to it.

import json
import os
import shlex
import signal
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOKAL = os.path.join(REPO, "target/release/lokal")

# Each model needs its weights in three packagings, one per engine family. The
# GGUF and MLX copies are downloaded separately (see README); a missing file is
# reported when that engine is asked for, not at import.
MODELS = {
    "smol": {
        "hf": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "gguf": os.path.expanduser(
            "~/.cache/huggingface/hub/models--unsloth--SmolLM2-135M-Instruct-GGUF/"
            "snapshots/9e6855bc4be717fca1ef21360a1db4b29d5c559a/SmolLM2-135M-Instruct-F16.gguf"),
        "omlx": "SmolLM2-135M-Instruct",
        "openai": "smollm2-135m-instruct",
    },
    "qwen": {
        "hf": "Qwen/Qwen2.5-0.5B-Instruct",
        "gguf": os.path.expanduser("~/.cache/gguf/qwen2.5-0.5b-instruct-fp16.gguf"),
        "omlx": "Qwen2.5-0.5B-Instruct-bf16",
        "openai": "qwen2.5-0.5b-instruct",
    },
}
DEFAULT_MODEL = "smol"

OMLX_DIR = os.path.expanduser("~/.omlx/models")
PORTS = {"lokal": 8080, "llamacpp": 8081, "omlx": 8082}


def _lokal_http(backend):
    return {
        "server": lambda m, ctx, par: [
            LOKAL, "serve", "-b", backend, "-m", m["hf"],
            "--port", str(PORTS["lokal"]), "--max-concurrent", str(par)],
        "ready": lambda m: ("POST", f"http://127.0.0.1:{PORTS['lokal']}/generate",
                            {"prompt": "hi", "max_tokens": 1, "temperature": 0}),
        "bench": lambda m: {"api": "lokal", "url": f"http://127.0.0.1:{PORTS['lokal']}"},
    }


def _lokal_cli(backend):
    # No server: the binary runs once per request. This is the only way to
    # measure prompts past serve mode's pooled-slot cap (src/batch.rs).
    return {
        "server": None,
        "ready": None,
        "bench": lambda m: {"api": "cli", "bin": LOKAL, "backend": backend,
                            "model-name": m["hf"]},
    }


ENGINES = {
    "lokal-metal": _lokal_http("metal"),
    "lokal-hybrid": _lokal_http("hybrid"),
    "lokal-metal-cli": _lokal_cli("metal"),
    "lokal-hybrid-cli": _lokal_cli("hybrid"),
    "llamacpp": {
        # llama.cpp splits one KV allocation across its parallel slots, so a
        # 4-way run at N tokens needs 4x the context a single-stream run does.
        "server": lambda m, ctx, par: [
            "llama-server", "-m", m["gguf"], "--port", str(PORTS["llamacpp"]),
            "-ngl", "99", "-c", str(ctx), "--parallel", str(par), "--no-webui"],
        "ready": lambda m: ("GET", f"http://127.0.0.1:{PORTS['llamacpp']}/health", None),
        "bench": lambda m: {"api": "openai", "url": f"http://127.0.0.1:{PORTS['llamacpp']}/v1",
                            "model-name": m["openai"]},
    },
    "omlx": {
        "server": lambda m, ctx, par: [
            "omlx", "serve", "--model-dir", OMLX_DIR, "--port", str(PORTS["omlx"]),
            "--api-key", "bench", "--log-level", "warning",
            "--max-concurrent-requests", str(par)],
        "ready": lambda m: ("GET", f"http://127.0.0.1:{PORTS['omlx']}/v1/models", None),
        # oMLX names a model after its directory, not the repo it came from.
        "bench": lambda m: {"api": "openai", "url": f"http://127.0.0.1:{PORTS['omlx']}/v1",
                            "model-name": m["omlx"], "bearer": "bench"},
    },
}
# Kept so older command lines and saved benchmark scripts keep working.
ENGINES["lokal-ane"] = ENGINES["lokal-hybrid"]
ENGINES["lokal-ane-cli"] = ENGINES["lokal-hybrid-cli"]
ENGINES["lokal-ane-http"] = ENGINES["lokal-hybrid"]
ENGINES["lokal-hybrid-http"] = ENGINES["lokal-hybrid"]
ENGINES["lokal-metal-http"] = ENGINES["lokal-metal"]


def resolve(key, model_key=DEFAULT_MODEL, ctx=8192, par=1):
    """(server command or None, ready probe or None, client flags) for one engine."""
    if key not in ENGINES:
        raise SystemExit(f"unknown engine {key!r}; known: {', '.join(sorted(ENGINES))}")
    if model_key not in MODELS:
        raise SystemExit(f"unknown model {model_key!r}; known: {', '.join(MODELS)}")
    spec, model = ENGINES[key], MODELS[model_key]
    cmd = spec["server"](model, ctx, par) if spec["server"] else None
    if cmd and cmd[0] == "llama-server" and not os.path.exists(model["gguf"]):
        raise SystemExit(f"llama.cpp needs a GGUF for {model_key} at {model['gguf']}")
    return cmd, (spec["ready"](model) if spec["ready"] else None), spec["bench"](model)


def bench_flags(key, model_key=DEFAULT_MODEL):
    """Client flags as a command line, for drivers that shell out to the bench."""
    _, _, bench = resolve(key, model_key)
    out = []
    for k, v in bench.items():
        out += [f"--{k}", str(v)]
    return out


def _probe(ready):
    method, url, payload = ready
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"authorization": "Bearer bench", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status == 200


def start(cmd, ready, log_path, timeout=600, quiet=False):
    """Launch a server and wait until it answers. Returns None for CLI engines."""
    if not cmd:
        return None
    if not quiet:
        print(f"  starting: {shlex.join(cmd)}", flush=True)
    log = open(log_path, "w")
    # New session: llama-server and omlx both spawn children, and killing the
    # whole process group is the only way to be sure nothing is left on the GPU.
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with code {proc.returncode}; log: {log_path}")
        try:
            if _probe(ready):
                if not quiet:
                    print("  server ready", flush=True)
                return proc
        except Exception:
            time.sleep(2)
    stop(proc)
    raise SystemExit(f"server never became ready at {ready[1]}; log: {log_path}")


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
