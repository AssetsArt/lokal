# summarize_longctx.py — turn results.jsonl into the markdown tables in README.md.
#
# results.jsonl is append-only: every run of bench_engines.py adds a line, and a
# re-measured cell simply appends a newer one. This reads the file, keeps the
# LAST line per (tag, engine, prompt size), and prints the tables. Nothing here
# computes a metric — it only arranges what was measured.
#
#   python3 benchmarks/summarize_longctx.py --tag longctx-baseline
#   python3 benchmarks/summarize_longctx.py --tag longctx-conc --metric concurrent

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Display order and pretty names; anything else falls through in file order.
ORDER = [
    ("lokal-ane-cli", "lokal `-b hybrid` (cli)"),
    ("lokal-metal-cli", "lokal `-b metal` (cli)"),
    ("lokal-ane-http", "lokal `-b hybrid` (serve)"),
    ("lokal-metal-http", "lokal `-b metal` (serve)"),
    ("llamacpp", "llama.cpp"),
    ("omlx", "oMLX"),
]


def load(path, tag):
    """Last row wins per (engine, size) — a re-run supersedes what it replaces."""
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if tag and r.get("tag") != tag:
                continue
            engine = r["name"].split(" @")[0]
            rows[(engine, r.get("prompt_target_tokens", 0))] = r
    return rows


def label(size):
    return f"{size // 1000}k" if size >= 1000 and size % 1000 == 0 else str(size)


def cell(r, metric):
    if r is None:
        return "—"
    if r.get("unsupported"):
        return "unsupported"
    if metric == "prefill":
        return f"{r['prefill_s']:.2f} s / {r['prefill_tps']:.0f} tok/s"
    if metric == "decode":
        gen = r["single_gen_tokens"]
        note = "" if gen >= r.get("max_tokens", 128) else f" ({gen} gen)"
        return f"{r['decode_tps']:.0f} tok/s{note}"
    if metric == "tokens":
        return str(r["prompt_tokens"])
    if metric == "split":
        # Where the prompt was actually processed. The ANE only ever covers the
        # first s+p positions of its windowed graph; everything past that is a
        # Metal prefill, and that tail is what grows with context.
        a = r.get("ane_prefill")
        if not a:
            return "all Metal"
        tail_tok = r["prompt_tokens"] - a["tokens"]
        tail_s = r["prefill_s"] - a["secs"]
        if tail_tok <= 0:
            return f"ANE {a['tokens']} tok / {a['secs']:.2f} s (no tail)"
        return (f"ANE {a['tokens']} tok / {a['secs']:.2f} s + "
                f"Metal {tail_tok} tok / {tail_s:.2f} s")
    return f"{r['concurrent4_agg_tps']:.0f} tok/s" if r.get("concurrent4_agg_tps") else "—"


def table(rows, engines, sizes, metric):
    head = "| prompt | " + " | ".join(name for _, name in engines) + " |"
    rule = "|---" * (len(engines) + 1) + "|"
    body = [
        "| " + label(size) + " | "
        + " | ".join(cell(rows.get((key, size)), metric) for key, _ in engines) + " |"
        for size in sizes
    ]
    return "\n".join([head, rule, *body])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(HERE, "results.jsonl"))
    ap.add_argument("--tag", default="longctx-baseline")
    ap.add_argument("--metric", default="", help="prefill | decode | tokens | concurrent (default: all)")
    args = ap.parse_args()

    rows = load(args.file, args.tag)
    if not rows:
        raise SystemExit(f"no rows tagged {args.tag!r} in {args.file}")
    present = {k for k, _ in rows}
    engines = [(k, n) for k, n in ORDER if k in present]
    engines += [(k, k) for k in sorted(present) if k not in dict(ORDER)]
    sizes = sorted({s for _, s in rows})

    titles = {"prefill": "Prefill (wall / tokens per second)",
              "decode": "Decode (marginal tokens per second)",
              "tokens": "Prompt length as each engine tokenized it",
              "split": "Where lokal prefilled it (ANE window vs Metal tail)",
              "concurrent": "Concurrent aggregate throughput"}
    for metric in ([args.metric] if args.metric else ["prefill", "decode", "tokens"]):
        print(f"\n**{titles[metric]}**\n")
        print(table(rows, engines, sizes, metric))


if __name__ == "__main__":
    main()
