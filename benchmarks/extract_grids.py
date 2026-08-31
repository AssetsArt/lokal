#!/usr/bin/env python3
"""Emit ggml's i-quant lookup tables into Rust and Metal source.

The grids are thousands of entries — iq1s_grid alone is 2048 u64s. They are
copied by MACHINE, never by hand and never through anyone's context window:
retyping them is impossible to review and re-deriving them is meaningless,
because they are fitted codebooks, not formulas.

Regenerating is idempotent: each output region is delimited by sentinels and
replaced wholesale.
"""
import re, sys, pathlib

SRC = pathlib.Path(sys.argv[1])          # ggml-common.h
RUST_OUT = pathlib.Path(sys.argv[2])     # src/lowmem/iq_grids.rs
METAL = pathlib.Path(sys.argv[3])        # src/gpu/kernels.metal
GGML_REF = sys.argv[4]                   # provenance string

WANT = ["kmask_iq2xs", "ksigns_iq2xs", "iq2xxs_grid", "iq2xs_grid",
        "iq2s_grid", "iq3xxs_grid", "iq3s_grid", "iq1s_grid"]
CTYPE = {"uint8_t": ("u8", "uchar"), "uint32_t": ("u32", "uint"),
         "uint64_t": ("u64", "ulong"), "int8_t": ("i8", "char")}

text = SRC.read_text()
defines = {m.group(1): int(m.group(2))
           for m in re.finditer(r"#define\s+(NGRID_IQ1S)\s+(\d+)", text)}

tables = {}
for m in re.finditer(r"GGML_TABLE_BEGIN\((\w+),\s*(\w+),\s*(\w+)\)(.*?)GGML_TABLE_END\(\)",
                     text, re.S):
    ctype, name, size, body = m.groups()
    if name not in WANT:
        continue
    n = defines.get(size, None) if not size.isdigit() else int(size)
    vals = re.findall(r"0x[0-9a-fA-F]+|-?\d+", body)
    assert n is None or len(vals) == n, f"{name}: got {len(vals)} values, header says {n}"
    tables[name] = (ctype, vals)

missing = [w for w in WANT if w not in tables]
assert not missing, f"tables not found: {missing}"

def wrap(vals, per, indent):
    out = []
    for i in range(0, len(vals), per):
        out.append(indent + ", ".join(vals[i:i + per]) + ",")
    return "\n".join(out)

# ---- Rust ----
r = [f"//! ggml i-quant lookup tables, emitted by benchmarks/extract_grids.py.",
     f"//! Source: {GGML_REF}. DO NOT EDIT BY HAND — regenerate.",
     f"//! These are fitted codebooks, not formulas: they cannot be re-derived,",
     f"//! only copied, which is why a machine copies them.", ""]
for name, (ctype, vals) in tables.items():
    rt = CTYPE[ctype][0]
    sfx = "" if rt in ("u8", "i8") else ""
    r.append(f"#[rustfmt::skip]")
    r.append(f"pub(crate) const {name.upper()}: [{rt}; {len(vals)}] = [")
    r.append(wrap(vals, 8 if rt == "u64" else 16, "    "))
    r.append("];\n")
RUST_OUT.write_text("\n".join(r))

# ---- Metal ----
m_lines = [f"// BEGIN GENERATED IQ TABLES (benchmarks/extract_grids.py, {GGML_REF})",
           "// Fitted codebooks copied verbatim from ggml-common.h. Do not edit."]
for name, (ctype, vals) in tables.items():
    mt = CTYPE[ctype][1]
    m_lines.append(f"constant {mt} lm_{name}[{len(vals)}] = {{")
    m_lines.append(wrap(vals, 8 if mt == "ulong" else 16, "    ").rstrip(","))
    m_lines.append("};")
m_lines.append("// END GENERATED IQ TABLES")
block = "\n".join(m_lines)

mt_text = METAL.read_text()
sentinel_re = re.compile(r"// BEGIN GENERATED IQ TABLES.*?// END GENERATED IQ TABLES", re.S)
if sentinel_re.search(mt_text):
    mt_text = sentinel_re.sub(block, mt_text)
else:
    anchor = "// ggml's kvalues_iq4nl (ggml-common.h), verbatim"
    assert anchor in mt_text, "metal anchor not found"
    mt_text = mt_text.replace(anchor, block + "\n\n" + anchor, 1)
METAL.write_text(mt_text)

print(f"emitted {len(tables)} tables: " +
      ", ".join(f"{n}[{len(v)}]" for n, (_, v) in tables.items()))
