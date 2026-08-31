//! GGUF checkpoint reading (lane gguf-loader, rehomed by gguf-unify): the
//! v2/v3 container parser — mmap, header walk, metadata KVs, tensor tables —
//! plus the file-level size accounting the routing guards read. Arch mapping
//! lives in arch.rs, the tokenizer build in tokenizer.rs, the dequant layer
//! in dequant.rs.
//!
//! Format facts (magic/version/kv/tensor-info order, type ids, key and tensor
//! names) are transcribed from a read-only llama.cpp checkout
//! (ggml/include/gguf.h, ggml/src/gguf.cpp), not from memory. Little-endian
//! files only.

use crate::gguf::dequant::{GgmlType, GgufTensor};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::Path;

// ---------- metadata values ----------

/// One GGUF metadata value. Arrays are kept as typed vectors — the tokenizer
/// arrays run to 150k+ entries, so they are parsed once, never re-walked.
pub enum GgufValue {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    ArrStr(Vec<String>),
    ArrI64(Vec<i64>),
    ArrF64(Vec<f64>),
}

impl GgufValue {
    fn kind(&self) -> &'static str {
        match self {
            Self::U64(_) => "uint",
            Self::I64(_) => "int",
            Self::F64(_) => "float",
            Self::Bool(_) => "bool",
            Self::Str(_) => "string",
            Self::ArrStr(_) => "string array",
            Self::ArrI64(_) => "int array",
            Self::ArrF64(_) => "float array",
        }
    }
}

// ---------- bounds-checked little-endian cursor ----------

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> crate::Result<&'a [u8]> {
        let end = self.p.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| {
            format!("GGUF truncated: need {n} bytes at offset {}, file has {}", self.p, self.b.len())
        })?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> crate::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> crate::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    /// A GGUF string: u64 length prefix + raw bytes (not NUL-terminated).
    fn str(&mut self) -> crate::Result<String> {
        let n = self.u64()? as usize;
        if n > 1 << 24 {
            return Err(format!("GGUF string of {n} bytes at offset {} — corrupt length", self.p).into());
        }
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    /// One metadata value of GGUF type id `ty` (enum gguf_type in gguf.h).
    /// Scalars widen to 64-bit; nested arrays are rejected (nothing writes them).
    fn value(&mut self, ty: u32, key: &str) -> crate::Result<GgufValue> {
        Ok(match ty {
            0 => GgufValue::U64(self.take(1)?[0] as u64),
            1 => GgufValue::I64(self.take(1)?[0] as i8 as i64),
            2 => GgufValue::U64(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => GgufValue::I64(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => GgufValue::U64(self.u32()? as u64),
            5 => GgufValue::I64(self.u32()? as i32 as i64),
            6 => GgufValue::F64(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => GgufValue::Bool(self.take(1)?[0] != 0),
            8 => GgufValue::Str(self.str()?),
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(self.u64()? as i64),
            12 => GgufValue::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                if n > 1 << 26 {
                    return Err(format!("GGUF array {key}: {n} entries — corrupt length").into());
                }
                match et {
                    8 => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(self.str()?);
                        }
                        GgufValue::ArrStr(v)
                    }
                    6 | 12 => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(match self.value(et, key)? {
                                GgufValue::F64(x) => x,
                                _ => unreachable!(),
                            });
                        }
                        GgufValue::ArrF64(v)
                    }
                    // bools ride the int rail as 0/1 — llama.cpp writes
                    // per-layer flag arrays (qwen35.attention.recurrent_layers)
                    // as GGUF bool arrays.
                    0..=5 | 7 | 10 | 11 => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(match self.value(et, key)? {
                                GgufValue::U64(x) => x as i64,
                                GgufValue::I64(x) => x,
                                GgufValue::Bool(x) => x as i64,
                                _ => unreachable!(),
                            });
                        }
                        GgufValue::ArrI64(v)
                    }
                    other => return Err(format!("GGUF array {key}: element type {other} unsupported").into()),
                }
            }
            other => return Err(format!("GGUF key {key}: value type {other} unsupported").into()),
        })
    }
}

// ---------- the parsed file ----------

pub(crate) struct TensorInfo {
    pub(crate) name: String,
    /// ROW-MAJOR: GGUF's fastest-varying-first `ne` is reversed at parse, so
    /// dims == [rows, cols] for a 2-D weight (matching TensorMeta.shape).
    dims: Vec<usize>,
    ty: GgmlType,
    /// Absolute byte range inside the mmap.
    range: std::ops::Range<usize>,
}

pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub(crate) kv: HashMap<String, GgufValue>,
    pub(crate) infos: Vec<TensorInfo>,
    by_name: HashMap<String, usize>,
    data_start: usize,
}

impl GgufFile {
    pub fn open(path: &Path) -> crate::Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut rd = Rd { b: &mmap, p: 0 };

        if rd.take(4)? != b"GGUF" {
            return Err(format!("{} is not a GGUF file (bad magic)", path.display()).into());
        }
        let version = rd.u32()?;
        if version.swap_bytes() == 2 || version.swap_bytes() == 3 {
            return Err("big-endian GGUF files are not supported — re-export little-endian".into());
        }
        if !(2..=3).contains(&version) {
            return Err(format!("GGUF version {version} unsupported (v2 and v3 only)").into());
        }
        let n_tensors = rd.u64()? as usize;
        let n_kv = rd.u64()? as usize;

        let mut kv = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = rd.str()?;
            let ty = rd.u32()?;
            let val = rd.value(ty, &key)?;
            kv.insert(key, val);
        }

        // Tensor infos: name, n_dims (u32), ne[] (u64 each, fastest first),
        // ggml type (u32), offset (u64, relative to the data section).
        let mut raw = Vec::with_capacity(n_tensors);
        // type name -> (tensor count, first tensor seen with it)
        let mut unsupported: std::collections::BTreeMap<&'static str, (usize, String)> =
            std::collections::BTreeMap::new();
        for _ in 0..n_tensors {
            let name = rd.str()?;
            let n_dims = rd.u32()? as usize;
            if n_dims == 0 || n_dims > 4 {
                return Err(format!("tensor {name}: {n_dims} dimensions").into());
            }
            let mut ne = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                ne.push(rd.u64()? as usize);
            }
            let ty_id = rd.u32()?;
            let offset = rd.u64()? as usize;
            // Unsupported types are COLLECTED, not raised here. Failing on the
            // first offending tensor made the human discover this file's
            // requirements one re-download at a time; a mixed low-bit file can
            // carry four or five types at once, and they should learn all of
            // them from one run. Every field for this tensor has been read, so
            // skipping it keeps the reader in sync.
            let ty = match GgmlType::from_gguf(ty_id) {
                Ok(t) => t,
                Err(tyname) => {
                    unsupported.entry(tyname).or_insert((0usize, name)).0 += 1;
                    continue;
                }
            };
            if ne[0] % ty.blk_elems() != 0 {
                return Err(format!(
                    "tensor {name}: row length {} is not a whole number of {:?} blocks",
                    ne[0], ty
                )
                .into());
            }
            raw.push((name, ne, ty, offset));
        }

        // One sweep, one verdict: every type this build cannot run, with how
        // many tensors carry it and one example each, so a mixed low-bit file
        // costs the reader one run rather than one re-download per type.
        if !unsupported.is_empty() {
            let n_types = unsupported.len();
            let total: usize = unsupported.values().map(|(c, _)| c).sum();
            let detail = unsupported
                .iter()
                .map(|(ty, (count, example))| {
                    let plural = if *count == 1 { "tensor" } else { "tensors" };
                    format!("{ty} ({count} {plural}, e.g. {example})")
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "this checkpoint uses {n_types} quantization type(s) lokal does not run, \
                 across {total} of {n_tensors} tensors: {detail}. \
                 Re-download the model as Q4_K_M or Q8_0."
            )
            .into());
        }

        // Data section starts at the next alignment boundary after the header;
        // each tensor's offset is relative to it. Alignment-padding mistakes
        // here parse "successfully" into garbage — the llama-cli cross-check
        // gate exists for exactly that, and every range is bounds-checked.
        let align = match kv.get("general.alignment") {
            Some(GgufValue::U64(a)) if *a > 0 && a.is_power_of_two() => *a as usize,
            None => 32,
            Some(v) => return Err(format!("general.alignment is invalid ({})", v.kind()).into()),
        };
        let data_start = rd.p.next_multiple_of(align);

        let mut infos = Vec::with_capacity(raw.len());
        let mut by_name = HashMap::with_capacity(raw.len());
        for (name, ne, ty, offset) in raw {
            let n_elems: usize = ne.iter().product();
            let nbytes = ty.row_bytes(n_elems);
            let start = data_start
                .checked_add(offset)
                .ok_or_else(|| format!("tensor {name}: offset overflow"))?;
            let end = start
                .checked_add(nbytes)
                .filter(|&e| e <= mmap.len())
                .ok_or_else(|| {
                    format!(
                        "tensor {name}: {nbytes} bytes at data offset {offset} run past the file"
                    )
                })?;
            let mut dims = ne;
            dims.reverse(); // fastest-first → row-major
            if by_name.insert(name.clone(), infos.len()).is_some() {
                return Err(format!("tensor {name} appears twice").into());
            }
            infos.push(TensorInfo { name, dims, ty, range: start..end });
        }

        Ok(Self { mmap, version, kv, infos, by_name, data_start })
    }

    pub fn n_tensors(&self) -> usize {
        self.infos.len()
    }

    /// One TSV line per tensor — name, dims (row-major), type, byte size, and
    /// data-relative offset. The llama-gguf cross-check gate diffs the last
    /// three columns against the reference reader, which is what catches an
    /// off-by-alignment parse that "succeeds" into garbage.
    pub fn dump_tsv(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for i in &self.infos {
            let dims: Vec<String> = i.dims.iter().map(|d| d.to_string()).collect();
            let _ = writeln!(
                out,
                "{}\t{}\t{:?}\t{}\t{}",
                i.name,
                dims.join("x"),
                i.ty,
                i.range.len(),
                i.range.start - self.data_start
            );
        }
        out
    }

    pub fn tensors(&self) -> impl Iterator<Item = GgufTensor<'_>> {
        self.infos.iter().map(|i| GgufTensor {
            name: i.name.clone(),
            dims: i.dims.clone(),
            ty: i.ty,
            data: &self.mmap[i.range.clone()],
        })
    }

    pub fn tensor(&self, name: &str) -> crate::Result<GgufTensor<'_>> {
        let i = self
            .by_name
            .get(name)
            .map(|&i| &self.infos[i])
            .ok_or_else(|| format!("tensor {name} not in the GGUF file"))?;
        Ok(GgufTensor {
            name: i.name.clone(),
            dims: i.dims.clone(),
            ty: i.ty,
            data: &self.mmap[i.range.clone()],
        })
    }

    // -- typed metadata accessors; errors name the key and what was found --

    fn kv(&self, key: &str) -> crate::Result<&GgufValue> {
        self.kv.get(key).ok_or_else(|| format!("GGUF metadata key {key} is missing").into())
    }
    pub fn get_usize(&self, key: &str) -> crate::Result<usize> {
        match self.kv(key)? {
            GgufValue::U64(x) => Ok(*x as usize),
            GgufValue::I64(x) if *x >= 0 => Ok(*x as usize),
            v => Err(format!("GGUF key {key} is a {}, expected an integer", v.kind()).into()),
        }
    }
    pub fn get_f32(&self, key: &str) -> crate::Result<f32> {
        match self.kv(key)? {
            GgufValue::F64(x) => Ok(*x as f32),
            v => Err(format!("GGUF key {key} is a {}, expected a float", v.kind()).into()),
        }
    }
    pub fn get_str(&self, key: &str) -> crate::Result<&str> {
        match self.kv(key)? {
            GgufValue::Str(s) => Ok(s),
            v => Err(format!("GGUF key {key} is a {}, expected a string", v.kind()).into()),
        }
    }
    pub fn get_arr_str(&self, key: &str) -> crate::Result<&[String]> {
        match self.kv(key)? {
            GgufValue::ArrStr(v) => Ok(v),
            v => Err(format!("GGUF key {key} is a {}, expected a string array", v.kind()).into()),
        }
    }
    pub fn get_arr_i64(&self, key: &str) -> crate::Result<&[i64]> {
        match self.kv(key)? {
            GgufValue::ArrI64(v) => Ok(v),
            v => Err(format!("GGUF key {key} is a {}, expected an int array", v.kind()).into()),
        }
    }
    pub fn has_key(&self, key: &str) -> bool {
        self.kv.contains_key(key)
    }
}

// ---------- architecture metadata (D4) ----------

/// Every tensor's f32 expansion, summed — the honest RAM cost of running a
/// quantized file on the full-materialization backends. Note this can
/// legitimately exceed the safetensors twin's params x 4: some GGUFs
/// materialize a duplicate output.weight where safetensors ties it to the
/// embedding table, and we materialize what the file carries.
pub fn expanded_f32_bytes(g: &GgufFile) -> usize {
    g.infos.iter().map(|i| i.dims.iter().product::<usize>() * 4).sum()
}

/// Total file bytes actually holding tensor data (for the expansion-ratio line).
pub fn quant_bytes(g: &GgufFile) -> usize {
    g.infos.iter().map(|i| i.range.len()).sum()
}

/// Physical RAM, for the fits-in-RAM check (hw.memsize via sysctlbyname —
/// no libc dependency in the tree, so the one symbol is declared here).
pub fn phys_ram_bytes() -> usize {
    // Test override: the expansion-guard gate proves the refusal without
    // downloading a model that actually exceeds RAM.
    if let Some(mb) = std::env::var("LOKAL_ASSUME_RAM_MB").ok().and_then(|v| v.parse::<usize>().ok())
    {
        return mb << 20;
    }
    extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *const std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }
    let mut v: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let rc = unsafe {
        sysctlbyname(
            c"hw.memsize".as_ptr(),
            &mut v as *mut u64 as *mut _,
            &mut len,
            std::ptr::null(),
            0,
        )
    };
    if rc == 0 && v > 0 {
        v as usize
    } else {
        16 << 30 // sysctl failing is not a reason to refuse a load
    }
}

/// One-line summary for load banners and the lane-2 handoff error.
pub fn summary(g: &GgufFile) -> String {
    let mut counts: HashMap<GgmlType, usize> = HashMap::new();
    for i in &g.infos {
        *counts.entry(i.ty).or_default() += 1;
    }
    let mut parts: Vec<String> =
        counts.iter().map(|(ty, n)| format!("{n} {ty:?}")).collect();
    parts.sort();
    format!("GGUF v{}: {} tensors ({})", g.version, g.infos.len(), parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// D3: ONE sweep, ONE verdict. The old behaviour raised on the first
    /// offending tensor, so a mixed file taught its requirements one
    /// re-download at a time — the human hit exactly that.
    #[test]
    fn refusal_names_every_unsupported_type_at_once() {
        let p = crate::gguf::testutil::write_tmp("lowbit", &crate::gguf::testutil::mixed_lowbit_gguf());
        let err = match GgufFile::open(&p) {
            Ok(_) => panic!("a file of unsupported types must not open"),
            Err(e) => e.to_string(),
        };
        std::fs::remove_file(p).ok();
        for want in ["Q8_1", "Q5_1", "Q4_1"] {
            assert!(err.contains(want), "error should name {want}: {err}");
        }
        // Counts and an example per type, so the reader can see which tensors.
        assert!(err.contains("Q8_1 (2 tensors, e.g. token_embd.weight)"), "{err}");
        assert!(err.contains("Q5_1 (1 tensor, e.g."), "singular reads right: {err}");
        assert!(err.contains("3 quantization type(s)"), "{err}");
        assert!(err.contains("across 4 of 5 tensors"), "{err}");
        // The supported type must not be listed as a PROBLEM — it may still
        // appear in the recommendation, which is the point of the sentence.
        let listed = err.split(". Re-download").next().unwrap();
        assert!(!listed.contains("Q8_0"), "Q8_0 is supported and must not be listed: {err}");
        assert!(err.contains("Re-download the model as Q4_K_M or Q8_0"), "{err}");
    }

    #[test]
    fn parses_the_synthetic_file() {
        let p = crate::gguf::testutil::write_tmp("parse", &crate::gguf::testutil::tiny_gguf(3, None));
        let g = GgufFile::open(&p).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.n_tensors(), 1);
        assert_eq!(g.get_str("general.architecture").unwrap(), "llama");
        let t = g.tensor("t").unwrap();
        assert_eq!(t.dims, vec![2, 32]); // ne reversed to row-major
        assert_eq!(t.ty, GgmlType::Q8_0);
        assert_eq!(t.data.len(), 68);
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn respects_general_alignment() {
        let p = crate::gguf::testutil::write_tmp("align", &crate::gguf::testutil::tiny_gguf(2, Some(64)));
        let g = GgufFile::open(&p).unwrap();
        assert_eq!(g.tensor("t").unwrap().data.len(), 68);
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn rejects_big_endian_and_truncation() {
        let mut be = crate::gguf::testutil::tiny_gguf(3, None);
        be[4..8].copy_from_slice(&3u32.to_be_bytes());
        let p = crate::gguf::testutil::write_tmp("reject", &be);
        let open_err = |p: &std::path::Path| match GgufFile::open(p) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        let err = open_err(&p);
        assert!(err.contains("big-endian"), "{err}");
        let mut short = crate::gguf::testutil::tiny_gguf(3, None);
        short.truncate(short.len() - 40);
        std::fs::write(&p, &short).unwrap();
        let err = open_err(&p);
        assert!(err.contains("truncated") || err.contains("run past"), "{err}");
        std::fs::remove_file(p).ok();
    }
}
