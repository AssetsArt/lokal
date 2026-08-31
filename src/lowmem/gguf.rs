//! GGUF checkpoint reading (lane gguf-loader): the container parser, the
//! arch-metadata → ModelConfig mapping, the tokenizer built from the embedded
//! vocab, and the dequant-everything loader the cpu/metal backends use when a
//! model genuinely fits RAM. The shared seam types (GgmlType, GgufTensor,
//! dequant_row_ref) live in src/lowmem/manifest.rs.
//!
//! Format facts (magic/version/kv/tensor-info order, type ids, key and tensor
//! names, pre-tokenizer regexes) are transcribed from a read-only llama.cpp
//! checkout (ggml/include/gguf.h, ggml/src/gguf.cpp, src/llama-arch.cpp,
//! src/llama-vocab.cpp), not from memory. Little-endian files only.

use crate::lowmem::manifest::{dequant_row_ref, GgmlType, GgufTensor};
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
                    0..=5 | 10 | 11 => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(match self.value(et, key)? {
                                GgufValue::U64(x) => x as i64,
                                GgufValue::I64(x) => x,
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

struct TensorInfo {
    name: String,
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
    kv: HashMap<String, GgufValue>,
    infos: Vec<TensorInfo>,
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
            let ty = GgmlType::from_gguf(ty_id).map_err(|tyname| {
                // With Q5_0 in the set (ruling daa86b0f), Q4_K_M is a safe
                // recommendation again — small models' mixed files load.
                format!(
                    "tensor {name} is {tyname}, which lokal does not run — \
                     re-download the model as Q4_K_M or Q8_0"
                )
            })?;
            if ne[0] % ty.blk_elems() != 0 {
                return Err(format!(
                    "tensor {name}: row length {} is not a whole number of {:?} blocks",
                    ne[0], ty
                )
                .into());
            }
            raw.push((name, ne, ty, offset));
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

/// What the runtime needs to know about a GGUF model beyond ModelConfig.
/// qwen3's per-head q/k RMSNorm and explicit head_dim travel here — lane 2
/// applies them; the cpu/metal forward does not know them and refuses qwen3.
pub struct GgufArch {
    pub arch: String,
    pub qk_norm: bool,
    /// Explicit head dim from metadata. qwen3 violates hidden/n_heads — never
    /// derive this.
    pub head_dim: usize,
}

/// `general.architecture` + the per-arch keys → the shared ModelConfig.
pub fn model_config(g: &GgufFile) -> crate::Result<(crate::config::ModelConfig, GgufArch)> {
    let arch = g.get_str("general.architecture")?.to_string();
    if !matches!(arch.as_str(), "llama" | "qwen2" | "qwen3") {
        return Err(format!(
            "GGUF architecture \"{arch}\" is not supported (llama, qwen2, qwen3 are)"
        )
        .into());
    }
    let k = |suffix: &str| format!("{arch}.{suffix}");
    let heads = g.get_usize(&k("attention.head_count"))?;
    let hidden = g.get_usize(&k("embedding_length"))?;
    // vocab: the token list is authoritative; the explicit key is the fallback
    // for a file that carries no tokenizer.
    let vocab_size = match g.kv.get("tokenizer.ggml.tokens") {
        Some(GgufValue::ArrStr(t)) => t.len(),
        _ => g.get_usize(&k("vocab_size"))?,
    };
    let head_dim = match g.get_usize(&k("attention.key_length")) {
        Ok(x) => x,
        Err(_) => hidden / heads,
    };
    let qk_norm = g.has_key("tokenizer.ggml.tokens") && arch == "qwen3"
        || g.infos.iter().any(|i| i.name.ends_with("attn_q_norm.weight"));

    let cfg = crate::config::ModelConfig {
        architectures: vec![format!("gguf:{arch}")],
        hidden_size: hidden,
        intermediate_size: g.get_usize(&k("feed_forward_length"))?,
        num_hidden_layers: g.get_usize(&k("block_count"))?,
        num_attention_heads: heads,
        num_key_value_heads: g.get_usize(&k("attention.head_count_kv"))?,
        vocab_size,
        rms_norm_eps: g.get_f32(&k("attention.layer_norm_rms_epsilon"))?,
        rope_theta: g.get_f32(&k("rope.freq_base")).unwrap_or(10000.0),
        max_position_embeddings: g.get_usize(&k("context_length")).unwrap_or(4096),
        eos_token_id: match g.get_usize("tokenizer.ggml.eos_token_id") {
            Ok(id) => crate::config::EosIds::One(id as u32),
            Err(_) => crate::config::EosIds::default(),
        },
    };
    Ok((cfg, GgufArch { arch, qk_norm, head_dim }))
}

// ---------- tensor names: GGUF (llama-arch.cpp) → HF (the rest of the tree) ----------

/// Map one GGUF tensor name to the HF name every backend in this tree uses.
/// Returns None for tensors the forward pass does not consume (rope_freqs).
pub fn hf_name(gguf: &str) -> Option<String> {
    fn tail(s: &str) -> Option<(usize, &str)> {
        let rest = s.strip_prefix("blk.")?;
        let (n, rest) = rest.split_once('.')?;
        Some((n.parse().ok()?, rest))
    }
    Some(match gguf {
        "token_embd.weight" => "model.embed_tokens.weight".into(),
        "output_norm.weight" => "model.norm.weight".into(),
        "output.weight" => "lm_head.weight".into(),
        "rope_freqs.weight" => return None,
        _ => {
            let (i, rest) = tail(gguf)?;
            let (mid, kind) = rest.rsplit_once('.')?;
            if !matches!(kind, "weight" | "bias") {
                return None;
            }
            let hf_mid = match mid {
                "attn_norm" => "input_layernorm".into(),
                "ffn_norm" => "post_attention_layernorm".into(),
                "attn_q" => "self_attn.q_proj".into(),
                "attn_k" => "self_attn.k_proj".into(),
                "attn_v" => "self_attn.v_proj".into(),
                "attn_output" => "self_attn.o_proj".into(),
                "attn_q_norm" => "self_attn.q_norm".into(),
                "attn_k_norm" => "self_attn.k_norm".into(),
                "ffn_gate" => "mlp.gate_proj".into(),
                "ffn_up" => "mlp.up_proj".into(),
                "ffn_down" => "mlp.down_proj".into(),
                other => format!("gguf.{other}"),
            };
            format!("model.layers.{i}.{hf_mid}.{kind}")
        }
    })
}

// ---------- the fits-in-RAM loader (revised D6) ----------

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

/// llama-arch GGUFs store q_proj/k_proj with llama.cpp's RoPE permute
/// (LlamaModel.permute in the converter): per head,
///   gguf[h*hd + d*2 + a] = hf[h*hd + a*(hd/2) + d],  a in {0,1}.
/// Verified numerically row-by-row on SmolLM2 (lane note). qwen2 checkpoints
/// are NOT permuted (proved bit-exact without it). Undo it at load so every
/// consumer sees the HF layout the whole tree assumes. Public for lane 2:
/// GgufTensor.data is the file's bytes, i.e. still permuted for llama-arch.
pub fn unpermute_llama_qk(data: &mut Vec<f32>, rows: usize, cols: usize, head_dim: usize) {
    debug_assert_eq!(rows % head_dim, 0);
    let mut out = vec![0f32; data.len()];
    for h in 0..rows / head_dim {
        for j in 0..head_dim {
            let (a, d) = (j / (head_dim / 2), j % (head_dim / 2));
            let src = h * head_dim + d * 2 + a;
            out[(h * head_dim + j) * cols..][..cols]
                .copy_from_slice(&data[src * cols..][..cols]);
        }
    }
    *data = out;
}

/// Dequantize the whole checkpoint to f32 under HF names — what `-b cpu` and
/// `-b metal` eat. The caller has already run the fits-in-RAM check.
pub fn load_f32(g: &GgufFile) -> crate::Result<crate::weights::TensorMap> {
    let (_, arch) = model_config(g)?;
    let undo_permute = arch.arch == "llama";
    let mut map = crate::weights::TensorMap::new();
    for t in g.tensors() {
        let Some(name) = hf_name(&t.name) else { continue };
        let n: usize = t.dims.iter().product();
        let mut out = vec![0f32; n];
        dequant_row_ref(t.ty, t.data, &mut out);
        if undo_permute
            && t.dims.len() == 2
            && (t.name.ends_with("attn_q.weight") || t.name.ends_with("attn_k.weight"))
        {
            unpermute_llama_qk(&mut out, t.dims[0], t.dims[1], arch.head_dim);
        }
        map.insert(name, out);
    }
    Ok(map)
}

// ---------- tokenizer from the embedded vocab (D5) ----------

/// Build a `tokenizers::Tokenizer` equivalent to the model's HF tokenizer.json:
/// NFC → Split(arch regex) → ByteLevel, BPE from tokenizer.ggml.tokens/merges,
/// control tokens (token_type 3) registered as special so they never round-trip
/// through byte decoding (llama3's reserved tokens break naive decoders).
pub fn build_tokenizer(g: &GgufFile) -> crate::Result<tokenizers::Tokenizer> {
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::unicode::NFC;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::pre_tokenizers::sequence::Sequence;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
    use tokenizers::{AddedToken, SplitDelimiterBehavior};

    let model = g.get_str("tokenizer.ggml.model")?;
    if model != "gpt2" {
        return Err(format!(
            "GGUF tokenizer model \"{model}\" is not supported yet — byte-level BPE (\"gpt2\") only; \
             use the safetensors checkpoint for this model"
        )
        .into());
    }
    let tokens = g.get_arr_str("tokenizer.ggml.tokens")?;
    let merges_raw = g.get_arr_str("tokenizer.ggml.merges")?;
    let token_type = g.get_arr_i64("tokenizer.ggml.token_type").unwrap_or(&[]);

    let vocab: tokenizers::models::bpe::Vocab =
        tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    // Merges are "left right" pairs; byte-level tokens never contain a real
    // space (space is Ġ), so the first space is the separator.
    let merges: Vec<(String, String)> = merges_raw
        .iter()
        .map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .ok_or_else(|| format!("malformed merge entry {m:?}"))
        })
        .collect::<Result<_, _>>()?;

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| format!("BPE build: {e}"))?;
    let mut tok = tokenizers::Tokenizer::new(bpe);

    // The split regex is per pre-tokenizer family (tokenizer.ggml.pre), each
    // string lifted from the model's own tokenizer.json via llama-vocab.cpp.
    let pre = g.get_str("tokenizer.ggml.pre").unwrap_or("gpt2");
    let regex = match pre {
        "qwen2" => {
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        }
        "llama3" | "llama-bpe" => {
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        }
        // GPT-2's original — the ecosystem default for unmarked BPE files.
        _ => r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    };
    let split = Split::new(SplitPattern::Regex(regex.into()), SplitDelimiterBehavior::Isolated, false)
        .map_err(|e| format!("pre-tokenizer regex: {e}"))?;
    let _ = tok.with_normalizer(Some(NFC));
    let _ = tok.with_pre_tokenizer(Some(Sequence::new(vec![
        split.into(),
        ByteLevel::new(false, false, false).into(),
    ])));
    let _ = tok.with_decoder(Some(ByteLevel::new(false, false, false)));

    // token_type 3 = control (llama.cpp's LLAMA_TOKEN_TYPE_CONTROL).
    let specials: Vec<AddedToken> = token_type
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == 3)
        .filter_map(|(i, _)| tokens.get(i))
        .map(|s| AddedToken::from(s.clone(), true))
        .collect();
    if !specials.is_empty() {
        let _ = tok.add_special_tokens(specials);
    }
    Ok(tok)
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

    /// A tiny synthetic v3 file: one string KV, one Q8_0 tensor of ne=[32, 2]
    /// at offset 0. Exercises the header walk, alignment, dim reversal, and
    /// bounds checks without touching a real checkpoint.
    fn tiny_gguf(version: u32, align: Option<u64>) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&version.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        let n_kv = 1 + align.is_some() as u64;
        b.extend_from_slice(&n_kv.to_le_bytes());
        let put_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        put_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes()); // string
        put_str(&mut b, "llama");
        if let Some(a) = align {
            put_str(&mut b, "general.alignment");
            b.extend_from_slice(&4u32.to_le_bytes()); // u32
            b.extend_from_slice(&(a as u32).to_le_bytes());
        }
        // tensor info: name, n_dims=2, ne=[32,2] (fastest first), type Q8_0, offset 0
        put_str(&mut b, "t");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&32u64.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        // pad to alignment, then 2 rows x 34 bytes of Q8_0 data
        let align = align.unwrap_or(32) as usize;
        while b.len() % align != 0 {
            b.push(0);
        }
        b.extend_from_slice(&vec![0u8; 68]);
        b
    }

    fn write_tmp(label: &str, bytes: &[u8]) -> std::path::PathBuf {
        // One file per TEST, not per process: the tests run in parallel and a
        // shared name makes one test open another's bytes.
        let p = std::env::temp_dir()
            .join(format!("lokal-gguf-test-{}-{label}.gguf", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn parses_the_synthetic_file() {
        let p = write_tmp("parse", &tiny_gguf(3, None));
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
        let p = write_tmp("align", &tiny_gguf(2, Some(64)));
        let g = GgufFile::open(&p).unwrap();
        assert_eq!(g.tensor("t").unwrap().data.len(), 68);
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn rejects_big_endian_and_truncation() {
        let mut be = tiny_gguf(3, None);
        be[4..8].copy_from_slice(&3u32.to_be_bytes());
        let p = write_tmp("reject", &be);
        let open_err = |p: &std::path::Path| match GgufFile::open(p) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        };
        let err = open_err(&p);
        assert!(err.contains("big-endian"), "{err}");
        let mut short = tiny_gguf(3, None);
        short.truncate(short.len() - 40);
        std::fs::write(&p, &short).unwrap();
        let err = open_err(&p);
        assert!(err.contains("truncated") || err.contains("run past"), "{err}");
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn unpermute_inverts_the_converter_permute() {
        // Forward permute exactly as conversion/llama.py does it, on a 2-head,
        // hd=4, cols=1 toy: reshape(h, 2, hd/2, in).swapaxes(1,2).reshape.
        let hd = 4usize;
        let rows = 8usize;
        let hf: Vec<f32> = (0..rows as u32).map(|x| x as f32).collect();
        let mut gguf = vec![0f32; rows];
        for h in 0..rows / hd {
            for a in 0..2 {
                for d in 0..hd / 2 {
                    gguf[h * hd + d * 2 + a] = hf[h * hd + a * (hd / 2) + d];
                }
            }
        }
        let mut got = gguf.clone();
        unpermute_llama_qk(&mut got, rows, 1, hd);
        assert_eq!(got, hf);
    }

    #[test]
    fn maps_gguf_names_to_hf() {
        assert_eq!(hf_name("token_embd.weight").as_deref(), Some("model.embed_tokens.weight"));
        assert_eq!(
            hf_name("blk.7.attn_output.weight").as_deref(),
            Some("model.layers.7.self_attn.o_proj.weight")
        );
        assert_eq!(hf_name("blk.0.attn_q.bias").as_deref(), Some("model.layers.0.self_attn.q_proj.bias"));
        assert_eq!(
            hf_name("blk.3.attn_k_norm.weight").as_deref(),
            Some("model.layers.3.self_attn.k_norm.weight")
        );
        assert_eq!(hf_name("rope_freqs.weight"), None);
    }

    // ---- real-file tests: run by the gates with `--ignored` ----

    fn qwen_gguf() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join(".cache/gguf/qwen2.5-0.5b-instruct-fp16.gguf")
    }

    fn qwen_hf_tokenizer() -> std::path::PathBuf {
        let snaps = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join(".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots");
        std::fs::read_dir(snaps)
            .unwrap()
            .flatten()
            .map(|e| e.path().join("tokenizer.json"))
            .find(|p| p.is_file())
            .expect("Qwen HF snapshot with tokenizer.json")
    }

    /// qwen3 metadata: explicit head_dim (128 with hidden 1024 — violating
    /// hidden/n_heads) and the qk-norm tensors must surface through GgufArch.
    #[test]
    fn qwen3_arch_meta_from_synthetic_header() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let keys: &[(&str, u32, u64)] = &[
            ("qwen3.block_count", 4, 2),
            ("qwen3.embedding_length", 4, 1024),
            ("qwen3.feed_forward_length", 4, 3072),
            ("qwen3.attention.head_count", 4, 16),
            ("qwen3.attention.head_count_kv", 4, 8),
            ("qwen3.attention.key_length", 4, 128),
            ("qwen3.vocab_size", 4, 1000),
        ];
        b.extend_from_slice(&((keys.len() + 2) as u64).to_le_bytes());
        let put_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        put_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut b, "qwen3");
        for (k, ty, v) in keys {
            put_str(&mut b, k);
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&(*v as u32).to_le_bytes());
        }
        put_str(&mut b, "qwen3.attention.layer_norm_rms_epsilon");
        b.extend_from_slice(&6u32.to_le_bytes());
        b.extend_from_slice(&1e-6f32.to_le_bytes());
        // one qk-norm tensor so the detection has something to see
        put_str(&mut b, "blk.0.attn_q_norm.weight");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&128u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&0u64.to_le_bytes());
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend_from_slice(&vec![0u8; 512]);
        let p = write_tmp("qwen3", &b);
        let g = GgufFile::open(&p).unwrap();
        let (cfg, arch) = model_config(&g).unwrap();
        assert_eq!(arch.arch, "qwen3");
        assert_eq!(arch.head_dim, 128, "head_dim must come from metadata, never hidden/n_heads");
        assert!(arch.qk_norm);
        assert_eq!(cfg.hidden_size / cfg.num_attention_heads, 64); // the identity qwen3 breaks
        // The expansion arithmetic the fits-in-RAM guard runs: 128 f32 elems.
        assert_eq!(expanded_f32_bytes(&g), 128 * 4);
        assert_eq!(quant_bytes(&g), 128 * 4);
        std::fs::remove_file(p).ok();
    }

    fn hf_tokenizer_json(repo_dir: &str) -> std::path::PathBuf {
        let snaps = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join(".cache/huggingface/hub")
            .join(repo_dir)
            .join("snapshots");
        std::fs::read_dir(snaps)
            .unwrap()
            .flatten()
            .map(|e| e.path().join("tokenizer.json"))
            .find(|p| p.is_file())
            .expect("snapshot with tokenizer.json")
    }

    #[test]
    #[ignore = "needs the local GGUF + HF checkpoints"]
    fn gguf_tokenizer_matches_hf_on_mixed_corpus() {
        let smol_gguf = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(
            ".cache/huggingface/hub/models--unsloth--SmolLM2-135M-Instruct-GGUF/snapshots/9e6855bc4be717fca1ef21360a1db4b29d5c559a/SmolLM2-135M-Instruct-F16.gguf",
        );
        for (gguf, hf_json) in [
            (qwen_gguf(), hf_tokenizer_json("models--Qwen--Qwen2.5-0.5B-Instruct")),
            (smol_gguf, hf_tokenizer_json("models--HuggingFaceTB--SmolLM2-135M-Instruct")),
        ] {
            check_tokenizer_pair(&gguf, &hf_json);
        }
    }

    fn check_tokenizer_pair(gguf: &std::path::Path, hf_json: &std::path::Path) {
        let g = GgufFile::open(gguf).unwrap();
        let ours = build_tokenizer(&g).unwrap();
        let hf = tokenizers::Tokenizer::from_file(hf_json).unwrap();
        let corpus = [
            "สวัสดีครับ วันนี้อากาศดีมาก ๆ เลยนะครับ",
            "ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ ทำให้ tokenizer ต้องทำงานหนัก",
            "Hello, world! I'll say it again: don't panic.",
            "Mixed ไทย English และ 中文 plus ελληνικά in one line",
            "🎉🚀 emoji test 👨‍👩‍👧‍👦 with a ZWJ family and 🇹🇭 a flag",
            "numbers 1234567890, 3.14159, 1e-9, 0xDEADBEEF",
            "code: fn main() { println!(\"hi\"); }\n\tindented\ttabs  and   runs of spaces",
            "trailing spaces   \nและบรรทัดใหม่\r\nwindows line endings",
            "",
            " leading space",
        ];
        for line in corpus {
            let a = ours.encode(line, false).unwrap();
            let b = hf.encode(line, false).unwrap();
            assert_eq!(a.get_ids(), b.get_ids(), "encode mismatch on {line:?}");
            let da = ours.decode(a.get_ids(), true).unwrap();
            let db = hf.decode(b.get_ids(), true).unwrap();
            assert_eq!(da, db, "decode mismatch on {line:?}");
        }
    }

    #[test]
    #[ignore = "needs the local Qwen GGUF checkpoint"]
    fn qwen_gguf_config_matches_known_shape() {
        let g = GgufFile::open(&qwen_gguf()).unwrap();
        let (cfg, arch) = model_config(&g).unwrap();
        assert_eq!(arch.arch, "qwen2");
        assert!(!arch.qk_norm);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.vocab_size, 151936);
        assert!(cfg.is_eos(151645));
    }
}

#[cfg(test)]
mod equivalence_tests {
    use super::*;

    /// Name-mapping + dequant proof with ZERO tolerance: an F16 GGUF converted
    /// from the same snapshot the safetensors arm loads must dequantize to the
    /// exact same f32 values — bf16 is exactly f16-representable, so the
    /// converter's f16 rounding is lossless here. Asset built with
    /// convert_hf_to_gguf.py from snapshot 7ae55760. Deliberately NOT Qwen's
    /// own fp16 upload: that file carries OLDER instruct weights than today's
    /// snapshot (embed frozen, everything else differs — task note 39617bde).
    #[test]
    #[ignore = "needs the local same-snapshot GGUF + safetensors checkpoints"]
    fn fp16_gguf_matches_safetensors_bit_exact() {
        let home = std::env::var("HOME").unwrap();
        let g = GgufFile::open(
            &std::path::Path::new(&home)
                .join(".cache/gguf/qwen2.5-0.5b-instruct-f16-from-7ae55760.gguf"),
        )
        .unwrap();
        let snaps = std::path::PathBuf::from(&home)
            .join(".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots");
        let st_dir = std::fs::read_dir(snaps)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .find(|p| p.join("model.safetensors").is_file())
            .unwrap();
        let ours = load_f32(&g).unwrap();
        let theirs = crate::weights::load(&st_dir).unwrap();
        let mut checked = 0usize;
        for (name, st) in &theirs {
            let Some(gg) = ours.get(name) else {
                panic!("safetensors tensor {name} has no GGUF counterpart");
            };
            assert_eq!(gg.len(), st.len(), "{name} length");
            for (i, (a, b)) in gg.iter().zip(st).enumerate() {
                // Compare AFTER f16 rounding — the form every engine stores.
                // bf16 is exactly f16-representable in the normal range, but
                // f16 SUBNORMALS (|x| < 2^-14) round; both paths round them
                // identically, so post-round bits must still be equal.
                let (fa, fb) = (half::f16::from_f32(*a), half::f16::from_f32(*b));
                assert!(
                    fa.to_bits() == fb.to_bits(),
                    "{name}[{i}]: {a} vs {b} — f16 bits {:04x} != {:04x}",
                    fa.to_bits(),
                    fb.to_bits()
                );
            }
            checked += 1;
        }
        assert!(checked > 200, "only {checked} tensors compared");
        eprintln!("equivalence: {checked} tensors bit-exact at f16");
    }
}
