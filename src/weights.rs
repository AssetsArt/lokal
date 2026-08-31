//! Loads weights from .safetensors files into a HashMap<tensor name, Vec<f32>>.
//!
//! The safetensors format is refreshingly simple: a JSON header describing each
//! tensor's name/dtype/shape/byte range, followed by the raw data. The
//! `safetensors` crate parses the header; the numeric conversion to f32 is done
//! here (most checkpoints store bfloat16 to save space).

use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

pub type TensorMap = HashMap<String, Vec<f32>>;

pub fn load(dir: &Path) -> crate::Result<TensorMap> {
    let mut tensors = TensorMap::new();
    for name in shard_files(dir)? {
        // mmap lets the OS page the file in lazily instead of reading it up front.
        let file = std::fs::File::open(dir.join(&name))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;
        for (tensor_name, view) in st.tensors() {
            tensors.insert(tensor_name, to_f32(view.dtype(), view.data())?);
        }
    }
    Ok(tensors)
}

/// Weight file list: the single standard file, or multiple shards per the index.
/// (pub(crate): the lowmem manifest walks the same shards without loading them.)
pub(crate) fn shard_files(dir: &Path) -> crate::Result<Vec<String>> {
    if dir.join("model.safetensors").exists() {
        return Ok(vec!["model.safetensors".into()]);
    }
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        dir.join("model.safetensors.index.json"),
    )?)?;
    let mut files: Vec<String> = index["weight_map"]
        .as_object()
        .ok_or("index file has no weight_map")?
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    files.sort();
    files.dedup();
    Ok(files)
}

/// Convert raw little-endian data to f32.
/// bf16 is f32 with the mantissa truncated to 8 bits — same range, less precision.
pub(crate) fn to_f32(dtype: Dtype, data: &[u8]) -> crate::Result<Vec<f32>> {
    let out = match dtype {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        other => return Err(format!("unsupported dtype {other:?}").into()),
    };
    Ok(out)
}

/// The tensor-storage seam (docs/gguf-design.md §Tensor Abstraction): the
/// model graph must not know where tensors live. Two variants exist on main —
/// the eager f32 map (safetensors via `load`, GGUF via `gguf::load_f32`) and
/// the mmap-backed `LowMemSource` — and both satisfy this trait. It is a
/// LOADING seam: `dyn` here is off every hot path (construction only), and the
/// paged/view surface the streaming backends also need (row ranges, GPU spans)
/// is intentionally beyond it — an f32 seam must not pretend to cover paging.
pub(crate) trait TensorStore {
    fn has(&self, name: &str) -> bool;
    /// Element count without materializing (shape inference, e.g. head_dim).
    fn numel(&self, name: &str) -> Option<usize>;
    /// Materialize one tensor as owned f32. A store MAY hand its buffer over
    /// (the map does — second take errors); a view-backed store re-reads.
    fn take_f32(&mut self, name: &str) -> crate::Result<Vec<f32>>;
}

impl TensorStore for TensorMap {
    fn has(&self, name: &str) -> bool {
        self.contains_key(name)
    }
    fn numel(&self, name: &str) -> Option<usize> {
        self.get(name).map(Vec::len)
    }
    fn take_f32(&mut self, name: &str) -> crate::Result<Vec<f32>> {
        self.remove(name)
            .ok_or_else(|| format!("tensor {name} not found in the weight files").into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seam's move semantics on the eager map: a take hands the buffer
    /// over, so a second take of the same name is an error, and optional
    /// tensors probe with has() first (how from_store reads q_norm/bias).
    #[test]
    fn tensor_map_store_hands_buffers_over() {
        let mut t = TensorMap::new();
        t.insert("a.weight".into(), vec![1.0, 2.0]);
        assert!(TensorStore::has(&t, "a.weight"));
        assert_eq!(TensorStore::numel(&t, "a.weight"), Some(2));
        assert_eq!(t.take_f32("a.weight").unwrap(), vec![1.0, 2.0]);
        assert!(!TensorStore::has(&t, "a.weight"));
        assert!(t.take_f32("a.weight").is_err(), "second take must error");
        assert!(t.take_f32("missing").unwrap_err().to_string().contains("missing"));
    }

    /// The seam over the view-backed store: LowMemSource::take_f32 re-reads
    /// (nothing consumed) — both formats behind one trait is the whole point
    /// of spec §11. Real-file: the synthetic header is too bare for open().
    #[test]
    #[ignore = "needs the local Qwen GGUF checkpoint"]
    fn lowmem_source_satisfies_the_seam() {
        let p = crate::gguf::testutil::qwen_gguf();
        let mut src = crate::lowmem::LowMemSource::open(&p).unwrap();
        let store: &mut dyn TensorStore = &mut src;
        let name = "model.norm.weight";
        assert!(store.has(name));
        let a = store.take_f32(name).unwrap();
        let b = store.take_f32(name).unwrap();
        assert_eq!(a, b, "view-backed take consumes nothing");
        assert_eq!(store.numel(name), Some(a.len()));
        assert!(!a.is_empty());
    }
}
