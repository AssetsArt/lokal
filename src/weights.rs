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
