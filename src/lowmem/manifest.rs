//! WeightManifest — the mmap layer under -b lowmem.
//!
//! Opens every safetensors shard once, keeps the mmaps plus a name → location
//! table, and hands out bytes on demand. Nothing is read up front: the OS pages
//! data in as it is touched and stays free to drop clean pages under pressure,
//! which is what lets a model larger than RAM open at all.

use memmap2::{Advice, Mmap};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

/// Where one tensor's bytes live: shard index + absolute byte range, plus the
/// dtype and shape needed to interpret them.
pub(crate) struct TensorMeta {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub shard: usize,
    pub offset: usize,
    pub len: usize,
}

pub(crate) struct WeightManifest {
    shards: Vec<Mmap>,
    tensors: HashMap<String, TensorMeta>,
    /// Total parameter count, summed from the headers — no data was read for it.
    pub n_params: usize,
}

impl WeightManifest {
    pub fn open(dir: &Path) -> crate::Result<Self> {
        let mut shards = Vec::new();
        let mut tensors = HashMap::new();
        for (si, name) in crate::weights::shard_files(dir)?.into_iter().enumerate() {
            let file = std::fs::File::open(dir.join(&name))?;
            let mmap = unsafe { Mmap::map(&file)? };
            // Staging sweeps run front to back within a shard; tell the pager so
            // readahead works with us (random faults are what kill disk-backed runs).
            let _ = mmap.advise(Advice::Sequential);
            let st = SafeTensors::deserialize(&mmap)?;
            let base = mmap.as_ptr() as usize;
            for (tname, view) in st.tensors() {
                let data = view.data();
                tensors.insert(
                    tname,
                    TensorMeta {
                        dtype: view.dtype(),
                        shape: view.shape().to_vec(),
                        shard: si,
                        offset: data.as_ptr() as usize - base,
                        len: data.len(),
                    },
                );
            }
            drop(st);
            shards.push(mmap);
        }
        let n_params = tensors.values().map(|m| m.shape.iter().product::<usize>()).sum();
        Ok(Self { shards, tensors, n_params })
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn meta(&self, name: &str) -> crate::Result<&TensorMeta> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name} not found in the weight files").into())
    }

    /// The tensor's raw on-disk bytes.
    fn bytes(&self, m: &TensorMeta) -> &[u8] {
        &self.shards[m.shard][m.offset..m.offset + m.len]
    }

    /// Whole tensor converted to f32 — norms, biases, and the phase-1 eager
    /// path. The paged path must never call this on a large matrix.
    pub fn read_f32(&self, name: &str) -> crate::Result<Vec<f32>> {
        let m = self.meta(name)?;
        crate::weights::to_f32(m.dtype, self.bytes(m))
    }
}
