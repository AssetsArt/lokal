//! WeightManifest — the mmap layer under -b lowmem.
//!
//! Opens every safetensors shard once, keeps the mmaps plus a name → location
//! table, and hands out bytes on demand. Nothing is read up front: the OS pages
//! data in as it is touched and stays free to drop clean pages under pressure,
//! which is what lets a model larger than RAM open at all.

use memmap2::{Advice, Mmap};
use metal::{Buffer, Device, MTLResourceOptions};
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
    /// No-copy MTLBuffers wrapping each shard's mmap, so kernels read weight
    /// bytes straight from the page cache — the pattern llama.cpp ships for
    /// larger-than-RAM models on Apple silicon: file-backed pages stay
    /// reclaimable, fault in on first GPU touch, and never exist twice.
    /// (Human-directed exception to the spec's original never-hand-mmap-to-
    /// Metal rule, recorded on the task.) Declared before `shards` so the
    /// buffers drop before the mappings they alias.
    views: Vec<Buffer>,
    shards: Vec<Mmap>,
    tensors: HashMap<String, TensorMeta>,
    /// Total parameter count, summed from the headers — no data was read for it.
    pub n_params: usize,
}

// The views alias plain readonly mmaps; Metal buffer handles are documented
// thread-safe (same justification as the engines).
unsafe impl Send for WeightManifest {}
unsafe impl Sync for WeightManifest {}

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
        Ok(Self { views: Vec::new(), shards, tensors, n_params })
    }

    /// Create the GPU views (once, at engine build). mmap bases are page
    /// aligned; the length rounds up to the page the mapping already spans.
    pub fn make_gpu_views(&mut self, device: &Device) {
        const PAGE: usize = 16384;
        self.views = self
            .shards
            .iter()
            .map(|m| {
                device.new_buffer_with_bytes_no_copy(
                    m.as_ptr() as *const _,
                    m.len().next_multiple_of(PAGE) as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            })
            .collect();
    }

    /// GPU view + absolute byte offset for rows r0..r1 of a 2-D tensor, or
    /// None when views are absent or the span can't be read as ushorts.
    pub fn gpu_span(
        &self,
        name: &str,
        r0: usize,
        r1: usize,
    ) -> crate::Result<Option<(&Buffer, usize)>> {
        let m = self.meta(name)?;
        let n_rows = m.shape.first().copied().unwrap_or(0);
        if m.shape.len() != 2 || r1 > n_rows || r0 >= r1 {
            return Err(format!("gpu_span({name}, {r0}..{r1}): tensor has shape {:?}", m.shape).into());
        }
        let off = m.offset + r0 * (m.len / n_rows);
        Ok(match self.views.get(m.shard) {
            Some(b) if off % 2 == 0 => Some((b, off)),
            _ => None,
        })
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn meta(&self, name: &str) -> crate::Result<&TensorMeta> {
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

    /// Raw bytes of rows r0..r1 of a 2-D tensor — one contiguous mmap slice
    /// (rows are contiguous on disk), plus the dtype to interpret it. This is
    /// the read the pager stages pages through; the slice may be unaligned, so
    /// callers convert via byte chunks, never typed pointers.
    pub fn read_rows(&self, name: &str, r0: usize, r1: usize) -> crate::Result<(&[u8], Dtype)> {
        let m = self.meta(name)?;
        let n_rows = m.shape.first().copied().unwrap_or(0);
        if m.shape.len() != 2 || r1 > n_rows || r0 >= r1 {
            return Err(format!(
                "read_rows({name}, {r0}..{r1}): tensor has shape {:?}",
                m.shape
            )
            .into());
        }
        let row_bytes = m.len / n_rows;
        Ok((&self.bytes(m)[r0 * row_bytes..r1 * row_bytes], m.dtype))
    }
}

// ---------- The GGUF seam (frozen: lane gguf-kernels builds against this) ----------
