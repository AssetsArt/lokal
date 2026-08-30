//! WeightPool — an LRU byte budget of staged weight pages in shared MTLBuffers.
//!
//! A page is a ROW BLOCK of one 2-D tensor (rows are contiguous on disk, so a
//! page stages with one mmap read), capped at PAGE_BYTES; a tensor smaller than
//! that is one page. dtype conversion to f16 happens here, per page, on the way
//! in — never a whole-tensor or whole-model copy. Each resident page's buffer
//! is exactly its page's size and is accounted against the byte budget, which
//! keeps the budget arithmetic exact (challenge 8ca59508 against D3's
//! fixed-size slots: equal-size slots measured ~3× waste on real checkpoints).
//!
//! Eviction is LRU over a use-clock; pages referenced by the command buffer
//! currently being encoded are pinned and never evicted, which is the entire
//! correctness argument while every command buffer is waited on before the next
//! one opens. The epoch machinery for overlapped submission arrives with the
//! per-layer pipeline phase.

use super::manifest::WeightManifest;
use half::{bf16, f16};
use metal::{Buffer, Device, MTLResourceOptions};
use safetensors::Dtype;
use std::collections::HashMap;

/// Upper bound on one page's staged f16 bytes — row blocks are sized to it.
pub(super) const PAGE_BYTES: usize = 16 << 20;

/// One large weight matrix, paged by row blocks. `id` keys the pool map.
pub(super) struct PagedTensor {
    pub id: u32,
    pub name: String,
    pub in_dim: usize,
    pub out_dim: usize,
    pub rows_per_page: usize,
    pub n_pages: usize,
    /// Eagerly-resident f16 bias; encode binds the engine's shared zero buffer
    /// when absent (same convention as the metal backend).
    pub bias: Option<Buffer>,
}

impl PagedTensor {
    pub fn new(
        mf: &WeightManifest,
        id: u32,
        name: String,
        in_dim: usize,
        out_dim: usize,
        bias: Option<Buffer>,
    ) -> crate::Result<Self> {
        let m = mf.meta(&name)?;
        if m.shape != [out_dim, in_dim] {
            return Err(format!(
                "{name} has shape {:?} but the config implies [{out_dim}, {in_dim}]",
                m.shape
            )
            .into());
        }
        let rows_per_page = (PAGE_BYTES / (in_dim * 2)).clamp(1, out_dim);
        Ok(Self {
            id,
            name,
            in_dim,
            out_dim,
            rows_per_page,
            n_pages: out_dim.div_ceil(rows_per_page),
            bias,
        })
    }

    /// (first row, row count) of one block — the last block may be short.
    pub fn block_rows(&self, block: usize) -> (usize, usize) {
        let r0 = block * self.rows_per_page;
        (r0, self.rows_per_page.min(self.out_dim - r0))
    }

    /// Staged f16 bytes of one block.
    fn block_bytes(&self, block: usize) -> usize {
        self.block_rows(block).1 * self.in_dim * 2
    }
}

struct Page {
    buf: Buffer,
    bytes: usize,
    last_used: u64,
    pinned: bool,
}

pub(super) struct WeightPool {
    device: Device,
    budget: usize,
    used: usize,
    pages: HashMap<(u32, u32), Page>,
    clock: u64,
    /// f32→f16 clips past 65504 — flag an overflowing checkpoint once, at the
    /// first page that actually clips (cheaper than a full scan at open).
    overflow_warned: bool,
}

impl WeightPool {
    pub fn new(device: &Device, budget_bytes: usize) -> Self {
        Self {
            device: device.clone(),
            budget: budget_bytes.max(4 * PAGE_BYTES), // never below 4 pages (D9 floor)
            used: 0,
            pages: HashMap::new(),
            clock: 0,
            overflow_warned: false,
        }
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget
    }

    /// Make every (tensor, block) page resident and pin it for the command
    /// buffer about to be encoded. Pinning happens in two passes so a staging
    /// miss can never evict a page this same call just resolved.
    pub fn make_resident(
        &mut self,
        mf: &WeightManifest,
        pages: &[(&PagedTensor, usize)],
    ) -> crate::Result<()> {
        for (t, b) in pages {
            if let Some(p) = self.pages.get_mut(&(t.id, *b as u32)) {
                self.clock += 1;
                p.last_used = self.clock;
                p.pinned = true;
            }
        }
        for (t, b) in pages {
            let key = (t.id, *b as u32);
            if self.pages.contains_key(&key) {
                continue;
            }
            let need = t.block_bytes(*b);
            while self.used + need > self.budget {
                self.evict_lru().map_err(|_| {
                    format!(
                        "weight pool exhausted: one encode set needs more than the {} MB budget — raise the memory budget",
                        self.budget >> 20
                    )
                })?;
            }
            let buf = self
                .device
                .new_buffer(need as u64, MTLResourceOptions::StorageModeShared);
            self.stage(mf, t, *b, &buf)?;
            self.clock += 1;
            self.used += need;
            self.pages.insert(
                key,
                Page { buf, bytes: need, last_used: self.clock, pinned: true },
            );
        }
        Ok(())
    }

    pub fn get(&self, t: &PagedTensor, block: usize) -> &Buffer {
        &self.pages[&(t.id, block as u32)].buf
    }

    /// The encoded command buffer completed — every page is evictable again.
    pub fn unpin_all(&mut self) {
        for p in self.pages.values_mut() {
            p.pinned = false;
        }
    }

    fn evict_lru(&mut self) -> Result<(), ()> {
        let key = self
            .pages
            .iter()
            .filter(|(_, p)| !p.pinned)
            .min_by_key(|(_, p)| p.last_used)
            .map(|(k, _)| *k)
            .ok_or(())?;
        let p = self.pages.remove(&key).unwrap();
        self.used -= p.bytes;
        Ok(())
    }

    /// Read the block's rows from the mmap and convert into the buffer — the
    /// ONLY place weight bytes are copied, and only ever one page's worth.
    fn stage(
        &mut self,
        mf: &WeightManifest,
        t: &PagedTensor,
        block: usize,
        dst: &Buffer,
    ) -> crate::Result<()> {
        let (r0, rows) = t.block_rows(block);
        let (src, dtype) = mf.read_rows(&t.name, r0, r0 + rows)?;
        let dp = dst.contents() as *mut u16;
        // The mmap slice can be unaligned (safetensors headers are arbitrary
        // lengths), so conversion walks byte pairs/quads — never typed pointers.
        let mut clipped = false;
        match dtype {
            Dtype::F16 => unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dp as *mut u8, src.len());
            },
            Dtype::BF16 => {
                for (i, b) in src.chunks_exact(2).enumerate() {
                    let v = bf16::from_le_bytes([b[0], b[1]]).to_f32();
                    clipped |= v.abs() > f16::MAX.to_f32();
                    unsafe { *dp.add(i) = f16::from_f32(v).to_bits() };
                }
            }
            Dtype::F32 => {
                for (i, b) in src.chunks_exact(4).enumerate() {
                    let v = f32::from_le_bytes(b.try_into().unwrap());
                    clipped |= v.abs() > f16::MAX.to_f32();
                    unsafe { *dp.add(i) = f16::from_f32(v).to_bits() };
                }
            }
            other => return Err(format!("unsupported dtype {other:?} in {}", t.name).into()),
        }
        if clipped && !self.overflow_warned {
            self.overflow_warned = true;
            eprintln!(
                "lowmem: warning — {} holds values beyond f16 range; they clip at ±65504",
                t.name
            );
        }
        Ok(())
    }
}
