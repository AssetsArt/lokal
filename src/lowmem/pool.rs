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
use half::f16;
use metal::{Buffer, Device, MTLResourceOptions};
use rayon::prelude::*;
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
    /// Referenced by the command buffer currently being encoded.
    pinned: bool,
    /// The last command-buffer epoch that referenced this page — eviction
    /// refuses it until that epoch is marked completed.
    epoch: u64,
}

/// make_resident's verdict when the budget is tight.
pub(super) enum Admit {
    Ready,
    /// Every eviction candidate is still referenced by an in-flight command
    /// buffer — wait one out, mark it completed, and retry.
    NeedWait,
}

enum Evict {
    Done,
    AllInFlight,
    Exhausted,
}

pub(super) struct WeightPool {
    device: Device,
    budget: usize,
    used: usize,
    pages: HashMap<(u32, u32), Page>,
    clock: u64,
    /// Next command-buffer epoch to hand out (engine-global, monotonic).
    epoch: u64,
    /// Highest epoch whose command buffer has finished on the GPU.
    completed: u64,
    /// bf16 pages land as RAW BITS (memcpy) and convert on the GPU: each entry
    /// is a buffer awaiting a bf16_to_f16_inplace dispatch, which the session
    /// encodes at the head of the same command buffer that first reads it.
    pending_convert: Vec<(Buffer, usize)>,
    /// Evicted pages park their buffers here by size instead of freeing them:
    /// a fresh MTLBuffer costs an allocation plus a zero-fill page fault for
    /// every byte written — in thrash mode that doubled the staging traffic.
    /// Freelist bytes still count against the budget (free_bytes).
    free: HashMap<usize, Vec<Buffer>>,
    free_bytes: usize,
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
            epoch: 0,
            completed: 0,
            pending_convert: Vec::new(),
            free: HashMap::new(),
            free_bytes: 0,
            overflow_warned: false,
        }
    }

    /// Buffers staged since the last call that still hold raw bf16 bits — the
    /// caller encodes their conversion before any dispatch that reads them.
    pub fn take_pending_converts(&mut self) -> Vec<(Buffer, usize)> {
        std::mem::take(&mut self.pending_convert)
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget
    }

    /// A new command buffer's epoch. The caller stamps its pages with it via
    /// make_resident and reports completion through mark_completed.
    pub fn begin_cb(&mut self) -> u64 {
        self.epoch += 1;
        self.epoch
    }

    /// The command buffer for `epoch` finished on the GPU — its pages become
    /// eviction candidates. Command buffers on one queue complete in commit
    /// order, so marking the newest also covers everything older.
    pub fn mark_completed(&mut self, epoch: u64) {
        self.completed = self.completed.max(epoch);
    }

    /// Make every (tensor, block) page resident, pin it, and stamp it with the
    /// command buffer's epoch. Pinning happens in two passes so a staging miss
    /// can never evict a page this same call just resolved. NeedWait means the
    /// only eviction candidates belong to in-flight command buffers.
    pub fn make_resident(
        &mut self,
        mf: &WeightManifest,
        pages: &[(&PagedTensor, usize)],
        epoch: u64,
    ) -> crate::Result<Admit> {
        for (t, b) in pages {
            if let Some(p) = self.pages.get_mut(&(t.id, *b as u32)) {
                self.clock += 1;
                p.last_used = self.clock;
                p.pinned = true;
                p.epoch = epoch;
            }
        }
        for (t, b) in pages {
            let key = (t.id, *b as u32);
            if self.pages.contains_key(&key) {
                continue;
            }
            let need = t.block_bytes(*b);
            // Make room: a parked buffer of the RIGHT size satisfies the request
            // without growing the total; otherwise shrink (drop wrong-size parked
            // buffers, then evict — eviction PARKS, so a right-size victim is
            // picked up by the reuse check one iteration later).
            loop {
                if self.free.get(&need).is_some_and(|v| !v.is_empty()) {
                    break;
                }
                if self.used + self.free_bytes + need <= self.budget {
                    break;
                }
                if self.free_bytes > 0 {
                    let sz = self
                        .free
                        .iter()
                        .find(|(_, v)| !v.is_empty())
                        .map(|(s, _)| *s)
                        .expect("free_bytes > 0 implies a parked buffer");
                    self.free.get_mut(&sz).unwrap().pop();
                    self.free_bytes -= sz;
                    continue;
                }
                match self.evict_mru() {
                    Evict::Done => {}
                    Evict::AllInFlight => return Ok(Admit::NeedWait),
                    Evict::Exhausted => {
                        return Err(format!(
                            "weight pool exhausted: one encode set needs more than the {} MB budget — raise the memory budget",
                            self.budget >> 20
                        )
                        .into())
                    }
                }
            }
            let buf = match self.free.get_mut(&need).and_then(|v| v.pop()) {
                Some(b) => {
                    self.free_bytes -= need;
                    b // warm pages: no allocation, no zero-fill faults
                }
                None => self
                    .device
                    .new_buffer(need as u64, MTLResourceOptions::StorageModeShared),
            };
            self.stage(mf, t, *b, &buf)?;
            self.clock += 1;
            self.used += need;
            self.pages.insert(
                key,
                Page { buf, bytes: need, last_used: self.clock, pinned: true, epoch },
            );
        }
        Ok(Admit::Ready)
    }

    pub fn get(&self, t: &PagedTensor, block: usize) -> &Buffer {
        &self.pages[&(t.id, block as u32)].buf
    }

    /// The command buffer for the current encode is committed — pages stay
    /// protected by their epoch until mark_completed says the GPU is done.
    pub fn unpin_all(&mut self) {
        for p in self.pages.values_mut() {
            p.pinned = false;
        }
    }

    /// Evict the MOST-recently-used unpinned page. The pool's access pattern is
    /// a repeating scan (layer 0..L, every chunk and every token), and LRU is
    /// pessimal on loops — it evicts each page moments before its reuse, so a
    /// model over budget restaged WHOLE every token. MRU keeps a stable
    /// resident prefix (those pages never touch the bus again) and streams only
    /// the overflow through one hot slot: per-token traffic drops from
    /// model_bytes to model_bytes − pool_bytes.
    fn evict_mru(&mut self) -> Evict {
        let mut best: Option<(&(u32, u32), &Page)> = None;
        let mut in_flight_only = false;
        for (k, p) in &self.pages {
            if p.pinned {
                continue;
            }
            if p.epoch > self.completed {
                in_flight_only = true;
                continue;
            }
            if best.map(|(_, b)| p.last_used > b.last_used).unwrap_or(true) {
                best = Some((k, p));
            }
        }
        match best.map(|(k, _)| *k) {
            Some(key) => {
                let p = self.pages.remove(&key).unwrap();
                self.used -= p.bytes;
                self.free_bytes += p.bytes;
                self.free.entry(p.bytes).or_default().push(p.buf);
                Evict::Done
            }
            None if in_flight_only => Evict::AllInFlight,
            None => Evict::Exhausted,
        }
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
        // Rows convert in parallel: staging is the disk-side "CPU load" bar of
        // the overlap diagram, and a serial convert would cap it at ~2 GB/s.
        let out = unsafe { std::slice::from_raw_parts_mut(dp, rows * t.in_dim) };
        let clipped = match dtype {
            Dtype::F16 => {
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dp as *mut u8, src.len());
                }
                false
            }
            Dtype::BF16 => {
                // Raw bits only — the GPU converts in place before first use
                // (bf16_to_f16_inplace). The CPU's share of a bf16 page is a
                // copy, parallelized because a single core's ~10 GB/s memcpy is
                // the whole thrash-mode budget (~1 GB restaged per token when
                // the model exceeds the pool).
                let out_b =
                    unsafe { std::slice::from_raw_parts_mut(dp as *mut u8, src.len()) };
                out_b
                    .par_chunks_mut(1 << 20)
                    .zip(src.par_chunks(1 << 20))
                    .for_each(|(d, s)| d.copy_from_slice(s));
                self.pending_convert.push((dst.clone(), rows * t.in_dim));
                false
            }
            Dtype::F32 => out
                .par_chunks_mut(t.in_dim)
                .zip(src.par_chunks(t.in_dim * 4))
                .map(|(d, s)| {
                    let mut c = false;
                    for (o, b) in d.iter_mut().zip(s.chunks_exact(4)) {
                        let v = f32::from_le_bytes(b.try_into().unwrap());
                        c |= v.abs() > f16::MAX.to_f32();
                        *o = f16::from_f32(v).to_bits();
                    }
                    c
                })
                .reduce(|| false, |a, b| a | b),
            other => return Err(format!("unsupported dtype {other:?} in {}", t.name).into()),
        };
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
