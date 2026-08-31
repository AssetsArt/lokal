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

use super::{LowMemSource, SrcType};
use half::f16;
use metal::{Buffer, Device, MTLResourceOptions};
use rayon::prelude::*;
use std::collections::HashMap;

/// Upper bound on one page's staged f16 bytes — row blocks are sized to it.
pub(super) const PAGE_BYTES: usize = 16 << 20;

/// One large weight matrix, paged by row blocks. `id` keys the pool map.
pub(super) struct PagedTensor {
    pub id: u32,
    pub name: String,
    pub in_dim: usize,
    pub out_dim: usize,
    /// What the CHECKPOINT holds. Quant pages stay quantized in the pool — the
    /// 4x residency is the whole point — so this drives page sizing, the
    /// pipeline selector, and whether a page can be bound direct.
    pub ty: SrcType,
    pub rows_per_page: usize,
    pub n_pages: usize,
    /// Eagerly-resident f16 bias; encode binds the engine's shared zero buffer
    /// when absent (same convention as the metal backend).
    pub bias: Option<Buffer>,
}

impl PagedTensor {
    pub fn new(
        src: &LowMemSource,
        id: u32,
        name: String,
        in_dim: usize,
        out_dim: usize,
        bias: Option<Buffer>,
    ) -> crate::Result<Self> {
        let shape = src.shape(&name)?;
        if shape != [out_dim, in_dim] {
            return Err(format!(
                "{name} has shape {shape:?} but the config implies [{out_dim}, {in_dim}]"
            )
            .into());
        }
        let ty = src.src_type(&name)?;
        if let SrcType::Quant(t) = ty {
            // A row must be a whole number of blocks: rows are the unit the pool
            // pages and the kernels index, and a block straddling two rows has
            // no meaning. K-quants block by 256, and every real checkpoint's
            // in_dim is a multiple of it — name the tensor when one is not.
            let be = t.blk_elems();
            if in_dim % be != 0 {
                return Err(format!(
                    "{name}: {t:?} blocks {be} elements but the row is {in_dim} wide \
                     ({in_dim} % {be} != 0) — this checkpoint cannot be paged"
                )
                .into());
            }
            if ty.qtype() == u32::MAX {
                return Err(format!("{name}: {t:?} has no GPU dequant path yet").into());
            }
        }
        // Pages are sized in CHECKPOINT bytes, so a Q4 page carries ~4x the rows
        // a bf16 page does at the same byte cost — which is the residency win.
        let rows_per_page = (PAGE_BYTES / ty.row_bytes(in_dim).max(1)).clamp(1, out_dim);
        Ok(Self {
            id,
            name,
            in_dim,
            out_dim,
            ty,
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

    /// Pool bytes of one block. Quant blocks are stored raw, so this is the
    /// checkpoint's own row size, not an f16 expansion.
    fn block_bytes(&self, block: usize) -> usize {
        self.block_rows(block).1 * self.ty.row_bytes(self.in_dim)
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

/// One GPU stage-in: convert `elems` bf16 values at `src_off` bytes into the
/// checkpoint's mmap view straight into the f16 pool page `dst`.
pub(super) struct PendingConvert {
    pub src: Buffer,
    pub src_off: usize,
    pub dst: Buffer,
    pub elems: usize,
}

/// Where one decode dispatch reads a weight block from.
pub(super) enum Bind {
    /// Staged f16 in the pool (resident or just admitted).
    Pool(Buffer),
    /// The checkpoint's raw bf16 through the mmap view at this byte offset —
    /// pages the budget has no room for stream from the page cache, read once,
    /// never staged and never evicting anything.
    Direct(Buffer, usize),
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
    /// bf16 pages stage GPU-side: each entry is a bf16_to_f16_copy dispatch
    /// (mmap view → pool page) that the session encodes at the head of the
    /// same command buffer that first reads the page. No CPU byte is moved.
    pending_convert: Vec<PendingConvert>,
    /// Evicted pages park their buffers here by size instead of freeing them:
    /// a fresh MTLBuffer costs an allocation plus a zero-fill page fault for
    /// every byte written — in thrash mode that doubled the staging traffic.
    /// Freelist bytes still count against the budget (free_bytes).
    free: HashMap<usize, Vec<Buffer>>,
    free_bytes: usize,
    /// f32→f16 clips past 65504 — flag an overflowing checkpoint once, at the
    /// first page that actually clips (cheaper than a full scan at open).
    overflow_warned: bool,
    /// LOKAL_LOWMEM_QDIRECT=1: let over-budget QUANT pages read straight from
    /// the checkpoint instead of staging. Correct in every run measured at the
    /// alignment gate below, but the gate's threshold is empirical rather than
    /// explained, so the default keeps quant pages on the staged path.
    qdirect: bool,
    /// Staging counters. A model that fits the pool must stage every page once
    /// and never again — the residency promise is exactly "zero stage-ins per
    /// decode step", and a gate cannot assert that from free text.
    stats: PoolStats,
}

/// What the pool has moved since it was built. Cheap to copy, so a caller can
/// snapshot it at a phase boundary and diff the two.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct PoolStats {
    pub stage_ins: u64,
    pub stage_bytes: u64,
    pub evictions: u64,
    /// Over-budget decode reads that bypassed the pool and streamed straight
    /// from the checkpoint. These move bus bytes without ever staging a page,
    /// so a stage-in count alone would call a streaming run resident.
    pub direct_binds: u64,
    pub direct_bytes: u64,
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
            qdirect: std::env::var("LOKAL_LOWMEM_QDIRECT").is_ok_and(|v| v == "1"),
            stats: PoolStats::default(),
        }
    }

    /// Staging counters so far — diff two snapshots to get one phase's traffic.
    pub fn stats(&self) -> PoolStats {
        self.stats
    }

    /// Stage-ins queued since the last call — the caller encodes them before
    /// any dispatch that reads the pages they fill.
    pub fn take_pending_converts(&mut self) -> Vec<PendingConvert> {
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
        src: &LowMemSource,
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
            self.stage(src, t, *b, &buf)?;
            self.stats.stage_ins += 1;
            self.stats.stage_bytes += need as u64;
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

    /// Decode admission for one block. Resident pages pin and bind from the
    /// pool; a block with budget room stages in (growing the resident prefix);
    /// past the budget line it binds DIRECT — matvec work reads each streamed
    /// byte once, so staging would only add traffic. Falls back to the staged
    /// path (with eviction) when the span has no usable GPU view.
    pub fn bind_decode(
        &mut self,
        src: &LowMemSource,
        t: &PagedTensor,
        block: usize,
        epoch: u64,
    ) -> crate::Result<Result<Bind, Admit>> {
        let key = (t.id, block as u32);
        if let Some(p) = self.pages.get_mut(&key) {
            self.clock += 1;
            p.last_used = self.clock;
            p.pinned = true;
            p.epoch = epoch;
            return Ok(Ok(Bind::Pool(p.buf.clone())));
        }
        let need = t.block_bytes(block);
        if self.used + self.free_bytes + need > self.budget
            // bf16 needs its own pipeline to read raw checkpoint bytes; a QUANT
            // page needs nothing special, because the pool page and the file
            // span hold the SAME quant blocks — the staged pipeline reads
            // either one. That is the streaming promise: over-budget quant
            // weights cross the bus as quant bytes, once.
            && (t.ty == SrcType::BF16 || (t.ty.is_quant() && self.qdirect))
        {
            let (r0, rows) = t.block_rows(block);
            if let Some((view, off)) = src.gpu_span(&t.name, r0, r0 + rows)? {
                // 64, not 4, for quant. MEASURED, not derived: on Qwen3-0.6B
                // Q4_K_M at a 200 MB pool, spans admitted at 4- or 16-byte
                // alignment decode into garbage while the SAME code at 64 and
                // 128 is exact, and Q4_K alone reproduces it, so it is not a
                // per-type bug. Every offset here is already 32-aligned (GGUF
                // aligns tensor data to 32 and the view base to a page), so
                // what fails is precisely the 32-mod-64 spans. Until the reason
                // is understood this path stays OFF by default — see qdirect.
                let need_align = if t.ty.is_quant() { 64 } else { 4 };
                if off % need_align == 0 {
                    self.stats.direct_binds += 1;
                    self.stats.direct_bytes += need as u64;
                    return Ok(Ok(Bind::Direct(view.clone(), off)));
                }
            }
        }
        match self.make_resident(src, &[(t, block)], epoch)? {
            Admit::Ready => Ok(Ok(Bind::Pool(self.get(t, block).clone()))),
            Admit::NeedWait => Ok(Err(Admit::NeedWait)),
        }
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
                self.stats.evictions += 1;
                Evict::Done
            }
            None if in_flight_only => Evict::AllInFlight,
            None => Evict::Exhausted,
        }
    }

    /// Which GGUF row holds HF row `r`, undoing llama.cpp's q/k permute.
///
/// The converter reshapes (n_head, 2, hd/2, cols) and swaps the middle axes, so
/// the file's row h*hd + d*2 + p carries HF's row h*hd + p*(hd/2) + d. We need
/// that read backwards — given the HF row we are filling, which file row holds
/// it — and the map is NOT an involution (it is one only at hd == 2), so
/// applying the forward direction here scrambles q/k into fluent nonsense.
/// The mapping never leaves its head.
fn unpermuted_src_row(r: usize, head_dim: usize) -> usize {
    let (h, q) = (r / head_dim, r % head_dim);
    let half = head_dim / 2;
    let (p, d) = (q / half, q % half);
    h * head_dim + d * 2 + p
}

/// Read the block's rows from the mmap and convert into the buffer — the
    /// ONLY place weight bytes are copied, and only ever one page's worth.
    fn stage(
        &mut self,
        src: &LowMemSource,
        t: &PagedTensor,
        block: usize,
        dst: &Buffer,
    ) -> crate::Result<()> {
        let (r0, rows) = t.block_rows(block);
        // llama.cpp stores llama-arch q/k under a row permute that matches
        // GGML's adjacent-pair RoPE; lokal rotates halves, so the permute is
        // undone HERE, as the page materializes. It is a pure row reorder, so
        // it works on quant blocks untouched — and it keeps the kernels clean,
        // which is the point: a compensating shuffle inside dequant would make
        // every future llama-arch GGUF silently wrong.
        let gathered;
        let bytes: &[u8] = match src.unpermute_head_dim(&t.name) {
            Some(hd) => {
                let rb = t.ty.row_bytes(t.in_dim);
                let mut v = Vec::with_capacity(rows * rb);
                for i in 0..rows {
                    let sr = Self::unpermuted_src_row(r0 + i, hd);
                    v.extend_from_slice(src.read_rows(&t.name, sr, sr + 1)?);
                }
                gathered = v;
                &gathered
            }
            None => src.read_rows(&t.name, r0, r0 + rows)?,
        };
        let dp = dst.contents() as *mut u16;
        // Quantized pages are stored EXACTLY as the checkpoint holds them: the
        // pool's whole reason to exist is that a Q4 model needs a quarter of the
        // bytes, and dequantizing at stage time would hand all of it back.
        if t.ty.is_quant() {
            assert_eq!(bytes.len(), t.block_bytes(block), "quant page size mismatch");
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dp as *mut u8, bytes.len());
            }
            return Ok(());
        }
        // The mmap slice can be unaligned (safetensors headers are arbitrary
        // lengths), so conversion walks byte pairs/quads — never typed pointers.
        // Rows convert in parallel: staging is the disk-side "CPU load" bar of
        // the overlap diagram, and a serial convert would cap it at ~2 GB/s.
        let out = unsafe { std::slice::from_raw_parts_mut(dp, rows * t.in_dim) };
        let clipped = match t.ty {
            SrcType::F16 => {
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), dp as *mut u8, bytes.len());
                }
                false
            }
            SrcType::BF16 => {
                // The GPU stages the page itself, reading the checkpoint bytes
                // through the mmap's no-copy view — zero CPU bytes moved. The
                // CPU fallback (view missing or an unaligned span) converts in
                // parallel like the F32 path.
                if let Some((view, off)) = src.gpu_span(&t.name, r0, r0 + rows)? {
                    self.pending_convert.push(PendingConvert {
                        src: view.clone(),
                        src_off: off,
                        dst: dst.clone(),
                        elems: rows * t.in_dim,
                    });
                    false
                } else {
                    out.par_chunks_mut(t.in_dim)
                        .zip(bytes.par_chunks(t.in_dim * 2))
                        .map(|(d, s)| {
                            let mut c = false;
                            for (o, b) in d.iter_mut().zip(s.chunks_exact(2)) {
                                let v = half::bf16::from_le_bytes([b[0], b[1]]).to_f32();
                                c |= v.abs() > f16::MAX.to_f32();
                                *o = f16::from_f32(v).to_bits();
                            }
                            c
                        })
                        .reduce(|| false, |a, b| a | b)
                }
            }
            SrcType::F32 => out
                .par_chunks_mut(t.in_dim)
                .zip(bytes.par_chunks(t.in_dim * 4))
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
            SrcType::Quant(_) => unreachable!("quant pages return above"),
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

#[cfg(test)]
mod permute_tests {
    use super::*;

    /// llama.cpp's forward permute, straight from convert_hf_to_gguf.py's
    /// reshape/swapaxes: file row `r` carries this HF row.
    fn forward(r: usize, hd: usize) -> usize {
        let (h, rem) = (r / hd, r % hd);
        let (d, p) = (rem / 2, rem % 2);
        h * hd + p * (hd / 2) + d
    }

    /// The two must compose to the identity. They did not once — the forward
    /// map was used for both directions, and a 135M model answered that the
    /// capital of Thailand is a city in the United States.
    #[test]
    fn unpermute_inverts_llama_cpp_permute() {
        for hd in [2usize, 4, 64, 128] {
            for r in 0..hd * 3 {
                assert_eq!(forward(WeightPool::unpermuted_src_row(r, hd), hd), r, "hd={hd} r={r}");
            }
        }
    }

    /// Every row maps somewhere distinct inside its own head — a permutation,
    /// not a collapse.
    #[test]
    fn unpermute_is_a_within_head_bijection() {
        let hd = 64;
        for h in 0..3 {
            let mut seen: Vec<usize> =
                (h * hd..(h + 1) * hd).map(|r| WeightPool::unpermuted_src_row(r, hd)).collect();
            seen.sort_unstable();
            assert_eq!(seen, (h * hd..(h + 1) * hd).collect::<Vec<_>>());
        }
    }
}

#[cfg(test)]
mod quant_oracle {
    //! The oracle gate (gguf-kernels D2, written FIRST per the lane order):
    //! GPU dequantization must match the CPU reference BIT-FOR-BIT on
    //! adversarial blocks. Until gguf-loader's seam freezes, `ref_dequant_row`
    //! below is a placeholder implementing exact ggml semantics (transcribed
    //! from ggml-quants.c); at seam-freeze it is swapped for
    //! `manifest::dequant_row_ref` in one line — and because this shim was
    //! derived independently of Tiësto's, the swap also cross-checks HIS
    //! implementation against ggml.
    //!
    //! Negative control (run once, 2026-08-31): planting the classic
    //! get_scale_min_k4 bug (q[j+4] instead of q[j] donating the min's top
    //! bits) fails the gate at Q4_K block 1 elem 128 — the gate can fail.

    use crate::gpu::metal as gpu;
    use half::f16;
    use metal::{CompileOptions, Device, FunctionConstantValues, MTLDataType, MTLResourceOptions, MTLSize};

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum QType {
        Q8_0 = 2,
        Q4_0 = 3,
        Q4K = 4,
        Q6K = 5,
        Q5K = 6,
        Q5_0 = 7,
        Q2K = 8,
        Q3K = 9,
        IQ4NL = 10,
        IQ4XS = 11,
        IQ3XXS = 12,
        IQ3S = 13,
    }

    impl QType {
        /// The seam's enum for the same encoding. The oracle keys types by the
        /// kernel's LM_W_QTYPE selector, which is deliberately not ggml's id.
        fn seam(self) -> super::super::manifest::GgmlType {
            use super::super::manifest::GgmlType as G;
            match self {
                QType::Q8_0 => G::Q8_0,
                QType::Q4_0 => G::Q4_0,
                QType::IQ3XXS => G::IQ3_XXS,
                QType::IQ3S => G::IQ3_S,
                QType::IQ4NL => G::IQ4_NL,
                QType::IQ4XS => G::IQ4_XS,
                QType::Q2K => G::Q2_K,
                QType::Q3K => G::Q3_K,
                QType::Q4K => G::Q4_K,
                QType::Q6K => G::Q6_K,
                QType::Q5K => G::Q5_K,
                QType::Q5_0 => G::Q5_0,
            }
        }

        fn blk_elems(self) -> usize {
            match self {
                QType::Q8_0 | QType::Q4_0 | QType::Q5_0 | QType::IQ4NL => 32,
                QType::Q4K | QType::Q6K | QType::Q5K | QType::Q2K | QType::Q3K
                | QType::IQ4XS | QType::IQ3XXS | QType::IQ3S => 256,
            }
        }
        fn blk_bytes(self) -> usize {
            match self {
                QType::Q8_0 => 34,
                QType::Q4_0 => 18,
                QType::Q5_0 => 22,
                QType::Q4K => 144,
                QType::Q6K => 210,
                QType::Q5K => 176,
                QType::Q2K => 84,
                QType::Q3K => 110,
                QType::IQ4NL => 18,
                QType::IQ4XS => 136,
                QType::IQ3XXS => 98,
                QType::IQ3S => 110,
            }
        }
    }

    /// The seam's codebook, referenced (not copied) so a divergence is
    /// impossible by construction — the shim's independence is in how it
    /// INDEXES the table, not in retyping 16 magic numbers.
    use super::super::iq_grids::{
        IQ3S_GRID as IQ3S, IQ3XXS_GRID as IQ3XXS, KMASK_IQ2XS as KMASK, KSIGNS_IQ2XS as KSIGNS,
    };
    use super::super::manifest::KVALUES_IQ4NL as KV;

    fn f16_at(b: &[u8]) -> f32 {
        f16::from_le_bytes([b[0], b[1]]).to_f32()
    }

    // ggml's get_scale_min_k4, exactly — including q[j] (not q[j+4]) donating
    // the min's top bits in the second half.
    fn scale_min_k4(j: usize, q: &[u8]) -> (u32, u32) {
        if j < 4 {
            ((q[j] & 63) as u32, (q[j + 4] & 63) as u32)
        } else {
            (
                ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)) as u32,
                ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as u32,
            )
        }
    }

    /// Placeholder for the seam's dequant_row_ref (exact ggml semantics,
    /// strict IEEE f32 — no fma, same expression shapes as ggml-quants.c).
    fn ref_dequant_row(ty: QType, src: &[u8], out: &mut [f32]) {
        let (be, bb) = (ty.blk_elems(), ty.blk_bytes());
        assert_eq!(out.len() % be, 0);
        assert_eq!(src.len(), out.len() / be * bb);
        for (blk, y) in src.chunks_exact(bb).zip(out.chunks_exact_mut(be)) {
            match ty {
                // Q2_K and Q3_K, written in ggml's SEQUENTIAL loop order
                // (n, j, l writing forward) rather than the seam's per-element
                // index decomposition. Two different readings of the same C: if
                // they agree the reading is right, which one reference alone
                // can never tell you.
                // IQ4 pair, in ggml's sequential loop order rather than the
                // seam's per-element decomposition.
                // The IQ3 pair in ggml's sequential order: walk ib32 groups
                // forward writing y as it goes, rather than the seam's
                // per-element index decomposition.
                QType::IQ3XXS => {
                    let d = f16_at(blk);
                    let mut o = 0usize;
                    for ib32 in 0..8 {
                        let sas = 2 + 64 + 4 * ib32;
                        let aux32 =
                            u32::from_le_bytes([blk[sas], blk[sas + 1], blk[sas + 2], blk[sas + 3]]);
                        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                        for l in 0..4 {
                            let signs = KSIGNS[((aux32 >> (7 * l)) & 127) as usize];
                            let g1 = IQ3XXS[blk[2 + ib32 * 8 + 2 * l] as usize];
                            let g2 = IQ3XXS[blk[2 + ib32 * 8 + 2 * l + 1] as usize];
                            for j in 0..4 {
                                let s1 = if signs & KMASK[j] != 0 { -1.0 } else { 1.0 };
                                let s2 = if signs & KMASK[j + 4] != 0 { -1.0 } else { 1.0 };
                                y[o + j] = db * ((g1 >> (8 * j)) & 0xFF) as f32 * s1;
                                y[o + j + 4] = db * ((g2 >> (8 * j)) & 0xFF) as f32 * s2;
                            }
                            o += 8;
                        }
                    }
                }
                QType::IQ3S => {
                    let d = f16_at(blk);
                    let mut o = 0usize;
                    for g in 0..8usize {
                        let sc_byte = blk[106 + g / 2];
                        let sc = if g % 2 == 0 { sc_byte & 0xF } else { sc_byte >> 4 };
                        let db = d * (1 + 2 * sc as u32) as f32;
                        let qh = blk[66 + g] as usize;
                        for l in 0..4usize {
                            let i1 = blk[2 + g * 8 + 2 * l] as usize | ((qh << (8 - 2 * l)) & 256);
                            let i2 = blk[2 + g * 8 + 2 * l + 1] as usize | ((qh << (7 - 2 * l)) & 256);
                            let (g1, g2) = (IQ3S[i1], IQ3S[i2]);
                            let sg = blk[74 + g * 4 + l];
                            for j in 0..4 {
                                let s1 = if sg & KMASK[j] != 0 { -1.0 } else { 1.0 };
                                let s2 = if sg & KMASK[j + 4] != 0 { -1.0 } else { 1.0 };
                                y[o + j] = db * ((g1 >> (8 * j)) & 0xFF) as f32 * s1;
                                y[o + j + 4] = db * ((g2 >> (8 * j)) & 0xFF) as f32 * s2;
                            }
                            o += 8;
                        }
                    }
                }
                QType::IQ4NL => {
                    let d = f16_at(blk);
                    for j in 0..16 {
                        let q = blk[2 + j];
                        y[j] = d * KV[(q & 0xF) as usize] as f32;
                        y[j + 16] = d * KV[(q >> 4) as usize] as f32;
                    }
                }
                QType::IQ4XS => {
                    let d = f16_at(blk);
                    let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
                    let mut o = 0usize;
                    for ib in 0..8 {
                        let ls = ((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
                            | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
                        let dl = d * (ls - 32) as f32;
                        let qs = &blk[8 + ib * 16..8 + ib * 16 + 16];
                        for j in 0..16 {
                            y[o + j] = dl * KV[(qs[j] & 0xF) as usize] as f32;
                            y[o + j + 16] = dl * KV[(qs[j] >> 4) as usize] as f32;
                        }
                        o += 32;
                    }
                }
                QType::Q2K => {
                    let d = f16_at(&blk[80..]);
                    let dmin = f16_at(&blk[82..]);
                    let mut o = 0usize;
                    let mut is = 0usize;
                    for n in 0..2 {
                        let q = &blk[16 + n * 32..16 + n * 32 + 32];
                        for j in 0..4 {
                            for half in 0..2 {
                                let sc = blk[is];
                                is += 1;
                                let dl = d * (sc & 0xF) as f32;
                                let ml = dmin * (sc >> 4) as f32;
                                for l in 0..16 {
                                    let v = (q[half * 16 + l] >> (2 * j)) & 3;
                                    y[o] = dl * v as f32 - ml;
                                    o += 1;
                                }
                            }
                        }
                    }
                }
                QType::Q3K => {
                    let d_all = f16_at(&blk[108..]);
                    // Same 12->16 six-bit unpack, but read one index at a time
                    // instead of ggml's four-word shuffle.
                    let sc_at = |k: usize| -> u8 {
                        let (g, m) = (k / 4, k % 4);
                        let lo = if g < 2 {
                            blk[96 + 4 * g + m] & 0xF
                        } else {
                            (blk[96 + 4 * (g - 2) + m] >> 4) & 0xF
                        };
                        let hi = (blk[96 + 8 + m] >> (2 * g)) & 3;
                        lo | (hi << 4)
                    };
                    let mut o = 0usize;
                    let mut is = 0usize;
                    let mut m: u8 = 1;
                    for n in 0..2 {
                        let q = &blk[32 + n * 32..32 + n * 32 + 32];
                        for j in 0..4 {
                            for half in 0..2 {
                                let dl = d_all * (sc_at(is) as i32 - 32) as f32;
                                is += 1;
                                for l in 0..16 {
                                    let v = ((q[half * 16 + l] >> (2 * j)) & 3) as i32;
                                    let hi = if blk[half * 16 + l] & m != 0 { 0 } else { 4 };
                                    y[o] = dl * (v - hi) as f32;
                                    o += 1;
                                }
                            }
                            m <<= 1;
                        }
                    }
                }
                QType::Q8_0 => {
                    let d = f16_at(blk);
                    for j in 0..32 {
                        y[j] = (blk[2 + j] as i8) as f32 * d;
                    }
                }
                QType::Q4_0 => {
                    let d = f16_at(blk);
                    for j in 0..16 {
                        let x0 = (blk[2 + j] & 0x0F) as i32 - 8;
                        let x1 = (blk[2 + j] >> 4) as i32 - 8;
                        y[j] = x0 as f32 * d;
                        y[j + 16] = x1 as f32 * d;
                    }
                }
                QType::Q4K => {
                    let d = f16_at(blk);
                    let min = f16_at(&blk[2..]);
                    let scales = &blk[4..16];
                    let qs = &blk[16..144];
                    let mut yy = 0usize;
                    let mut is = 0usize;
                    let mut qoff = 0usize;
                    for _ in (0..256).step_by(64) {
                        let (sc, m) = scale_min_k4(is, scales);
                        let d1 = d * sc as f32;
                        let m1 = min * m as f32;
                        let (sc, m) = scale_min_k4(is + 1, scales);
                        let d2 = d * sc as f32;
                        let m2 = min * m as f32;
                        for l in 0..32 {
                            y[yy] = d1 * (qs[qoff + l] & 0xF) as f32 - m1;
                            yy += 1;
                        }
                        for l in 0..32 {
                            y[yy] = d2 * (qs[qoff + l] >> 4) as f32 - m2;
                            yy += 1;
                        }
                        qoff += 32;
                        is += 2;
                    }
                }
                QType::Q5_0 => {
                    let d = f16_at(blk);
                    let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
                    for j in 0..16 {
                        let xh_0 = ((qh >> j) << 4) & 0x10;
                        let xh_1 = (qh >> (j + 12)) & 0x10;
                        let x0 = ((blk[6 + j] & 0x0F) as u32 | xh_0) as i32 - 16;
                        let x1 = ((blk[6 + j] >> 4) as u32 | xh_1) as i32 - 16;
                        y[j] = x0 as f32 * d;
                        y[j + 16] = x1 as f32 * d;
                    }
                }
                QType::Q5K => {
                    let d = f16_at(blk);
                    let min = f16_at(&blk[2..]);
                    let scales = &blk[4..16];
                    let qh = &blk[16..48];
                    let qs = &blk[48..176];
                    let mut is = 0usize;
                    let (mut u1, mut u2) = (1u8, 2u8);
                    for j in (0..256).step_by(64) {
                        let ql = &qs[j / 2..j / 2 + 32];
                        let (sc, m) = scale_min_k4(is, scales);
                        let d1 = d * sc as f32;
                        let m1 = min * m as f32;
                        let (sc, m) = scale_min_k4(is + 1, scales);
                        let d2 = d * sc as f32;
                        let m2 = min * m as f32;
                        for l in 0..32 {
                            let hi1: u32 = if qh[l] & u1 != 0 { 16 } else { 0 };
                            let hi2: u32 = if qh[l] & u2 != 0 { 16 } else { 0 };
                            y[j + l] = d1 * ((ql[l] & 0x0F) as u32 + hi1) as f32 - m1;
                            y[j + 32 + l] = d2 * ((ql[l] >> 4) as u32 + hi2) as f32 - m2;
                        }
                        is += 2;
                        u1 <<= 2;
                        u2 <<= 2;
                    }
                }
                QType::Q6K => {
                    let d = f16_at(&blk[208..]);
                    for n in (0..256).step_by(128) {
                        let ql = &blk[n / 2..];
                        let qh = &blk[128 + n / 4..];
                        let sc = &blk[192 + n / 16..];
                        for l in 0..32 {
                            let is = l / 16;
                            let q1 = ((ql[l] & 0xF) as i32 | (((qh[l] >> 0) & 3) as i32) << 4) - 32;
                            let q2 = ((ql[l + 32] & 0xF) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
                            let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
                            let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;
                            y[n + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                            y[n + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                            y[n + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                            y[n + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                        }
                    }
                }
            }
        }
    }

    /// Adversarial rows: all-zero, subnormal scales with extreme quants, max
    /// f16 scale with max-magnitude quants, and dense LCG-patterned bytes that
    /// exercise every scale high-bit combination. Inf/NaN f16 scales are out
    /// of domain (no real checkpoint carries them; NaN payloads differ across
    /// engines) and deliberately excluded.
    fn adversarial_rows(ty: QType, n_blocks: usize, n_rows: usize) -> Vec<u8> {
        let bb = ty.blk_bytes();
        let mut out = Vec::with_capacity(n_rows * n_blocks * bb);
        let mut lcg: u32 = 0x2545_F491;
        let mut next = || {
            lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
            (lcg >> 16) as u8
        };
        let scale_bytes: [[u8; 2]; 4] = [
            f16::from_f32(0.0).to_le_bytes(),
            [0x01, 0x00],                       // smallest positive subnormal
            [0xFF, 0x83],                       // negative subnormal
            f16::MAX.to_le_bytes(),             // 65504
        ];
        for row in 0..n_rows {
            for blk in 0..n_blocks {
                let variant = (row * n_blocks + blk) % 4;
                let mut b = vec![0u8; bb];
                match variant {
                    0 => {} // all-zero block
                    _ => {
                        for x in b.iter_mut() {
                            *x = next();
                        }
                        let sb = scale_bytes[variant];
                        match ty {
                            QType::Q8_0 | QType::Q4_0 | QType::Q5_0 => b[0..2].copy_from_slice(&sb),
                            QType::Q4K | QType::Q5K => {
                                b[0..2].copy_from_slice(&sb);
                                b[2..4].copy_from_slice(&scale_bytes[3 - variant + 1]);
                            }
                            QType::Q6K => b[208..210].copy_from_slice(&sb),
                            // Q2_K's d/dmin sit at the END of the block, not the start.
                            QType::Q2K => {
                                b[80..82].copy_from_slice(&sb);
                                b[82..84].copy_from_slice(&scale_bytes[3 - variant + 1]);
                            }
                            QType::Q3K => b[108..110].copy_from_slice(&sb),
                            QType::IQ4NL | QType::IQ4XS | QType::IQ3XXS | QType::IQ3S => {
                                b[0..2].copy_from_slice(&sb)
                            }
                        }
                        if variant == 3 {
                            // max-magnitude quants under the max scale
                            match ty {
                                QType::Q8_0 => b[2..34].fill(0x80),
                                QType::Q4_0 => b[2..18].fill(0x0F),
                                QType::Q5_0 => b[2..22].fill(0xFF), // qh + nibbles all set
                                QType::Q4K => {
                                    b[4..16].fill(0xFF); // all scale/min bits set
                                    b[16..144].fill(0xFF);
                                }
                                QType::Q5K => {
                                    b[4..176].fill(0xFF); // scales, high bits, nibbles
                                }
                                // Grid indices stay RANDOM (a filled 0xFF index
                                // is a valid but singular grid entry); what the
                                // max case drives is scales and signs.
                                QType::IQ3XXS => b[66..98].fill(0xFF),
                                QType::IQ3S => {
                                    b[66..74].fill(0xFF);  // qh: ninth bit set everywhere
                                    b[74..106].fill(0xFF); // all signs negative
                                    b[106..110].fill(0xFF); // scales maxed
                                }
                                QType::IQ4NL => b[2..18].fill(0xFF), // codebook index 15 everywhere
                                QType::IQ4XS => {
                                    b[2..8].fill(0xFF);   // scales_h + scales_l all set -> ls 63
                                    b[8..136].fill(0xFF); // every nibble -> codebook 15
                                }
                                QType::Q2K => {
                                    b[0..16].fill(0xFF);  // every scale AND min maxed
                                    b[16..80].fill(0xFF); // every 2-bit quant = 3
                                }
                                QType::Q3K => {
                                    b[0..32].fill(0x00);  // hmask CLEAR: the -4 path everywhere
                                    b[32..96].fill(0xFF); // quants all 3
                                    b[96..108].fill(0xFF); // scales all 63 -> +31 after bias
                                }
                                QType::Q6K => {
                                    b[0..192].fill(0xFF);
                                    b[192..208].fill(0x80); // scales = -128
                                }
                            }
                        }
                    }
                }
                out.extend_from_slice(&b);
            }
        }
        out
    }

    #[test]
    fn gpu_dequant_matches_reference_bit_for_bit() {
        let device = Device::system_default().expect("Metal device required for the oracle");
        // PRECISE library: fast-math would license fma fusion and reordering,
        // and the gate is bit equality with strict-IEEE Rust — the shipped
        // quant pipelines are built from this same fast-math-off source.
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(&gpu::shader_source(128), &opts)
            .expect("kernels.metal compiles");

        for ty in [
            QType::Q8_0,
            QType::Q4_0,
            QType::Q5_0,
            QType::Q4K,
            QType::Q6K,
            QType::Q5K,
            QType::Q2K,
            QType::Q3K,
            QType::IQ4NL,
            QType::IQ4XS,
            QType::IQ3XXS,
            QType::IQ3S,
        ] {
            let (n_blocks, n_rows) = (8usize, 3usize);
            let cols = n_blocks * ty.blk_elems();
            let row_bytes = n_blocks * ty.blk_bytes();
            let src = adversarial_rows(ty, n_blocks, n_rows);

            // THREE-way, on purpose. `want` is the SEAM's reference — the one
            // production actually calls — while the shim below was written
            // independently from ggml-quants.c before the seam existed. GPU vs
            // seam catches a kernel bug; shim vs seam catches a shared
            // misreading of ggml, which no single reference can catch alone.
            let mut want = vec![0f32; n_rows * cols];
            let mut shim = vec![0f32; n_rows * cols];
            for r in 0..n_rows {
                let row = &src[r * row_bytes..(r + 1) * row_bytes];
                super::super::manifest::dequant_row_ref(
                    ty.seam(),
                    row,
                    &mut want[r * cols..(r + 1) * cols],
                );
                ref_dequant_row(ty, row, &mut shim[r * cols..(r + 1) * cols]);
            }
            for (i, (a, b)) in want.iter().zip(&shim).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{ty:?}: seam dequant_row_ref and the independent shim disagree at elem {i} \
                     ({a} vs {b}) — one of them misreads ggml-quants.c"
                );
            }

            let consts = FunctionConstantValues::new();
            let tyv = ty as u32;
            consts.set_constant_value_at_index(&tyv as *const u32 as *const _, MTLDataType::UInt, 25);
            let f = lib.get_function("lm_dequant_oracle", Some(consts)).expect("oracle fn");
            let pipe = device.new_compute_pipeline_state_with_function(&f).expect("oracle pipe");

            let src_buf = device.new_buffer_with_data(
                src.as_ptr() as *const _,
                src.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let out_buf = device.new_buffer(
                (want.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let queue = device.new_command_queue();
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&src_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            let p: [u32; 3] = [cols as u32, row_bytes as u32, n_rows as u32];
            enc.set_bytes(2, 12, p.as_ptr() as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(cols.div_ceil(32) as u64, n_rows as u64, 1),
                MTLSize::new(32, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let got = unsafe {
                std::slice::from_raw_parts(out_buf.contents() as *const f32, want.len())
            };
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    g.to_bits() == w.to_bits(),
                    "{ty:?}: col {} row {} (block {}, elem {}): gpu {g:e} ({:#010x}) != ref {w:e} ({:#010x})",
                    i % cols,
                    i / cols,
                    (i % cols) / ty.blk_elems(),
                    (i % cols) % ty.blk_elems(),
                    g.to_bits(),
                    w.to_bits(),
                );
            }
        }
    }
}
