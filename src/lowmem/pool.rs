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

    /// Decode admission for one block. Resident pages pin and bind from the
    /// pool; a block with budget room stages in (growing the resident prefix);
    /// past the budget line it binds DIRECT — matvec work reads each streamed
    /// byte once, so staging would only add traffic. Falls back to the staged
    /// path (with eviction) when the span has no usable GPU view.
    pub fn bind_decode(
        &mut self,
        mf: &WeightManifest,
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
            && mf.meta(&t.name)?.dtype == Dtype::BF16 // the direct pipes read raw bf16
        {
            let (r0, rows) = t.block_rows(block);
            if let Some((view, off)) = mf.gpu_span(&t.name, r0, r0 + rows)? {
                if off % 4 == 0 {
                    return Ok(Ok(Bind::Direct(view.clone(), off)));
                }
            }
        }
        match self.make_resident(mf, &[(t, block)], epoch)? {
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
                // The GPU stages the page itself, reading the checkpoint bytes
                // through the mmap's no-copy view — zero CPU bytes moved. The
                // CPU fallback (view missing or an unaligned span) converts in
                // parallel like the F32 path.
                if let Some((view, off)) = mf.gpu_span(&t.name, r0, r0 + rows)? {
                    self.pending_convert.push(PendingConvert {
                        src: view.clone(),
                        src_off: off,
                        dst: dst.clone(),
                        elems: rows * t.in_dim,
                    });
                    false
                } else {
                    out.par_chunks_mut(t.in_dim)
                        .zip(src.par_chunks(t.in_dim * 2))
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
    }

    impl QType {
        fn blk_elems(self) -> usize {
            match self {
                QType::Q8_0 | QType::Q4_0 => 32,
                QType::Q4K | QType::Q6K => 256,
            }
        }
        fn blk_bytes(self) -> usize {
            match self {
                QType::Q8_0 => 34,
                QType::Q4_0 => 18,
                QType::Q4K => 144,
                QType::Q6K => 210,
            }
        }
    }

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
                            QType::Q8_0 | QType::Q4_0 => b[0..2].copy_from_slice(&sb),
                            QType::Q4K => {
                                b[0..2].copy_from_slice(&sb);
                                b[2..4].copy_from_slice(&scale_bytes[3 - variant + 1]);
                            }
                            QType::Q6K => b[208..210].copy_from_slice(&sb),
                        }
                        if variant == 3 {
                            // max-magnitude quants under the max scale
                            match ty {
                                QType::Q8_0 => b[2..34].fill(0x80),
                                QType::Q4_0 => b[2..18].fill(0x0F),
                                QType::Q4K => {
                                    b[4..16].fill(0xFF); // all scale/min bits set
                                    b[16..144].fill(0xFF);
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

        for ty in [QType::Q8_0, QType::Q4_0, QType::Q4K, QType::Q6K] {
            let (n_blocks, n_rows) = (8usize, 3usize);
            let cols = n_blocks * ty.blk_elems();
            let row_bytes = n_blocks * ty.blk_bytes();
            let src = adversarial_rows(ty, n_blocks, n_rows);

            let mut want = vec![0f32; n_rows * cols];
            for r in 0..n_rows {
                ref_dequant_row(
                    ty,
                    &src[r * row_bytes..(r + 1) * row_bytes],
                    &mut want[r * cols..(r + 1) * cols],
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
