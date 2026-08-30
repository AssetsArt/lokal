//! The lowmem forward pass — per-layer command buffers over pooled weight pages.
//!
//! Structure mirrors MetalSession::run_from dispatch by dispatch, with three
//! deliberate differences:
//!   - every weight matmul walks a paged tensor block by block (matmul_pg /
//!     matvec with buffer offsets) — nothing assumes a whole tensor resident;
//!   - one command buffer per LAYER, waited on before the next opens, so the
//!     pool's pin set is exact (per-layer overlap arrives in the next phase);
//!   - the embedding lookup is a CPU-side gather straight from the mmap: a
//!     chunk touches at most `chunk` scattered rows, which never justifies
//!     keeping a vocab × hidden table resident.

use super::pool::PagedTensor;
use super::pool::WeightPool;
use super::{LayerWeights, LowMemEngine};
use crate::engine::Session;
use crate::gpu::metal as gpu;
use half::{bf16, f16};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use safetensors::Dtype;

// ---- kernel parameter structs: byte-exact mirrors of kernels.metal ----
// (kernels.metal is the contract; the metal backend keeps its own copies.)

#[repr(C)]
struct MatvecParams {
    in_dim: u32,
    out_dim: u32,
}
#[repr(C)]
struct MatmulPagedParams {
    in_dim: u32,
    out_dim: u32,
    n_rows: u32,
    y_stride: u32,
}
#[repr(C)]
struct NormParams {
    dim: u32,
    eps: f32,
}
#[repr(C)]
struct RopeQkPrefillParams {
    head_dim: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    pos0: u32,
    theta: f32,
    n_rows: u32,
}
#[repr(C)]
struct RopeQkParams {
    head_dim: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    pos: u32,
    theta: f32,
}
#[repr(C)]
struct AttnParams {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    pos0: u32,
    max_seq: u32,
    n_rows: u32,
}
#[repr(C)]
struct AttnDecParams {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    pos: u32,
    n_splits: u32,
}
#[repr(C)]
struct ElemParams {
    dim: u32,
}

fn set_bytes<T>(enc: &ComputeCommandEncoderRef, idx: u64, v: &T) {
    enc.set_bytes(idx, size_of::<T>() as u64, v as *const T as *const _);
}

pub(super) struct LowMemSession<'a> {
    e: &'a LowMemEngine,
    max_seq: usize,
    chunk: usize,
    k_cache: Vec<Buffer>, // per layer: f16 [max_seq (+ flash slack) × kv_dim]
    v_cache: Vec<Buffer>,
    x: Buffer,
    xn: Buffer,
    q: Buffer,
    att: Buffer,
    xb: Buffer,
    gate: Buffer,
    up: Buffer,
    /// f32 staging for the chunk's fresh K then V rows (converted into the
    /// caches by f32_to_f16) — [chunk × kv_dim] each half.
    kvs: Buffer,
    /// The flash kernel also emits a half copy of its output; lowmem's o_proj
    /// reads the f32 one, so this is write-only ballast.
    xh: Buffer,
    scores: Buffer,
    partials: Buffer,
    logits: Buffer, // one row: [vocab]
}

impl<'a> LowMemSession<'a> {
    pub fn new(e: &'a LowMemEngine, max_seq: usize) -> Self {
        let cfg = &e.cfg;
        let d = &e.device;
        let chunk = gpu::PREFILL_CHUNK.min(max_seq);
        let (h, kvd) = (cfg.hidden_size, cfg.kv_dim());
        Self {
            k_cache: (0..cfg.num_hidden_layers)
                .map(|_| gpu::f16_empty_buffer(d, (max_seq + gpu::FLASH_C) * kvd))
                .collect(),
            v_cache: (0..cfg.num_hidden_layers)
                .map(|_| gpu::f16_empty_buffer(d, (max_seq + gpu::FLASH_C) * kvd))
                .collect(),
            x: gpu::f32_buffer(d, chunk * h),
            xn: gpu::f32_buffer(d, chunk * h),
            q: gpu::f32_buffer(d, chunk * h),
            att: gpu::f32_buffer(d, chunk * h),
            xb: gpu::f32_buffer(d, chunk * h),
            gate: gpu::f32_buffer(d, chunk * cfg.intermediate_size),
            up: gpu::f32_buffer(d, chunk * cfg.intermediate_size),
            kvs: gpu::f32_buffer(d, 2 * chunk * kvd),
            xh: gpu::f16_empty_buffer(d, chunk * h),
            scores: if cfg.head_dim() == gpu::FLASH_HEAD_DIM {
                gpu::f32_buffer(d, 1) // flash path never reads it — stub binding
            } else {
                gpu::f32_buffer(d, chunk * cfg.num_attention_heads * max_seq)
            },
            partials: gpu::f32_buffer(
                d,
                cfg.num_attention_heads
                    * max_seq.div_ceil(gpu::ATTN_SPLIT)
                    * (cfg.head_dim() + 2),
            ),
            logits: gpu::f32_buffer(d, cfg.vocab_size),
            e,
            max_seq,
            chunk,
        }
    }

    /// CPU-side embedding gather: one mmap row per token, converted straight
    /// into the x buffer (unified memory).
    fn embed_gather(&self, ids: &[u32]) -> crate::Result<()> {
        let h = self.e.cfg.hidden_size;
        let xp = self.x.contents() as *mut f32;
        for (i, &id) in ids.iter().enumerate() {
            let (row, dtype) =
                self.e.manifest.read_rows("model.embed_tokens.weight", id as usize, id as usize + 1)?;
            let dst = unsafe { std::slice::from_raw_parts_mut(xp.add(i * h), h) };
            match dtype {
                Dtype::F32 => {
                    for (o, b) in dst.iter_mut().zip(row.chunks_exact(4)) {
                        *o = f32::from_le_bytes(b.try_into().unwrap());
                    }
                }
                Dtype::BF16 => {
                    for (o, b) in dst.iter_mut().zip(row.chunks_exact(2)) {
                        *o = bf16::from_le_bytes([b[0], b[1]]).to_f32();
                    }
                }
                Dtype::F16 => {
                    for (o, b) in dst.iter_mut().zip(row.chunks_exact(2)) {
                        *o = f16::from_le_bytes([b[0], b[1]]).to_f32();
                    }
                }
                other => return Err(format!("unsupported embed dtype {other:?}").into()),
            }
        }
        Ok(())
    }

    // ---- encode helpers ----

    fn enc_rmsnorm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_off: u64,
        weight: &Buffer,
        y: &Buffer,
        n_rows: usize,
    ) {
        let p = NormParams { dim: self.e.cfg.hidden_size as u32, eps: self.e.cfg.rms_norm_eps };
        enc.set_compute_pipeline_state(&self.e.pipes.rmsnorm);
        enc.set_buffer(0, Some(x), x_off);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(y), 0);
        set_bytes(enc, 3, &p);
        enc.dispatch_thread_groups(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Prefill Y = X·Wᵀ + bias over every block of a paged tensor. `y_stride`
    /// is the logical tensor's full out_dim; each block writes its column span.
    fn enc_matmul_paged(
        &self,
        enc: &ComputeCommandEncoderRef,
        pool: &WeightPool,
        t: &PagedTensor,
        x: &Buffer,
        y: &Buffer,
        n_rows: usize,
        y_base: u64,
        y_stride: usize,
    ) {
        for blk in 0..t.n_pages {
            let (r0, rows) = t.block_rows(blk);
            let p = MatmulPagedParams {
                in_dim: t.in_dim as u32,
                out_dim: rows as u32,
                n_rows: n_rows as u32,
                y_stride: y_stride as u32,
            };
            enc.set_compute_pipeline_state(&self.e.pipes.matmul_pg);
            enc.set_buffer(0, Some(pool.get(t, blk)), 0);
            match &t.bias {
                Some(b) => enc.set_buffer(1, Some(b), (r0 * 2) as u64),
                None => enc.set_buffer(1, Some(&self.e.zero_bias), 0),
            }
            enc.set_buffer(2, Some(x), 0);
            enc.set_buffer(3, Some(y), y_base + (r0 * 4) as u64);
            set_bytes(enc, 4, &p);
            enc.dispatch_thread_groups(
                MTLSize::new((rows as u64).div_ceil(64), (n_rows as u64).div_ceil(32), 1),
                MTLSize::new(128, 1, 1),
            );
        }
    }

    /// Decode y[r0..] = W_block·x + bias per block — matvec / matvec_h /
    /// matvec_acc all share the binding layout. `y_elem` is the output's
    /// element size in bytes (4 for f32, 2 for the f16 caches).
    #[allow(clippy::too_many_arguments)]
    fn enc_matvec_paged(
        &self,
        enc: &ComputeCommandEncoderRef,
        pool: &WeightPool,
        pipe: &ComputePipelineState,
        t: &PagedTensor,
        x: &Buffer,
        y: &Buffer,
        y_base: u64,
        y_elem: u64,
    ) {
        for blk in 0..t.n_pages {
            let (r0, rows) = t.block_rows(blk);
            let p = MatvecParams { in_dim: t.in_dim as u32, out_dim: rows as u32 };
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(pool.get(t, blk)), 0);
            match &t.bias {
                Some(b) => enc.set_buffer(1, Some(b), (r0 * 2) as u64),
                None => enc.set_buffer(1, Some(&self.e.zero_bias), 0),
            }
            enc.set_buffer(2, Some(x), 0);
            enc.set_buffer(3, Some(y), y_base + r0 as u64 * y_elem);
            set_bytes(enc, 4, &p);
            gpu::dispatch_simdgroup_rows(enc, rows as u32);
        }
    }

    fn enc_f32_to_f16(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &Buffer,
        src_off: u64,
        dst: &Buffer,
        dst_off: u64,
        dim: usize,
    ) {
        let d = dim as u32;
        enc.set_compute_pipeline_state(&self.e.pipes.f32_to_f16);
        enc.set_buffer(0, Some(src), src_off);
        enc.set_buffer(1, Some(dst), dst_off);
        set_bytes(enc, 2, &d);
        gpu::dispatch_grid(enc, dim);
    }

    fn enc_elem(
        &self,
        enc: &ComputeCommandEncoderRef,
        pipe: &ComputePipelineState,
        a: &Buffer,
        b: &Buffer,
        dim: usize,
    ) {
        let p = ElemParams { dim: dim as u32 };
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        set_bytes(enc, 2, &p);
        gpu::dispatch_grid(enc, dim);
    }

    /// The prefill body of one layer (also serves head_dims outside the fused
    /// decode's reach at n_rows == 1). Mirrors MetalSession::run_from's prefill
    /// arm minus the tensor-ops staging.
    #[allow(clippy::too_many_arguments)]
    fn encode_layer_prefill(
        &self,
        enc: &ComputeCommandEncoderRef,
        pool: &WeightPool,
        lw: &LayerWeights,
        l: usize,
        pos0: usize,
        n: usize,
        kv_byte_off: u64,
    ) {
        let e = self.e;
        let cfg = &e.cfg;
        let (h, hd, kvd) = (cfg.hidden_size, cfg.head_dim(), cfg.kv_dim());
        let v_base = (self.chunk * kvd * 4) as u64; // V's half of the kvs staging

        // Attention half.
        self.enc_rmsnorm(enc, &self.x, 0, &lw.input_ln, &self.xn, n);
        self.enc_matmul_paged(enc, pool, &lw.q, &self.xn, &self.q, n, 0, h);
        self.enc_matmul_paged(enc, pool, &lw.k, &self.xn, &self.kvs, n, 0, kvd);
        self.enc_matmul_paged(enc, pool, &lw.v, &self.xn, &self.kvs, n, v_base, kvd);
        self.enc_f32_to_f16(enc, &self.kvs, 0, &self.k_cache[l], kv_byte_off, n * kvd);
        self.enc_f32_to_f16(enc, &self.kvs, v_base, &self.v_cache[l], kv_byte_off, n * kvd);
        {
            let p = RopeQkPrefillParams {
                head_dim: hd as u32,
                n_q_heads: cfg.num_attention_heads as u32,
                n_kv_heads: cfg.num_key_value_heads as u32,
                pos0: pos0 as u32,
                theta: cfg.rope_theta,
                n_rows: n as u32,
            };
            enc.set_compute_pipeline_state(&e.pipes.rope_qk_prefill);
            enc.set_buffer(0, Some(&self.q), 0);
            enc.set_buffer(1, Some(&self.k_cache[l]), kv_byte_off);
            set_bytes(enc, 2, &p);
            gpu::dispatch_grid(
                enc,
                n * (cfg.num_attention_heads + cfg.num_key_value_heads) * hd / 2,
            );
        }
        let p = AttnParams {
            head_dim: hd as u32,
            n_heads: cfg.num_attention_heads as u32,
            n_kv_heads: cfg.num_key_value_heads as u32,
            pos0: pos0 as u32,
            max_seq: self.max_seq as u32,
            n_rows: n as u32,
        };
        if hd == gpu::FLASH_HEAD_DIM {
            enc.set_compute_pipeline_state(&e.pipes.attention_flash);
            enc.set_buffer(0, Some(&self.q), 0);
            enc.set_buffer(1, Some(&self.k_cache[l]), 0);
            enc.set_buffer(2, Some(&self.v_cache[l]), 0);
            enc.set_buffer(3, Some(&self.att), 0);
            set_bytes(enc, 4, &p);
            enc.set_buffer(5, Some(&self.xh), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(
                    cfg.num_attention_heads as u64,
                    n.div_ceil(gpu::FLASH_Q) as u64,
                    1,
                ),
                MTLSize::new(gpu::FLASH_THREADS as u64, 1, 1),
            );
        } else {
            enc.set_compute_pipeline_state(&e.pipes.attention_fallback);
            enc.set_buffer(0, Some(&self.q), 0);
            enc.set_buffer(1, Some(&self.k_cache[l]), 0);
            enc.set_buffer(2, Some(&self.v_cache[l]), 0);
            enc.set_buffer(3, Some(&self.scores), 0);
            enc.set_buffer(4, Some(&self.att), 0);
            set_bytes(enc, 5, &p);
            enc.dispatch_thread_groups(
                MTLSize::new(cfg.num_attention_heads as u64, n as u64, 1),
                MTLSize::new(256, 1, 1),
            );
        }
        self.enc_matmul_paged(enc, pool, &lw.o, &self.att, &self.xb, n, 0, h);
        self.enc_elem(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);

        // SwiGLU MLP half.
        self.enc_rmsnorm(enc, &self.x, 0, &lw.post_ln, &self.xn, n);
        self.enc_matmul_paged(enc, pool, &lw.gate, &self.xn, &self.gate, n, 0, cfg.intermediate_size);
        self.enc_matmul_paged(enc, pool, &lw.up, &self.xn, &self.up, n, 0, cfg.intermediate_size);
        self.enc_elem(enc, &e.pipes.silu_mul, &self.gate, &self.up, n * cfg.intermediate_size);
        self.enc_matmul_paged(enc, pool, &lw.down, &self.gate, &self.xb, n, 0, h);
        self.enc_elem(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);
    }

    /// One decode step (n == 1): matvec-family kernels, flash-decoding attention.
    fn encode_layer_decode(
        &self,
        enc: &ComputeCommandEncoderRef,
        pool: &WeightPool,
        lw: &LayerWeights,
        l: usize,
        pos: usize,
        kv_byte_off: u64,
    ) {
        let e = self.e;
        let cfg = &e.cfg;
        let hd = cfg.head_dim();

        self.enc_rmsnorm(enc, &self.x, 0, &lw.input_ln, &self.xn, 1);
        self.enc_matvec_paged(enc, pool, &e.pipes.matvec, &lw.q, &self.xn, &self.q, 0, 4);
        self.enc_matvec_paged(enc, pool, &e.pipes.matvec_h, &lw.k, &self.xn, &self.k_cache[l], kv_byte_off, 2);
        self.enc_matvec_paged(enc, pool, &e.pipes.matvec_h, &lw.v, &self.xn, &self.v_cache[l], kv_byte_off, 2);
        {
            let p = RopeQkParams {
                head_dim: hd as u32,
                n_q_heads: cfg.num_attention_heads as u32,
                n_kv_heads: cfg.num_key_value_heads as u32,
                pos: pos as u32,
                theta: cfg.rope_theta,
            };
            enc.set_compute_pipeline_state(&e.pipes.rope_qk_decode);
            enc.set_buffer(0, Some(&self.q), 0);
            enc.set_buffer(1, Some(&self.k_cache[l]), kv_byte_off);
            set_bytes(enc, 2, &p);
            gpu::dispatch_grid(enc, (cfg.num_attention_heads + cfg.num_key_value_heads) * hd / 2);
        }
        {
            let n_splits = (pos + 1).div_ceil(gpu::ATTN_SPLIT);
            let p = AttnDecParams {
                head_dim: hd as u32,
                n_heads: cfg.num_attention_heads as u32,
                n_kv_heads: cfg.num_key_value_heads as u32,
                pos: pos as u32,
                n_splits: n_splits as u32,
            };
            let (grid_x, tg_mem) = e.gqa;
            enc.set_compute_pipeline_state(&e.pipes.attn_dec_partial);
            enc.set_buffer(0, Some(&self.q), 0);
            enc.set_buffer(1, Some(&self.k_cache[l]), 0);
            enc.set_buffer(2, Some(&self.v_cache[l]), 0);
            enc.set_buffer(3, Some(&self.partials), 0);
            set_bytes(enc, 4, &p);
            for (i, len) in tg_mem.iter().enumerate() {
                enc.set_threadgroup_memory_length(i as u64, *len);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(grid_x, n_splits as u64, 1),
                MTLSize::new(gpu::DEC_TG as u64, 1, 1),
            );
            enc.set_compute_pipeline_state(&e.pipes.attn_dec_reduce);
            enc.set_buffer(0, Some(&self.partials), 0);
            enc.set_buffer(1, Some(&self.att), 0);
            set_bytes(enc, 2, &p);
            enc.dispatch_thread_groups(
                MTLSize::new(cfg.num_attention_heads as u64, 1, 1),
                MTLSize::new(hd as u64, 1, 1),
            );
        }
        self.enc_matvec_paged(enc, pool, &e.pipes.matvec_acc, &lw.o, &self.att, &self.x, 0, 4);

        self.enc_rmsnorm(enc, &self.x, 0, &lw.post_ln, &self.xn, 1);
        // SwiGLU: gate and up share [inter, h], so their pages split identically.
        for blk in 0..lw.gate.n_pages {
            let (r0, rows) = lw.gate.block_rows(blk);
            let p = MatvecParams { in_dim: lw.gate.in_dim as u32, out_dim: rows as u32 };
            enc.set_compute_pipeline_state(&e.pipes.matvec_swiglu);
            enc.set_buffer(0, Some(pool.get(&lw.gate, blk)), 0);
            enc.set_buffer(1, Some(pool.get(&lw.up, blk)), 0);
            enc.set_buffer(2, Some(&self.xn), 0);
            enc.set_buffer(3, Some(&self.gate), (r0 * 4) as u64);
            set_bytes(enc, 4, &p);
            gpu::dispatch_simdgroup_rows(enc, rows as u32);
        }
        self.enc_matvec_paged(enc, pool, &e.pipes.matvec_acc, &lw.down, &self.gate, &self.x, 0, 4);
    }

    /// Process `n` tokens at positions pos0.. — one command buffer per layer.
    fn run(&mut self, ids: &[u32], pos0: usize, want_logits: bool) -> crate::Result<Vec<f32>> {
        let e = self.e;
        let cfg = &e.cfg;
        let (h, hd, kvd) = (cfg.hidden_size, cfg.head_dim(), cfg.kv_dim());
        let n = ids.len();
        let kv_byte_off = (pos0 * kvd * 2) as u64;
        let fused_decode = n == 1 && hd <= gpu::DEC_TG && hd.is_multiple_of(4);
        // One session encodes at a time — concurrent serve sessions serialize
        // here and stay correct (documented D10 behavior).
        let mut pool = e.pool.lock().map_err(|_| "lowmem pool lock poisoned")?;

        self.embed_gather(ids)?;

        for (l, lw) in e.layers.iter().enumerate() {
            let pages = layer_pages(lw);
            pool.make_resident(&e.manifest, &pages)?;
            let cb = e.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            if fused_decode {
                self.encode_layer_decode(enc, &pool, lw, l, pos0, kv_byte_off);
            } else {
                self.encode_layer_prefill(enc, &pool, lw, l, pos0, n, kv_byte_off);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            pool.unpin_all();
        }

        if !want_logits {
            return Ok(Vec::new());
        }

        // Final norm on the last row, then the lm_head in page groups small
        // enough to pin — the ~vocab × hidden matrix never sits resident whole.
        let lm = &e.lm_head;
        let group = (pool.budget_bytes() / 2 / super::pool::PAGE_BYTES).max(1);
        let mut blk = 0;
        let mut first = true;
        while blk < lm.n_pages {
            let hi = (blk + group).min(lm.n_pages);
            let pages: Vec<_> = (blk..hi).map(|b| (lm, b)).collect();
            pool.make_resident(&e.manifest, &pages)?;
            let cb = e.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            if first {
                self.enc_rmsnorm(enc, &self.x, ((n - 1) * h * 4) as u64, &e.final_norm, &self.xn, 1);
                first = false;
            }
            for b in blk..hi {
                let (r0, rows) = lm.block_rows(b);
                let p = MatvecParams { in_dim: lm.in_dim as u32, out_dim: rows as u32 };
                enc.set_compute_pipeline_state(&e.pipes.matvec);
                enc.set_buffer(0, Some(pool.get(lm, b)), 0);
                enc.set_buffer(1, Some(&e.zero_bias), 0);
                enc.set_buffer(2, Some(&self.xn), 0);
                enc.set_buffer(3, Some(&self.logits), (r0 * 4) as u64);
                set_bytes(enc, 4, &p);
                gpu::dispatch_simdgroup_rows(enc, rows as u32);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            pool.unpin_all();
            blk = hi;
        }

        let logits = unsafe {
            std::slice::from_raw_parts(self.logits.contents() as *const f32, cfg.vocab_size)
        };
        Ok(logits.to_vec())
    }
}

fn layer_pages(lw: &LayerWeights) -> Vec<(&PagedTensor, usize)> {
    let mut v = Vec::new();
    for t in [&lw.q, &lw.k, &lw.v, &lw.o, &lw.gate, &lw.up, &lw.down] {
        for b in 0..t.n_pages {
            v.push((t, b));
        }
    }
    v
}

impl Session for LowMemSession<'_> {
    fn forward(&mut self, token: u32, pos: usize) -> crate::Result<Vec<f32>> {
        self.run(&[token], pos, true)
    }

    /// Chunked prefill, same shape as the metal backend's: later chunks attend
    /// to earlier chunks' K,V through the cache via pos0.
    fn prefill(&mut self, ids: &[u32]) -> crate::Result<Vec<f32>> {
        let mut pos0 = 0;
        let mut logits = Vec::new();
        for chunk in ids.chunks(self.chunk) {
            let last = pos0 + chunk.len() == ids.len();
            logits = self.run(chunk, pos0, last)?;
            pos0 += chunk.len();
        }
        Ok(logits)
    }
}
