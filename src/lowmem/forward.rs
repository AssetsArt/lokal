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

use super::pool::{Admit, Bind, PagedTensor, PendingConvert, PoolStats, WeightPool};
use super::{dequant_row_ref, AttnWeights, Fam, FullAttn, LayerWeights, LowMemEngine, SrcType};

use crate::engine::Session;
use crate::gpu::metal as gpu;
use half::{bf16, f16};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

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
struct RopeParams {
    head_dim: u32,
    n_heads: u32,
    pos0: u32,
    theta: f32,
    n_rows: u32,
    /// Leading dims that rotate; the rest of each head passes through. Equals
    /// head_dim everywhere except qwen35 (rope.dimension_count 64 of 256).
    rot_dim: u32,
}
#[repr(C)]
struct RopeQkParams {
    head_dim: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    pos: u32,
    theta: f32,
    /// Leading dims that rotate; head_dim except on qwen35 (partial RoPE).
    rot_dim: u32,
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
    chunk: usize,
    /// Per layer: f16 [cap × kv_dim] — the sink region plus the ring. Sized by
    /// the WINDOW, not the context: this is the bounded-KV promise (D5).
    k_cache: Vec<Buffer>,
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
    /// qwen35 only: per-linear-layer recurrent state (conv + delta), f32,
    /// read-modify-written once per step by the kernel lane. ONE live state
    /// per sequence — no rollback in v1: a session serves exactly one prompt
    /// and serve builds a fresh session per request, so nothing ever rewinds.
    deltanet: Option<crate::gpu::metal::DeltaNetStates>,
    /// qwen35 only: the deltanet block's working buffers. `None` on every other
    /// architecture, so nothing else pays for them.
    ds: Option<DeltaScratch>,
    /// qwen35 only: the joint Q+gate projection de-interleaved. Two compact
    /// [rows][heads][head_dim] tensors, so qk-norm, RoPE and attention keep
    /// seeing an ordinary Q and never learn the interleaved stride. Separate
    /// buffers rather than an in-place compaction because compacting in place
    /// races: head h's destination overlaps head h+1's source.
    qg: Option<(Buffer, Buffer)>,
    /// Pool counters as they stood when prefill finished, so the drop line can
    /// report decode's staging on its own. Decode is where residency is proved:
    /// prefill always stages the model once, decode must stage nothing.
    mark: PoolStats,
    decode_steps: u64,
}

/// The gated-deltanet block's scratch. Sized from the deltanet geometry, not
/// from hidden_size: conv_channels (6144 on the 2B) is WIDER than hidden (2048)
/// and wider than q_proj_dim, so none of the existing buffers can be borrowed
/// for it — the same width-confusion that has already cost this repo two
/// overruns, avoided here by giving the block its own buffers.
struct DeltaScratch {
    /// [chunk x conv_channels] — the joint qkv projection, conv's input.
    qkv: Buffer,
    /// [chunk x d_inner] — the `z` gate, pre-silu.
    z: Buffer,
    /// [chunk x n_v_heads] each — the two tiny per-head projections.
    alpha: Buffer,
    beta_p: Buffer,
    /// [chunk x n_v_heads] each — the activated gates, one slice per token.
    g: Buffer,
    beta: Buffer,
    /// [chunk x conv_channels] — post-silu conv output, split into q|k|v.
    conv_out: Buffer,
    /// [chunk x d_inner] — the block's output, before the out projection.
    dout: Buffer,
}

impl DeltaScratch {
    fn new(d: &metal::Device, dims: crate::deltanet_ref::DeltaDims, chunk: usize) -> Self {
        let (c, inner, hv) = (dims.conv_channels(), dims.d_inner(), dims.n_v_heads);
        Self {
            qkv: gpu::f32_buffer(d, chunk * c),
            z: gpu::f32_buffer(d, chunk * inner),
            alpha: gpu::f32_buffer(d, chunk * hv),
            beta_p: gpu::f32_buffer(d, chunk * hv),
            // Chunk-wide, mirroring metal's DeltaScratch — byte-identical under
            // the per-token loop, and the precondition for batching it.
            g: gpu::f32_buffer(d, chunk * hv),
            beta: gpu::f32_buffer(d, chunk * hv),
            conv_out: gpu::f32_buffer(d, chunk * c),
            dout: gpu::f32_buffer(d, chunk * inner),
        }
    }
}

impl<'a> LowMemSession<'a> {
    pub fn new(e: &'a LowMemEngine, max_seq: usize) -> Self {
        let cfg = &e.cfg;
        let d = &e.device;
        let cap = e.win.cap;
        let chunk = gpu::PREFILL_CHUNK.min(max_seq);
        let (h, kvd) = (cfg.hidden_size, e.dims.kv_dim);
        // The two-kind state schedule (docs/gguf-design.md §state seam) — the
        // one source for which slot a layer gets; the stub-vs-full choice below
        // and the DeltaNetStates slots both follow it.
        let sched = crate::gpu::metal::state_schedule(
            cfg.num_hidden_layers,
            e.deltanet_layout.as_ref(),
        );
        let recurrent =
            |l: usize| sched[l] == crate::gpu::metal::LayerStateKind::Recurrent;
        Self {
            // KV EXISTS ONLY ON THE FULL-ATTENTION LAYERS. On qwen35 that is 6 of
            // 24 (one in full_attention_interval); the gated-deltanet blocks
            // carry recurrent state instead and never touch a cache. Allocating
            // all 24 would blow the budget line D computed by 4x on the KV term
            // while nothing ever read three quarters of it.
            //
            // The slot is KEPT (a one-element stub) rather than the vector
            // compacted, so `self.k_cache[l]` still means layer l everywhere and
            // no call site needs a second index that could drift out of step
            // with lane D's is_recurrent map — which is the source of truth.
            k_cache: (0..cfg.num_hidden_layers)
                .map(|l| gpu::f16_empty_buffer(d, if recurrent(l) { 1 } else { cap * kvd }))
                .collect(),
            v_cache: (0..cfg.num_hidden_layers)
                .map(|l| gpu::f16_empty_buffer(d, if recurrent(l) { 1 } else { cap * kvd }))
                .collect(),
            x: gpu::f32_buffer(d, chunk * h),
            xn: gpu::f32_buffer(d, chunk * h),
            // q and att hold per-head rows: chunk * q_dim, which equals
            // chunk * hidden everywhere EXCEPT qwen3 (explicit head_dim makes
            // q_dim = 2*hidden on the 0.6B) — sizing them by `h` was a real
            // overflow, found by Tiësto fixing the same bug in metal (b3bd4db6).
            // q holds the PROJECTION's output, which on qwen35 is 2x the
            // attention width (joint Q+gate) — size by q_proj_dim, not q_dim.
            q: gpu::f32_buffer(d, chunk * e.dims.q_proj_dim),
            att: gpu::f32_buffer(d, chunk * e.dims.q_dim),
            xb: gpu::f32_buffer(d, chunk * h),
            gate: gpu::f32_buffer(d, chunk * cfg.intermediate_size),
            up: gpu::f32_buffer(d, chunk * cfg.intermediate_size),
            kvs: gpu::f32_buffer(d, 2 * chunk * kvd),
            // xh feeds o_proj with att's half copy — q_dim wide, not hidden.
            xh: gpu::f16_empty_buffer(d, chunk * e.dims.q_dim),
            scores: if e.dims.head_dim == gpu::FLASH_HEAD_DIM {
                gpu::f32_buffer(d, 1) // flash path never reads it — stub binding
            } else {
                gpu::f32_buffer(d, chunk * cfg.num_attention_heads * cap)
            },
            deltanet: e.deltanet_layout.as_ref().map(|l| {
                let st = crate::gpu::metal::DeltaNetStates::new(d, l);
                for (i, k) in sched.iter().enumerate() {
                    debug_assert_eq!(
                        st.layers[i].is_some(),
                        *k == crate::gpu::metal::LayerStateKind::Recurrent,
                        "state slot kind drifted from the schedule at layer {i}"
                    );
                }
                st
            }),
            ds: e.delta_dims().map(|dd| DeltaScratch::new(d, dd, chunk)),
            qg: (e.dims.q_proj_dim != e.dims.q_dim).then(|| {
                (gpu::f32_buffer(d, chunk * e.dims.q_dim), gpu::f32_buffer(d, chunk * e.dims.q_dim))
            }),
            partials: gpu::f32_buffer(
                d,
                cfg.num_attention_heads * (cap / gpu::ATTN_SPLIT) * (e.dims.head_dim + 2),
            ),
            logits: gpu::f32_buffer(d, cfg.vocab_size),
            mark: PoolStats::default(),
            decode_steps: 0,
            e,
            chunk,
        }
    }

    /// Destination spans for writing positions [pos0, pos0+n) into the store:
    /// (first chunk row, first slot, length). At most three — the sink part,
    /// then the ring part split once at the wrap.
    fn write_spans(&self, pos0: usize, n: usize) -> Vec<(usize, usize, usize)> {
        let win = &self.e.win;
        let end = pos0 + n;
        let mut spans = Vec::with_capacity(3);
        if pos0 < win.sink {
            spans.push((0, pos0, win.sink.min(end) - pos0));
        }
        let mut p = pos0.max(win.sink);
        while p < end {
            let rel = (p - win.sink) % win.ring;
            let len = (win.ring - rel).min(end - p);
            spans.push((p - pos0, win.sink_pad + rel, len));
            p += len;
        }
        spans
    }

    /// CPU-side embedding gather: one mmap row per token, converted straight
    /// into the x buffer (unified memory).
    fn embed_gather(&self, ids: &[u32]) -> crate::Result<()> {
        let h = self.e.cfg.hidden_size;
        let xp = self.x.contents() as *mut f32;
        const EMBED: &str = "model.embed_tokens.weight";
        let ty = self.e.source.src_type(EMBED)?;
        for (i, &id) in ids.iter().enumerate() {
            let row = self.e.source.read_rows(EMBED, id as usize, id as usize + 1)?;
            let dst = unsafe { std::slice::from_raw_parts_mut(xp.add(i * h), h) };
            match ty {
                SrcType::F32 => {
                    for (o, b) in dst.iter_mut().zip(row.chunks_exact(4)) {
                        *o = f32::from_le_bytes(b.try_into().unwrap());
                    }
                }
                SrcType::BF16 => {
                    for (o, b) in dst.iter_mut().zip(row.chunks_exact(2)) {
                        *o = bf16::from_le_bytes([b[0], b[1]]).to_f32();
                    }
                }
                SrcType::F16 => {
                    for (o, b) in dst.iter_mut().zip(row.chunks_exact(2)) {
                        *o = f16::from_le_bytes([b[0], b[1]]).to_f32();
                    }
                }
                // A quantized embedding table stays quantized on disk and is
                // gathered a row at a time — 512 scattered rows per chunk never
                // justified a resident vocab x hidden table, quantized or not,
                // and the reference dequant is exact (D3).
                SrcType::Quant(t) => dequant_row_ref(t, row, dst),
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

    /// RMSNorm over rows of an arbitrary width — qk-norm normalizes each HEAD
    /// (dim = head_dim), not each token, so one token contributes n_heads rows.
    /// Safe in place: the kernel reduces the row into threadgroup memory and
    /// barriers before any thread writes back, so no thread reads a value
    /// another has already scaled.
    fn enc_rmsnorm_dim(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_off: u64,
        weight: &Buffer,
        n_rows: usize,
        dim: usize,
    ) {
        let p = NormParams { dim: dim as u32, eps: self.e.cfg.rms_norm_eps };
        enc.set_compute_pipeline_state(&self.e.pipes.rmsnorm);
        enc.set_buffer(0, Some(x), x_off);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(x), x_off);
        set_bytes(enc, 3, &p);
        enc.dispatch_thread_groups(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// The same, on an f16 buffer in place — decode writes K straight into the
    /// cache, so its qk-norm has to happen there rather than on an f32 staging
    /// copy the way prefill's does.
    fn enc_rmsnorm_h(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_off: u64,
        weight: &Buffer,
        n_rows: usize,
        dim: usize,
    ) {
        let p = NormParams { dim: dim as u32, eps: self.e.cfg.rms_norm_eps };
        enc.set_compute_pipeline_state(&self.e.pipes.rmsnorm_h_inplace);
        enc.set_buffer(0, Some(x), x_off);
        enc.set_buffer(1, Some(weight), 0);
        set_bytes(enc, 2, &p);
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
            enc.set_compute_pipeline_state(self.e.matmul_pipe(t.ty));
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
    /// matvec_acc all share the binding layout. Each block reads either its
    /// staged f16 pool page or the raw bf16 checkpoint span through the mmap
    /// view (the LM_W_BF16-specialized pipeline). `y_elem` is the output's
    /// element size in bytes (4 for f32, 2 for the f16 caches).
    #[allow(clippy::too_many_arguments)]
    fn enc_matvec_bound(
        &self,
        enc: &ComputeCommandEncoderRef,
        fam: Fam,
        t: &PagedTensor,
        binds: &[Bind],
        x: &Buffer,
        y: &Buffer,
        y_base: u64,
        y_elem: u64,
    ) {
        for (blk, bind) in binds.iter().enumerate() {
            let (r0, rows) = t.block_rows(blk);
            let p = MatvecParams { in_dim: t.in_dim as u32, out_dim: rows as u32 };
            match bind {
                Bind::Pool(buf) => {
                    enc.set_compute_pipeline_state(self.e.staged_pipe(t.ty, fam));
                    enc.set_buffer(0, Some(buf), 0);
                }
                Bind::Direct(view, off) => {
                    // Quant blocks read identically from a pool page or the
                    // checkpoint, so they keep their staged pipeline; only
                    // bf16 needs the raw-checkpoint specialization.
                    enc.set_compute_pipeline_state(match t.ty.is_quant() {
                        true => self.e.staged_pipe(t.ty, fam),
                        false => self.e.direct_pipe(fam),
                    });
                    enc.set_buffer(0, Some(view), *off as u64);
                }
            }
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

    /// GPU stage-ins: convert bf16 spans straight from the checkpoint's mmap
    /// view into their f16 pool pages — encoded at the HEAD of the command
    /// buffer that first reads them (serial encoder = ordered). The buffer
    /// binds at a 4-byte-aligned offset; the sub-offset rides in p.y.
    fn enc_pending_converts(&self, enc: &ComputeCommandEncoderRef, conv: &[PendingConvert]) {
        for c in conv {
            let base = (c.src_off & !3) as u64;
            let p: [u32; 2] = [c.elems as u32, ((c.src_off & 3) >> 1) as u32];
            enc.set_compute_pipeline_state(&self.e.pipes.bf16_to_f16);
            enc.set_buffer(0, Some(&c.src), base);
            enc.set_buffer(1, Some(&c.dst), 0);
            set_bytes(enc, 2, &p);
            enc.set_buffer(3, Some(&self.e.clip_flag), 0);
            gpu::dispatch_grid(enc, c.elems);
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
    /// arm minus the tensor-ops staging; K/V land in the sink+ring store, in at
    /// most three contiguous spans, and RoPE runs with TRUE absolute positions.
    /// qwen35's gated-deltanet block, for `n` tokens already projected into the
    /// deltanet scratch. Token by token, because this is the DECODE form — plan
    /// v1 runs prefill through it one position at a time (the chunked form with
    /// solve_tri/cumsum is a later lane). Both recurrent states roll across the
    /// loop, so the token order here is load-bearing.
    ///
    /// Op order is llama.cpp's (src/models/qwen35.cpp build_linear_attn), and
    /// the whole chain is verified against lane B's reference on real weights by
    /// gpu::metal's `deltanet_block_matches_reference_on_real_weights`. Dispatches
    /// inside one compute encoder are serial, so the chain needs no barriers.
    fn enc_delta_block(
        &self,
        enc: &ComputeCommandEncoderRef,
        la: &super::LinearAttn,
        l: usize,
        n: usize,
    ) {
        let e = self.e;
        let d = e.delta_dims().expect("deltanet dims on a qwen35 checkpoint");
        let ds = self.ds.as_ref().expect("deltanet scratch on a qwen35 session");
        let st = self.deltanet.as_ref().expect("qwen35 states").layers[l]
            .as_ref()
            .expect("every linear layer owns recurrent state");
        let eps = e.cfg.rms_norm_eps;
        let (s_dim, hv, hk) = (d.d_state, d.n_v_heads, d.n_k_heads);
        let (key_dim, inner, c_all) = (s_dim * hk, d.d_inner(), d.conv_channels());

        // Decode keeps the per-token chain byte-for-byte — see the metal twin's
        // note: it is one token, and it is the path that just earned 11x here.
        if n > 1 {
            self.enc_delta_block_chunk(enc, la, l, n);
            return;
        }

        for t in 0..n {
            let (goff, coff, zoff) = ((t * hv * 4) as u64, (t * c_all * 4) as u64, (t * inner * 4) as u64);

            // 1. the two per-head scalars. beta is SIGMOIDED inside the kernel.
            enc.set_compute_pipeline_state(&e.pipes.delta_gates);
            enc.set_buffer(0, Some(&ds.alpha), goff);
            enc.set_buffer(1, Some(&ds.beta_p), goff);
            enc.set_buffer(2, Some(&la.a), 0);
            enc.set_buffer(3, Some(&la.dt_bias), 0);
            enc.set_buffer(4, Some(&ds.g), goff);
            enc.set_buffer(5, Some(&ds.beta), goff);
            set_bytes(enc, 6, &(hv as u32));
            set_bytes(enc, 7, &1u32); // n_tokens — see the metal twin's note
            gpu::dispatch_grid(enc, hv);

            // 2. depthwise conv + silu, rolling the conv state in place.
            #[repr(C)]
            struct SsmConvParams {
                channels: u32,
                d_conv: u32,
            }
            enc.set_compute_pipeline_state(&e.pipes.ssm_conv_decode);
            enc.set_buffer(0, Some(&st.conv), 0);
            enc.set_buffer(1, Some(&ds.qkv), coff);
            enc.set_buffer(2, Some(&la.conv1d), 0);
            enc.set_buffer(3, Some(&ds.conv_out), coff);
            set_bytes(enc, 4, &SsmConvParams { channels: c_all as u32, d_conv: d.d_conv as u32 });
            gpu::dispatch_grid(enc, c_all);

            // 3. l2-normalise q and k per K head, in place on their slices.
            for off in [0u64, (key_dim * 4) as u64] {
                enc.set_compute_pipeline_state(&e.pipes.l2norm_rows);
                enc.set_buffer(0, Some(&ds.conv_out), coff + off);
                set_bytes(enc, 1, &(s_dim as u32));
                set_bytes(enc, 2, &eps);
                set_bytes(enc, 3, &0u32); // tok_stride — one token, y index 0
                enc.dispatch_thread_groups(MTLSize::new(hk as u64, 1, 1), MTLSize::new(256, 1, 1));
            }

            // 4. the delta rule. q/k/v are three views of the conv output.
            #[repr(C)]
            struct DeltaStepParams {
                d_state: u32,
                n_v_heads: u32,
                group: u32,
            }
            enc.set_compute_pipeline_state(&e.pipes.delta_decode_step);
            enc.set_buffer(0, Some(&st.delta), 0);
            enc.set_buffer(1, Some(&ds.conv_out), coff);
            enc.set_buffer(2, Some(&ds.conv_out), coff + (key_dim * 4) as u64);
            enc.set_buffer(3, Some(&ds.conv_out), coff + (2 * key_dim * 4) as u64);
            enc.set_buffer(4, Some(&ds.g), goff);
            enc.set_buffer(5, Some(&ds.beta), goff);
            enc.set_buffer(6, Some(&ds.dout), zoff);
            set_bytes(
                enc,
                7,
                &DeltaStepParams { d_state: s_dim as u32, n_v_heads: hv as u32, group: (hv / hk) as u32 },
            );
            gpu::dispatch_grid(enc, s_dim * hv);

            // 5. per-head RMSNorm gated by silu(z), in place on this row.
            enc.set_compute_pipeline_state(&e.pipes.gated_output_norm);
            enc.set_buffer(0, Some(&ds.dout), zoff);
            enc.set_buffer(1, Some(&la.ssm_norm), 0);
            enc.set_buffer(2, Some(&ds.z), zoff);
            set_bytes(enc, 3, &(s_dim as u32));
            set_bytes(enc, 4, &eps);
            enc.dispatch_thread_groups(MTLSize::new(hv as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// De-interleave the joint Q+gate projection, returning the buffer the
    /// attention path should treat as Q. On every architecture without a joint
    /// projection this is a no-op that hands back `self.q` unchanged.
    fn enc_split_qg(&self, enc: &ComputeCommandEncoderRef, n: usize) -> &Buffer {
        let Some((q, gate)) = &self.qg else { return &self.q };
        let e = self.e;
        let (hd, heads) = (e.dims.head_dim, e.cfg.num_attention_heads);
        #[repr(C)]
        struct QGSplitParams {
            head_dim: u32,
            n_heads: u32,
            n_rows: u32,
        }
        enc.set_compute_pipeline_state(&e.pipes.split_q_gate);
        enc.set_buffer(0, Some(&self.q), 0);
        enc.set_buffer(1, Some(q), 0);
        enc.set_buffer(2, Some(gate), 0);
        set_bytes(enc, 3, &QGSplitParams { head_dim: hd as u32, n_heads: heads as u32, n_rows: n as u32 });
        gpu::dispatch_grid(enc, n * heads * hd);
        q
    }

    /// qwen35's attention output gate: attn · sigmoid(gate), applied AFTER
    /// attention and BEFORE wo (qwen35.cpp:327-331). A no-op elsewhere.
    fn enc_apply_qgate(&self, enc: &ComputeCommandEncoderRef, n: usize) {
        let Some((_, gate)) = &self.qg else { return };
        let e = self.e;
        enc.set_compute_pipeline_state(&e.pipes.attn_out_gate);
        enc.set_buffer(0, Some(&self.att), 0);
        enc.set_buffer(1, Some(gate), 0);
        set_bytes(enc, 2, &((n * e.dims.q_dim) as u32));
        gpu::dispatch_grid(enc, n * e.dims.q_dim);
    }

    /// qwen35's linear block, prefill form: the four projections, the deltanet
    /// chain token by token, then the output projection and the residual add.
    /// Mirrors what the attention half does around it, so the caller's shared
    /// norms and MLP need no special case.
    fn enc_delta_prefill(
        &self,
        enc: &ComputeCommandEncoderRef,
        pool: &WeightPool,
        la: &super::LinearAttn,
        l: usize,
        n: usize,
    ) {
        let e = self.e;
        let h = e.cfg.hidden_size;
        let d = e.delta_dims().expect("deltanet dims on a qwen35 checkpoint");
        let ds = self.ds.as_ref().expect("deltanet scratch on a qwen35 session");
        let (hv, inner, c_all) = (d.n_v_heads, d.d_inner(), d.conv_channels());
        self.enc_matmul_paged(enc, pool, &la.qkv, &self.xn, &ds.qkv, n, 0, c_all);
        self.enc_matmul_paged(enc, pool, &la.z_gate, &self.xn, &ds.z, n, 0, inner);
        self.enc_matmul_paged(enc, pool, &la.alpha, &self.xn, &ds.alpha, n, 0, hv);
        self.enc_matmul_paged(enc, pool, &la.beta, &self.xn, &ds.beta_p, n, 0, hv);
        self.enc_delta_block(enc, la, l, n);
        self.enc_matmul_paged(enc, pool, &la.out, &ds.dout, &self.xb, n, 0, h);
        self.enc_elem(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);
    }

    /// The same block in decode form: matvec-family projections against the
    /// bound pages, one token, and the output projection ACCUMULATED straight
    /// into x — exactly how the attention path spends its o_proj.
    /// The chunk-wide twin of `enc_delta_block` — memo §3/§4. lowmem's decode
    /// encoder is SERIAL, so unlike the metal side there are no explicit
    /// barriers here: consecutive dispatches on a serial encoder are already
    /// ordered, including the write-after-read on the conv window between
    /// `ssm_conv_prefill` and `ssm_conv_roll`.
    fn enc_delta_block_chunk(
        &self,
        enc: &ComputeCommandEncoderRef,
        la: &super::LinearAttn,
        l: usize,
        n: usize,
    ) {
        let e = self.e;
        let d = e.delta_dims().expect("deltanet dims on a qwen35 checkpoint");
        let ds = self.ds.as_ref().expect("deltanet scratch on a qwen35 session");
        let st = self.deltanet.as_ref().expect("qwen35 states").layers[l]
            .as_ref()
            .expect("every linear layer owns recurrent state");
        let eps = e.cfg.rms_norm_eps;
        let (s_dim, hv, hk) = (d.d_state, d.n_v_heads, d.n_k_heads);
        let (key_dim, inner, c_all) = (s_dim * hk, d.d_inner(), d.conv_channels());
        #[repr(C)]
        struct SsmConvBatchParams {
            channels: u32,
            d_conv: u32,
            n_tokens: u32,
        }
        #[repr(C)]
        struct DeltaStepParams {
            d_state: u32,
            n_v_heads: u32,
            group: u32,
        }
        let cp = SsmConvBatchParams {
            channels: c_all as u32,
            d_conv: d.d_conv as u32,
            n_tokens: n as u32,
        };

        // 1. gates for every token.
        enc.set_compute_pipeline_state(&e.pipes.delta_gates);
        enc.set_buffer(0, Some(&ds.alpha), 0);
        enc.set_buffer(1, Some(&ds.beta_p), 0);
        enc.set_buffer(2, Some(&la.a), 0);
        enc.set_buffer(3, Some(&la.dt_bias), 0);
        enc.set_buffer(4, Some(&ds.g), 0);
        enc.set_buffer(5, Some(&ds.beta), 0);
        set_bytes(enc, 6, &(hv as u32));
        set_bytes(enc, 7, &(n as u32));
        gpu::dispatch_grid(enc, n * hv);

        // 2. conv for every (token, channel), then the window rolled once.
        enc.set_compute_pipeline_state(&e.pipes.ssm_conv_prefill);
        enc.set_buffer(0, Some(&st.conv), 0);
        enc.set_buffer(1, Some(&ds.qkv), 0);
        enc.set_buffer(2, Some(&la.conv1d), 0);
        enc.set_buffer(3, Some(&ds.conv_out), 0);
        set_bytes(enc, 4, &cp);
        gpu::dispatch_grid(enc, n * c_all);
        enc.set_compute_pipeline_state(&e.pipes.ssm_conv_roll);
        enc.set_buffer(0, Some(&st.conv), 0);
        enc.set_buffer(1, Some(&ds.qkv), 0);
        set_bytes(enc, 2, &cp);
        gpu::dispatch_grid(enc, c_all);

        // 3. l2-normalise q and k for every token.
        for off in [0u64, (key_dim * 4) as u64] {
            enc.set_compute_pipeline_state(&e.pipes.l2norm_rows);
            enc.set_buffer(0, Some(&ds.conv_out), off);
            set_bytes(enc, 1, &(s_dim as u32));
            set_bytes(enc, 2, &eps);
            set_bytes(enc, 3, &(c_all as u32));
            enc.dispatch_thread_groups(
                MTLSize::new(hk as u64, n as u64, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // 4. the delta rule — one dispatch for the whole chunk, or the
        //    per-token fallback when the tile will not fit this device.
        match gpu::delta_tile(e.device.max_threadgroup_memory_length() as usize, s_dim) {
            Some(tile) => {
                enc.set_compute_pipeline_state(&e.pipes.delta_prefill_step);
                enc.set_buffer(0, Some(&st.delta), 0);
                enc.set_buffer(1, Some(&ds.conv_out), 0);
                enc.set_buffer(2, Some(&ds.conv_out), (key_dim * 4) as u64);
                enc.set_buffer(3, Some(&ds.conv_out), (2 * key_dim * 4) as u64);
                enc.set_buffer(4, Some(&ds.g), 0);
                enc.set_buffer(5, Some(&ds.beta), 0);
                enc.set_buffer(6, Some(&ds.dout), 0);
                set_bytes(
                    enc,
                    7,
                    &gpu::DeltaChunkParams {
                        d_state: s_dim as u32,
                        n_v_heads: hv as u32,
                        group: (hv / hk) as u32,
                        n_tokens: n as u32,
                        tok_stride: c_all as u32,
                        out_stride: inner as u32,
                        gate_stride: hv as u32,
                        tile: tile as u32,
                    },
                );
                enc.set_threadgroup_memory_length(0, (tile * s_dim * 4) as u64);
                enc.dispatch_thread_groups(
                    MTLSize::new((hv * (s_dim / tile)) as u64, 1, 1),
                    MTLSize::new(tile as u64, 1, 1),
                );
            }
            None => {
                for t in 0..n {
                    let (goff, coff, zoff) =
                        ((t * hv * 4) as u64, (t * c_all * 4) as u64, (t * inner * 4) as u64);
                    enc.set_compute_pipeline_state(&e.pipes.delta_decode_step);
                    enc.set_buffer(0, Some(&st.delta), 0);
                    enc.set_buffer(1, Some(&ds.conv_out), coff);
                    enc.set_buffer(2, Some(&ds.conv_out), coff + (key_dim * 4) as u64);
                    enc.set_buffer(3, Some(&ds.conv_out), coff + (2 * key_dim * 4) as u64);
                    enc.set_buffer(4, Some(&ds.g), goff);
                    enc.set_buffer(5, Some(&ds.beta), goff);
                    enc.set_buffer(6, Some(&ds.dout), zoff);
                    set_bytes(
                        enc,
                        7,
                        &DeltaStepParams {
                            d_state: s_dim as u32,
                            n_v_heads: hv as u32,
                            group: (hv / hk) as u32,
                        },
                    );
                    gpu::dispatch_grid(enc, s_dim * hv);
                }
            }
        }

        // 5. the gated output norm — rows are contiguous across (token, head).
        enc.set_compute_pipeline_state(&e.pipes.gated_output_norm);
        enc.set_buffer(0, Some(&ds.dout), 0);
        enc.set_buffer(1, Some(&la.ssm_norm), 0);
        enc.set_buffer(2, Some(&ds.z), 0);
        set_bytes(enc, 3, &(s_dim as u32));
        set_bytes(enc, 4, &eps);
        enc.dispatch_thread_groups(
            MTLSize::new((n * hv) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    fn enc_delta_decode(
        &self,
        enc: &ComputeCommandEncoderRef,
        plan: &AttnPlan,
        la: &super::LinearAttn,
        l: usize,
    ) {
        let ds = self.ds.as_ref().expect("deltanet scratch on a qwen35 session");
        let (p_qkv, p_z, p_out, p_alpha, p_beta) = plan.linear();
        self.enc_matvec_bound(enc, Fam::Mv, &la.qkv, p_qkv, &self.xn, &ds.qkv, 0, 4);
        self.enc_matvec_bound(enc, Fam::Mv, &la.z_gate, p_z, &self.xn, &ds.z, 0, 4);
        self.enc_matvec_bound(enc, Fam::Mv, &la.alpha, p_alpha, &self.xn, &ds.alpha, 0, 4);
        self.enc_matvec_bound(enc, Fam::Mv, &la.beta, p_beta, &self.xn, &ds.beta_p, 0, 4);
        self.enc_delta_block(enc, la, l, 1);
        self.enc_matvec_bound(enc, Fam::MvA, &la.out, p_out, &ds.dout, &self.x, 0, 4);
    }

    /// The full-attention half of a layer: the Q/K/V projections through the
    /// output projection's residual add. Split out so qwen35's linear blocks can
    /// take their own route BETWEEN the two norms, which both kinds share.
    #[allow(clippy::too_many_arguments)]
    fn enc_attn_prefill(
        &self,
        enc: &ComputeCommandEncoderRef,
        pool: &WeightPool,
        fa: &FullAttn,
        l: usize,
        pos0: usize,
        n: usize,
    ) {
        let e = self.e;
        let cfg = &e.cfg;
        let (h, hd, kvd) = (cfg.hidden_size, self.e.dims.head_dim, self.e.dims.kv_dim);
        let rot = self.e.dims.rot_dim; // == hd except on qwen35
        let v_base = (self.chunk * kvd * 4) as u64; // V's half of the kvs staging

        // Attention half.
        self.enc_matmul_paged(enc, pool, &fa.q, &self.xn, &self.q, n, 0, self.e.dims.q_proj_dim);
        // qwen35 projects Q and the output gate together, interleaved per head.
        // De-interleave once here so everything downstream sees an ordinary Q.
        let qbuf = self.enc_split_qg(enc, n);
        self.enc_matmul_paged(enc, pool, &fa.k, &self.xn, &self.kvs, n, 0, kvd);
        self.enc_matmul_paged(enc, pool, &fa.v, &self.xn, &self.kvs, n, v_base, kvd);
        // qwen3 normalizes every head of q and k before RoPE. q is f32 in its
        // own buffer; k is still in the f32 staging half, which is why this
        // lands BEFORE the f32_to_f16 spans below rather than after.
        if let (Some(qn), Some(kn)) = (&fa.q_norm, &fa.k_norm) {
            self.enc_rmsnorm_dim(enc, qbuf, 0, qn, n * cfg.num_attention_heads, hd);
            self.enc_rmsnorm_dim(enc, &self.kvs, 0, kn, n * cfg.num_key_value_heads, hd);
        }
        // RoPE q as one launch, then per destination span: convert the fresh
        // K/V rows into the store and rotate K there by its true positions.
        {
            let p = RopeParams {
                head_dim: hd as u32,
                n_heads: cfg.num_attention_heads as u32,
                pos0: pos0 as u32,
                theta: cfg.rope_theta,
                n_rows: n as u32,
                rot_dim: rot as u32,
            };
            enc.set_compute_pipeline_state(&e.pipes.rope);
            enc.set_buffer(0, Some(qbuf), 0);
            set_bytes(enc, 1, &p);
            gpu::dispatch_grid(enc, n * cfg.num_attention_heads * rot / 2);
        }
        for &(row, slot, len) in &self.write_spans(pos0, n) {
            let src_off = (row * kvd * 4) as u64;
            let dst_off = (slot * kvd * 2) as u64;
            self.enc_f32_to_f16(enc, &self.kvs, src_off, &self.k_cache[l], dst_off, len * kvd);
            self.enc_f32_to_f16(enc, &self.kvs, v_base + src_off, &self.v_cache[l], dst_off, len * kvd);
            let p = RopeParams {
                head_dim: hd as u32,
                n_heads: cfg.num_key_value_heads as u32,
                pos0: (pos0 + row) as u32,
                theta: cfg.rope_theta,
                n_rows: len as u32,
                rot_dim: rot as u32,
            };
            enc.set_compute_pipeline_state(&e.pipes.rope_h);
            enc.set_buffer(0, Some(&self.k_cache[l]), dst_off);
            set_bytes(enc, 1, &p);
            gpu::dispatch_grid(enc, len * cfg.num_key_value_heads * rot / 2);
        }
        let p = AttnParams {
            head_dim: hd as u32,
            n_heads: cfg.num_attention_heads as u32,
            n_kv_heads: cfg.num_key_value_heads as u32,
            pos0: pos0 as u32,
            max_seq: e.win.cap as u32, // the fallback kernel's scores stride
            n_rows: n as u32,
        };
        if hd == gpu::FLASH_HEAD_DIM {
            enc.set_compute_pipeline_state(&e.pipes.attention_flash);
            enc.set_buffer(0, Some(qbuf), 0);
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
            enc.set_buffer(0, Some(qbuf), 0);
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
        self.enc_apply_qgate(enc, n);
        self.enc_matmul_paged(enc, pool, &fa.o, &self.att, &self.xb, n, 0, h);
        self.enc_elem(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_layer_prefill(
        &self,
        enc: &ComputeCommandEncoderRef,
        pool: &WeightPool,
        lw: &LayerWeights,
        l: usize,
        pos0: usize,
        n: usize,
    ) {
        let e = self.e;
        let cfg = &e.cfg;
        let h = cfg.hidden_size;
        // Both block kinds share the two norms and the SwiGLU MLP; only what
        // sits between them differs.
        self.enc_rmsnorm(enc, &self.x, 0, &lw.input_ln, &self.xn, n);
        match &lw.attn {
            AttnWeights::Full(fa) => self.enc_attn_prefill(enc, pool, fa, l, pos0, n),
            AttnWeights::Linear(la) => self.enc_delta_prefill(enc, pool, la, l, n),
        }

        // SwiGLU MLP half.
        self.enc_rmsnorm(enc, &self.x, 0, &lw.post_ln, &self.xn, n);
        self.enc_matmul_paged(enc, pool, &lw.gate, &self.xn, &self.gate, n, 0, cfg.intermediate_size);
        self.enc_matmul_paged(enc, pool, &lw.up, &self.xn, &self.up, n, 0, cfg.intermediate_size);
        self.enc_elem(enc, &e.pipes.silu_mul, &self.gate, &self.up, n * cfg.intermediate_size);
        self.enc_matmul_paged(enc, pool, &lw.down, &self.gate, &self.xb, n, 0, h);
        self.enc_elem(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);
    }

    /// One decode step (n == 1): matvec-family kernels, flash-decoding attention
    /// over the WHOLE bounded store (every split dispatched; masks sort validity
    /// — the split count is a constant, which is what makes decode cost flat).
    /// The full-attention half of a decode step: projections, qk-norm, rope,
    /// flash-decoding attention, and the output projection accumulated into x.
    fn enc_attn_decode(
        &self,
        enc: &ComputeCommandEncoderRef,
        plan: &AttnPlan,
        fa: &FullAttn,
        l: usize,
        pos: usize,
    ) {
        let (pq, pk, pv, po) = plan.full();
        let e = self.e;
        let cfg = &e.cfg;
        let hd = e.dims.head_dim;
        let kv_byte_off = (e.win.slot_of(pos) * e.dims.kv_dim * 2) as u64;

        self.enc_matvec_bound(enc, Fam::Mv, &fa.q, pq, &self.xn, &self.q, 0, 4);
        let qbuf = self.enc_split_qg(enc, 1);
        self.enc_matvec_bound(enc, Fam::MvH, &fa.k, pk, &self.xn, &self.k_cache[l], kv_byte_off, 2);
        self.enc_matvec_bound(enc, Fam::MvH, &fa.v, pv, &self.xn, &self.v_cache[l], kv_byte_off, 2);
        if let (Some(qn), Some(kn)) = (&fa.q_norm, &fa.k_norm) {
            self.enc_rmsnorm_dim(enc, qbuf, 0, qn, cfg.num_attention_heads, hd);
            self.enc_rmsnorm_h(enc, &self.k_cache[l], kv_byte_off, kn, cfg.num_key_value_heads, hd);
        }
        {
            let p = RopeQkParams {
                head_dim: hd as u32,
                n_q_heads: cfg.num_attention_heads as u32,
                n_kv_heads: cfg.num_key_value_heads as u32,
                pos: pos as u32,
                theta: cfg.rope_theta,
                rot_dim: self.e.dims.rot_dim as u32,
            };
            enc.set_compute_pipeline_state(&e.pipes.rope_qk_decode);
            enc.set_buffer(0, Some(qbuf), 0);
            enc.set_buffer(1, Some(&self.k_cache[l]), kv_byte_off);
            set_bytes(enc, 2, &p);
            gpu::dispatch_grid(enc, (cfg.num_attention_heads + cfg.num_key_value_heads) * self.e.dims.rot_dim / 2);
        }
        {
            let n_splits = e.win.cap / gpu::ATTN_SPLIT;
            let p = AttnDecParams {
                head_dim: hd as u32,
                n_heads: cfg.num_attention_heads as u32,
                n_kv_heads: cfg.num_key_value_heads as u32,
                pos: pos as u32,
                n_splits: n_splits as u32,
            };
            let (grid_x, tg_mem) = e.gqa;
            enc.set_compute_pipeline_state(&e.pipes.attn_dec_partial);
            enc.set_buffer(0, Some(qbuf), 0);
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
        self.enc_apply_qgate(enc, 1);
        self.enc_matvec_bound(enc, Fam::MvA, &fa.o, po, &self.att, &self.x, 0, 4);
    }

    fn encode_layer_decode(
        &self,
        enc: &ComputeCommandEncoderRef,
        plan: &LayerPlan,
        lw: &LayerWeights,
        l: usize,
        pos: usize,
    ) {
        let e = self.e;
        let cfg = &e.cfg;
        // Shared with the linear block: the two norms and the MLP.
        self.enc_rmsnorm(enc, &self.x, 0, &lw.input_ln, &self.xn, 1);
        match &lw.attn {
            AttnWeights::Full(fa) => self.enc_attn_decode(enc, &plan.attn, fa, l, pos),
            AttnWeights::Linear(la) => self.enc_delta_decode(enc, &plan.attn, la, l),
        }

        self.enc_rmsnorm(enc, &self.x, 0, &lw.post_ln, &self.xn, 1);
        // SwiGLU: gate and up share [inter, h], so their pages split identically.
        // The fused kernel needs both operands in the SAME mode per block; a
        // mixed block pair falls back to two matvecs + the elementwise silu_mul.
        let mixed = plan
            .gate
            .iter()
            .zip(&plan.up)
            .any(|(g, u)| !matches!((g, u), (Bind::Pool(_), Bind::Pool(_)) | (Bind::Direct(..), Bind::Direct(..))));
        if mixed {
            self.enc_matvec_bound(enc, Fam::Mv, &lw.gate, &plan.gate, &self.xn, &self.gate, 0, 4);
            self.enc_matvec_bound(enc, Fam::Mv, &lw.up, &plan.up, &self.xn, &self.up, 0, 4);
            self.enc_elem(enc, &e.pipes.silu_mul, &self.gate, &self.up, cfg.intermediate_size);
        } else {
            for (blk, (bg, bu)) in plan.gate.iter().zip(&plan.up).enumerate() {
                let (r0, rows) = lw.gate.block_rows(blk);
                let p = MatvecParams { in_dim: lw.gate.in_dim as u32, out_dim: rows as u32 };
                match (bg, bu) {
                    (Bind::Pool(g), Bind::Pool(u)) => {
                        enc.set_compute_pipeline_state(e.swiglu_pipe(lw.gate.ty));
                        enc.set_buffer(0, Some(g), 0);
                        enc.set_buffer(1, Some(u), 0);
                    }
                    (Bind::Direct(g, go), Bind::Direct(u, uo)) => {
                        enc.set_compute_pipeline_state(&e.direct.matvec_swiglu);
                        enc.set_buffer(0, Some(g), *go as u64);
                        enc.set_buffer(1, Some(u), *uo as u64);
                    }
                    _ => unreachable!("mixed pairs took the split path"),
                }
                enc.set_buffer(2, Some(&self.xn), 0);
                enc.set_buffer(3, Some(&self.gate), (r0 * 4) as u64);
                set_bytes(enc, 4, &p);
                gpu::dispatch_simdgroup_rows(enc, rows as u32);
            }
        }
        self.enc_matvec_bound(enc, Fam::MvA, &lw.down, &plan.down, &self.gate, &self.x, 0, 4);
    }

    /// Process `n` tokens at positions pos0.. — one command buffer per layer,
    /// committed WITHOUT waiting (D4): while the GPU runs layer N, the CPU
    /// stages layer N+1's pages and encodes its dispatches. Metal's hazard
    /// tracking orders the dependent buffers across command buffers; the pool's
    /// epoch stamps keep eviction away from in-flight pages. The one wait is at
    /// the end of the forward (or when the budget forces one out early).
    fn run(&mut self, ids: &[u32], pos0: usize, want_logits: bool) -> crate::Result<Vec<f32>> {
        let e = self.e;
        let cfg = &e.cfg;
        let (h, hd) = (cfg.hidden_size, self.e.dims.head_dim);
        let n = ids.len();
        // DEC_MAX_HD, not DEC_TG. attn_dec_gqa_walk was generalized past a
        // threadgroup-wide head_dim by quant-decode-hd256 (merge 46d5880) and this
        // backend dispatches the SAME kernel, so the ceiling here was the metal
        // path's old one left behind — qwen35 (hd 256) fell to the prefill-shaped
        // encoder for every decode step, matmuls at n = 1.
        //
        // Why this is not the deferral the dense f16 path took (metal.rs's
        // `NOTE: still DEC_TG`): that path stays capped because its FUSED enc_qkv
        // has never run at hd > 128 and no f16 checkpoint exercises it. lowmem has
        // no fused qkv — enc_attn_decode issues three separate bound matvecs — so
        // every kernel this condition admits is either head_dim-agnostic (matvec,
        // rmsnorm_dim, rope_qk_decode, which takes rot_dim) or the generalized
        // decode-attention pair. Same shape as the metal QUANT path this backend
        // is the twin of, and the same ceiling it now carries.
        let fused_decode = n == 1 && hd <= gpu::DEC_MAX_HD && hd.is_multiple_of(4);
        // One session encodes at a time — concurrent serve sessions serialize
        // here and stay correct (documented D10 behavior).
        let mut pool = e.pool.lock().map_err(|_| "lowmem pool lock poisoned")?;
        let mut inflight: Vec<(metal::CommandBuffer, u64)> = Vec::new();

        self.embed_gather(ids)?;

        for (l, lw) in e.layers.iter().enumerate() {
            let (ep, plan) = if fused_decode {
                let (ep, plan) = plan_decode(&mut pool, &e.source, lw, &mut inflight)?;
                (ep, Some(plan))
            } else {
                (admit(&mut pool, &e.source, &layer_pages(lw), &mut inflight)?, None)
            };
            let conv = pool.take_pending_converts();
            let cb = e.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.enc_pending_converts(enc, &conv);
            match &plan {
                Some(plan) => self.encode_layer_decode(enc, plan, lw, l, pos0),
                None => self.encode_layer_prefill(enc, &pool, lw, l, pos0, n),
            }
            enc.end_encoding();
            cb.commit();
            pool.unpin_all();
            retire(&mut pool, &mut inflight, cb, ep, e.sync);
        }

        if want_logits {
            // Final norm on the last row, then every lm_head block in ONE
            // command buffer: resident blocks read the pool, the rest read the
            // checkpoint directly — the vocab × hidden matrix never needs to
            // sit resident, staged, or grouped.
            let lm = &e.lm_head;
            let (ep, binds) = plan_tensor(&mut pool, &e.source, lm, &mut inflight)?;
            let conv = pool.take_pending_converts();
            let cb = e.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            self.enc_pending_converts(enc, &conv);
            self.enc_rmsnorm(enc, &self.x, ((n - 1) * h * 4) as u64, &e.final_norm, &self.xn, 1);
            self.enc_matvec_bound(
                enc,
                Fam::Mv,
                lm,
                &binds,
                &self.xn,
                &self.logits,
                0,
                4,
            );
            enc.end_encoding();
            cb.commit();
            pool.unpin_all();
            retire(&mut pool, &mut inflight, cb, ep, e.sync);
        }

        // Drain before the pool lock drops: queue order makes the last command
        // buffer's completion cover them all, and a clean pool (every epoch
        // completed) is what lets another session evict freely.
        if let Some((cb, _)) = inflight.last() {
            cb.wait_until_completed();
        }
        for (_, ep) in inflight.drain(..) {
            pool.mark_completed(ep);
        }
        // The GPU converter's overflow flag — warn once per process.
        if unsafe { *(e.clip_flag.contents() as *const u32) } != 0
            && !e.clip_warned.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!("lowmem: warning — checkpoint holds values beyond f16 range; they clip at ±65504");
        }

        if !want_logits {
            return Ok(Vec::new());
        }
        let logits = unsafe {
            std::slice::from_raw_parts(self.logits.contents() as *const f32, cfg.vocab_size)
        };
        Ok(logits.to_vec())
    }
}

/// Get an epoch and make `pages` resident under it, waiting out the oldest
/// in-flight command buffer whenever every eviction candidate is still on the
/// GPU. Returns the epoch the caller's command buffer must retire under.
fn admit(
    pool: &mut WeightPool,
    src: &super::LowMemSource,
    pages: &[(&PagedTensor, usize)],
    inflight: &mut Vec<(metal::CommandBuffer, u64)>,
) -> crate::Result<u64> {
    let ep = pool.begin_cb();
    loop {
        match pool.make_resident(src, pages, ep)? {
            Admit::Ready => return Ok(ep),
            Admit::NeedWait => {
                if inflight.is_empty() {
                    return Err("lowmem: pool wait requested with nothing in flight (bug)".into());
                }
                let (cb, cbep) = inflight.remove(0);
                cb.wait_until_completed();
                pool.mark_completed(cbep);
            }
        }
    }
}

/// Book-keep a just-committed command buffer: in sync mode wait it out on the
/// spot; otherwise push it in flight and opportunistically retire any that
/// already finished (queue order — the front finishes first).
fn retire(
    pool: &mut WeightPool,
    inflight: &mut Vec<(metal::CommandBuffer, u64)>,
    cb: &metal::CommandBufferRef,
    ep: u64,
    sync: bool,
) {
    if sync {
        cb.wait_until_completed();
        pool.mark_completed(ep);
        return;
    }
    inflight.push((cb.to_owned(), ep));
    while inflight
        .first()
        .map(|(c, _)| c.status() == metal::MTLCommandBufferStatus::Completed)
        .unwrap_or(false)
    {
        let (_, done) = inflight.remove(0);
        pool.mark_completed(done);
    }
}

fn layer_pages(lw: &LayerWeights) -> Vec<(&PagedTensor, usize)> {
    let mut v = Vec::new();
    for t in lw.paged() {
        for b in 0..t.n_pages {
            v.push((t, b));
        }
    }
    v
}

/// Where each decode dispatch reads a layer's weights from.
struct LayerPlan {
    attn: AttnPlan,
    gate: Vec<Bind>,
    up: Vec<Bind>,
    down: Vec<Bind>,
}

/// Page bindings for the attention half, mirroring `AttnWeights` arm for arm.
/// Kept an enum for the same reason the weights are: the arms bind disjoint
/// tensor sets, and a plan that silently held the wrong arm's pages would
/// underfeed the layer at dispatch time rather than fail to compile.
enum AttnPlan {
    Full { q: Vec<Bind>, k: Vec<Bind>, v: Vec<Bind>, o: Vec<Bind> },
    Linear { qkv: Vec<Bind>, z_gate: Vec<Bind>, out: Vec<Bind>, alpha: Vec<Bind>, beta: Vec<Bind> },
}

impl AttnPlan {
    /// The linear block's bindings.
    fn linear(&self) -> (&[Bind], &[Bind], &[Bind], &[Bind], &[Bind]) {
        match self {
            AttnPlan::Linear { qkv, z_gate, out, alpha, beta } => (qkv, z_gate, out, alpha, beta),
            AttnPlan::Full { .. } => unreachable!("attention block dispatched to the linear path"),
        }
    }

    /// The full-attention bindings, for the paths that only handle them.
    fn full(&self) -> (&[Bind], &[Bind], &[Bind], &[Bind]) {
        match self {
            AttnPlan::Full { q, k, v, o } => (q, k, v, o),
            AttnPlan::Linear { .. } => unreachable!("linear block dispatched to the attention path"),
        }
    }
}

/// Bind one tensor's blocks for a decode-style dispatch, waiting out in-flight
/// command buffers when a staging admission needs room.
fn plan_tensor(
    pool: &mut WeightPool,
    src: &super::LowMemSource,
    t: &PagedTensor,
    inflight: &mut Vec<(metal::CommandBuffer, u64)>,
) -> crate::Result<(u64, Vec<Bind>)> {
    let ep = pool.begin_cb();
    let mut binds = Vec::with_capacity(t.n_pages);
    let mut blk = 0;
    while blk < t.n_pages {
        match pool.bind_decode(src, t, blk, ep)? {
            Ok(b) => {
                binds.push(b);
                blk += 1;
            }
            Err(Admit::NeedWait) => {
                if inflight.is_empty() {
                    return Err("lowmem: pool wait requested with nothing in flight (bug)".into());
                }
                let (cb, cbep) = inflight.remove(0);
                cb.wait_until_completed();
                pool.mark_completed(cbep);
            }
            Err(Admit::Ready) => unreachable!(),
        }
    }
    Ok((ep, binds))
}

/// The whole layer's decode plan under ONE epoch.
fn plan_decode(
    pool: &mut WeightPool,
    src: &super::LowMemSource,
    lw: &LayerWeights,
    inflight: &mut Vec<(metal::CommandBuffer, u64)>,
) -> crate::Result<(u64, LayerPlan)> {
    let ep = pool.begin_cb();
    let mut bind_all = |t: &PagedTensor| -> crate::Result<Vec<Bind>> {
        let mut binds = Vec::with_capacity(t.n_pages);
        let mut blk = 0;
        while blk < t.n_pages {
            match pool.bind_decode(src, t, blk, ep)? {
                Ok(b) => {
                    binds.push(b);
                    blk += 1;
                }
                Err(Admit::NeedWait) => {
                    if inflight.is_empty() {
                        return Err("lowmem: pool wait requested with nothing in flight (bug)".into());
                    }
                    let (cb, cbep) = inflight.remove(0);
                    cb.wait_until_completed();
                    pool.mark_completed(cbep);
                }
                Err(Admit::Ready) => unreachable!(),
            }
        }
        Ok(binds)
    };
    let attn = match &lw.attn {
        AttnWeights::Full(f) => AttnPlan::Full {
            q: bind_all(&f.q)?,
            k: bind_all(&f.k)?,
            v: bind_all(&f.v)?,
            o: bind_all(&f.o)?,
        },
        AttnWeights::Linear(l) => AttnPlan::Linear {
            qkv: bind_all(&l.qkv)?,
            z_gate: bind_all(&l.z_gate)?,
            out: bind_all(&l.out)?,
            alpha: bind_all(&l.alpha)?,
            beta: bind_all(&l.beta)?,
        },
    };
    Ok((
        ep,
        LayerPlan {
            attn,
            gate: bind_all(&lw.gate)?,
            up: bind_all(&lw.up)?,
            down: bind_all(&lw.down)?,
        },
    ))
}

impl Session for LowMemSession<'_> {
    fn forward(&mut self, token: u32, pos: usize) -> crate::Result<Vec<f32>> {
        self.decode_steps += 1;
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
        // The prefill/decode boundary: everything staged from here on is decode
        // traffic, which is what the residency gate actually measures.
        if let Ok(pool) = self.e.pool.lock() {
            self.mark = pool.stats();
        }
        Ok(logits)
    }
}

/// `LOKAL_LOWMEM_STATS=1` prints one structured line per session at drop. It
/// exists for the residency gate: a checkpoint that fits the pool must show
/// `decode_stage_ins=0`, and asserting that on a parsed field beats grepping
/// prose that could change wording under us.
impl Drop for LowMemSession<'_> {
    fn drop(&mut self) {
        if !std::env::var("LOKAL_LOWMEM_STATS").is_ok_and(|v| v == "1") {
            return;
        }
        let Ok(pool) = self.e.pool.lock() else { return };
        let (end, m) = (pool.stats(), self.mark);
        eprintln!(
            "lowmem: stats prefill_stage_ins={} prefill_MB={} decode_stage_ins={} decode_MB={} \
             decode_direct_binds={} decode_direct_MB={} decode_steps={} evictions={}",
            m.stage_ins,
            m.stage_bytes >> 20,
            end.stage_ins - m.stage_ins,
            (end.stage_bytes - m.stage_bytes) >> 20,
            end.direct_binds - m.direct_binds,
            (end.direct_bytes - m.direct_bytes) >> 20,
            self.decode_steps,
            end.evictions,
        );
    }
}
