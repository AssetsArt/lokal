//! Metal backend — the full forward pass on Apple M-series GPUs.
//!
//! Works hand in hand with kernels.metal (read the two files together):
//!   engine creation: compile the kernels, convert weights f32 → f16, upload once
//!   forward:         encode a whole token's work (~15 dispatches × N layers) into a
//!                    single command buffer → submit once → wait → read back only logits
//!   prefill:         same, but processes chunks of up to 128 tokens as matrix-matrix
//!                    work — W is read once per chunk instead of once per token,
//!                    which is where the order-of-magnitude prefill speedup comes from
//!
//! Why not dispatch op by op the way the CPU code calls functions? One CPU↔GPU sync
//! costs on the order of a hundred microseconds — at ~450 dispatches per token that
//! would be slower than the CPU. Encoding everything into one command buffer (the GPU
//! runs it in order on its own) is the heart of this backend.

use crate::config::ModelConfig;
use crate::engine::{Engine, Session};
use crate::model::Model;
use half::f16;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState,
    Device, FunctionConstantValues, MTLDataType, MTLResourceOptions, MTLSize,
};

/// Maximum tokens processed together during prefill (one command buffer per chunk).
/// Bigger chunks amortize weight reads better but grow the scratch buffers
/// (the attention scores buffer in particular).
const PREFILL_CHUNK: usize = 512;

// ---------- kernel parameters passed via set_bytes — must match the MSL structs exactly ----------

#[repr(C)]
struct EmbedParams {
    dim: u32,
    n_rows: u32,
}
#[repr(C)]
struct MatvecParams {
    in_dim: u32,
    out_dim: u32,
}
#[repr(C)]
struct QkvParams {
    in_dim: u32,
    q_dim: u32,
    kv_dim: u32,
    kv_off: u32,
}
#[repr(C)]
struct QkvBatchParams {
    in_dim: u32,
    q_dim: u32,
    kv_dim: u32,
    max_seq: u32,
}
#[repr(C)]
struct RopeQkBatchParams {
    head_dim: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    theta: f32,
    max_seq: u32,
    kv_dim: u32,
    n_rows: u32,
}
#[repr(C)]
struct AttnDecBatchParams {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    max_seq: u32,
    kv_dim: u32,
    splits_max: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct RowMeta {
    pos: u32,
    slot: u32,
}
#[repr(C)]
struct MatmulParams {
    in_dim: u32,
    out_dim: u32,
    n_rows: u32,
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

/// A pipeline is a compiled kernel ready to dispatch — one per kernel in kernels.metal.
struct Pipelines {
    embed: ComputePipelineState,
    matvec: ComputePipelineState,
    matvec_acc: ComputePipelineState,
    matvec_swiglu: ComputePipelineState,
    matvec_qkv: ComputePipelineState,
    matvec_h: ComputePipelineState,
    matmul: ComputePipelineState,
    matmul_t: ComputePipelineState,
    f32_to_f16: ComputePipelineState,
    bias_add: ComputePipelineState,
    matmul_h: ComputePipelineState,
    rmsnorm: ComputePipelineState,
    rmsnorm_hf: ComputePipelineState,
    silu_mul_hf: ComputePipelineState,
    rope: ComputePipelineState,
    rope_h: ComputePipelineState,
    rope_qk_decode: ComputePipelineState,
    attention: ComputePipelineState,
    attention_prefill_flash: ComputePipelineState,
    attention_decode_partial: ComputePipelineState,
    attention_decode_reduce: ComputePipelineState,
    silu_mul: ComputePipelineState,
    add_inplace: ComputePipelineState,
    // Batched-decode variants (continuous batching in serve mode).
    matvec_qkv_batch: ComputePipelineState,
    matvec_acc_batch: ComputePipelineState,
    matvec_swiglu_batch: ComputePipelineState,
    rope_qk_batch: ComputePipelineState,
    attention_decode_partial_batch: ComputePipelineState,
    attention_decode_reduce_batch: ComputePipelineState,
}

/// Cached positions per flash-decoding window — must match ATTN_SPLIT in kernels.metal.
const ATTN_SPLIT: usize = 128;
/// Threads per decode-attention threadgroup — must match DEC_TG in kernels.metal.
const DEC_TG: usize = 128;
/// Max q heads one GQA decode threadgroup covers — must match MAX_GQA_CHUNK in kernels.metal.
const MAX_GQA_CHUNK: usize = 8;
/// The head_dim the flash prefill attention kernel is specialized for (FA_HD in
/// kernels.metal); other head sizes take the scores-scratch fallback kernel.
const FLASH_HEAD_DIM: usize = 64;
/// Max rows the logits buffer holds — the ceiling for one speculative verify batch.
const SPEC_MAX: usize = 8;

/// A linear layer on the GPU: f16 weights + f16 bias (all-zero when the model has none —
/// adding zero is free and avoids a branch in the kernel).
struct GpuLinear {
    w: Buffer,
    bias: Buffer,
    has_bias: bool,
    in_dim: u32,
    out_dim: u32,
}

struct GpuBlock {
    input_layernorm: Buffer,
    q_proj: GpuLinear,
    k_proj: GpuLinear,
    v_proj: GpuLinear,
    o_proj: GpuLinear,
    post_attention_layernorm: Buffer,
    gate_proj: GpuLinear,
    up_proj: GpuLinear,
    down_proj: GpuLinear,
}

pub struct MetalEngine {
    cfg: ModelConfig,
    device: Device,
    queue: CommandQueue,
    pipes: Pipelines,
    embed_tokens: Buffer, // f16 [vocab × hidden]
    blocks: Vec<GpuBlock>,
    norm: Buffer,
    lm_head: GpuLinear,
}

// Apple documents MTLDevice / MTLCommandQueue / MTLBuffer / MTLComputePipelineState as
// thread-safe, but the `metal` crate's wrappers don't declare it, so we assert it here.
// (The genuinely non-thread-safe objects — command buffers and encoders — only live
// inside a session, which is always used from a single thread.)
unsafe impl Send for MetalEngine {}
unsafe impl Sync for MetalEngine {}

/// Convert f32 → f16 and upload as a GPU buffer. StorageModeShared means unified
/// memory: on M-series chips the CPU and GPU see the same bytes — no PCIe copies
/// like on discrete GPUs.
fn f16_buffer(device: &Device, data: &[f32]) -> Buffer {
    let halves: Vec<u16> = data.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    device.new_buffer_with_data(
        halves.as_ptr() as *const _,
        (halves.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn f32_buffer(device: &Device, len: usize) -> Buffer {
    device.new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeShared)
}

/// Uninitialized f16 buffer — the KV cache's dtype.
fn f16_empty_buffer(device: &Device, len: usize) -> Buffer {
    device.new_buffer((len * 2) as u64, MTLResourceOptions::StorageModeShared)
}

impl MetalEngine {
    /// Takes a loaded CPU-side Model and moves it onto the GPU (the Model is dropped after).
    pub fn new(model: Model) -> crate::Result<Self> {
        let device = Device::system_default().ok_or("no Metal-capable GPU found")?;
        let queue = device.new_command_queue();

        // Kernels are compiled at runtime — edit kernels.metal and just cargo run again.
        let lib = device
            .new_library_with_source(include_str!("kernels.metal"), &CompileOptions::new())
            .map_err(|e| format!("failed to compile kernels.metal: {e}"))?;
        let pipe = |name: &str| -> crate::Result<ComputePipelineState> {
            let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("kernel {name}: {e}").into())
        };
        // The GQA decode kernels are specialized per model: function constant 0
        // (GQA_CHUNK) is the q-head group width one threadgroup covers, fixed here
        // so the per-head loops in the kernel unroll flat.
        let gqa_chunk = (model.cfg.num_attention_heads / model.cfg.num_key_value_heads)
            .min(MAX_GQA_CHUNK) as u32;
        let gqa_pipe = |name: &str| -> crate::Result<ComputePipelineState> {
            let consts = FunctionConstantValues::new();
            consts.set_constant_value_at_index(
                &gqa_chunk as *const u32 as *const _,
                MTLDataType::UInt,
                0,
            );
            let f = lib
                .get_function(name, Some(consts))
                .map_err(|e| format!("kernel {name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("kernel {name}: {e}").into())
        };
        let pipes = Pipelines {
            embed: pipe("embed")?,
            matvec: pipe("matvec")?,
            matvec_acc: pipe("matvec_acc")?,
            matvec_swiglu: pipe("matvec_swiglu")?,
            matvec_qkv: pipe("matvec_qkv")?,
            matvec_h: pipe("matvec_h")?,
            matmul: pipe("matmul")?,
            matmul_t: pipe("matmul_t")?,
            f32_to_f16: pipe("f32_to_f16")?,
            bias_add: pipe("bias_add")?,
            matmul_h: pipe("matmul_h")?,
            rmsnorm: pipe("rmsnorm")?,
            rmsnorm_hf: pipe("rmsnorm_hf")?,
            silu_mul_hf: pipe("silu_mul_hf")?,
            rope: pipe("rope")?,
            rope_h: pipe("rope_h")?,
            rope_qk_decode: pipe("rope_qk_decode")?,
            attention: pipe("attention")?,
            attention_prefill_flash: pipe("attention_prefill_flash")?,
            attention_decode_partial: gqa_pipe("attention_decode_partial")?,
            attention_decode_reduce: pipe("attention_decode_reduce")?,
            silu_mul: pipe("silu_mul")?,
            add_inplace: pipe("add_inplace")?,
            matvec_qkv_batch: pipe("matvec_qkv_batch")?,
            matvec_acc_batch: pipe("matvec_acc_batch")?,
            matvec_swiglu_batch: pipe("matvec_swiglu_batch")?,
            rope_qk_batch: pipe("rope_qk_batch")?,
            attention_decode_partial_batch: gqa_pipe("attention_decode_partial_batch")?,
            attention_decode_reduce_batch: pipe("attention_decode_reduce_batch")?,
        };

        fn lin(device: &Device, l: &crate::model::Linear) -> GpuLinear {
            let has_bias = l.bias.is_some();
            let zero_bias;
            let bias = match &l.bias {
                Some(b) => b,
                None => {
                    zero_bias = vec![0.0; l.out_dim];
                    &zero_bias
                }
            };
            GpuLinear {
                w: f16_buffer(device, &l.w),
                bias: f16_buffer(device, bias),
                has_bias,
                in_dim: l.in_dim as u32,
                out_dim: l.out_dim as u32,
            }
        }

        let blocks = model
            .blocks
            .iter()
            .map(|b| GpuBlock {
                input_layernorm: f16_buffer(&device, &b.input_layernorm),
                q_proj: lin(&device, &b.q_proj),
                k_proj: lin(&device, &b.k_proj),
                v_proj: lin(&device, &b.v_proj),
                o_proj: lin(&device, &b.o_proj),
                post_attention_layernorm: f16_buffer(&device, &b.post_attention_layernorm),
                gate_proj: lin(&device, &b.gate_proj),
                up_proj: lin(&device, &b.up_proj),
                down_proj: lin(&device, &b.down_proj),
            })
            .collect();

        eprintln!("Metal: {} — weights converted to f16 and resident on the GPU", device.name());
        Ok(Self {
            embed_tokens: f16_buffer(&device, &model.embed_tokens),
            norm: f16_buffer(&device, &model.norm),
            lm_head: lin(&device, &model.lm_head),
            cfg: model.cfg,
            queue,
            device,
            pipes,
        blocks,
        })
    }

    // ---------- one encode helper per kernel ----------
    // set_bytes passes tiny parameters (positions, sizes) without allocating buffers.
    // Everything takes n_rows: decode passes 1, prefill passes the chunk size —
    // the only difference is the dispatched grid.

    fn enc_embed(&self, enc: &ComputeCommandEncoderRef, ids: &Buffer, x: &Buffer, n_rows: usize) {
        let p = EmbedParams { dim: self.cfg.hidden_size as u32, n_rows: n_rows as u32 };
        enc.set_compute_pipeline_state(&self.pipes.embed);
        enc.set_buffer(0, Some(&self.embed_tokens), 0);
        enc.set_buffer(1, Some(ids), 0);
        enc.set_buffer(2, Some(x), 0);
        enc.set_bytes(3, size_of::<EmbedParams>() as u64, &p as *const _ as *const _);
        dispatch_grid(enc, n_rows * self.cfg.hidden_size);
    }

    /// Y (at byte offset y_off) = X·Wᵀ + bias for n_rows tokens.
    /// Single row → the matvec kernel (bandwidth-friendly); multiple rows → tiled matmul.
    /// The y offset lets k/v projections write straight into the cache — no copy.
    #[allow(clippy::too_many_arguments)]
    fn enc_linear(
        &self,
        enc: &ComputeCommandEncoderRef,
        l: &GpuLinear,
        x: &Buffer,
        x_off: u64,
        y: &Buffer,
        y_off: u64,
        n_rows: usize,
        xh: Option<&Buffer>,
        convert: bool,
    ) {
        self.enc_linear_with(&self.pipes.matvec, &self.pipes.matmul, enc, l, x, x_off, y, y_off, n_rows, xh, convert);
    }

    /// enc_linear writing f16 — the k/v projections, whose output IS the KV cache.
    #[allow(clippy::too_many_arguments)]
    fn enc_linear_kv(
        &self,
        enc: &ComputeCommandEncoderRef,
        l: &GpuLinear,
        x: &Buffer,
        y: &Buffer,
        y_off: u64,
        n_rows: usize,
    ) {
        self.enc_linear_with(&self.pipes.matvec_h, &self.pipes.matmul_h, enc, l, x, 0, y, y_off, n_rows, None, false);
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_linear_with(
        &self,
        matvec: &ComputePipelineState,
        matmul: &ComputePipelineState,
        enc: &ComputeCommandEncoderRef,
        l: &GpuLinear,
        x: &Buffer,
        x_off: u64,
        y: &Buffer,
        y_off: u64,
        n_rows: usize,
        xh: Option<&Buffer>,
        convert: bool,
    ) {
        if n_rows == 1 {
            let p = MatvecParams { in_dim: l.in_dim, out_dim: l.out_dim };
            enc.set_compute_pipeline_state(matvec);
            enc.set_buffer(0, Some(&l.w), 0);
            enc.set_buffer(1, Some(&l.bias), 0);
            enc.set_buffer(2, Some(x), x_off);
            enc.set_buffer(3, Some(y), y_off);
            enc.set_bytes(4, size_of::<MatvecParams>() as u64, &p as *const _ as *const _);
            dispatch_simdgroup_rows(enc, l.out_dim);
        } else if let Some(xh) = xh {
            // Tensor-ops path (Metal 4): run mpp matmul2d from the half input copy
            // straight into the f32 output; convert here only when no upstream
            // kernel already emitted the half copy. Bias only where a layer has one.
            let p = MatmulParams { in_dim: l.in_dim, out_dim: l.out_dim, n_rows: n_rows as u32 };
            if convert {
                let dim = (n_rows * l.in_dim as usize) as u32;
                enc.set_compute_pipeline_state(&self.pipes.f32_to_f16);
                enc.set_buffer(0, Some(x), x_off);
                enc.set_buffer(1, Some(xh), 0);
                enc.set_bytes(2, 4, &dim as *const _ as *const _);
                dispatch_grid(enc, dim as usize);
            }

            enc.set_compute_pipeline_state(&self.pipes.matmul_t);
            enc.set_buffer(0, Some(&l.w), 0);
            enc.set_buffer(1, Some(xh), 0);
            enc.set_buffer(2, Some(y), y_off);
            enc.set_bytes(3, size_of::<MatmulParams>() as u64, &p as *const _ as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((l.out_dim as u64).div_ceil(64), (n_rows as u64).div_ceil(32), 1),
                MTLSize::new(128, 1, 1),
            );

            if l.has_bias {
                enc.set_compute_pipeline_state(&self.pipes.bias_add);
                enc.set_buffer(0, Some(y), y_off);
                enc.set_buffer(1, Some(&l.bias), 0);
                enc.set_bytes(2, size_of::<MatmulParams>() as u64, &p as *const _ as *const _);
                dispatch_grid(enc, n_rows * l.out_dim as usize);
            }
        } else {
            let p = MatmulParams { in_dim: l.in_dim, out_dim: l.out_dim, n_rows: n_rows as u32 };
            enc.set_compute_pipeline_state(matmul);
            enc.set_buffer(0, Some(&l.w), 0);
            enc.set_buffer(1, Some(&l.bias), 0);
            enc.set_buffer(2, Some(x), x_off);
            enc.set_buffer(3, Some(y), y_off);
            enc.set_bytes(4, size_of::<MatmulParams>() as u64, &p as *const _ as *const _);
            // 2D grid: (tiles of 64 outputs) × (tiles of 32 tokens), 128 threads =
            // 4 simdgroups per tile — see MM_* in kernels.metal.
            let tiles_out = (l.out_dim as u64).div_ceil(64);
            let tiles_row = (n_rows as u64).div_ceil(32);
            enc.dispatch_thread_groups(
                MTLSize::new(tiles_out, tiles_row, 1),
                MTLSize::new(128, 1, 1),
            );
        }
    }

    fn enc_rmsnorm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        weight: &Buffer,
        y: &Buffer,
        n_rows: usize,
    ) {
        let p = NormParams { dim: self.cfg.hidden_size as u32, eps: self.cfg.rms_norm_eps };
        enc.set_compute_pipeline_state(&self.pipes.rmsnorm);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(y), 0);
        enc.set_bytes(3, size_of::<NormParams>() as u64, &p as *const _ as *const _);
        // One threadgroup (256 threads = NORM_TG) per row.
        enc.dispatch_thread_groups(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Prefill rmsnorm: writes the normalized row in f32 AND half (the half copy
    /// feeds the tensor-ops matmuls without a separate conversion pass).
    fn enc_rmsnorm_hf(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        weight: &Buffer,
        y: &Buffer,
        y_h: &Buffer,
        n_rows: usize,
    ) {
        let p = NormParams { dim: self.cfg.hidden_size as u32, eps: self.cfg.rms_norm_eps };
        enc.set_compute_pipeline_state(&self.pipes.rmsnorm_hf);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(y), 0);
        enc.set_buffer(3, Some(y_h), 0);
        enc.set_bytes(4, size_of::<NormParams>() as u64, &p as *const _ as *const _);
        enc.dispatch_thread_groups(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_off: u64,
        n_heads: usize,
        pos0: usize,
        n_rows: usize,
        f16: bool, // true = an f16 buffer (rows in the KV cache), false = f32 (q)
    ) {
        let hd = self.cfg.head_dim();
        let p = RopeParams {
            head_dim: hd as u32,
            n_heads: n_heads as u32,
            pos0: pos0 as u32,
            theta: self.cfg.rope_theta,
            n_rows: n_rows as u32,
        };
        enc.set_compute_pipeline_state(if f16 { &self.pipes.rope_h } else { &self.pipes.rope });
        enc.set_buffer(0, Some(x), x_off);
        enc.set_bytes(1, size_of::<RopeParams>() as u64, &p as *const _ as *const _);
        dispatch_grid(enc, n_rows * n_heads * hd / 2);
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        q: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        cache_off: u64,
        scores: &Buffer,
        out: &Buffer,
        pos0: usize,
        n_rows: usize,
        max_seq: usize,
        out_h: &Buffer,
    ) {
        let p = AttnParams {
            head_dim: self.cfg.head_dim() as u32,
            n_heads: self.cfg.num_attention_heads as u32,
            n_kv_heads: self.cfg.num_key_value_heads as u32,
            pos0: pos0 as u32,
            max_seq: max_seq as u32,
            n_rows: n_rows as u32,
        };
        if self.cfg.head_dim() == FLASH_HEAD_DIM {
            // Flash path: no scores scratch, one threadgroup per (head, 16-row tile).
            enc.set_compute_pipeline_state(&self.pipes.attention_prefill_flash);
            enc.set_buffer(0, Some(q), 0);
            enc.set_buffer(1, Some(k_cache), cache_off);
            enc.set_buffer(2, Some(v_cache), cache_off);
            enc.set_buffer(3, Some(out), 0);
            enc.set_bytes(4, size_of::<AttnParams>() as u64, &p as *const _ as *const _);
            enc.set_buffer(5, Some(out_h), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(self.cfg.num_attention_heads as u64, n_rows.div_ceil(32) as u64, 1),
                MTLSize::new(128, 1, 1),
            );
            return;
        }
        enc.set_compute_pipeline_state(&self.pipes.attention);
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k_cache), cache_off);
        enc.set_buffer(2, Some(v_cache), cache_off);
        enc.set_buffer(3, Some(scores), 0);
        enc.set_buffer(4, Some(out), 0);
        enc.set_bytes(5, size_of::<AttnParams>() as u64, &p as *const _ as *const _);
        // 2D grid: (query heads) × (query rows in the chunk) — one threadgroup per pair.
        enc.dispatch_thread_groups(
            MTLSize::new(self.cfg.num_attention_heads as u64, n_rows as u64, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Decode-only: x[row] += W·in + bias — o_proj/down_proj with the residual fused.
    fn enc_matvec_acc(&self, enc: &ComputeCommandEncoderRef, l: &GpuLinear, x: &Buffer, y: &Buffer) {
        let p = MatvecParams { in_dim: l.in_dim, out_dim: l.out_dim };
        enc.set_compute_pipeline_state(&self.pipes.matvec_acc);
        enc.set_buffer(0, Some(&l.w), 0);
        enc.set_buffer(1, Some(&l.bias), 0);
        enc.set_buffer(2, Some(x), 0);
        enc.set_buffer(3, Some(y), 0);
        enc.set_bytes(4, size_of::<MatvecParams>() as u64, &p as *const _ as *const _);
        dispatch_simdgroup_rows(enc, l.out_dim);
    }

    /// Decode-only: y = silu(Wg·x) * (Wu·x) — the SwiGLU inner step in one dispatch.
    fn enc_swiglu(
        &self,
        enc: &ComputeCommandEncoderRef,
        gate: &GpuLinear,
        up: &GpuLinear,
        x: &Buffer,
        y: &Buffer,
    ) {
        let p = MatvecParams { in_dim: gate.in_dim, out_dim: gate.out_dim };
        enc.set_compute_pipeline_state(&self.pipes.matvec_swiglu);
        enc.set_buffer(0, Some(&gate.w), 0);
        enc.set_buffer(1, Some(&up.w), 0);
        enc.set_buffer(2, Some(x), 0);
        enc.set_buffer(3, Some(y), 0);
        enc.set_bytes(4, size_of::<MatvecParams>() as u64, &p as *const _ as *const _);
        dispatch_simdgroup_rows(enc, gate.out_dim);
    }

    /// Decode-only: q,k,v projections in one dispatch, k and v written straight into
    /// this position's cache slot.
    #[allow(clippy::too_many_arguments)]
    fn enc_qkv(
        &self,
        enc: &ComputeCommandEncoderRef,
        blk: &GpuBlock,
        x: &Buffer,
        q: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        kv_off_elems: usize,
    ) {
        let p = QkvParams {
            in_dim: blk.q_proj.in_dim,
            q_dim: blk.q_proj.out_dim,
            kv_dim: blk.k_proj.out_dim,
            kv_off: kv_off_elems as u32,
        };
        enc.set_compute_pipeline_state(&self.pipes.matvec_qkv);
        enc.set_buffer(0, Some(&blk.q_proj.w), 0);
        enc.set_buffer(1, Some(&blk.q_proj.bias), 0);
        enc.set_buffer(2, Some(&blk.k_proj.w), 0);
        enc.set_buffer(3, Some(&blk.k_proj.bias), 0);
        enc.set_buffer(4, Some(&blk.v_proj.w), 0);
        enc.set_buffer(5, Some(&blk.v_proj.bias), 0);
        enc.set_buffer(6, Some(x), 0);
        enc.set_buffer(7, Some(q), 0);
        enc.set_buffer(8, Some(k_cache), 0);
        enc.set_buffer(9, Some(v_cache), 0);
        enc.set_bytes(10, size_of::<QkvParams>() as u64, &p as *const _ as *const _);
        dispatch_simdgroup_rows(enc, blk.q_proj.out_dim + 2 * blk.k_proj.out_dim);
    }

    /// Decode-only: RoPE on q and this position's new k cache row, one dispatch.
    fn enc_rope_qk(
        &self,
        enc: &ComputeCommandEncoderRef,
        q: &Buffer,
        k_cache: &Buffer,
        kv_byte_off: u64,
        pos: usize,
    ) {
        let hd = self.cfg.head_dim();
        let p = RopeQkParams {
            head_dim: hd as u32,
            n_q_heads: self.cfg.num_attention_heads as u32,
            n_kv_heads: self.cfg.num_key_value_heads as u32,
            pos: pos as u32,
            theta: self.cfg.rope_theta,
        };
        enc.set_compute_pipeline_state(&self.pipes.rope_qk_decode);
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k_cache), kv_byte_off);
        enc.set_bytes(2, size_of::<RopeQkParams>() as u64, &p as *const _ as *const _);
        dispatch_grid(enc, (self.cfg.num_attention_heads + self.cfg.num_key_value_heads) * hd / 2);
    }

    /// Decode-only attention (n_rows = 1): flash-decoding split. Falls back to the
    /// generic kernel via the caller when head_dim > DEC_TG.
    #[allow(clippy::too_many_arguments)]
    fn enc_attention_decode(
        &self,
        enc: &ComputeCommandEncoderRef,
        q: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        cache_off: u64,
        partials: &Buffer,
        out: &Buffer,
        pos: usize,
    ) {
        let n_splits = (pos + 1).div_ceil(ATTN_SPLIT);
        let p = AttnDecParams {
            head_dim: self.cfg.head_dim() as u32,
            n_heads: self.cfg.num_attention_heads as u32,
            n_kv_heads: self.cfg.num_key_value_heads as u32,
            pos: pos as u32,
            n_splits: n_splits as u32,
        };
        let heads = self.cfg.num_attention_heads as u64;
        let (grid_x, tg_mem) = self.gqa_decode_dims();
        enc.set_compute_pipeline_state(&self.pipes.attention_decode_partial);
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k_cache), cache_off);
        enc.set_buffer(2, Some(v_cache), cache_off);
        enc.set_buffer(3, Some(partials), 0);
        enc.set_bytes(4, size_of::<AttnDecParams>() as u64, &p as *const _ as *const _);
        for (i, len) in tg_mem.iter().enumerate() {
            enc.set_threadgroup_memory_length(i as u64, *len);
        }
        enc.dispatch_thread_groups(
            MTLSize::new(grid_x, n_splits as u64, 1),
            MTLSize::new(DEC_TG as u64, 1, 1),
        );

        enc.set_compute_pipeline_state(&self.pipes.attention_decode_reduce);
        enc.set_buffer(0, Some(partials), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_bytes(2, size_of::<AttnDecParams>() as u64, &p as *const _ as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(heads, 1, 1),
            MTLSize::new(self.cfg.head_dim() as u64, 1, 1),
        );
    }

    /// Grid width and threadgroup memory sizes for the GQA-aware decode partial
    /// kernels: one threadgroup per (kv head × group chunk, window), covering up to
    /// MAX_GQA_CHUNK q heads of one kv head's group.
    fn gqa_decode_dims(&self) -> (u64, [u64; 4]) {
        let cfg = &self.cfg;
        let group = cfg.num_attention_heads / cfg.num_key_value_heads;
        let chunk = group.min(MAX_GQA_CHUNK);
        let grid_x = (cfg.num_key_value_heads * group.div_ceil(chunk)) as u64;
        // Sizes must mirror the kernel's q_s / es / acc_red / red layouts, padded to
        // Metal's 16-byte threadgroup-allocation granularity.
        let f32s = |n: usize| (n * 4).next_multiple_of(16) as u64;
        (
            grid_x,
            [
                f32s(chunk * cfg.head_dim()),
                f32s(chunk * ATTN_SPLIT),
                f32s(DEC_TG * (chunk | 1)),
                f32s(chunk * (DEC_TG / 32) + chunk),
            ],
        )
    }

    fn enc_elementwise(
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
        enc.set_bytes(2, size_of::<ElemParams>() as u64, &p as *const _ as *const _);
        dispatch_grid(enc, dim);
    }
}

/// One-thread-per-element dispatch — kernels guard the tail with `if (gid < dim)`.
fn dispatch_grid(enc: &ComputeCommandEncoderRef, n: usize) {
    let tg = 256u64;
    enc.dispatch_thread_groups(MTLSize::new((n as u64).div_ceil(tg), 1, 1), MTLSize::new(tg, 1, 1));
}

/// One-simdgroup-per-output-row dispatch, 4 rows per threadgroup — shared by every
/// matvec-family kernel (they guard the tail with `if (row >= out_dim)`).
fn dispatch_simdgroup_rows(enc: &ComputeCommandEncoderRef, rows: u32) {
    let rows_per_tg = 4u64;
    let tgs = (rows as u64).div_ceil(rows_per_tg);
    enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(32 * rows_per_tg, 1, 1));
}

impl Engine for MetalEngine {
    fn name(&self) -> &'static str {
        "metal"
    }
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }
    fn session(&self, max_seq: usize) -> crate::Result<Box<dyn Session + '_>> {
        Ok(Box::new(self.raw_session(max_seq)))
    }
    fn batcher(&self, n_slots: usize, max_seq: usize) -> Option<Box<dyn crate::engine::Batcher + '_>> {
        self.make_batcher(n_slots, max_seq)
            .map(|b| Box::new(b) as Box<dyn crate::engine::Batcher>)
    }
}

impl MetalEngine {
    /// Build a session as a concrete type — the ane backend needs write_kv/prefill_from.
    pub(crate) fn raw_session(&self, max_seq: usize) -> MetalSession<'_> {
        let cfg = &self.cfg;
        let d = &self.device;
        let caches = (0..cfg.num_hidden_layers)
            .map(|_| f16_empty_buffer(d, max_seq * cfg.kv_dim()))
            .collect::<Vec<_>>();
        let v_caches = (0..cfg.num_hidden_layers)
            .map(|_| f16_empty_buffer(d, max_seq * cfg.kv_dim()))
            .collect::<Vec<_>>();
        let scratch = self.session_scratch(max_seq);
        self.session_with_cache(max_seq, caches, v_caches, 0, scratch)
    }

    /// GPU scratch for one session: activations, logits, attention partials, and the
    /// prefill scores buffer (chunk × heads × max_seq floats — by far the big one).
    /// Buffers are refcounted, so this can be allocated once and cloned into many
    /// sessions as long as no two of them run at the same time.
    fn session_scratch(&self, max_seq: usize) -> SessionScratch {
        let cfg = &self.cfg;
        let d = &self.device;
        let chunk = PREFILL_CHUNK.min(max_seq); // scratch sized per chunk (decode uses row 0 only)
        SessionScratch {
            ids: d.new_buffer((chunk * 4) as u64, MTLResourceOptions::StorageModeShared),
            x: f32_buffer(d, chunk * cfg.hidden_size),
            xn: f32_buffer(d, chunk * cfg.hidden_size),
            q: f32_buffer(d, chunk * cfg.hidden_size),
            att: f32_buffer(d, chunk * cfg.hidden_size),
            xb: f32_buffer(d, chunk * cfg.hidden_size),
            gate: f32_buffer(d, chunk * cfg.intermediate_size),
            up: f32_buffer(d, chunk * cfg.intermediate_size),
            logits: f32_buffer(d, SPEC_MAX * cfg.vocab_size),
            // The flash prefill path never touches scores; keep a 1-float stub so
            // the fallback binding stays valid without the (huge) allocation.
            scores: if cfg.head_dim() == FLASH_HEAD_DIM {
                f32_buffer(d, 1)
            } else {
                f32_buffer(d, chunk * cfg.num_attention_heads * max_seq)
            },
            partials: f32_buffer(
                d,
                cfg.num_attention_heads * max_seq.div_ceil(ATTN_SPLIT) * (cfg.head_dim() + 2),
            ),
            xh: f16_empty_buffer(d, chunk * cfg.hidden_size.max(cfg.intermediate_size)),
        }
    }

    /// A session whose KV cache lives inside a shared pool (continuous batching):
    /// the buffers are shared, kv_base points at this session's slot.
    pub(crate) fn session_with_cache(
        &self,
        max_seq: usize,
        k_cache: Vec<Buffer>,
        v_cache: Vec<Buffer>,
        kv_base: u64,
        scratch: SessionScratch,
    ) -> MetalSession<'_> {
        let SessionScratch { ids, x, xn, q, att, xb, gate, up, logits, scores, partials, xh } = scratch;
        MetalSession {
            ids,
            x,
            xn,
            q,
            att,
            xb,
            gate,
            up,
            logits,
            scores,
            partials,
            xh,
            k_cache,
            v_cache,
            kv_base,
            max_seq,
            engine: self,
        }
    }
}

/// The scratch buffers behind a MetalSession, separated from the KV cache so the
/// batcher can allocate them ONCE and reuse them for every admitted request —
/// re-allocating scores alone would cost ~hundreds of MB per admission.
#[derive(Clone)]
pub(crate) struct SessionScratch {
    ids: Buffer,
    x: Buffer,
    xn: Buffer,
    q: Buffer,
    att: Buffer,
    xb: Buffer,
    gate: Buffer,
    up: Buffer,
    logits: Buffer,
    scores: Buffer,
    partials: Buffer,
    /// Half staging for the tensor-ops matmul inputs (widest activation row).
    xh: Buffer,
}

/// Where a chunk's layer-`layer0` input comes from (see `MetalSession::run_from`).
#[derive(Clone, Copy)]
enum Source<'a> {
    Ids(&'a [u32]),
    Hidden(&'a [f32]),
}

/// One generation run's GPU state — the KV cache lives on the GPU and never leaves it.
pub(crate) struct MetalSession<'a> {
    engine: &'a MetalEngine,
    ids: Buffer,
    x: Buffer,
    xn: Buffer,
    q: Buffer,
    att: Buffer,
    xb: Buffer,
    gate: Buffer,
    up: Buffer,
    logits: Buffer,
    scores: Buffer,
    partials: Buffer, // decode attention: [heads × windows × (head_dim + 2)]
    xh: Buffer,       // half staging for tensor-ops matmul inputs
    k_cache: Vec<Buffer>,
    v_cache: Vec<Buffer>,
    kv_base: u64, // byte offset of this session's slot when the cache is pooled
    max_seq: usize,
}

impl MetalSession<'_> {
    /// Process n_rows tokens (positions pos0..pos0+n_rows) in one command buffer.
    /// The dispatch order below mirrors Model::forward on the CPU line by line.
    /// `logits_rows` = how many trailing positions need logits: 0 for intermediate
    /// prefill chunks, 1 for decode / the last chunk, n for speculative verification.
    fn run(&mut self, ids: &[u32], pos0: usize, logits_rows: usize) -> crate::Result<Vec<f32>> {
        self.run_from(Source::Ids(ids), ids.len(), pos0, 0, logits_rows)
    }

    /// The body of a chunk: `n` tokens at positions pos0.. through layers
    /// `layer0..L`, then optionally the final norm and lm_head. `src` says where
    /// the layer-`layer0` input comes from — token ids (embed on the GPU, the
    /// normal path) or a hidden state computed elsewhere (split prefill: the ANE
    /// ran layers 0..layer0 for this chunk, see src/ane.rs).
    fn run_from(
        &mut self,
        src: Source<'_>,
        n: usize,
        pos0: usize,
        layer0: usize,
        logits_rows: usize,
    ) -> crate::Result<Vec<f32>> {
        let e = self.engine;
        let cfg = &e.cfg;
        let h = cfg.hidden_size;
        let kv_byte_off = self.kv_base + (pos0 * cfg.kv_dim() * 2) as u64; // this chunk's first (f16) cache row

        // Unified memory: write the input straight into the buffer pre-commit.
        match src {
            Source::Ids(ids) => unsafe {
                std::ptr::copy_nonoverlapping(ids.as_ptr(), self.ids.contents() as *mut u32, n)
            },
            Source::Hidden(x) => unsafe {
                std::ptr::copy_nonoverlapping(x.as_ptr(), self.x.contents() as *mut f32, n * h)
            },
        }

        let cb = e.queue.new_command_buffer();
        // Serial encoder: dispatches execute in order, each seeing its predecessors'
        // results — no manual barriers needed.
        let enc = cb.new_compute_command_encoder();

        if matches!(src, Source::Ids(_)) {
            e.enc_embed(enc, &self.ids, &self.x, n);
        }
        let fused_decode = n == 1 && cfg.head_dim() <= DEC_TG && cfg.head_dim().is_multiple_of(4);
        for (l, blk) in e.blocks.iter().enumerate().skip(layer0) {
            if fused_decode {
                // Decode path: fused kernels — 9 dispatches per layer instead of 15.
                // Same math as the prefill path below, with qkv / swiglu / residual
                // adds folded into single launches and flash-decoding attention.
                e.enc_rmsnorm(enc, &self.x, &blk.input_layernorm, &self.xn, 1);
                let kv_off_elems = (self.kv_base / 2) as usize + pos0 * cfg.kv_dim();
                e.enc_qkv(enc, blk, &self.xn, &self.q, &self.k_cache[l], &self.v_cache[l], kv_off_elems);
                e.enc_rope_qk(enc, &self.q, &self.k_cache[l], kv_byte_off, pos0);
                e.enc_attention_decode(enc, &self.q, &self.k_cache[l], &self.v_cache[l], self.kv_base, &self.partials, &self.att, pos0);
                e.enc_matvec_acc(enc, &blk.o_proj, &self.att, &self.x);
                e.enc_rmsnorm(enc, &self.x, &blk.post_attention_layernorm, &self.xn, 1);
                e.enc_swiglu(enc, &blk.gate_proj, &blk.up_proj, &self.xn, &self.gate);
                e.enc_matvec_acc(enc, &blk.down_proj, &self.gate, &self.x);
                continue;
            }

            // Prefill path (and the rare head_dim > DEC_TG decode): tiled matmuls.
            // Attention half.
            e.enc_rmsnorm_hf(enc, &self.x, &blk.input_layernorm, &self.xn, &self.xh, n);
            e.enc_linear(enc, &blk.q_proj, &self.xn, 0, &self.q, 0, n, Some(&self.xh), false);
            e.enc_linear_kv(enc, &blk.k_proj, &self.xn, &self.k_cache[l], kv_byte_off, n);
            e.enc_linear_kv(enc, &blk.v_proj, &self.xn, &self.v_cache[l], kv_byte_off, n);
            e.enc_rope(enc, &self.q, 0, cfg.num_attention_heads, pos0, n, false);
            e.enc_rope(enc, &self.k_cache[l], kv_byte_off, cfg.num_key_value_heads, pos0, n, true);
            e.enc_attention(enc, &self.q, &self.k_cache[l], &self.v_cache[l], self.kv_base, &self.scores, &self.att, pos0, n, self.max_seq, &self.xh);
            e.enc_linear(enc, &blk.o_proj, &self.att, 0, &self.xb, 0, n, Some(&self.xh), e.cfg.head_dim() != FLASH_HEAD_DIM);
            e.enc_elementwise(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);

            // SwiGLU MLP half.
            e.enc_rmsnorm_hf(enc, &self.x, &blk.post_attention_layernorm, &self.xn, &self.xh, n);
            e.enc_linear(enc, &blk.gate_proj, &self.xn, 0, &self.gate, 0, n, Some(&self.xh), false);
            e.enc_linear(enc, &blk.up_proj, &self.xn, 0, &self.up, 0, n, Some(&self.xh), false);
            {
                let p = ElemParams { dim: (n * cfg.intermediate_size) as u32 };
                enc.set_compute_pipeline_state(&e.pipes.silu_mul_hf);
                enc.set_buffer(0, Some(&self.gate), 0);
                enc.set_buffer(1, Some(&self.up), 0);
                enc.set_buffer(2, Some(&self.xh), 0);
                enc.set_bytes(3, size_of::<ElemParams>() as u64, &p as *const _ as *const _);
                dispatch_grid(enc, n * cfg.intermediate_size);
            }
            e.enc_linear(enc, &blk.down_proj, &self.gate, 0, &self.xb, 0, n, Some(&self.xh), false);
            e.enc_elementwise(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);
        }
        if logits_rows > 0 {
            // Norm every row (cheap), then run the big lm_head only on the rows whose
            // logits are wanted — the final one for decode, all of them for verification.
            e.enc_rmsnorm_hf(enc, &self.x, &e.norm, &self.xn, &self.xh, n);
            let first = n - logits_rows;
            e.enc_linear(enc, &e.lm_head, &self.xn, (first * h * 4) as u64, &self.logits, 0, logits_rows, Some(&self.xh), true);
        }

        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed(); // the single sync point for the whole chunk

        if logits_rows == 0 {
            return Ok(Vec::new());
        }
        // Unified memory: read logits straight out of the buffer, no device copy.
        let logits = unsafe {
            std::slice::from_raw_parts(self.logits.contents() as *const f32, logits_rows * cfg.vocab_size)
        };
        Ok(logits.to_vec())
    }

    /// The model config, for callers (the ane backend) that only hold a session.
    pub(crate) fn config_ref(&self) -> &ModelConfig {
        &self.engine.cfg
    }

    /// Write K,V computed elsewhere (e.g. on the ANE) into the cache at pos0 onward,
    /// converting to the cache's f16 on the way. With unified memory this is the
    /// whole "device transfer".
    pub(crate) fn write_kv(&mut self, layer: usize, pos0: usize, k: &[f32], v: &[f32]) {
        let kvd = self.engine.cfg.kv_dim();
        let base = (self.kv_base / 2) as usize;
        unsafe {
            let kp = (self.k_cache[layer].contents() as *mut u16).add(base + pos0 * kvd);
            for (i, &x) in k.iter().enumerate() {
                *kp.add(i) = f16::from_f32(x).to_bits();
            }
            let vp = (self.v_cache[layer].contents() as *mut u16).add(base + pos0 * kvd);
            for (i, &x) in v.iter().enumerate() {
                *vp.add(i) = f16::from_f32(x).to_bits();
            }
        }
    }

    /// Same as `write_kv`, for K,V that already carry the cache's f16 bits — one
    /// memcpy per layer instead of a per-element convert. Split prefill converts
    /// on the ANE thread (it needs the f16 rows anyway, to feed the next chunk's
    /// past), which keeps the conversion off the thread driving the GPU.
    pub(crate) fn write_kv_bits(&mut self, layer: usize, pos0: usize, k: &[u16], v: &[u16]) {
        let kvd = self.engine.cfg.kv_dim();
        let base = (self.kv_base / 2) as usize;
        unsafe {
            let kp = (self.k_cache[layer].contents() as *mut u16).add(base + pos0 * kvd);
            std::ptr::copy_nonoverlapping(k.as_ptr(), kp, k.len());
            let vp = (self.v_cache[layer].contents() as *mut u16).add(base + pos0 * kvd);
            std::ptr::copy_nonoverlapping(v.as_ptr(), vp, v.len());
        }
    }

    /// How many rows the per-chunk scratch buffers hold — the hard ceiling on `n`
    /// for any single `run_from`. Split prefill checks its graph's chunk width
    /// against this before writing a hidden state into `x`.
    pub(crate) fn max_chunk_rows(&self) -> usize {
        PREFILL_CHUNK.min(self.max_seq)
    }

    /// Split prefill: finish a chunk whose leading `layer0` layers ran on another
    /// device. `x` is that device's hidden state for these `n` rows (row-major
    /// [n, hidden_size]); layers 0..layer0 of the KV cache must already hold this
    /// chunk's rows. Returns the last row's logits when `want_logits`.
    pub(crate) fn prefill_tail_layers(
        &mut self,
        x: &[f32],
        pos0: usize,
        n: usize,
        layer0: usize,
        want_logits: bool,
    ) -> crate::Result<Vec<f32>> {
        self.run_from(Source::Hidden(x), n, pos0, layer0, want_logits as usize)
    }

    /// Batch prefill continuing from pos0 (cache slots 0..pos0 must already be filled)
    /// — the ane backend uses this to take over after the ANE's portion.
    pub(crate) fn prefill_from(&mut self, ids: &[u32], mut pos0: usize) -> crate::Result<Vec<f32>> {
        let end = pos0 + ids.len();
        let mut logits = Vec::new();
        for chunk in ids.chunks(PREFILL_CHUNK) {
            let is_last = pos0 + chunk.len() == end;
            logits = self.run(chunk, pos0, is_last as usize)?;
            pos0 += chunk.len();
        }
        Ok(logits)
    }
}

// ---------- continuous batching ----------

/// The serve-mode batcher: a pooled KV cache ([slot][max_seq][kv_dim] per layer,
/// physical pages committed lazily by the OS) plus scratch for one batched decode
/// step. Prefill runs per request through a slot-backed MetalSession; decode
/// advances every active request in one submission, so one read of the weights
/// serves all of them.
pub(crate) struct MetalBatcher<'a> {
    engine: &'a MetalEngine,
    max_seq: usize,
    splits_max: usize,
    k_cache: Vec<Buffer>, // per layer: [n_slots × max_seq × kv_dim] f16
    v_cache: Vec<Buffer>,
    ids: Buffer,
    meta: Buffer,
    x: Buffer,
    xn: Buffer,
    q: Buffer,
    att: Buffer,
    gate: Buffer,
    logits: Buffer,
    partials: Buffer,
    /// One prefill-session scratch set shared by every slot admission (slot sessions
    /// are created and dropped one at a time on the scheduler thread).
    session_scratch: SessionScratch,
}

// Same justification as MetalEngine: every Metal object here is documented
// thread-safe, and the batcher itself lives on one scheduler thread.
unsafe impl Send for MetalBatcher<'_> {}

impl MetalEngine {
    pub(crate) fn make_batcher(&self, n_slots: usize, max_seq: usize) -> Option<MetalBatcher<'_>> {
        let cfg = &self.cfg;
        // The batched kernels are the fused-decode family — same requirements.
        if cfg.head_dim() > DEC_TG || !cfg.head_dim().is_multiple_of(4) || n_slots > SPEC_MAX {
            return None;
        }
        let d = &self.device;
        let (h, kvd) = (cfg.hidden_size, cfg.kv_dim());
        let splits_max = max_seq.div_ceil(ATTN_SPLIT);
        Some(MetalBatcher {
            k_cache: (0..cfg.num_hidden_layers)
                .map(|_| f16_empty_buffer(d, n_slots * max_seq * kvd))
                .collect(),
            v_cache: (0..cfg.num_hidden_layers)
                .map(|_| f16_empty_buffer(d, n_slots * max_seq * kvd))
                .collect(),
            ids: d.new_buffer((n_slots * 4) as u64, MTLResourceOptions::StorageModeShared),
            meta: d.new_buffer(
                (n_slots * size_of::<RowMeta>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            x: f32_buffer(d, n_slots * h),
            xn: f32_buffer(d, n_slots * h),
            q: f32_buffer(d, n_slots * h),
            att: f32_buffer(d, n_slots * h),
            gate: f32_buffer(d, n_slots * cfg.intermediate_size),
            logits: f32_buffer(d, n_slots * cfg.vocab_size),
            partials: f32_buffer(
                d,
                n_slots * cfg.num_attention_heads * splits_max * (cfg.head_dim() + 2),
            ),
            session_scratch: self.session_scratch(max_seq),
            splits_max,
            max_seq,
            engine: self,
        })
    }
}

impl MetalBatcher<'_> {
    /// A regular MetalSession whose cache is this pool's `slot` — prefill reuses the
    /// whole existing path (chunked matmul prefill, ANE handoff via write_kv).
    pub(crate) fn slot_session(&self, slot: usize) -> MetalSession<'_> {
        let kvd = self.engine.cfg.kv_dim();
        let base = (slot * self.max_seq * kvd * 2) as u64;
        self.engine.session_with_cache(
            self.max_seq,
            self.k_cache.clone(),
            self.v_cache.clone(),
            base,
            self.session_scratch.clone(),
        )
    }
}

impl crate::engine::Batcher for MetalBatcher<'_> {
    fn prefill(&mut self, slot: usize, ids: &[u32]) -> crate::Result<Vec<f32>> {
        self.slot_session(slot).prefill(ids)
    }

    /// One decode step for every active request — the fused decode dispatch
    /// sequence with a batch grid dimension, encoded into a single command buffer.
    fn decode_step(&mut self, rows: &[crate::engine::BatchRow]) -> crate::Result<Vec<Vec<f32>>> {
        let e = self.engine;
        let cfg = &e.cfg;
        let n = rows.len();
        let h = cfg.hidden_size;
        let (hd, kvd) = (cfg.head_dim(), cfg.kv_dim());
        unsafe {
            let idp = self.ids.contents() as *mut u32;
            let mp = self.meta.contents() as *mut RowMeta;
            for (i, r) in rows.iter().enumerate() {
                *idp.add(i) = r.token;
                *mp.add(i) = RowMeta { pos: r.pos as u32, slot: r.slot as u32 };
            }
        }
        let splits_now = rows.iter().map(|r| r.pos / ATTN_SPLIT + 1).max().unwrap_or(1);

        let cb = e.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        e.enc_embed(enc, &self.ids, &self.x, n);
        for (l, blk) in e.blocks.iter().enumerate() {
            e.enc_rmsnorm(enc, &self.x, &blk.input_layernorm, &self.xn, n);

            let p = QkvBatchParams {
                in_dim: h as u32,
                q_dim: blk.q_proj.out_dim,
                kv_dim: blk.k_proj.out_dim,
                max_seq: self.max_seq as u32,
            };
            enc.set_compute_pipeline_state(&e.pipes.matvec_qkv_batch);
            enc.set_buffer(0, Some(&blk.q_proj.w), 0);
            enc.set_buffer(1, Some(&blk.q_proj.bias), 0);
            enc.set_buffer(2, Some(&blk.k_proj.w), 0);
            enc.set_buffer(3, Some(&blk.k_proj.bias), 0);
            enc.set_buffer(4, Some(&blk.v_proj.w), 0);
            enc.set_buffer(5, Some(&blk.v_proj.bias), 0);
            enc.set_buffer(6, Some(&self.xn), 0);
            enc.set_buffer(7, Some(&self.q), 0);
            enc.set_buffer(8, Some(&self.k_cache[l]), 0);
            enc.set_buffer(9, Some(&self.v_cache[l]), 0);
            enc.set_bytes(10, size_of::<QkvBatchParams>() as u64, &p as *const _ as *const _);
            enc.set_buffer(11, Some(&self.meta), 0);
            let qkv_rows = (blk.q_proj.out_dim + 2 * blk.k_proj.out_dim) as u64;
            enc.dispatch_thread_groups(
                MTLSize::new(qkv_rows.div_ceil(4), n as u64, 1),
                MTLSize::new(128, 1, 1),
            );

            let p = RopeQkBatchParams {
                head_dim: hd as u32,
                n_q_heads: cfg.num_attention_heads as u32,
                n_kv_heads: cfg.num_key_value_heads as u32,
                theta: cfg.rope_theta,
                max_seq: self.max_seq as u32,
                kv_dim: kvd as u32,
                n_rows: n as u32,
            };
            enc.set_compute_pipeline_state(&e.pipes.rope_qk_batch);
            enc.set_buffer(0, Some(&self.q), 0);
            enc.set_buffer(1, Some(&self.k_cache[l]), 0);
            enc.set_buffer(2, Some(&self.meta), 0);
            enc.set_bytes(3, size_of::<RopeQkBatchParams>() as u64, &p as *const _ as *const _);
            dispatch_grid(enc, n * (cfg.num_attention_heads + cfg.num_key_value_heads) * hd / 2);

            let p = AttnDecBatchParams {
                head_dim: hd as u32,
                n_heads: cfg.num_attention_heads as u32,
                n_kv_heads: cfg.num_key_value_heads as u32,
                max_seq: self.max_seq as u32,
                kv_dim: kvd as u32,
                splits_max: self.splits_max as u32,
            };
            enc.set_compute_pipeline_state(&e.pipes.attention_decode_partial_batch);
            enc.set_buffer(0, Some(&self.q), 0);
            enc.set_buffer(1, Some(&self.k_cache[l]), 0);
            enc.set_buffer(2, Some(&self.v_cache[l]), 0);
            enc.set_buffer(3, Some(&self.partials), 0);
            enc.set_buffer(4, Some(&self.meta), 0);
            enc.set_bytes(5, size_of::<AttnDecBatchParams>() as u64, &p as *const _ as *const _);
            let (grid_x, tg_mem) = e.gqa_decode_dims();
            for (i, len) in tg_mem.iter().enumerate() {
                enc.set_threadgroup_memory_length(i as u64, *len);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(grid_x, splits_now as u64, n as u64),
                MTLSize::new(DEC_TG as u64, 1, 1),
            );
            enc.set_compute_pipeline_state(&e.pipes.attention_decode_reduce_batch);
            enc.set_buffer(0, Some(&self.partials), 0);
            enc.set_buffer(1, Some(&self.att), 0);
            enc.set_buffer(2, Some(&self.meta), 0);
            enc.set_bytes(3, size_of::<AttnDecBatchParams>() as u64, &p as *const _ as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(cfg.num_attention_heads as u64, n as u64, 1),
                MTLSize::new(hd as u64, 1, 1),
            );

            self.enc_matvec_batch(enc, &e.pipes.matvec_acc_batch, &blk.o_proj, &self.att, &self.x, n);
            e.enc_rmsnorm(enc, &self.x, &blk.post_attention_layernorm, &self.xn, n);
            self.enc_swiglu_batch(enc, blk, n);
            self.enc_matvec_batch(enc, &e.pipes.matvec_acc_batch, &blk.down_proj, &self.gate, &self.x, n);
        }
        e.enc_rmsnorm(enc, &self.x, &e.norm, &self.xn, n);
        e.enc_linear(enc, &e.lm_head, &self.xn, 0, &self.logits, 0, n, None, false);

        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let vocab = cfg.vocab_size;
        let flat = unsafe {
            std::slice::from_raw_parts(self.logits.contents() as *const f32, n * vocab)
        };
        Ok(flat.chunks(vocab).map(|c| c.to_vec()).collect())
    }
}

impl MetalBatcher<'_> {
    fn enc_matvec_batch(
        &self,
        enc: &ComputeCommandEncoderRef,
        pipe: &ComputePipelineState,
        l: &GpuLinear,
        x: &Buffer,
        y: &Buffer,
        n: usize,
    ) {
        let p = MatvecParams { in_dim: l.in_dim, out_dim: l.out_dim };
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&l.w), 0);
        enc.set_buffer(1, Some(&l.bias), 0);
        enc.set_buffer(2, Some(x), 0);
        enc.set_buffer(3, Some(y), 0);
        enc.set_bytes(4, size_of::<MatvecParams>() as u64, &p as *const _ as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((l.out_dim as u64).div_ceil(4), n as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    fn enc_swiglu_batch(&self, enc: &ComputeCommandEncoderRef, blk: &GpuBlock, n: usize) {
        let gate = &blk.gate_proj;
        let p = MatvecParams { in_dim: gate.in_dim, out_dim: gate.out_dim };
        enc.set_compute_pipeline_state(&self.engine.pipes.matvec_swiglu_batch);
        enc.set_buffer(0, Some(&gate.w), 0);
        enc.set_buffer(1, Some(&blk.up_proj.w), 0);
        enc.set_buffer(2, Some(&self.xn), 0);
        enc.set_buffer(3, Some(&self.gate), 0);
        enc.set_bytes(4, size_of::<MatvecParams>() as u64, &p as *const _ as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((gate.out_dim as u64).div_ceil(4), n as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }
}

impl Session for MetalSession<'_> {
    fn forward(&mut self, token: u32, pos: usize) -> crate::Result<Vec<f32>> {
        self.run(&[token], pos, 1)
    }

    /// Batch prefill: split the prompt into PREFILL_CHUNK-sized chunks. Later chunks
    /// automatically attend to earlier chunks' K,V through the cache (via pos0).
    fn prefill(&mut self, ids: &[u32]) -> crate::Result<Vec<f32>> {
        self.prefill_from(ids, 0)
    }

    /// Speculative verification: the whole batch in one submission, logits for every
    /// position. Falls back to the one-by-one loop past the logits buffer's capacity.
    fn forward_batch(&mut self, ids: &[u32], pos0: usize) -> crate::Result<Vec<Vec<f32>>> {
        if ids.len() > SPEC_MAX {
            return ids.iter().enumerate().map(|(i, &t)| self.forward(t, pos0 + i)).collect();
        }
        let vocab = self.engine.cfg.vocab_size;
        let flat = self.run(ids, pos0, ids.len())?;
        Ok(flat.chunks(vocab).map(|c| c.to_vec()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Not a correctness test: measures the host-side f32→f16 loop that write_kv runs
    /// on every ANE→Metal handoff, at a 6k-token Qwen-shaped handoff (kv_dim 128,
    /// 24 layers, K+V), without needing ANE graphs or a GPU. Run explicitly:
    ///   cargo test --release write_kv_conversion -- --ignored --nocapture
    #[test]
    #[ignore = "timing evidence, not a pass/fail check"]
    fn write_kv_conversion_microbench() {
        let (rows, kvd, layers) = (6144usize, 128usize, 24usize);
        let src: Vec<f32> = (0..rows * kvd).map(|i| (i % 251) as f32 * 0.01 - 1.0).collect();
        let mut dst = vec![0u16; rows * kvd];
        let t = std::time::Instant::now();
        for _ in 0..layers * 2 {
            // Mirrors the write_kv loop: scalar conversion through a raw pointer.
            let dp = std::hint::black_box(dst.as_mut_ptr());
            for (i, &x) in src.iter().enumerate() {
                unsafe { *dp.add(i) = f16::from_f32(x).to_bits() };
            }
        }
        std::hint::black_box(&dst);
        eprintln!(
            "write_kv-shaped handoff ({} rows × kv_dim {} × {} layers × K+V): {:?} total",
            rows,
            kvd,
            layers,
            t.elapsed()
        );
    }
}
