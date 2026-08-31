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
#[cfg(target_os = "macos")]
use crate::lowmem::{LowMemSource, SrcType};
use half::f16;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState,
    Device, FunctionConstantValues, MTLDataType, MTLResourceOptions, MTLSize,
};

/// Maximum tokens processed together during prefill (one command buffer per chunk).
/// Bigger chunks amortize weight reads better but grow the scratch buffers
/// (the attention scores buffer in particular).
pub(crate) const PREFILL_CHUNK: usize = 512;

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
struct RopeQkPrefillParams {
    head_dim: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    pos0: u32,
    theta: f32,
    n_rows: u32,
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
    matmul_th: ComputePipelineState,
    f32_to_f16: ComputePipelineState,
    bias_add: ComputePipelineState,
    matmul_h: ComputePipelineState,
    rmsnorm: ComputePipelineState,
    rmsnorm_h_inplace: ComputePipelineState,
    rmsnorm_hf: ComputePipelineState,
    silu_mul_hf: ComputePipelineState,
    rope: ComputePipelineState,
    rope_h: ComputePipelineState,
    rope_qk_prefill: ComputePipelineState,
    matmul_tb: ComputePipelineState,
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
pub(crate) const ATTN_SPLIT: usize = 128;
/// Threads per decode-attention threadgroup — must match DEC_TG in kernels.metal.
pub(crate) const DEC_TG: usize = 128;
/// Max q heads one GQA decode threadgroup covers — must match MAX_GQA_CHUNK in kernels.metal.
pub(crate) const MAX_GQA_CHUNK: usize = 8;
/// The head_dim the flash prefill attention kernel is specialized for (FA_HD in
/// kernels.metal); other head sizes take the scores-scratch fallback kernel.
pub(crate) const FLASH_HEAD_DIM: usize = 64;
/// Flash kernel dispatch geometry (FA_Q / FA_C / FA_THREADS in the shader).
pub(crate) const FLASH_Q: usize = 8;
pub(crate) const FLASH_C: usize = 64;
pub(crate) const FLASH_THREADS: usize = 128;
/// Token-rows per tensor-ops matmul tile (MM_TROWS in the shader).
const MM_TILE_ROWS: usize = 64;
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
    /// qwen3: per-head q/k RMSNorm (f16), pre-RoPE. None elsewhere.
    q_norm: Option<Buffer>,
    k_norm: Option<Buffer>,
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
    /// Opt-in sliding-window mode (--context-window): the geometry plus the
    /// attention pipelines re-specialized with lowmem's LM_* function
    /// constants. None = full causal, bit-for-bit the backend's only behavior
    /// before this mode existed.
    win: Option<WinState>,
    /// Quantized-GGUF execution (weights stay quant, dequantized on read).
    /// None = the dense f16 paths, untouched.
    quant: Option<QuantState>,
    /// qwen35 only: the recurrency map + state sizes sessions allocate from.
    /// None on every other architecture. Set by the engine constructor that
    /// lane C lands; sessions honor it today.
    pub(crate) q35_layout: Option<Qwen35Layout>,
    /// True geometry, derived from the checkpoint (qwen3 violates the
    /// hidden/n_heads identity, so cfg.head_dim()/kv_dim() are never read on
    /// hot paths — these are).
    dims: crate::lowmem::Dims,
}


/// lowmem's Dims constructor is module-private; same three lines, same reason
/// as the WindowCfg mirror (fields are pub, the formula is fixed).
fn dims_of(cfg: &ModelConfig, head_dim: Option<usize>) -> crate::lowmem::Dims {
    let hd = head_dim.unwrap_or_else(|| cfg.head_dim());
    crate::lowmem::Dims {
        hidden: cfg.hidden_size,
        head_dim: hd,
        q_dim: cfg.num_attention_heads * hd,
        kv_dim: cfg.num_key_value_heads * hd,
    }
}

/// The sliding-window add-on: geometry shared with lowmem, plus the three
/// attention pipelines built from the same kernel source with the LM_* window
/// constants set (everything else — GEMMs, rope, norms — is untouched).
pub(crate) struct WinState {
    pub cfg: crate::lowmem::WindowCfg,
    flash: ComputePipelineState,
    fallback: ComputePipelineState,
    dec_partial: ComputePipelineState,
}

/// One quantized weight matrix, resident as a span of the checkpoint's no-copy
/// mmap view (file pages stay reclaimable — the lowmem-proven pattern that lets
/// a 16.5 GB file run on a 32 GB box) or, for llama-arch q/k, a small staged
/// buffer holding the rows re-ordered out of llama.cpp's RoPE permute.
pub(crate) struct QuantLinear {
    w: Buffer,
    w_off: u64,
    bias: Buffer, // f16; the shared zero row when the projection is biasless
    in_dim: u32,
    out_dim: u32,
    sel: u32, // LM_W_QTYPE selector
}

/// One transformer block's quant weights + eagerly-resident f16 norms.
pub(crate) struct QuantBlock {
    input_layernorm: Buffer,
    post_attention_layernorm: Buffer,
    /// qwen3's per-head q/k RMSNorm weights (f16), pre-RoPE. None elsewhere.
    q_norm: Option<Buffer>,
    k_norm: Option<Buffer>,
    q_proj: QuantLinear,
    k_proj: QuantLinear,
    v_proj: QuantLinear,
    o_proj: QuantLinear,
    gate_proj: QuantLinear,
    up_proj: QuantLinear,
    down_proj: QuantLinear,
}

/// The pipelines one quant selector dispatches — built from the PRECISE
/// (fast-math-off) library so dequant math matches dequant_row_ref bit-for-bit
/// (gguf-kernels' rule; the fast library's fma contraction disagrees in the
/// last ulp on exactly the values a quantizer produces).
struct QuantPipes {
    matvec: ComputePipelineState,
    matvec_h: ComputePipelineState,
    matvec_acc: ComputePipelineState,
    matvec_swiglu: ComputePipelineState,
    matmul_pg: ComputePipelineState,
}

/// Everything the quant-GGUF execution path adds to the engine. None = the
/// backend is exactly what it was before this mode existed.
pub(crate) struct QuantState {
    source: LowMemSource,
    blocks: Vec<QuantBlock>,
    lm_head: QuantLinear,
    final_norm: Buffer,
    /// f32 embedding rows come off the CPU per token (dequant_row_ref) — no
    /// f16 table is ever materialized.
    embed_name: &'static str,
    pipes: std::collections::HashMap<u32, QuantPipes>,
    zero_bias: Buffer,
    n_params: usize,
}

impl QuantState {
    fn pipe(&self, sel: u32) -> &QuantPipes {
        self.pipes.get(&sel).expect("selector was built at engine construction")
    }
}

#[repr(C)]
struct MatmulPagedParams {
    in_dim: u32,
    out_dim: u32,
    n_rows: u32,
    y_stride: u32,
}

/// qwen35's recurrent session state: one conv + one delta buffer per LINEAR
/// trunk layer, f32, read-modify-written once per forward step. Sized by the
/// checkpoint's real dims (27B: conv 30,720 el + delta 786,432 el per layer,
/// 48 layers ≈ 150 MB per sequence — CONSTANT in context length). Attention
/// layers carry None; the MTP block is never in the map at all (layout length
/// is the TRUNK — the "17th attention layer" misconception must not reach an
/// allocator). One live state per sequence: no rollback in v1 — greedy CLI and
/// per-request serve sessions never rewind (documented limitation; llama.cpp's
/// snapshot ring is the reference if speculative decoding ever needs it).
pub(crate) struct Qwen35States {
    /// Index = trunk layer id. None = full-attention layer (KV lives there instead).
    pub layers: Vec<Option<Qwen35LayerState>>,
    pub conv_elems: usize,
    pub delta_elems: usize,
}

pub(crate) struct Qwen35LayerState {
    /// Rolling conv history, (d_conv−1)·C f32 — layout [channel][d_conv−1],
    /// oldest first (matches lowmem::qwen35_ref::conv_step).
    pub conv: Buffer,
    /// Delta-rule state [S, S, H_v] f32 — i the contraction index
    /// (s[i + j·S + h·S·S], matches lowmem::qwen35_ref::delta_decode_step).
    pub delta: Buffer,
}

/// What a qwen35-aware engine hands its sessions so they can allocate states:
/// the per-trunk-layer recurrency map plus the two per-layer element counts.
/// The C/D seam — changes go through the lead, never pairwise.
#[derive(Clone)]
pub(crate) struct Qwen35Layout {
    pub is_recurrent: Vec<bool>,
    pub conv_elems: usize,
    pub delta_elems: usize,
}

impl Qwen35Layout {
    /// MRoPE'S PRECONDITION, refused by name rather than silently assumed.
    ///
    /// qwen35 carries `rope.dimension_sections` and llama.cpp routes it through
    /// ggml_mrope_cache_init. For a TEXT batch that function is a no-op relative
    /// to plain rope — llama-batch.cpp:781-787 broadcasts ONE position into all
    /// four components, ops.cpp sets indep_sects = is_vision so the per-section
    /// resets never fire, and every one of the four thetas is multiplied by the
    /// same theta_scale each pair. All four therefore stay equal for the whole
    /// sequence, so whichever theta a sector selects is the plain-rope theta.
    /// `mrope_degenerates_to_rope_for_text_batches` pins that numerically.
    ///
    /// The REAL precondition is a property of the batch, not the checkpoint:
    /// this engine never constructs a vision batch. What metadata CAN tell us is
    /// whether a checkpoint belongs to the family that equivalence was verified
    /// on. A vision-capable variant reaches the same rope path and would be
    /// silently rotated with the wrong thetas — the one way this bites — so an
    /// unrecognised section layout is refused by name here instead.
    ///
    /// Deliberately NOT done: a sectioned rope kernel. Its best possible outcome
    /// is bit-identity with the rope we already ship, against a hand-derived
    /// index mapping that could be wrong (Detoro's ruling, task note e0449773).
    pub fn check_rope_sections(m: &crate::lowmem::gguf::Qwen35Meta) -> Result<(), String> {
        let s = &m.rope_sections;
        let sum: usize = s.iter().sum();
        if sum == 0 {
            return Err(format!(
                "qwen35: rope.dimension_sections is all zeros {s:?} — no rotary sections to \
                 map; refusing rather than rotating with an undefined layout"
            ));
        }
        if s[3] != 0 {
            return Err(format!(
                "qwen35: rope.dimension_sections {s:?} has a non-zero 4th (vision 'extra') \
                 section — this is a vision-capable variant. This engine only builds text \
                 batches, which is what makes MRoPE equivalent to plain rope, and it has no \
                 sectioned rope kernel. Refusing by name rather than rotating it as text."
            ));
        }
        Ok(())
    }

    /// The one meta→layout translation, so no caller re-derives sizes (where
    /// the 17-layer misconception would creep back in): the map is exactly the
    /// TRUNK's is_recurrent — the MTP block is not in the meta's map at all.
    pub fn from_meta(m: &crate::lowmem::gguf::Qwen35Meta) -> Self {
        Self {
            is_recurrent: m.is_recurrent.clone(),
            conv_elems: m.conv_state_elems,
            delta_elems: m.delta_state_elems,
        }
    }
}

impl Qwen35States {
    /// Zero-initialized states — zeroing is load-bearing: an empty conv
    /// history contributes silence and an empty delta state attends nothing.
    pub fn new(device: &Device, layout: &Qwen35Layout) -> Self {
        let layers = layout
            .is_recurrent
            .iter()
            .map(|&r| {
                r.then(|| Qwen35LayerState {
                    conv: f32_zero_buffer(device, layout.conv_elems),
                    delta: f32_zero_buffer(device, layout.delta_elems),
                })
            })
            .collect();
        Self { layers, conv_elems: layout.conv_elems, delta_elems: layout.delta_elems }
    }

    /// Back to the start-of-sequence state (a new prompt on a reused session).
    pub fn reset(&self) {
        for l in self.layers.iter().flatten() {
            unsafe {
                std::ptr::write_bytes(l.conv.contents() as *mut u8, 0, self.conv_elems * 4);
                std::ptr::write_bytes(l.delta.contents() as *mut u8, 0, self.delta_elems * 4);
            }
        }
    }

    /// The honest budget figure (what the lowmem plan and banners report).
    pub fn total_bytes(&self) -> usize {
        self.layers.iter().flatten().count() * (self.conv_elems + self.delta_elems) * 4
    }
}

/// Zero-filled f32 buffer (the f16 twin exists above; recurrent states are f32
/// because the delta rule accumulates into them across the whole sequence).
pub(crate) fn f32_zero_buffer(device: &Device, len: usize) -> Buffer {
    let buf = device.new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeShared);
    unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, len * 4) };
    buf
}

/// Destination spans for writing positions [pos0, pos0+n) into the windowed
/// store: (first chunk row, first slot, length) — at most the sink part plus
/// the ring part split once at the wrap (lowmem's write_spans, same shape).
fn win_write_spans(win: &crate::lowmem::WindowCfg, pos0: usize, n: usize) -> Vec<(usize, usize, usize)> {
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

/// lowmem's WindowCfg constructor is private to its module; this mirrors its
/// exact formula (sink_pad/ring 128-aligned, ring carries a full prefill chunk
/// of slack) — challenge 7c1a09cf tracks unifying the two at integration, and
/// a unit test pins this mirror to lowmem's own values.
pub(crate) fn window_cfg(w: usize, sink: usize) -> crate::Result<crate::lowmem::WindowCfg> {
    if w == 0 || sink > w {
        return Err(format!("invalid window config: window {w}, sink {sink}").into());
    }
    let sink_pad = sink.next_multiple_of(128);
    let ring = (w + PREFILL_CHUNK).next_multiple_of(128);
    Ok(crate::lowmem::WindowCfg { w, sink, sink_pad, ring, cap: sink_pad + ring })
}

// Apple documents MTLDevice / MTLCommandQueue / MTLBuffer / MTLComputePipelineState as
// thread-safe, but the `metal` crate's wrappers don't declare it, so we assert it here.
// (The genuinely non-thread-safe objects — command buffers and encoders — only live
// inside a session, which is always used from a single thread.)
unsafe impl Send for MetalEngine {}
unsafe impl Sync for MetalEngine {}

/// The full shader source with its injected compile-time constants — the one
/// place the #define preamble lives (the lowmem backend compiles its own
/// pipeline set from the same string).
pub(crate) fn shader_source(kvd: usize) -> String {
    format!(
        "#define MM_TROWS {MM_TILE_ROWS}\n#define FA_KVD {kvd}\n{}",
        include_str!("kernels.metal")
    )
}

/// Convert f32 → f16 and upload as a GPU buffer. StorageModeShared means unified
/// memory: on M-series chips the CPU and GPU see the same bytes — no PCIe copies
/// like on discrete GPUs.
pub(crate) fn f16_buffer(device: &Device, data: &[f32]) -> Buffer {
    let halves: Vec<u16> = data.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    device.new_buffer_with_data(
        halves.as_ptr() as *const _,
        (halves.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Width of one activation row that carries Q or attention output.
///
/// NOT hidden_size. q_dim = n_heads·head_dim is independent of hidden_size and
/// is LARGER on real configs — Qwen3-0.6B is hidden 1024 / q_dim 2048, and
/// qwen35's is hidden 5120 / q_dim 6144 — so a buffer sized by hidden_size is
/// overrun by the q/att writes, and the first symptom is nondeterminism rather
/// than a crash.
///
/// This exists as one function because the bug has now been found twice in two
/// places: once in session_scratch and once in the serve batcher, which was
/// missed when the first was fixed. Every path that allocates a Q- or
/// attention-width row goes through here so there is no third place.
pub(crate) fn attn_row_width(hidden: usize, q_dim: usize) -> usize {
    hidden.max(q_dim)
}

/// An f32 buffer holding `data`, for weights the qwen35 kernels read as f32.
///
/// The deltanet small tensors (ssm_conv1d, ssm_a, ssm_dt.bias, ssm_norm) are
/// stored F32 in the checkpoint and consumed as `device const float *` by the
/// kernels, which are gated BIT-FOR-BIT against an f32 CPU reference. Routing
/// them through the f16 path that norms use would narrow them and forfeit that
/// gate for no residency win — together they are a few hundred KB.
pub(crate) fn f32_buffer_from(device: &Device, data: &[f32]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

pub(crate) fn f32_buffer(device: &Device, len: usize) -> Buffer {
    device.new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeShared)
}

/// Zero-filled f16 buffer — the KV cache's dtype. The zeroing is load-bearing:
/// the flash prefill kernel reads K/V tiles straight from the cache, and a
/// tile's masked tail may touch rows no projection has written yet. Masked
/// scores never reach the output, but the P·V accumulate still multiplies the
/// raw values by zero — and 0 x NaN is NaN, so uninitialized rows must not be
/// able to hold NaN bit patterns.
pub(crate) fn f16_empty_buffer(device: &Device, len: usize) -> Buffer {
    let buf = device.new_buffer((len * 2) as u64, MTLResourceOptions::StorageModeShared);
    unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, len * 2) };
    buf
}

impl MetalEngine {
    /// Takes a loaded CPU-side Model and moves it onto the GPU (the Model is
    /// dropped after). `win`: Some((window, sink)) opts into sliding-window attention — the
    /// window pipelines must be specialized while the shader library is alive,
    /// which is why the choice happens at engine build, not per session.
    pub fn new_with_window(model: Model, win: Option<(usize, usize)>) -> crate::Result<Self> {
        let device = Device::system_default().ok_or("no Metal-capable GPU found")?;
        let queue = device.new_command_queue();

        // Kernels are compiled at runtime — edit kernels.metal and just cargo run again.
        let lib = device
            .new_library_with_source(&shader_source(model.kv_dim), &CompileOptions::new())
            .map_err(|e| format!("failed to compile kernels.metal: {e}"))?;
        let dims = dims_of(&model.cfg, Some(model.head_dim));
        let pipes = Self::build_pipelines(&device, &lib, &model.cfg)?;
        let win_state = Self::build_win_state(&device, &lib, &model.cfg, dims.kv_dim, win)?;
        Self::finish_dense(device, queue, model, pipes, win_state, dims)
    }

    /// The full dense pipeline set — code moved verbatim from the constructor
    /// so the quant path can build the identical set from its own library.
    fn build_pipelines(
        device: &Device,
        lib: &metal::Library,
        cfg: &ModelConfig,
    ) -> crate::Result<Pipelines> {
        let pipe = |name: &str| -> crate::Result<ComputePipelineState> {
            let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("kernel {name}: {e}").into())
        };
        // Kernels that REFERENCE function constants (the lowmem LM_* window set)
        // must be built through the specialization API even with no constant
        // set: the empty set resolves is_function_constant_defined() to false
        // and compiles exactly the unwindowed code this backend always ran.
        let default_spec = |name: &str| -> crate::Result<ComputePipelineState> {
            let f = lib
                .get_function(name, Some(FunctionConstantValues::new()))
                .map_err(|e| format!("kernel {name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("kernel {name}: {e}").into())
        };
        // The GQA decode kernels are specialized per model: function constant 0
        // (GQA_CHUNK) is the q-head group width one threadgroup covers, fixed here
        // so the per-head loops in the kernel unroll flat.
        let gqa_chunk =
            (cfg.num_attention_heads / cfg.num_key_value_heads).min(MAX_GQA_CHUNK) as u32;
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
            // The matvec family references the lowmem LM_W_BF16 function
            // constant through dot_wx — unspecialized builds are refused, so
            // these take the empty specialization (identical code).
            matvec: default_spec("matvec")?,
            matvec_acc: default_spec("matvec_acc")?,
            matvec_swiglu: default_spec("matvec_swiglu")?,
            matvec_qkv: default_spec("matvec_qkv")?,
            matvec_h: default_spec("matvec_h")?,
            matmul: pipe("matmul")?,
            matmul_t: pipe("matmul_t")?,
            matmul_th: pipe("matmul_th")?,
            f32_to_f16: pipe("f32_to_f16")?,
            bias_add: pipe("bias_add")?,
            matmul_h: pipe("matmul_h")?,
            rmsnorm: pipe("rmsnorm")?,
            rmsnorm_h_inplace: pipe("rmsnorm_h_inplace")?,
            rmsnorm_hf: pipe("rmsnorm_hf")?,
            silu_mul_hf: pipe("silu_mul_hf")?,
            rope: pipe("rope")?,
            rope_h: pipe("rope_h")?,
            rope_qk_prefill: pipe("rope_qk_prefill")?,
            matmul_tb: pipe("matmul_tb")?,
            rope_qk_decode: pipe("rope_qk_decode")?,
            attention: default_spec("attention")?,
            attention_prefill_flash: default_spec("attention_prefill_flash")?,
            attention_decode_partial: gqa_pipe("attention_decode_partial")?,
            attention_decode_reduce: pipe("attention_decode_reduce")?,
            silu_mul: pipe("silu_mul")?,
            add_inplace: pipe("add_inplace")?,
            matvec_qkv_batch: default_spec("matvec_qkv_batch")?,
            matvec_acc_batch: default_spec("matvec_acc_batch")?,
            matvec_swiglu_batch: default_spec("matvec_swiglu_batch")?,
            rope_qk_batch: pipe("rope_qk_batch")?,
            attention_decode_partial_batch: gqa_pipe("attention_decode_partial_batch")?,
            attention_decode_reduce_batch: pipe("attention_decode_reduce_batch")?,
        };
        Ok(pipes)
    }

    /// Window mode: re-specialize the three attention kernels with the LM_*
    /// constants (indices 20-23, exactly as lowmem builds them; GQA_CHUNK
    /// rides at 0 for the decode kernel).
    fn build_win_state(
        device: &Device,
        lib: &metal::Library,
        cfg: &ModelConfig,
        kvd: usize,
        win: Option<(usize, usize)>,
    ) -> crate::Result<Option<WinState>> {
        let gqa_chunk =
            (cfg.num_attention_heads / cfg.num_key_value_heads).min(MAX_GQA_CHUNK) as u32;
        let win_state = match win {
            None => None,
            Some((w, sink)) => {
                let wc = window_cfg(w, sink)?;
                let win_pipe = |name: &str, gqa: bool| -> crate::Result<ComputePipelineState> {
                    let consts = FunctionConstantValues::new();
                    if gqa {
                        consts.set_constant_value_at_index(
                            &gqa_chunk as *const u32 as *const _,
                            MTLDataType::UInt,
                            0,
                        );
                    }
                    for (v, idx) in [
                        (wc.sink as u32, 20u64),
                        (wc.sink_pad as u32, 21),
                        (wc.ring as u32, 22),
                        (wc.w as u32, 23),
                    ] {
                        consts.set_constant_value_at_index(
                            &v as *const u32 as *const _,
                            MTLDataType::UInt,
                            idx,
                        );
                    }
                    let f = lib
                        .get_function(name, Some(consts))
                        .map_err(|e| format!("kernel {name}: {e}"))?;
                    device
                        .new_compute_pipeline_state_with_function(&f)
                        .map_err(|e| format!("kernel {name}: {e}").into())
                };
                let kv_mb = cfg.num_hidden_layers * wc.cap * kvd * 2 * 2;
                eprintln!(
                    "Metal window mode: window {} (+{} sink) — KV is a ring of {} slots/layer, {:.0} MB total, flat in context length",
                    wc.w,
                    wc.sink,
                    wc.cap,
                    kv_mb as f64 / 1e6
                );
                Some(WinState {
                    cfg: wc,
                    flash: win_pipe("attention_prefill_flash", false)?,
                    fallback: win_pipe("attention", false)?,
                    dec_partial: win_pipe("attention_decode_partial", true)?,
                })
            }
        };
        Ok(win_state)
    }

    /// The tail of dense construction (weights to f16 buffers) — moved
    /// verbatim; the quant path never enters here.
    fn finish_dense(
        device: Device,
        queue: CommandQueue,
        model: Model,
        pipes: Pipelines,
        win_state: Option<WinState>,
        dims: crate::lowmem::Dims,
    ) -> crate::Result<Self> {
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
                q_norm: b.q_norm.as_ref().map(|w| f16_buffer(&device, w)),
                k_norm: b.k_norm.as_ref().map(|w| f16_buffer(&device, w)),
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
            win: win_state,
            quant: None,
            q35_layout: None,
            dims,
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
        conc: bool,
    ) {
        self.enc_linear_with(&self.pipes.matvec, &self.pipes.matmul, enc, l, x, x_off, y, y_off, n_rows, xh, convert, conc);
    }

    /// enc_linear writing f16 — the k/v projections, whose output IS the KV cache.
    /// With a staged half input (the rmsnorm half copy) and multiple rows, the
    /// tensor-ops kernel takes it; single rows and the batched decode path keep
    /// the matvec/simdgroup kernels.
    #[allow(clippy::too_many_arguments)]
    fn enc_linear_kv(
        &self,
        enc: &ComputeCommandEncoderRef,
        l: &GpuLinear,
        x: &Buffer,
        y: &Buffer,
        y_off: u64,
        n_rows: usize,
        xh: Option<&Buffer>,
    ) {
        let (Some(xh), true) = (xh, n_rows > 1) else {
            self.enc_linear_with(&self.pipes.matvec_h, &self.pipes.matmul_h, enc, l, x, 0, y, y_off, n_rows, None, false, false);
            return;
        };
        let p = MatmulParams { in_dim: l.in_dim, out_dim: l.out_dim, n_rows: n_rows as u32 };
        enc.set_compute_pipeline_state(&self.pipes.matmul_th);
        enc.set_buffer(0, Some(&l.w), 0);
        enc.set_buffer(1, Some(xh), 0);
        enc.set_buffer(2, Some(y), y_off);
        enc.set_buffer(3, Some(&l.bias), 0);
        enc.set_bytes(4, size_of::<MatmulParams>() as u64, &p as *const _ as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((l.out_dim as u64).div_ceil(64), (n_rows as u64).div_ceil(MM_TILE_ROWS as u64), 1),
            MTLSize::new(128, 1, 1),
        );
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
        conc: bool,
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
                if conc {
                    enc.memory_barrier_with_resources(&[xh]); // matmul_t reads the fresh half copy
                }
            }

            // Biased layers take matmul_tb (bias folded into the store epilogue,
            // value-identical to matmul_t + bias_add); the rest stay on matmul_t.
            if l.has_bias {
                enc.set_compute_pipeline_state(&self.pipes.matmul_tb);
                enc.set_buffer(0, Some(&l.w), 0);
                enc.set_buffer(1, Some(xh), 0);
                enc.set_buffer(2, Some(y), y_off);
                enc.set_buffer(3, Some(&l.bias), 0);
                enc.set_bytes(4, size_of::<MatmulParams>() as u64, &p as *const _ as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new((l.out_dim as u64).div_ceil(64), (n_rows as u64).div_ceil(MM_TILE_ROWS as u64), 1),
                    MTLSize::new(128, 1, 1),
                );
                return;
            }
            enc.set_compute_pipeline_state(&self.pipes.matmul_t);
            enc.set_buffer(0, Some(&l.w), 0);
            enc.set_buffer(1, Some(xh), 0);
            enc.set_buffer(2, Some(y), y_off);
            enc.set_bytes(3, size_of::<MatmulParams>() as u64, &p as *const _ as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((l.out_dim as u64).div_ceil(64), (n_rows as u64).div_ceil(MM_TILE_ROWS as u64), 1),
                MTLSize::new(128, 1, 1),
            );

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

    /// qwen3's per-head q/k norm, f32 rows in place (rmsnorm with x == y and a
    /// caller-chosen row width — one "row" here is one HEAD, dim = head_dim).
    fn enc_rmsnorm_dim(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        weight: &Buffer,
        n_rows: usize,
        dim: usize,
    ) {
        let p = NormParams { dim: dim as u32, eps: self.cfg.rms_norm_eps };
        enc.set_compute_pipeline_state(&self.pipes.rmsnorm);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_buffer(2, Some(x), 0); // in place
        enc.set_bytes(3, size_of::<NormParams>() as u64, &p as *const _ as *const _);
        enc.dispatch_thread_groups(MTLSize::new(n_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Same, for f16 rows already sitting in the KV cache (k after a fused
    /// decode projection) — the rmsnorm_h_inplace kernel gguf-kernels shipped.
    fn enc_rmsnorm_h_inplace(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_off: u64,
        weight: &Buffer,
        n_rows: usize,
        dim: usize,
    ) {
        let p = NormParams { dim: dim as u32, eps: self.cfg.rms_norm_eps };
        enc.set_compute_pipeline_state(&self.pipes.rmsnorm_h_inplace);
        enc.set_buffer(0, Some(x), x_off);
        enc.set_buffer(1, Some(weight), 0);
        enc.set_bytes(2, size_of::<NormParams>() as u64, &p as *const _ as *const _);
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
        let hd = self.dims.head_dim;
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
            head_dim: self.dims.head_dim as u32,
            n_heads: self.cfg.num_attention_heads as u32,
            n_kv_heads: self.cfg.num_key_value_heads as u32,
            pos0: pos0 as u32,
            max_seq: max_seq as u32,
            n_rows: n_rows as u32,
        };
        if self.dims.head_dim == FLASH_HEAD_DIM {
            // Flash path: no scores scratch, one threadgroup per (head, 16-row tile).
            enc.set_compute_pipeline_state(match &self.win {
                Some(w) => &w.flash,
                None => &self.pipes.attention_prefill_flash,
            });
            enc.set_buffer(0, Some(q), 0);
            enc.set_buffer(1, Some(k_cache), cache_off);
            enc.set_buffer(2, Some(v_cache), cache_off);
            enc.set_buffer(3, Some(out), 0);
            enc.set_bytes(4, size_of::<AttnParams>() as u64, &p as *const _ as *const _);
            enc.set_buffer(5, Some(out_h), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(
                    self.cfg.num_attention_heads as u64,
                    n_rows.div_ceil(FLASH_Q) as u64,
                    1,
                ),
                MTLSize::new(FLASH_THREADS as u64, 1, 1),
            );
            return;
        }
        enc.set_compute_pipeline_state(match &self.win {
            Some(w) => &w.fallback,
            None => &self.pipes.attention,
        });
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
        enc.set_compute_pipeline_state(&self.pipes.f32_to_f16);
        enc.set_buffer(0, Some(src), src_off);
        enc.set_buffer(1, Some(dst), dst_off);
        enc.set_bytes(2, 4, &d as *const u32 as *const _);
        dispatch_grid(enc, dim);
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
    // qwen35 NOTE — do not "fix" this by adding MRoPE sectioning. qwen35 ships
    // rope.dimension_sections and llama.cpp routes it through
    // ggml_mrope_cache_init, but for a TEXT batch that is provably the same
    // rope as this one: one position is broadcast into all four components, so
    // the four thetas start equal and are all scaled by theta_scale every pair,
    // and the sector select can never pick a different value. Pinned by
    // qwen35_kernel_oracle::mrope_degenerates_to_rope_for_text_batches (with a
    // vision-path negative control), and vision-capable checkpoints are refused
    // by name in Qwen35Layout::check_rope_sections. A sectioned kernel's best
    // possible outcome is bit-identity with this one.
    fn enc_rope_qk(
        &self,
        enc: &ComputeCommandEncoderRef,
        q: &Buffer,
        k_cache: &Buffer,
        kv_byte_off: u64,
        pos: usize,
    ) {
        let hd = self.dims.head_dim;
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
        // Window mode: every split of the bounded store is dispatched (the
        // constant split count is what makes decode cost flat); the LM_*
        // kernel masks validity per slot. Full causal keeps the growing count.
        let n_splits = match &self.win {
            Some(w) => w.cfg.cap / ATTN_SPLIT,
            None => (pos + 1).div_ceil(ATTN_SPLIT),
        };
        let p = AttnDecParams {
            head_dim: self.dims.head_dim as u32,
            n_heads: self.cfg.num_attention_heads as u32,
            n_kv_heads: self.cfg.num_key_value_heads as u32,
            pos: pos as u32,
            n_splits: n_splits as u32,
        };
        let heads = self.cfg.num_attention_heads as u64;
        let (grid_x, tg_mem) = self.gqa_decode_dims();
        enc.set_compute_pipeline_state(match &self.win {
            Some(w) => &w.dec_partial,
            None => &self.pipes.attention_decode_partial,
        });
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
            MTLSize::new(self.dims.head_dim as u64, 1, 1),
        );
    }

    fn gqa_decode_dims(&self) -> (u64, [u64; 4]) {
        gqa_decode_dims(&self.cfg, self.dims.head_dim)
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

/// Grid width and threadgroup memory sizes for the GQA-aware decode partial
/// kernels: one threadgroup per (kv head × group chunk, window), covering up to
/// MAX_GQA_CHUNK q heads of one kv head's group. Free function because the
/// lowmem backend dispatches the same kernels from its own engine.
pub(crate) fn gqa_decode_dims(cfg: &ModelConfig, head_dim: usize) -> (u64, [u64; 4]) {
    let group = cfg.num_attention_heads / cfg.num_key_value_heads;
    let chunk = group.min(MAX_GQA_CHUNK);
    let grid_x = (cfg.num_key_value_heads * group.div_ceil(chunk)) as u64;
    // Sizes must mirror the kernel's q_s / es / acc_red / red layouts, padded to
    // Metal's 16-byte threadgroup-allocation granularity.
    let f32s = |n: usize| (n * 4).next_multiple_of(16) as u64;
    (
        grid_x,
        [
            f32s(chunk * head_dim),
            f32s(chunk * ATTN_SPLIT),
            f32s(DEC_TG * (chunk | 1)),
            f32s(chunk * (DEC_TG / 32) + chunk),
        ],
    )
}

/// One-thread-per-element dispatch — kernels guard the tail with `if (gid < dim)`.
pub(crate) fn dispatch_grid(enc: &ComputeCommandEncoderRef, n: usize) {
    let tg = 256u64;
    enc.dispatch_thread_groups(MTLSize::new((n as u64).div_ceil(tg), 1, 1), MTLSize::new(tg, 1, 1));
}

/// One-simdgroup-per-output-row dispatch, 4 rows per threadgroup — shared by every
/// matvec-family kernel (they guard the tail with `if (row >= out_dim)`).
pub(crate) fn dispatch_simdgroup_rows(enc: &ComputeCommandEncoderRef, rows: u32) {
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
        if self.win.is_some() || self.quant.is_some() {
            // Window mode (D4) and quant-GGUF v1 both serve through per-request
            // sessions; the batcher pool keeps the dense full-causal layout.
            return None;
        }
        self.make_batcher(n_slots, max_seq)
            .map(|b| Box::new(b) as Box<dyn crate::engine::Batcher>)
    }
}

impl MetalEngine {
    /// Build a session as a concrete type — the ane backend needs write_kv/prefill_from.
    /// Build the engine straight from a quantized GGUF: weights stay in their
    /// on-disk encoding behind one no-copy view, norms/biases materialize as
    /// the usual f16 buffers, and the qtype pipelines come from the precise
    /// library. No f32 expansion happens anywhere on this path.
    pub fn new_gguf_quant(
        path: &std::path::Path,
        cfg: ModelConfig,
        win: Option<(usize, usize)>,
    ) -> crate::Result<Self> {
        use std::collections::HashMap;
        let device = Device::system_default().ok_or("no Metal-capable GPU found")?;
        let queue = device.new_command_queue();
        let mut source = LowMemSource::open(path)?;
        source.make_gpu_views(&device);
        let dims = dims_of(&cfg, source.head_dim());

        let shader = shader_source(dims.kv_dim);
        let lib = device
            .new_library_with_source(&shader, &CompileOptions::new())
            .map_err(|e| format!("failed to compile kernels.metal: {e}"))?;
        // (identical construction to new_with_window below — the dense
        // pipelines keep the fast library and their exact existing code)
        let dense = Self::build_pipelines(&device, &lib, &cfg)?;
        let win_state = Self::build_win_state(&device, &lib, &cfg, dims.kv_dim, win)?;

        // Quant pipelines: precise fast-math-off library, one set per selector
        // present in the file (gguf-kernels doctrine — see lowmem's twin).
        let quant_types = source.quant_types();
        let precise = CompileOptions::new();
        precise.set_fast_math_enabled(false);
        let plib = device
            .new_library_with_source(&shader, &precise)
            .map_err(|e| format!("failed to compile kernels.metal (precise): {e}"))?;
        let qpipe = |name: &str, sel: u32| -> crate::Result<ComputePipelineState> {
            let consts = FunctionConstantValues::new();
            consts.set_constant_value_at_index(&sel as *const u32 as *const _, MTLDataType::UInt, 25);
            let f = plib.get_function(name, Some(consts)).map_err(|e| format!("kernel {name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("kernel {name}: {e}").into())
        };
        let mut pipes: HashMap<u32, QuantPipes> = HashMap::new();
        let mut f16_sel_needed = false;
        for ty in &quant_types {
            let sel = SrcType::Quant(*ty).qtype();
            if sel == u32::MAX {
                // NON-EXHAUSTIVE on purpose: the lowbit-quants lane grows the
                // seam in parallel; a type lands here only when wired.
                return Err(format!(
                    "GGUF type {ty:?} has no metal quant pipeline yet — re-download as Q4_K_M or Q8_0"
                )
                .into());
            }
            if !pipes.contains_key(&sel) {
                pipes.insert(
                    sel,
                    QuantPipes {
                        matvec: qpipe("matvec", sel)?,
                        matvec_h: qpipe("matvec_h", sel)?,
                        matvec_acc: qpipe("matvec_acc", sel)?,
                        matvec_swiglu: qpipe("matvec_swiglu", sel)?,
                        matmul_pg: qpipe("matmul_pg", sel)?,
                    },
                );
            }
            let _ = &mut f16_sel_needed;
        }
        // F16/F32 tensors in a mixed file (fp16 GGUFs, some heads) run under
        // selector 0 — same kernels, precise library for uniformity.
        if !pipes.contains_key(&0) {
            pipes.insert(
                0,
                QuantPipes {
                    matvec: qpipe("matvec", 0)?,
                    matvec_h: qpipe("matvec_h", 0)?,
                    matvec_acc: qpipe("matvec_acc", 0)?,
                    matvec_swiglu: qpipe("matvec_swiglu", 0)?,
                    matmul_pg: qpipe("matmul_pg", 0)?,
                },
            );
        }

        let zero_bias = f16_empty_buffer(&device, cfg.hidden_size.max(cfg.intermediate_size).max(cfg.vocab_size));
        let norm_buf = |name: &str| -> crate::Result<Buffer> {
            Ok(f16_buffer(&device, &source.read_f32(name)?))
        };
        let qlin = |source: &LowMemSource, device: &Device, zero_bias: &Buffer, name: &str, in_dim: usize, out_dim: usize| -> crate::Result<QuantLinear> {
            let ty = source.src_type(&format!("{name}.weight"))?;
            let sel = match ty {
                SrcType::Quant(t) => {
                    let sel = SrcType::Quant(t).qtype();
                    if sel == u32::MAX {
                        return Err(format!("{name}: {t:?} has no metal quant pipeline yet").into());
                    }
                    sel
                }
                // F16/F32 rows run the selector-0 kernels; the f32 case never
                // occurs for 2-D weights in practice (norms are the f32 ones).
                SrcType::F16 | SrcType::F32 => 0,
                SrcType::BF16 => return Err(format!("{name}: bf16 inside a GGUF is unsupported").into()),
            };
            let shape = source.shape(&format!("{name}.weight"))?;
            if shape != vec![out_dim, in_dim] {
                return Err(format!("{name}: shape {shape:?}, expected [{out_dim}, {in_dim}]").into());
            }
            let bias = match source.has(&format!("{name}.bias")) {
                true => f16_buffer(device, &source.read_f32(&format!("{name}.bias"))?),
                false => zero_bias.clone(),
            };
            let wname = format!("{name}.weight");
            let (w, w_off) = match source.gpu_span(&wname, 0, out_dim)? {
                Some((view, off)) => (view.clone(), off as u64),
                None => {
                    // llama-arch q/k: rows sit RoPE-permuted in the file, so a
                    // single span cannot be handed to the GPU — stage the rows
                    // re-ordered once (small: q/k only), still in quant bytes.
                    let hd = source
                        .unpermute_head_dim(&wname)
                        .ok_or_else(|| format!("{name}: no GPU span and no permute reason"))?;
                    let rb = ty.row_bytes(in_dim);
                    let mut staged = vec![0u8; out_dim * rb];
                    for r in 0..out_dim {
                        let (h, j) = (r / hd, r % hd);
                        let (a, d) = (j / (hd / 2), j % (hd / 2));
                        let src_row = h * hd + d * 2 + a;
                        let row = source.read_rows(&wname, src_row, src_row + 1)?;
                        staged[r * rb..(r + 1) * rb].copy_from_slice(row);
                    }
                    let buf = device.new_buffer_with_data(
                        staged.as_ptr() as *const _,
                        staged.len() as u64,
                        MTLResourceOptions::StorageModeShared,
                    );
                    (buf, 0)
                }
            };
            Ok(QuantLinear { w, w_off, bias, in_dim: in_dim as u32, out_dim: out_dim as u32, sel })
        };

        let (h, kvd, inter) = (cfg.hidden_size, dims.kv_dim, cfg.intermediate_size);
        let q_dim = dims.q_dim;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            blocks.push(QuantBlock {
                input_layernorm: norm_buf(&format!("{p}.input_layernorm.weight"))?,
                post_attention_layernorm: norm_buf(&format!("{p}.post_attention_layernorm.weight"))?,
                q_proj: qlin(&source, &device, &zero_bias, &format!("{p}.self_attn.q_proj"), h, q_dim)?,
                k_proj: qlin(&source, &device, &zero_bias, &format!("{p}.self_attn.k_proj"), h, kvd)?,
                v_proj: qlin(&source, &device, &zero_bias, &format!("{p}.self_attn.v_proj"), h, kvd)?,
                o_proj: qlin(&source, &device, &zero_bias, &format!("{p}.self_attn.o_proj"), q_dim, h)?,
                q_norm: match source.has(&format!("{p}.self_attn.q_norm.weight")) {
                    true => Some(f16_buffer(&device, &source.read_f32(&format!("{p}.self_attn.q_norm.weight"))?)),
                    false => None,
                },
                k_norm: match source.has(&format!("{p}.self_attn.k_norm.weight")) {
                    true => Some(f16_buffer(&device, &source.read_f32(&format!("{p}.self_attn.k_norm.weight"))?)),
                    false => None,
                },
                gate_proj: qlin(&source, &device, &zero_bias, &format!("{p}.mlp.gate_proj"), h, inter)?,
                up_proj: qlin(&source, &device, &zero_bias, &format!("{p}.mlp.up_proj"), h, inter)?,
                down_proj: qlin(&source, &device, &zero_bias, &format!("{p}.mlp.down_proj"), inter, h)?,
            });
        }
        let final_norm = norm_buf("model.norm.weight")?;
        let lm_head_name =
            if source.has("lm_head.weight") { "lm_head" } else { "model.embed_tokens" };
        let lm_head = qlin(&source, &device, &zero_bias, lm_head_name, h, cfg.vocab_size)?;
        let n_params = source.n_params();

        eprintln!(
            "Metal quant: {} stays {:?} on the GPU — dequantized on read, no f32 expansion",
            path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            quant_types,
        );

        Ok(Self {
            pipes: dense,
            embed_tokens: f16_empty_buffer(&device, 1), // unused on the quant path (CPU gather)
            blocks: Vec::new(),
            norm: f16_empty_buffer(&device, 1),
            lm_head: GpuLinear {
                w: f16_empty_buffer(&device, 1),
                bias: zero_bias.clone(),
                has_bias: false,
                in_dim: 0,
                out_dim: 0,
            },
            cfg,
            queue,
            device,
            win: win_state,
            q35_layout: None, // lane C's constructor sets this for qwen35 files
            dims,
            quant: Some(QuantState {
                source,
                blocks,
                lm_head,
                final_norm,
                embed_name: "model.embed_tokens.weight",
                pipes,
                zero_bias,
                n_params,
            }),
        })
    }

    pub(crate) fn raw_session(&self, max_seq: usize) -> MetalSession<'_> {
        let cfg = &self.cfg;
        let d = &self.device;
        // Window mode: KV is a ring of cap slots per layer — O(window), not
        // O(context) — exactly lowmem's store layout so the LM_* kernels read
        // it unchanged.
        let kv_slots = match &self.win {
            Some(w) => w.cfg.cap,
            None => max_seq + FLASH_C,
        };
        let caches = (0..cfg.num_hidden_layers)
            .map(|_| f16_empty_buffer(d, kv_slots * self.dims.kv_dim))
            .collect::<Vec<_>>();
        let v_caches = (0..cfg.num_hidden_layers)
            .map(|_| f16_empty_buffer(d, kv_slots * self.dims.kv_dim))
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
        // Attention scratch is strided by the KV extent: max_seq full-causal,
        // the ring cap under a window.
        let kv_extent = match &self.win {
            Some(w) => w.cfg.cap,
            None => max_seq,
        };
        SessionScratch {
            ids: d.new_buffer((chunk * 4) as u64, MTLResourceOptions::StorageModeShared),
            x: f32_buffer(d, chunk * cfg.hidden_size),
            xn: f32_buffer(d, chunk * cfg.hidden_size),
            // q and attention rows are q_dim wide — qwen3's q_dim (heads x 128)
            // is 2x hidden, so sizing these by hidden overflows into whatever
            // the allocator placed next (the first symptom is nondeterminism).
            q: f32_buffer(d, chunk * attn_row_width(cfg.hidden_size, self.dims.q_dim)),
            att: f32_buffer(d, chunk * attn_row_width(cfg.hidden_size, self.dims.q_dim)),
            xb: f32_buffer(d, chunk * cfg.hidden_size),
            gate: f32_buffer(d, chunk * cfg.intermediate_size),
            up: f32_buffer(d, chunk * cfg.intermediate_size),
            logits: f32_buffer(d, SPEC_MAX * cfg.vocab_size),
            // The flash prefill path never touches scores; keep a 1-float stub so
            // the fallback binding stays valid without the (huge) allocation.
            scores: if self.dims.head_dim == FLASH_HEAD_DIM {
                f32_buffer(d, 1)
            } else {
                f32_buffer(d, chunk * cfg.num_attention_heads * kv_extent)
            },
            partials: f32_buffer(
                d,
                cfg.num_attention_heads * kv_extent.div_ceil(ATTN_SPLIT) * (self.dims.head_dim + 2),
            ),
            xh: f16_empty_buffer(
                d,
                chunk * cfg.hidden_size.max(cfg.intermediate_size).max(self.dims.q_dim),
            ),
            // Window mode stages fresh K/V rows in f32 here, then scatters them
            // into the ring as f16 spans (lowmem's exact write path). One float
            // of stub keeps the binding cheap when the mode is off.
            kvs: if self.win.is_some() || self.quant.is_some() {
                f32_buffer(d, 2 * chunk * self.dims.kv_dim)
            } else {
                f32_buffer(d, 1)
            },
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
        let SessionScratch { ids, x, xn, q, att, xb, gate, up, logits, scores, partials, xh, kvs } =
            scratch;
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
            kvs,
            q35: self.q35_layout.as_ref().map(|l| Qwen35States::new(&self.device, l)),
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
    /// Window mode's fresh-K/V f32 staging (1-float stub otherwise).
    kvs: Buffer,
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
    kvs: Buffer,      // window mode: fresh K/V f32 staging before the ring scatter
    /// qwen35: the per-linear-layer recurrent states (None elsewhere).
    q35: Option<Qwen35States>,
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
        if self.engine.quant.is_some() {
            return self.run_from_quant(src, n, pos0, layer0, logits_rows);
        }
        let e = self.engine;
        let cfg = &e.cfg;
        let h = cfg.hidden_size;
        // This chunk's first (f16) cache row. Under a window the row is the
        // RING SLOT of pos0 — for decode (n == 1) that is the whole story;
        // prefill writes go through the staging + span-scatter path instead.
        let kv_slot0 = match &e.win {
            Some(w) => w.cfg.slot_of(pos0),
            None => pos0,
        };
        let kv_byte_off = self.kv_base + (kv_slot0 * e.dims.kv_dim * 2) as u64;

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
        let fused_decode = n == 1 && e.dims.head_dim <= DEC_TG && e.dims.head_dim.is_multiple_of(4);
        // Decode keeps the serial encoder (every dispatch depends on the previous
        // one anyway). Prefill uses a concurrent encoder with explicit barriers so
        // independent dispatches — q/k/v projections, the rope pair, gate/up — can
        // overlap; each barrier names exactly the buffers the next stage reads.
        let enc = if fused_decode {
            cb.new_compute_command_encoder()
        } else {
            cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent)
        };
        let conc = !fused_decode;
        macro_rules! bar {
            ($($b:expr),+) => { if conc { enc.memory_barrier_with_resources(&[$($b),+]) } };
        }

        if matches!(src, Source::Ids(_)) {
            e.enc_embed(enc, &self.ids, &self.x, n);
            bar!(&self.x);
        }
        for (l, blk) in e.blocks.iter().enumerate().skip(layer0) {
            if fused_decode {
                // Decode path: fused kernels — 9 dispatches per layer instead of 15.
                // Same math as the prefill path below, with qkv / swiglu / residual
                // adds folded into single launches and flash-decoding attention.
                e.enc_rmsnorm(enc, &self.x, &blk.input_layernorm, &self.xn, 1);
                let kv_off_elems = (self.kv_base / 2) as usize + kv_slot0 * e.dims.kv_dim;
                e.enc_qkv(enc, blk, &self.xn, &self.q, &self.k_cache[l], &self.v_cache[l], kv_off_elems);
                if let (Some(qn), Some(kn)) = (&blk.q_norm, &blk.k_norm) {
                    e.enc_rmsnorm_dim(enc, &self.q, qn, cfg.num_attention_heads, e.dims.head_dim);
                    e.enc_rmsnorm_h_inplace(enc, &self.k_cache[l], kv_byte_off, kn, cfg.num_key_value_heads, e.dims.head_dim);
                }
                e.enc_rope_qk(enc, &self.q, &self.k_cache[l], kv_byte_off, pos0);
                e.enc_attention_decode(enc, &self.q, &self.k_cache[l], &self.v_cache[l], self.kv_base, &self.partials, &self.att, pos0);
                e.enc_matvec_acc(enc, &blk.o_proj, &self.att, &self.x);
                e.enc_rmsnorm(enc, &self.x, &blk.post_attention_layernorm, &self.xn, 1);
                e.enc_swiglu(enc, &blk.gate_proj, &blk.up_proj, &self.xn, &self.gate);
                e.enc_matvec_acc(enc, &blk.down_proj, &self.gate, &self.x);
                continue;
            }

            // Prefill path (and the rare head_dim > DEC_TG decode): tiled matmuls.
            // Attention half. Barrier-free groups: q/k/v projections (disjoint
            // outputs, shared read-only input), the two ropes, gate/up.
            e.enc_rmsnorm_hf(enc, &self.x, &blk.input_layernorm, &self.xn, &self.xh, n);
            bar!(&self.xn, &self.xh);
            e.enc_linear(enc, &blk.q_proj, &self.xn, 0, &self.q, 0, n, Some(&self.xh), false, conc);
            if let Some(w) = &e.win {
                // Window mode (lowmem's exact write path): project fresh K/V
                // into the f32 staging, rope q standalone, then per destination
                // span convert the rows into the ring as f16 and rotate K there
                // by its TRUE positions — storage is slots, RoPE is absolute.
                let kvd = e.dims.kv_dim;
                let hd = e.dims.head_dim;
                let v_base = self.kvs.length() / 2; // bytes: V's half of the staging
                e.enc_linear(enc, &blk.k_proj, &self.xn, 0, &self.kvs, 0, n, Some(&self.xh), false, conc);
                e.enc_linear(enc, &blk.v_proj, &self.xn, 0, &self.kvs, v_base, n, Some(&self.xh), false, conc);
                bar!(&self.q, &self.kvs);
                if let (Some(qn), Some(kn)) = (&blk.q_norm, &blk.k_norm) {
                    e.enc_rmsnorm_dim(enc, &self.q, qn, n * cfg.num_attention_heads, hd);
                    e.enc_rmsnorm_dim(enc, &self.kvs, kn, n * cfg.num_key_value_heads, hd);
                    bar!(&self.q, &self.kvs);
                }
                let rp = RopeParams {
                    head_dim: hd as u32,
                    n_heads: cfg.num_attention_heads as u32,
                    pos0: pos0 as u32,
                    theta: cfg.rope_theta,
                    n_rows: n as u32,
                };
                enc.set_compute_pipeline_state(&e.pipes.rope);
                enc.set_buffer(0, Some(&self.q), 0);
                enc.set_bytes(1, size_of::<RopeParams>() as u64, &rp as *const _ as *const _);
                dispatch_grid(enc, n * cfg.num_attention_heads * hd / 2);
                for &(row, slot, len) in &win_write_spans(&w.cfg, pos0, n) {
                    let src_off = (row * kvd * 4) as u64;
                    let dst_off = (slot * kvd * 2) as u64;
                    e.enc_f32_to_f16(enc, &self.kvs, src_off, &self.k_cache[l], dst_off, len * kvd);
                    e.enc_f32_to_f16(enc, &self.kvs, v_base + src_off, &self.v_cache[l], dst_off, len * kvd);
                    // Concurrent encoder (unlike lowmem's serial one): the rope
                    // reads the rows the convert just wrote — order them.
                    bar!(&self.k_cache[l]);
                    let rp = RopeParams {
                        head_dim: hd as u32,
                        n_heads: cfg.num_key_value_heads as u32,
                        pos0: (pos0 + row) as u32,
                        theta: cfg.rope_theta,
                        n_rows: len as u32,
                    };
                    enc.set_compute_pipeline_state(&e.pipes.rope_h);
                    enc.set_buffer(0, Some(&self.k_cache[l]), dst_off);
                    enc.set_bytes(1, size_of::<RopeParams>() as u64, &rp as *const _ as *const _);
                    dispatch_grid(enc, len * cfg.num_key_value_heads * hd / 2);
                }
                // kvs rides in the barrier too: the NEXT layer's projections
                // overwrite the staging the converts above still read.
                bar!(&self.q, &self.k_cache[l], &self.v_cache[l], &self.kvs);
            } else {
                e.enc_linear_kv(enc, &blk.k_proj, &self.xn, &self.k_cache[l], kv_byte_off, n, Some(&self.xh));
                e.enc_linear_kv(enc, &blk.v_proj, &self.xn, &self.v_cache[l], kv_byte_off, n, Some(&self.xh));
                bar!(&self.q, &self.k_cache[l], &self.v_cache[l]);
                if let (Some(qn), Some(kn)) = (&blk.q_norm, &blk.k_norm) {
                    // qwen3: every head of q (f32) and the fresh k rows (f16,
                    // already in the cache) normalize BEFORE RoPE.
                    e.enc_rmsnorm_dim(enc, &self.q, qn, n * cfg.num_attention_heads, e.dims.head_dim);
                    e.enc_rmsnorm_h_inplace(enc, &self.k_cache[l], kv_byte_off, kn, n * cfg.num_key_value_heads, e.dims.head_dim);
                    bar!(&self.q, &self.k_cache[l]);
                }
                // One fused launch rotates q (f32) and the fresh k cache rows (f16);
                // per-element math identical to the separate rope/rope_h dispatches.
                let hd = e.dims.head_dim;
                let rp = RopeQkPrefillParams {
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
                enc.set_bytes(2, size_of::<RopeQkPrefillParams>() as u64, &rp as *const _ as *const _);
                dispatch_grid(enc, n * (cfg.num_attention_heads + cfg.num_key_value_heads) * hd / 2);
                bar!(&self.q, &self.k_cache[l]);
            }
            {
                let kv_extent = match &e.win {
                    Some(w) => w.cfg.cap, // the fallback kernel's scores stride
                    None => self.max_seq,
                };
                e.enc_attention(enc, &self.q, &self.k_cache[l], &self.v_cache[l], self.kv_base, &self.scores, &self.att, pos0, n, kv_extent, &self.xh);
                bar!(&self.att, &self.xh, &self.scores);
            }
            e.enc_linear(enc, &blk.o_proj, &self.att, 0, &self.xb, 0, n, Some(&self.xh), e.dims.head_dim != FLASH_HEAD_DIM, conc);
            bar!(&self.xb);
            e.enc_elementwise(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);
            bar!(&self.x);

            // SwiGLU MLP half.
            e.enc_rmsnorm_hf(enc, &self.x, &blk.post_attention_layernorm, &self.xn, &self.xh, n);
            bar!(&self.xn, &self.xh);
            e.enc_linear(enc, &blk.gate_proj, &self.xn, 0, &self.gate, 0, n, Some(&self.xh), false, conc);
            e.enc_linear(enc, &blk.up_proj, &self.xn, 0, &self.up, 0, n, Some(&self.xh), false, conc);
            bar!(&self.gate, &self.up);
            {
                let p = ElemParams { dim: (n * cfg.intermediate_size) as u32 };
                enc.set_compute_pipeline_state(&e.pipes.silu_mul_hf);
                enc.set_buffer(0, Some(&self.gate), 0);
                enc.set_buffer(1, Some(&self.up), 0);
                enc.set_buffer(2, Some(&self.xh), 0);
                enc.set_bytes(3, size_of::<ElemParams>() as u64, &p as *const _ as *const _);
                dispatch_grid(enc, n * cfg.intermediate_size);
            }
            bar!(&self.gate, &self.xh); // silu_mul_hf writes both copies
            e.enc_linear(enc, &blk.down_proj, &self.gate, 0, &self.xb, 0, n, Some(&self.xh), false, conc);
            bar!(&self.xb);
            e.enc_elementwise(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);
            bar!(&self.x);
        }
        if logits_rows > 0 {
            // Norm every row (cheap), then run the big lm_head only on the rows whose
            // logits are wanted — the final one for decode, all of them for verification.
            e.enc_rmsnorm_hf(enc, &self.x, &e.norm, &self.xn, &self.xh, n);
            bar!(&self.xn, &self.xh);
            let first = n - logits_rows;
            e.enc_linear(enc, &e.lm_head, &self.xn, (first * h * 4) as u64, &self.logits, 0, logits_rows, Some(&self.xh), true, conc);
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

    /// One matvec-family dispatch against a quant (or f16-in-GGUF) weight.
    /// `y_elem` is the output element width: 4 = f32 buffers, 2 = the f16 cache.
    #[allow(clippy::too_many_arguments)]
    fn enc_qmv(
        &self,
        enc: &ComputeCommandEncoderRef,
        pipe: &ComputePipelineState,
        l: &QuantLinear,
        x: &Buffer,
        y: &Buffer,
        y_base: u64,
    ) {
        let p = MatvecParams { in_dim: l.in_dim, out_dim: l.out_dim };
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&l.w), l.w_off);
        enc.set_buffer(1, Some(&l.bias), 0);
        enc.set_buffer(2, Some(x), 0);
        enc.set_buffer(3, Some(y), y_base);
        enc.set_bytes(4, size_of::<MatvecParams>() as u64, &p as *const _ as *const _);
        dispatch_simdgroup_rows(enc, l.out_dim);
    }

    /// One prefill GEMM through matmul_pg's dequant-tile staging: Y = X·Wᵀ+b,
    /// X read as f32 rows at `x_off` bytes, Y written f32 at `y_base` bytes.
    #[allow(clippy::too_many_arguments)]
    fn enc_qmm(
        &self,
        enc: &ComputeCommandEncoderRef,
        pipe: &ComputePipelineState,
        l: &QuantLinear,
        x: &Buffer,
        x_off: u64,
        y: &Buffer,
        y_base: u64,
        n_rows: usize,
    ) {
        let p = MatmulPagedParams {
            in_dim: l.in_dim,
            out_dim: l.out_dim,
            n_rows: n_rows as u32,
            y_stride: l.out_dim,
        };
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&l.w), l.w_off);
        enc.set_buffer(1, Some(&l.bias), 0);
        enc.set_buffer(2, Some(x), x_off);
        enc.set_buffer(3, Some(y), y_base);
        enc.set_bytes(4, size_of::<MatmulPagedParams>() as u64, &p as *const _ as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((l.out_dim as u64).div_ceil(64), (n_rows as u64).div_ceil(32), 1),
            MTLSize::new(128, 1, 1),
        );
    }

    /// run_from's twin for quant-GGUF weights: same chunk walk, same attention
    /// and KV layout (full causal unless the window is on — enc_attention and
    /// the span writer already switch on it), weights dequantized on read via
    /// the precise-library pipelines. Embeddings gather on the CPU per token
    /// through dequant_row_ref — no f16 table exists.
    fn run_from_quant(
        &mut self,
        src: Source<'_>,
        n: usize,
        pos0: usize,
        layer0: usize,
        logits_rows: usize,
    ) -> crate::Result<Vec<f32>> {
        let e = self.engine;
        let q = e.quant.as_ref().expect("quant path entered without state");
        let cfg = &e.cfg;
        let (h, kvd, hd) = (cfg.hidden_size, e.dims.kv_dim, e.dims.head_dim);
        let kv_slot0 = match &e.win {
            Some(w) => w.cfg.slot_of(pos0),
            None => pos0,
        };
        let kv_byte_off = self.kv_base + (kv_slot0 * kvd * 2) as u64;

        match src {
            Source::Ids(ids) => {
                // CPU embedding gather: one quant row per token, dequantized
                // straight into the x buffer (unified memory).
                let xp = self.x.contents() as *mut f32;
                for (i, &id) in ids.iter().enumerate() {
                    let row = q.source.read_rows(q.embed_name, id as usize, id as usize + 1)?;
                    let ty = q.source.src_type(q.embed_name)?;
                    let dst = unsafe { std::slice::from_raw_parts_mut(xp.add(i * h), h) };
                    match ty {
                        SrcType::Quant(t) => crate::lowmem::manifest::dequant_row_ref(t, row, dst),
                        SrcType::F16 => {
                            for (j, c) in row.chunks_exact(2).enumerate() {
                                dst[j] = f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
                            }
                        }
                        SrcType::F32 => {
                            for (j, c) in row.chunks_exact(4).enumerate() {
                                dst[j] = f32::from_le_bytes(c.try_into().unwrap());
                            }
                        }
                        SrcType::BF16 => return Err("bf16 rows inside a GGUF are unsupported".into()),
                    }
                }
            }
            Source::Hidden(x) => unsafe {
                std::ptr::copy_nonoverlapping(x.as_ptr(), self.x.contents() as *mut f32, n * h)
            },
        }

        let cb = e.queue.new_command_buffer();
        let fused_decode = n == 1 && hd <= DEC_TG && hd.is_multiple_of(4);
        let enc = if fused_decode {
            cb.new_compute_command_encoder()
        } else {
            cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent)
        };
        let conc = !fused_decode;
        macro_rules! bar {
            ($($b:expr),+) => { if conc { enc.memory_barrier_with_resources(&[$($b),+]) } };
        }

        let v_base = self.kvs.length() / 2;
        for (l, blk) in q.blocks.iter().enumerate().skip(layer0) {
            if fused_decode {
                e.enc_rmsnorm(enc, &self.x, &blk.input_layernorm, &self.xn, 1);
                self.enc_qmv(enc, &q.pipe(blk.q_proj.sel).matvec, &blk.q_proj, &self.xn, &self.q, 0);
                self.enc_qmv(enc, &q.pipe(blk.k_proj.sel).matvec_h, &blk.k_proj, &self.xn, &self.k_cache[l], kv_byte_off);
                self.enc_qmv(enc, &q.pipe(blk.v_proj.sel).matvec_h, &blk.v_proj, &self.xn, &self.v_cache[l], kv_byte_off);
                if let (Some(qn), Some(kn)) = (&blk.q_norm, &blk.k_norm) {
                    e.enc_rmsnorm_dim(enc, &self.q, qn, cfg.num_attention_heads, hd);
                    e.enc_rmsnorm_h_inplace(enc, &self.k_cache[l], kv_byte_off, kn, cfg.num_key_value_heads, hd);
                }
                e.enc_rope_qk(enc, &self.q, &self.k_cache[l], kv_byte_off, pos0);
                e.enc_attention_decode(enc, &self.q, &self.k_cache[l], &self.v_cache[l], self.kv_base, &self.partials, &self.att, pos0);
                self.enc_qmv(enc, &q.pipe(blk.o_proj.sel).matvec_acc, &blk.o_proj, &self.att, &self.x, 0);
                e.enc_rmsnorm(enc, &self.x, &blk.post_attention_layernorm, &self.xn, 1);
                // gate/up dispatch separately: a mixed-quant file may hold the
                // two halves in different encodings, and matvec_swiglu assumes
                // one selector for both.
                self.enc_qmv(enc, &q.pipe(blk.gate_proj.sel).matvec, &blk.gate_proj, &self.xn, &self.gate, 0);
                self.enc_qmv(enc, &q.pipe(blk.up_proj.sel).matvec, &blk.up_proj, &self.xn, &self.up, 0);
                let p = ElemParams { dim: cfg.intermediate_size as u32 };
                enc.set_compute_pipeline_state(&e.pipes.silu_mul);
                enc.set_buffer(0, Some(&self.gate), 0);
                enc.set_buffer(1, Some(&self.up), 0);
                enc.set_bytes(2, size_of::<ElemParams>() as u64, &p as *const _ as *const _);
                dispatch_grid(enc, cfg.intermediate_size);
                self.enc_qmv(enc, &q.pipe(blk.down_proj.sel).matvec_acc, &blk.down_proj, &self.gate, &self.x, 0);
                continue;
            }

            // Prefill: rmsnorm (f32), quant GEMMs, staged K/V into the cache.
            e.enc_rmsnorm(enc, &self.x, &blk.input_layernorm, &self.xn, n);
            bar!(&self.xn);
            self.enc_qmm(enc, &q.pipe(blk.q_proj.sel).matmul_pg, &blk.q_proj, &self.xn, 0, &self.q, 0, n);
            self.enc_qmm(enc, &q.pipe(blk.k_proj.sel).matmul_pg, &blk.k_proj, &self.xn, 0, &self.kvs, 0, n);
            self.enc_qmm(enc, &q.pipe(blk.v_proj.sel).matmul_pg, &blk.v_proj, &self.xn, 0, &self.kvs, v_base, n);
            bar!(&self.q, &self.kvs);
            if let (Some(qn), Some(kn)) = (&blk.q_norm, &blk.k_norm) {
                // qwen3: per-head norm before RoPE — q in place (f32), k while
                // still in the f32 staging half (why this precedes the spans).
                e.enc_rmsnorm_dim(enc, &self.q, qn, n * cfg.num_attention_heads, hd);
                e.enc_rmsnorm_dim(enc, &self.kvs, kn, n * cfg.num_key_value_heads, hd);
                bar!(&self.q, &self.kvs);
            }
            {
                let rp = RopeParams {
                    head_dim: hd as u32,
                    n_heads: cfg.num_attention_heads as u32,
                    pos0: pos0 as u32,
                    theta: cfg.rope_theta,
                    n_rows: n as u32,
                };
                enc.set_compute_pipeline_state(&e.pipes.rope);
                enc.set_buffer(0, Some(&self.q), 0);
                enc.set_bytes(1, size_of::<RopeParams>() as u64, &rp as *const _ as *const _);
                dispatch_grid(enc, n * cfg.num_attention_heads * hd / 2);
            }
            let spans: Vec<(usize, usize, usize)> = match &e.win {
                Some(w) => win_write_spans(&w.cfg, pos0, n),
                None => vec![(0, pos0, n)],
            };
            for &(row, slot, len) in &spans {
                let src_off = (row * kvd * 4) as u64;
                let dst_off = self.kv_base + (slot * kvd * 2) as u64;
                e.enc_f32_to_f16(enc, &self.kvs, src_off, &self.k_cache[l], dst_off, len * kvd);
                e.enc_f32_to_f16(enc, &self.kvs, v_base + src_off, &self.v_cache[l], dst_off, len * kvd);
                bar!(&self.k_cache[l]);
                let rp = RopeParams {
                    head_dim: hd as u32,
                    n_heads: cfg.num_key_value_heads as u32,
                    pos0: (pos0 + row) as u32,
                    theta: cfg.rope_theta,
                    n_rows: len as u32,
                };
                enc.set_compute_pipeline_state(&e.pipes.rope_h);
                enc.set_buffer(0, Some(&self.k_cache[l]), dst_off);
                enc.set_bytes(1, size_of::<RopeParams>() as u64, &rp as *const _ as *const _);
                dispatch_grid(enc, len * cfg.num_key_value_heads * hd / 2);
            }
            bar!(&self.q, &self.k_cache[l], &self.v_cache[l], &self.kvs);
            {
                let kv_extent = match &e.win {
                    Some(w) => w.cfg.cap,
                    None => self.max_seq,
                };
                e.enc_attention(enc, &self.q, &self.k_cache[l], &self.v_cache[l], self.kv_base, &self.scores, &self.att, pos0, n, kv_extent, &self.xh);
                bar!(&self.att, &self.scores);
            }
            self.enc_qmm(enc, &q.pipe(blk.o_proj.sel).matmul_pg, &blk.o_proj, &self.att, 0, &self.xb, 0, n);
            bar!(&self.xb);
            e.enc_elementwise(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);
            bar!(&self.x);

            e.enc_rmsnorm(enc, &self.x, &blk.post_attention_layernorm, &self.xn, n);
            bar!(&self.xn);
            self.enc_qmm(enc, &q.pipe(blk.gate_proj.sel).matmul_pg, &blk.gate_proj, &self.xn, 0, &self.gate, 0, n);
            self.enc_qmm(enc, &q.pipe(blk.up_proj.sel).matmul_pg, &blk.up_proj, &self.xn, 0, &self.up, 0, n);
            bar!(&self.gate, &self.up);
            {
                let p = ElemParams { dim: (n * cfg.intermediate_size) as u32 };
                enc.set_compute_pipeline_state(&e.pipes.silu_mul);
                enc.set_buffer(0, Some(&self.gate), 0);
                enc.set_buffer(1, Some(&self.up), 0);
                enc.set_bytes(2, size_of::<ElemParams>() as u64, &p as *const _ as *const _);
                dispatch_grid(enc, n * cfg.intermediate_size);
            }
            bar!(&self.gate);
            self.enc_qmm(enc, &q.pipe(blk.down_proj.sel).matmul_pg, &blk.down_proj, &self.gate, 0, &self.xb, 0, n);
            bar!(&self.xb);
            e.enc_elementwise(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);
            bar!(&self.x);
        }

        if logits_rows > 0 {
            e.enc_rmsnorm(enc, &self.x, &q.final_norm, &self.xn, n);
            bar!(&self.xn);
            let first = n - logits_rows;
            if logits_rows == 1 && !conc {
                self.enc_qmv(enc, &q.pipe(q.lm_head.sel).matvec, &q.lm_head, &self.xn, &self.logits, 0);
            } else {
                self.enc_qmm(enc, &q.pipe(q.lm_head.sel).matmul_pg, &q.lm_head, &self.xn, (first * h * 4) as u64, &self.logits, 0, logits_rows);
            }
        }

        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        if logits_rows == 0 {
            return Ok(Vec::new());
        }
        let logits = unsafe {
            std::slice::from_raw_parts(self.logits.contents() as *const f32, logits_rows * cfg.vocab_size)
        };
        Ok(logits.to_vec())
    }

    /// The model config, for callers (the ane backend) that only hold a session.
    pub(crate) fn config_ref(&self) -> &ModelConfig {
        &self.engine.cfg
    }

    /// The engine's window size, if the session runs in window mode.
    pub(crate) fn window(&self) -> Option<usize> {
        self.engine.win.as_ref().map(|w| w.cfg.w)
    }

    /// Write K,V computed elsewhere (e.g. on the ANE) into the cache at pos0 onward,
    /// converting to the cache's f16 on the way. With unified memory this is the
    /// whole "device transfer".
    pub(crate) fn write_kv(&mut self, layer: usize, pos0: usize, k: &[f32], v: &[f32]) {
        let kvd = self.engine.dims.kv_dim;
        let base = (self.kv_base / 2) as usize;
        unsafe {
            let kp = self.k_cache[layer].contents() as *mut u16;
            let vp = self.v_cache[layer].contents() as *mut u16;
            for r in 0..k.len() / kvd {
                let slot = self.kv_slot(pos0 + r);
                for i in 0..kvd {
                    *kp.add(base + slot * kvd + i) = f16::from_f32(k[r * kvd + i]).to_bits();
                    *vp.add(base + slot * kvd + i) = f16::from_f32(v[r * kvd + i]).to_bits();
                }
            }
        }
    }

    /// Position -> cache row: identity full-causal, the ring slot under a window.
    fn kv_slot(&self, p: usize) -> usize {
        match &self.engine.win {
            Some(w) => w.cfg.slot_of(p),
            None => p,
        }
    }

    /// Same as `write_kv`, for K,V that already carry the cache's f16 bits — one
    /// memcpy per layer instead of a per-element convert. Split prefill converts
    /// on the ANE thread (it needs the f16 rows anyway, to feed the next chunk's
    /// past), which keeps the conversion off the thread driving the GPU.
    pub(crate) fn write_kv_bits(&mut self, layer: usize, pos0: usize, k: &[u16], v: &[u16]) {
        let kvd = self.engine.dims.kv_dim;
        let base = (self.kv_base / 2) as usize;
        unsafe {
            let kp = self.k_cache[layer].contents() as *mut u16;
            let vp = self.v_cache[layer].contents() as *mut u16;
            for r in 0..k.len() / kvd {
                // Per-row: the ring can wrap mid-chunk under a window (full
                // causal degenerates to the old two-memcpy copy, row by row).
                let slot = self.kv_slot(pos0 + r);
                std::ptr::copy_nonoverlapping(
                    k.as_ptr().add(r * kvd),
                    kp.add(base + slot * kvd),
                    kvd,
                );
                std::ptr::copy_nonoverlapping(
                    v.as_ptr().add(r * kvd),
                    vp.add(base + slot * kvd),
                    kvd,
                );
            }
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
    /// The engine's configured window size (None = full causal).
    pub(crate) fn window_size(&self) -> Option<usize> {
        self.win.as_ref().map(|w| w.cfg.w)
    }

    pub(crate) fn make_batcher(&self, n_slots: usize, max_seq: usize) -> Option<MetalBatcher<'_>> {
        let cfg = &self.cfg;
        // qwen35 v1: continuous batching needs per-slot recurrent state with
        // decode continuity, and decode_step never goes through a session (a
        // fresh slot_session would hand every step a zeroed state). Serve
        // falls back to per-request sessions, each owning its own state —
        // the same v1 shape as quant files in the batcher.
        if self.q35_layout.is_some() {
            return None;
        }
        // The batched kernels are the fused-decode family — same requirements.
        if self.dims.head_dim > DEC_TG || !self.dims.head_dim.is_multiple_of(4) || n_slots > SPEC_MAX {
            return None;
        }
        let d = &self.device;
        let (h, kvd) = (cfg.hidden_size, self.dims.kv_dim);
        let splits_max = max_seq.div_ceil(ATTN_SPLIT);
        Some(MetalBatcher {
            k_cache: (0..cfg.num_hidden_layers)
                .map(|_| f16_empty_buffer(d, (n_slots * max_seq + FLASH_C) * kvd))
                .collect(),
            v_cache: (0..cfg.num_hidden_layers)
                .map(|_| f16_empty_buffer(d, (n_slots * max_seq + FLASH_C) * kvd))
                .collect(),
            ids: d.new_buffer((n_slots * 4) as u64, MTLResourceOptions::StorageModeShared),
            meta: d.new_buffer(
                (n_slots * size_of::<RowMeta>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            x: f32_buffer(d, n_slots * h),
            xn: f32_buffer(d, n_slots * h),
            // q and att are Q-WIDTH, not hidden-width — matvec_qkv_batch below
            // writes blk.q_proj.out_dim per row. Sizing these by hidden_size
            // overran them on every config where q_dim > hidden_size (Qwen3-0.6B
            // and 4B among them); session_scratch had the same bug and this
            // path was missed when that one was fixed.
            q: f32_buffer(d, n_slots * attn_row_width(h, self.dims.q_dim)),
            att: f32_buffer(d, n_slots * attn_row_width(h, self.dims.q_dim)),
            gate: f32_buffer(d, n_slots * cfg.intermediate_size),
            logits: f32_buffer(d, n_slots * cfg.vocab_size),
            partials: f32_buffer(
                d,
                n_slots * cfg.num_attention_heads * splits_max * (self.dims.head_dim + 2),
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
        let kvd = self.engine.dims.kv_dim;
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
        let (hd, kvd) = (self.engine.dims.head_dim, self.engine.dims.kv_dim);
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
        e.enc_linear(enc, &e.lm_head, &self.xn, 0, &self.logits, 0, n, None, false, false);

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
    /// q_dim is NOT hidden_size, and on real configs it is bigger. This pins
    /// the invariant that every Q/attention-width allocation must use, because
    /// getting it wrong overruns the buffer silently — the failure shows up as
    /// nondeterminism, not as a crash, which is the worst way to find it.
    ///
    /// Found twice: session_scratch first, then the serve batcher, which was
    /// still sizing q/att by hidden_size long after the first fix. The numbers
    /// below are real configs, not invented ones.
    #[test]
    fn attn_rows_are_q_wide_not_hidden_wide() {
        use super::attn_row_width;
        // Qwen3-0.6B: hidden 1024, 16 heads x 128 = q_dim 2048.
        assert_eq!(attn_row_width(1024, 2048), 2048, "0.6B q rows are 2x hidden");
        // qwen35 27B: hidden 5120, q_dim 6144.
        assert_eq!(attn_row_width(5120, 6144), 6144, "qwen35 q rows exceed hidden");
        // Qwen3-8B: hidden 4096, 32 heads x 128 = q_dim 4096 — equal, and the
        // reason the bug hid for so long: the models people tested were exactly
        // the ones where the two happen to coincide.
        assert_eq!(attn_row_width(4096, 4096), 4096);
        // A hidden-dominant config must still get hidden.
        assert_eq!(attn_row_width(8192, 4096), 8192);
    }

    #[test]
    fn window_cfg_mirror_matches_lowmem_formula() {
        // Pins the metal-side mirror (challenge 7c1a09cf) to lowmem's values:
        // sink_pad = sink aligned to 128, ring = (w + PREFILL_CHUNK) aligned.
        let w = super::window_cfg(2048, 4).unwrap();
        assert_eq!((w.w, w.sink, w.sink_pad, w.ring, w.cap), (2048, 4, 128, 2560, 2688));
        assert_eq!(w.slot_of(3), 3);
        assert_eq!(w.slot_of(4), 128);
        assert_eq!(w.slot_of(4 + 2560), 128); // ring wrap
        assert!(super::window_cfg(0, 0).is_err());
        assert!(super::window_cfg(64, 128).is_err());
    }

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

    /// qwen35 session-state lifecycle: allocate → zeroed, read-modify-write
    /// sticks within a sequence, reset → zeroed again; attention layers hold
    /// no state; the map's length is exactly the trunk (the MTP block cannot
    /// even be expressed, so its attention can never allocate).
    #[test]
    fn qwen35_states_lifecycle() {
        let device = Device::system_default().expect("metal device");
        // Interval-4 trunk of 8: layers 3 and 7 are attention.
        let layout = Qwen35Layout {
            is_recurrent: (0..8).map(|i| (i + 1) % 4 != 0).collect(),
            conv_elems: 6,
            delta_elems: 10,
        };
        let st = Qwen35States::new(&device, &layout);
        assert_eq!(st.layers.len(), 8, "map length is the trunk, nothing more");
        for (i, l) in st.layers.iter().enumerate() {
            assert_eq!(l.is_some(), (i + 1) % 4 != 0, "layer {i}");
        }
        assert_eq!(st.total_bytes(), 6 * (6 + 10) * 4);

        let read = |b: &Buffer, n: usize| unsafe {
            std::slice::from_raw_parts(b.contents() as *const f32, n).to_vec()
        };
        let l0 = st.layers[0].as_ref().unwrap();
        assert_eq!(read(&l0.conv, 6), vec![0.0; 6], "fresh conv state is silence");
        assert_eq!(read(&l0.delta, 10), vec![0.0; 10], "fresh delta state attends nothing");

        // The kernel lane's contract: read, modify, write back — and the write
        // must still be there on the next step of the SAME sequence.
        unsafe {
            let c = l0.conv.contents() as *mut f32;
            for i in 0..6 {
                *c.add(i) = read(&l0.conv, 6)[i] + (i as f32 + 1.0);
            }
            *(l0.delta.contents() as *mut f32).add(9) = -2.5;
        }
        assert_eq!(read(&l0.conv, 6), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(read(&l0.delta, 10)[9], -2.5);
        // Other layers are untouched — no aliasing between layer buffers.
        let l1 = st.layers[1].as_ref().unwrap();
        assert_eq!(read(&l1.conv, 6), vec![0.0; 6]);

        // New sequence on a reused session: reset returns to start-of-sequence.
        st.reset();
        assert_eq!(read(&l0.conv, 6), vec![0.0; 6]);
        assert_eq!(read(&l0.delta, 10), vec![0.0; 10]);
    }

    /// The one canonical meta→layout translation carries the real sizes and
    /// only the trunk's map.
    #[test]
    fn qwen35_layout_from_meta_is_trunk_shaped() {
        let meta = crate::lowmem::gguf::Qwen35Meta {
            trunk_layers: 64,
            nextn_layers: 1,
            full_attention_interval: 4,
            is_recurrent: (0..64).map(|i| (i + 1) % 4 != 0).collect(),
            d_conv: 4,
            d_state: 128,
            n_group: 16,
            dt_rank: 48,
            d_inner: 6144,
            rope_sections: [11, 11, 10, 0],
            conv_state_elems: 30_720,
            delta_state_elems: 786_432,
        };
        let layout = Qwen35Layout::from_meta(&meta);
        assert_eq!(layout.is_recurrent.len(), 64);
        assert_eq!(layout.is_recurrent.iter().filter(|&&r| r).count(), 48);
        assert_eq!(layout.conv_elems, 30_720);
        assert_eq!(layout.delta_elems, 786_432);
        // The 27B budget figure, from the layout alone (no allocation):
        // 48 × (30,720 + 786,432) × 4 bytes ≈ 149 MB per sequence.
        let bytes: usize = layout.is_recurrent.iter().filter(|&&r| r).count()
            * (layout.conv_elems + layout.delta_elems)
            * 4;
        assert_eq!(bytes >> 20, 149);
    }
}

#[cfg(test)]
mod qwen35_kernel_oracle {
    //! GPU kernels vs lane B's CPU reference (src/lowmem/qwen35_ref.rs),
    //! bit-for-bit. Same doctrine as the quant oracle: the reference is the
    //! subject, the GPU is the thing under test, and a negative control proves
    //! the comparison can fail.
    use crate::gpu::metal as gpu;
    use crate::lowmem::qwen35_ref as rf;
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

    /// Small but structurally real: several channels, the true d_conv, and a
    /// state that is genuinely rolled across steps rather than used once.
    fn dims() -> rf::DeltaDims {
        rf::DeltaDims { d_state: 8, n_v_heads: 4, n_k_heads: 2, d_conv: 4 }
    }

    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    /// Run ssm_conv_decode for `steps` tokens, returning every step's output
    /// and the final state — the rolling is half the semantics, so a
    /// single-shot comparison would miss a broken roll entirely.
    fn gpu_conv(d: rf::DeltaDims, state0: &[f32], xs: &[Vec<f32>], w: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let device = Device::system_default().expect("Metal device required");
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(&gpu::shader_source(128), &opts)
            .expect("kernels.metal compiles");
        let f = lib.get_function("ssm_conv_decode", None).expect("ssm_conv_decode");
        let pipe = device.new_compute_pipeline_state_with_function(&f).expect("pipeline");
        let queue = device.new_command_queue();
        let c_all = d.conv_channels();

        let bytes = |v: &[f32]| (std::mem::size_of_val(v)) as u64;
        let st = device.new_buffer_with_data(
            state0.as_ptr() as *const _, bytes(state0), MTLResourceOptions::StorageModeShared);
        let wb = device.new_buffer_with_data(
            w.as_ptr() as *const _, bytes(w), MTLResourceOptions::StorageModeShared);
        let out = device.new_buffer((c_all * 4) as u64, MTLResourceOptions::StorageModeShared);

        #[repr(C)]
        struct P { channels: u32, d_conv: u32 }
        let p = P { channels: c_all as u32, d_conv: d.d_conv as u32 };
        let mut outs = Vec::new();
        for x in xs {
            let xb = device.new_buffer_with_data(
                x.as_ptr() as *const _, bytes(x), MTLResourceOptions::StorageModeShared);
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&st), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&wb), 0);
            enc.set_buffer(3, Some(&out), 0);
            enc.set_bytes(4, std::mem::size_of::<P>() as u64,
                          &p as *const P as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(((c_all + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            outs.push(unsafe { std::slice::from_raw_parts(out.contents() as *const f32, c_all) }.to_vec());
        }
        let final_state =
            unsafe { std::slice::from_raw_parts(st.contents() as *const f32, state0.len()) }.to_vec();
        (outs, final_state)
    }

    #[test]
    fn conv_decode_matches_reference_bit_for_bit() {
        let d = dims();
        let (c_all, k) = (d.conv_channels(), d.d_conv);
        let mut seed = 0x2545_F491u32;
        let state0: Vec<f32> = (0..c_all * (k - 1)).map(|_| lcg(&mut seed)).collect();
        let w: Vec<f32> = (0..c_all * k).map(|_| lcg(&mut seed)).collect();
        let xs: Vec<Vec<f32>> =
            (0..5).map(|_| (0..c_all).map(|_| lcg(&mut seed)).collect()).collect();

        let (gpu_outs, gpu_state) = gpu_conv(d, &state0, &xs, &w);

        // Two different bars, because two different things are being checked.
        //
        // The DOT is pure multiply-add and must be BIT-EXACT: we control the
        // order and, with contraction off, the GPU reproduces ggml's scalar
        // form exactly. Anything less means a real arithmetic divergence.
        //
        // The SiLU output cannot be bit-exact: it goes through exp(), and no
        // transcendental is bit-specified across implementations — Metal's and
        // libm's differ, and `precise::exp` does not close it either (measured,
        // not assumed). So this gate bounds the END-TO-END value at 2 ulp.
        //
        // That the DOT itself is bit-exact was established separately, by
        // running this same comparison with silu removed from the kernel: it
        // failed by 1 ulp until `#pragma clang fp contract(off)` stopped Metal
        // fusing the multiply-add, then matched exactly. The bound below is
        // therefore exp's floor and nothing else, which is why 2 ulp is an
        // assertion rather than a shrug.
        let mut ref_state = state0.clone();
        let mut worst_ulp = 0i64;
        for (step, x) in xs.iter().enumerate() {
            let want = rf::conv_step(&d, &mut ref_state, x, &w);
            for (i, (a, b)) in want.iter().zip(&gpu_outs[step]).enumerate() {
                let ulp = (a.to_bits() as i64 - b.to_bits() as i64).abs();
                assert!(
                    ulp <= 2,
                    "step {step} channel {i}: reference {a} vs gpu {b} ({ulp} ulp) — \
                     more than exp()'s spread means a real divergence"
                );
                worst_ulp = worst_ulp.max(ulp);
            }
        }
        // Pin the observed spread: if it ever grows, something changed that is
        // not exp's last bit.
        assert!(worst_ulp <= 2, "worst {worst_ulp} ulp across the run");
        // The rolled state must match too: an off-by-one roll produces correct
        // FIRST outputs and wrong later ones, so only comparing step 0 would
        // pass a broken kernel.
        for (i, (a, b)) in ref_state.iter().zip(&gpu_state).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "final state slot {i}");
        }
    }

    /// The comparison must be able to FAIL: feeding the reference a state that
    /// differs in one slot has to break bit-equality somewhere.
    #[test]
    fn conv_oracle_has_a_negative_control() {
        let d = dims();
        let (c_all, k) = (d.conv_channels(), d.d_conv);
        let mut seed = 99u32;
        let state0: Vec<f32> = (0..c_all * (k - 1)).map(|_| lcg(&mut seed)).collect();
        let w: Vec<f32> = (0..c_all * k).map(|_| lcg(&mut seed)).collect();
        let xs: Vec<Vec<f32>> = vec![(0..c_all).map(|_| lcg(&mut seed)).collect()];

        let (gpu_outs, _) = gpu_conv(d, &state0, &xs, &w);
        let mut perturbed = state0.clone();
        perturbed[0] = perturbed[0] + 1.0;
        let mut rs = perturbed;
        let want = rf::conv_step(&d, &mut rs, &xs[0], &w);
        assert!(
            want.iter().zip(&gpu_outs[0]).any(|(a, b)| a.to_bits() != b.to_bits()),
            "a perturbed state must change the output, or the oracle proves nothing"
        );
    }

    /// Run delta_decode_step for `steps` tokens against one persistent state,
    /// returning every step's output and the final state.
    #[allow(clippy::too_many_arguments)]
    fn gpu_delta(
        d: rf::DeltaDims,
        state0: &[f32],
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        g: &[Vec<f32>],
        beta: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        let device = Device::system_default().expect("Metal device required");
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(&gpu::shader_source(128), &opts)
            .expect("kernels.metal compiles");
        let f = lib.get_function("delta_decode_step", None).expect("delta_decode_step");
        let pipe = device.new_compute_pipeline_state_with_function(&f).expect("pipeline");
        let queue = device.new_command_queue();

        let bytes = |v: &[f32]| (std::mem::size_of_val(v)) as u64;
        let shared = MTLResourceOptions::StorageModeShared;
        let st = device.new_buffer_with_data(state0.as_ptr() as *const _, bytes(state0), shared);
        let n_out = d.d_state * d.n_v_heads;
        let out = device.new_buffer((n_out * 4) as u64, shared);

        #[repr(C)]
        struct P {
            d_state: u32,
            n_v_heads: u32,
            group: u32,
        }
        let p = P {
            d_state: d.d_state as u32,
            n_v_heads: d.n_v_heads as u32,
            group: (d.n_v_heads / d.n_k_heads) as u32,
        };

        let mut outs = Vec::new();
        for step in 0..q.len() {
            let up = |x: &[f32]| device.new_buffer_with_data(x.as_ptr() as *const _, bytes(x), shared);
            let (qb, kb, vb) = (up(&q[step]), up(&k[step]), up(&v[step]));
            let (gb, bb) = (up(&g[step]), up(&beta[step]));
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&st), 0);
            enc.set_buffer(1, Some(&qb), 0);
            enc.set_buffer(2, Some(&kb), 0);
            enc.set_buffer(3, Some(&vb), 0);
            enc.set_buffer(4, Some(&gb), 0);
            enc.set_buffer(5, Some(&bb), 0);
            enc.set_buffer(6, Some(&out), 0);
            enc.set_bytes(7, std::mem::size_of::<P>() as u64, &p as *const P as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(((n_out + 63) / 64) as u64, 1, 1),
                MTLSize::new(64, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            outs.push(
                unsafe { std::slice::from_raw_parts(out.contents() as *const f32, n_out) }.to_vec(),
            );
        }
        let final_state =
            unsafe { std::slice::from_raw_parts(st.contents() as *const f32, state0.len()) }.to_vec();
        (outs, final_state)
    }

    /// Inputs for `steps` delta steps. `decay` chooses the g values: `None`
    /// means g = 0 exactly, which is what makes the bit-exact test possible.
    #[allow(clippy::type_complexity)]
    fn delta_inputs(
        d: rf::DeltaDims,
        steps: usize,
        seed: &mut u32,
        decay: bool,
    ) -> (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (s, hv, hk) = (d.d_state, d.n_v_heads, d.n_k_heads);
        let state0: Vec<f32> = (0..d.delta_state_elems()).map(|_| lcg(seed)).collect();
        fn mk(n: usize, seed: &mut u32) -> Vec<f32> {
            (0..n).map(|_| lcg(seed)).collect()
        }
        let q: Vec<Vec<f32>> = (0..steps).map(|_| mk(s * hk, seed)).collect();
        let k: Vec<Vec<f32>> = (0..steps).map(|_| mk(s * hk, seed)).collect();
        let v: Vec<Vec<f32>> = (0..steps).map(|_| mk(s * hv, seed)).collect();
        // g is a log-decay: the model's g = a·softplus(α+bias) is <= 0, so eᵍ
        // is a contraction. Feeding positive g would let the state blow up over
        // steps and the comparison would be measuring overflow, not the kernel.
        let g: Vec<Vec<f32>> = (0..steps)
            .map(|_| {
                (0..hv)
                    .map(|_| if decay { -(lcg(seed).abs()) } else { 0.0 })
                    .collect()
            })
            .collect();
        let beta: Vec<Vec<f32>> = (0..steps).map(|_| mk(hv, seed)).collect();
        (state0, q, k, v, g, beta)
    }

    /// With g = 0 the decay factor is exp(0) = 1.0 — exact in every
    /// implementation — so the ENTIRE kernel is add/multiply and must be
    /// bit-for-bit. This is the load-bearing test: it pins the state layout,
    /// the K-head broadcast, the 1/√S scaling, the update-then-read ordering
    /// inside the second loop, and the summation order of both dots. Only the
    /// decay itself is left to the bounded test below, so nothing can hide
    /// inside a tolerance.
    #[test]
    fn delta_decode_matches_reference_bit_for_bit() {
        let d = dims();
        let mut seed = 0x9E37_79B9u32;
        let (state0, q, k, v, g, beta) = delta_inputs(d, 5, &mut seed, false);
        let (gpu_outs, gpu_state) = gpu_delta(d, &state0, &q, &k, &v, &g, &beta);

        let mut ref_state = state0.clone();
        for step in 0..q.len() {
            let want =
                rf::delta_decode_step(&d, &mut ref_state, &q[step], &k[step], &v[step], &g[step], &beta[step]);
            for (i, (a, b)) in want.iter().zip(&gpu_outs[step]).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "step {step} slot {i}: reference {a} vs gpu {b} — with g=0 this kernel \
                     is pure multiply-add and has no excuse for a difference"
                );
            }
        }
        // Several steps AND the final state, because the state is the half that
        // a single-step comparison cannot see: a kernel that writes the right
        // outputs from a subtly wrong state passes step 0 and fails step 3.
        for (i, (a, b)) in ref_state.iter().zip(&gpu_state).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "final state slot {i}");
        }
    }

    /// With a live decay the kernel calls exp(), which is where bit-equality
    /// ends: Metal's exp and the host's differ in the last bit (measured on the
    /// conv kernel; `precise::exp` does not close it). Every element of the
    /// state is scaled by that value every step, so that difference is carried
    /// forward and amplified by the recurrence.
    ///
    /// So this test does NOT assert a hand-picked tolerance. It MEASURES the
    /// conditioning first — how far the reference drifts from ITSELF when
    /// exp(g) is moved by one rounding unit — and then requires the GPU to sit
    /// inside a small multiple of that. If the kernel's only divergence is
    /// exp's last bit, it lands there by construction; if it has a real
    /// arithmetic bug, no amount of ill-conditioning in the recurrence excuses
    /// it, because the yardstick is derived from the reference alone and a GPU
    /// bug cannot inflate it.
    ///
    /// A fixed tolerance would have been the wrong instrument here: the first
    /// draft of this test asserted 1e-5 and passed, and the measured drift was
    /// 5.5e-6 — about 46 ulp. That number is neither "exp's last bit" nor a
    /// bug; it is the recurrence's gain, and the only way to tell those apart
    /// is to measure the gain separately.
    #[test]
    fn delta_decode_bounded_once_the_decay_is_live() {
        let d = dims();
        let mut seed = 0x0DEF_ACEDu32;
        let (state0, q, k, v, g, beta) = delta_inputs(d, 5, &mut seed, true);
        let (gpu_outs, gpu_state) = gpu_delta(d, &state0, &q, &k, &v, &g, &beta);

        // exp(g + δ) = exp(g)·(1 + δ), so δ = 2⁻²⁴ — f32's rounding unit — is
        // exactly a one-rounding move in the decay factor, applied to every
        // head at once (the worst case: the GPU's per-head last-bit errors
        // point in arbitrary directions).
        let delta = (2f32).powi(-24);
        let g_nudged: Vec<Vec<f32>> =
            g.iter().map(|row| row.iter().map(|x| x + delta).collect()).collect();

        let run = |gs: &[Vec<f32>]| -> (Vec<Vec<f32>>, Vec<f32>) {
            let mut st = state0.clone();
            let outs = (0..q.len())
                .map(|t| rf::delta_decode_step(&d, &mut st, &q[t], &k[t], &v[t], &gs[t], &beta[t]))
                .collect();
            (outs, st)
        };
        let (ref_outs, ref_state) = run(&g);
        let (nudged_outs, nudged_state) = run(&g_nudged);

        let worst = |a: &[Vec<f32>], b: &[Vec<f32>], sa: &[f32], sb: &[f32]| -> f32 {
            let o = a
                .iter()
                .zip(b)
                .flat_map(|(x, y)| x.iter().zip(y))
                .fold(0f32, |m, (x, y)| m.max(rel_err(*x, *y)));
            sa.iter().zip(sb).fold(o, |m, (x, y)| m.max(rel_err(*x, *y)))
        };
        let yardstick = worst(&ref_outs, &nudged_outs, &ref_state, &nudged_state);
        let measured = worst(&ref_outs, &gpu_outs, &ref_state, &gpu_state);

        // The yardstick must be non-zero, or it is not measuring anything and
        // the ratio below would be vacuous.
        assert!(yardstick > 0.0, "one-rounding nudge changed nothing — the sensitivity probe is broken");
        // 4x, and the factor is derived rather than chosen: the yardstick moves
        // every head's decay in the SAME direction, and heads never mix in this
        // kernel, so a GPU whose only fault is exp's last bit cannot exceed ~2x
        // it (exp(g) for g<0 lands in (0,1), where one ulp is between 2⁻²⁴ and
        // 2⁻²³ relative). 4x is that ceiling with one doubling of margin.
        // Measured on M-series at this seed: gpu 5.5e-6, yardstick 1.9e-5,
        // ratio 0.29 — inside the envelope and 13x below the bar.
        assert!(
            measured <= 4.0 * yardstick,
            "gpu drift {measured:e} exceeds 4x the reference's own one-rounding sensitivity {yardstick:e} — that is more than exp's last bit can explain"
        );
    }

    fn rel_err(a: f32, b: f32) -> f32 {
        let scale = a.abs().max(b.abs());
        if scale == 0.0 { 0.0 } else { (a - b).abs() / scale }
    }

    /// Run attn_out_gate over a flat vector.
    fn gpu_attn_gate(attn: &[f32], gate: &[f32]) -> Vec<f32> {
        let device = Device::system_default().expect("Metal device required");
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(&gpu::shader_source(128), &opts)
            .expect("kernels.metal compiles");
        let f = lib.get_function("attn_out_gate", None).expect("attn_out_gate");
        let pipe = device.new_compute_pipeline_state_with_function(&f).expect("pipeline");
        let queue = device.new_command_queue();
        let shared = MTLResourceOptions::StorageModeShared;
        let ab = device.new_buffer_with_data(
            attn.as_ptr() as *const _, std::mem::size_of_val(attn) as u64, shared);
        let gb = device.new_buffer_with_data(
            gate.as_ptr() as *const _, std::mem::size_of_val(gate) as u64, shared);
        let n = attn.len() as u32;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&ab), 0);
        enc.set_buffer(1, Some(&gb), 0);
        enc.set_bytes(2, 4, &n as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(((attn.len() + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        unsafe { std::slice::from_raw_parts(ab.contents() as *const f32, attn.len()) }.to_vec()
    }

    /// The EXACT half for the out-gate. σ has three arguments where it is
    /// exactly representable in any implementation — σ(0) = 0.5 (exp(0) = 1
    /// exactly), σ(+big) = 1 and σ(−big) = 0 (exp underflows, 1+0 = 1) — so
    /// feeding those pins the pairing, the indexing and the multiply with no
    /// tolerance at all. The +big case is the valuable one: it asserts the
    /// kernel leaves values ALONE when the gate is open, which a scaling bug
    /// would break while still passing a σ(0) test.
    #[test]
    fn attn_out_gate_is_exact_at_saturation_and_zero() {
        let mut seed = 0x7A11_0000u32;
        let attn: Vec<f32> = (0..256).map(|_| lcg(&mut seed)).collect();
        let gate: Vec<f32> = (0..256)
            .map(|i| match i % 3 { 0 => 0.0, 1 => 100.0, _ => -100.0 })
            .collect();
        let got = gpu_attn_gate(&attn, &gate);
        let mut want = attn.clone();
        rf::attn_out_gate(&mut want, &gate);
        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "slot {i}: reference {a} vs gpu {b}");
        }
    }

    /// The BOUNDED half — and the number in it was NOT picked, because when I
    /// picked one I got it wrong.
    ///
    /// I asserted 2 ulp here first, on the strength of the conv kernel's
    /// measured silu spread. It FAILED at 3 ulp, and the reason is structural,
    /// not a fluke: σ(g) = 1/(1+exp(−g)) rounds three more times AFTER exp —
    /// the add, the reciprocal, and the product — so a one-ulp exp error can
    /// surface as three. conv's silu happened to land inside 2 and I
    /// generalised from it.
    ///
    /// Bumping 2 to 3 would have been exactly the quiet gate-widening this lane
    /// exists to prevent, so the bound is measured instead: perturb exp by one
    /// rounding inside the reference's own σ, see how far the result moves, and
    /// require the GPU inside 4x that. Then the bar tracks the arithmetic
    /// rather than the last number that happened to pass.
    #[test]
    fn attn_out_gate_bounded_by_sigmoids_measured_floor() {
        let mut seed = 0x7A11_0001u32;
        let attn: Vec<f32> = (0..1024).map(|_| lcg(&mut seed) * 4.0).collect();
        let gate: Vec<f32> = (0..1024).map(|_| lcg(&mut seed) * 8.0).collect();
        let got = gpu_attn_gate(&attn, &gate);
        let mut want = attn.clone();
        rf::attn_out_gate(&mut want, &gate);

        // σ with exp moved by one f32 rounding — the only thing the GPU is
        // allowed to be doing differently.
        let nudged: Vec<f32> = attn
            .iter()
            .zip(&gate)
            .map(|(a, g)| {
                // f32::EPSILON (2⁻²³), not 2⁻²⁴: 1.0 + 2⁻²⁴ rounds straight
                // back to 1.0, so the smaller nudge is a no-op and the probe
                // measures nothing. The guard below caught exactly that.
                let e = (-g).exp() * (1.0 + f32::EPSILON);
                a * (1.0 / (1.0 + e))
            })
            .collect();
        let yardstick = want
            .iter()
            .zip(&nudged)
            .fold(0f32, |m, (a, b)| m.max(rel_err(*a, *b)));
        let measured = want.iter().zip(&got).fold(0f32, |m, (a, b)| m.max(rel_err(*a, *b)));
        assert!(yardstick > 0.0, "the one-rounding nudge changed nothing — probe is broken");
        assert!(
            measured <= 4.0 * yardstick,
            "gpu drift {measured:e} exceeds 4x sigmoid's measured one-rounding \
             floor {yardstick:e} — more than exp's last bit can explain"
        );
    }

    /// The out-gate's negative control: the gate is applied ELEMENTWISE and
    /// paired by index, so rotating the gate by one must break the match. A
    /// kernel that read the gate at the wrong offset — the exact hazard, since
    /// the gate rides interleaved in the joint Q projection — would otherwise
    /// look perfect.
    #[test]
    fn attn_out_gate_oracle_sees_a_misaligned_gate() {
        let mut seed = 0x7A11_0002u32;
        let attn: Vec<f32> = (0..256).map(|_| lcg(&mut seed)).collect();
        let gate: Vec<f32> = (0..256).map(|_| lcg(&mut seed) * 4.0).collect();
        let got = gpu_attn_gate(&attn, &gate);
        let mut rotated: Vec<f32> = gate[1..].to_vec();
        rotated.push(gate[0]);
        let mut want = attn.clone();
        rf::attn_out_gate(&mut want, &rotated);
        assert!(
            want.iter().zip(&got).any(|(a, b)| a.to_bits() != b.to_bits()),
            "a rotated gate must change the output, or the pairing is untested"
        );
    }

    /// Run gated_output_norm (one threadgroup per V head).
    fn gpu_gated_norm(d: rf::DeltaDims, o: &[f32], w: &[f32], z: &[f32], eps: f32) -> Vec<f32> {
        let device = Device::system_default().expect("Metal device required");
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(&gpu::shader_source(128), &opts)
            .expect("kernels.metal compiles");
        let f = lib.get_function("gated_output_norm", None).expect("gated_output_norm");
        let pipe = device.new_compute_pipeline_state_with_function(&f).expect("pipeline");
        let queue = device.new_command_queue();
        let shared = MTLResourceOptions::StorageModeShared;
        let mk = |v: &[f32]| device.new_buffer_with_data(
            v.as_ptr() as *const _, std::mem::size_of_val(v) as u64, shared);
        let (ob, wb, zb) = (mk(o), mk(w), mk(z));
        let dim = d.d_state as u32;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&ob), 0);
        enc.set_buffer(1, Some(&wb), 0);
        enc.set_buffer(2, Some(&zb), 0);
        enc.set_bytes(3, 4, &dim as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(d.n_v_heads as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        unsafe { std::slice::from_raw_parts(ob.contents() as *const f32, o.len()) }.to_vec()
    }

    fn norm_dims() -> rf::DeltaDims {
        rf::DeltaDims { d_state: 128, n_v_heads: 4, n_k_heads: 2, d_conv: 4 }
    }

    /// The EXACT half for the gated norm, built by removing BOTH inexact
    /// ingredients at once rather than tolerating them:
    ///   * integer o and w keep Σo² exactly representable, so the f32 tree sum
    ///     and the reference's f64 sum agree bit-for-bit (and dividing by a
    ///     power-of-two head width is exact);
    ///   * z = +100 makes silu(z) = z EXACTLY (exp(−100) underflows, 1+0 = 1,
    ///     σ = 1), so the gate contributes a real multiply rather than a
    ///     transcendental — and unlike z = 0 it does not collapse the output to
    ///     zeros, which would have made the test vacuous.
    /// What remains pinned exactly is the whole operation order: mean, one
    /// reciprocal root, and the (o·scale)·w association the reference uses.
    #[test]
    fn gated_output_norm_matches_reference_where_the_sums_agree() {
        let d = norm_dims();
        let (s, eps) = (d.d_state, 1e-5f32);
        let mut seed = 0xBEEF_0001u32;
        let o: Vec<f32> = (0..s * d.n_v_heads).map(|_| (lcg(&mut seed) * 8.0).round()).collect();
        let w: Vec<f32> = (0..s).map(|_| (lcg(&mut seed) * 4.0).round()).collect();
        let z: Vec<f32> = vec![100.0; s * d.n_v_heads];
        let got = gpu_gated_norm(d, &o, &w, &z, eps);
        let mut want = o.clone();
        rf::gated_output_norm(&d, &mut want, &w, &z, eps);
        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "slot {i}: reference {a} vs gpu {b}");
        }
    }

    /// The BOUNDED half. Two inexact ingredients now, so the yardstick is the
    /// larger of the two measured floors: the f64→f32 accumulation gap (probed
    /// on the reference itself) and one σ rounding. Same rule as everywhere in
    /// this lane — the tolerance may only ever cover what was proven separately
    /// above, and it is measured rather than picked.
    #[test]
    fn gated_output_norm_bounded_by_its_two_measured_floors() {
        let d = norm_dims();
        let (s, eps) = (d.d_state, 1e-5f32);
        let mut seed = 0xBEEF_0002u32;
        let o: Vec<f32> = (0..s * d.n_v_heads).map(|_| lcg(&mut seed)).collect();
        let w: Vec<f32> = (0..s).map(|_| lcg(&mut seed)).collect();
        let z: Vec<f32> = (0..s * d.n_v_heads).map(|_| lcg(&mut seed) * 4.0).collect();
        let got = gpu_gated_norm(d, &o, &w, &z, eps);

        let mut want = o.clone();
        rf::gated_output_norm(&d, &mut want, &w, &z, eps);

        // The accumulation probe: the same maths with the mean narrowed to f32.
        let mut narrowed = o.clone();
        for h in 0..d.n_v_heads {
            let oh = &mut narrowed[h * s..(h + 1) * s];
            let mean: f32 = oh.iter().map(|&v| v * v).sum::<f32>() / s as f32;
            let scale = 1.0 / (mean + eps).sqrt();
            for i in 0..s {
                oh[i] = oh[i] * scale * w[i] * rf::silu(z[h * s + i]);
            }
        }
        let accum_gap = want
            .iter()
            .zip(&narrowed)
            .fold(0f32, |m, (a, b)| m.max(rel_err(*a, *b)));
        // One f32 rounding is 2⁻²⁴ relative; σ costs about one.
        let sigma_floor = (2f32).powi(-24);
        let yardstick = accum_gap.max(sigma_floor);
        let measured = want.iter().zip(&got).fold(0f32, |m, (a, b)| m.max(rel_err(*a, *b)));
        assert!(accum_gap > 0.0, "narrowing the mean changed nothing — probe is broken");
        assert!(
            measured <= 4.0 * yardstick,
            "gpu drift {measured:e} exceeds 4x the measured floor {yardstick:e} \
             (accumulation {accum_gap:e}, sigma {sigma_floor:e})"
        );
    }

    /// The gated norm's negative control. `w` is [S] and SHARED by every head,
    /// so it must be indexed WITHIN the head — a kernel that walked it by
    /// global position would produce head 0 correctly and every later head
    /// wrong. Rotating w must break the match; that it does is what makes the
    /// exact test above evidence about indexing and not just about arithmetic.
    #[test]
    fn gated_output_norm_oracle_sees_a_misaligned_weight() {
        let d = norm_dims();
        let (s, eps) = (d.d_state, 1e-5f32);
        let mut seed = 0xBEEF_0003u32;
        let o: Vec<f32> = (0..s * d.n_v_heads).map(|_| lcg(&mut seed)).collect();
        let w: Vec<f32> = (0..s).map(|_| lcg(&mut seed)).collect();
        let z: Vec<f32> = (0..s * d.n_v_heads).map(|_| lcg(&mut seed) * 4.0).collect();
        let got = gpu_gated_norm(d, &o, &w, &z, eps);
        let mut rotated: Vec<f32> = w[1..].to_vec();
        rotated.push(w[0]);
        let mut want = o.clone();
        rf::gated_output_norm(&d, &mut want, &rotated, &z, eps);
        assert!(
            want.iter().zip(&got).any(|(a, b)| a.to_bits() != b.to_bits()),
            "a rotated ssm_norm weight must change the output, or w's indexing is untested"
        );
    }

    /// (a) of the MRoPE ruling: the equivalence PINNED, not remembered.
    ///
    /// Both theta sequences are transcribed from ggml (ops.cpp
    /// ggml_rope_cache_init and ggml_mrope_cache_init, read on this box, not
    /// recalled) — only the theta each pair uses, since that is the entire
    /// claim; rope_yarn is common to both paths and would only add noise.
    ///
    /// For a text batch all four position components carry the SAME position,
    /// so theta_t/h/w/e start equal, and the loop multiplies all four by
    /// theta_scale every pair — they can never diverge, so the sector select is
    /// a no-op. That is why no sectioned kernel is needed.
    ///
    /// The negative control is in the same test on purpose: flipping
    /// indep_sects (llama.cpp's vision path, where the thetas ARE reset at
    /// section boundaries) must make the sequences differ. Without it, a
    /// transcription that accidentally computed plain rope twice would "prove"
    /// the equivalence while checking nothing.
    #[test]
    fn mrope_degenerates_to_rope_for_text_batches() {
        // ops.cpp: theta_scale = powf(freq_base, -2/n_dims).
        let (ne0, freq_base) = (256usize, 1.0e6f32);
        let theta_scale = freq_base.powf(-2.0 / ne0 as f32);
        let sections = [11usize, 11, 10, 0];

        // ggml_rope_cache_init: one theta, scaled once per pair.
        let rope_thetas = |base: f32| -> Vec<f32> {
            let mut theta = base;
            (0..ne0 / 2)
                .map(|_| {
                    let t = theta;
                    theta *= theta_scale;
                    t
                })
                .collect()
        };

        // ggml_mrope_cache_init, non-interleaved branch.
        let mrope_thetas = |bases: [f32; 4], indep_sects: bool| -> Vec<f32> {
            let sect_dims: usize = sections.iter().sum();
            let sec_w = sections[1] + sections[0];
            let sec_e = sections[2] + sec_w;
            let mut th = bases;
            (0..ne0 / 2)
                .map(|pair| {
                    let sector = pair % sect_dims;
                    if indep_sects {
                        if sector == 0 {
                            th[0] = bases[0];
                        } else if sector == sections[0] {
                            th[1] = bases[1];
                        } else if sector == sec_w {
                            th[2] = bases[2];
                        } else if sector == sec_e {
                            th[3] = bases[3];
                        }
                    }
                    let t = if sector >= sections[0] && sector < sec_w {
                        th[1]
                    } else if sector >= sec_w && sector < sec_w + sections[2] {
                        th[2]
                    } else if sector >= sec_w + sections[2] {
                        th[3]
                    } else {
                        th[0]
                    };
                    for x in th.iter_mut() {
                        *x *= theta_scale;
                    }
                    t
                })
                .collect()
        };

        let mut differing_pairs = 0;
        for pos in 0..6u32 {
            let base = pos as f32;
            // A TEXT batch: one position broadcast into all four components.
            let plain = rope_thetas(base);
            let m = mrope_thetas([base; 4], false);
            for (i, (a, b)) in plain.iter().zip(&m).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "pos {pos} pair {i}: plain rope {a} vs mrope {b} — the equivalence this \
                     lane relies on to ship NO sectioned kernel does not hold"
                );
            }
            // The control: the vision path must NOT agree, or this proves nothing.
            let vision = mrope_thetas([base; 4], true);
            differing_pairs += plain.iter().zip(&vision).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        }
        assert!(
            differing_pairs > 0,
            "indep_sects made no difference — the probe cannot see a sectioned rope, \
             so the equivalence above is not evidence"
        );
    }

    /// (b) of the MRoPE ruling: the precondition is refused BY NAME. A
    /// vision-capable checkpoint reaches the same rope path and would be
    /// silently rotated as text — the only way this finding can bite.
    #[test]
    fn vision_section_layout_is_refused_by_name() {
        let mut meta = crate::lowmem::gguf::Qwen35Meta {
            trunk_layers: 64,
            nextn_layers: 1,
            full_attention_interval: 4,
            is_recurrent: (0..64).map(|i| (i + 1) % 4 != 0).collect(),
            d_conv: 4,
            d_state: 128,
            n_group: 16,
            dt_rank: 48,
            d_inner: 6144,
            rope_sections: [11, 11, 10, 0], // the real 27B layout
            conv_state_elems: 30_720,
            delta_state_elems: 786_432,
        };
        assert!(super::Qwen35Layout::check_rope_sections(&meta).is_ok(), "the real 27B layout must pass");

        meta.rope_sections = [11, 11, 10, 4];
        let err = super::Qwen35Layout::check_rope_sections(&meta).unwrap_err();
        assert!(err.contains("vision"), "the refusal must name why: {err}");

        meta.rope_sections = [0, 0, 0, 0];
        assert!(super::Qwen35Layout::check_rope_sections(&meta).is_err(), "all-zero sections must refuse");
    }

    /// Run l2norm_rows over `rows` (one threadgroup per row) and read them back.
    fn gpu_l2norm(rows: &[Vec<f32>], eps: f32) -> Vec<Vec<f32>> {
        let device = Device::system_default().expect("Metal device required");
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(&gpu::shader_source(128), &opts)
            .expect("kernels.metal compiles");
        let f = lib.get_function("l2norm_rows", None).expect("l2norm_rows");
        let pipe = device.new_compute_pipeline_state_with_function(&f).expect("pipeline");
        let queue = device.new_command_queue();

        let dim = rows[0].len();
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let buf = device.new_buffer_with_data(
            flat.as_ptr() as *const _,
            std::mem::size_of_val(&flat[..]) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&buf), 0);
        let d32 = dim as u32;
        enc.set_bytes(1, 4, &d32 as *const u32 as *const _);
        enc.set_bytes(2, 4, &eps as *const f32 as *const _);
        // One threadgroup per row, NORM_TG threads wide — the kernel strides.
        enc.dispatch_thread_groups(MTLSize::new(rows.len() as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let out = unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, flat.len()) };
        out.chunks(dim).map(|c| c.to_vec()).collect()
    }

    /// The EXACT half. The only thing that can differ between this kernel and
    /// the reference is the accumulation of Σx² (f32 tree here, f64 sequential
    /// there), so this test removes that variable instead of tolerating it:
    /// with small-integer inputs every partial sum is exactly representable, so
    /// both accumulations agree BIT-FOR-BIT in any order. Everything else — the
    /// eps clamp, the single reciprocal, the scaling, the row striding — is
    /// then pinned exactly, with no tolerance to hide in.
    ///
    /// (√ is safe to compare: rounding a f64 square root to f32 gives the
    /// correctly-rounded f32 result, because 53 ≥ 2·24+2, so the reference's
    /// f64-then-narrow and the GPU's native f32 √ cannot disagree here.)
    #[test]
    fn l2norm_matches_reference_where_the_sums_agree() {
        let dim = 128; // the real d_state
        let eps = 1e-6f32;
        let mut seed = 0xC0FF_EE11u32;
        let rows: Vec<Vec<f32>> = (0..6)
            .map(|_| (0..dim).map(|_| (lcg(&mut seed) * 16.0).round()).collect())
            .collect();
        let got = gpu_l2norm(&rows, eps);
        for (r, row) in rows.iter().enumerate() {
            let mut want = row.clone();
            rf::l2_norm(&mut want, eps);
            for (i, (a, b)) in want.iter().zip(&got[r]).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "row {r} slot {i}: reference {a} vs gpu {b} — on exactly-summable \
                     input this kernel has no excuse for a difference"
                );
            }
        }
    }

    /// The eps clamp, pinned on the one input that can expose it. A zero row is
    /// reachable in practice (split_qkv l2-normalises per K head, and a head
    /// whose conv output is all zeros produces one), and without the clamp the
    /// kernel computes 1/0 = inf and then 0·inf = NaN while the reference
    /// returns zeros. No random test data ever generates this row, which is
    /// exactly why it gets its own test rather than a seed.
    #[test]
    fn l2norm_zero_row_is_zero_not_nan() {
        let dim = 128;
        let eps = 1e-6f32;
        let rows = vec![vec![0f32; dim], vec![1f32; dim]];
        let got = gpu_l2norm(&rows, eps);
        assert!(
            got[0].iter().all(|v| v.is_finite()),
            "zero row produced non-finite values — the eps clamp is missing or wrong"
        );
        let mut want = rows[0].clone();
        rf::l2_norm(&mut want, eps);
        for (i, (a, b)) in want.iter().zip(&got[0]).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "zero row slot {i}");
        }
        // The neighbouring row must still be normalised normally — a clamp that
        // fires on every row would also pass the check above.
        let n = (dim as f32).sqrt();
        assert!(
            (got[1][0] - 1.0 / n).abs() < 1e-6,
            "the clamp must not fire on an ordinary row: got {}",
            got[1][0]
        );
    }

    /// The BOUNDED half, calibrated the same way as the delta step: the f32-vs-
    /// f64 accumulation gap is measured on the reference itself (same maths,
    /// sum narrowed to f32) and the GPU is required to sit inside a small
    /// multiple of it. The GPU sums in a different ORDER as well (a simd tree,
    /// not a sequential walk), which is why the allowance is 4x and not 1x.
    #[test]
    fn l2norm_bounded_by_the_measured_accumulation_gap() {
        let dim = 128;
        let eps = 1e-6f32;
        let mut seed = 0x1234_ABCDu32;
        let rows: Vec<Vec<f32>> = (0..6)
            .map(|_| (0..dim).map(|_| lcg(&mut seed)).collect())
            .collect();
        let got = gpu_l2norm(&rows, eps);

        let f32_sum_variant = |row: &[f32]| -> Vec<f32> {
            let sum: f32 = row.iter().map(|&v| v * v).sum();
            let scale = 1.0 / sum.sqrt().max(eps);
            row.iter().map(|v| v * scale).collect()
        };
        let (mut worst_gpu, mut yardstick) = (0f32, 0f32);
        for (r, row) in rows.iter().enumerate() {
            let mut want = row.clone();
            rf::l2_norm(&mut want, eps);
            let narrowed = f32_sum_variant(row);
            for i in 0..dim {
                worst_gpu = worst_gpu.max(rel_err(want[i], got[r][i]));
                yardstick = yardstick.max(rel_err(want[i], narrowed[i]));
            }
        }
        assert!(yardstick > 0.0, "narrowing the sum to f32 changed nothing — probe is broken");
        assert!(
            worst_gpu <= 4.0 * yardstick,
            "gpu drift {worst_gpu:e} exceeds 4x the measured f64→f32 accumulation gap \
             {yardstick:e} — more than the missing f64 can explain"
        );
    }

    /// The comparison must be able to see a TRANSPOSED state. The layout is
    /// s[i + j*S + h*S*S] with i the contraction index; a kernel that swapped
    /// i and j would produce finite, plausible numbers. Feeding the reference a
    /// per-head transposed state has to break the match — if it does not, the
    /// oracle is not testing the layout at all.
    #[test]
    fn delta_oracle_sees_a_transposed_state() {
        let d = dims();
        let mut seed = 0x5EED_1234u32;
        let (state0, q, k, v, g, beta) = delta_inputs(d, 1, &mut seed, false);
        let (gpu_outs, _) = gpu_delta(d, &state0, &q, &k, &v, &g, &beta);

        let s = d.d_state;
        let mut transposed = state0.clone();
        for h in 0..d.n_v_heads {
            for j in 0..s {
                for i in 0..s {
                    transposed[h * s * s + j * s + i] = state0[h * s * s + i * s + j];
                }
            }
        }
        let want = rf::delta_decode_step(&d, &mut transposed, &q[0], &k[0], &v[0], &g[0], &beta[0]);
        assert!(
            want.iter().zip(&gpu_outs[0]).any(|(a, b)| a.to_bits() != b.to_bits()),
            "a transposed state must change the output, or the layout is untested"
        );
    }
}
