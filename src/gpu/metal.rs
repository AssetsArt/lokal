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
    /// Leading dims that rotate; the rest of each head passes through. Equals
    /// head_dim on every architecture except qwen35 (rope.dimension_count 64
    /// inside head_dim 256).
    ///
    /// THIS FIELD'S ABSENCE HERE WAS THE e61c260 REGRESSION. kernels.metal
    /// gained it and forward.rs mirrored it; this copy did not, and Metal
    /// matches these by LAYOUT, so the kernel read it from uninitialised memory
    /// past the struct and RoPE rotated the wrong span — fluent, wrong output
    /// on every GGUF model under -b metal. `mirror_structs_match_the_metal_source`
    /// now fails if any mirror drifts from kernels.metal again.
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
    // qwen35's gated-deltanet block — shared kernel source with lowmem, so the
    // two engines dispatch the SAME code and byte-identity is achievable.
    ssm_conv_decode: ComputePipelineState,
    /// The chunk-wide conv pair: one dispatch for every (token, channel), then
    /// one that rolls the window forward by the whole chunk.
    ssm_conv_prefill: ComputePipelineState,
    ssm_conv_roll: ComputePipelineState,
    delta_decode_step: ComputePipelineState,
    delta_gates: ComputePipelineState,
    l2norm_rows: ComputePipelineState,
    gated_output_norm: ComputePipelineState,
    split_q_gate: ComputePipelineState,
    attn_out_gate: ComputePipelineState,
}

/// Cached positions per flash-decoding window — must match ATTN_SPLIT in kernels.metal.
pub(crate) const ATTN_SPLIT: usize = 128;
/// Threads per decode-attention threadgroup — must match DEC_TG in kernels.metal.
pub(crate) const DEC_TG: usize = 128;
/// Output dims one decode-attention thread accumulates — must match MAX_DEC_DPT in
/// kernels.metal. 1 while head_dim <= DEC_TG; qwen35's 256 needs 2.
pub(crate) const MAX_DEC_DPT: usize = 2;
/// The largest head_dim the fused decode path can serve. Above it the decode
/// attention kernel has no thread geometry and the caller must fall back to the
/// prefill-shaped encoder — which is what made qwen35 decode cost a prefill per
/// token before this constant existed.
pub(crate) const DEC_MAX_HD: usize = DEC_TG * MAX_DEC_DPT;
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
    pub(crate) deltanet_layout: Option<DeltaNetLayout>,
    /// The deltanet geometry the kernels take as parameters. Stored as
    /// DeltaDims (Copy, and what both the kernels and deltanet_ref speak)
    /// rather than the raw meta, exactly as the lowmem engine does.
    deltanet_dims: Option<crate::deltanet_ref::DeltaDims>,
    /// True geometry, derived from the checkpoint (qwen3 violates the
    /// hidden/n_heads identity, so cfg.head_dim()/kv_dim() are never read on
    /// hot paths — these are).
    dims: crate::lowmem::Dims,
}


/// lowmem's Dims constructor is module-private; same three lines, same reason
/// as the WindowCfg mirror (fields are pub, the formula is fixed).
fn dims_of(
    cfg: &ModelConfig,
    head_dim: Option<usize>,
    joint_q_gate: bool,
    rot_dim: Option<usize>,
) -> crate::lowmem::Dims {
    let hd = head_dim.unwrap_or_else(|| cfg.head_dim());
    let q_dim = cfg.num_attention_heads * hd;
    crate::lowmem::Dims {
        hidden: cfg.hidden_size,
        head_dim: hd,
        q_dim,
        kv_dim: cfg.num_key_value_heads * hd,
        // qwen35 projects Q and the output gate jointly; everyone else does not.
        q_proj_dim: if joint_q_gate { 2 * q_dim } else { q_dim },
        rot_dim: rot_dim.unwrap_or(hd),
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
    /// What sits between the two norms. Dense checkpoints are `Full` on every
    /// layer; qwen35 alternates — `Linear` on the gated-deltanet blocks, `Full`
    /// on one in `full_attention_interval`. The FFN triple below is shared
    /// because both kinds genuinely carry it.
    ///
    /// Same shape as the lowmem side's `AttnWeights` (src/lowmem/mod.rs), and
    /// deliberately so: two engines walking the same checkpoint should disagree
    /// about scheduling, not about what a layer IS.
    attn: QuantAttn,
    gate_proj: QuantLinear,
    up_proj: QuantLinear,
    down_proj: QuantLinear,
}

/// The two shapes a trunk layer can take on the quant path. An enum rather than
/// a bag of Options because the arms share no attention tensor at all: a linear
/// block has no q/k/v/o and an attention block has no conv or recurrent state.
pub(crate) enum QuantAttn {
    Full(Box<QuantFullAttn>),
    Linear(Box<QuantLinearAttn>),
}

pub(crate) struct QuantFullAttn {
    /// qwen3's per-head q/k RMSNorm weights (f16), pre-RoPE. None elsewhere.
    q_norm: Option<Buffer>,
    k_norm: Option<Buffer>,
    /// On qwen35 this projects Q and the output gate JOINTLY — out_dim is
    /// 2·n_heads·head_dim, `[q(hd)|gate(hd)]` interleaved per head.
    q_proj: QuantLinear,
    k_proj: QuantLinear,
    v_proj: QuantLinear,
    o_proj: QuantLinear,
}

/// qwen35's gated-deltanet block. Roles transcribed from llama.cpp
/// (src/models/qwen35.cpp build_linear_attn); mirrors lowmem's `LinearAttn`.
pub(crate) struct QuantLinearAttn {
    /// hidden -> conv_channels (2·n_group·d_state + d_inner).
    qkv: QuantLinear,
    /// hidden -> d_inner; the `z` that gates the normalised output.
    z_gate: QuantLinear,
    /// d_inner -> hidden.
    out: QuantLinear,
    /// hidden -> n_v_heads, the delta gate's pre-activation.
    alpha: QuantLinear,
    /// hidden -> n_v_heads. THE SIGMOID IS THE KERNEL'S JOB (delta_gates):
    /// qwen35.cpp:366 activates this projection and deltanet_ref's
    /// delta_decode_step takes beta already activated.
    beta: QuantLinear,
    /// f32, not f16: these are stored F32 and the kernels read them as
    /// `device const float *` while being gated bit-for-bit against an f32
    /// reference. [channels][d_conv].
    conv1d: Buffer,
    /// [n_v_heads] f32 (ggml's A_NOSCAN).
    a: Buffer,
    /// [n_v_heads] f32.
    dt_bias: Buffer,
    /// [d_state] f32, the output stage's per-head RMSNorm weight.
    ssm_norm: Buffer,
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
/// The gated-deltanet block's working buffers on the metal path.
///
/// Its own buffers, not borrowed ones: conv_channels (6144 on the 2B) is WIDER
/// than hidden (2048) AND wider than q_proj_dim (4096), so nothing existing is
/// big enough. Same reasoning — and the same widths — as lowmem's DeltaScratch.
/// Allocated only when the checkpoint is a deltanet hybrid.
struct DeltaScratch {
    qkv: Buffer,
    z: Buffer,
    alpha: Buffer,
    beta_p: Buffer,
    g: Buffer,
    beta: Buffer,
    conv_out: Buffer,
    dout: Buffer,
}

impl DeltaScratch {
    fn new(d: &Device, dims: crate::deltanet_ref::DeltaDims, chunk: usize) -> Self {
        let (c, inner, hv) = (dims.conv_channels(), dims.d_inner(), dims.n_v_heads);
        Self {
            qkv: f32_buffer(d, chunk * c),
            z: f32_buffer(d, chunk * inner),
            alpha: f32_buffer(d, chunk * hv),
            beta_p: f32_buffer(d, chunk * hv),
            // CHUNK-WIDE, not one token. The per-token loop below still writes
            // one slice at a time and reads the same slice back, so this is
            // byte-identical today — it is what lets the batched kernels that
            // follow address every token of a chunk at once.
            g: f32_buffer(d, chunk * hv),
            beta: f32_buffer(d, chunk * hv),
            conv_out: f32_buffer(d, chunk * c),
            dout: f32_buffer(d, chunk * inner),
        }
    }
}

pub(crate) struct DeltaNetStates {
    /// Index = trunk layer id. None = full-attention layer (KV lives there instead).
    pub layers: Vec<Option<DeltaNetLayerState>>,
    pub conv_elems: usize,
    pub delta_elems: usize,
}

pub(crate) struct DeltaNetLayerState {
    /// Rolling conv history, (d_conv−1)·C f32 — layout [channel][d_conv−1],
    /// oldest first (matches deltanet_ref::conv_step).
    pub conv: Buffer,
    /// Delta-rule state [S, S, H_v] f32 — i the contraction index. The GPU
    /// layout is TRANSPOSED relative to the reference: s[j + i·S + h·S·S] here,
    /// s[i + j·S + h·S·S] in deltanet_ref, so that a thread's column is strided
    /// and adjacent threads read adjacent addresses (kernels.metal
    /// delta_decode_step carries the full note). Allocation, zeroing and reset
    /// are unaffected — zero is layout-invariant — and nothing outside the
    /// kernel and its oracle reads the buffer's interior.
    pub delta: Buffer,
}

/// docs/gguf-design.md §Attention & per-layer state: a trunk layer owns
/// exactly ONE kind of per-layer state — a KV cache slot (attention layers)
/// or a recurrent slot (conv window + delta state, the deltanet layers).
/// `state_schedule` is the ONE place the kind is decided; construction reads
/// it instead of re-deriving from the layout, and `cache[l]` keeps meaning
/// layer l everywhere (stub buffers, never compaction — the recorded rule).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LayerStateKind {
    Kv,
    Recurrent,
}

/// The per-layer state schedule: dense checkpoints are all-Kv; a deltanet
/// hybrid follows its recurrency map. Length is exactly the trunk.
pub(crate) fn state_schedule(
    n_layers: usize,
    layout: Option<&DeltaNetLayout>,
) -> Vec<LayerStateKind> {
    (0..n_layers)
        .map(|l| match layout {
            Some(q) if q.is_recurrent.get(l).copied().unwrap_or(false) => {
                LayerStateKind::Recurrent
            }
            _ => LayerStateKind::Kv,
        })
        .collect()
}

/// Per-layer KV cache length in f16 elements, one entry per trunk layer: the
/// full ring on a Kv layer, a one-element stub on a Recurrent one. The stub is
/// a legal binding that nothing reads — the recurrent layers' state is conv +
/// delta, and their forward branch binds no cache at all. Keeping the slot
/// (never compacting the vector) is what lets `k_cache[l]` mean layer l
/// everywhere; it is also exactly what LowMemSession::new does.
pub(crate) fn kv_cache_elems(
    sched: &[LayerStateKind],
    kv_slots: usize,
    kv_dim: usize,
) -> Vec<usize> {
    sched
        .iter()
        .map(|k| match k {
            LayerStateKind::Recurrent => 1,
            LayerStateKind::Kv => kv_slots * kv_dim,
        })
        .collect()
}

/// What a qwen35-aware engine hands its sessions so they can allocate states:
/// the per-trunk-layer recurrency map plus the two per-layer element counts.
/// The C/D seam — changes go through the lead, never pairwise.
#[derive(Clone)]
pub(crate) struct DeltaNetLayout {
    pub is_recurrent: Vec<bool>,
    pub conv_elems: usize,
    pub delta_elems: usize,
}

impl DeltaNetLayout {

    /// The one meta→layout translation, so no caller re-derives sizes (where
    /// the 17-layer misconception would creep back in): the map is exactly the
    /// TRUNK's is_recurrent — the MTP block is not in the meta's map at all.
    pub fn from_meta(m: &crate::gguf::Qwen35Meta) -> Self {
        Self {
            is_recurrent: m.is_recurrent.clone(),
            conv_elems: m.conv_state_elems,
            delta_elems: m.delta_state_elems,
        }
    }
}

impl MetalEngine {
    /// The deltanet geometry, or None on a non-hybrid checkpoint.
    fn deltanet_dims(&self) -> Option<crate::deltanet_ref::DeltaDims> {
        self.deltanet_dims
    }

    /// THIS engine's per-layer state schedule. Every consumer inside the
    /// backend comes through here, so recurrency is read from the layout in
    /// exactly one place (the T3 rule) and nothing re-derives it from
    /// `is_recurrent` on its own.
    fn state_schedule(&self) -> Vec<LayerStateKind> {
        state_schedule(self.cfg.num_hidden_layers, self.deltanet_layout.as_ref())
    }

    /// Stubbing a recurrent layer's KV cache is sound only because the forward
    /// loop's own branch and the schedule agree layer for layer: a
    /// QuantAttn::Full block binds k_cache[l], a Linear block never does, and
    /// the f16 path has no linear variant at all, so it binds every layer
    /// unconditionally. Checked once per session construction, where a
    /// mismatch is a named panic instead of a GPU write past a one-element
    /// buffer — the failure mode that has no other symptom.
    fn assert_schedule_matches_graph(&self, sched: &[LayerStateKind]) {
        match &self.quant {
            Some(q) => {
                assert_eq!(
                    q.blocks.len(),
                    sched.len(),
                    "quant block count and the state schedule disagree"
                );
                for (l, blk) in q.blocks.iter().enumerate() {
                    let kind = match &blk.attn {
                        QuantAttn::Full(_) => LayerStateKind::Kv,
                        QuantAttn::Linear(_) => LayerStateKind::Recurrent,
                    };
                    assert_eq!(
                        kind, sched[l],
                        "layer {l}: block kind and the state schedule disagree — \
                         a stubbed KV cache would be bound by the attention half"
                    );
                }
            }
            None => assert!(
                self.deltanet_layout.is_none(),
                "deltanet layout on the f16 path — every f16 layer binds its KV cache"
            ),
        }
    }
}

impl DeltaNetStates {
    /// Zero-initialized states — zeroing is load-bearing: an empty conv
    /// history contributes silence and an empty delta state attends nothing.
    pub fn new(device: &Device, layout: &DeltaNetLayout) -> Self {
        let layers = layout
            .is_recurrent
            .iter()
            .map(|&r| {
                r.then(|| DeltaNetLayerState {
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
        // Construction-time seam checks (docs/gguf-design.md §FFN/§Norm): the
        // fused swiglu and rmsnorm pipelines are the only forms compiled in.
        match model.cfg.activation()? {
            crate::config::Activation::SwiGLU => {}
        }
        match model.cfg.norm_type() {
            crate::config::NormType::RmsNormPre => {}
        }
        let device = Device::system_default().ok_or("no Metal-capable GPU found")?;
        let queue = device.new_command_queue();

        // Kernels are compiled at runtime — edit kernels.metal and just cargo run again.
        let lib = device
            .new_library_with_source(&shader_source(model.kv_dim), &CompileOptions::new())
            .map_err(|e| format!("failed to compile kernels.metal: {e}"))?;
        let dims = dims_of(&model.cfg, Some(model.head_dim), false, None);
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
            ssm_conv_decode: pipe("ssm_conv_decode")?,
            ssm_conv_prefill: pipe("ssm_conv_prefill")?,
            ssm_conv_roll: pipe("ssm_conv_roll")?,
            delta_decode_step: pipe("delta_decode_step")?,
            delta_gates: pipe("delta_gates")?,
            l2norm_rows: pipe("l2norm_rows")?,
            gated_output_norm: pipe("gated_output_norm")?,
            split_q_gate: pipe("split_q_gate")?,
            attn_out_gate: pipe("attn_out_gate")?,
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
                gate_proj: lin(&device, &b.ffn.dense().gate_proj),
                up_proj: lin(&device, &b.ffn.dense().up_proj),
                down_proj: lin(&device, &b.ffn.dense().down_proj),
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
            deltanet_layout: None,
            deltanet_dims: None,
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
            rot_dim: hd as u32,
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
    // deltanet_kernel_oracle::mrope_degenerates_to_rope_for_text_batches (with a
    // vision-path negative control), and vision-capable checkpoints are refused
    // by name in Qwen35Meta::check_rope_sections. A sectioned kernel's best
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
        let rot = self.dims.rot_dim; // == hd except on qwen35 (partial RoPE)
        let p = RopeQkParams {
            head_dim: hd as u32,
            n_q_heads: self.cfg.num_attention_heads as u32,
            n_kv_heads: self.cfg.num_key_value_heads as u32,
            pos: pos as u32,
            theta: self.cfg.rope_theta,
            rot_dim: rot as u32,
        };
        enc.set_compute_pipeline_state(&self.pipes.rope_qk_decode);
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k_cache), kv_byte_off);
        enc.set_bytes(2, size_of::<RopeQkParams>() as u64, &p as *const _ as *const _);
        // Grid sized by rot_dim/2: the tail threads must not exist, or they
        // would rotate exactly the dims that have to pass through.
        dispatch_grid(enc, (self.cfg.num_attention_heads + self.cfg.num_key_value_heads) * rot / 2);
    }

    /// Decode-only attention (n_rows = 1): flash-decoding split. Falls back to the
    /// generic kernel via the caller when head_dim > DEC_MAX_HD.
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
            f32s(acc_red_elems(chunk, head_dim)),
            f32s(chunk * (DEC_TG / 32) + chunk),
        ],
    )
}

/// Elements in the decode-attention `acc_red` scratch. The kernel indexes it by
/// (position lane, output dim), so it holds P x head_dim entries: DEC_TG while
/// head_dim <= DEC_TG (P = DEC_TG/head_dim lanes over head_dim dims), head_dim
/// above it (P collapses to 1 and each thread carries several dims). One named
/// rule because Rust allocates this buffer and Metal indexes it — the two must
/// agree, and an under-allocation here is a threadgroup-memory overrun with no
/// symptom. ACC_STRIDE's `| 1` odd stride is mirrored from the shader.
fn acc_red_elems(chunk: usize, head_dim: usize) -> usize {
    DEC_TG.max(head_dim) * (chunk | 1)
}

/// Host-side mirror of the shader's lm_row_bytes, for the probe's byte
/// accounting only — a GB/s number is meaningless without the exact byte count,
/// and reading it off the shader is not possible from Rust. Kept next to nothing
/// else so it cannot be mistaken for a sizing rule anything dispatches against.
fn lm_row_bytes_host(sel: u32, in_dim: u32) -> u64 {
    let n32 = (in_dim / 32) as u64;
    let n256 = (in_dim / 256) as u64;
    match sel {
        2 => n32 * 34, 3 => n32 * 18, 4 => n256 * 144, 5 => n256 * 210,
        6 => n256 * 176, 7 => n32 * 22, 8 => n256 * 84, 9 => n256 * 110,
        10 => n32 * 18, 11 => n256 * 136, 12 => n256 * 98, 13 => n256 * 110,
        14 => n256 * 66, 15 => n256 * 74, 16 => n256 * 82, 17 => n256 * 50,
        18 => n256 * 56,
        _ => (in_dim as u64) * 2,
    }
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
        match cfg.activation()? {
            crate::config::Activation::SwiGLU => {}
        }
        match cfg.norm_type() {
            crate::config::NormType::RmsNormPre => {}
        }
        use std::collections::HashMap;
        let device = Device::system_default().ok_or("no Metal-capable GPU found")?;
        let queue = device.new_command_queue();
        let mut source = LowMemSource::open(path)?;
        source.make_gpu_views(&device);
        let dn_meta = source.qwen35();
        let dims = dims_of(
            &cfg,
            source.head_dim(),
            dn_meta.is_some(),
            dn_meta.as_ref().map(|m| 2 * m.rope_sections.iter().sum::<usize>()),
        );

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
        let q_dim = dims.q_proj_dim; // joint Q+gate on qwen35; == q_dim elsewhere
        // qwen35's recurrency map decides which shape each trunk layer takes.
        // None on every other checkpoint, which then takes the Full arm for all
        // layers exactly as before.
        let q35 = source.qwen35();
        let f32_buf = |name: &str| -> crate::Result<Buffer> {
            Ok(f32_buffer_from(&device, &source.read_f32(name)?))
        };
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let recurrent = q35.as_ref().is_some_and(|m| m.is_recurrent[i]);
            let attn = if recurrent {
                let m = q35.as_ref().expect("recurrent implies qwen35 meta");
                let conv_channels = 2 * m.n_group * m.d_state + m.d_inner;
                QuantAttn::Linear(Box::new(QuantLinearAttn {
                    qkv: qlin(&source, &device, &zero_bias, &format!("{p}.gguf.attn_qkv"), h, conv_channels)?,
                    z_gate: qlin(&source, &device, &zero_bias, &format!("{p}.gguf.attn_gate"), h, m.d_inner)?,
                    out: qlin(&source, &device, &zero_bias, &format!("{p}.gguf.ssm_out"), m.d_inner, h)?,
                    alpha: qlin(&source, &device, &zero_bias, &format!("{p}.gguf.ssm_alpha"), h, m.dt_rank)?,
                    beta: qlin(&source, &device, &zero_bias, &format!("{p}.gguf.ssm_beta"), h, m.dt_rank)?,
                    conv1d: f32_buf(&format!("{p}.gguf.ssm_conv1d.weight"))?,
                    a: f32_buf(&format!("{p}.gguf.ssm_a"))?,
                    dt_bias: f32_buf(&format!("{p}.gguf.ssm_dt.bias"))?,
                    ssm_norm: f32_buf(&format!("{p}.gguf.ssm_norm.weight"))?,
                }))
            } else {
                QuantAttn::Full(Box::new(QuantFullAttn {
                    q_proj: qlin(&source, &device, &zero_bias, &format!("{p}.self_attn.q_proj"), h, q_dim)?,
                    k_proj: qlin(&source, &device, &zero_bias, &format!("{p}.self_attn.k_proj"), h, kvd)?,
                    v_proj: qlin(&source, &device, &zero_bias, &format!("{p}.self_attn.v_proj"), h, kvd)?,
                    o_proj: qlin(&source, &device, &zero_bias, &format!("{p}.self_attn.o_proj"), dims.q_dim, h)?,
                    q_norm: match source.has(&format!("{p}.self_attn.q_norm.weight")) {
                        true => Some(f16_buffer(&device, &source.read_f32(&format!("{p}.self_attn.q_norm.weight"))?)),
                        false => None,
                    },
                    k_norm: match source.has(&format!("{p}.self_attn.k_norm.weight")) {
                        true => Some(f16_buffer(&device, &source.read_f32(&format!("{p}.self_attn.k_norm.weight"))?)),
                        false => None,
                    },
                }))
            };
            blocks.push(QuantBlock {
                input_layernorm: norm_buf(&format!("{p}.input_layernorm.weight"))?,
                post_attention_layernorm: norm_buf(&match q35.is_some() {
                    true => format!("{p}.gguf.post_attention_norm.weight"),
                    false => format!("{p}.post_attention_layernorm.weight"),
                })?,
                attn,
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
            deltanet_layout: q35.as_ref().map(DeltaNetLayout::from_meta),
            deltanet_dims: q35.as_ref().map(|m| crate::deltanet_ref::DeltaDims {
                d_state: m.d_state,
                n_v_heads: m.dt_rank,
                n_k_heads: m.n_group,
                d_conv: m.d_conv,
            }),
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

    /// DIAGNOSTIC, env-gated (`LOKAL_MATVEC_PROBE=1`), never on a normal path.
    ///
    /// The quant matvec family reads a uniform 65-70 GB/s on a ~200 GB/s box and
    /// this lane has now falsified two explanations for it from the inside:
    /// packed 4-byte loads did not move lm_head (6.492 -> 6.532 ms), and
    /// read-amplification cannot be it because Q8_0 amplifies 1.0x while Q6_K
    /// amplifies 2.55x and both measure the same GB/s. What no per-type kernel
    /// change can test is the DISPATCH SHAPE itself, which is identical for every
    /// type. So: run the same bytes through the same shape with the dequant cost
    /// removed (probe_shape), and through a flat coalesced stream (probe_linear).
    /// The real matvec runs beside them as the in-run reference, so all three
    /// numbers come from one process, one buffer and one machine state.
    /// SYNTHETIC microbenchmark for a weight type no file on this box carries.
    /// `LOKAL_MATVEC_PROBE=synth:<sel>:<in_dim>:<out_dim>` builds a tensor of
    /// VALID blocks for that selector and times the real matvec over it.
    ///
    /// It exists because unsloth/Qwen3.5-{0.8B,2B}-GGUF have no IQ1 tag at all
    /// (checked: IQ2_M, IQ2_XXS, IQ3_XXS, IQ4_XS and up, no IQ1_*), so the type
    /// that is 86.7% of the 27B IQ1_M step cannot be timed here on real weights.
    /// Every number it prints is labelled SYNTHETIC and supports NO claim about
    /// the 27B — it compares two binaries on identical bytes, nothing more.
    /// Block validity follows the same rules the one-hot gate uses.
    fn run_synth_probe(&self, spec: &str) {
        let parts: Vec<&str> = spec.split(':').collect();
        let (Ok(sel), Ok(in_dim), Ok(out_dim)) = (
            parts.first().unwrap_or(&"").parse::<u32>(),
            parts.get(1).unwrap_or(&"").parse::<usize>(),
            parts.get(2).unwrap_or(&"").parse::<usize>(),
        ) else {
            eprintln!("probe fail stage=spec want=synth:<sel>:<in_dim>:<out_dim> got={spec}");
            return;
        };
        let blk = match sel {
            5 => 210usize,
            17 => 50,
            18 => 56,
            _ => {
                eprintln!("probe fail stage=selector sel={sel} has no synthetic block rule");
                return;
            }
        };
        let sb = in_dim / 256;
        let row_bytes = sb * blk;
        let mut w = vec![0u8; out_dim * row_bytes];
        let mut st = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        for r in 0..out_dim {
            for b in 0..sb {
                let base = r * row_bytes + b * blk;
                for i in 0..blk {
                    w[base + i] = (next() >> 24) as u8;
                }
                let d_bits: u16 = 0x3C00 | ((next() >> 20) as u16 & 0x03FF);
                match sel {
                    5 => {
                        w[base + 208] = (d_bits & 0xFF) as u8;
                        w[base + 209] = (d_bits >> 8) as u8;
                    }
                    17 => {
                        w[base] = (d_bits & 0xFF) as u8;
                        w[base + 1] = (d_bits >> 8) as u8;
                    }
                    _ => w[base + 55] = (w[base + 55] & 0x0F) | 0x30,
                }
            }
        }
        let d = &self.device;
        let precise = CompileOptions::new();
        precise.set_fast_math_enabled(false);
        let Ok(lib) = d.new_library_with_source(&shader_source(self.dims.kv_dim), &precise) else {
            eprintln!("probe fail stage=library");
            return;
        };
        let consts = FunctionConstantValues::new();
        consts.set_constant_value_at_index(&sel as *const u32 as *const _, MTLDataType::UInt, 25);
        let Ok(f) = lib.get_function("matvec", Some(consts)) else {
            eprintln!("probe fail stage=function");
            return;
        };
        let Ok(pipe) = d.new_compute_pipeline_state_with_function(&f) else {
            eprintln!("probe fail stage=pipeline");
            return;
        };
        let wbuf = d.new_buffer_with_data(
            w.as_ptr() as *const _,
            w.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let x = f32_buffer(d, in_dim);
        unsafe {
            let xp = x.contents() as *mut f32;
            for i in 0..in_dim {
                *xp.add(i) = 1.0;
            }
        }
        let bias = f16_empty_buffer(d, out_dim.max(8));
        unsafe { std::ptr::write_bytes(bias.contents() as *mut u8, 0, 2 * out_dim.max(8)) };
        let y = f32_buffer(d, out_dim.max(8));
        let p = MatvecParams { in_dim: in_dim as u32, out_dim: out_dim as u32 };
        let mut ns: Vec<u128> = Vec::new();
        for _ in 0..7 {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&wbuf), 0);
            enc.set_buffer(1, Some(&bias), 0);
            enc.set_buffer(2, Some(&x), 0);
            enc.set_buffer(3, Some(&y), 0);
            enc.set_bytes(4, size_of::<MatvecParams>() as u64, &p as *const _ as *const _);
            dispatch_simdgroup_rows(enc, out_dim as u32);
            enc.end_encoding();
            let t = std::time::Instant::now();
            cb.commit();
            cb.wait_until_completed();
            ns.push(t.elapsed().as_nanos());
        }
        ns.sort_unstable();
        let med = ns[ns.len() / 2];
        eprintln!(
            "probe kind=SYNTHETIC sel={sel} in_dim={in_dim} out_dim={out_dim} ms={:.3} \
             gbps={:.1} bytes={} samples={} note=no-real-file-for-this-type",
            med as f64 / 1e6,
            w.len() as f64 / (med as f64 / 1e9) / 1e9,
            w.len(),
            ns.len()
        );
    }

    fn run_matvec_probe(&self) {
        if let Ok(spec) = std::env::var("LOKAL_MATVEC_PROBE") {
            if let Some(rest) = spec.strip_prefix("synth:") {
                return self.run_synth_probe(rest);
            }
        }
        let Some(q) = &self.quant else {
            eprintln!("probe skip reason=not-a-quant-engine");
            return;
        };
        let l = &q.lm_head;
        let bytes = (l.out_dim as u64) * lm_row_bytes_host(l.sel, l.in_dim);
        let d = &self.device;
        // Own library so the probe cannot perturb the pipelines a real run uses;
        // fast-math off to match the family it is standing in for.
        let precise = CompileOptions::new();
        precise.set_fast_math_enabled(false);
        let lib = match d.new_library_with_source(&shader_source(self.dims.kv_dim), &precise) {
            Ok(v) => v,
            Err(e) => { eprintln!("probe fail stage=library err={e}"); return; }
        };
        let build = |name: &str| -> Option<ComputePipelineState> {
            let consts = FunctionConstantValues::new();
            consts.set_constant_value_at_index(
                &l.sel as *const u32 as *const _, MTLDataType::UInt, 25);
            let f = match lib.get_function(name, Some(consts)) {
                Ok(f) => f,
                Err(e) => { eprintln!("probe fail stage=function name={name} err={e}"); return None; }
            };
            match d.new_compute_pipeline_state_with_function(&f) {
                Ok(p) => Some(p),
                Err(e) => { eprintln!("probe fail stage=pipeline name={name} err={e}"); None }
            }
        };
        let (Some(p_shape), Some(p_linear)) = (build("probe_shape"), build("probe_linear")) else {
            return;
        };
        let x = f32_buffer(d, l.in_dim as usize);
        let y = f32_buffer(d, (l.out_dim as usize).max(1 << 16));
        let params = MatvecParams { in_dim: l.in_dim, out_dim: l.out_dim };
        // Rule 9's spirit: the first pass carries warmup, so time several and
        // report the median, never a single sample.
        let time_one = |kind: &str, run: &dyn Fn(&ComputeCommandEncoderRef)| {
            let mut ns: Vec<u128> = Vec::new();
            for _ in 0..5 {
                let cb = self.queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                run(enc);
                enc.end_encoding();
                let t = std::time::Instant::now();
                cb.commit();
                cb.wait_until_completed();
                ns.push(t.elapsed().as_nanos());
            }
            ns.sort_unstable();
            let med = ns[ns.len() / 2];
            eprintln!(
                "probe kind={kind} ms={:.3} gbps={:.1} bytes={bytes} samples={}",
                med as f64 / 1e6,
                bytes as f64 / (med as f64 / 1e9) / 1e9,
                ns.len()
            );
        };
        time_one("matvec_real", &|enc| {
            self.enc_qmv_probe(enc, &q.pipe(l.sel).matvec, l, &x, &y);
        });
        time_one("probe_shape", &|enc| {
            enc.set_compute_pipeline_state(&p_shape);
            enc.set_buffer(0, Some(&l.w), l.w_off);
            enc.set_buffer(1, Some(&y), 0);
            enc.set_bytes(2, size_of::<MatvecParams>() as u64, &params as *const _ as *const _);
            dispatch_simdgroup_rows(enc, l.out_dim);
        });
        time_one("probe_linear", &|enc| {
            enc.set_compute_pipeline_state(&p_linear);
            enc.set_buffer(0, Some(&l.w), l.w_off);
            enc.set_buffer(1, Some(&y), 0);
            enc.set_bytes(2, size_of::<MatvecParams>() as u64, &params as *const _ as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1024, 1, 1), MTLSize::new(64, 1, 1));
        });
    }

    /// The probe's copy of enc_qmv's binding, so the diagnostic never reaches
    /// into a session's buffers.
    fn enc_qmv_probe(
        &self,
        enc: &ComputeCommandEncoderRef,
        pipe: &ComputePipelineState,
        l: &QuantLinear,
        x: &Buffer,
        y: &Buffer,
    ) {
        let p = MatvecParams { in_dim: l.in_dim, out_dim: l.out_dim };
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&l.w), l.w_off);
        enc.set_buffer(1, Some(&l.bias), 0);
        enc.set_buffer(2, Some(x), 0);
        enc.set_buffer(3, Some(y), 0);
        enc.set_bytes(4, size_of::<MatvecParams>() as u64, &p as *const _ as *const _);
        dispatch_simdgroup_rows(enc, l.out_dim);
    }

    pub(crate) fn raw_session(&self, max_seq: usize) -> MetalSession<'_> {
        // Diagnostic only, and one-shot: see run_matvec_probe.
        static PROBE: std::sync::Once = std::sync::Once::new();
        PROBE.call_once(|| {
            if std::env::var_os("LOKAL_MATVEC_PROBE").is_some() {
                self.run_matvec_probe();
            }
        });
        let cfg = &self.cfg;
        let d = &self.device;
        // Window mode: KV is a ring of cap slots per layer — O(window), not
        // O(context) — exactly lowmem's store layout so the LM_* kernels read
        // it unchanged.
        let kv_slots = match &self.win {
            Some(w) => w.cfg.cap,
            None => max_seq + FLASH_C,
        };
        // A recurrent layer's per-layer state is the conv window plus the
        // delta state (DeltaNetStates, allocated below); it never touches a KV
        // cache, because the quant forward takes the QuantAttn::Linear branch,
        // which binds neither k_cache[l] nor v_cache[l]. A full cap × kv_dim
        // buffer there is RAM nobody ever reads — on qwen35 that is three of
        // every four trunk layers. lowmem has stubbed them since the hybrid
        // shipped (LowMemSession::new); this is the same stub, off the same
        // schedule, so the two backends allocate the same shape.
        //
        // The SLOT is kept (a one-element buffer, still a legal binding)
        // rather than the vector compacted, so `k_cache[l]` keeps meaning
        // layer l at every call site and no second index can drift out of
        // step with the recurrency map.
        let sched = self.state_schedule();
        self.assert_schedule_matches_graph(&sched);
        let elems = kv_cache_elems(&sched, kv_slots, self.dims.kv_dim);
        debug_assert_eq!(elems.len(), cfg.num_hidden_layers);
        let caches =
            elems.iter().map(|&n| f16_empty_buffer(d, n)).collect::<Vec<_>>();
        let v_caches =
            elems.iter().map(|&n| f16_empty_buffer(d, n)).collect::<Vec<_>>();
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
            q: f32_buffer(d, chunk * attn_row_width(cfg.hidden_size, self.dims.q_proj_dim)),
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
                chunk * cfg.hidden_size.max(cfg.intermediate_size).max(self.dims.q_proj_dim),
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
            deltanet: self.deltanet_layout.as_ref().map(|l| DeltaNetStates::new(&self.device, l)),
            ds: self
                .deltanet_dims()
                .map(|dd| DeltaScratch::new(&self.device, dd, PREFILL_CHUNK.min(max_seq))),
            qg: (self.dims.q_proj_dim != self.dims.q_dim).then(|| {
                let chunk = PREFILL_CHUNK.min(max_seq);
                (f32_buffer(&self.device, chunk * self.dims.q_dim),
                 f32_buffer(&self.device, chunk * self.dims.q_dim))
            }),
            k_cache,
            v_cache,
            state: self.state_schedule(),
            kv_base,
            max_seq,
            timing: GpuTiming::from_env(self),
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
    deltanet: Option<DeltaNetStates>,
    /// Deltanet working buffers; None on every non-hybrid checkpoint.
    ds: Option<DeltaScratch>,
    /// Joint Q+gate de-interleaved: (compact q, gate). None unless the
    /// projection is wider than attention (qwen35).
    qg: Option<(Buffer, Buffer)>,
    k_cache: Vec<Buffer>,
    v_cache: Vec<Buffer>,
    /// The per-layer state schedule this session's caches were allocated
    /// against: Kv layers own a real cache, Recurrent layers own a stub. Kept
    /// so the KV write path refuses a stubbed layer by name.
    state: Vec<LayerStateKind>,
    kv_base: u64, // byte offset of this session's slot when the cache is pooled
    max_seq: usize,
    /// GPU phase attribution — `None` unless LOKAL_GPU_TIMING is set, which is
    /// every run that is not this lane's diagnosis. See the timing section.
    timing: Option<Box<GpuTiming>>,
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
        // NOTE: still DEC_TG, not DEC_MAX_HD. The decode attention kernel would serve
        // hd 256 here too, but the rest of this path (enc_qkv above all) has never run
        // at hd > 128 and no f16 checkpoint in the gate table exercises it — lifting
        // it would ship untested surface for no model that exists. The quant path is
        // where qwen35 lives; deferred deliberately, per the lane plan.
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

            // Prefill path (and the rare head_dim > DEC_MAX_HD decode): tiled matmuls.
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
                    rot_dim: hd as u32,
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
                        rot_dim: hd as u32,
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
    /// The deltanet scratch, which every caller below has already established
    /// exists (a Linear arm implies a deltanet checkpoint implies the scratch).
    fn ds_ref(&self) -> &DeltaScratch {
        self.ds.as_ref().expect("a deltanet layer implies deltanet scratch")
    }

    /// De-interleave the joint Q+gate projection, returning the buffer the
    /// attention path should treat as Q. A no-op returning `self.q` on every
    /// checkpoint whose projection is not joint.
    fn enc_split_qg(&self, enc: &ComputeCommandEncoderRef, n: usize, conc: bool) -> &Buffer {
        let Some((qc, gate)) = &self.qg else { return &self.q };
        let e = self.engine;
        let (hd, heads) = (e.dims.head_dim, e.cfg.num_attention_heads);
        #[repr(C)]
        struct QGSplitParams {
            head_dim: u32,
            n_heads: u32,
            n_rows: u32,
        }
        // WAR ORDERING, and the reason this lane's oracle went red at long prompts:
        // the PREVIOUS attention layer READ this same `gate` buffer in
        // enc_apply_qgate. On a concurrent encoder nothing stops this layer's
        // write from landing before that read retires, and the deltanet layers
        // in between are not a barrier — they touch different resources. So the
        // buffers are ordered BEFORE the write, not only after it.
        if conc {
            enc.memory_barrier_with_resources(&[qc, gate]);
        }
        enc.set_compute_pipeline_state(&e.pipes.split_q_gate);
        enc.set_buffer(0, Some(&self.q), 0);
        enc.set_buffer(1, Some(qc), 0);
        enc.set_buffer(2, Some(gate), 0);
        let p = QGSplitParams { head_dim: hd as u32, n_heads: heads as u32, n_rows: n as u32 };
        enc.set_bytes(3, size_of::<QGSplitParams>() as u64, &p as *const _ as *const _);
        dispatch_grid(enc, n * heads * hd);
        if conc {
            enc.memory_barrier_with_resources(&[qc, gate]);
        }
        qc
    }

    /// attn · sigmoid(gate), applied AFTER attention and BEFORE wo
    /// (qwen35.cpp:327-331). A no-op elsewhere.
    fn enc_apply_qgate(&self, enc: &ComputeCommandEncoderRef, n: usize, conc: bool) {
        let Some((_, gate)) = &self.qg else { return };
        let e = self.engine;
        let n_elem = n * e.dims.q_dim;
        enc.set_compute_pipeline_state(&e.pipes.attn_out_gate);
        enc.set_buffer(0, Some(&self.att), 0);
        enc.set_buffer(1, Some(gate), 0);
        let n32 = n_elem as u32;
        enc.set_bytes(2, 4, &n32 as *const u32 as *const _);
        dispatch_grid(enc, n_elem);
        if conc {
            enc.memory_barrier_with_resources(&[&self.att]);
        }
    }

    /// qwen35's gated-deltanet block for `n` already-projected tokens.
    ///
    /// THE BARRIERS ARE THE WHOLE DIFFICULTY OF THIS PORT. lowmem runs this on a
    /// SERIAL encoder and gets ordering for free. Metal's prefill encoder is
    /// CONCURRENT, and this chain is strictly sequential twice over: each stage
    /// reads the previous stage's output, AND both recurrent states are
    /// read-modify-written IN PLACE across the token loop — token t+1's conv
    /// reads the window token t just rolled, and its delta step reads the state
    /// token t just mutated. So there is a barrier between every stage and, the
    /// part that is easy to miss, between every TOKEN.
    ///
    /// A missing barrier here does not crash; it produces output that still
    /// looks like text. The window-mode lane hit exactly this porting lowmem's
    /// attention into this encoder, and only byte-identity on a tiny prompt
    /// caught it. That is why this lane's oracle is metal==lowmem byte-identical
    /// over 4+ runs per side rather than one run of eyeballed text.
    fn enc_delta_block(
        &self,
        enc: &ComputeCommandEncoderRef,
        la: &QuantLinearAttn,
        l: usize,
        n: usize,
        conc: bool,
    ) {
        let e = self.engine;
        let d = e.deltanet_dims().expect("deltanet dims on a deltanet checkpoint");
        let ds = self.ds_ref();
        let st = self.deltanet.as_ref().expect("deltanet states").layers[l]
            .as_ref()
            .expect("every linear layer owns recurrent state");
        let eps = e.cfg.rms_norm_eps;
        let (s_dim, hv, hk) = (d.d_state, d.n_v_heads, d.n_k_heads);
        let (key_dim, inner, c_all) = (s_dim * hk, d.d_inner(), d.conv_channels());
        macro_rules! bar {
            ($($b:expr),+) => { if conc { enc.memory_barrier_with_resources(&[$($b),+]) } };
        }

        // DECODE keeps the per-token chain byte-for-byte. It is one token, so
        // there is nothing to batch, and it is the path that just earned an 11x
        // on lowmem and a 15x here — re-routing it in the lane that rewrites
        // prefill would put both at risk for no gain. The batched path's n = 1
        // case is proven EQUAL to this one as a test (memo §6 T2's constructed
        // bit-exact half), not assumed by shipping through it.
        if n > 1 {
            self.enc_delta_block_chunk(enc, la, l, n, conc);
            return;
        }

        for t in 0..n {
            let (goff, coff, zoff) =
                ((t * hv * 4) as u64, (t * c_all * 4) as u64, (t * inner * 4) as u64);

            enc.set_compute_pipeline_state(&e.pipes.delta_gates);
            enc.set_buffer(0, Some(&ds.alpha), goff);
            enc.set_buffer(1, Some(&ds.beta_p), goff);
            enc.set_buffer(2, Some(&la.a), 0);
            enc.set_buffer(3, Some(&la.dt_bias), 0);
            enc.set_buffer(4, Some(&ds.g), goff);
            enc.set_buffer(5, Some(&ds.beta), goff);
            let hv32 = hv as u32;
            enc.set_bytes(6, 4, &hv32 as *const u32 as *const _);
            // n_tokens. NOT optional: Metal binds by index at runtime, so a
            // buffer this kernel declares and the caller never sets is read from
            // whatever was there — no compile error, no crash, just a guard
            // computed from garbage. That is what this lane shipped for one
            // build, and the cell table is what caught it.
            let one_tok = 1u32;
            enc.set_bytes(7, 4, &one_tok as *const u32 as *const _);
            dispatch_grid(enc, hv);
            bar!(&ds.g, &ds.beta);

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
            let cp = SsmConvParams { channels: c_all as u32, d_conv: d.d_conv as u32 };
            enc.set_bytes(4, size_of::<SsmConvParams>() as u64, &cp as *const _ as *const _);
            dispatch_grid(enc, c_all);
            bar!(&ds.conv_out, &st.conv);

            let s32 = s_dim as u32;
            for off in [0u64, (key_dim * 4) as u64] {
                enc.set_compute_pipeline_state(&e.pipes.l2norm_rows);
                enc.set_buffer(0, Some(&ds.conv_out), coff + off);
                enc.set_bytes(1, 4, &s32 as *const u32 as *const _);
                enc.set_bytes(2, 4, &eps as *const f32 as *const _);
                let no_stride = 0u32; // one token: the grid's y index is 0
                enc.set_bytes(3, 4, &no_stride as *const u32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(hk as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            bar!(&ds.conv_out);

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
            let dp = DeltaStepParams {
                d_state: s32,
                n_v_heads: hv as u32,
                group: (hv / hk) as u32,
            };
            enc.set_bytes(7, size_of::<DeltaStepParams>() as u64, &dp as *const _ as *const _);
            dispatch_grid(enc, s_dim * hv);
            bar!(&ds.dout, &st.delta);

            enc.set_compute_pipeline_state(&e.pipes.gated_output_norm);
            enc.set_buffer(0, Some(&ds.dout), zoff);
            enc.set_buffer(1, Some(&la.ssm_norm), 0);
            enc.set_buffer(2, Some(&ds.z), zoff);
            enc.set_bytes(3, 4, &s32 as *const u32 as *const _);
            enc.set_bytes(4, 4, &eps as *const f32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(hv as u64, 1, 1), MTLSize::new(256, 1, 1));
            // Closing the token: the next iteration reads BOTH states in place.
            bar!(&ds.dout, &st.conv, &st.delta);
        }
    }

    /// A timed step creates hundreds of autoreleased encoder objects, and a
    /// Rust binary has no ambient autorelease pool — without one the diagnosis
    /// mode would grow the process for the length of the run. The untimed path
    /// keeps the exact call it has always had.
    /// The same six stages over a WHOLE prefill chunk. Four of them have no
    /// cross-token dependency at all and collapse to one dispatch each; the
    /// delta rule keeps its per-token loop here and loses it in the next commit.
    ///
    /// Ordering, which is the whole difficulty on the CONCURRENT prefill
    /// encoder: `ssm_conv_roll` WRITES the window `ssm_conv_prefill` READ, so
    /// the barrier between them names `st.conv` — a write-after-read hazard,
    /// the kind that leaves no trace when it fires because every early token
    /// simply convolves against a window from the wrong end of the chunk and
    /// still produces fluent text.
    fn enc_delta_block_chunk(
        &self,
        enc: &ComputeCommandEncoderRef,
        la: &QuantLinearAttn,
        l: usize,
        n: usize,
        conc: bool,
    ) {
        let e = self.engine;
        let d = e.deltanet_dims().expect("deltanet dims on a deltanet checkpoint");
        let ds = self.ds_ref();
        let st = self.deltanet.as_ref().expect("deltanet states").layers[l]
            .as_ref()
            .expect("every linear layer owns recurrent state");
        let eps = e.cfg.rms_norm_eps;
        let (s_dim, hv, hk) = (d.d_state, d.n_v_heads, d.n_k_heads);
        let (key_dim, inner, c_all) = (s_dim * hk, d.d_inner(), d.conv_channels());
        let s32 = s_dim as u32;
        macro_rules! bar {
            ($($b:expr),+) => { if conc { enc.memory_barrier_with_resources(&[$($b),+]) } };
        }
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
        // 1. the two per-head scalars, every token at once.
        enc.set_compute_pipeline_state(&e.pipes.delta_gates);
        enc.set_buffer(0, Some(&ds.alpha), 0);
        enc.set_buffer(1, Some(&ds.beta_p), 0);
        enc.set_buffer(2, Some(&la.a), 0);
        enc.set_buffer(3, Some(&la.dt_bias), 0);
        enc.set_buffer(4, Some(&ds.g), 0);
        enc.set_buffer(5, Some(&ds.beta), 0);
        let hv32 = hv as u32;
        let n32 = n as u32;
        enc.set_bytes(6, 4, &hv32 as *const u32 as *const _);
        enc.set_bytes(7, 4, &n32 as *const u32 as *const _);
        dispatch_grid(enc, n * hv);
        bar!(&ds.g, &ds.beta);

        // 2. the conv for every (token, channel), then the window rolled once
        //    for the whole chunk.
        enc.set_compute_pipeline_state(&e.pipes.ssm_conv_prefill);
        enc.set_buffer(0, Some(&st.conv), 0);
        enc.set_buffer(1, Some(&ds.qkv), 0);
        enc.set_buffer(2, Some(&la.conv1d), 0);
        enc.set_buffer(3, Some(&ds.conv_out), 0);
        enc.set_bytes(4, size_of::<SsmConvBatchParams>() as u64, &cp as *const _ as *const _);
        dispatch_grid(enc, n * c_all);
        // WAR on st.conv, plus the read-after-write on conv_out.
        bar!(&ds.conv_out, &st.conv);
        enc.set_compute_pipeline_state(&e.pipes.ssm_conv_roll);
        enc.set_buffer(0, Some(&st.conv), 0);
        enc.set_buffer(1, Some(&ds.qkv), 0);
        enc.set_bytes(2, size_of::<SsmConvBatchParams>() as u64, &cp as *const _ as *const _);
        dispatch_grid(enc, c_all);
        bar!(&st.conv);

        // 3. l2-normalise q and k for every token — the grid's y axis is the
        //    token and `tok_stride` walks it.
        let cstride = c_all as u32;
        for off in [0u64, (key_dim * 4) as u64] {
            enc.set_compute_pipeline_state(&e.pipes.l2norm_rows);
            enc.set_buffer(0, Some(&ds.conv_out), off);
            enc.set_bytes(1, 4, &s32 as *const u32 as *const _);
            enc.set_bytes(2, 4, &eps as *const f32 as *const _);
            enc.set_bytes(3, 4, &cstride as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(hk as u64, n as u64, 1),
                MTLSize::new(256, 1, 1),
            );
        }
        bar!(&ds.conv_out);

        // 4. the delta rule — still one dispatch per token; the state is
        //    read-modify-written, so token t+1 must see token t's writes.
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
            let dp = DeltaStepParams {
                d_state: s32,
                n_v_heads: hv as u32,
                group: (hv / hk) as u32,
            };
            enc.set_bytes(7, size_of::<DeltaStepParams>() as u64, &dp as *const _ as *const _);
            dispatch_grid(enc, s_dim * hv);
            bar!(&ds.dout, &st.delta);
        }

        // 5. the gated output norm. Head h of token t sits at (t·H_v + h)·S, so
        //    the rows are already contiguous across the chunk and this needs no
        //    stride — one threadgroup per (token, head).
        enc.set_compute_pipeline_state(&e.pipes.gated_output_norm);
        enc.set_buffer(0, Some(&ds.dout), 0);
        enc.set_buffer(1, Some(&la.ssm_norm), 0);
        enc.set_buffer(2, Some(&ds.z), 0);
        enc.set_bytes(3, 4, &s32 as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((n * hv) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        bar!(&ds.dout);
    }

    fn run_from_quant(
        &mut self,
        src: Source<'_>,
        n: usize,
        pos0: usize,
        layer0: usize,
        logits_rows: usize,
    ) -> crate::Result<Vec<f32>> {
        if self.timing.is_some() {
            objc2::rc::autoreleasepool(|_| {
                self.run_from_quant_inner(src, n, pos0, layer0, logits_rows)
            })
        } else {
            self.run_from_quant_inner(src, n, pos0, layer0, logits_rows)
        }
    }

    fn run_from_quant_inner(
        &mut self,
        src: Source<'_>,
        n: usize,
        pos0: usize,
        layer0: usize,
        logits_rows: usize,
    ) -> crate::Result<Vec<f32>> {
        let t_step = std::time::Instant::now();
        // Out of the session for the step so the phase encoder can hold it
        // mutably while `self` keeps handing out buffers; put back on the way
        // out (every early return below is an Err, which ends the run anyway).
        let mut timing = self.timing.take();
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
                        SrcType::Quant(t) => crate::gguf::dequant_row_ref(t, row, dst),
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
        // Decode takes the fused path up to DEC_MAX_HD. It used to stop at DEC_TG,
        // which excluded head_dim 256 — i.e. the WHOLE qwen35 arch, including the 48
        // of 64 deltanet layers that have no attention constraint at all — and made
        // every decode step run the concurrent prefill encoder: tiled matmuls at
        // n = 1 instead of matvecs, ~80x off the memory-bound ceiling on the 27B.
        // The only piece that was ever hd-bound is the decode attention kernel, now
        // generalized; the Full-attn fused branch below (joint Q+gate split, per-head
        // qk-norm, partial rope) was already written for qwen35 and simply never ran.
        let fused_decode = n == 1 && hd <= DEC_MAX_HD && hd.is_multiple_of(4);
        let conc = !fused_decode;
        let cpu_pre_ns = t_step.elapsed().as_nanos();
        // One encoder when timing is off — the construction this path has
        // always used, byte for byte. One encoder PER PHASE RUN when it is on.
        let mut pe = PhaseEnc::new(cb, conc, timing.as_deref_mut());
        // A mark with zero dispatches never cuts an encoder, so the two qwen35
        // -only helpers below cost nothing on checkpoints that lack the joint
        // Q+gate projection.
        let qg_n = u32::from(self.qg.is_some());
        let dn_disp = 6 * n as u32; // the deltanet chain's kernels per token
        macro_rules! bar {
            ($($b:expr),+) => { if conc { pe.cur().memory_barrier_with_resources(&[$($b),+]) } };
        }

        let v_base = self.kvs.length() / 2;
        let rot = e.dims.rot_dim; // == hd except on qwen35 (partial rope)
        for (l, blk) in q.blocks.iter().enumerate().skip(layer0) {
            if fused_decode {
                e.enc_rmsnorm(pe.at(Phase::Norm, 1), &self.x, &blk.input_layernorm, &self.xn, 1);
                match &blk.attn {
                    QuantAttn::Linear(la) => {
                        // Gated-deltanet block: four projections, the six-kernel
                        // chain one token at a time, then the out projection
                        // accumulated straight into x (what o_proj does below).
                        self.enc_qmv(pe.at(Phase::Mm(MmRole::Dn, la.qkv.sel), 1), &q.pipe(la.qkv.sel).matvec, &la.qkv, &self.xn, &self.ds_ref().qkv, 0);
                        self.enc_qmv(pe.at(Phase::Mm(MmRole::Dn, la.z_gate.sel), 1), &q.pipe(la.z_gate.sel).matvec, &la.z_gate, &self.xn, &self.ds_ref().z, 0);
                        self.enc_qmv(pe.at(Phase::Mm(MmRole::Dn, la.alpha.sel), 1), &q.pipe(la.alpha.sel).matvec, &la.alpha, &self.xn, &self.ds_ref().alpha, 0);
                        self.enc_qmv(pe.at(Phase::Mm(MmRole::Dn, la.beta.sel), 1), &q.pipe(la.beta.sel).matvec, &la.beta, &self.xn, &self.ds_ref().beta_p, 0);
                        self.enc_delta_block(pe.at(Phase::Delta, dn_disp), la, l, 1, conc);
                        self.enc_qmv(pe.at(Phase::Mm(MmRole::Dn, la.out.sel), 1), &q.pipe(la.out.sel).matvec_acc, &la.out, &self.ds_ref().dout, &self.x, 0);
                    }
                    QuantAttn::Full(fa) => {
                self.enc_qmv(pe.at(Phase::Mm(MmRole::Attn, fa.q_proj.sel), 1), &q.pipe(fa.q_proj.sel).matvec, &fa.q_proj, &self.xn, &self.q, 0);
                let qbuf = self.enc_split_qg(pe.at(Phase::Attn, qg_n), 1, conc);
                self.enc_qmv(pe.at(Phase::Mm(MmRole::Attn, fa.k_proj.sel), 1), &q.pipe(fa.k_proj.sel).matvec_h, &fa.k_proj, &self.xn, &self.k_cache[l], kv_byte_off);
                self.enc_qmv(pe.at(Phase::Mm(MmRole::Attn, fa.v_proj.sel), 1), &q.pipe(fa.v_proj.sel).matvec_h, &fa.v_proj, &self.xn, &self.v_cache[l], kv_byte_off);
                if let (Some(qn), Some(kn)) = (&fa.q_norm, &fa.k_norm) {
                    e.enc_rmsnorm_dim(pe.at(Phase::Norm, 1), qbuf, qn, cfg.num_attention_heads, hd);
                    e.enc_rmsnorm_h_inplace(pe.at(Phase::Norm, 1), &self.k_cache[l], kv_byte_off, kn, cfg.num_key_value_heads, hd);
                }
                e.enc_rope_qk(pe.at(Phase::Rope, 1), qbuf, &self.k_cache[l], kv_byte_off, pos0);
                // two kernels: the flash-decoding partials, then the merge.
                e.enc_attention_decode(pe.at(Phase::Attn, 2), qbuf, &self.k_cache[l], &self.v_cache[l], self.kv_base, &self.partials, &self.att, pos0);
                self.enc_apply_qgate(pe.at(Phase::Attn, qg_n), 1, conc);
                self.enc_qmv(pe.at(Phase::Mm(MmRole::Attn, fa.o_proj.sel), 1), &q.pipe(fa.o_proj.sel).matvec_acc, &fa.o_proj, &self.att, &self.x, 0);
                    }
                }
                e.enc_rmsnorm(pe.at(Phase::Norm, 1), &self.x, &blk.post_attention_layernorm, &self.xn, 1);
                // gate/up dispatch separately: a mixed-quant file may hold the
                // two halves in different encodings, and matvec_swiglu assumes
                // one selector for both.
                self.enc_qmv(pe.at(Phase::Mm(MmRole::Mlp, blk.gate_proj.sel), 1), &q.pipe(blk.gate_proj.sel).matvec, &blk.gate_proj, &self.xn, &self.gate, 0);
                self.enc_qmv(pe.at(Phase::Mm(MmRole::Mlp, blk.up_proj.sel), 1), &q.pipe(blk.up_proj.sel).matvec, &blk.up_proj, &self.xn, &self.up, 0);
                {
                    let enc = pe.at(Phase::Elem, 1);
                    let p = ElemParams { dim: cfg.intermediate_size as u32 };
                    enc.set_compute_pipeline_state(&e.pipes.silu_mul);
                    enc.set_buffer(0, Some(&self.gate), 0);
                    enc.set_buffer(1, Some(&self.up), 0);
                    enc.set_bytes(2, size_of::<ElemParams>() as u64, &p as *const _ as *const _);
                    dispatch_grid(enc, cfg.intermediate_size);
                }
                self.enc_qmv(pe.at(Phase::Mm(MmRole::Mlp, blk.down_proj.sel), 1), &q.pipe(blk.down_proj.sel).matvec_acc, &blk.down_proj, &self.gate, &self.x, 0);
                continue;
            }

            // Prefill: rmsnorm (f32), quant GEMMs, staged K/V into the cache.
            e.enc_rmsnorm(pe.at(Phase::Norm, 1), &self.x, &blk.input_layernorm, &self.xn, n);
            bar!(&self.xn);
            let attn_out_done = match &blk.attn {
                QuantAttn::Linear(la) => {
                    // Gated-deltanet block, prefill form: the four projections,
                    // then the decode-form chain token by token (plan v1), then
                    // the out projection and the residual add.
                    self.enc_qmm(pe.at(Phase::Mm(MmRole::Dn, la.qkv.sel), 1), &q.pipe(la.qkv.sel).matmul_pg, &la.qkv, &self.xn, 0, &self.ds_ref().qkv, 0, n);
                    self.enc_qmm(pe.at(Phase::Mm(MmRole::Dn, la.z_gate.sel), 1), &q.pipe(la.z_gate.sel).matmul_pg, &la.z_gate, &self.xn, 0, &self.ds_ref().z, 0, n);
                    self.enc_qmm(pe.at(Phase::Mm(MmRole::Dn, la.alpha.sel), 1), &q.pipe(la.alpha.sel).matmul_pg, &la.alpha, &self.xn, 0, &self.ds_ref().alpha, 0, n);
                    self.enc_qmm(pe.at(Phase::Mm(MmRole::Dn, la.beta.sel), 1), &q.pipe(la.beta.sel).matmul_pg, &la.beta, &self.xn, 0, &self.ds_ref().beta_p, 0, n);
                    bar!(&self.ds_ref().qkv, &self.ds_ref().z, &self.ds_ref().alpha, &self.ds_ref().beta_p);
                    self.enc_delta_block(pe.at(Phase::Delta, dn_disp), la, l, n, conc);
                    bar!(&self.ds_ref().dout);
                    self.enc_qmm(pe.at(Phase::Mm(MmRole::Dn, la.out.sel), 1), &q.pipe(la.out.sel).matmul_pg, &la.out, &self.ds_ref().dout, 0, &self.xb, 0, n);
                    bar!(&self.xb);
                    e.enc_elementwise(pe.at(Phase::Elem, 1), &e.pipes.add_inplace, &self.x, &self.xb, n * h);
                    bar!(&self.x);
                }
                QuantAttn::Full(fa) => {
                self.enc_qmm(pe.at(Phase::Mm(MmRole::Attn, fa.q_proj.sel), 1), &q.pipe(fa.q_proj.sel).matmul_pg, &fa.q_proj, &self.xn, 0, &self.q, 0, n);
                self.enc_qmm(pe.at(Phase::Mm(MmRole::Attn, fa.k_proj.sel), 1), &q.pipe(fa.k_proj.sel).matmul_pg, &fa.k_proj, &self.xn, 0, &self.kvs, 0, n);
                self.enc_qmm(pe.at(Phase::Mm(MmRole::Attn, fa.v_proj.sel), 1), &q.pipe(fa.v_proj.sel).matmul_pg, &fa.v_proj, &self.xn, 0, &self.kvs, v_base, n);
                bar!(&self.q, &self.kvs);
                // qwen35 projects Q and the output gate together, interleaved per
                // head; split once so qk-norm, RoPE and attention keep seeing an
                // ordinary compact Q. A no-op on every other checkpoint.
                let qbuf = self.enc_split_qg(pe.at(Phase::Attn, qg_n), n, conc);
                if let (Some(qn), Some(kn)) = (&fa.q_norm, &fa.k_norm) {
                    // qwen3: per-head norm before RoPE — q in place (f32), k while
                    // still in the f32 staging half (why this precedes the spans).
                    //
                    // NORMALIZE `qbuf`, NEVER `self.q`. On a joint Q+gate
                    // checkpoint (qwen35) enc_split_qg has already drained
                    // self.q into the compact q/gate pair and nothing reads it
                    // again, so norming self.q here writes to a dead buffer and
                    // hands attention an UN-NORMALIZED Q — fluent, wrong output
                    // on prefill only, which is what this lane spent a session
                    // chasing. On every other arch qbuf IS self.q, so `qbuf` is
                    // correct in both cases and `self.q` is correct in only one.
                    // The decode half below has always used qbuf; keep them same.
                    e.enc_rmsnorm_dim(pe.at(Phase::Norm, 1), qbuf, qn, n * cfg.num_attention_heads, hd);
                    e.enc_rmsnorm_dim(pe.at(Phase::Norm, 1), &self.kvs, kn, n * cfg.num_key_value_heads, hd);
                    bar!(qbuf, &self.kvs);
                }
                {
                    let enc = pe.at(Phase::Rope, 1);
                    let rp = RopeParams {
                        head_dim: hd as u32,
                        n_heads: cfg.num_attention_heads as u32,
                        pos0: pos0 as u32,
                        theta: cfg.rope_theta,
                        n_rows: n as u32,
                        rot_dim: rot as u32,
                    };
                    enc.set_compute_pipeline_state(&e.pipes.rope);
                    enc.set_buffer(0, Some(qbuf), 0);
                    enc.set_bytes(1, size_of::<RopeParams>() as u64, &rp as *const _ as *const _);
                    dispatch_grid(enc, n * cfg.num_attention_heads * rot / 2);
                }
                let spans: Vec<(usize, usize, usize)> = match &e.win {
                    Some(w) => win_write_spans(&w.cfg, pos0, n),
                    None => vec![(0, pos0, n)],
                };
                for &(row, slot, len) in &spans {
                    let src_off = (row * kvd * 4) as u64;
                    let dst_off = self.kv_base + (slot * kvd * 2) as u64;
                    e.enc_f32_to_f16(pe.at(Phase::KvStage, 1), &self.kvs, src_off, &self.k_cache[l], dst_off, len * kvd);
                    e.enc_f32_to_f16(pe.at(Phase::KvStage, 1), &self.kvs, v_base + src_off, &self.v_cache[l], dst_off, len * kvd);
                    bar!(&self.k_cache[l]);
                    let enc = pe.at(Phase::Rope, 1);
                    let rp = RopeParams {
                        head_dim: hd as u32,
                        n_heads: cfg.num_key_value_heads as u32,
                        pos0: (pos0 + row) as u32,
                        theta: cfg.rope_theta,
                        n_rows: len as u32,
                        rot_dim: rot as u32,
                    };
                    enc.set_compute_pipeline_state(&e.pipes.rope_h);
                    enc.set_buffer(0, Some(&self.k_cache[l]), dst_off);
                    enc.set_bytes(1, size_of::<RopeParams>() as u64, &rp as *const _ as *const _);
                    dispatch_grid(enc, len * cfg.num_key_value_heads * hd / 2);
                }
                bar!(qbuf, &self.k_cache[l], &self.v_cache[l], &self.kvs);
                {
                    let kv_extent = match &e.win {
                        Some(w) => w.cfg.cap,
                        None => self.max_seq,
                    };
                    e.enc_attention(pe.at(Phase::Attn, 1), qbuf, &self.k_cache[l], &self.v_cache[l], self.kv_base, &self.scores, &self.att, pos0, n, kv_extent, &self.xh);
                    bar!(&self.att, &self.scores);
                }
                self.enc_apply_qgate(pe.at(Phase::Attn, qg_n), n, conc);
                self.enc_qmm(pe.at(Phase::Mm(MmRole::Attn, fa.o_proj.sel), 1), &q.pipe(fa.o_proj.sel).matmul_pg, &fa.o_proj, &self.att, 0, &self.xb, 0, n);
                bar!(&self.xb);
                e.enc_elementwise(pe.at(Phase::Elem, 1), &e.pipes.add_inplace, &self.x, &self.xb, n * h);
                bar!(&self.x);
                }
            };
            let _ = attn_out_done;

            e.enc_rmsnorm(pe.at(Phase::Norm, 1), &self.x, &blk.post_attention_layernorm, &self.xn, n);
            bar!(&self.xn);
            self.enc_qmm(pe.at(Phase::Mm(MmRole::Mlp, blk.gate_proj.sel), 1), &q.pipe(blk.gate_proj.sel).matmul_pg, &blk.gate_proj, &self.xn, 0, &self.gate, 0, n);
            self.enc_qmm(pe.at(Phase::Mm(MmRole::Mlp, blk.up_proj.sel), 1), &q.pipe(blk.up_proj.sel).matmul_pg, &blk.up_proj, &self.xn, 0, &self.up, 0, n);
            bar!(&self.gate, &self.up);
            {
                let enc = pe.at(Phase::Elem, 1);
                let p = ElemParams { dim: (n * cfg.intermediate_size) as u32 };
                enc.set_compute_pipeline_state(&e.pipes.silu_mul);
                enc.set_buffer(0, Some(&self.gate), 0);
                enc.set_buffer(1, Some(&self.up), 0);
                enc.set_bytes(2, size_of::<ElemParams>() as u64, &p as *const _ as *const _);
                dispatch_grid(enc, n * cfg.intermediate_size);
            }
            bar!(&self.gate);
            self.enc_qmm(pe.at(Phase::Mm(MmRole::Mlp, blk.down_proj.sel), 1), &q.pipe(blk.down_proj.sel).matmul_pg, &blk.down_proj, &self.gate, 0, &self.xb, 0, n);
            bar!(&self.xb);
            e.enc_elementwise(pe.at(Phase::Elem, 1), &e.pipes.add_inplace, &self.x, &self.xb, n * h);
            bar!(&self.x);
        }

        if logits_rows > 0 {
            e.enc_rmsnorm(pe.at(Phase::Norm, 1), &self.x, &q.final_norm, &self.xn, n);
            bar!(&self.xn);
            let first = n - logits_rows;
            if logits_rows == 1 && !conc {
                self.enc_qmv(pe.at(Phase::Mm(MmRole::Head, q.lm_head.sel), 1), &q.pipe(q.lm_head.sel).matvec, &q.lm_head, &self.xn, &self.logits, 0);
            } else {
                self.enc_qmm(pe.at(Phase::Mm(MmRole::Head, q.lm_head.sel), 1), &q.pipe(q.lm_head.sel).matmul_pg, &q.lm_head, &self.xn, (first * h * 4) as u64, &self.logits, 0, logits_rows);
            }
        }

        pe.end();
        let t_commit = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let gpu_wait_ns = t_commit.elapsed().as_nanos();
        let cpu_enc_ns = t_commit.duration_since(t_step).as_nanos() - cpu_pre_ns;

        let t_post = std::time::Instant::now();
        let out = if logits_rows == 0 {
            Vec::new()
        } else {
            let logits = unsafe {
                std::slice::from_raw_parts(self.logits.contents() as *const f32, logits_rows * cfg.vocab_size)
            };
            logits.to_vec()
        };
        if let Some(t) = timing.as_deref_mut() {
            let kind = if n == 1 { "decode" } else { "prefill" };
            t.report(
                kind,
                n,
                pos0,
                t_step.elapsed().as_nanos(),
                cpu_pre_ns,
                cpu_enc_ns,
                gpu_wait_ns,
                t_post.elapsed().as_nanos(),
            );
        }
        self.timing = timing;
        Ok(out)
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
        self.assert_kv_layer(layer);
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

    /// Refuse a KV write aimed at a layer whose cache is a stub — its state is
    /// recurrent and lives in `deltanet` instead. Once per layer per chunk, so
    /// free next to the copy it guards, and it turns a future caller that
    /// routes a hybrid checkpoint's linear layer through the ANE path into a
    /// named panic rather than silent device-memory corruption.
    fn assert_kv_layer(&self, layer: usize) {
        assert_eq!(
            self.state[layer],
            LayerStateKind::Kv,
            "write_kv on layer {layer}: that layer's state is recurrent, its KV cache is a stub"
        );
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
        self.assert_kv_layer(layer);
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
        if self.deltanet_layout.is_some() {
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
            q: f32_buffer(d, n_slots * attn_row_width(h, self.dims.q_proj_dim)),
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

// ==================== GPU time attribution (LOKAL_GPU_TIMING) ====================
//
// DIAGNOSIS ONLY, and default OFF. With the variable unset this file dispatches
// exactly what it dispatched before the mode existed — same encoder
// construction, same kernel order, same buffers — so every identity gate holds
// byte for byte. The whole off-path cost is one `Option::is_none` per phase
// mark.
//
// MECHANISM, and why this one and not the obvious one. Apple M-series report
// `supportsCounterSampling(AtStageBoundary) == true` and every other sampling
// point FALSE (measured, M1 Pro / macOS 26), so per-DISPATCH counters do not
// exist on this hardware at any price. What does exist is a timestamp pair per
// COMPUTE ENCODER, attached through MTLComputePassDescriptor. So a timed step
// splits its one encoder into one encoder per contiguous run of same-phase
// dispatches — still ONE command buffer, ONE commit, ONE wait. The GPU work is
// unchanged; only the phase boundaries become observable.
//
// ITS DISTORTION, measured rather than assumed (`gputime hdr` carries it):
//   * each encoder boundary costs a fixed slice INSIDE the measured window
//     (encoder start + drain) and a fixed GAP between windows;
//   * on the CONCURRENT prefill encoder a boundary additionally serializes
//     overlap that the unsplit encoder would have had.
// `calib_inside_ns` / `calib_gap_ns` are that constant, measured in-process on
// this device against a real pipeline, so any phase's raw ns can be corrected
// by its own encoder count. `LOKAL_GPU_TIMING=total` runs the step as ONE
// encoder (two samples, no splitting) — the undistorted reference the harness
// compares timed against untimed throughput with.
//
// metal-rs 0.33 BUG WORKED AROUND HERE: `CounterSampleBufferRef::
// resolve_counter_range` builds `Vec::with_capacity(n)` and then passes
// `size_of_val(data.as_slice())` — the length of an EMPTY vec, i.e. 0 — to
// `-getBytes:length:`, so it copies nothing and always returns zeros. The
// resolve below goes straight to the ObjC selector through objc2 (already a
// dependency of this crate) for that reason.

/// What a run of dispatches is doing. The tag is compared to decide where one
/// timed encoder ends and the next begins, so two adjacent dispatches with the
/// same tag cost one encoder, not two.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// rmsnorm in all its shapes (f32, per-head, in-place-half).
    Norm,
    /// rope / rope_h / rope_qk_decode.
    Rope,
    /// attention proper, plus the q/gate split and output gate that only exist
    /// to feed it.
    Attn,
    /// the gated-deltanet six-kernel chain (delta_gates, ssm_conv_decode,
    /// l2norm_rows x2, delta_decode_step, gated_output_norm).
    Delta,
    /// silu_mul, add_inplace — the cheap elementwise glue.
    Elem,
    /// f32 -> f16 staging of fresh K/V rows into the cache.
    KvStage,
    /// a weight multiply: which weight, and the quant selector it dispatches.
    Mm(MmRole, u32),
    /// everything not currently being isolated (see the filter).
    Other,
}

/// Which weight a `Phase::Mm` is multiplying. The quant family answers "is the
/// i-quant grid dequant the problem"; the role answers "which matmul do we
/// fix", and the two questions have different answers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MmRole {
    /// q / k / v / o projections of a full-attention block.
    Attn,
    /// the gated-deltanet block's projections (qkv, z, alpha, beta, out).
    Dn,
    /// gate / up / down.
    Mlp,
    /// the vocab head.
    Head,
}

impl MmRole {
    fn name(self) -> &'static str {
        match self {
            MmRole::Attn => "attn",
            MmRole::Dn => "dn",
            MmRole::Mlp => "mlp",
            MmRole::Head => "head",
        }
    }
}

/// Selector -> short type label for the table. This MIRRORS `SrcType::qtype()`
/// (src/lowmem/mod.rs) and is deliberately the cheap kind of mirror: an
/// unmapped selector prints `selN` — visible in the table rather than silent —
/// and a wrong label cannot change one bit of what the GPU computes. The test
/// `sel_names_cover_every_quant_type` keeps the known set honest.
fn sel_name(sel: u32) -> String {
    let s = match sel {
        0 => "f16",
        1 => "bf16",
        2 => "q8_0",
        3 => "q4_0",
        4 => "q4_k",
        5 => "q6_k",
        6 => "q5_k",
        7 => "q5_0",
        8 => "q2_k",
        9 => "q3_k",
        10 => "iq4_nl",
        11 => "iq4_xs",
        12 => "iq3_xxs",
        13 => "iq3_s",
        14 => "iq2_xxs",
        15 => "iq2_xs",
        16 => "iq2_s",
        17 => "iq1_s",
        18 => "iq1_m",
        _ => return format!("sel{sel}"),
    };
    s.to_string()
}

impl Phase {
    fn name(self) -> String {
        match self {
            Phase::Norm => "norm".to_string(),
            Phase::Rope => "rope".to_string(),
            Phase::Attn => "attn".to_string(),
            Phase::Delta => "delta".to_string(),
            Phase::Elem => "elem".to_string(),
            Phase::KvStage => "kvstage".to_string(),
            Phase::Other => "other".to_string(),
            Phase::Mm(r, sel) => format!("mm:{}:{}", r.name(), sel_name(sel)),
        }
    }
}

/// A `#[repr(C)]` NSRange we can hand to objc2 (metal-rs's own NSRange carries
/// no `Encode` impl).
#[repr(C)]
#[derive(Clone, Copy)]
struct NsRange {
    location: usize,
    length: usize,
}
unsafe impl objc2::encode::Encode for NsRange {
    const ENCODING: objc2::encode::Encoding = objc2::encode::Encoding::Struct(
        "_NSRange",
        &[usize::ENCODING, usize::ENCODING],
    );
}

/// Sentinel Metal writes when a sample could not be taken.
const COUNTER_ERROR: u64 = u64::MAX;

/// Read `n` timestamps out of a counter sample buffer. See the metal-rs bug
/// note at the top of this section for why this is not
/// `resolve_counter_range`.
fn resolve_timestamps(sb: &metal::CounterSampleBufferRef, n: usize) -> Vec<u64> {
    use metal::foreign_types::ForeignTypeRef;
    if n == 0 {
        return Vec::new();
    }
    unsafe {
        let obj = sb.as_ptr() as *mut objc2::runtime::AnyObject;
        let r = NsRange { location: 0, length: n };
        let data: *mut objc2::runtime::AnyObject =
            objc2::msg_send![obj, resolveCounterRange: r];
        if data.is_null() {
            return Vec::new();
        }
        let bytes: *const u8 = objc2::msg_send![data, bytes];
        let len: usize = objc2::msg_send![data, length];
        if bytes.is_null() || len < n * 8 {
            return Vec::new();
        }
        std::slice::from_raw_parts(bytes as *const u64, n).to_vec()
    }
}

/// The per-process timing state: the sample buffer, the phase filter, and the
/// per-step encoder ledger. `None` on the session unless LOKAL_GPU_TIMING is set.
pub(crate) struct GpuTiming {
    sb: metal::CounterSampleBuffer,
    /// How many encoders one step may split into before the sample buffer runs
    /// out. Metal caps a sample buffer at 32768 bytes = 4096 timestamps.
    cap_enc: usize,
    /// `None` = isolate every phase (mode `1`); `Some(v)` = only phases whose
    /// name starts with one of these split out, the rest merge into `other`
    /// (`total` is the empty vector: nothing splits, one encoder per step).
    filter: Option<Vec<String>>,
    /// One entry per timed encoder used by the step in flight.
    tags: Vec<Phase>,
    disp: Vec<u32>,
    /// The step ran out of samples and stopped splitting — its phase sums are
    /// short and the harness must say so rather than average them in.
    overflow: bool,
    dec_seq: u32,
    pre_seq: u32,
}

impl GpuTiming {
    /// Build from the environment. Returns None when timing is off, which is
    /// the only state any non-diagnostic run is ever in.
    fn from_env(e: &MetalEngine) -> Option<Box<GpuTiming>> {
        let raw = std::env::var("LOKAL_GPU_TIMING").ok()?;
        let v = raw.trim();
        if v.is_empty() || v == "0" || v.eq_ignore_ascii_case("off") {
            return None;
        }
        let filter = match v {
            "1" | "full" | "all" => None,
            "total" | "none" => Some(Vec::new()),
            other => Some(other.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()),
        };
        let ts = e.device.counter_sets().into_iter().find(|c| c.name() == "timestamp")?;
        if !e.device.supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary) {
            eprintln!("gputime hdr error=\"device does not support AtStageBoundary counter sampling — timing disabled\"");
            return None;
        }
        // 4096 timestamps is the device maximum (32768 B); two per encoder.
        let n_samples: u64 = 4096;
        let desc = metal::CounterSampleBufferDescriptor::new();
        desc.set_counter_set(&ts);
        desc.set_sample_count(n_samples);
        desc.set_storage_mode(metal::MTLStorageMode::Shared);
        let sb = match e.device.new_counter_sample_buffer_with_descriptor(&desc) {
            Ok(sb) => sb,
            Err(err) => {
                eprintln!("gputime hdr error=\"counter sample buffer: {err}\"");
                return None;
            }
        };
        let t = GpuTiming {
            sb,
            cap_enc: (n_samples / 2) as usize,
            filter,
            tags: Vec::new(),
            disp: Vec::new(),
            overflow: false,
            dec_seq: 0,
            pre_seq: 0,
        };
        let (inside, gap) = t.calibrate(e);
        let mode = match &t.filter {
            None => "full".to_string(),
            Some(v) if v.is_empty() => "total".to_string(),
            Some(v) => format!("isolate:{}", v.join("+")),
        };
        eprintln!(
            "gputime hdr device=\"{}\" mode={} cap_enc={} calib_inside_ns={:.0} calib_gap_ns={:.0} unit=ns",
            e.device.name(),
            mode,
            t.cap_enc,
            inside,
            gap
        );
        Some(Box::new(t))
    }

    /// What an encoder boundary costs on THIS device, measured against a real
    /// pipeline (add_inplace over one element — the cheapest dispatch the
    /// engine already owns; this lane may not add a kernel). `inside` is the
    /// fixed slice each encoder's own measured window carries, `gap` the dead
    /// time between windows. Reported, never silently subtracted.
    fn calibrate(&self, e: &MetalEngine) -> (f64, f64) {
        const N: usize = 128;
        let a = f32_buffer(&e.device, 4);
        let b = f32_buffer(&e.device, 4);
        let p = ElemParams { dim: 1 };
        let mut inside = Vec::new();
        let mut gap = Vec::new();
        for _ in 0..3 {
            // A: one encoder, N dispatches (samples 0,1).
            let cb = e.queue.new_command_buffer();
            let pd = metal::ComputePassDescriptor::new();
            let att = pd.sample_buffer_attachments().object_at(0).unwrap();
            att.set_sample_buffer(&self.sb);
            att.set_start_of_encoder_sample_index(0);
            att.set_end_of_encoder_sample_index(1);
            let enc = cb.compute_command_encoder_with_descriptor(pd);
            for _ in 0..N {
                enc.set_compute_pipeline_state(&e.pipes.add_inplace);
                enc.set_buffer(0, Some(&a), 0);
                enc.set_buffer(1, Some(&b), 0);
                enc.set_bytes(2, size_of::<ElemParams>() as u64, &p as *const _ as *const _);
                dispatch_grid(enc, 1);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            // B: N encoders, one dispatch each (samples 2 .. 2+2N).
            let cb = e.queue.new_command_buffer();
            for i in 0..N {
                let pd = metal::ComputePassDescriptor::new();
                let att = pd.sample_buffer_attachments().object_at(0).unwrap();
                att.set_sample_buffer(&self.sb);
                att.set_start_of_encoder_sample_index((2 + 2 * i) as u64);
                att.set_end_of_encoder_sample_index((2 + 2 * i + 1) as u64);
                let enc = cb.compute_command_encoder_with_descriptor(pd);
                enc.set_compute_pipeline_state(&e.pipes.add_inplace);
                enc.set_buffer(0, Some(&a), 0);
                enc.set_buffer(1, Some(&b), 0);
                enc.set_bytes(2, size_of::<ElemParams>() as u64, &p as *const _ as *const _);
                dispatch_grid(enc, 1);
                enc.end_encoding();
            }
            cb.commit();
            cb.wait_until_completed();
            let ts = resolve_timestamps(&self.sb, 2 + 2 * N);
            if ts.len() < 2 + 2 * N || ts.iter().any(|&t| t == COUNTER_ERROR) {
                return (f64::NAN, f64::NAN);
            }
            let one = ts[1].saturating_sub(ts[0]) as f64;
            let sum: f64 = (0..N)
                .map(|i| ts[2 + 2 * i + 1].saturating_sub(ts[2 + 2 * i]) as f64)
                .sum();
            let span = ts[2 + 2 * N - 1].saturating_sub(ts[2]) as f64;
            inside.push((sum - one) / N as f64);
            gap.push((span - sum) / (N - 1) as f64);
        }
        let med = |mut v: Vec<f64>| {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        (med(inside), med(gap))
    }

    /// The tag an encoder is opened under: the phase itself when it is being
    /// isolated, `Other` when it is not. `full` isolates everything, `total`
    /// isolates nothing — one code path covers both.
    fn key_of(&self, p: Phase) -> Phase {
        match &self.filter {
            None => p,
            Some(names) => {
                let n = p.name();
                if names.iter().any(|f| n.starts_with(f.as_str())) {
                    p
                } else {
                    Phase::Other
                }
            }
        }
    }

    fn begin_step(&mut self) {
        self.tags.clear();
        self.disp.clear();
        self.overflow = false;
    }

    /// Attach the next free sample pair to `pd`, or None when the buffer is full.
    fn attach(&mut self, pd: &metal::ComputePassDescriptorRef, tag: Phase) -> bool {
        let i = self.tags.len();
        if i >= self.cap_enc {
            self.overflow = true;
            return false;
        }
        let att = pd.sample_buffer_attachments().object_at(0).unwrap();
        att.set_sample_buffer(&self.sb);
        att.set_start_of_encoder_sample_index((2 * i) as u64);
        att.set_end_of_encoder_sample_index((2 * i + 1) as u64);
        self.tags.push(tag);
        self.disp.push(0);
        true
    }

    fn add_dispatches(&mut self, n: u32) {
        if let Some(d) = self.disp.last_mut() {
            *d += n;
        }
    }

    /// Resolve the step's samples and print it: one `step` line, then one
    /// `phase` line per phase that ran. Structured key=value on stderr so the
    /// harness never has to grep free text (protocol:gate-scripts rule 3).
    #[allow(clippy::too_many_arguments)]
    fn report(
        &mut self,
        kind: &str,
        n: usize,
        pos0: usize,
        wall_ns: u128,
        cpu_pre_ns: u128,
        cpu_enc_ns: u128,
        gpu_wait_ns: u128,
        cpu_post_ns: u128,
    ) {
        let n_enc = self.tags.len();
        let ts = resolve_timestamps(&self.sb, 2 * n_enc);
        let seq = if kind == "decode" {
            let s = self.dec_seq;
            self.dec_seq += 1;
            s
        } else {
            let s = self.pre_seq;
            self.pre_seq += 1;
            s
        };
        let mut sums: Vec<(Phase, u64, u32, u32)> = Vec::new(); // phase, ns, encoders, dispatches
        let mut bad = 0u32;
        let mut empty = 0u32;
        let mut total = 0u64;
        let (mut first, mut last) = (u64::MAX, 0u64);
        if ts.len() == 2 * n_enc {
            for i in 0..n_enc {
                // AN ENCODER WITH NO DISPATCHES NEVER HAS ITS COUNTERS WRITTEN.
                // Its sample pair still holds whatever the PREVIOUS step wrote
                // there, which reads as a perfectly plausible interval — that
                // stale pair is what made gpu_span_ns come out at half a second
                // for a 32 ms decode step. Skip them by construction rather
                // than trusting that none exist.
                if self.disp[i] == 0 {
                    empty += 1;
                    continue;
                }
                let (s, e) = (ts[2 * i], ts[2 * i + 1]);
                if s == COUNTER_ERROR || e == COUNTER_ERROR || e < s {
                    bad += 1;
                    continue;
                }
                first = first.min(s);
                last = last.max(e);
                let d = e - s;
                total += d;
                let tag = self.tags[i];
                match sums.iter_mut().find(|(p, _, _, _)| *p == tag) {
                    Some(row) => {
                        row.1 += d;
                        row.2 += 1;
                        row.3 += self.disp[i];
                    }
                    None => sums.push((tag, d, 1, self.disp[i])),
                }
            }
        } else {
            bad = n_enc as u32;
        }
        let span = if first == u64::MAX { 0 } else { last - first };
        sums.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!(
            "gputime step kind={kind} seq={seq} n={n} pos0={pos0} enc={n_enc} bad={bad} \
empty={empty} overflow={} wall_ns={wall_ns} cpu_pre_ns={cpu_pre_ns} cpu_encode_ns={cpu_enc_ns} \
gpu_wait_ns={gpu_wait_ns} cpu_post_ns={cpu_post_ns} gpu_sum_ns={total} gpu_span_ns={span}",
            u8::from(self.overflow)
        );
        for (p, ns, enc, disp) in sums {
            eprintln!(
                "gputime phase kind={kind} seq={seq} name={} ns={ns} enc={enc} disp={disp}",
                p.name()
            );
        }
    }
}

/// The encoder in flight, plus where to cut the next one. When timing is off
/// this is a thin newtype over the single encoder the engine has always used
/// and `at()` is a null check.
struct PhaseEnc<'a, 'b> {
    cb: &'a metal::CommandBufferRef,
    conc: bool,
    /// `None` until the first marked dispatch: an encoder opened before its
    /// phase is known would be an EMPTY encoder, and Metal does not write
    /// counter samples for one (see `report`).
    enc: Option<&'a ComputeCommandEncoderRef>,
    tm: Option<&'b mut GpuTiming>,
    key: Phase,
    /// The sample buffer filled up; stop splitting rather than churn encoders.
    stopped: bool,
}

impl<'a, 'b> PhaseEnc<'a, 'b> {
    fn new(cb: &'a metal::CommandBufferRef, conc: bool, tm: Option<&'b mut GpuTiming>) -> Self {
        match tm {
            // OFF PATH — byte-for-byte the construction this file has always used.
            None => PhaseEnc {
                cb,
                conc,
                enc: Some(if conc {
                    cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent)
                } else {
                    cb.new_compute_command_encoder()
                }),
                tm: None,
                key: Phase::Other,
                stopped: true,
            },
            Some(t) => {
                t.begin_step();
                PhaseEnc { cb, conc, enc: None, tm: Some(t), key: Phase::Other, stopped: false }
            }
        }
    }

    fn open(
        cb: &'a metal::CommandBufferRef,
        conc: bool,
        tm: Option<&mut GpuTiming>,
        tag: Phase,
    ) -> &'a ComputeCommandEncoderRef {
        let dt = if conc {
            metal::MTLDispatchType::Concurrent
        } else {
            metal::MTLDispatchType::Serial
        };
        if let Some(t) = tm {
            let pd = metal::ComputePassDescriptor::new();
            pd.set_dispatch_type(dt);
            if t.attach(pd, tag) {
                return cb.compute_command_encoder_with_descriptor(pd);
            }
        }
        if conc {
            cb.compute_command_encoder_with_dispatch_type(dt)
        } else {
            cb.new_compute_command_encoder()
        }
    }

    /// Mark the next `nd` dispatches as belonging to phase `p`, and return the
    /// encoder to put them in — a new one when the phase changed.
    ///
    /// `nd == 0` NEVER cuts an encoder. Two call sites (the joint Q+gate split
    /// and the output gate) dispatch nothing on checkpoints without a joint
    /// projection, and cutting there would leave an empty encoder holding a
    /// stale sample pair.
    fn at(&mut self, p: Phase, nd: u32) -> &'a ComputeCommandEncoderRef {
        let Some(t) = self.tm.as_deref_mut() else {
            return self.enc.expect("the untimed path opens its encoder eagerly");
        };
        if nd == 0 {
            if let Some(enc) = self.enc {
                return enc;
            }
        }
        if !self.stopped {
            let k = t.key_of(p);
            if k != self.key || self.enc.is_none() {
                if let Some(enc) = self.enc {
                    enc.end_encoding();
                }
                let before = t.tags.len();
                self.enc = Some(Self::open(self.cb, self.conc, Some(&mut *t), k));
                if t.tags.len() == before {
                    // out of samples: this and every later encoder is untimed
                    self.stopped = true;
                }
                self.key = k;
            }
        } else if self.enc.is_none() {
            self.enc = Some(Self::open(self.cb, self.conc, None, self.key));
        }
        t.add_dispatches(nd);
        self.enc.expect("just opened")
    }

    /// The encoder currently open, for barriers that must ride the same one.
    fn cur(&mut self) -> &'a ComputeCommandEncoderRef {
        if self.enc.is_none() {
            let tm = self.tm.as_deref_mut();
            let key = self.key;
            self.enc = Some(Self::open(self.cb, self.conc, tm, key));
        }
        self.enc.expect("just opened")
    }

    fn end(self) {
        if let Some(enc) = self.enc {
            enc.end_encoding();
        }
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
    /// `sel_name` (the GPU-attribution table's type labels) mirrors
    /// `SrcType::qtype()` in src/lowmem/mod.rs. It is the CHEAP kind of mirror
    /// — a wrong label cannot move one computed bit, only mislabel a
    /// diagnostic row — but a selector the table has never heard of is exactly
    /// the silent drift the guard below exists for, so every type the seam
    /// maps to a metal pipeline must have a distinct name here.
    #[test]
    fn sel_names_cover_every_quant_type() {
        use crate::gguf::GgmlType as G;
        use crate::lowmem::SrcType;
        let all = [
            G::F32, G::F16, G::Q4_0, G::Q5_0, G::Q8_0, G::Q2_K, G::Q3_K, G::Q4_K,
            G::Q5_K, G::Q6_K, G::IQ1_S, G::IQ1_M, G::IQ2_XXS, G::IQ2_XS, G::IQ2_S,
            G::IQ3_XXS, G::IQ3_S, G::IQ4_NL, G::IQ4_XS,
        ];
        let mut by_sel: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
        for t in all {
            let sel = SrcType::Quant(t).qtype();
            if sel == u32::MAX {
                continue; // not wired to a metal pipeline yet — refused earlier
            }
            let name = super::sel_name(sel);
            assert!(
                !name.starts_with("sel"),
                "GGUF type {t:?} maps to selector {sel}, which has no attribution label"
            );
            let prev = by_sel.insert(sel, name.clone());
            assert!(prev.is_none() || prev.as_deref() == Some(name.as_str()));
        }
        let mut names: Vec<&String> = by_sel.values().collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "two selectors share one attribution label");
        assert_eq!(super::sel_name(SrcType::F16.qtype()), "f16");
        assert_eq!(super::sel_name(SrcType::BF16.qtype()), "bf16");
    }

    /// THE MIRROR GUARD — the class this lane exists to close.
    ///
    /// kernels.metal's `*Params` structs are mirrored by hand in Rust, in more
    /// than one file, and Metal binds them BY LAYOUT: a mirror missing a field
    /// makes the kernel read that field from uninitialised memory past the end
    /// of the struct. Nothing crashes. RoPE just rotates the wrong span and the
    /// model produces fluent, wrong text — which is what e61c260 shipped, on
    /// every GGUF model under `-b metal`, for three merges.
    ///
    /// The compiler cannot catch it. Rust forces field-completeness on the
    /// struct you EDIT; a duplicate declaration in another file stays
    /// internally consistent and silently disagrees with the Metal side. So the
    /// check has to be textual, and it has to run.
    ///
    /// Both sides are PARSED rather than listed here on purpose: a hand-written
    /// table of expected fields is one more mirror, and it would drift exactly
    /// like the one that caused this.
    #[test]
    fn mirror_structs_match_the_metal_source() {
        // metal type -> the Rust type a mirror must use for the same 4/8 bytes
        fn canon(t: &str) -> Option<&'static str> {
            Some(match t {
                "uint" => "u32",
                "int" => "i32",
                "float" => "f32",
                "ulong" => "u64",
                "long" => "i64",
                "ushort" => "u16",
                _ => return None,
            })
        }

        /// `struct Name { ... }` bodies from Metal source: (field, rust-type).
        fn parse_metal(src: &str) -> std::collections::HashMap<String, Vec<(String, String)>> {
            let mut out = std::collections::HashMap::new();
            let mut cur: Option<(String, Vec<(String, String)>)> = None;
            for line in src.lines() {
                let t = line.trim();
                if let Some(rest) = t.strip_prefix("struct ") {
                    if let Some(name) = rest.strip_suffix(" {") {
                        cur = Some((name.to_string(), Vec::new()));
                        continue;
                    }
                }
                if let Some((name, fields)) = cur.as_mut() {
                    if t == "};" || t == "}" {
                        out.insert(name.clone(), std::mem::take(fields));
                        cur = None;
                        continue;
                    }
                    // `uint head_dim;` possibly followed by a // comment
                    let decl = t.split("//").next().unwrap_or("").trim().trim_end_matches(';');
                    let mut w = decl.split_whitespace();
                    if let (Some(ty), Some(field), None) = (w.next(), w.next(), w.next()) {
                        if let Some(rt) = canon(ty) {
                            fields.push((field.to_string(), rt.to_string()));
                        }
                    }
                }
            }
            out
        }

        /// The same for Rust `struct Name { field: ty, }` (any indentation).
        fn parse_rust(src: &str) -> Vec<(String, Vec<(String, String)>)> {
            let mut out = Vec::new();
            let mut cur: Option<(String, Vec<(String, String)>)> = None;
            for line in src.lines() {
                let t = line.trim();
                if let Some(rest) = t.strip_prefix("struct ") {
                    if let Some(name) = rest.strip_suffix(" {") {
                        cur = Some((name.to_string(), Vec::new()));
                        continue;
                    }
                }
                if let Some((name, fields)) = cur.as_mut() {
                    if t == "}" {
                        out.push((name.clone(), std::mem::take(fields)));
                        cur = None;
                        continue;
                    }
                    if t.starts_with("///") || t.starts_with("//") || t.is_empty() {
                        continue;
                    }
                    if let Some((f, ty)) = t.trim_end_matches(',').split_once(':') {
                        fields.push((f.trim().to_string(), ty.trim().to_string()));
                    }
                }
            }
            out
        }

        let metal = parse_metal(include_str!("kernels.metal"));
        assert!(metal.len() > 10, "parsed only {} Metal structs — parser broken", metal.len());

        let mut checked = 0;
        let mut unmatched = Vec::new();
        for (file, src) in [
            ("src/gpu/metal.rs", include_str!("metal.rs")),
            ("src/lowmem/forward.rs", include_str!("../lowmem/forward.rs")),
        ] {
            for (name, rust_fields) in parse_rust(src) {
                let Some(metal_fields) = metal.get(&name) else {
                    // A Rust struct named *Params that matches NO Metal struct is
                    // the guard's blind spot: the loop above skips it silently, so
                    // renaming either side disables the check instead of failing
                    // it. Collect them and fail loudly rather than `continue`.
                    if name.ends_with("Params") {
                        unmatched.push(format!("{file}: {name}"));
                    }
                    continue;
                };
                checked += 1;
                assert_eq!(
                    &rust_fields, metal_fields,
                    "{file}: `struct {name}` has drifted from its kernels.metal definition. \
                     Metal binds these BY LAYOUT, so a mismatch makes the kernel read a field \
                     from uninitialised memory and produce fluent, wrong output — it does not \
                     crash. Update every Rust mirror when you change the Metal struct."
                );
            }
        }
        assert!(
            unmatched.is_empty(),
            "these Rust `*Params` structs mirror no struct in kernels.metal, so NOTHING \
             checks them: {unmatched:?}. Either the Metal struct was renamed (rename the \
             mirror to match) or this is not a kernel mirror (then it must not be called \
             *Params). A silently unchecked mirror is how e61c260 shipped."
        );
        // The guard needs its own negative control: if the parsers silently
        // matched nothing, every assertion above is vacuous and the test would
        // pass on a codebase where every mirror had drifted. The floor tracks
        // the real population (30 at the metal-deltanet lane) rather than a
        // token handful, so LOSING most of the mirrors fails here too — an
        // `>= 8` floor happily passes a run that checked eight of thirty.
        assert!(
            checked >= 24,
            "compared only {checked} mirrors — the duplicated *Params structs number about \
             thirty; the parser or the naming convention has changed and most mirrors are \
             now going unchecked"
        );
    }

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
    fn deltanet_states_lifecycle() {
        let device = Device::system_default().expect("metal device");
        // Interval-4 trunk of 8: layers 3 and 7 are attention.
        let layout = DeltaNetLayout {
            is_recurrent: (0..8).map(|i| (i + 1) % 4 != 0).collect(),
            conv_elems: 6,
            delta_elems: 10,
        };
        let st = DeltaNetStates::new(&device, &layout);
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

    /// The two-kind state schedule (T3): dense is all-Kv; the hybrid follows
    /// its map index-for-index; and the DeltaNetStates slots agree with the
    /// schedule layer by layer — the invariant every session construction
    /// leans on.
    #[test]
    fn state_schedule_is_two_kind_and_agrees_with_states() {
        use LayerStateKind::*;
        assert!(state_schedule(4, None).iter().all(|k| *k == Kv), "dense = all Kv");

        let layout = DeltaNetLayout {
            is_recurrent: (0..8).map(|i| (i + 1) % 4 != 0).collect(),
            conv_elems: 6,
            delta_elems: 10,
        };
        let sched = state_schedule(8, Some(&layout));
        assert_eq!(sched.len(), 8);
        assert_eq!(sched.iter().filter(|k| **k == Kv).count(), 2);
        for (l, k) in sched.iter().enumerate() {
            assert_eq!(*k == Recurrent, layout.is_recurrent[l], "layer {l}");
        }
        let device = Device::system_default().expect("metal device");
        let st = DeltaNetStates::new(&device, &layout);
        for (l, k) in sched.iter().enumerate() {
            assert_eq!(st.layers[l].is_some(), *k == Recurrent, "slot kind, layer {l}");
        }
    }

    /// The KV stub (this lane): a Recurrent layer gets a one-element cache,
    /// never a cap × kv_dim one — nothing reads it, because that layer's
    /// forward branch binds no cache. The vector is NOT compacted: index l is
    /// still layer l, which is what every call site assumes.
    #[test]
    fn kv_cache_is_stubbed_exactly_on_recurrent_layers() {
        use LayerStateKind::*;
        let (slots, kvd) = (4096usize, 4096usize);

        // Dense: every layer keeps its full cache, byte for byte what the
        // backend allocated before the stub existed.
        let dense = state_schedule(8, None);
        assert_eq!(kv_cache_elems(&dense, slots, kvd), vec![slots * kvd; 8]);

        // Hybrid: qwen35's one-in-four map (layers 3 and 7 are attention).
        let layout = DeltaNetLayout {
            is_recurrent: (0..8).map(|i| (i + 1) % 4 != 0).collect(),
            conv_elems: 6,
            delta_elems: 10,
        };
        let sched = state_schedule(8, Some(&layout));
        let elems = kv_cache_elems(&sched, slots, kvd);
        assert_eq!(elems.len(), 8, "one slot per trunk layer, not compacted");
        for (l, n) in elems.iter().enumerate() {
            match sched[l] {
                Recurrent => assert_eq!(*n, 1, "layer {l} stubbed"),
                Kv => assert_eq!(*n, slots * kvd, "layer {l} keeps its ring"),
            }
        }
        // The win, in the shape the plan states it: KV elements drop to the
        // attention layers' share (2 of 8 here), both caches.
        let before: usize = 8 * slots * kvd;
        let after: usize = elems.iter().sum();
        // Exactly the two attention layers' rings, plus 6 stub elements —
        // a quarter of what the backend allocated before this lane.
        assert_eq!(after, 2 * slots * kvd + 6);
        assert!(after < before / 3, "stub must be a real reduction");
    }

    /// THE ONE-HOT BIT-EXACT GATE (ruling d04120ac, made FIRST-order by 4b25ec14).
    ///
    /// The bit-exact dequant oracle in pool.rs covers `lm_dequant_*` — the
    /// per-element path — and nothing covered `dot_wx` and the run/staging
    /// machinery the decode matvec actually uses. That gap is not theoretical:
    /// a wrong quarter index mapping in this lane produced fluent garbage
    /// ("the capital of Thailand is called _Annaka_ Annaka Annaka") with no
    /// crash and no NaN, and it was caught by a 16-cell token gate AFTER a speed
    /// number had already been published off it.
    ///
    /// The trick that makes this exact rather than tolerance-based: drive the
    /// REAL matvec kernel with an x that is 1.0 at a single column and 0.0
    /// everywhere else. Every accumulator then carries exactly one nonzero term
    /// (0 + a = a and a * 0 = 0 are exact in IEEE), so y[row] must equal the CPU
    /// reference's dequantised weight at that column BIT-FOR-BIT — whatever
    /// decomposition, lane mapping or threadgroup staging the kernel uses
    /// internally. It pins index remapping independently of any accumulation
    /// question, which is precisely the class of defect that slipped through.
    ///
    /// The reference is `gguf::dequant_row_ref`, which this lane must not touch
    /// and which has its own bit-exact gate against ggml semantics — so this is a
    /// real chain to an independent oracle, not a mirror of the code under test.
    ///
    /// MARKED #[ignore] AND WHY, because it is not a slow-test exemption: this
    /// test does real GPU work, and running Metal work in parallel with the
    /// other GPU tests makes THEM fail — pool.rs's dequant oracle and the
    /// deltanet l2norm oracle both read ZEROS out of their own output buffers.
    /// The suite is 82/0 without this test and fails deterministically with it,
    /// while this test passes alone and its negative control fires, so the
    /// arithmetic on both sides is fine and what is exposed is that those tests
    /// are not concurrency-safe. Fixing that spans src/lowmem/pool.rs, outside
    /// this lane's boundary, so it is a challenge and not a silent edit. Until
    /// it is ruled, this runs in the `--ignored` pass, which this lane's gates
    /// require anyway.
    #[test]
    #[ignore = "GPU tests are not concurrency-safe; see the lane challenge"]
    fn onehot_probe_matches_the_cpu_reference_bit_for_bit() {
        use crate::gguf::GgmlType;
        let device = Device::system_default().expect("metal device");
        // One entry per type whose read path this arc rewrites. The harness is
        // type-agnostic apart from two things: the LM_W_QTYPE selector, and which
        // bytes of a synthetic block must be constrained so the block is VALID.
        // Everything else — grid indices, scale nibbles, quant bits — is random,
        // and for both IQ1 types that is provably safe: their grid index is 11
        // bits wide (byte | 3 bits) and lm_iq1s_grid has exactly 2048 entries, so
        // no random byte pattern can read out of bounds.
        let types: &[(u32, crate::gguf::GgmlType, usize)] = &[
            (5, crate::gguf::GgmlType::Q6_K, 210),
            (17, crate::gguf::GgmlType::IQ1_S, 50),
            (18, crate::gguf::GgmlType::IQ1_M, 56),
        ];
        for &(SEL, TY, BLK) in types {
        let precise = CompileOptions::new();
        precise.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(&shader_source(FLASH_HEAD_DIM), &precise)
            .expect("library");
        let consts = FunctionConstantValues::new();
        consts.set_constant_value_at_index(&SEL as *const u32 as *const _, MTLDataType::UInt, 25);
        let f = lib.get_function("matvec", Some(consts)).expect("matvec");
        let pipe = device
            .new_compute_pipeline_state_with_function(&f)
            .expect("pipeline");

        // in_dim 512 exercises the short-row case (fewer 32-element runs than a
        // simdgroup has lanes); 2048 is the lm_head width where every lane is fed.
        for &in_dim in &[512usize, 2048usize] {
            let rows = 2usize;
            let sb = in_dim / 256;
            let row_bytes = sb * BLK;
            let mut w = vec![0u8; rows * row_bytes];
            // Deterministic filler. Every ql/qh/scale byte pattern is a valid
            // Q6_K block; only `d` is an f16 and a random one would be NaN ~3%
            // of the time, so it is written into [1,2) — finite, varied, and the
            // only type-specific knowledge this harness needs.
            let mut st = 0x2545_F491_4F6C_DD1Du64;
            let mut next = || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                st
            };
            for r in 0..rows {
                for b in 0..sb {
                    let base = r * row_bytes + b * BLK;
                    for i in 0..BLK {
                        w[base + i] = (next() >> 24) as u8;
                    }
                    // Keep the block's f16 scale FINITE. A random f16 is NaN or
                    // Inf about 3% of the time and NaN != NaN would turn this
                    // gate into a coin flip. [1,2) is finite and still varied.
                    let d_bits: u16 = 0x3C00 | ((next() >> 20) as u16 & 0x03FF);
                    let put = |w: &mut Vec<u8>, off: usize, bits: u16| {
                        w[base + off] = (bits & 0xFF) as u8;
                        w[base + off + 1] = (bits >> 8) as u8;
                    };
                    match SEL {
                        5 => put(&mut w, 208, d_bits),   // Q6_K: d at the end
                        17 => put(&mut w, 0, d_bits),    // IQ1_S: d first
                        // IQ1_M has NO d field: its f16 is assembled from the TOP
                        // NIBBLE of each of the four scale u16s, and the top
                        // nibble of the LAST one supplies sign + the high
                        // exponent bits. Forcing it to 3 gives exponent 011xx —
                        // finite for every value the other three nibbles take,
                        // so the rest stays random.
                        18 => w[base + 55] = (w[base + 55] & 0x0F) | 0x30,
                        _ => unreachable!("no block-validity rule for selector {SEL}"),
                    }
                }
            }
            let mut expect = vec![0f32; rows * in_dim];
            for r in 0..rows {
                dequant_row_ref_for_test(
                    TY,
                    &w[r * row_bytes..(r + 1) * row_bytes],
                    &mut expect[r * in_dim..(r + 1) * in_dim],
                );
            }

            let wbuf = device.new_buffer_with_data(
                w.as_ptr() as *const _,
                w.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let bias = f16_empty_buffer(&device, rows.max(8));
            unsafe { std::ptr::write_bytes(bias.contents() as *mut u8, 0, 2 * rows.max(8)) };
            // Every one of these types maps indices with PERIOD 256 — the
            // superblock — so testing
            // every column of a long row re-tests the same mapping over and over.
            // Two superblocks' worth of columns is full coverage of the mapping,
            // and in_dim 512 vs 2048 covers the two lane regimes (fewer 32-element
            // runs than a simdgroup has lanes, and more).
            //
            // Keeping the sweep short is not just speed: the first version
            // committed and waited 2560 times and monopolised the device hard
            // enough that the pre-existing pool.rs dequant oracle read ZEROS out
            // of its own output buffer when the suite ran both in parallel. Its
            // arithmetic was not wrong and neither was this test's; this test was
            // a bad citizen. (A single-command-buffer version SIGSEGVs inside
            // Metal at any size — not root-caused, not shipped, and not needed
            // once the sweep is bounded by the mapping's period.)
            let cols: Vec<usize> = if in_dim <= 512 {
                (0..in_dim).collect()
            } else {
                (0..256).chain(in_dim - 256..in_dim).collect()
            };
            let x = f32_buffer(&device, in_dim);
            let y = f32_buffer(&device, rows.max(8));
            let queue = device.new_command_queue();
            let p = MatvecParams { in_dim: in_dim as u32, out_dim: rows as u32 };
            for &c in &cols {
                unsafe {
                    let xp = x.contents() as *mut f32;
                    std::ptr::write_bytes(xp, 0, in_dim * 4);
                    *xp.add(c) = 1.0;
                }
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&wbuf), 0);
                enc.set_buffer(1, Some(&bias), 0);
                enc.set_buffer(2, Some(&x), 0);
                enc.set_buffer(3, Some(&y), 0);
                enc.set_bytes(
                    4,
                    size_of::<MatvecParams>() as u64,
                    &p as *const _ as *const _,
                );
                dispatch_simdgroup_rows(enc, rows as u32);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let got = unsafe { std::slice::from_raw_parts(y.contents() as *const f32, rows) };
                for r in 0..rows {
                    // The kernel finishes with `sum + bias`, and the bias here is
                    // zero — so a weight that dequantises to NEGATIVE zero comes
                    // back as POSITIVE zero, because (-0.0) + (+0.0) = +0.0 in
                    // IEEE. That is the kernel's real arithmetic, not a defect,
                    // and the gate caught it on its first run (in_dim 512, col 17,
                    // GPU +0 vs reference -0). Adding the same zero to the
                    // reference reproduces the kernel's own final operation
                    // instead of special-casing zeros: every nonzero value is
                    // unchanged by it, so the comparison stays bit-exact.
                    let want = expect[r * in_dim + c] + 0.0;
                    assert_eq!(
                        got[r].to_bits(),
                        want.to_bits(),
                        "in_dim {in_dim} row {r} col {c}: GPU {} vs CPU reference {} \
                         — the decode weight-read path disagrees with gguf::dequant_row_ref \
                         at a single column, which is an INDEX defect, not an accumulation one",
                        got[r],
                        want
                    );
                }
            }
        }
        }
    }

    /// Thin wrapper so the gate names its reference explicitly: this lane may
    /// read `gguf::dequant_row_ref` but must never change it — it is the other
    /// half of this gate's independence.
    fn dequant_row_ref_for_test(ty: crate::gguf::GgmlType, src: &[u8], out: &mut [f32]) {
        crate::gguf::dequant_row_ref(ty, src, out)
    }

    /// The decode-attention geometry lives in two languages: kernels.metal
    /// hard-codes it as #defines and Rust dispatches against consts that must
    /// equal them, with nothing but a comment linking the pair. This lane made
    /// that pairing load-bearing — MAX_DEC_DPT decides how many output dims one
    /// thread carries, so a Rust const LARGER than the shader's would dispatch
    /// head_dims the kernel cannot hold and write past acc_red, which on a GPU
    /// has no symptom at all. Parsed, not listed: a hand-written table would be
    /// one more mirror to drift.
    #[test]
    fn decode_geometry_defines_match_the_metal_source() {
        // kvd only feeds the FA_KVD preamble; the #defines below are literal.
        let src = shader_source(FLASH_HEAD_DIM);
        let define = |name: &str| -> usize {
            let needle = format!("#define {name} ");
            let line = src
                .lines()
                .map(str::trim_start)
                .find(|l| l.starts_with(&needle))
                .unwrap_or_else(|| {
                    panic!("kernels.metal has no `#define {name}` — the Rust const \
                            that mirrors it is now pinned to nothing")
                });
            line[needle.len()..]
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("`#define {name}` is not a plain integer"))
        };
        assert_eq!(define("ATTN_SPLIT"), ATTN_SPLIT, "ATTN_SPLIT");
        assert_eq!(define("DEC_TG"), DEC_TG, "DEC_TG");
        assert_eq!(define("MAX_DEC_DPT"), MAX_DEC_DPT, "MAX_DEC_DPT");
        assert_eq!(define("MAX_GQA_CHUNK"), MAX_GQA_CHUNK, "MAX_GQA_CHUNK");
        assert_eq!(DEC_MAX_HD, DEC_TG * MAX_DEC_DPT, "the dispatch ceiling is the product");
    }

    /// acc_red is the one scratch that had to grow for head_dim > DEC_TG, and it
    /// is ALLOCATED in Rust while INDEXED in Metal. Both regimes pinned: the
    /// hd <= DEC_TG sizing must not move (every dense cell in the gate table was
    /// captured through it), and hd = 256 must get one entry per dim.
    #[test]
    fn acc_red_covers_both_head_dim_regimes() {
        // hd <= DEC_TG: P = DEC_TG/hd lanes x hd dims = DEC_TG entries. This is
        // the pre-existing sizing, unchanged.
        for hd in [64usize, 96, 128] {
            assert_eq!(acc_red_elems(4, hd), DEC_TG * 5, "hd {hd} sizing moved");
        }
        // hd > DEC_TG: one position lane, every dim.
        assert_eq!(acc_red_elems(4, 256), 256 * 5);
        // Enough for the largest head_dim the dispatch will ever admit.
        assert!(acc_red_elems(1, DEC_MAX_HD) >= DEC_MAX_HD);
        // Odd stride (the shader's ACC_STRIDE = GQA_CHUNK | 1) is preserved.
        assert_eq!(acc_red_elems(1, 128) / DEC_TG, 1);
        assert_eq!(acc_red_elems(8, 128) / DEC_TG, 9);
    }

    /// The one canonical meta→layout translation carries the real sizes and
    /// only the trunk's map.
    #[test]
    fn deltanet_layout_from_meta_is_trunk_shaped() {
        let meta = crate::gguf::Qwen35Meta {
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
        let layout = DeltaNetLayout::from_meta(&meta);
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
mod deltanet_kernel_oracle {
    //! GPU kernels vs lane B's CPU reference (src/deltanet_ref.rs),
    //! bit-for-bit. Same doctrine as the quant oracle: the reference is the
    //! subject, the GPU is the thing under test, and a negative control proves
    //! the comparison can fail.
    use crate::gpu::metal as gpu;
    use crate::deltanet_ref as rf;
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
    /// Swap i and j per head. The GPU state is transposed relative to the
    /// reference (see delta_decode_step's layout note), so the oracle permutes
    /// on the way in and back on the way out; the permutation is its own
    /// inverse, which is why one helper serves both directions. Note this is the
    /// DELTA state only — gpu_conv's state is untouched by this lane.
    fn permute_delta_state(d: rf::DeltaDims, src: &[f32]) -> Vec<f32> {
        let s = d.d_state;
        let mut out = vec![0f32; src.len()];
        for h in 0..d.n_v_heads {
            for j in 0..s {
                for i in 0..s {
                    out[h * s * s + i * s + j] = src[h * s * s + j * s + i];
                }
            }
        }
        out
    }

    /// The oracle's delta step. `reference_layout = true` is the real path: the
    /// caller hands reference-layout state and gets reference-layout state back.
    /// `false` uploads the state RAW, i.e. in the pre-transpose orientation —
    /// which is what the negative control needs, because a half-done transpose
    /// (kernel moved, caller not) is exactly that.
    fn gpu_delta(
        d: rf::DeltaDims,
        state0: &[f32],
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        g: &[Vec<f32>],
        beta: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        gpu_delta_layout(d, state0, q, k, v, g, beta, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn gpu_delta_layout(
        d: rf::DeltaDims,
        state0: &[f32],
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        g: &[Vec<f32>],
        beta: &[Vec<f32>],
        reference_layout: bool,
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
        let uploaded = if reference_layout {
            permute_delta_state(d, state0)
        } else {
            state0.to_vec()
        };
        let st =
            device.new_buffer_with_data(uploaded.as_ptr() as *const _, bytes(&uploaded), shared);
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
        let raw =
            unsafe { std::slice::from_raw_parts(st.contents() as *const f32, state0.len()) }.to_vec();
        let final_state = if reference_layout { permute_delta_state(d, &raw) } else { raw };
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
        let mrope_thetas = |bases: [f32; 4], indep_sects: bool, is_imrope: bool| -> Vec<f32> {
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
                    let t = if is_imrope {
                        // The branch qwen35 ACTUALLY takes: llama-model.cpp:2708
                        // maps LLM_ARCH_QWEN35 to LLAMA_ROPE_TYPE_IMROPE, so the
                        // interleaved selector runs, not the contiguous one.
                        if sector % 3 == 1 && sector < 3 * sections[1] {
                            th[1]
                        } else if sector % 3 == 2 && sector < 3 * sections[2] {
                            th[2]
                        } else if sector % 3 == 0 && sector < 3 * sections[0] {
                            th[0]
                        } else {
                            th[3]
                        }
                    } else if sector >= sections[0] && sector < sec_w {
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
            // Both selectors, because the equivalence argument covers both and
            // only one of them is the one qwen35 runs.
            let m = mrope_thetas([base; 4], false, false);
            let m_i = mrope_thetas([base; 4], false, true);
            for (i, ((a, b), c)) in plain.iter().zip(&m).zip(&m_i).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "pos {pos} pair {i}: plain rope {a} vs mrope {b} — the equivalence this \
                     lane relies on to ship NO sectioned kernel does not hold"
                );
                assert_eq!(
                    a.to_bits(),
                    c.to_bits(),
                    "pos {pos} pair {i}: plain rope {a} vs INTERLEAVED mrope {c} — this is the \
                     selector qwen35 actually uses (LLAMA_ROPE_TYPE_IMROPE)"
                );
            }
            // The control: the vision path must NOT agree, or this proves nothing.
            let vision = mrope_thetas([base; 4], true, true);
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
        let mut meta = crate::gguf::Qwen35Meta {
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
        assert!(meta.check_rope_sections().is_ok(), "the real 27B layout must pass");

        meta.rope_sections = [11, 11, 10, 4];
        let err = meta.check_rope_sections().unwrap_err();
        assert!(err.contains("vision"), "the refusal must name why: {err}");

        meta.rope_sections = [0, 0, 0, 0];
        assert!(meta.check_rope_sections().is_err(), "all-zero sections must refuse");
    }

    fn gpu_delta_gates(alpha: &[f32], beta_in: &[f32], a: &[f32], dt: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let device = Device::system_default().expect("Metal device required");
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(&gpu::shader_source(128), &opts)
            .expect("kernels.metal compiles");
        let f = lib.get_function("delta_gates", None).expect("delta_gates");
        let pipe = device.new_compute_pipeline_state_with_function(&f).expect("pipeline");
        let queue = device.new_command_queue();
        let shared = MTLResourceOptions::StorageModeShared;
        let mk = |v: &[f32]| device.new_buffer_with_data(
            v.as_ptr() as *const _, std::mem::size_of_val(v) as u64, shared);
        let n = alpha.len();
        let (ab, bb, aab, dtb) = (mk(alpha), mk(beta_in), mk(a), mk(dt));
        let gout = device.new_buffer((n * 4) as u64, shared);
        let bout = device.new_buffer((n * 4) as u64, shared);
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        for (i, b) in [&ab, &bb, &aab, &dtb, &gout, &bout].iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        let n32 = n as u32;
        enc.set_bytes(6, 4, &n32 as *const u32 as *const _);
        let one = 1u32; // this helper drives one token's worth of heads
        enc.set_bytes(7, 4, &one as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(((n + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let rd = |b: &metal::Buffer| unsafe {
            std::slice::from_raw_parts(b.contents() as *const f32, n)
        }.to_vec();
        (rd(&gout), rd(&bout))
    }

    /// The EXACT half. softplus's own x > 20 shortcut is what makes it
    /// possible: above the threshold the reference returns x UNCHANGED, so no
    /// transcendental runs on either side and g = a·(alpha+dt_bias) is pure
    /// arithmetic. sigma is exact at the same three arguments as the out-gate.
    /// This pins the pairing of the four inputs — the thing most likely to be
    /// wrong — with no tolerance at all.
    #[test]
    fn delta_gates_are_exact_above_the_softplus_shortcut() {
        let n = 16; // the 2B's n_v_heads
        let mut seed = 0x5150_0001u32;
        let dt: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
        // alpha chosen so alpha + dt_bias is comfortably past 20 in every slot.
        let alpha: Vec<f32> = (0..n).map(|i| 40.0 + i as f32).collect();
        let a: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 2.0).collect();
        let beta_in: Vec<f32> = (0..n)
            .map(|i| match i % 3 { 0 => 0.0, 1 => 100.0, _ => -100.0 })
            .collect();
        let (g, beta) = gpu_delta_gates(&alpha, &beta_in, &a, &dt);
        for h in 0..n {
            let want_g = rf::delta_gate(alpha[h], dt[h], a[h]);
            assert_eq!(want_g.to_bits(), g[h].to_bits(), "g[{h}]: {want_g} vs {}", g[h]);
            let want_b = rf::sigmoid(beta_in[h]);
            assert_eq!(want_b.to_bits(), beta[h].to_bits(), "beta[{h}]");
        }
        // The shortcut must actually be what is being exercised, or this test
        // is silently a transcendental comparison that happened to pass.
        assert!(alpha.iter().zip(&dt).all(|(x, b)| x + b > 20.0));
    }

    /// The BOUNDED half, below the shortcut where log and exp both run. Bound
    /// is measured, not chosen: nudge exp by one rounding inside the
    /// reference's own softplus and sigma, and require the GPU inside 4x the
    /// resulting self-drift.
    #[test]
    fn delta_gates_bounded_below_the_shortcut() {
        let n = 48; // the 27B's n_v_heads
        let mut seed = 0x5150_0002u32;
        let alpha: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 4.0).collect();
        let dt: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
        let a: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 2.0).collect();
        let beta_in: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 6.0).collect();
        let (g, beta) = gpu_delta_gates(&alpha, &beta_in, &a, &dt);

        let eps = 1.0 + f32::EPSILON;
        let mut worst = 0f32;
        let mut yard = 0f32;
        for h in 0..n {
            let x = alpha[h] + dt[h];
            assert!(x <= 20.0, "slot {h} took the shortcut — this test must exercise log/exp");
            let want_g = rf::delta_gate(alpha[h], dt[h], a[h]);
            let nudged_g = a[h] * (1.0 + x.exp() * eps).ln();
            worst = worst.max(rel_err(want_g, g[h]));
            yard = yard.max(rel_err(want_g, nudged_g));
            let want_b = rf::sigmoid(beta_in[h]);
            let nudged_b = 1.0 / (1.0 + (-beta_in[h]).exp() * eps);
            worst = worst.max(rel_err(want_b, beta[h]));
            yard = yard.max(rel_err(want_b, nudged_b));
        }
        assert!(yard > 0.0, "the one-rounding nudge changed nothing — probe is broken");
        assert!(worst <= 4.0 * yard, "gpu drift {worst:e} exceeds 4x the measured floor {yard:e}");
    }

    /// One deltanet block on the GPU: the six kernels in llama.cpp's order,
    /// one command buffer per token. Dispatches inside a compute encoder are
    /// serial by default, so the chain's ordering needs no barriers here.
    #[allow(clippy::too_many_arguments)]
    fn gpu_delta_block(
        d: rf::DeltaDims,
        conv1d: &[f32],
        a: &[f32],
        dt: &[f32],
        ssm_norm: &[f32],
        qkv: &[Vec<f32>],
        z: &[Vec<f32>],
        alpha: &[Vec<f32>],
        beta_p: &[Vec<f32>],
        eps: f32,
    ) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        let device = Device::system_default().expect("Metal device required");
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(&gpu::shader_source(128), &opts)
            .expect("kernels.metal compiles");
        let pipe = |n: &str| {
            let f = lib.get_function(n, None).unwrap_or_else(|_| panic!("{n}"));
            device.new_compute_pipeline_state_with_function(&f).expect("pipeline")
        };
        let (p_gates, p_conv, p_l2, p_delta, p_norm) = (
            pipe("delta_gates"),
            pipe("ssm_conv_decode"),
            pipe("l2norm_rows"),
            pipe("delta_decode_step"),
            pipe("gated_output_norm"),
        );
        let queue = device.new_command_queue();
        let shared = MTLResourceOptions::StorageModeShared;
        let up = |v: &[f32]| device.new_buffer_with_data(
            v.as_ptr() as *const _, std::mem::size_of_val(v) as u64, shared);
        let (s_dim, hv, hk) = (d.d_state, d.n_v_heads, d.n_k_heads);
        let key_dim = s_dim * hk;
        let c_all = d.conv_channels();

        let (b_conv1d, b_a, b_dt, b_norm) = (up(conv1d), up(a), up(dt), up(ssm_norm));
        let conv_state = gpu::f32_zero_buffer(&device, d.conv_state_elems());
        let delta_state = gpu::f32_zero_buffer(&device, d.delta_state_elems());
        let b_conv_out = gpu::f32_buffer(&device, c_all);
        let b_g = gpu::f32_buffer(&device, hv);
        let b_beta = gpu::f32_buffer(&device, hv);
        let b_out = gpu::f32_buffer(&device, s_dim * hv);

        let mut outs = Vec::new();
        for t in 0..qkv.len() {
            let (b_qkv, b_z, b_alpha, b_bp) = (up(&qkv[t]), up(&z[t]), up(&alpha[t]), up(&beta_p[t]));
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();

            // 1. the two per-head scalars
            enc.set_compute_pipeline_state(&p_gates);
            for (i, b) in [&b_alpha, &b_bp, &b_a, &b_dt, &b_g, &b_beta].iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            let hv32 = hv as u32;
            enc.set_bytes(6, 4, &hv32 as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(64, 1, 1));

            // 2. depthwise conv + silu, rolling the conv state
            #[repr(C)]
            struct CP { channels: u32, d_conv: u32 }
            let cp = CP { channels: c_all as u32, d_conv: d.d_conv as u32 };
            enc.set_compute_pipeline_state(&p_conv);
            enc.set_buffer(0, Some(&conv_state), 0);
            enc.set_buffer(1, Some(&b_qkv), 0);
            enc.set_buffer(2, Some(&b_conv1d), 0);
            enc.set_buffer(3, Some(&b_conv_out), 0);
            enc.set_bytes(4, std::mem::size_of::<CP>() as u64, &cp as *const CP as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(((c_all + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));

            // 3. l2-normalise q and k per K head, in place, on their own slices
            let s32 = s_dim as u32;
            for off in [0u64, (key_dim * 4) as u64] {
                enc.set_compute_pipeline_state(&p_l2);
                enc.set_buffer(0, Some(&b_conv_out), off);
                enc.set_bytes(1, 4, &s32 as *const u32 as *const _);
                enc.set_bytes(2, 4, &eps as *const f32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(hk as u64, 1, 1), MTLSize::new(256, 1, 1));
            }

            // 4. the delta rule; q/k/v are three views of the conv output
            #[repr(C)]
            struct DP { d_state: u32, n_v_heads: u32, group: u32 }
            let dp = DP { d_state: s32, n_v_heads: hv32, group: (hv / hk) as u32 };
            enc.set_compute_pipeline_state(&p_delta);
            enc.set_buffer(0, Some(&delta_state), 0);
            enc.set_buffer(1, Some(&b_conv_out), 0);
            enc.set_buffer(2, Some(&b_conv_out), (key_dim * 4) as u64);
            enc.set_buffer(3, Some(&b_conv_out), (2 * key_dim * 4) as u64);
            enc.set_buffer(4, Some(&b_g), 0);
            enc.set_buffer(5, Some(&b_beta), 0);
            enc.set_buffer(6, Some(&b_out), 0);
            enc.set_bytes(7, std::mem::size_of::<DP>() as u64, &dp as *const DP as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(((s_dim * hv + 63) / 64) as u64, 1, 1), MTLSize::new(64, 1, 1));

            // 5. per-head RMSNorm gated by silu(z), in place on the output
            enc.set_compute_pipeline_state(&p_norm);
            enc.set_buffer(0, Some(&b_out), 0);
            enc.set_buffer(1, Some(&b_norm), 0);
            enc.set_buffer(2, Some(&b_z), 0);
            enc.set_bytes(3, 4, &s32 as *const u32 as *const _);
            enc.set_bytes(4, 4, &eps as *const f32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(hv as u64, 1, 1), MTLSize::new(256, 1, 1));

            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            outs.push(
                unsafe { std::slice::from_raw_parts(b_out.contents() as *const f32, s_dim * hv) }
                    .to_vec(),
            );
        }
        let rd = |b: &metal::Buffer, n: usize| unsafe {
            std::slice::from_raw_parts(b.contents() as *const f32, n)
        }.to_vec();
        // The delta state comes back in the GPU's TRANSPOSED layout and the
        // caller compares it against the reference's, so permute it here. The
        // conv state is not transposed and must not be touched. The upload side
        // needs nothing: this oracle starts the delta state at zero, and zero is
        // layout-invariant. (§7 enumerated gpu_delta/delta_inputs as the sites;
        // this third one is the same class and is inside the boundary, so it is
        // implementation judgment rather than a plan divergence worth a
        // challenge — but it is the site the memo's list missed, and the
        // existing test is what found it.)
        (
            outs,
            rd(&conv_state, d.conv_state_elems()),
            permute_delta_state(d, &rd(&delta_state, d.delta_state_elems())),
        )
    }

    /// THE SINGLE-LAYER IDENTITY GATE: one real deltanet block, on real
    /// weights, through the six kernels composed in llama.cpp's order, against
    /// lane B's reference doing the same thing on the CPU.
    ///
    /// Why this exists when every kernel already has its own oracle: unit gates
    /// prove each kernel computes what it claims IN ISOLATION. They structurally
    /// cannot catch the errors that actually kill a port — feeding conv the
    /// wrong slice, splitting q/k/v at the wrong offsets, forgetting beta's
    /// sigmoid, applying the gated norm before the delta step, using the K-head
    /// count where the V-head count belongs. Those are composition errors, and
    /// only a composed comparison sees them.
    ///
    /// The projections are computed on the CPU for BOTH sides from the same
    /// real (dequantized) weights, so the two paths are fed byte-identical
    /// inputs and the only thing under test is the deltanet chain itself.
    /// Ordinary matmuls are covered elsewhere and would just add noise here.
    ///
    /// Multi-token, because both states roll: a block that is right for token 0
    /// and wrong afterwards is the exact failure a single-shot test misses.
    #[test]
    #[ignore]
    fn deltanet_block_matches_reference_on_real_weights() {
        use crate::lowmem::LowMemSource;
        let Some(path) = crate::lowmem::tests_qwen35_gguf() else {
            panic!("Qwen3.5-2B GGUF not in the HF cache — this gate needs the real file")
        };
        let src = LowMemSource::open(&path).expect("opens");
        let meta = src.qwen35().expect("qwen35 meta");
        assert!(meta.is_recurrent[0], "layer 0 of the 2B is a linear block");
        let d = rf::DeltaDims {
            d_state: meta.d_state,
            n_v_heads: meta.dt_rank,
            n_k_heads: meta.n_group,
            d_conv: meta.d_conv,
        };
        let (hidden, eps) = (2048usize, 1e-6f32);
        let rd = |n: &str| src.read_f32(n).unwrap_or_else(|e| panic!("{n}: {e}"));
        let p = "model.layers.0";
        let w_qkv = rd(&format!("{p}.gguf.attn_qkv.weight"));
        let w_z = rd(&format!("{p}.gguf.attn_gate.weight"));
        let w_alpha = rd(&format!("{p}.gguf.ssm_alpha.weight"));
        let w_beta = rd(&format!("{p}.gguf.ssm_beta.weight"));
        let conv1d = rd(&format!("{p}.gguf.ssm_conv1d.weight"));
        let ssm_a = rd(&format!("{p}.gguf.ssm_a"));
        let dt_bias = rd(&format!("{p}.gguf.ssm_dt.bias"));
        let ssm_norm = rd(&format!("{p}.gguf.ssm_norm.weight"));

        // Shapes are row-major [rows, cols] (gguf.rs:280 reverses ne), so a
        // projection is W[out][in] and conv1d is [channel][tap].
        assert_eq!(w_qkv.len(), d.conv_channels() * hidden);
        assert_eq!(conv1d.len(), d.conv_channels() * d.d_conv);
        assert_eq!(ssm_a.len(), d.n_v_heads);
        assert_eq!(ssm_norm.len(), d.d_state);

        let matvec = |w: &[f32], x: &[f32], out_dim: usize| -> Vec<f32> {
            (0..out_dim)
                .map(|o| {
                    let row = &w[o * hidden..(o + 1) * hidden];
                    row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()
                })
                .collect()
        };

        let steps = 4;
        let mut seed = 0x4C41_5945u32;
        let xs: Vec<Vec<f32>> = (0..steps)
            .map(|_| (0..hidden).map(|_| lcg(&mut seed) * 0.5).collect())
            .collect();
        let qkv: Vec<Vec<f32>> = xs.iter().map(|x| matvec(&w_qkv, x, d.conv_channels())).collect();
        let z: Vec<Vec<f32>> = xs.iter().map(|x| matvec(&w_z, x, d.d_inner())).collect();
        let alpha: Vec<Vec<f32>> = xs.iter().map(|x| matvec(&w_alpha, x, d.n_v_heads)).collect();
        let beta_p: Vec<Vec<f32>> = xs.iter().map(|x| matvec(&w_beta, x, d.n_v_heads)).collect();

        // ---- the reference block ----
        let mut ref_conv = vec![0f32; d.conv_state_elems()];
        let mut ref_state = vec![0f32; d.delta_state_elems()];
        let mut want: Vec<Vec<f32>> = Vec::new();
        for t in 0..steps {
            let conv_out = rf::conv_step(&d, &mut ref_conv, &qkv[t], &conv1d);
            let (q, k, v) = rf::split_qkv(&d, &conv_out, eps);
            let g: Vec<f32> = (0..d.n_v_heads)
                .map(|h| rf::delta_gate(alpha[t][h], dt_bias[h], ssm_a[h]))
                .collect();
            let b: Vec<f32> = beta_p[t].iter().map(|&x| rf::sigmoid(x)).collect();
            let mut o = rf::delta_decode_step(&d, &mut ref_state, &q, &k, &v, &g, &b);
            rf::gated_output_norm(&d, &mut o, &ssm_norm, &z[t], eps);
            want.push(o);
        }

        // ---- the same block on the GPU ----
        let (got, gpu_conv, gpu_state) =
            gpu_delta_block(d, &conv1d, &ssm_a, &dt_bias, &ssm_norm, &qkv, &z, &alpha, &beta_p, eps);

        // THE PROBE MUST MODEL EVERY DIVERGENCE THE GPU ACTUALLY HAS, or the
        // ratio is meaningless. My first version nudged only the decay and the
        // state came out at 6x it — which looked like a composition error and
        // was not: this chain ALSO differs in l2_norm's and the gated norm's
        // accumulation (f64 in the reference, f32 on a GPU that has no f64) and
        // in sigma on beta. So the probe below is the same block computed with
        // the GPU's OWN precision choices — f32 reductions throughout, exp moved
        // one rounding — and the GPU is required to sit within 4x its distance
        // from the f64 reference. Anything the probe does not model would still
        // show up, which is the point.
        let l2_f32 = |x: &mut [f32], eps: f32| {
            let sum: f32 = x.iter().map(|&v| v * v).sum();
            let scale = 1.0 / sum.sqrt().max(eps);
            for v in x.iter_mut() {
                *v *= scale;
            }
        };
        let nudge = 1.0 + f32::EPSILON;
        let mut nudged_conv = vec![0f32; d.conv_state_elems()];
        let mut nudged_state = vec![0f32; d.delta_state_elems()];
        let mut nudged: Vec<Vec<f32>> = Vec::new();
        for t in 0..steps {
            let conv_out = rf::conv_step(&d, &mut nudged_conv, &qkv[t], &conv1d);
            let key_dim = d.d_state * d.n_k_heads;
            let mut q = conv_out[..key_dim].to_vec();
            let mut k = conv_out[key_dim..2 * key_dim].to_vec();
            let v = conv_out[2 * key_dim..2 * key_dim + d.d_inner()].to_vec();
            for h in 0..d.n_k_heads {
                l2_f32(&mut q[h * d.d_state..(h + 1) * d.d_state], eps);
                l2_f32(&mut k[h * d.d_state..(h + 1) * d.d_state], eps);
            }
            let g: Vec<f32> = (0..d.n_v_heads)
                .map(|h| rf::delta_gate(alpha[t][h], dt_bias[h], ssm_a[h]) * nudge)
                .collect();
            let b: Vec<f32> = beta_p[t].iter().map(|&x| rf::sigmoid(x) * nudge).collect();
            let mut o = rf::delta_decode_step(&d, &mut nudged_state, &q, &k, &v, &g, &b);
            // the gated norm with an f32 mean, as the GPU computes it
            for h in 0..d.n_v_heads {
                let oh = &mut o[h * d.d_state..(h + 1) * d.d_state];
                let mean: f32 = oh.iter().map(|&x| x * x).sum::<f32>() / d.d_state as f32;
                let scale = 1.0 / (mean + eps).sqrt();
                for i in 0..d.d_state {
                    oh[i] = oh[i] * scale * ssm_norm[i] * rf::silu(z[t][h * d.d_state + i]);
                }
            }
            nudged.push(o);
        }

        // METRIC: max |a-b| over the tensor, divided by the tensor's own
        // largest magnitude. NOT per-element relative error — the delta state
        // spans seven orders of magnitude (it starts at zero and grows as a sum
        // of outer products), so per-element relative error is dominated by
        // entries around 1e-9 that carry no information. The first draft of
        // this gate used per-element and "failed" on a slot holding -3.20e-9
        // against -3.23e-9 while the tensor's max was 5.4e-2. The same metric is
        // applied to the reference-vs-nudged probe, so the ratio stays honest.
        let rel_inf = |a: &[f32], b: &[f32]| -> f32 {
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(f32::MIN_POSITIVE);
            a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
        };
        let flat = |v: &[Vec<f32>]| -> Vec<f32> { v.iter().flatten().copied().collect() };
        let (fw, fg, fn_) = (flat(&want), flat(&got), flat(&nudged));

        let out_yard = rel_inf(&fw, &fn_);
        let out_gap = rel_inf(&fw, &fg);
        let state_yard = rel_inf(&ref_state, &nudged_state);
        let state_gap = rel_inf(&ref_state, &gpu_state);
        let conv_gap = rel_inf(&ref_conv, &gpu_conv);

        assert!(out_yard > 0.0, "the one-rounding nudge changed nothing — probe is broken");
        // The conv state is pure add/multiply with contraction off: no exp, no
        // reduction, nothing to excuse a difference. It must be EXACT, and that
        // it is proves the conv half of the composition (slice offsets, the
        // rolling window, the channel packing) independently of the tolerance
        // the rest of the chain needs.
        assert!(
            ref_conv.iter().zip(&gpu_conv).all(|(a, b)| a.to_bits() == b.to_bits()),
            "the conv state must be bit-exact, gap {conv_gap:e}"
        );
        assert!(
            out_gap <= 4.0 * out_yard,
            "deltanet block output drift {out_gap:e} exceeds 4x the reference's own \
             one-rounding sensitivity {out_yard:e} — that is a composition error, not noise"
        );
        // The state must track too: a block whose OUTPUT matches while its state
        // drifts is right until it suddenly is not.
        assert!(
            state_gap <= 4.0 * state_yard,
            "delta state drift {state_gap:e} exceeds 4x the probe's {state_yard:e}"
        );
    }

    /// Partial RoPE: qwen35 rotates only rope.dimension_count (64) of each
    /// 256-wide head and leaves the rest ALONE. The tail is the half worth
    /// testing — rotating it would still produce a valid rotation of a real
    /// vector, so nothing would crash or NaN; the model would just quietly be a
    /// different model. So this asserts the tail is BIT-IDENTICAL to the input,
    /// and that the rotated prefix matches a full-width rope of that size.
    #[test]
    fn rope_rotates_only_the_leading_rot_dim() {
        let (head_dim, rot, n_heads, n_rows) = (256usize, 64usize, 2usize, 3usize);
        let mut seed = 0x0A0B_0C0Du32;
        let x: Vec<f32> = (0..n_rows * n_heads * head_dim).map(|_| lcg(&mut seed)).collect();

        let device = Device::system_default().expect("Metal device required");
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(&gpu::shader_source(128), &opts)
            .expect("kernels.metal compiles");
        let f = lib.get_function("rope", None).expect("rope");
        let pipe = device.new_compute_pipeline_state_with_function(&f).expect("pipeline");
        let queue = device.new_command_queue();
        let buf = device.new_buffer_with_data(
            x.as_ptr() as *const _,
            std::mem::size_of_val(&x[..]) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        #[repr(C)]
        struct P { head_dim: u32, n_heads: u32, pos0: u32, theta: f32, n_rows: u32, rot_dim: u32 }
        let p = P {
            head_dim: head_dim as u32,
            n_heads: n_heads as u32,
            pos0: 5,
            theta: 1.0e7,
            n_rows: n_rows as u32,
            rot_dim: rot as u32,
        };
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&buf), 0);
        enc.set_bytes(1, std::mem::size_of::<P>() as u64, &p as *const P as *const _);
        gpu::dispatch_grid(enc, n_rows * n_heads * rot / 2);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, x.len()) };

        let half = rot / 2;
        for row in 0..n_rows {
            for h in 0..n_heads {
                let base = (row * n_heads + h) * head_dim;
                for i in 0..half {
                    let freq = (1.0e7f32).powf(-2.0 * i as f32 / rot as f32);
                    let angle = (5 + row) as f32 * freq;
                    let (s, c) = (angle.sin(), angle.cos());
                    let (a, b) = (x[base + i], x[base + i + half]);
                    let (wa, wb) = (a * c - b * s, a * s + b * c);
                    // The prefix is checked to transcendental tolerance, not
                    // bit-for-bit: the kernel reaches its angle through Metal's
                    // pow and sincos and the CPU through libm's, which differ in
                    // the last bits. THIS TEST'S CLAIM IS STRUCTURAL — which
                    // dims move and which do not — and rope's numeric fidelity
                    // is already covered end-to-end by the dense-model identity
                    // gates. Saying so beats asserting a ulp count that is
                    // really measuring two libm implementations.
                    for (want, have) in [(wa, got[base + i]), (wb, got[base + i + half])] {
                        assert!(
                            rel_err(want, have) < 1e-5,
                            "row {row} head {h} dim {i}: {want} vs {have}"
                        );
                    }
                    // ...and it must actually have rotated, or "matches within
                    // tolerance" would also pass on a kernel that did nothing.
                    assert!(
                        got[base + i].to_bits() != x[base + i].to_bits()
                            || got[base + i + half].to_bits() != x[base + i + half].to_bits(),
                        "row {row} head {h} dim {i} did not rotate at all"
                    );
                }
                // THE TAIL MUST BE UNTOUCHED, bit for bit.
                for i in rot..head_dim {
                    assert_eq!(
                        x[base + i].to_bits(),
                        got[base + i].to_bits(),
                        "row {row} head {h} dim {i} was rotated but must pass through"
                    );
                }
            }
        }
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
        let stride0 = 0u32; // single token: the y index is 0, so the stride is unused
        enc.set_bytes(3, 4, &stride0 as *const u32 as *const _);
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

    /// RE-POINTED, not deleted (docs/deltanet-chain-design.md §7 risk 1). This
    /// control used to fire when the REFERENCE was fed a transposed state, back
    /// when GPU and reference shared one layout. They no longer do: the kernel
    /// reads s[j + i*S + h*S*S] and the reference s[i + j*S + h*S*S]. What must
    /// be caught now is a HALF-DONE transpose — kernel moved, caller not — which
    /// is exactly uploading the state in the old orientation, and which produces
    /// finite, plausible, wrong numbers that nothing else would notice.
    ///
    /// This is not hypothetical: while landing this lane the kernel was
    /// transposed before the oracle was, and the existing tests failed
    /// immediately. This control is what keeps that true for the next change.
    #[test]
    fn delta_oracle_sees_the_old_state_orientation() {
        let d = dims();
        let mut seed = 0x5EED_1234u32;
        let (state0, q, k, v, g, beta) = delta_inputs(d, 1, &mut seed, false);
        let (right, right_state) = gpu_delta_layout(d, &state0, &q, &k, &v, &g, &beta, true);
        let (wrong, wrong_state) = gpu_delta_layout(d, &state0, &q, &k, &v, &g, &beta, false);
        assert!(
            right[0].iter().zip(&wrong[0]).any(|(a, b)| a.to_bits() != b.to_bits()),
            "uploading the state in the OLD orientation must change the output — if it \
             does not, the kernel is not reading the layout it claims and a half-done \
             transpose would ship silently"
        );
        assert!(
            right_state.iter().zip(&wrong_state).any(|(a, b)| a.to_bits() != b.to_bits()),
            "the resulting STATE must differ too: the kernel writes through the same \
             addresses it reads, so a control checking only the output would miss a \
             transpose done half-way on the write side"
        );
    }
}
