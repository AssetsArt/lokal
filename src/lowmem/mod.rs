//! -b lowmem — a disk-backed, bounded-memory backend.
//!
//! A different philosophy from metal/hybrid: those move the whole model onto
//! the GPU and win on speed; lowmem promises a bounded, predictable footprint
//! and accepts what that costs. The mmapped weight files are the source of
//! truth; RAM holds a working set of staged pages, never a copy of the model.
//!
//! Layer weights live in a fixed WeightPool (pool.rs) and are staged from the
//! mmap per row-block page as the forward pass reaches them; only norms and
//! biases (a few hundred KB) are eagerly resident. The forward pass itself is
//! in forward.rs. Still to land: per-layer commit overlap, the KV ring with
//! windowed attention, and the --memory-budget arithmetic.

mod forward;
pub(crate) mod gguf;
pub(crate) mod qwen35_ref;
pub(crate) mod iq_grids;
pub(crate) mod manifest;
#[allow(dead_code)] // consumers arrive with lane C
mod pool;

use crate::config::ModelConfig;
use crate::engine::{Engine, Session};
use crate::gpu::metal as gpu;
use manifest::{dequant_row_ref, GgmlType, WeightManifest};
use metal::{Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, FunctionConstantValues, MTLDataType, MTLResourceOptions};
use pool::{PagedTensor, WeightPool, PAGE_BYTES};
use safetensors::Dtype;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

pub use crate::engine::LowMemOpts;

/// Default total working-set budget in MB (D9).
const BUDGET_MB_DEFAULT: usize = 4096;
/// Fixed runtime overhead estimate: binary, tokenizer, Metal runtime, shader
/// library, session bookkeeping — everything that is neither weights, KV, nor
/// activation scratch. Estimated from measured phys_footprint minus the
/// accounted parts on this machine's runs; deliberately round.
const OVERHEAD_MB: usize = 256;

/// D9's closed-form budget split. Pure arithmetic so the refuse-to-start path
/// is testable without a GPU.
struct MemoryPlan {
    kv_bytes: usize,
    act_bytes: usize,
    /// qwen35 only: recurrent conv+delta state, f32, per sequence. A FIXED
    /// term like activations — constant in context length, never per-token.
    state_bytes: usize,
    pool_bytes: usize,
}

/// Attention widths, taken from the checkpoint rather than derived.
///
/// ModelConfig::head_dim() computes hidden_size / n_heads, which qwen3 breaks:
/// it states head_dim explicitly and it does NOT satisfy that identity
/// (Qwen3-0.6B is hidden 1024, 16 heads, head_dim 128, so q_proj is
/// [2048, 1024], not [1024, 1024]). Every width inside lowmem comes from here.
#[derive(Clone, Copy)]
pub(crate) struct Dims {
    pub hidden: usize,
    pub head_dim: usize,
    /// n_heads * head_dim — the q and o projections' outer width.
    pub q_dim: usize,
    /// n_kv_heads * head_dim.
    pub kv_dim: usize,
}

impl Dims {
    fn new(cfg: &ModelConfig, head_dim: Option<usize>) -> Self {
        let head_dim = head_dim.unwrap_or_else(|| cfg.head_dim());
        Dims {
            hidden: cfg.hidden_size,
            head_dim,
            q_dim: cfg.num_attention_heads * head_dim,
            kv_dim: cfg.num_key_value_heads * head_dim,
        }
    }
}

fn memory_plan(
    cfg: &ModelConfig,
    dims: Dims,
    win: &WindowCfg,
    budget_mb: usize,
    q35: Option<&gguf::Qwen35Meta>,
) -> crate::Result<MemoryPlan> {
    let (h, hd, kvd) = (dims.hidden, dims.head_dim, dims.kv_dim);
    let chunk = gpu::PREFILL_CHUNK;
    // KV store: K and V, f16, cap slots per layer — closed-form in the window.
    // On qwen35 KV exists ONLY on the full-attention trunk layers (16 of 64 on
    // the 27B); the 48 linear layers carry the fixed recurrent state instead.
    // The MTP block is outside the meta's map entirely, so its attention can
    // never be counted here (the "17th attention layer" misconception).
    let n_kv_layers = match q35 {
        Some(m) => m.is_recurrent.iter().filter(|&&r| !r).count(),
        None => cfg.num_hidden_layers,
    };
    let state_bytes = q35.map_or(0, |m| {
        let n_linear = m.is_recurrent.iter().filter(|&&r| r).count();
        n_linear * (m.conv_state_elems + m.delta_state_elems) * 4
    });
    let kv_bytes = n_kv_layers * win.cap * kvd * 2 * 2;
    // One session's activation scratch, mirroring LowMemSession::new.
    let scores = if hd == gpu::FLASH_HEAD_DIM {
        4
    } else {
        chunk * cfg.num_attention_heads * win.cap * 4
    };
    let act_bytes = 3 * chunk * h * 4                    // x, xn, xb
        + 2 * chunk * dims.q_dim * 4                     // q, att (q_dim != h on qwen3)
        + 2 * chunk * cfg.intermediate_size * 4          // gate, up
        + 2 * chunk * kvd * 4                            // kvs staging
        + chunk * dims.q_dim * 2                         // xh (o_proj input)
        + scores
        + cfg.num_attention_heads * (win.cap / gpu::ATTN_SPLIT) * (hd + 2) * 4
        + cfg.vocab_size * 4;                            // logits
    let fixed = kv_bytes + act_bytes + state_bytes + (OVERHEAD_MB << 20);
    let budget = budget_mb << 20;
    let floor = 4 * PAGE_BYTES;
    if budget < fixed + floor {
        return Err(format!(
            "--memory-budget {budget_mb} MB cannot hold the working set: KV {} MB (window {} × {} layers){} + activations {} MB + runtime overhead {} MB leaves less than the {} MB weight-pool floor — raise the budget or shrink --context-window",
            kv_bytes >> 20,
            win.w,
            n_kv_layers,
            match state_bytes {
                0 => String::new(),
                b => format!(" + recurrent state {} MB (constant in ctx)", b >> 20),
            },
            act_bytes >> 20,
            OVERHEAD_MB,
            floor >> 20,
        )
        .into());
    }
    Ok(MemoryPlan { kv_bytes, act_bytes, state_bytes, pool_bytes: budget - fixed })
}

/// Sliding-window attention geometry. The KV store per layer is a SINK region
/// (positions 0..sink pinned forever, StreamingLLM-style) followed by a RING of
/// `ring` slots holding the last window of positions — storage is closed-form
/// and independent of context length. `ring` carries the window plus a full
/// prefill chunk of slack so every row of an in-flight chunk still sees its
/// whole window; both regions are 128-aligned so attention tiles and decode
/// splits never straddle them.
#[derive(Clone, Copy)]
pub(crate) struct WindowCfg {
    pub w: usize,
    pub sink: usize,
    pub sink_pad: usize,
    pub ring: usize,
    pub cap: usize,
}

impl WindowCfg {
    pub(crate) fn new(w: usize, sink: usize) -> crate::Result<Self> {
        if w == 0 || sink > w {
            return Err(format!("invalid window config: window {w}, sink {sink}").into());
        }
        let sink_pad = sink.next_multiple_of(128);
        let ring = (w + gpu::PREFILL_CHUNK).next_multiple_of(128);
        Ok(Self { w, sink, sink_pad, ring, cap: sink_pad + ring })
    }

    /// The store slot holding position `p`.
    pub fn slot_of(&self, p: usize) -> usize {
        if p < self.sink { p } else { self.sink_pad + (p - self.sink) % self.ring }
    }
}

/// The pipelines the lowmem forward dispatches — a subset of kernels.metal,
/// compiled from the same source string as the metal backend's.
pub(crate) struct Pipes {
    pub rmsnorm: ComputePipelineState,
    /// qwen3 qk-norm on the f16 KV cache, in place.
    pub rmsnorm_h_inplace: ComputePipelineState,
    pub matvec: ComputePipelineState,
    pub matvec_h: ComputePipelineState,
    pub matvec_acc: ComputePipelineState,
    pub matvec_swiglu: ComputePipelineState,
    pub matmul_pg: ComputePipelineState,
    pub f32_to_f16: ComputePipelineState,
    pub bf16_to_f16: ComputePipelineState,
    pub rope: ComputePipelineState,
    pub rope_h: ComputePipelineState,
    pub rope_qk_decode: ComputePipelineState,
    /// The three attention pipelines are the shared kernels SPECIALIZED with the
    /// LM_* window function constants — lowmem is windowed by construction.
    pub attention_flash: ComputePipelineState,
    pub attention_fallback: ComputePipelineState,
    pub attn_dec_partial: ComputePipelineState,
    pub attn_dec_reduce: ComputePipelineState,
    pub silu_mul: ComputePipelineState,
    pub add_inplace: ComputePipelineState,
}

/// The matvec family specialized with LM_W_BF16: weight buffers are RAW bf16
/// checkpoint bytes read through the mmap views (values still round through
/// f16, so results match the staged pipelines bit for bit).
pub(crate) struct DirectPipes {
    pub matvec: ComputePipelineState,
    pub matvec_h: ComputePipelineState,
    pub matvec_acc: ComputePipelineState,
    pub matvec_swiglu: ComputePipelineState,
}

/// The matvec family plus the prefill GEMM, specialized for ONE quantized
/// weight encoding. Built from the PRECISE library: Metal's fast math contracts
/// multiply-add pairs, which silently breaks bit-for-bit agreement with the
/// strict-IEEE Rust reference the oracle gate compares against. The existing
/// f16/bf16 pipelines keep the fast library, so -b metal's numerics never move.
pub(crate) struct QuantPipes {
    pub matvec: ComputePipelineState,
    pub matvec_h: ComputePipelineState,
    pub matvec_acc: ComputePipelineState,
    pub matvec_swiglu: ComputePipelineState,
    pub matmul_pg: ComputePipelineState,
}

/// Which member of the matvec family a call site wants; the weight's own type
/// picks the specialization.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fam {
    Mv,
    MvH,
    MvA,
}

/// One transformer block: eagerly-resident norms, paged projection matrices.
pub(crate) struct LayerWeights {
    pub input_ln: Buffer,
    pub post_ln: Buffer,
    /// qwen3's per-head q/k RMSNorm weights (head_dim each), applied to every
    /// head of q and k before RoPE. None on architectures without qk-norm.
    pub q_norm: Option<Buffer>,
    pub k_norm: Option<Buffer>,
    pub q: PagedTensor,
    pub k: PagedTensor,
    pub v: PagedTensor,
    pub o: PagedTensor,
    pub gate: PagedTensor,
    pub up: PagedTensor,
    pub down: PagedTensor,
}

/// What a weight's elements are, across both checkpoint formats. Safetensors
/// gives f32/f16/bf16; GGUF adds the quantized block types, whose rows are not
/// element-addressable — a row is a whole number of blocks, and the pool holds
/// those blocks RAW (dequantizing at stage time would forfeit the 4x residency
/// that is this backend's reason to exist).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SrcType {
    F32,
    F16,
    BF16,
    Quant(GgmlType),
}

impl SrcType {
    /// Bytes one row of `cols` elements occupies in the checkpoint.
    pub fn row_bytes(self, cols: usize) -> usize {
        match self {
            SrcType::F32 => cols * 4,
            SrcType::F16 | SrcType::BF16 => cols * 2,
            SrcType::Quant(t) => t.row_bytes(cols),
        }
    }

    pub fn is_quant(self) -> bool {
        matches!(self, SrcType::Quant(_))
    }

    /// The LM_W_QTYPE function-constant selector this type builds pipelines
    /// under. Kept beside row_bytes so the two never drift apart.
    pub fn qtype(self) -> u32 {
        match self {
            SrcType::F32 | SrcType::F16 => 0,
            SrcType::BF16 => 1,
            SrcType::Quant(GgmlType::Q8_0) => 2,
            SrcType::Quant(GgmlType::Q4_0) => 3,
            SrcType::Quant(GgmlType::Q4_K) => 4,
            SrcType::Quant(GgmlType::Q6_K) => 5,
            SrcType::Quant(GgmlType::Q5_K) => 6,
            SrcType::Quant(GgmlType::Q5_0) => 7,
            SrcType::Quant(GgmlType::Q2_K) => 8,
            SrcType::Quant(GgmlType::Q3_K) => 9,
            SrcType::Quant(GgmlType::IQ4_NL) => 10,
            SrcType::Quant(GgmlType::IQ4_XS) => 11,
            SrcType::Quant(GgmlType::IQ3_XXS) => 12,
            SrcType::Quant(GgmlType::IQ3_S) => 13,
            SrcType::Quant(GgmlType::IQ2_XXS) => 14,
            SrcType::Quant(GgmlType::IQ2_XS) => 15,
            SrcType::Quant(GgmlType::IQ2_S) => 16,
            SrcType::Quant(GgmlType::IQ1_S) => 17,
            SrcType::Quant(GgmlType::IQ1_M) => 18,
            SrcType::Quant(_) => u32::MAX, // refused at PagedTensor::new
        }
    }
}

/// The checkpoint behind the pool. Safetensors and GGUF differ in how a row is
/// found and what it holds, and in nothing else the pool cares about, so the
/// difference is confined here rather than smeared through pool.rs.
///
/// This lives in lowmem's own module, not in the seam: manifest.rs belongs to
/// the loader lane, and the abstraction the seam ruling named never landed
/// (challenge on gguf-kernels). Both variants are built from public seam items.
pub(crate) enum LowMemSource {
    Safetensors(WeightManifest),
    Gguf(Box<GgufSource>),
}

/// A GGUF file plus the HF-name index lowmem addresses tensors by. The rest of
/// the backend speaks HF names (`model.layers.0.self_attn.q_proj.weight`); the
/// file speaks `blk.0.attn_q.weight`.
pub(crate) struct GgufSource {
    file: gguf::GgufFile,
    /// One no-copy Metal view over the file's mmap, plus the page-aligned host
    /// address it starts at, so a tensor's span becomes (view, byte offset).
    view: Option<Buffer>,
    base: usize,
    /// HF name -> the GGUF name that carries it.
    by_hf: HashMap<String, String>,
    arch: gguf::GgufArch,
    n_params: usize,
}

/// The one qwen35 tensor `gguf::hf_name` cannot express: `blk.N.ssm_a` carries
/// NO `.weight`/`.bias` suffix, so the generic mapper's `rsplit_once('.')`
/// fails and the tensor never enters the name index — it would be invisible to
/// the loader rather than merely awkwardly named.
///
/// Every other deltanet tensor already survives, via the mapper's
/// `gguf.<mid>` fallback (`blk.0.ssm_conv1d.weight` becomes
/// `model.layers.0.gguf.ssm_conv1d.weight`), so this handles the one exception
/// rather than duplicating the mapper. It lives here, not in gguf.rs, because
/// that file belongs to the loader lane; this is a lowmem-side index fix and
/// changes no shared behaviour — `hf_name` is consulted first and wins.
fn qwen35_hf_name(gguf: &str) -> Option<String> {
    let rest = gguf.strip_prefix("blk.")?;
    let (n, mid) = rest.split_once('.')?;
    let i: usize = n.parse().ok()?;
    match mid {
        "ssm_a" => Some(format!("model.layers.{i}.gguf.ssm_a")),
        _ => None,
    }
}

impl LowMemSource {
    pub fn open(path: &Path) -> crate::Result<Self> {
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
            let file = gguf::GgufFile::open(path)?;
            let (_, arch) = gguf::model_config(&file)?;
            let mut by_hf = HashMap::new();
            let mut n_params = 0;
            for t in file.tensors() {
                n_params += t.dims.iter().product::<usize>();
                if let Some(hf) = gguf::hf_name(&t.name) {
                    by_hf.insert(hf, t.name.clone());
                } else if let Some(hf) = qwen35_hf_name(&t.name) {
                    by_hf.insert(hf, t.name.clone());
                }
            }
            Ok(LowMemSource::Gguf(Box::new(GgufSource {
                file,
                by_hf,
                arch,
                n_params,
                view: None,
                base: 0,
            })))
        } else {
            Ok(LowMemSource::Safetensors(WeightManifest::open(path)?))
        }
    }

    pub fn make_gpu_views(&mut self, device: &Device) {
        const PAGE: usize = 16384;
        match self {
            LowMemSource::Safetensors(mf) => mf.make_gpu_views(device),
            LowMemSource::Gguf(g) => {
                // The mmap base is not exposed, but every tensor points into
                // it, so the lowest tensor address rounded DOWN to a page is a
                // mapped, page-aligned address — which is all Metal needs.
                let (mut lo, mut hi) = (usize::MAX, 0usize);
                for t in g.file.tensors() {
                    let p = t.data.as_ptr() as usize;
                    lo = lo.min(p);
                    hi = hi.max(p + t.data.len());
                }
                if lo == usize::MAX {
                    return;
                }
                let base = lo & !(PAGE - 1);
                g.base = base;
                g.view = Some(device.new_buffer_with_bytes_no_copy(
                    base as *const _,
                    (hi - base).next_multiple_of(PAGE) as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                ));
            }
        }
    }

    pub fn n_tensors(&self) -> usize {
        match self {
            LowMemSource::Safetensors(mf) => mf.n_tensors(),
            LowMemSource::Gguf(g) => g.file.n_tensors(),
        }
    }

    /// Bytes the pool holds for the whole model — the checkpoint's own encoding
    /// for GGUF (quant blocks stay quantized), f16 for safetensors.
    pub fn staged_bytes(&self) -> usize {
        match self {
            LowMemSource::Safetensors(mf) => mf.n_params * 2,
            LowMemSource::Gguf(g) => g.file.tensors().map(|t| t.data.len()).sum(),
        }
    }

    pub fn n_params(&self) -> usize {
        match self {
            LowMemSource::Safetensors(mf) => mf.n_params,
            LowMemSource::Gguf(g) => g.n_params,
        }
    }

    pub fn is_gguf(&self) -> bool {
        matches!(self, LowMemSource::Gguf(_))
    }

    /// The distinct quantized types this checkpoint actually uses. Pipelines
    /// are built per type PRESENT, never per type supported: a dtype selector
    /// as a function constant multiplies pipeline builds, and compiling the
    /// whole matvec family for six types a file never mentions costs seconds
    /// of startup for nothing.
    pub fn quant_types(&self) -> Vec<GgmlType> {
        let LowMemSource::Gguf(g) = self else { return Vec::new() };
        let mut v: Vec<GgmlType> = Vec::new();
        for t in g.file.tensors() {
            if !matches!(t.ty, GgmlType::F32 | GgmlType::F16) && !v.contains(&t.ty) {
                v.push(t.ty);
            }
        }
        v
    }

    /// Explicit head dim when the checkpoint states one (GGUF always does).
    /// None means "derive it", which is right for every safetensors model here.
    /// qwen35 only: the hybrid-trunk metadata (recurrency map + state sizes).
    /// None for every other architecture and for safetensors.
    pub fn qwen35(&self) -> Option<gguf::Qwen35Meta> {
        match self {
            LowMemSource::Gguf(g) if g.arch.arch == "qwen35" => {
                gguf::qwen35_meta(&g.file).ok()
            }
            _ => None,
        }
    }

    pub fn head_dim(&self) -> Option<usize> {
        match self {
            LowMemSource::Safetensors(_) => None,
            LowMemSource::Gguf(g) => Some(g.arch.head_dim),
        }
    }

    /// qwen3-style per-head q/k norm, straight from the file's metadata.
    pub fn qk_norm(&self) -> bool {
        match self {
            LowMemSource::Safetensors(mf) => mf.has("model.layers.0.self_attn.q_norm.weight"),
            LowMemSource::Gguf(g) => g.arch.qk_norm,
        }
    }

    pub fn has(&self, name: &str) -> bool {
        match self {
            LowMemSource::Safetensors(mf) => mf.has(name),
            LowMemSource::Gguf(g) => g.by_hf.contains_key(name),
        }
    }

    pub fn shape(&self, name: &str) -> crate::Result<Vec<usize>> {
        match self {
            LowMemSource::Safetensors(mf) => Ok(mf.meta(name)?.shape.clone()),
            LowMemSource::Gguf(g) => Ok(g.tensor(name)?.dims.clone()),
        }
    }

    pub fn src_type(&self, name: &str) -> crate::Result<SrcType> {
        match self {
            LowMemSource::Safetensors(mf) => Ok(match mf.meta(name)?.dtype {
                Dtype::F32 => SrcType::F32,
                Dtype::F16 => SrcType::F16,
                Dtype::BF16 => SrcType::BF16,
                other => return Err(format!("unsupported dtype {other:?} in {name}").into()),
            }),
            LowMemSource::Gguf(g) => Ok(match g.tensor(name)?.ty {
                GgmlType::F32 => SrcType::F32,
                GgmlType::F16 => SrcType::F16,
                t => SrcType::Quant(t),
            }),
        }
    }

    /// Rows `r0..r1` as the checkpoint stores them. Rows are contiguous in both
    /// formats, so this is a slice, never a copy.
    pub fn read_rows(&self, name: &str, r0: usize, r1: usize) -> crate::Result<&[u8]> {
        match self {
            LowMemSource::Safetensors(mf) => Ok(mf.read_rows(name, r0, r1)?.0),
            LowMemSource::Gguf(g) => {
                let t = g.tensor(name)?;
                let cols = *t.dims.last().ok_or("gguf tensor has no dims")?;
                let rb = t.ty.row_bytes(cols);
                let rows = t.dims[0];
                if r1 > rows || r0 >= r1 {
                    return Err(format!("read_rows({name}, {r0}..{r1}): tensor has {rows} rows").into());
                }
                Ok(&t.data[r0 * rb..r1 * rb])
            }
        }
    }

    pub fn gpu_span(
        &self,
        name: &str,
        r0: usize,
        r1: usize,
    ) -> crate::Result<Option<(&Buffer, usize)>> {
        match self {
            LowMemSource::Safetensors(mf) => mf.gpu_span(name, r0, r1),
            LowMemSource::Gguf(g) => {
                let Some(view) = &g.view else { return Ok(None) };
                let t = g.tensor(name)?;
                let cols = *t.dims.last().ok_or("gguf tensor has no dims")?;
                if r1 > t.dims[0] || r0 >= r1 {
                    return Err(format!("gpu_span({name}, {r0}..{r1})").into());
                }
                // A permuted tensor's rows are not contiguous in the file, so
                // there is no single span to hand the GPU — those stage.
                if self.unpermute_head_dim(name).is_some() {
                    return Ok(None);
                }
                Ok(Some((view, t.data.as_ptr() as usize - g.base + r0 * t.ty.row_bytes(cols))))
            }
        }
    }

    /// Whole tensor as f32 — for the eagerly-resident small tensors (norms,
    /// biases) and the embedding gather. llama-arch q/k come back UNPERMUTED:
    /// llama.cpp stores them rotated for GGML's adjacent-pair RoPE, and lokal
    /// rotates halves (kernels.metal rope pairs head[i] with head[i+half_dim]).
    pub fn read_f32(&self, name: &str) -> crate::Result<Vec<f32>> {
        match self {
            LowMemSource::Safetensors(mf) => mf.read_f32(name),
            LowMemSource::Gguf(g) => {
                let t = g.tensor(name)?;
                let cols = *t.dims.last().unwrap_or(&1);
                let rows = t.dims.iter().product::<usize>() / cols.max(1);
                let mut out = vec![0f32; rows * cols];
                for r in 0..rows {
                    let rb = t.ty.row_bytes(cols);
                    dequant_row_ref(t.ty, &t.data[r * rb..(r + 1) * rb], &mut out[r * cols..(r + 1) * cols]);
                }
                if let Some(hd) = self.unpermute_head_dim(name) {
                    gguf::unpermute_llama_qk(&mut out, rows, cols, hd);
                }
                Ok(out)
            }
        }
    }

    /// Some(head_dim) when this tensor is a llama-arch GGUF q/k that must have
    /// llama.cpp's RoPE permute undone as it materializes. None everywhere else
    /// — safetensors is never permuted, and neither is any non-q/k tensor.
    pub fn unpermute_head_dim(&self, name: &str) -> Option<usize> {
        let LowMemSource::Gguf(g) = self else { return None };
        if g.arch.arch != "llama" {
            return None;
        }
        (name.ends_with("self_attn.q_proj.weight") || name.ends_with("self_attn.k_proj.weight"))
            .then_some(g.arch.head_dim)
    }
}

impl GgufSource {
    fn tensor(&self, hf: &str) -> crate::Result<manifest::GgufTensor<'_>> {
        let gg = self.by_hf.get(hf).ok_or_else(|| format!("{hf}: not in this GGUF"))?;
        self.file.tensor(gg)
    }
}

pub struct LowMemEngine {
    cfg: ModelConfig,
    dims: Dims,
    source: LowMemSource,
    device: Device,
    queue: CommandQueue,
    pipes: Pipes,
    direct: DirectPipes,
    /// Quant pipelines by LM_W_QTYPE selector, built only for the types this
    /// checkpoint actually contains.
    quant: HashMap<u32, QuantPipes>,
    layers: Vec<LayerWeights>,
    final_norm: Buffer,
    lm_head: PagedTensor,
    /// Shared all-zero f16 bias for biasless projections (same convention as
    /// the metal backend), sized for the largest page's row count.
    zero_bias: Buffer,
    pool: Mutex<WeightPool>,
    /// Decode-attention dispatch geometry, precomputed from the config.
    gqa: (u64, [u64; 4]),
    /// Sliding-window geometry — the backend's core semantic.
    win: WindowCfg,
    /// qwen35 only: what sessions allocate their recurrent state from.
    /// None on every other architecture (existing paths untouched).
    q35_layout: Option<crate::gpu::metal::Qwen35Layout>,
    /// LOKAL_LOWMEM_SYNC=1: wait out every command buffer before the next —
    /// the bisect mode for anything that smells like an eviction race.
    sync: bool,
    /// The GPU bf16 converter's overflow flag (one u32, set on any clip) and
    /// the once-only latch for the warning it feeds.
    clip_flag: Buffer,
    clip_warned: std::sync::atomic::AtomicBool,
}

// Same justification as MetalEngine: Apple documents these Metal objects as
// thread-safe; the mutable pool sits behind the Mutex above.
unsafe impl Send for LowMemEngine {}
unsafe impl Sync for LowMemEngine {}

impl LowMemEngine {
    /// Built from the model DIRECTORY, not a loaded Model — nothing here ever
    /// materializes the full model in RAM.
    pub fn new(dir: &Path, cfg: ModelConfig, opts: &LowMemOpts) -> crate::Result<Self> {
        let t0 = Instant::now();
        let mut source = LowMemSource::open(dir)?;
        eprintln!(
            "lowmem: {} {} tensors | {:.1}M params (headers parsed in {:.2}s)",
            if source.is_gguf() { "gguf" } else { "manifest" },
            source.n_tensors(),
            source.n_params() as f64 / 1e6,
            t0.elapsed().as_secs_f64(),
        );

        let dims = Dims::new(&cfg, source.head_dim());
        let device = Device::system_default().ok_or("no Metal-capable GPU found")?;
        source.make_gpu_views(&device);
        let queue = device.new_command_queue();
        let shader = gpu::shader_source(dims.kv_dim);
        let lib = device
            .new_library_with_source(&shader, &CompileOptions::new())
            .map_err(|e| format!("failed to compile kernels.metal: {e}"))?;
        // Every build goes through the specialization API: kernels that
        // reference function constants (the matvec family via dot_wx, the
        // attention set via LM_*) refuse the unspecialized path, and the empty
        // set compiles identical code for the rest.
        let pipe = |name: &str| -> crate::Result<ComputePipelineState> {
            let f = lib
                .get_function(name, Some(FunctionConstantValues::new()))
                .map_err(|e| format!("kernel {name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("kernel {name}: {e}").into())
        };
        // Weight-encoding selector (LM_W_QTYPE at index 25): 1 = raw bf16 over
        // the mmap views; 2..7 = the GGUF quant types, which build from the
        // PRECISE library below.
        let build = |lib: &metal::Library,
                     name: &str,
                     qtype: u32|
         -> crate::Result<ComputePipelineState> {
            let consts = FunctionConstantValues::new();
            consts.set_constant_value_at_index(
                &qtype as *const u32 as *const _,
                MTLDataType::UInt,
                25,
            );
            let f = lib
                .get_function(name, Some(consts))
                .map_err(|e| format!("kernel {name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("kernel {name}: {e}").into())
        };
        let qtype_pipe =
            |name: &str, qtype: u32| -> crate::Result<ComputePipelineState> { build(&lib, name, qtype) };

        // Dequant math must match dequant_row_ref BIT-FOR-BIT, and Metal's
        // default fast math contracts a*b+c into an fma whose intermediate
        // keeps extra precision — the two then disagree in the last ulp on
        // exactly the values a quantizer produces. Only the quant pipelines
        // pay for this; everything else stays on the fast library.
        let quant_types = source.quant_types();
        let mut quant: HashMap<u32, QuantPipes> = HashMap::new();
        if !quant_types.is_empty() {
            let precise = CompileOptions::new();
            precise.set_fast_math_enabled(false);
            let plib = device
                .new_library_with_source(&shader, &precise)
                .map_err(|e| format!("failed to compile kernels.metal (precise): {e}"))?;
            let t_pipes = Instant::now();
            for ty in &quant_types {
                let sel = SrcType::Quant(*ty).qtype();
                // A type with no selector has no GPU path; PagedTensor::new
                // refuses it by tensor name, which is the useful message.
                if sel == u32::MAX || quant.contains_key(&sel) {
                    continue;
                }
                quant.insert(
                    sel,
                    QuantPipes {
                        matvec: build(&plib, "matvec", sel)?,
                        matvec_h: build(&plib, "matvec_h", sel)?,
                        matvec_acc: build(&plib, "matvec_acc", sel)?,
                        matvec_swiglu: build(&plib, "matvec_swiglu", sel)?,
                        matmul_pg: build(&plib, "matmul_pg", sel)?,
                    },
                );
            }
            eprintln!(
                "lowmem: {} quant pipeline set(s) for {:?} (precise, fast-math off) in {:.2}s",
                quant.len(),
                quant_types,
                t_pipes.elapsed().as_secs_f64(),
            );
        }
        let env_usize = |k: &str, d: usize| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        // Flags first, env second (debug), defaults last.
        let win = WindowCfg::new(
            opts.context_window.unwrap_or_else(|| env_usize("LOKAL_LOWMEM_WINDOW", 2048)),
            opts.attention_sink.unwrap_or_else(|| env_usize("LOKAL_LOWMEM_SINK", 4)),
        )?;

        let gqa_chunk =
            (cfg.num_attention_heads / cfg.num_key_value_heads).min(gpu::MAX_GQA_CHUNK) as u32;
        // The windowed attention pipelines: the SHARED kernels specialized with
        // the LM_* function constants (indices 20-23; GQA_CHUNK rides at 0 for
        // the decode partial like the metal backend's own pipeline).
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
                (win.sink as u32, 20u64),
                (win.sink_pad as u32, 21),
                (win.ring as u32, 22),
                (win.w as u32, 23),
            ] {
                consts.set_constant_value_at_index(&v as *const u32 as *const _, MTLDataType::UInt, idx);
            }
            let f = lib
                .get_function(name, Some(consts))
                .map_err(|e| format!("kernel {name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("kernel {name}: {e}").into())
        };
        let pipes = Pipes {
            rmsnorm: pipe("rmsnorm")?,
            rmsnorm_h_inplace: pipe("rmsnorm_h_inplace")?,
            matvec: pipe("matvec")?,
            matvec_h: pipe("matvec_h")?,
            matvec_acc: pipe("matvec_acc")?,
            matvec_swiglu: pipe("matvec_swiglu")?,
            matmul_pg: pipe("matmul_pg")?,
            f32_to_f16: pipe("f32_to_f16")?,
            bf16_to_f16: pipe("bf16_to_f16_copy")?,
            rope: pipe("rope")?,
            rope_h: pipe("rope_h")?,
            rope_qk_decode: pipe("rope_qk_decode")?,
            attention_flash: win_pipe("attention_prefill_flash", false)?,
            attention_fallback: win_pipe("attention", false)?,
            attn_dec_partial: win_pipe("attention_decode_partial", true)?,
            attn_dec_reduce: pipe("attention_decode_reduce")?,
            silu_mul: pipe("silu_mul")?,
            add_inplace: pipe("add_inplace")?,
        };
        let direct = DirectPipes {
            matvec: qtype_pipe("matvec", 1)?,
            matvec_h: qtype_pipe("matvec_h", 1)?,
            matvec_acc: qtype_pipe("matvec_acc", 1)?,
            matvec_swiglu: qtype_pipe("matvec_swiglu", 1)?,
        };

        // Small weights (norms, biases): eagerly resident, a few hundred KB —
        // they land in D9's fixed-overhead term, not the pool.
        let small = |name: String| -> crate::Result<Buffer> {
            Ok(gpu::f16_buffer(&device, &source.read_f32(&name)?))
        };
        let mut next_id = 0u32;
        let mut mk = |prefix: String, in_dim: usize, out_dim: usize| -> crate::Result<PagedTensor> {
            let bias_name = format!("{prefix}.bias");
            let bias = if source.has(&bias_name) {
                Some(gpu::f16_buffer(&device, &source.read_f32(&bias_name)?))
            } else {
                None
            };
            let t = PagedTensor::new(
                &source,
                next_id,
                format!("{prefix}.weight"),
                in_dim,
                out_dim,
                bias,
            )?;
            next_id += 1;
            Ok(t)
        };

        let (h, kv) = (cfg.hidden_size, dims.kv_dim);
        let qk_norm = source.qk_norm();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(LayerWeights {
                input_ln: small(format!("{p}.input_layernorm.weight"))?,
                post_ln: small(format!("{p}.post_attention_layernorm.weight"))?,
                q_norm: match qk_norm {
                    true => Some(small(format!("{p}.self_attn.q_norm.weight"))?),
                    false => None,
                },
                k_norm: match qk_norm {
                    true => Some(small(format!("{p}.self_attn.k_norm.weight"))?),
                    false => None,
                },
                q: mk(format!("{p}.self_attn.q_proj"), h, dims.q_dim)?,
                k: mk(format!("{p}.self_attn.k_proj"), h, kv)?,
                v: mk(format!("{p}.self_attn.v_proj"), h, kv)?,
                o: mk(format!("{p}.self_attn.o_proj"), dims.q_dim, h)?,
                gate: mk(format!("{p}.mlp.gate_proj"), h, cfg.intermediate_size)?,
                up: mk(format!("{p}.mlp.up_proj"), h, cfg.intermediate_size)?,
                down: mk(format!("{p}.mlp.down_proj"), cfg.intermediate_size, h)?,
            });
        }
        let final_norm = small("model.norm.weight".into())?;
        // Tied weights: no lm_head tensor means the pager reads the embedding
        // table's rows — same bytes, no copy anywhere.
        let lm_name =
            if source.has("lm_head.weight") { "lm_head" } else { "model.embed_tokens" };
        let lm_head = mk(lm_name.into(), h, cfg.vocab_size)?;

        let max_rows = layers
            .iter()
            .flat_map(|l| [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down])
            .chain([&lm_head])
            .map(|t| t.rows_per_page)
            .max()
            .unwrap_or(1);
        let zero_bias = gpu::f16_buffer(&device, &vec![0.0; max_rows]);

        // The budget split (D9). LOKAL_LOWMEM_POOL_MB stays as a debug override
        // that pins the pool directly, bypassing the arithmetic.
        let budget_mb = opts.memory_budget_mb.unwrap_or(BUDGET_MB_DEFAULT);
        let q35_meta = source.qwen35();
        let plan = memory_plan(&cfg, dims, &win, budget_mb, q35_meta.as_ref())?;
        let q35_layout =
            q35_meta.as_ref().map(crate::gpu::metal::Qwen35Layout::from_meta);
        let pool_bytes = match std::env::var("LOKAL_LOWMEM_POOL_MB") {
            Ok(v) => v.parse::<usize>().map(|mb| mb << 20).unwrap_or(plan.pool_bytes),
            Err(_) => plan.pool_bytes,
        };
        let pool = WeightPool::new(&device, pool_bytes);
        // What the pool would actually hold. NOT params*2: that assumes every
        // weight stages as f16, which is the whole thing a quantized checkpoint
        // is not — it reported a 19.8 GB Q4 file as needing 61 GB, and turned
        // the disk-bound estimate into fiction on exactly the models this
        // backend exists for.
        let staged_bytes = source.staged_bytes();
        if staged_bytes > pool_bytes {
            // The ANE-compile lesson: long silent work must announce itself. And
            // the decode line pre-answers "why is the GPU idle": past the budget
            // every token re-streams the non-resident remainder from disk, so the
            // GPU's few ms of matvec work per token round to zero on a power
            // gauge — the machine is busy reading, not computing.
            let stream_bytes = staged_bytes - pool_bytes;
            eprintln!(
                "lowmem: model needs {:.1} GB staged but the weight pool holds {:.1} GB — running disk-bound: \
                 prefill streams the whole model once per {} tokens (~{:.1}s per sweep at SSD speed), \
                 decode streams the non-resident {:.1} GB EVERY token (~{:.1}s/token; the GPU will look idle — \
                 it is waiting on the disk). A quantized checkpoint that fits the pool removes this entirely.",
                staged_bytes as f64 / (1 << 30) as f64,
                pool_bytes as f64 / (1 << 30) as f64,
                gpu::PREFILL_CHUNK,
                staged_bytes as f64 / 2.5e9,
                stream_bytes as f64 / (1 << 30) as f64,
                stream_bytes as f64 / 2.5e9,
            );
        }
        // The one-line budget arithmetic, printed at load (D9).
        let n_kv_layers = q35_meta
            .as_ref()
            .map_or(cfg.num_hidden_layers, |m| {
                m.is_recurrent.iter().filter(|&&r| !r).count()
            });
        eprintln!(
            "lowmem: {} — budget {} MB = weights {} MB (paged, ≤{} MB pages) + KV {} MB (window {} +{} sink × {} layers){} + activations {} MB + overhead {} MB",
            device.name(),
            budget_mb,
            pool_bytes >> 20,
            PAGE_BYTES >> 20,
            plan.kv_bytes >> 20,
            win.w,
            win.sink,
            n_kv_layers,
            match plan.state_bytes {
                0 => String::new(),
                // Per SEQUENCE, not per token — the whole point of the term.
                b => format!(" + recurrent state {} MB (constant in ctx)", b >> 20),
            },
            plan.act_bytes >> 20,
            OVERHEAD_MB,
        );

        let clip_flag =
            device.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        unsafe { *(clip_flag.contents() as *mut u32) = 0 };

        Ok(Self {
            gqa: gpu::gqa_decode_dims(&cfg, dims.head_dim),
            sync: std::env::var("LOKAL_LOWMEM_SYNC").is_ok_and(|v| v == "1"),
            win,
            q35_layout,
            clip_flag,
            clip_warned: std::sync::atomic::AtomicBool::new(false),
            cfg,
            source,
            dims,
            quant,
            device,
            queue,
            pipes,
            direct,
            layers,
            final_norm,
            lm_head,
            zero_bias,
            pool: Mutex::new(pool),
        })
    }
}

impl LowMemEngine {
    /// The pipeline that reads THIS tensor's encoding. Quant weights land on
    /// the precise library's specialization; everything else keeps the f16
    /// pipelines the safetensors path has always used, so those numerics are
    /// untouched by construction.
    pub(crate) fn staged_pipe(&self, ty: SrcType, fam: Fam) -> &ComputePipelineState {
        match self.quant.get(&ty.qtype()).filter(|_| ty.is_quant()) {
            Some(q) => match fam {
                Fam::Mv => &q.matvec,
                Fam::MvH => &q.matvec_h,
                Fam::MvA => &q.matvec_acc,
            },
            None => match fam {
                Fam::Mv => &self.pipes.matvec,
                Fam::MvH => &self.pipes.matvec_h,
                Fam::MvA => &self.pipes.matvec_acc,
            },
        }
    }

    /// The direct-read (mmap) sibling. Only bf16 has one today: a quant span
    /// bound direct would need its own view plumbing, and until that lands a
    /// quant page always goes through the pool.
    pub(crate) fn direct_pipe(&self, fam: Fam) -> &ComputePipelineState {
        match fam {
            Fam::Mv => &self.direct.matvec,
            Fam::MvH => &self.direct.matvec_h,
            Fam::MvA => &self.direct.matvec_acc,
        }
    }

    pub(crate) fn matmul_pipe(&self, ty: SrcType) -> &ComputePipelineState {
        match self.quant.get(&ty.qtype()).filter(|_| ty.is_quant()) {
            Some(q) => &q.matmul_pg,
            None => &self.pipes.matmul_pg,
        }
    }

    pub(crate) fn swiglu_pipe(&self, ty: SrcType) -> &ComputePipelineState {
        match self.quant.get(&ty.qtype()).filter(|_| ty.is_quant()) {
            Some(q) => &q.matvec_swiglu,
            None => &self.pipes.matvec_swiglu,
        }
    }
}

impl Engine for LowMemEngine {
    fn name(&self) -> &'static str {
        "lowmem"
    }
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }
    fn session(&self, max_seq: usize) -> crate::Result<Box<dyn Session + '_>> {
        Ok(Box::new(forward::LowMemSession::new(self, max_seq)))
    }
    // batcher() stays None: serve mode falls back to per-request sessions. The
    // pool lock in run() makes concurrent sessions correct, merely slow.
}

#[cfg(test)]
mod tests {

    // ---- qwen35 real-file tests: run by the gates with `--ignored` ----

    fn qwen35_gguf() -> Option<std::path::PathBuf> {
        let snaps = std::path::PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/huggingface/hub/models--unsloth--Qwen3.5-2B-GGUF/snapshots");
        std::fs::read_dir(snaps)
            .ok()?
            .flatten()
            .flat_map(|e| std::fs::read_dir(e.path()).into_iter().flatten().flatten())
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "gguf"))
    }

    /// Every tensor the qwen35 forward pass will ask for must be ADDRESSABLE
    /// before any of it is written. The hybrid has two different block shapes
    /// and they share only the norms and the FFN, so a name that silently fails
    /// to resolve would surface much later as a confusing load error in the
    /// middle of a graph — this asserts the whole set up front, per layer, on
    /// the real checkpoint.
    ///
    /// It also pins the one name the generic mapper cannot express: `ssm_a`
    /// carries no `.weight` suffix, so without `qwen35_hf_name` it is absent
    /// from the index entirely rather than merely oddly named.
    #[test]
    #[ignore]
    fn qwen35_2b_every_tensor_resolves() {
        let Some(path) = qwen35_gguf() else {
            panic!("Qwen3.5-2B GGUF not in the HF cache — this gate needs the real file");
        };
        let src = LowMemSource::open(&path).expect("opens");
        let meta = src.qwen35().expect("qwen35 meta parses");
        assert_eq!(meta.is_recurrent.len(), meta.trunk_layers);

        let mut linear = 0;
        let mut full = 0;
        for i in 0..meta.trunk_layers {
            let p = format!("model.layers.{i}");
            // Shared by both block kinds.
            for n in [
                format!("{p}.input_layernorm.weight"),
                format!("{p}.gguf.post_attention_norm.weight"),
                format!("{p}.mlp.gate_proj.weight"),
                format!("{p}.mlp.up_proj.weight"),
                format!("{p}.mlp.down_proj.weight"),
            ] {
                assert!(src.has(&n), "layer {i}: missing {n}");
            }
            if meta.is_recurrent[i] {
                linear += 1;
                for n in [
                    format!("{p}.gguf.attn_qkv.weight"),
                    format!("{p}.gguf.attn_gate.weight"),
                    format!("{p}.gguf.ssm_conv1d.weight"),
                    format!("{p}.gguf.ssm_a"),
                    format!("{p}.gguf.ssm_alpha.weight"),
                    format!("{p}.gguf.ssm_beta.weight"),
                    format!("{p}.gguf.ssm_dt.bias"),
                    format!("{p}.gguf.ssm_norm.weight"),
                    format!("{p}.gguf.ssm_out.weight"),
                ] {
                    assert!(src.has(&n), "linear layer {i}: missing {n}");
                }
            } else {
                full += 1;
                for n in [
                    format!("{p}.self_attn.q_proj.weight"),
                    format!("{p}.self_attn.k_proj.weight"),
                    format!("{p}.self_attn.v_proj.weight"),
                    format!("{p}.self_attn.o_proj.weight"),
                    format!("{p}.self_attn.q_norm.weight"),
                    format!("{p}.self_attn.k_norm.weight"),
                ] {
                    assert!(src.has(&n), "attention layer {i}: missing {n}");
                }
            }
        }
        // full_attention_interval 4 over 24 blocks: 6 attention, 18 linear.
        assert_eq!((linear, full), (18, 6), "hybrid split on the 2B");
        assert!(src.has("model.embed_tokens.weight") && src.has("model.norm.weight"));
    }

    /// The shapes the kernels were written against, read off the real file
    /// rather than assumed. conv_channels is the load-bearing one: the joint
    /// qkv projection's output width must EQUAL 2·n_group·d_state + d_inner, or
    /// the conv kernel is reading a differently-packed row than lane B's
    /// reference and every gate downstream is meaningless.
    #[test]
    #[ignore]
    fn qwen35_2b_shapes_match_the_kernel_assumptions() {
        let Some(path) = qwen35_gguf() else { panic!("Qwen3.5-2B GGUF not in the HF cache") };
        let src = LowMemSource::open(&path).expect("opens");
        let meta = src.qwen35().expect("qwen35 meta parses");
        let d = crate::lowmem::qwen35_ref::DeltaDims {
            d_state: meta.d_state,
            n_v_heads: meta.dt_rank,
            n_k_heads: meta.n_group,
            d_conv: meta.d_conv,
        };
        assert_eq!(d.d_inner(), meta.d_inner, "d_inner = dt_rank · d_state");
        // 2B: 2·16·128 + 2048 = 6144, which must be attn_qkv's output width.
        assert_eq!(d.conv_channels(), 6144);
        assert_eq!(meta.conv_state_elems, (meta.d_conv - 1) * d.conv_channels());
        assert_eq!(meta.delta_state_elems, meta.d_state * meta.d_inner);
    }
    use super::*;
    use crate::config::EosIds;

    fn qwen05b_cfg() -> ModelConfig {
        ModelConfig {
            architectures: vec!["Qwen2ForCausalLM".into()],
            hidden_size: 896,
            intermediate_size: 4864,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            max_position_embeddings: 32768,
            eos_token_id: EosIds::default(),
        }
    }

    /// D9's refuse-to-start path: an impossible budget errors with the
    /// arithmetic in the message, and a feasible one splits exactly.
    #[test]
    fn budget_arithmetic_refuses_impossible_budgets() {
        let cfg = qwen05b_cfg();
        let win = WindowCfg::new(2048, 4).unwrap();
        let err = match memory_plan(&cfg, Dims::new(&cfg, None), &win, 300, None) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a 300 MB budget must be refused"),
        };
        assert!(err.contains("--memory-budget 300"), "{err}");
        assert!(err.contains("weight-pool floor"), "{err}");
        let plan = memory_plan(&cfg, Dims::new(&cfg, None), &win, 4096, None).unwrap();
        let total = plan.kv_bytes
            + plan.act_bytes
            + plan.state_bytes
            + (OVERHEAD_MB << 20)
            + plan.pool_bytes;
        assert_eq!(total, 4096 << 20);
        assert_eq!(plan.state_bytes, 0, "no recurrent term without qwen35 meta");
        assert!(plan.pool_bytes >= 4 * PAGE_BYTES);
    }

    /// A trunk-shaped meta for budget tests: interval-4 recurrency over
    /// `trunk` layers with the REAL 27B per-layer state sizes.
    fn q35_meta_27b(trunk: usize) -> gguf::Qwen35Meta {
        let is_recurrent: Vec<bool> = (0..trunk).map(|i| (i + 1) % 4 != 0).collect();
        gguf::Qwen35Meta {
            trunk_layers: trunk,
            nextn_layers: 1,
            full_attention_interval: 4,
            is_recurrent,
            d_conv: 4,
            d_state: 128,
            n_group: 16,
            dt_rank: 48,
            d_inner: 6144,
            rope_sections: [11, 11, 10, 0],
            conv_state_elems: 3 * (6144 + 2 * 16 * 128), // 30,720
            delta_state_elems: 128 * 6144,               // 786,432
        }
    }

    /// The qwen35 budget shape: KV over the 16 ATTENTION layers only, plus a
    /// fixed recurrent-state term that is constant in the window (i.e. in
    /// context) — and no trace of the MTP block's attention (the map's length
    /// is the trunk, so a 17th KV layer cannot even be expressed).
    #[test]
    fn budget_arithmetic_qwen35_kv_on_attention_layers_only() {
        let mut cfg = qwen05b_cfg();
        cfg.num_hidden_layers = 64;
        let dims = Dims::new(&cfg, None);
        let meta = q35_meta_27b(64);
        assert_eq!(meta.is_recurrent.iter().filter(|&&r| !r).count(), 16);

        let win = WindowCfg::new(2048, 4).unwrap();
        let plan = memory_plan(&cfg, dims, &win, 4096, Some(&meta)).unwrap();
        // KV: 16 layers, f16 K+V — identical to a 16-layer dense model, and
        // 4x smaller than the 64-layer misread.
        assert_eq!(plan.kv_bytes, 16 * win.cap * dims.kv_dim * 2 * 2);
        // State: 48 linear layers × (conv + delta) × f32 — the 27B's ≈150 MB.
        assert_eq!(plan.state_bytes, 48 * (30_720 + 786_432) * 4);
        assert_eq!(plan.state_bytes >> 20, 149);
        let total = plan.kv_bytes
            + plan.act_bytes
            + plan.state_bytes
            + (OVERHEAD_MB << 20)
            + plan.pool_bytes;
        assert_eq!(total, 4096 << 20);

        // Constant in ctx: a 4x window moves KV, never the state term.
        let wide = WindowCfg::new(8192, 4).unwrap();
        let p2 = memory_plan(&cfg, dims, &wide, 4096, Some(&meta)).unwrap();
        assert_eq!(p2.state_bytes, plan.state_bytes);
        assert!(p2.kv_bytes > plan.kv_bytes);

        // The refusal text carries the new term and the honest layer count.
        let err = match memory_plan(&cfg, dims, &win, 300, Some(&meta)) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a 300 MB budget must be refused"),
        };
        assert!(err.contains("recurrent state 149 MB"), "{err}");
        assert!(err.contains("× 16 layers"), "{err}");
    }
}
