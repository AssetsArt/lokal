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
pub(crate) mod manifest;
mod pool;

use crate::config::ModelConfig;
use crate::engine::{Engine, Session};
use crate::gpu::metal as gpu;
use manifest::WeightManifest;
use metal::{Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, FunctionConstantValues, MTLDataType};
use pool::{PagedTensor, WeightPool, PAGE_BYTES};
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
    pool_bytes: usize,
}

fn memory_plan(cfg: &ModelConfig, win: &WindowCfg, budget_mb: usize) -> crate::Result<MemoryPlan> {
    let (h, hd, kvd) = (cfg.hidden_size, cfg.head_dim(), cfg.kv_dim());
    let chunk = gpu::PREFILL_CHUNK;
    // KV store: K and V, f16, cap slots per layer — closed-form in the window.
    let kv_bytes = cfg.num_hidden_layers * win.cap * kvd * 2 * 2;
    // One session's activation scratch, mirroring LowMemSession::new.
    let scores = if hd == gpu::FLASH_HEAD_DIM {
        4
    } else {
        chunk * cfg.num_attention_heads * win.cap * 4
    };
    let act_bytes = 5 * chunk * h * 4                    // x, xn, q, att, xb
        + 2 * chunk * cfg.intermediate_size * 4          // gate, up
        + 2 * chunk * kvd * 4                            // kvs staging
        + chunk * h * 2                                  // xh
        + scores
        + cfg.num_attention_heads * (win.cap / gpu::ATTN_SPLIT) * (hd + 2) * 4
        + cfg.vocab_size * 4;                            // logits
    let fixed = kv_bytes + act_bytes + (OVERHEAD_MB << 20);
    let budget = budget_mb << 20;
    let floor = 4 * PAGE_BYTES;
    if budget < fixed + floor {
        return Err(format!(
            "--memory-budget {budget_mb} MB cannot hold the working set: KV {} MB (window {} × {} layers) + activations {} MB + runtime overhead {} MB leaves less than the {} MB weight-pool floor — raise the budget or shrink --context-window",
            kv_bytes >> 20,
            win.w,
            cfg.num_hidden_layers,
            act_bytes >> 20,
            OVERHEAD_MB,
            floor >> 20,
        )
        .into());
    }
    Ok(MemoryPlan { kv_bytes, act_bytes, pool_bytes: budget - fixed })
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

/// One transformer block: eagerly-resident norms, paged projection matrices.
pub(crate) struct LayerWeights {
    pub input_ln: Buffer,
    pub post_ln: Buffer,
    pub q: PagedTensor,
    pub k: PagedTensor,
    pub v: PagedTensor,
    pub o: PagedTensor,
    pub gate: PagedTensor,
    pub up: PagedTensor,
    pub down: PagedTensor,
}

pub struct LowMemEngine {
    cfg: ModelConfig,
    manifest: WeightManifest,
    device: Device,
    queue: CommandQueue,
    pipes: Pipes,
    direct: DirectPipes,
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
        let mut manifest = WeightManifest::open(dir)?;
        eprintln!(
            "lowmem: manifest {} tensors | {:.1}M params (headers parsed in {:.2}s)",
            manifest.n_tensors(),
            manifest.n_params as f64 / 1e6,
            t0.elapsed().as_secs_f64(),
        );

        let device = Device::system_default().ok_or("no Metal-capable GPU found")?;
        manifest.make_gpu_views(&device);
        let queue = device.new_command_queue();
        let lib = device
            .new_library_with_source(&gpu::shader_source(cfg.kv_dim()), &CompileOptions::new())
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
        // the mmap views; 2..6 = the GGUF quant types (those come from the
        // precise fast-math-off library when the engine wires them).
        let qtype_pipe = |name: &str, qtype: u32| -> crate::Result<ComputePipelineState> {
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
            Ok(gpu::f16_buffer(&device, &manifest.read_f32(&name)?))
        };
        let mut next_id = 0u32;
        let mut mk = |prefix: String, in_dim: usize, out_dim: usize| -> crate::Result<PagedTensor> {
            let bias_name = format!("{prefix}.bias");
            let bias = if manifest.has(&bias_name) {
                Some(gpu::f16_buffer(&device, &manifest.read_f32(&bias_name)?))
            } else {
                None
            };
            let t = PagedTensor::new(
                &manifest,
                next_id,
                format!("{prefix}.weight"),
                in_dim,
                out_dim,
                bias,
            )?;
            next_id += 1;
            Ok(t)
        };

        let (h, kv) = (cfg.hidden_size, cfg.kv_dim());
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(LayerWeights {
                input_ln: small(format!("{p}.input_layernorm.weight"))?,
                post_ln: small(format!("{p}.post_attention_layernorm.weight"))?,
                q: mk(format!("{p}.self_attn.q_proj"), h, h)?,
                k: mk(format!("{p}.self_attn.k_proj"), h, kv)?,
                v: mk(format!("{p}.self_attn.v_proj"), h, kv)?,
                o: mk(format!("{p}.self_attn.o_proj"), h, h)?,
                gate: mk(format!("{p}.mlp.gate_proj"), h, cfg.intermediate_size)?,
                up: mk(format!("{p}.mlp.up_proj"), h, cfg.intermediate_size)?,
                down: mk(format!("{p}.mlp.down_proj"), cfg.intermediate_size, h)?,
            });
        }
        let final_norm = small("model.norm.weight".into())?;
        // Tied weights: no lm_head tensor means the pager reads the embedding
        // table's rows — same bytes, no copy anywhere.
        let lm_name =
            if manifest.has("lm_head.weight") { "lm_head" } else { "model.embed_tokens" };
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
        let plan = memory_plan(&cfg, &win, budget_mb)?;
        let pool_bytes = match std::env::var("LOKAL_LOWMEM_POOL_MB") {
            Ok(v) => v.parse::<usize>().map(|mb| mb << 20).unwrap_or(plan.pool_bytes),
            Err(_) => plan.pool_bytes,
        };
        let pool = WeightPool::new(&device, pool_bytes);
        let staged_bytes = manifest.n_params * 2; // f16 in the pool, whatever the disk dtype
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
        eprintln!(
            "lowmem: {} — budget {} MB = weights {} MB (paged, ≤{} MB pages) + KV {} MB (window {} +{} sink × {} layers) + activations {} MB + overhead {} MB",
            device.name(),
            budget_mb,
            pool_bytes >> 20,
            PAGE_BYTES >> 20,
            plan.kv_bytes >> 20,
            win.w,
            win.sink,
            cfg.num_hidden_layers,
            plan.act_bytes >> 20,
            OVERHEAD_MB,
        );

        let clip_flag =
            device.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        unsafe { *(clip_flag.contents() as *mut u32) = 0 };

        Ok(Self {
            gqa: gpu::gqa_decode_dims(&cfg),
            sync: std::env::var("LOKAL_LOWMEM_SYNC").is_ok_and(|v| v == "1"),
            win,
            clip_flag,
            clip_warned: std::sync::atomic::AtomicBool::new(false),
            cfg,
            manifest,
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
        let err = match memory_plan(&cfg, &win, 300) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a 300 MB budget must be refused"),
        };
        assert!(err.contains("--memory-budget 300"), "{err}");
        assert!(err.contains("weight-pool floor"), "{err}");
        let plan = memory_plan(&cfg, &win, 4096).unwrap();
        let total = plan.kv_bytes + plan.act_bytes + (OVERHEAD_MB << 20) + plan.pool_bytes;
        assert_eq!(total, 4096 << 20);
        assert!(plan.pool_bytes >= 4 * PAGE_BYTES);
    }
}
