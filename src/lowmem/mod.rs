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
mod manifest;
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

/// Pool size until --memory-budget lands (D9's closed-form arithmetic replaces
/// this: pool = budget − KV − activations − fixed overhead, default budget 4096).
const POOL_MB_DEFAULT: usize = 3072;

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
    fn new(w: usize, sink: usize) -> crate::Result<Self> {
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
}

// Same justification as MetalEngine: Apple documents these Metal objects as
// thread-safe; the mutable pool sits behind the Mutex above.
unsafe impl Send for LowMemEngine {}
unsafe impl Sync for LowMemEngine {}

impl LowMemEngine {
    /// Built from the model DIRECTORY, not a loaded Model — nothing here ever
    /// materializes the full model in RAM.
    pub fn new(dir: &Path, cfg: ModelConfig) -> crate::Result<Self> {
        let t0 = Instant::now();
        let manifest = WeightManifest::open(dir)?;
        eprintln!(
            "lowmem: manifest {} tensors | {:.1}M params (headers parsed in {:.2}s)",
            manifest.n_tensors(),
            manifest.n_params as f64 / 1e6,
            t0.elapsed().as_secs_f64(),
        );

        let device = Device::system_default().ok_or("no Metal-capable GPU found")?;
        let queue = device.new_command_queue();
        let lib = device
            .new_library_with_source(&gpu::shader_source(cfg.kv_dim()), &CompileOptions::new())
            .map_err(|e| format!("failed to compile kernels.metal: {e}"))?;
        let pipe = |name: &str| -> crate::Result<ComputePipelineState> {
            let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("kernel {name}: {e}").into())
        };
        let env_usize = |k: &str, d: usize| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        // Debug overrides until the --context-window / --attention-sink flags
        // land with the CLI phase.
        let win = WindowCfg::new(
            env_usize("LOKAL_LOWMEM_WINDOW", 2048),
            env_usize("LOKAL_LOWMEM_SINK", 4),
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

        // Debug override until --memory-budget lands; the budget arithmetic
        // (D9) replaces this as the public knob.
        let pool_mb = std::env::var("LOKAL_LOWMEM_POOL_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(POOL_MB_DEFAULT);
        let pool = WeightPool::new(&device, pool_mb << 20);
        let kv_mb = cfg.num_hidden_layers * win.cap * cfg.kv_dim() * 2 * 2 >> 20;
        eprintln!(
            "lowmem: {} — weight pool {} MB (pages ≤ {} MB) | window {} +{} sink, KV ring {} MB total",
            device.name(),
            pool_mb,
            PAGE_BYTES >> 20,
            win.w,
            win.sink,
            kv_mb,
        );

        Ok(Self {
            gqa: gpu::gqa_decode_dims(&cfg),
            sync: std::env::var("LOKAL_LOWMEM_SYNC").is_ok_and(|v| v == "1"),
            win,
            cfg,
            manifest,
            device,
            queue,
            pipes,
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
