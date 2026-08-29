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
    Device, MTLResourceOptions, MTLSize,
};

/// Maximum tokens processed together during prefill (one command buffer per chunk).
/// Bigger chunks amortize weight reads better but grow the scratch buffers
/// (the attention scores buffer in particular).
const PREFILL_CHUNK: usize = 128;

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
struct AttnParams {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    pos0: u32,
    max_seq: u32,
}
#[repr(C)]
struct ElemParams {
    dim: u32,
}

/// A pipeline is a compiled kernel ready to dispatch — one per kernel in kernels.metal.
struct Pipelines {
    embed: ComputePipelineState,
    matvec: ComputePipelineState,
    matmul: ComputePipelineState,
    rmsnorm: ComputePipelineState,
    rope: ComputePipelineState,
    attention: ComputePipelineState,
    silu_mul: ComputePipelineState,
    add_inplace: ComputePipelineState,
}

/// A linear layer on the GPU: f16 weights + f16 bias (all-zero when the model has none —
/// adding zero is free and avoids a branch in the kernel).
struct GpuLinear {
    w: Buffer,
    bias: Buffer,
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
        let pipes = Pipelines {
            embed: pipe("embed")?,
            matvec: pipe("matvec")?,
            matmul: pipe("matmul")?,
            rmsnorm: pipe("rmsnorm")?,
            rope: pipe("rope")?,
            attention: pipe("attention")?,
            silu_mul: pipe("silu_mul")?,
            add_inplace: pipe("add_inplace")?,
        };

        fn lin(device: &Device, l: &crate::model::Linear) -> GpuLinear {
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
    ) {
        if n_rows == 1 {
            let p = MatvecParams { in_dim: l.in_dim, out_dim: l.out_dim };
            enc.set_compute_pipeline_state(&self.pipes.matvec);
            enc.set_buffer(0, Some(&l.w), 0);
            enc.set_buffer(1, Some(&l.bias), 0);
            enc.set_buffer(2, Some(x), x_off);
            enc.set_buffer(3, Some(y), y_off);
            enc.set_bytes(4, size_of::<MatvecParams>() as u64, &p as *const _ as *const _);
            // 4 simdgroups (W rows) per threadgroup — the grid covers every row.
            let rows_per_tg = 4u64;
            let tgs = (l.out_dim as u64).div_ceil(rows_per_tg);
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(32 * rows_per_tg, 1, 1));
        } else {
            let p = MatmulParams { in_dim: l.in_dim, out_dim: l.out_dim, n_rows: n_rows as u32 };
            enc.set_compute_pipeline_state(&self.pipes.matmul);
            enc.set_buffer(0, Some(&l.w), 0);
            enc.set_buffer(1, Some(&l.bias), 0);
            enc.set_buffer(2, Some(x), x_off);
            enc.set_buffer(3, Some(y), y_off);
            enc.set_bytes(4, size_of::<MatmulParams>() as u64, &p as *const _ as *const _);
            // 2D grid: (tiles of 32 outputs) × (tiles of 8 tokens) — see MM_* in kernels.metal.
            let tiles_out = (l.out_dim as u64).div_ceil(32);
            let tiles_row = (n_rows as u64).div_ceil(8);
            enc.dispatch_thread_groups(
                MTLSize::new(tiles_out, tiles_row, 1),
                MTLSize::new(256, 1, 1),
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

    #[allow(clippy::too_many_arguments)]
    fn enc_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_off: u64,
        n_heads: usize,
        pos0: usize,
        n_rows: usize,
    ) {
        let hd = self.cfg.head_dim();
        let p = RopeParams {
            head_dim: hd as u32,
            n_heads: n_heads as u32,
            pos0: pos0 as u32,
            theta: self.cfg.rope_theta,
            n_rows: n_rows as u32,
        };
        enc.set_compute_pipeline_state(&self.pipes.rope);
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
        scores: &Buffer,
        out: &Buffer,
        pos0: usize,
        n_rows: usize,
        max_seq: usize,
    ) {
        let p = AttnParams {
            head_dim: self.cfg.head_dim() as u32,
            n_heads: self.cfg.num_attention_heads as u32,
            n_kv_heads: self.cfg.num_key_value_heads as u32,
            pos0: pos0 as u32,
            max_seq: max_seq as u32,
        };
        enc.set_compute_pipeline_state(&self.pipes.attention);
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k_cache), 0);
        enc.set_buffer(2, Some(v_cache), 0);
        enc.set_buffer(3, Some(scores), 0);
        enc.set_buffer(4, Some(out), 0);
        enc.set_bytes(5, size_of::<AttnParams>() as u64, &p as *const _ as *const _);
        // 2D grid: (query heads) × (query rows in the chunk) — one threadgroup per pair.
        enc.dispatch_thread_groups(
            MTLSize::new(self.cfg.num_attention_heads as u64, n_rows as u64, 1),
            MTLSize::new(256, 1, 1),
        );
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
}

impl MetalEngine {
    /// Build a session as a concrete type — the ane backend needs write_kv/prefill_from.
    pub(crate) fn raw_session(&self, max_seq: usize) -> MetalSession<'_> {
        let cfg = &self.cfg;
        let d = &self.device;
        let chunk = PREFILL_CHUNK.min(max_seq); // scratch sized per chunk (decode uses row 0 only)
        MetalSession {
            ids: d.new_buffer((chunk * 4) as u64, MTLResourceOptions::StorageModeShared),
            x: f32_buffer(d, chunk * cfg.hidden_size),
            xn: f32_buffer(d, chunk * cfg.hidden_size),
            q: f32_buffer(d, chunk * cfg.hidden_size),
            att: f32_buffer(d, chunk * cfg.hidden_size),
            xb: f32_buffer(d, chunk * cfg.hidden_size),
            gate: f32_buffer(d, chunk * cfg.intermediate_size),
            up: f32_buffer(d, chunk * cfg.intermediate_size),
            logits: f32_buffer(d, cfg.vocab_size),
            scores: f32_buffer(d, chunk * cfg.num_attention_heads * max_seq),
            k_cache: (0..cfg.num_hidden_layers).map(|_| f32_buffer(d, max_seq * cfg.kv_dim())).collect(),
            v_cache: (0..cfg.num_hidden_layers).map(|_| f32_buffer(d, max_seq * cfg.kv_dim())).collect(),
            max_seq,
            engine: self,
        }
    }
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
    k_cache: Vec<Buffer>,
    v_cache: Vec<Buffer>,
    max_seq: usize,
}

impl MetalSession<'_> {
    /// Process n_rows tokens (positions pos0..pos0+n_rows) in one command buffer.
    /// The dispatch order below mirrors Model::forward on the CPU line by line.
    /// want_logits=false for intermediate prefill chunks that only fill the KV cache.
    fn run(&mut self, ids: &[u32], pos0: usize, want_logits: bool) -> crate::Result<Vec<f32>> {
        let e = self.engine;
        let cfg = &e.cfg;
        let (h, n) = (cfg.hidden_size, ids.len());
        let kv_byte_off = (pos0 * cfg.kv_dim() * 4) as u64; // this chunk's first cache slot

        // Push the token ids to the GPU (unified memory: write into the buffer pre-commit).
        unsafe { std::ptr::copy_nonoverlapping(ids.as_ptr(), self.ids.contents() as *mut u32, n) };

        let cb = e.queue.new_command_buffer();
        // Serial encoder: dispatches execute in order, each seeing its predecessors'
        // results — no manual barriers needed.
        let enc = cb.new_compute_command_encoder();

        e.enc_embed(enc, &self.ids, &self.x, n);
        for (l, blk) in e.blocks.iter().enumerate() {
            // Attention half.
            e.enc_rmsnorm(enc, &self.x, &blk.input_layernorm, &self.xn, n);
            e.enc_linear(enc, &blk.q_proj, &self.xn, 0, &self.q, 0, n);
            e.enc_linear(enc, &blk.k_proj, &self.xn, 0, &self.k_cache[l], kv_byte_off, n);
            e.enc_linear(enc, &blk.v_proj, &self.xn, 0, &self.v_cache[l], kv_byte_off, n);
            e.enc_rope(enc, &self.q, 0, cfg.num_attention_heads, pos0, n);
            e.enc_rope(enc, &self.k_cache[l], kv_byte_off, cfg.num_key_value_heads, pos0, n);
            e.enc_attention(enc, &self.q, &self.k_cache[l], &self.v_cache[l], &self.scores, &self.att, pos0, n, self.max_seq);
            e.enc_linear(enc, &blk.o_proj, &self.att, 0, &self.xb, 0, n);
            e.enc_elementwise(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);

            // SwiGLU MLP half.
            e.enc_rmsnorm(enc, &self.x, &blk.post_attention_layernorm, &self.xn, n);
            e.enc_linear(enc, &blk.gate_proj, &self.xn, 0, &self.gate, 0, n);
            e.enc_linear(enc, &blk.up_proj, &self.xn, 0, &self.up, 0, n);
            e.enc_elementwise(enc, &e.pipes.silu_mul, &self.gate, &self.up, n * cfg.intermediate_size);
            e.enc_linear(enc, &blk.down_proj, &self.gate, 0, &self.xb, 0, n);
            e.enc_elementwise(enc, &e.pipes.add_inplace, &self.x, &self.xb, n * h);
        }
        if want_logits {
            // Only the last position's logits matter — norm every row (cheap), then
            // run the big lm_head matvec on the final row alone.
            e.enc_rmsnorm(enc, &self.x, &e.norm, &self.xn, n);
            e.enc_linear(enc, &e.lm_head, &self.xn, ((n - 1) * h * 4) as u64, &self.logits, 0, 1);
        }

        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed(); // the single sync point for the whole chunk

        if !want_logits {
            return Ok(Vec::new());
        }
        // Unified memory: read logits straight out of the buffer, no device copy.
        let logits =
            unsafe { std::slice::from_raw_parts(self.logits.contents() as *const f32, cfg.vocab_size) };
        Ok(logits.to_vec())
    }

    /// Write K,V computed elsewhere (e.g. on the ANE) into the cache at pos0 onward.
    /// With unified memory, "transferring between devices" is just a memcpy.
    pub(crate) fn write_kv(&mut self, layer: usize, pos0: usize, k: &[f32], v: &[f32]) {
        let kvd = self.engine.cfg.kv_dim();
        unsafe {
            let kp = (self.k_cache[layer].contents() as *mut f32).add(pos0 * kvd);
            std::ptr::copy_nonoverlapping(k.as_ptr(), kp, k.len());
            let vp = (self.v_cache[layer].contents() as *mut f32).add(pos0 * kvd);
            std::ptr::copy_nonoverlapping(v.as_ptr(), vp, v.len());
        }
    }

    /// Batch prefill continuing from pos0 (cache slots 0..pos0 must already be filled)
    /// — the ane backend uses this to take over after the ANE's portion.
    pub(crate) fn prefill_from(&mut self, ids: &[u32], mut pos0: usize) -> crate::Result<Vec<f32>> {
        let end = pos0 + ids.len();
        let mut logits = Vec::new();
        for chunk in ids.chunks(PREFILL_CHUNK) {
            let is_last = pos0 + chunk.len() == end;
            logits = self.run(chunk, pos0, is_last)?;
            pos0 += chunk.len();
        }
        Ok(logits)
    }
}

impl Session for MetalSession<'_> {
    fn forward(&mut self, token: u32, pos: usize) -> crate::Result<Vec<f32>> {
        self.run(&[token], pos, true)
    }

    /// Batch prefill: split the prompt into PREFILL_CHUNK-sized chunks. Later chunks
    /// automatically attend to earlier chunks' K,V through the cache (via pos0).
    fn prefill(&mut self, ids: &[u32]) -> crate::Result<Vec<f32>> {
        self.prefill_from(ids, 0)
    }
}
