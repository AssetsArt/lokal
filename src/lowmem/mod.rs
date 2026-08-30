//! -b lowmem — a disk-backed, bounded-memory backend.
//!
//! A different philosophy from metal/hybrid: those move the whole model onto
//! the GPU and win on speed; lowmem promises a bounded, predictable footprint
//! and accepts what that costs. The mmapped weight files are the source of
//! truth; RAM holds a working set, never a copy of the model.
//!
//! Phase 1 (this commit): the module boundary and the manifest. Construction
//! reads every tensor through the manifest, so its parsing and conversion are
//! exercised end to end — but the weights still end up eagerly resident via the
//! existing Metal engine. Paging, the buffer pool, windowed attention, and the
//! memory budget land in later phases.

mod manifest;

use crate::config::ModelConfig;
use crate::engine::{Engine, Session};
use crate::gpu::metal::MetalEngine;
use crate::model::{Block, Linear, Model};
use manifest::WeightManifest;
use std::path::Path;
use std::time::Instant;

pub struct LowMemEngine {
    /// The mmap layer — later phases page weights out of this on demand.
    #[allow(dead_code)] // phase 1 reads through it only during construction
    manifest: WeightManifest,
    /// Phase-1 compute: the full Metal engine, weights eagerly resident.
    metal: MetalEngine,
}

impl LowMemEngine {
    /// Built from the model DIRECTORY, not a loaded Model — the point of this
    /// backend is never materializing the full model in RAM (phase 1 still
    /// does internally; the seam is what this constructor establishes).
    pub fn new(dir: &Path, cfg: ModelConfig) -> crate::Result<Self> {
        let t0 = Instant::now();
        let manifest = WeightManifest::open(dir)?;
        eprintln!(
            "lowmem: manifest {} tensors | {:.1}M params (headers parsed in {:.2}s)",
            manifest.n_tensors(),
            manifest.n_params as f64 / 1e6,
            t0.elapsed().as_secs_f64(),
        );
        let model = build_model(&manifest, cfg)?;
        let metal = MetalEngine::new(model)?;
        Ok(Self { manifest, metal })
    }
}

impl Engine for LowMemEngine {
    fn name(&self) -> &'static str {
        "lowmem"
    }
    fn config(&self) -> &ModelConfig {
        self.metal.config()
    }
    fn session(&self, max_seq: usize) -> crate::Result<Box<dyn Session + '_>> {
        self.metal.session(max_seq)
    }
    // batcher() stays None: serve mode falls back to per-request sessions,
    // which is the documented behavior for this backend.
}

/// Mirror of Model::load's wiring, reading through the manifest instead of
/// weights::load — proves the manifest end to end before paging exists.
fn build_model(mf: &WeightManifest, cfg: ModelConfig) -> crate::Result<Model> {
    let (h, kv) = (cfg.hidden_size, cfg.kv_dim());
    let lin = |name: &str, in_dim: usize, out_dim: usize| -> crate::Result<Linear> {
        let w = mf.read_f32(&format!("{name}.weight"))?;
        if w.len() != in_dim * out_dim {
            return Err(format!(
                "{name}.weight has {} values but the config implies {out_dim}×{in_dim}",
                w.len()
            )
            .into());
        }
        let bias_name = format!("{name}.bias");
        let bias = if mf.has(&bias_name) { Some(mf.read_f32(&bias_name)?) } else { None };
        Ok(Linear { w, bias, in_dim, out_dim })
    };

    let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        blocks.push(Block {
            input_layernorm: mf.read_f32(&format!("{p}.input_layernorm.weight"))?,
            q_proj: lin(&format!("{p}.self_attn.q_proj"), h, h)?,
            k_proj: lin(&format!("{p}.self_attn.k_proj"), h, kv)?,
            v_proj: lin(&format!("{p}.self_attn.v_proj"), h, kv)?,
            o_proj: lin(&format!("{p}.self_attn.o_proj"), h, h)?,
            post_attention_layernorm: mf.read_f32(&format!("{p}.post_attention_layernorm.weight"))?,
            gate_proj: lin(&format!("{p}.mlp.gate_proj"), h, cfg.intermediate_size)?,
            up_proj: lin(&format!("{p}.mlp.up_proj"), h, cfg.intermediate_size)?,
            down_proj: lin(&format!("{p}.mlp.down_proj"), cfg.intermediate_size, h)?,
        });
    }
    let embed_tokens = mf.read_f32("model.embed_tokens.weight")?;
    let norm = mf.read_f32("model.norm.weight")?;
    // Small models often tie weights: no separate lm_head, the embedding table is reused.
    let lm_head_w =
        if mf.has("lm_head.weight") { mf.read_f32("lm_head.weight")? } else { embed_tokens.clone() };
    let lm_head = Linear { w: lm_head_w, bias: None, in_dim: h, out_dim: cfg.vocab_size };

    Ok(Model { n_params: mf.n_params, cfg, embed_tokens, blocks, norm, lm_head })
}
