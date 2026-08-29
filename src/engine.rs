//! The layer that makes compute backends swappable: CPU (default), Metal GPU,
//! ANE — and future ones (CUDA, ROCm).
//!
//! The split is two-level:
//!   - Engine  = loaded weights, ready to run — read-only, shared across threads/requests
//!   - Session = the state of one generation run (the growing KV cache) — single-threaded
//!
//! This separation lets the server handle concurrent requests without any locks.

use crate::config::ModelConfig;
use crate::model::{KvCache, Model};

pub trait Engine: Send + Sync {
    fn name(&self) -> &'static str;
    fn config(&self) -> &ModelConfig;
    /// Start a new generation run with a KV cache sized for `max_seq` tokens.
    fn session(&self, max_seq: usize) -> crate::Result<Box<dyn Session + '_>>;
}

pub trait Session {
    /// Process one token at position `pos` → logits (same contract as Model::forward).
    fn forward(&mut self, token: u32, pos: usize) -> crate::Result<Vec<f32>>;

    /// Process the whole prompt (filling the KV cache) → logits for the last position.
    /// The default walks token by token; backends with batch support override it.
    /// This seam is also where alternative prefill hardware (e.g. the ANE) plugs in.
    fn prefill(&mut self, ids: &[u32]) -> crate::Result<Vec<f32>> {
        let mut logits = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            logits = self.forward(id, pos)?;
        }
        Ok(logits)
    }

    /// Process a short batch starting at `pos0` → logits for EVERY position, not just
    /// the last. This is speculative decoding's verification step: the target model
    /// checks a whole block of draft tokens in one pass. The default loops over
    /// `forward`; the Metal backend overrides it with one batched submission.
    fn forward_batch(&mut self, ids: &[u32], pos0: usize) -> crate::Result<Vec<Vec<f32>>> {
        ids.iter().enumerate().map(|(i, &t)| self.forward(t, pos0 + i)).collect()
    }
}

/// Build an engine by backend name — adding a backend means adding one arm here
/// (see src/gpu/mod.rs for the checklist). `model_dir` is for backends that keep
/// extra files next to the model (the ane backend looks for prefill-*.mlmodelc).
pub fn create(
    backend: &str,
    model: Model,
    model_dir: &std::path::Path,
) -> crate::Result<Box<dyn Engine>> {
    let _ = model_dir; // unused on platforms without a backend that needs it
    match backend {
        "cpu" => Ok(Box::new(CpuEngine { model })),
        #[cfg(target_os = "macos")]
        "metal" => Ok(Box::new(crate::gpu::metal::MetalEngine::new(model)?)),
        #[cfg(target_os = "macos")]
        "ane" => Ok(Box::new(crate::ane::AneEngine::new(model, model_dir)?)),
        other => Err(format!(
            "unknown backend \"{other}\" — available: cpu{}",
            if cfg!(target_os = "macos") { ", metal, ane" } else { "" }
        )
        .into()),
    }
}

// ---------- CPU backend: wraps the reference Model + KvCache in the traits ----------

pub struct CpuEngine {
    pub model: Model,
}

struct CpuSession<'a> {
    model: &'a Model,
    cache: KvCache,
}

impl Engine for CpuEngine {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn config(&self) -> &ModelConfig {
        &self.model.cfg
    }
    fn session(&self, max_seq: usize) -> crate::Result<Box<dyn Session + '_>> {
        Ok(Box::new(CpuSession {
            model: &self.model,
            cache: KvCache::new(&self.model.cfg, max_seq),
        }))
    }
}

impl Session for CpuSession<'_> {
    fn forward(&mut self, token: u32, pos: usize) -> crate::Result<Vec<f32>> {
        Ok(self.model.forward(token, pos, &mut self.cache))
    }
}
