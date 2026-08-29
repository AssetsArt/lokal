//! Model resolution through the standard Hugging Face cache.
//!
//! lokal shares the exact cache used by transformers, candle, and every other
//! HF-ecosystem tool (`~/.cache/huggingface/hub`, overridable via HF_HOME):
//! anything already downloaded on this machine is reused as-is, and anything
//! lokal downloads becomes available to those tools. Gated and private repos
//! work automatically when a token is present (HF_TOKEN or `hf auth login`).
//!
//! A resolved model is a snapshot directory holding the three files inference
//! needs: config.json, tokenizer.json, and the safetensors weights.

use hf_hub::HFClientSync;
use std::path::{Path, PathBuf};

/// Accepts either a Hub repo id (e.g. "HuggingFaceTB/SmolLM2-135M") or a local
/// directory. Returns the directory containing the model files.
pub fn resolve_model(spec: &str) -> crate::Result<PathBuf> {
    let p = Path::new(spec);
    if p.is_dir() {
        return Ok(p.to_path_buf());
    }
    let (owner, name) = spec.split_once('/').ok_or_else(|| {
        format!("model spec \"{spec}\" is neither a local directory nor an owner/name repo id")
    })?;

    let client = HFClientSync::new()?;
    let repo = client.model(owner, name);
    // Fetch only what inference needs (the * covers sharded weight files).
    let patterns: Vec<String> =
        ["config.json", "tokenizer.json", "model*.safetensors", "model.safetensors.index.json"]
            .iter()
            .map(|s| s.to_string())
            .collect();

    // Try online first so new revisions are picked up; if the network is
    // unavailable, fall back to whatever the cache already holds.
    let snapshot = repo
        .snapshot_download()
        .allow_patterns(patterns.clone())
        .send()
        .or_else(|_| {
            repo.snapshot_download()
                .allow_patterns(patterns)
                .local_files_only(true)
                .send()
        })?;
    Ok(snapshot)
}
