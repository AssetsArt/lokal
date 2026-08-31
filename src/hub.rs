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
    if p.is_dir() || p.is_file() {
        // A file spec is a GGUF checkpoint (main.rs routes on the extension).
        return Ok(p.to_path_buf());
    }
    let (owner, name) = spec.split_once('/').ok_or_else(|| {
        format!("model spec \"{spec}\" is neither a local path nor an owner/name repo id")
    })?;

    // owner/repo/FILE.gguf: fetch that single file into the normal HF cache.
    if let Some((repo_name, file)) = name.split_once('/') {
        if !file.to_ascii_lowercase().ends_with(".gguf") {
            return Err(format!(
                "model spec \"{spec}\" names a file inside a repo — only .gguf files resolve that way"
            )
            .into());
        }
        let client = HFClientSync::new()?;
        let repo = client.model(owner, repo_name);
        let one = vec![file.to_string()];
        let snapshot = repo
            .snapshot_download()
            .allow_patterns(one.clone())
            .send()
            .or_else(|_| {
                repo.snapshot_download().allow_patterns(one).local_files_only(true).send()
            })?;
        let path = snapshot.join(file);
        if !path.is_file() {
            return Err(format!("{owner}/{repo_name} has no file named {file}").into());
        }
        return Ok(path);
    }

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

/// Where a model's exported Core ML graphs live: a lokal-owned directory,
/// deliberately OUTSIDE the Hugging Face cache. The graphs used to be written
/// next to the weights inside the snapshot directory, which had two failure
/// modes seen in the wild: an upstream revision bump orphans them (new snapshot
/// dir, minutes of re-export), and `hf cache delete` removes gigabytes of
/// artifacts it never created. This function is the ONE implementation of the
/// path rule — `lokal path --graphs`, the ane backend, and (via run.sh) the
/// exporter all resolve through it.
pub struct GraphLocation {
    /// The lokal-owned graph directory. Not necessarily created yet.
    pub dir: PathBuf,
    /// "owner/name" for a Hub model, the directory path for a local one.
    /// For messages and re-export commands, so errors can name the model.
    pub model: String,
    /// Basename of the HF snapshot the model resolved to; None for a local
    /// directory. Recorded in graphs.json and checked on load — graphs built
    /// from one revision's weights must not run against another's.
    pub revision: Option<String>,
}

/// `$LOKAL_GRAPH_DIR` if set (it then names ONE model's graph directory), else
/// `${XDG_CACHE_HOME:-~/.cache}/lokal/coreml/<slug>/`. The slug is the repo id
/// with `/` → `--` (matching the HF cache's own naming, so the two are easy to
/// correlate), or for a local model the directory name plus 8 hex of a
/// stable hash of its canonical path — two checkouts both named "model" must not collide.
/// Deliberately no hash sub-directory: `ls ~/.cache/lokal/coreml` must read as
/// a list of models.
pub fn graph_location(resolved: &Path) -> GraphLocation {
    let canon = resolved.canonicalize().unwrap_or_else(|_| resolved.to_path_buf());
    // A HF snapshot lives at …/models--<owner>--<name>/snapshots/<revision>/;
    // anything shaped differently is a local directory.
    let hub_parts = (|| {
        let rev = canon.file_name()?.to_str()?;
        let snapshots = canon.parent()?;
        if snapshots.file_name()? != "snapshots" {
            return None;
        }
        let repo = snapshots.parent()?.file_name()?.to_str()?;
        Some((repo.strip_prefix("models--")?.to_string(), rev.to_string()))
    })();
    let (slug, model, revision) = match hub_parts {
        Some((slug, rev)) => {
            let model = slug.replacen("--", "/", 1);
            (slug, model, Some(rev))
        }
        None => {
            let name = canon
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".into());
            let digest = path_hash_hex(canon.to_string_lossy().as_bytes());
            (format!("{name}-{digest}"), canon.display().to_string(), None)
        }
    };
    let dir = match std::env::var_os("LOKAL_GRAPH_DIR") {
        Some(d) if !d.is_empty() => PathBuf::from(d),
        _ => graph_cache_base().join(slug),
    };
    GraphLocation { dir, model, revision }
}

/// `${XDG_CACHE_HOME:-~/.cache}/lokal/coreml` — the parent of every per-model
/// graph directory. Public so a missing-graphs error can list which models DO
/// have graphs here (the classic mistake is exporting for `…-Instruct` and
/// running the base model).
pub fn graph_cache_base() -> PathBuf {
    match std::env::var_os("XDG_CACHE_HOME") {
        Some(x) if !x.is_empty() => PathBuf::from(x),
        _ => {
            std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default().join(".cache")
        }
    }
    .join("lokal")
    .join("coreml")
}

/// FNV-1a 64-bit, used only to give local-directory models a stable slug
/// suffix — a cache-directory NAME, not a security boundary (lead ruling,
/// 2026-08-30). Deliberately NOT std's DefaultHasher: its output is documented
/// unstable across Rust releases, so a toolchain upgrade would change the slug
/// and orphan every exported graph — the exact bug this module exists to kill.
fn path_hash_hex(data: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{:08x}", h as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_hash_matches_fnv1a_vectors() {
        // Published FNV-1a 64 values, low 32 bits hexed.
        assert_eq!(path_hash_hex(b""), "84222325");
        assert_eq!(path_hash_hex(b"a"), "8601ec8c");
        assert_eq!(path_hash_hex(b"foobar"), "f73967e8");
    }

    #[test]
    fn hub_snapshot_maps_to_readable_slug() {
        // The path need not exist: canonicalize falls back to the path as given.
        let loc = graph_location(Path::new(
            "/nonexistent/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f",
        ));
        assert!(loc.dir.ends_with("lokal/coreml/HuggingFaceTB--SmolLM2-135M-Instruct"));
        assert_eq!(loc.model, "HuggingFaceTB/SmolLM2-135M-Instruct");
        assert_eq!(loc.revision.as_deref(), Some("12fd25f"));
    }

    #[test]
    fn local_dir_slug_carries_a_path_hash() {
        let a = graph_location(Path::new("/nonexistent/checkouts/a/model"));
        let b = graph_location(Path::new("/nonexistent/checkouts/b/model"));
        assert!(a.revision.is_none());
        // Same directory name, different paths — the 8-hex suffix keeps them apart.
        let (an, bn) = (a.dir.file_name().unwrap(), b.dir.file_name().unwrap());
        assert_ne!(an, bn);
        assert!(an.to_string_lossy().starts_with("model-"));
        assert_eq!(an.to_string_lossy().len(), "model-".len() + 8);
    }
}
