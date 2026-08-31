//! Model hyperparameters, read straight from a Hugging Face config.json.
//!
//! Field names match the JSON keys exactly so the file and the code can be
//! compared side by side.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub hidden_size: usize,             // width of each token's hidden state
    pub intermediate_size: usize,       // MLP inner width (usually ~2.7x hidden)
    pub num_hidden_layers: usize,       // number of transformer blocks
    pub num_attention_heads: usize,     // query heads
    pub num_key_value_heads: usize,     // K/V heads — fewer than query heads under GQA
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,                // RoPE frequency base (original paper used 10000)
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize, // maximum context length the model was trained for
    #[serde(default)]
    pub eos_token_id: EosIds,
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_max_pos() -> usize {
    4096
}

impl ModelConfig {
    pub fn load(path: &Path) -> crate::Result<Self> {
        let cfg: Self = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        // The forward pass implemented here is the Llama architecture. These families all
        // share it (Qwen2 differs only by q/k/v biases, which model.rs detects from the
        // weight files on its own).
        const SUPPORTED: &[&str] = &["LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM"];
        if let Some(a) = cfg.architectures.first() {
            // The Qwen3 family (Qwen3*, Qwen3_5*) is KNOWN to break the Llama
            // walk this path implements: config.json states an explicit
            // head_dim that violates the hidden/heads identity head_dim()
            // assumes, and the checkpoints carry q/k-norm weights the dense
            // walks never apply. Running one as-Llama reads q/k at the wrong
            // width — degenerate CLI output, garbage nondeterministic serve —
            // so refuse by name and point at the path that actually works.
            // The as-Llama attempt below stays for genuinely llama-shaped
            // architectures we merely haven't tested.
            if a.starts_with("Qwen3") {
                return Err(format!(
                    "architecture {a} is not supported from safetensors: its explicit \
                     head_dim and qk-norm do not fit the Llama forward pass this path \
                     implements — run the GGUF instead (-m owner/repo:Q4_K_M or a local \
                     .gguf); safetensors support for this architecture is not wired yet"
                )
                .into());
            }
            if !SUPPORTED.contains(&a.as_str()) {
                eprintln!("warning: architecture {a} is untested — attempting to run it as Llama");
            }
        }
        Ok(cfg)
    }

    /// Vector width of a single attention head.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Total width of K (or V) per token — smaller than hidden_size under GQA.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_token_id.contains(id)
    }
}

/// config.json stores eos_token_id as either a single id or a list, depending on the model.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosIds {
    One(u32),
    Many(Vec<u32>),
}

impl Default for EosIds {
    fn default() -> Self {
        Self::Many(Vec::new())
    }
}

impl EosIds {
    pub fn contains(&self, id: u32) -> bool {
        match self {
            Self::One(x) => *x == id,
            Self::Many(xs) => xs.contains(&id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_arch(arch: &str) -> crate::Result<ModelConfig> {
        let json = format!(
            r#"{{"architectures":["{arch}"],"hidden_size":64,"intermediate_size":128,
                "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
                "vocab_size":100,"rms_norm_eps":1e-6}}"#
        );
        let dir = std::env::temp_dir().join(format!("lokal-cfg-test-{arch}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, json).unwrap();
        let r = ModelConfig::load(&path);
        let _ = std::fs::remove_dir_all(&dir);
        r
    }

    /// The Qwen3 family refuses by NAME with the working alternative in the
    /// text — never the silent as-Llama walk that produced degenerate output.
    #[test]
    fn qwen3_family_is_refused_by_name() {
        for arch in ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "Qwen3_5ForCausalLM"] {
            let err = match load_arch(arch) {
                Err(e) => e.to_string(),
                Ok(_) => panic!("{arch} must be refused"),
            };
            assert!(err.contains(arch), "{err}");
            assert!(err.contains("not supported from safetensors"), "{err}");
            assert!(err.contains(".gguf"), "names the working alternative: {err}");
        }
    }

    /// Genuinely llama-shaped unknowns keep the as-Llama attempt (the warning
    /// goes to stderr — behavior, not assertable text, is what matters here),
    /// and the supported list loads silently.
    #[test]
    fn llama_shaped_archs_still_load() {
        for arch in ["LlamaForCausalLM", "Qwen2ForCausalLM", "SomeNewLlamaCloneForCausalLM"] {
            assert!(load_arch(arch).is_ok(), "{arch} must load");
        }
        // Qwen2 < Qwen3 lexically but shares the prefix up to the digit —
        // prove the refusal never swallows it.
        assert!(load_arch("Qwen2ForCausalLM").is_ok());
    }
}
