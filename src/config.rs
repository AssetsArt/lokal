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
