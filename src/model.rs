//! The reference implementation: a complete Llama-family transformer forward
//! pass in one file, in plain Rust.
//!
//! Path of a single token (SmolLM2 / TinyLlama / Qwen2 / Mistral all share this):
//!
//!   token id ──→ embedding ──→ [ Block × N ] ──→ RMSNorm ──→ lm_head ──→ logits
//!
//!   Inside one Block:
//!     x ──→ RMSNorm ──→ attention (RoPE + KV cache + GQA) ──→ add back to x (residual)
//!       ──→ RMSNorm ──→ SwiGLU MLP                         ──→ add back to x (residual)

use crate::config::ModelConfig;
use crate::math::{dot, matvec, rmsnorm, silu, softmax};
use crate::weights::TensorStore;
use std::path::Path;

/// A plain linear layer: y = W·x (+ bias), with W shaped [out, in].
/// (Fields are pub(crate) so GPU backends in src/gpu/ can upload them.)
pub(crate) struct Linear {
    pub(crate) w: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>, // Llama has no biases; Qwen2 has them on q/k/v
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
}

impl Linear {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = matvec(&self.w, x, self.out_dim, self.in_dim);
        if let Some(b) = &self.bias {
            for (yi, bi) in y.iter_mut().zip(b) {
                *yi += bi;
            }
        }
        y
    }
}

/// One transformer block's weights — field names match the safetensors tensor names.
pub(crate) struct Block {
    pub(crate) input_layernorm: Vec<f32>, // norm before attention
    pub(crate) q_proj: Linear,
    pub(crate) k_proj: Linear,
    pub(crate) v_proj: Linear,
    pub(crate) o_proj: Linear,
    /// qwen3: per-head RMSNorm on q and k, applied before RoPE. None elsewhere.
    pub(crate) q_norm: Option<Vec<f32>>,
    pub(crate) k_norm: Option<Vec<f32>>,
    pub(crate) post_attention_layernorm: Vec<f32>, // norm before the MLP
    pub(crate) gate_proj: Linear,
    pub(crate) up_proj: Linear,
    pub(crate) down_proj: Linear,
}

impl Block {
    fn n_params(&self) -> usize {
        let lin = |l: &Linear| l.w.len() + l.bias.as_ref().map_or(0, Vec::len);
        self.input_layernorm.len()
            + self.post_attention_layernorm.len()
            + self.q_norm.as_ref().map_or(0, Vec::len)
            + self.k_norm.as_ref().map_or(0, Vec::len)
            + [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj, &self.gate_proj, &self.up_proj, &self.down_proj]
                .into_iter()
                .map(lin)
                .sum::<usize>()
    }
}

pub struct Model {
    pub cfg: ModelConfig,
    pub(crate) embed_tokens: Vec<f32>, // [vocab × hidden] token id → vector lookup table
    pub(crate) blocks: Vec<Block>,
    pub(crate) norm: Vec<f32>,   // final RMSNorm before the logits
    pub(crate) lm_head: Linear,  // [vocab × hidden] hidden state → score per vocab entry
    pub n_params: usize,
    /// Per-head width, DERIVED from q_proj's actual shape — qwen3 violates the
    /// hidden/n_heads identity (128 vs 64 on the 0.6B), so cfg.head_dim() is
    /// wrong there and every consumer reads these instead.
    pub head_dim: usize,
    pub kv_dim: usize,
}

/// Attention's memory: K and V for every past token, so they are never recomputed.
/// This is why generating token 500 costs about the same as token 1.
pub struct KvCache {
    k: Vec<Vec<f32>>, // per layer: [max_seq × kv_dim], ordered by position
    v: Vec<Vec<f32>>,
    kv_dim: usize,
}

impl KvCache {
    pub fn new(cfg: &ModelConfig, max_seq: usize, kv_dim: usize) -> Self {
        Self {
            k: vec![vec![0.0; max_seq * kv_dim]; cfg.num_hidden_layers],
            v: vec![vec![0.0; max_seq * kv_dim]; cfg.num_hidden_layers],
            kv_dim,
        }
    }
}

impl Model {
    /// Wire the loaded tensors (name → values) into the model structure.
    pub fn load(dir: &Path, cfg: ModelConfig) -> crate::Result<Self> {
        Self::from_store(cfg, &mut crate::weights::load(dir)?)
    }

    /// The tail `load` always contained, behind the TensorStore seam so any
    /// checkpoint source that can hand over f32 tensors under HF names enters
    /// with the same numerics (safetensors map, GGUF dequant — spec §11: the
    /// graph must not know where tensors live).
    pub fn from_store(cfg: ModelConfig, t: &mut dyn TensorStore) -> crate::Result<Self> {
        // The enums resolve ONCE, before the layer loop — the forward pass
        // below IS the SwiGLU/RMSNorm-pre implementation, and these matches
        // stop compiling the day a second variant exists.
        match cfg.activation()? {
            crate::config::Activation::SwiGLU => {}
        }
        match cfg.norm_type() {
            crate::config::NormType::RmsNormPre => {}
        }
        let h = cfg.hidden_size;
        // Head width comes from the checkpoint itself: qwen3's q_proj is
        // [n_heads*128, hidden] with hidden/n_heads = 64 — trusting the config
        // identity would slice every head wrong.
        let head_dim = t
            .numel("model.layers.0.self_attn.q_proj.weight")
            .map(|n| n / h / cfg.num_attention_heads)
            .unwrap_or_else(|| cfg.head_dim());
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv = cfg.num_key_value_heads * head_dim;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            blocks.push(Block {
                input_layernorm: take(t, &format!("{p}.input_layernorm.weight"))?,
                q_proj: linear(t, &format!("{p}.self_attn.q_proj"), h, q_dim)?,
                k_proj: linear(t, &format!("{p}.self_attn.k_proj"), h, kv)?,
                v_proj: linear(t, &format!("{p}.self_attn.v_proj"), h, kv)?,
                o_proj: linear(t, &format!("{p}.self_attn.o_proj"), q_dim, h)?,
                q_norm: take_opt(t, &format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: take_opt(t, &format!("{p}.self_attn.k_norm.weight"))?,
                post_attention_layernorm: take(t, &format!("{p}.post_attention_layernorm.weight"))?,
                gate_proj: linear(t, &format!("{p}.mlp.gate_proj"), h, cfg.intermediate_size)?,
                up_proj: linear(t, &format!("{p}.mlp.up_proj"), h, cfg.intermediate_size)?,
                down_proj: linear(t, &format!("{p}.mlp.down_proj"), cfg.intermediate_size, h)?,
            });
        }
        let embed_tokens = take(t, "model.embed_tokens.weight")?;
        let norm = take(t, "model.norm.weight")?;
        // Small models often tie weights: no separate lm_head, the embedding table is reused.
        let lm_head_w = match take_opt(t, "lm_head.weight")? {
            Some(w) => w,
            None => embed_tokens.clone(),
        };
        let lm_head = Linear { w: lm_head_w, bias: None, in_dim: h, out_dim: cfg.vocab_size };

        let n_params = embed_tokens.len()
            + norm.len()
            + lm_head.w.len()
            + blocks.iter().map(Block::n_params).sum::<usize>();
        Ok(Self { cfg, embed_tokens, blocks, norm, lm_head, n_params, head_dim, kv_dim: kv })
    }

    /// Process one token at position `pos` → logits (raw scores over the whole vocabulary).
    pub fn forward(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let cfg = &self.cfg;
        let (h, hd) = (cfg.hidden_size, self.head_dim);

        // 1. Embedding lookup: token id → hidden_size vector.
        let tok = token as usize;
        let mut x = self.embed_tokens[tok * h..(tok + 1) * h].to_vec();

        // 2. Run the blocks — x accumulates refinements through the residual stream.
        for (layer, blk) in self.blocks.iter().enumerate() {
            // First half: attention (this token looks back at everything before it).
            let xn = rmsnorm(&x, &blk.input_layernorm, cfg.rms_norm_eps);
            let mut q = blk.q_proj.forward(&xn);
            let mut k = blk.k_proj.forward(&xn);
            let v = blk.v_proj.forward(&xn);

            // qwen3 normalizes every head of q and k before RoPE.
            if let (Some(qn), Some(kn)) = (&blk.q_norm, &blk.k_norm) {
                for head in q.chunks_exact_mut(hd) {
                    let normed = rmsnorm(head, qn, cfg.rms_norm_eps);
                    head.copy_from_slice(&normed);
                }
                for head in k.chunks_exact_mut(hd) {
                    let normed = rmsnorm(head, kn, cfg.rms_norm_eps);
                    head.copy_from_slice(&normed);
                }
            }

            // RoPE encodes position by rotating q,k — no separate position embedding needed.
            rope(&mut q, pos, hd, cfg.rope_theta);
            rope(&mut k, pos, hd, cfg.rope_theta);

            // Append this token's k,v to the cache after the earlier positions.
            let kvd = cache.kv_dim;
            cache.k[layer][pos * kvd..(pos + 1) * kvd].copy_from_slice(&k);
            cache.v[layer][pos * kvd..(pos + 1) * kvd].copy_from_slice(&v);

            let att = attention(&q, &cache.k[layer], &cache.v[layer], pos, cfg, hd, cache.kv_dim);
            let att = blk.o_proj.forward(&att);
            for i in 0..h {
                x[i] += att[i]; // residual connection
            }

            // Second half: SwiGLU MLP (per-token computation on what attention gathered).
            let xn = rmsnorm(&x, &blk.post_attention_layernorm, cfg.rms_norm_eps);
            let gate = blk.gate_proj.forward(&xn);
            let up = blk.up_proj.forward(&xn);
            // silu(gate) acts as a per-dimension gate over the up projection.
            let inner: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            let out = blk.down_proj.forward(&inner);
            for i in 0..h {
                x[i] += out[i]; // residual connection
            }
        }

        // 3. Final norm, then project to a score for every token in the vocabulary.
        let xn = rmsnorm(&x, &self.norm, cfg.rms_norm_eps);
        self.lm_head.forward(&xn)
    }
}

/// Rotary Position Embedding (RoPE): pairs dimension i with i+head_dim/2 and rotates
/// each pair by angle pos·theta^(-2i/head_dim). Early pairs spin fast (sensitive to
/// nearby positions), later pairs spin slowly (capture long-range distance).
/// Key property: attention scores end up depending only on the *distance* between tokens.
fn rope(x: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;
    for head in x.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            let (sin, cos) = (pos as f32 * freq).sin_cos();
            let (a, b) = (head[i], head[i + half]);
            head[i] = a * cos - b * sin;
            head[i + half] = a * sin + b * cos;
        }
    }
}

/// Scaled dot-product attention for one query token against every cached position (0..=pos).
/// The causal mask comes for free: the cache only contains the past.
///
/// Supports GQA (Grouped-Query Attention): several query heads share one K/V head to
/// shrink the KV cache — e.g. SmolLM2 has 9 query heads but only 3 K/V heads.
fn attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    pos: usize,
    cfg: &ModelConfig,
    hd: usize,
    kvd: usize,
) -> Vec<f32> {
    let group = cfg.num_attention_heads / cfg.num_key_value_heads; // query heads per kv head
    let scale = 1.0 / (hd as f32).sqrt(); // keeps dot products from growing with dimension

    let mut out = vec![0.0; cfg.num_attention_heads * hd];
    for (h, o) in out.chunks_exact_mut(hd).enumerate() {
        let qh = &q[h * hd..(h + 1) * hd];
        let kv_off = (h / group) * hd; // where this query head's K/V head lives in each cache row

        // Attention scores against every past position, softmaxed into weights.
        let mut scores: Vec<f32> = (0..=pos)
            .map(|t| dot(qh, &k_cache[t * kvd + kv_off..t * kvd + kv_off + hd]) * scale)
            .collect();
        softmax(&mut scores);

        // This head's output: the weighted average of every position's v vector.
        for (t, s) in scores.iter().enumerate() {
            let vt = &v_cache[t * kvd + kv_off..t * kvd + kv_off + hd];
            for i in 0..hd {
                o[i] += s * vt[i];
            }
        }
    }
    out
}

// ---------- weight-loading helpers ----------

fn take(t: &mut dyn TensorStore, name: &str) -> crate::Result<Vec<f32>> {
    t.take_f32(name)
}

fn take_opt(t: &mut dyn TensorStore, name: &str) -> crate::Result<Option<Vec<f32>>> {
    match t.has(name) {
        true => Ok(Some(t.take_f32(name)?)),
        false => Ok(None),
    }
}

fn linear(t: &mut dyn TensorStore, name: &str, in_dim: usize, out_dim: usize) -> crate::Result<Linear> {
    let w = take(t, &format!("{name}.weight"))?;
    if w.len() != in_dim * out_dim {
        return Err(format!(
            "{name}.weight has {} values but the config implies {out_dim}×{in_dim}",
            w.len()
        )
        .into());
    }
    let bias = take_opt(t, &format!("{name}.bias"))?;
    Ok(Linear { w, bias, in_dim, out_dim })
}
