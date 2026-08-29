//! The generation loop shared by CLI and HTTP server modes.
//!
//! Inference has two phases:
//!   1. prefill — feed the prompt to fill the KV cache (nothing is generated yet)
//!   2. decode  — sample the next token from the logits → feed it back → repeat
//!      until EOS or the token budget runs out

use crate::engine::Engine;
use crate::sampler::Sampler;
use serde::Deserialize;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Generation options — deserializes directly from an HTTP request's JSON body.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GenOptions {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32, // 0 = greedy (fully reproducible, great for debugging)
    pub top_p: f32,
    pub seed: Option<u64>, // None = seed from the clock
    pub chat: bool,        // wrap the prompt in a chat template (for -Instruct models)
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 200,
            temperature: 0.7,
            top_p: 0.9,
            seed: None,
            chat: false,
        }
    }
}

pub struct GenOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

/// ChatML template — Instruct models (SmolLM2-Instruct, Qwen2.5-Instruct) are fine-tuned
/// to answer text in exactly this shape; a bare prompt won't get a conversational reply.
fn chatml(user_msg: &str) -> String {
    format!("<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n")
}

/// Run one prefill + decode pass. `on_text` fires whenever new text is ready to show
/// (the CLI streams it; the server ignores it and reads GenOutput.text at the end).
pub fn generate(
    engine: &dyn Engine,
    tokenizer: &Tokenizer,
    opt: &GenOptions,
    mut on_text: impl FnMut(&str),
) -> crate::Result<GenOutput> {
    let cfg = engine.config();

    // Text → token ids.
    let full_prompt = if opt.chat { chatml(&opt.prompt) } else { opt.prompt.clone() };
    let ids = tokenizer.encode(full_prompt.as_str(), true)?.get_ids().to_vec();
    if ids.is_empty() {
        return Err("empty prompt".into());
    }

    // Size the KV cache to what will actually be used (prompt + budget), not full context.
    let max_seq = (ids.len() + opt.max_tokens).min(cfg.max_position_embeddings);
    if ids.len() >= max_seq {
        return Err(format!("prompt ({} tokens) exceeds the model's context window", ids.len()).into());
    }
    let mut session = engine.session(max_seq)?;

    // Prefill: process the whole prompt to fill the KV cache. Each backend decides how:
    // the CPU walks token by token (the trait's default), Metal runs matrix-matrix
    // chunks, and the ANE backend sends the prompt through Core ML in one shot.
    let t = Instant::now();
    let mut logits = session.prefill(&ids)?;
    let prefill_secs = t.elapsed().as_secs_f64();

    // Decode: sample a token → feed it back into the model → repeat.
    let seed = opt
        .seed
        .unwrap_or_else(|| std::time::UNIX_EPOCH.elapsed().unwrap_or_default().as_nanos() as u64);
    let mut sampler = Sampler::new(opt.temperature, opt.top_p, seed);
    let t = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    let mut emitted = String::new();
    while ids.len() + generated.len() < max_seq {
        let next = sampler.sample(&mut logits);
        if cfg.is_eos(next) {
            break;
        }
        generated.push(next);

        // Decode the full sequence and emit only the newly grown suffix. A single UTF-8
        // character (Thai, emoji, ...) can span several tokens; decoding tokens one at a
        // time would print garbage at those boundaries.
        let text = tokenizer.decode(&generated, true)?;
        if text.starts_with(&emitted) && !text.ends_with('\u{FFFD}') {
            on_text(&text[emitted.len()..]);
            emitted = text;
        }

        logits = session.forward(next, ids.len() + generated.len() - 1)?;
    }
    let decode_secs = t.elapsed().as_secs_f64();

    // Flush any characters still held back by the incomplete-UTF-8 check.
    let text = tokenizer.decode(&generated, true)?;
    if text.starts_with(&emitted) {
        on_text(&text[emitted.len()..]);
    }

    Ok(GenOutput {
        text,
        prompt_tokens: ids.len(),
        generated_tokens: generated.len(),
        prefill_secs,
        decode_secs,
    })
}
