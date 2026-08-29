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
/// With a `draft` engine and temperature 0, decoding is speculative — see `speculative`.
pub fn generate(
    engine: &dyn Engine,
    draft: Option<&dyn Engine>,
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

    if let Some(d) = draft {
        if opt.temperature == 0.0 {
            return speculative(engine, d, tokenizer, opt, &ids, on_text);
        }
        eprintln!("note: --draft accelerates greedy decoding (-t 0) only; sampling uses the standard loop");
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

/// Draft block bounds. Bigger blocks amortize the target's verification better but
/// waste more draft work when a proposal is rejected early, so the block size adapts:
/// grow on a fully accepted block, shrink when nothing was accepted. The upper bound
/// keeps the verify batch (block + 1) within the Metal logits buffer (SPEC_MAX = 8).
const SPEC_MAX_DRAFT: usize = 7;
const SPEC_START_DRAFT: usize = 4;

/// Speculative decoding, greedy and exact: the draft proposes up to SPEC_DRAFT tokens,
/// the target verifies the whole block in ONE batched forward, and the accepted prefix
/// plus the target's own next token are emitted. Because acceptance compares argmax
/// against argmax, the output is token-identical to running the target alone at
/// temperature 0 — the draft only changes speed, never content.
///
/// The KV caches make this cheap to unwind: both backends write K,V by explicit
/// position, so a rejected token's cache rows are simply overwritten on the next
/// round — no copying, no truncation.
fn speculative(
    target: &dyn Engine,
    draft: &dyn Engine,
    tokenizer: &Tokenizer,
    opt: &GenOptions,
    ids: &[u32],
    mut on_text: impl FnMut(&str),
) -> crate::Result<GenOutput> {
    let cfg = target.config();
    let max_seq = (ids.len() + opt.max_tokens)
        .min(cfg.max_position_embeddings)
        .min(draft.config().max_position_embeddings);
    if ids.len() >= max_seq {
        return Err(format!("prompt ({} tokens) exceeds the model's context window", ids.len()).into());
    }
    let mut st = target.session(max_seq)?;
    let mut sd = draft.session(max_seq)?;

    // Both models prefill the prompt — each keeps its own KV cache.
    let t = Instant::now();
    let mut logits = st.prefill(ids)?;
    sd.prefill(ids)?;
    let prefill_secs = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut seq = ids.to_vec(); // prompt + everything generated so far
    let mut emitted = String::new();
    let mut draft_pos = ids.len(); // draft's cache is valid for seq[..draft_pos]
    let (mut proposed, mut accepted) = (0usize, 0usize);
    let mut gamma = SPEC_START_DRAFT;

    while seq.len() < max_seq {
        // The target's next token is already decided by its current logits.
        let next = argmax(&logits);
        if cfg.is_eos(next) {
            break;
        }
        seq.push(next);
        if seq.len() >= max_seq {
            break;
        }

        // Catch the draft up on every accepted token it hasn't seen (rewrites any
        // cache rows a rejected proposal left behind), ending with `next` — whose
        // draft logits propose the first block token. One batched pass.
        let caught = sd.forward_batch(&seq[draft_pos..], draft_pos)?;
        let mut dlogits = caught.into_iter().last().expect("catch-up is never empty");
        draft_pos = seq.len();

        // Draft a block (greedy), staying inside the cache.
        let room = max_seq - seq.len();
        let block = gamma.min(room);
        let mut proposal: Vec<u32> = Vec::with_capacity(block);
        while proposal.len() < block {
            let d = argmax(&dlogits);
            if cfg.is_eos(d) {
                break; // let the target's own logits decide whether to stop
            }
            proposal.push(d);
            if proposal.len() < block {
                dlogits = sd.forward(d, draft_pos)?;
                draft_pos += 1;
            }
        }
        proposed += proposal.len();

        // Verify: one batched target pass over `next` + the proposal gives the
        // target's argmax after every prefix — accept while it agrees.
        let batch: Vec<u32> = std::iter::once(next).chain(proposal.iter().copied()).collect();
        let all = st.forward_batch(&batch, seq.len() - 1)?;
        let mut k = 0;
        while k < proposal.len() && seq.len() < max_seq && argmax(&all[k]) == proposal[k] {
            seq.push(proposal[k]);
            k += 1;
        }
        accepted += k;
        if !proposal.is_empty() {
            if k == proposal.len() {
                gamma = (gamma + 1).min(SPEC_MAX_DRAFT);
            } else if k == 0 {
                gamma = (gamma - 1).max(1);
            }
        }
        // Logits after the last accepted token drive the next round's `next` —
        // the correction on a miss, a free extra token on a full accept.
        logits = all.into_iter().nth(k).expect("k < batch len");
        // Anything the draft ran past the accepted prefix is invalid — rewind.
        draft_pos = draft_pos.min(seq.len());

        let text = tokenizer.decode(&seq[ids.len()..], true)?;
        if text.starts_with(&emitted) && !text.ends_with('\u{FFFD}') {
            on_text(&text[emitted.len()..]);
            emitted = text;
        }
    }
    let decode_secs = t.elapsed().as_secs_f64();

    let generated = seq.len() - ids.len();
    if proposed > 0 {
        eprintln!(
            "  speculative: {accepted}/{proposed} draft tokens accepted ({:.0}%)",
            100.0 * accepted as f64 / proposed as f64
        );
    }
    let text = tokenizer.decode(&seq[ids.len()..], true)?;
    if text.starts_with(&emitted) {
        on_text(&text[emitted.len()..]);
    }

    Ok(GenOutput {
        text,
        prompt_tokens: ids.len(),
        generated_tokens: generated,
        prefill_secs,
        decode_secs,
    })
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}
