//! Continuous batching for serve mode.
//!
//! One scheduler thread owns a Batcher (a pool of KV slots plus a batched decode
//! step). Requests queue on a channel; whenever a slot is free the next request is
//! admitted (its prompt prefills into the slot — on the hybrid backend that runs on
//! the ANE), and every loop iteration advances ALL active requests by one token in
//! a single GPU submission. That is what lifts aggregate throughput past the
//! single-stream ceiling: the weights are read once per step regardless of how many
//! requests are generating.
//!
//! Requests beyond the slot count simply wait in the channel (FIFO), which replaces
//! the semaphore the per-request path uses.

use crate::engine::{BatchRow, Engine};
use crate::generate::{chatml, GenOptions, GenOutput};
use crate::sampler::Sampler;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Cap on the pooled per-slot sequence length. Slots are lazily paged so a large
/// value costs virtual address space, not RAM — the cap just keeps one request from
/// claiming a model's entire (possibly 32k+) context window of scratch.
const POOL_SEQ_CAP: usize = 8192;

pub struct Job {
    pub opt: GenOptions,
    pub resp: tokio::sync::oneshot::Sender<crate::Result<GenOutput>>,
}

/// Spawn the scheduler thread. Returns None when the backend has no batcher
/// (e.g. cpu) — the server then keeps its per-request path.
pub fn spawn(
    engine: Arc<dyn Engine>,
    tokenizer: Arc<Tokenizer>,
    n_slots: usize,
) -> Option<mpsc::Sender<Job>> {
    let max_seq = engine.config().max_position_embeddings.min(POOL_SEQ_CAP);
    engine.batcher(n_slots, max_seq)?; // probe support before committing a thread
    let (tx, rx) = mpsc::channel::<Job>();
    std::thread::spawn(move || run(engine, tokenizer, rx, n_slots, max_seq));
    Some(tx)
}

struct Active {
    slot: usize,
    seq_len: usize, // positions committed to the KV cache
    max_seq: usize, // this request's own budget (prompt + max_tokens, ≤ pool)
    prompt_tokens: usize,
    generated: Vec<u32>,
    logits: Vec<f32>, // pending logits — the source of the next token
    sampler: Sampler,
    prefill_secs: f64,
    decode_start: Instant,
    resp: tokio::sync::oneshot::Sender<crate::Result<GenOutput>>,
}

fn run(
    engine: Arc<dyn Engine>,
    tokenizer: Arc<Tokenizer>,
    rx: mpsc::Receiver<Job>,
    n_slots: usize,
    pool_seq: usize,
) {
    let mut batcher = engine.batcher(n_slots, pool_seq).expect("probed in spawn");
    let cfg = engine.config();
    let mut free: Vec<usize> = (0..n_slots).rev().collect();
    let mut active: Vec<Active> = Vec::new();

    loop {
        // Admit while there are free slots: block when idle, otherwise only take
        // what is already queued.
        while !free.is_empty() {
            let job = if active.is_empty() {
                match rx.recv() {
                    Ok(j) => j,
                    Err(_) => return, // server shut down
                }
            } else {
                match rx.try_recv() {
                    Ok(j) => j,
                    Err(_) => break,
                }
            };
            let slot = free.pop().expect("checked non-empty");
            match admit(batcher.as_mut(), &tokenizer, cfg, slot, pool_seq, job) {
                Some(a) => active.push(a),
                None => free.push(slot), // request failed or answered instantly
            }
        }

        // Sample each active request's next token from its pending logits; requests
        // that stop (EOS / budget) finish now, the rest form the batch. Semantics
        // mirror the single-stream loop: EOS is not emitted, and the final budgeted
        // token is emitted but not fed back.
        let mut rows = Vec::with_capacity(active.len());
        let mut i = 0;
        while i < active.len() {
            let a = &mut active[i];
            let next = a.sampler.sample(&mut a.logits);
            let mut stopped = cfg.is_eos(next);
            if !stopped {
                a.generated.push(next);
                stopped = a.seq_len + 1 >= a.max_seq;
            }
            if stopped {
                let a = active.swap_remove(i);
                free.push(a.slot);
                finish(&tokenizer, a);
                continue;
            }
            rows.push(BatchRow { token: next, pos: a.seq_len, slot: a.slot });
            i += 1;
        }
        if rows.is_empty() {
            continue;
        }

        // One submission advances every request (rows and active are index-aligned).
        match batcher.decode_step(&rows) {
            Ok(all) => {
                for (a, logits) in active.iter_mut().zip(all) {
                    a.seq_len += 1;
                    a.logits = logits;
                }
            }
            Err(e) => {
                // A failed step is fatal for the requests in flight — report and reset.
                let msg = e.to_string();
                for a in active.drain(..) {
                    free.push(a.slot);
                    let _ = a.resp.send(Err(msg.clone().into()));
                }
            }
        }
    }
}

/// Encode + prefill one request into `slot`. Returns None if the request never
/// becomes active (bad input, or the prompt alone filled the budget).
fn admit(
    batcher: &mut dyn crate::engine::Batcher,
    tokenizer: &Tokenizer,
    cfg: &crate::config::ModelConfig,
    slot: usize,
    pool_seq: usize,
    job: Job,
) -> Option<Active> {
    let opt = &job.opt;
    let full_prompt = if opt.chat { chatml(&opt.prompt) } else { opt.prompt.clone() };
    let ids = match tokenizer.encode(full_prompt.as_str(), true) {
        Ok(e) => e.get_ids().to_vec(),
        Err(e) => {
            let _ = job.resp.send(Err(e.to_string().into()));
            return None;
        }
    };
    let max_seq = (ids.len() + opt.max_tokens).min(pool_seq).min(cfg.max_position_embeddings);
    if ids.is_empty() || ids.len() >= max_seq {
        let _ = job.resp.send(Err(format!(
            "prompt ({} tokens) is empty or exceeds the {max_seq}-token budget",
            ids.len()
        )
        .into()));
        return None;
    }

    let t = Instant::now();
    let logits = match batcher.prefill(slot, &ids) {
        Ok(l) => l,
        Err(e) => {
            let _ = job.resp.send(Err(e));
            return None;
        }
    };
    let seed = opt
        .seed
        .unwrap_or_else(|| std::time::UNIX_EPOCH.elapsed().unwrap_or_default().as_nanos() as u64);
    Some(Active {
        slot,
        seq_len: ids.len(),
        max_seq,
        prompt_tokens: ids.len(),
        generated: Vec::new(),
        logits,
        sampler: Sampler::new(opt.temperature, opt.top_p, seed),
        prefill_secs: t.elapsed().as_secs_f64(),
        decode_start: Instant::now(),
        resp: job.resp,
    })
}

fn finish(tokenizer: &Tokenizer, a: Active) {
    let out = tokenizer
        .decode(&a.generated, true)
        .map(|text| GenOutput {
            text,
            prompt_tokens: a.prompt_tokens,
            generated_tokens: a.generated.len(),
            prefill_secs: a.prefill_secs,
            decode_secs: a.decode_start.elapsed().as_secs_f64(),
        })
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() });
    let _ = a.resp.send(out);
}
