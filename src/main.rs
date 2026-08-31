//! lokal — a local-first LLM inference engine written from scratch in Rust.
//!
//! The whole pipeline at a glance:
//!
//!   prompt ──tokenizer──→ token ids ──[model forward, token by token]──→ logits
//!          ──sampler──→ next token ──fed back into forward──→ ... ──decode──→ text
//!
//! Module map: hub.rs (downloads) → config.rs (hyperparameters) → weights.rs (tensor loading)
//! → model.rs (★ the forward pass) → generate.rs (prefill/decode loop) → sampler.rs
//! → math.rs (CPU kernels) → engine.rs (backend abstraction) → gpu/ + ane.rs (accelerators)
//! → server.rs (HTTP mode). See DESIGN.md for the full architecture.

#[cfg(target_os = "macos")]
mod ane;
mod batch;
mod config;
mod engine;
mod generate;
mod gpu;
mod hub;
#[cfg(target_os = "macos")]
mod lowmem;
mod math;
mod model;
mod sampler;
mod server;
mod weights;

use generate::GenOptions;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

struct Args {
    model: String,
    draft: Option<String>,
    backend: String,
    serve: bool,
    path_only: bool,
    graphs_path: bool,
    port: u16,
    max_concurrent: usize,
    lowmem: engine::LowMemOpts,
    opt: GenOptions,
}

const USAGE: &str = "lokal — run LLMs on your own machine, fast and simple

Usage:
  lokal [options]           one-shot generation
  lokal serve [options]     HTTP server (POST /generate)
  lokal path [-m <model>]   download if needed, then print the model's local directory
  lokal path --graphs       …print the model's Core ML graph directory instead (creating it)

Options:
  -m, --model <repo|dir>   Hugging Face repo or local directory [HuggingFaceTB/SmolLM2-135M]
      --draft <repo|dir>   smaller same-tokenizer model for speculative decoding (greedy only)
  -b, --backend <name>     cpu, metal (Apple GPU), hybrid (Neural Engine + GPU together),
                           lowmem (disk-backed paged inference — optimized for models larger
                           than available RAM; uses a bounded attention window and may reduce
                           long-context quality) [cpu]
      --memory-budget <MB> lowmem only: total working-set budget [4096]
      --context-window <N> sliding attention window in tokens — opt-in on metal/hybrid
                           (KV memory goes O(window); the model loses sight of tokens
                           beyond the window, trading long-range recall for flat cost),
                           always on for lowmem [lowmem: 2048]
      --attention-sink <N> with --context-window: pinned initial tokens, 0 disables [4]
  -p, --prompt <text>      prompt text [\"Once upon a time\"]
  -n, --max-tokens <N>     maximum number of tokens to generate [200]
  -t, --temperature <T>    0 = greedy (fully deterministic), higher = more adventurous [0.7]
      --top-p <P>          nucleus sampling threshold [0.9]
      --seed <N>           RNG seed for reproducible sampling
      --chat               wrap the prompt in a chat template (use with -Instruct models)
      --port <N>           port for serve mode [8080]
      --max-concurrent <N> serve mode: requests generating at once, the rest queue [4]
  -h, --help               show this help";

impl Args {
    fn parse() -> Result<Self> {
        let mut a = Self {
            model: "HuggingFaceTB/SmolLM2-135M".into(),
            draft: None,
            backend: "cpu".into(),
            serve: false,
            path_only: false,
            graphs_path: false,
            port: 8080,
            max_concurrent: 4,
            lowmem: engine::LowMemOpts::default(),
            opt: GenOptions { prompt: "Once upon a time".into(), ..Default::default() },
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut val = || it.next().ok_or(format!("{flag} requires a value"));
            match flag.as_str() {
                "serve" => a.serve = true,
                "path" => a.path_only = true,
                "--graphs" => a.graphs_path = true,
                "-m" | "--model" => a.model = val()?,
                "--draft" => a.draft = Some(val()?),
                "-b" | "--backend" => a.backend = val()?,
                "-p" | "--prompt" => a.opt.prompt = val()?,
                "-n" | "--max-tokens" => a.opt.max_tokens = val()?.parse()?,
                "-t" | "--temperature" => a.opt.temperature = val()?.parse()?,
                "--top-p" => a.opt.top_p = val()?.parse()?,
                "--seed" => a.opt.seed = Some(val()?.parse()?),
                "--chat" => a.opt.chat = true,
                "--port" => a.port = val()?.parse()?,
                "--max-concurrent" => a.max_concurrent = val()?.parse()?,
                "--memory-budget" => a.lowmem.memory_budget_mb = Some(val()?.parse()?),
                "--context-window" => a.lowmem.context_window = Some(val()?.parse()?),
                "--attention-sink" => a.lowmem.attention_sink = Some(val()?.parse()?),
                "-h" | "--help" => {
                    println!("{USAGE}");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option {other} (see --help)").into()),
            }
        }
        Ok(a)
    }
}

fn main() {
    // Print errors with Display rather than Debug so messages stay human-readable.
    if let Err(e) = run() {
        eprintln!("\nerror: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse()?;

    // Three ingredients: config, tokenizer, weights.
    if args.graphs_path && !args.path_only {
        return Err("--graphs only makes sense with `lokal path` (see --help)".into());
    }

    let dir = hub::resolve_model(&args.model)?;
    if args.path_only {
        if args.graphs_path {
            // The one place scripts (run.sh) resolve the lokal-owned graph
            // directory — same rule the ane backend applies at load time.
            let loc = hub::graph_location(&dir);
            std::fs::create_dir_all(&loc.dir)?;
            println!("{}", loc.dir.display());
        } else {
            println!("{}", dir.display());
        }
        return Ok(());
    }
    // --context-window/--attention-sink opt metal and hybrid into sliding-window
    // attention (lowmem keeps its own wiring); --memory-budget stays lowmem-only.
    let win: Option<(usize, usize)> = match (args.backend.as_str(), args.lowmem.context_window) {
        ("lowmem", _) | (_, None) => None,
        (_, Some(w)) => Some((w, args.lowmem.attention_sink.unwrap_or(4))),
    };
    if win.is_some() {
        if args.lowmem.memory_budget_mb.is_some() {
            return Err("--memory-budget applies to -b lowmem only".into());
        }
        if args.draft.is_some() {
            return Err(
                "--draft cannot be combined with --context-window — speculative verification \
                 under a window is unproven; drop one of the two"
                    .into(),
            );
        }
    }

    // A .gguf file is its own checkpoint format: config and tokenizer come out
    // of the file itself, weights are (possibly) quantized. Everything else
    // stays the config.json + tokenizer.json + safetensors directory layout.
    let is_gguf =
        dir.is_file() && dir.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    #[cfg(not(target_os = "macos"))]
    if is_gguf {
        return Err("GGUF checkpoints need the macOS build (Metal / lowmem)".into());
    }

    #[cfg(target_os = "macos")]
    let (cfg, tokenizer, engine) = if is_gguf {
        gguf_setup(&args, &dir, win)?
    } else {
        let cfg = config::ModelConfig::load(&dir.join("config.json"))?;
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))?;
        (cfg, tokenizer, None)
    };
    #[cfg(not(target_os = "macos"))]
    let (cfg, tokenizer, engine) = {
        let cfg = config::ModelConfig::load(&dir.join("config.json"))?;
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))?;
        (cfg, tokenizer, None)
    };

    // Wrap the model in the selected backend (cpu uses it directly, metal uploads
    // to the GPU). lowmem builds itself from the model directory — it must never
    // go through Model::load's full-RAM materialization (see src/lowmem/).
    let engine = match engine {
        Some(e) => e, // already built by the GGUF path
        None if args.backend == "lowmem" => {
            engine::create_lowmem(&dir, cfg.clone(), &args.lowmem)?
        }
        None if args.lowmem.memory_budget_mb.is_some() => {
            return Err("--memory-budget applies to -b lowmem only".into());
        }
        None if args.lowmem.attention_sink.is_some() && args.lowmem.context_window.is_none() => {
            return Err("--attention-sink needs --context-window".into());
        }
        None => {
            let t0 = Instant::now();
            let model = model::Model::load(&dir, cfg.clone())?;
            eprintln!(
                "{} | {} layers | hidden {} | {} q heads / {} kv | vocab {} | {:.1}M params (loaded in {:.1}s)",
                args.model,
                cfg.num_hidden_layers,
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.vocab_size,
                model.n_params as f64 / 1e6,
                t0.elapsed().as_secs_f64(),
            );
            engine::create(&args.backend, model, &dir, win)?
        }
    };
    eprintln!("backend: {}", engine.name());

    // Optional draft model for speculative decoding. It must share the target's
    // tokenizer (same model family); vocab size is the cheap proxy check. The draft
    // never uses the ane backend — its prompt share is small and the ANE graphs may
    // not exist for it.
    let draft = match &args.draft {
        None => None,
        Some(name) => {
            let ddir = hub::resolve_model(name)?;
            let dcfg = config::ModelConfig::load(&ddir.join("config.json"))?;
            if dcfg.vocab_size != cfg.vocab_size {
                return Err(format!(
                    "draft model {name} has vocab {} but the target has {} — they must share a tokenizer",
                    dcfg.vocab_size, cfg.vocab_size
                )
                .into());
            }
            let dmodel = model::Model::load(&ddir, dcfg)?;
            // A draft model is small: composite/paged backends hand it plain metal.
            let dbackend = match args.backend.as_str() {
                "hybrid" | "ane" | "lowmem" => "metal",
                b => b,
            };
            let dengine = engine::create(dbackend, dmodel, &ddir, None)?;
            eprintln!("draft: {name} ({})", dengine.name());
            Some(dengine)
        }
    };

    if args.serve {
        return server::serve(
            Arc::from(engine),
            draft.map(Arc::from),
            Arc::new(tokenizer),
            args.port,
            args.max_concurrent,
        );
    }

    // LOKAL_DEBUG_TOPK=N: prefill the prompt, print the final position's top-N
    // (token id, logit) pairs, and exit. The perturbation gates assert on these
    // directly — an argmax flip is never implied by a logits change
    // (protocol:gate-scripts), so window semantics are proved at logit level.
    if let Some(k) = std::env::var("LOKAL_DEBUG_TOPK").ok().and_then(|v| v.parse::<usize>().ok()) {
        let ids = tokenizer
            .encode(args.opt.prompt.as_str(), true)
            .map_err(|e| format!("tokenize: {e}"))?
            .get_ids()
            .to_vec();
        println!("ntok\t{}", ids.len()); // gates assert equal lengths across perturbations
        let mut session = engine.session(ids.len() + 1)?;
        let logits = session.prefill(&ids)?;
        let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (id, logit) in ranked.into_iter().take(k) {
            println!("topk\t{id}\t{logit:.6}");
        }
        return Ok(());
    }

    // CLI mode: echo the prompt, then stream generated text after it.
    if args.opt.chat {
        println!("{}", args.opt.prompt); // in chat mode, put question and answer on separate lines
    } else {
        print!("{}", args.opt.prompt);
    }
    std::io::stdout().flush()?;

    let out = generate::generate(engine.as_ref(), draft.as_deref(), &tokenizer, &args.opt, |piece| {
        print!("{piece}");
        let _ = std::io::stdout().flush();
    })?;
    println!();

    eprintln!(
        "\nprefill {} tokens in {:.2}s ({:.1} tok/s) | generated {} tokens in {:.2}s ({:.1} tok/s)",
        out.prompt_tokens,
        out.prefill_secs,
        out.prompt_tokens as f64 / out.prefill_secs.max(1e-9),
        out.generated_tokens,
        out.decode_secs,
        out.generated_tokens as f64 / out.decode_secs.max(1e-9),
    );
    Ok(())
}

/// Build (config, tokenizer, engine) from a GGUF checkpoint (revised D6):
/// cpu/metal dequantize everything to f32 when the EXPANDED weights fit RAM,
/// lowmem is the bounded path, hybrid cannot run GGUF at all.
#[cfg(target_os = "macos")]
fn gguf_setup(
    args: &Args,
    path: &std::path::Path,
    win: Option<(usize, usize)>,
) -> Result<(config::ModelConfig, tokenizers::Tokenizer, Option<Box<dyn engine::Engine>>)> {
    use lowmem::gguf;
    let g = gguf::GgufFile::open(path)?;
    // LOKAL_GGUF_INFO=1: dump the tensor table and exit — the cross-check gate
    // diffs this against `llama-cli --verbose` (parser-only, so it works even
    // on files whose tokenizer lokal refuses).
    if std::env::var_os("LOKAL_GGUF_INFO").is_some_and(|v| v == "1") {
        print!("{}", g.dump_tsv());
        println!("total\t{}", g.n_tensors());
        std::process::exit(0);
    }
    let (cfg, arch) = gguf::model_config(&g)?;
    let tokenizer = gguf::build_tokenizer(&g)?;
    eprintln!(
        "{} | {} ({}) | {} layers | hidden {} | {} q heads / {} kv | vocab {}",
        args.model,
        gguf::summary(&g),
        arch.arch,
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.vocab_size,
    );
    // qwen35 runs on -b lowmem. Two things are still refused BY NAME rather
    // than attempted, because both would otherwise produce plausible wrong
    // output instead of an error:
    //   * a checkpoint whose rope sections are not the text-broadcast layout
    //     the MRoPE equivalence was verified on (see Qwen35Layout::
    //     check_rope_sections — this engine has no sectioned rope kernel and
    //     would silently rotate a vision variant as if it were text);
    //   * every backend except lowmem, which is where the gated-deltanet block
    //     is wired. The metal backend builds its own per-layer tensors and has
    //     no linear-block path yet, so it would fail on a tensor name at best
    //     and mis-read the joint Q+gate projection at worst.
    if arch.arch == "qwen35" {
        let m = lowmem::gguf::qwen35_meta(&g)?;
        crate::gpu::metal::Qwen35Layout::check_rope_sections(&m)?;
        if args.backend != "lowmem" {
            return Err(format!(
                "qwen35 runs on -b lowmem only ({} trunk layers: {} full-attention + {} \
                 gated-deltanet{}); -b {} has no gated-deltanet path yet",
                m.trunk_layers,
                m.is_recurrent.iter().filter(|r| !**r).count(),
                m.is_recurrent.iter().filter(|r| **r).count(),
                if m.nextn_layers > 0 { ", +1 MTP block (skipped)" } else { "" },
                args.backend,
            )
            .into());
        }
        eprintln!(
            "qwen35: {} trunk layers — {} full-attention + {} gated-deltanet{}",
            m.trunk_layers,
            m.is_recurrent.iter().filter(|r| !**r).count(),
            m.is_recurrent.iter().filter(|r| **r).count(),
            if m.nextn_layers > 0 { ", +1 MTP block (skipped)" } else { "" },
        );
    }
    let engine = match args.backend.as_str() {
        "lowmem" => Some(engine::create_lowmem(path, cfg.clone(), &args.lowmem)?),
        "hybrid" | "ane" => {
            return Err("the hybrid backend cannot run a GGUF: its ANE prefill graphs are \
                 exported from safetensors (tools/export_prefill.py) — use -b metal, or the \
                 model's safetensors checkpoint"
                .into())
        }
        "metal" if std::env::var_os("LOKAL_GGUF_EXPAND").is_none() => {
            // Quant execution (D1): weights stay in their on-disk encoding and
            // dequantize on read — no f32 expansion, so the fits check is the
            // FILE size, which is what lets a 27B Q4 run on a 32 GB machine.
            // qwen3's per-head q/k norm is wired (metal-qwen3 lane), so the
            // arch needs no refusal here. LOKAL_GGUF_EXPAND=1 forces the old
            // expand-to-f32 path (the D3 identity gate compares the two).
            if args.lowmem.memory_budget_mb.is_some() {
                return Err("--memory-budget applies to -b lowmem only".into());
            }
            let engine = engine::create_metal_quant(path, cfg.clone(), win)?;
            Some(engine)
        }
        "cpu" | "metal" => {
            if args.lowmem.memory_budget_mb.is_some() {
                return Err("--memory-budget applies to -b lowmem only".into());
            }
            // The honest RAM cost is the EXPANDED f32 size, not the file size —
            // a 4 GB Q4 file is ~28 GB of f32. Checked before any allocation.
            let expanded = gguf::expanded_f32_bytes(&g);
            let ram = gguf::phys_ram_bytes();
            if expanded > ram {
                return Err(format!(
                    "this GGUF expands to {:.1} GB of f32 weights, but the machine has {:.1} GB \
                     of RAM — run it with -b lowmem (weights stay quantized and paged there)",
                    expanded as f64 / 1e9,
                    ram as f64 / 1e9
                )
                .into());
            }
            let quant = gguf::quant_bytes(&g);
            if expanded >= 4 * quant {
                eprintln!(
                    "note: this {:.1} GB file becomes {:.1} GB of f32 in RAM on -b {} — \
                     -b lowmem keeps the weights quantized instead",
                    quant as f64 / 1e9,
                    expanded as f64 / 1e9,
                    args.backend
                );
            }
            let t0 = Instant::now();
            let model = model::Model::from_tensors(cfg.clone(), gguf::load_f32(&g)?)?;
            eprintln!(
                "dequantized {:.1}M params to f32 in {:.1}s",
                model.n_params as f64 / 1e6,
                t0.elapsed().as_secs_f64()
            );
            let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
            Some(engine::create(&args.backend, model, parent, win)?)
        }
        other => {
            return Err(format!(
                "unknown backend \"{other}\" for a GGUF — available: cpu, metal, lowmem"
            )
            .into())
        }
    };
    Ok((cfg, tokenizer, engine))
}
