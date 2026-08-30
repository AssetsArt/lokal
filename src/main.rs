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
      --context-window <N> lowmem only: sliding attention window in tokens [2048]
      --attention-sink <N> lowmem only: pinned initial tokens, 0 disables [4]
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
    let cfg = config::ModelConfig::load(&dir.join("config.json"))?;
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))?;

    // Wrap the model in the selected backend (cpu uses it directly, metal uploads
    // to the GPU). lowmem builds itself from the model directory — it must never
    // go through Model::load's full-RAM materialization (see src/lowmem/).
    let engine = if args.backend == "lowmem" {
        engine::create_lowmem(&dir, cfg.clone(), &args.lowmem)?
    } else if args.lowmem.any_set() {
        return Err(
            "--memory-budget, --context-window, and --attention-sink apply to -b lowmem only"
                .into(),
        );
    } else {
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
        engine::create(&args.backend, model, &dir)?
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
            let dengine = engine::create(dbackend, dmodel, &ddir)?;
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
