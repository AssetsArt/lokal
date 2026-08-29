//! HTTP server mode: serves the model over POST /generate (JSON in → JSON out) via hyper.
//!
//! The minimal hyper 1.x setup: tokio accepts TCP connections → hyper speaks HTTP
//! → service_fn hands requests to our handler.
//!
//! Design note worth stealing: model weights are read-only, so the engine is shared
//! as an Arc across threads, while all mutable generation state (KV cache, sampler)
//! is created per request — concurrent requests need no locks at all.

use crate::engine::Engine;
use crate::generate::{generate, GenOptions};
use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::json;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Semaphore;

pub fn serve(
    engine: Arc<dyn Engine>,
    draft: Option<Arc<dyn Engine>>,
    tokenizer: Arc<Tokenizer>,
    port: u16,
    max_concurrent: usize,
) -> crate::Result<()> {
    // main() is plain sync code — build the tokio runtime here at the async boundary.
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
        // Admission control: the GPU's aggregate decode throughput saturates at a few
        // concurrent generations (measured ~4 on an M1 Pro) — beyond that, extra
        // concurrency only splits the same tokens/sec across more requests and holds
        // more KV caches in RAM. The semaphore is a FIFO queue: max_concurrent
        // requests generate, the rest wait their turn.
        let sem = Arc::new(Semaphore::new(max_concurrent.max(1)));
        eprintln!("listening on http://127.0.0.1:{port} ({max_concurrent} concurrent, rest queue) — try:");
        eprintln!("  curl http://127.0.0.1:{port}/generate -d '{{\"prompt\": \"Once upon a time\"}}'");
        loop {
            let (stream, _) = listener.accept().await?;
            let (engine, draft, tokenizer, sem) =
                (engine.clone(), draft.clone(), tokenizer.clone(), sem.clone());
            // One connection = one task, so a slow connection never blocks the others.
            tokio::spawn(async move {
                let service = service_fn(move |req| {
                    handle(req, engine.clone(), draft.clone(), tokenizer.clone(), sem.clone())
                });
                let conn = http1::Builder::new().serve_connection(TokioIo::new(stream), service);
                if let Err(e) = conn.await {
                    eprintln!("connection error: {e}");
                }
            });
        }
    })
}

async fn handle(
    req: Request<Incoming>,
    engine: Arc<dyn Engine>,
    draft: Option<Arc<dyn Engine>>,
    tokenizer: Arc<Tokenizer>,
    sem: Arc<Semaphore>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    if (req.method(), req.uri().path()) != (&Method::POST, "/generate") {
        return Ok(reply(
            StatusCode::NOT_FOUND,
            json!({"error": "single endpoint: POST /generate — JSON body with prompt (required), max_tokens, temperature, top_p, seed, chat"}),
        ));
    }

    let body = req.collect().await?.to_bytes();
    let opt: GenOptions = match serde_json::from_slice(&body) {
        Ok(o) => o,
        Err(e) => {
            return Ok(reply(StatusCode::BAD_REQUEST, json!({"error": format!("invalid JSON: {e}")})))
        }
    };

    // Wait for a generation slot (FIFO). The permit moves into the blocking task so
    // the slot frees exactly when generation finishes.
    let permit = sem.acquire_owned().await.expect("semaphore is never closed");

    // Heavy compute must leave the async threads, or it stalls tokio's event loop.
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        generate(engine.as_ref(), draft.as_deref(), &tokenizer, &opt, |_| {})
    })
    .await;

    Ok(match result {
        Ok(Ok(out)) => reply(
            StatusCode::OK,
            json!({
                "text": out.text,
                "prompt_tokens": out.prompt_tokens,
                "generated_tokens": out.generated_tokens,
                "tok_per_sec": out.generated_tokens as f64 / out.decode_secs.max(1e-9),
            }),
        ),
        Ok(Err(e)) => reply(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": e.to_string()})),
        Err(e) => reply(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": e.to_string()})),
    })
}

fn reply(status: StatusCode, value: serde_json::Value) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(value.to_string())))
        .expect("built from constants; cannot fail")
}
