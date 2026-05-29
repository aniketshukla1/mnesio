//! MCP server entry point.
//!
//! Read newline-delimited JSON-RPC requests from stdin, dispatch
//! through [`mneme_mcp::handler::handle_request`], and write
//! responses to stdout. Logging is configured to write only to
//! stderr so it doesn't corrupt the protocol channel.
//!
//! Environment:
//!
//! - `MNEME_DATA`     — fjall data directory (default `./mneme-data`)
//! - `MNEME_EMBEDDER` — `mock` (default) or `fastembed`
//! - `RUST_LOG`       — standard tracing-subscriber filter

use anyhow::Result;
use mneme_mcp::context::AppContext;
use mneme_mcp::handler::handle_request;
use mneme_mcp::protocol::{error_codes, Request, Response, ResponseError};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> Result<()> {
    // CRITICAL: stderr-only logging. stdout is the JSON-RPC channel;
    // a stray log line would corrupt the protocol.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let data_dir = std::env::var("MNEME_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./mneme-data"));
    let embedder_choice = std::env::var("MNEME_EMBEDDER").unwrap_or_else(|_| "mock".into());

    tracing::info!(
        data = %data_dir.display(),
        embedder = %embedder_choice,
        "mneme-mcp: booting"
    );

    let ctx = Arc::new(AppContext::open(&data_dir, &embedder_choice).await?);
    tracing::info!("mneme-mcp: ready — waiting for JSON-RPC over stdio");

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = reader.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Parse the line. Parse errors return a JSON-RPC error
        // response with a null id (we don't know what the id was if
        // the request didn't parse).
        let response = match serde_json::from_str::<Request>(trimmed) {
            Ok(req) => match handle_request(&ctx, req).await {
                Some(r) => Some(r),
                None => continue, // notification — nothing to write
            },
            Err(e) => {
                tracing::warn!(error = %e, line = %trimmed, "parse error");
                Some(Response::failure(
                    serde_json::Value::Null,
                    ResponseError::new(error_codes::PARSE_ERROR, format!("invalid JSON: {e}")),
                ))
            }
        };
        if let Some(resp) = response {
            let line = serde_json::to_string(&resp)?;
            stdout.write_all(line.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }
    tracing::info!("mneme-mcp: stdin closed — shutting down");
    Ok(())
}
