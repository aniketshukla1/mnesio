//! MCP server entry point.
//!
//! Read newline-delimited JSON-RPC requests from stdin, dispatch
//! through [`mnesio_mcp::handler::handle_request`], and write
//! responses to stdout. Logging is configured to write only to
//! stderr so it doesn't corrupt the protocol channel.
//!
//! Environment:
//!
//! - `MNESIO_DATA`     — fjall data directory (default `./mnesio-data`)
//! - `MNESIO_EMBEDDER` — `mock` (default) or `fastembed`
//! - `RUST_LOG`       — standard tracing-subscriber filter

use anyhow::Result;
use mnesio_mcp::context::AppContext;
use mnesio_mcp::handler::process_line;
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

    let data_dir = std::env::var("MNESIO_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./mnesio-data"));
    let embedder_choice = std::env::var("MNESIO_EMBEDDER").unwrap_or_else(|_| "mock".into());

    tracing::info!(
        data = %data_dir.display(),
        embedder = %embedder_choice,
        "mnesio-mcp: booting"
    );

    let ctx = Arc::new(AppContext::open(&data_dir, &embedder_choice).await?);
    tracing::info!("mnesio-mcp: ready — waiting for JSON-RPC over stdio");

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = reader.next_line().await? {
        // All parsing + dispatch + JSON-RPC error classification lives in
        // `process_line` (unit-tested against adversarial input). `None`
        // means a blank line or a notification — nothing to write.
        if let Some(resp) = process_line(&ctx, &line).await {
            let out = serde_json::to_string(&resp)?;
            stdout.write_all(out.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }
    tracing::info!("mnesio-mcp: stdin closed — shutting down");
    Ok(())
}
