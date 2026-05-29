//! [`AppContext`] — shared state every tool handler reads.
//!
//! Mirrors the boot sequence in `mneme-server/src/main.rs` but stays
//! lean: no demo writer, no procedural worker, no embedding worker.
//! The MCP server is a thin tool wrapper around the Phase-0 stack.
//!
//! The async embedding pipeline normally fills in `MemoryEmbedded`
//! events asynchronously, keeping the write path < 5 ms. Slice F
//! takes the synchronous path: writes append both `MemoryWritten`
//! AND `MemoryEmbedded` before returning. That's slower (~50-200ms
//! per write depending on embedder) but the LLM tool-call latency
//! is dominated by the LLM round-trip itself, so the tradeoff is
//! benign here. A future slice can move embedding back behind a
//! queue if we ever care.

use anyhow::Result;
use mneme_core::traits::MaterializedView;
use mneme_core::{Embedder, EventLog};
use mneme_index::{Bm25View, HybridRetriever, MockEmbedder, SnippetSynthesizer, VectorView};
use mneme_store::FjallEventLog;
use std::path::Path;
use std::sync::Arc;

/// Everything a tool handler needs at hand. Cheap to clone (all
/// fields are `Arc`).
#[derive(Clone)]
pub struct AppContext {
    pub log: Arc<dyn EventLog>,
    pub vector: Arc<VectorView>,
    pub bm25: Arc<Bm25View>,
    pub embedder: Arc<dyn Embedder>,
    pub retriever: Arc<HybridRetriever>,
    pub synthesizer: Arc<dyn mneme_core::Synthesizer>,
}

impl AppContext {
    /// Boot the context against an on-disk fjall data dir + the
    /// requested embedder. The retrieval views are replayed from the
    /// log so the server starts from a consistent state.
    ///
    /// `embedder_choice`: `"mock"` or `"fastembed"` (any other value
    /// errors). MCP defaults to `mock` so the binary boots instantly
    /// without downloading model weights — production deployments
    /// that want real semantic embeddings pass `"fastembed"`.
    pub async fn open(data_dir: &Path, embedder_choice: &str) -> Result<Self> {
        let log = FjallEventLog::open(data_dir)?;
        let log_trait: Arc<dyn EventLog> = log.clone();

        let embedder: Arc<dyn Embedder> = match embedder_choice {
            "mock" => Arc::new(MockEmbedder::new(32)),
            "fastembed" => Arc::new(mneme_index::FastEmbedEmbedder::new()?),
            other => {
                anyhow::bail!("unknown embedder choice {other:?}; expected `mock` or `fastembed`")
            }
        };

        // Reject mixing embedders against an existing log — same rule
        // mneme-server applies.
        let entries = log_trait.read_from(None).await?;
        for entry in &entries {
            if let mneme_core::event::Event::MemoryEmbedded { model_id, .. } = &entry.event {
                if model_id != embedder.model_id() {
                    anyhow::bail!(
                        "log contains embeddings from {model_id:?} but configured embedder is {:?}; \
                         clear MNEME_DATA or set MNEME_EMBEDDER to match",
                        embedder.model_id()
                    );
                }
            }
        }

        let vector = Arc::new(VectorView::new(
            embedder.dim(),
            embedder.model_id().to_string(),
        ));
        let bm25 = Arc::new(Bm25View::new()?);
        for entry in &entries {
            vector.apply(entry).await?;
            bm25.apply(entry).await?;
        }

        let retriever = Arc::new(HybridRetriever::new(
            vector.clone(),
            bm25.clone(),
            embedder.clone(),
        ));
        let synthesizer: Arc<dyn mneme_core::Synthesizer> = Arc::new(SnippetSynthesizer::new());

        Ok(Self {
            log: log_trait,
            vector,
            bm25,
            embedder,
            retriever,
            synthesizer,
        })
    }
}
