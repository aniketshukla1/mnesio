//! The public traits. These are the seams the rest of the workspace is
//! built around — each materialized view, the LLM client, and the embedder
//! are all swappable behind a trait (report §7, "abstract behind a trait").

use crate::event::LogEntry;
use crate::types::*;
use async_trait::async_trait;

/// The append-only event log: the system of record.
#[async_trait]
pub trait EventLog: Send + Sync {
    async fn append(&self, event: crate::event::Event) -> Result<Id, MnesioError>;
    /// Stream entries with id strictly greater than `after` (tail / replay).
    async fn read_from(&self, after: Option<Id>) -> Result<Vec<LogEntry>, MnesioError>;
}

/// Anything that consumes the event tail to maintain derived state.
#[async_trait]
pub trait MaterializedView: Send + Sync {
    fn name(&self) -> &str;
    async fn apply(&self, entry: &LogEntry) -> Result<(), MnesioError>;
    /// Id of the last entry this view has durably processed.
    async fn checkpoint(&self) -> Result<Option<Id>, MnesioError>;
}

/// Retrieval surface — implemented by the hybrid orchestrator over the
/// vector, BM25 and graph views.
#[async_trait]
pub trait Retriever: Send + Sync {
    async fn search(&self, query: &Query) -> Result<Vec<Hit>, MnesioError>;
}

/// A scoped, filtered retrieval request.
#[derive(Debug, Clone)]
pub struct Query {
    pub text: String,
    pub scope: Scope,
    pub k: usize,
    pub time_filter: Option<time::OffsetDateTime>,
}

/// A retrieval result with an explainable score breakdown.
#[derive(Debug, Clone)]
pub struct Hit {
    pub memory: MemoryRef,
    pub score: f32,
    /// Per-signal contributions (vector, bm25, graph, recency, ...).
    pub breakdown: Vec<(String, f32)>,
}

/// LLM client seam — keep the rest of the system provider-agnostic.
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<String, MnesioError>;
}

/// Embedding seam — fastembed today, swappable later (report §7 caveat).
///
/// Every implementation must expose a stable [`Embedder::model_id`]. The event
/// log records this id alongside the embedding so a view rebuild can detect a
/// mismatched embedder at startup and refuse to silently mix vector spaces
/// (long-form project description §4.5).
#[async_trait]
pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    /// Stable identifier for the (model, dimension) pair. Two embedders that
    /// produce comparable vector spaces must return the same `model_id`.
    fn model_id(&self) -> &str;
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, MnesioError>;
}

#[derive(Debug, thiserror::Error)]
pub enum MnesioError {
    #[error("storage: {0}")]
    Storage(String),
    #[error("index: {0}")]
    Index(String),
    #[error("llm: {0}")]
    Llm(String),
    #[error("scope violation: {0} may not access {1}")]
    ScopeViolation(String, String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
