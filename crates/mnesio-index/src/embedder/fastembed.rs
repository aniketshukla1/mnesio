//! `fastembed`-backed embedder.
//!
//! Local ONNX inference via the [`fastembed`] crate. Default model is
//! `BAAI/bge-small-en-v1.5` (384-dim) — the shipped default in the long-form
//! project description §4.5. On first use, fastembed downloads the model
//! (~30MB) to its cache directory; subsequent runs are offline.
//!
//! `fastembed::TextEmbedding::embed` is synchronous and CPU-bound (it runs
//! ONNX inference). We wrap it in `tokio::task::spawn_blocking` so the async
//! runtime keeps making progress while inference runs — this is what lets
//! mnesio satisfy hard rule #5 (write path stays fast) with a real embedder
//! plus an async embedding worker.
//!
//! Only compiled when the `fastembed` feature is enabled (default-on for
//! production builds; CI / tests opt out via `--no-default-features`).

use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use mnesio_core::{Embedder, MnesioError};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Default embedder used by the server when no override is requested.
/// `BGESmallENV15` is the canonical "small + fast + good" choice at 384-dim;
/// upgrades to `BGELargeENV15` (1024-dim) or others can swap one constant.
pub const DEFAULT_MODEL: EmbeddingModel = EmbeddingModel::BGESmallENV15;

/// Stable identifier we record in the event log alongside every embedding.
/// Two `FastEmbedEmbedder`s built around the same `EmbeddingModel` must
/// produce the same `model_id` so the startup mismatch check can match
/// them deterministically.
pub fn model_id_for(model: &EmbeddingModel) -> String {
    format!("fastembed-{model:?}")
}

/// Lazily-initialised `fastembed::TextEmbedding`. `embed` takes `&mut self`
/// in the upstream crate, so callers hold a `Mutex` and serialise inference
/// through it. `tokio::sync::Mutex` (not `std::sync::Mutex`) so we can hold
/// the guard across the await for the `spawn_blocking` join.
pub struct FastEmbedEmbedder {
    model: Arc<Mutex<TextEmbedding>>,
    model_id: String,
    dim: usize,
    /// Optional batch size override passed to `TextEmbedding::embed`. `None`
    /// lets fastembed pick a sensible default (256 at the time of writing).
    batch_size: Option<usize>,
}

impl FastEmbedEmbedder {
    /// Construct a new embedder with the default model
    /// (`BAAI/bge-small-en-v1.5`).
    pub fn new() -> Result<Self, MnesioError> {
        Self::with_model(DEFAULT_MODEL)
    }

    /// Construct a new embedder with an explicit model choice. Initialisation
    /// triggers a model download on first use (network required); subsequent
    /// calls hit the local cache.
    pub fn with_model(model: EmbeddingModel) -> Result<Self, MnesioError> {
        let info = TextEmbedding::get_model_info(&model)
            .map_err(|e| MnesioError::Other(anyhow::anyhow!("fastembed model info: {e}")))?;
        let dim = info.dim;
        let model_id = model_id_for(&model);
        tracing::info!(
            model = ?model,
            dim,
            "loading fastembed model (first run will download ~30MB to the cache dir)"
        );
        let inner = TextEmbedding::try_new(InitOptions::new(model))
            .map_err(|e| MnesioError::Other(anyhow::anyhow!("fastembed init: {e}")))?;
        Ok(Self {
            model: Arc::new(Mutex::new(inner)),
            model_id,
            dim,
            batch_size: None,
        })
    }

    /// Override the batch size passed to `TextEmbedding::embed`. Tune only
    /// if you have a specific reason; the upstream default works well.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }
}

#[async_trait]
impl Embedder for FastEmbedEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, MnesioError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // `TextEmbedding::embed` is sync + CPU-bound; offload via
        // spawn_blocking so the runtime keeps draining other tasks.
        let model = self.model.clone();
        let owned: Vec<String> = texts.to_vec();
        let batch = self.batch_size;
        let result = tokio::task::spawn_blocking(move || {
            // We hold the std-style lock inside the blocking thread —
            // `tokio::sync::Mutex` is the wrong type here because we're
            // not crossing an await *inside* the closure. Use blocking_lock.
            let mut guard = model.blocking_lock();
            guard.embed(owned, batch)
        })
        .await
        .map_err(|e| MnesioError::Other(anyhow::anyhow!("embed join: {e}")))?
        .map_err(|e| MnesioError::Other(anyhow::anyhow!("fastembed: {e}")))?;
        Ok(result)
    }
}
