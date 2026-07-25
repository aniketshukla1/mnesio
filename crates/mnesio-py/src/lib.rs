//! # mnesio — Python bindings
//!
//! Pip-installable extension exposing mnesio's core APIs to Python.
//! Three primitives: `write_memory`, `search`, `record_outcome` —
//! same surface area as the MCP server, but reachable from any
//! Python agent framework.
//!
//! The pyo3 facade in this module is intentionally thin: every method
//! delegates to [`client::InnerClient`], which is pure Rust and
//! testable without a Python interpreter. The split keeps the test
//! matrix small and ensures both the Python and Rust surfaces benefit
//! from the same set of unit tests.

// pyo3's `#[pymethods]` macro generates `Result::Ok(value.into())` to
// support transparent error-type erasure; clippy flags those as
// useless when the inner type already matches `PyErr`. Suppress here
// at the crate level — the false positives are inside macro
// expansion so we can't site-specifically allow them.
#![allow(clippy::useless_conversion)]

pub mod client;

use crate::client::{InnerClient, RecordOutcomeArgs, SearchArgs, WriteMemoryArgs};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// Map an internal Rust error to a Python exception. Anything
/// argument-shaped becomes `ValueError`; everything else becomes
/// `RuntimeError`.
fn map_err(e: anyhow::Error) -> PyErr {
    let msg = e.to_string();
    if msg.contains("must be non-empty")
        || msg.contains("not a valid ULID")
        || msg.contains("unknown embedder")
    {
        PyValueError::new_err(msg)
    } else {
        PyRuntimeError::new_err(msg)
    }
}

/// `Client` — the entry point Python users instantiate.
///
/// Construct against a data directory + an embedder choice. The
/// embedder is opened synchronously: `fastembed` triggers a one-time
/// model download (~30MB) on first use; `mock` is instant.
#[pyclass]
pub struct Client {
    inner: Mutex<InnerClient>,
}

#[pymethods]
impl Client {
    #[new]
    #[pyo3(signature = (data_dir, embedder = "mock"))]
    fn new(data_dir: PathBuf, embedder: &str) -> PyResult<Self> {
        let inner = InnerClient::open(&data_dir, embedder).map_err(map_err)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Append a memory. Returns the new memory's id (ULID string).
    /// The memory is searchable as soon as this call returns.
    #[pyo3(signature = (content, tenant = "default", tags = None))]
    fn write_memory(
        &self,
        content: String,
        tenant: &str,
        tags: Option<Vec<String>>,
    ) -> PyResult<String> {
        let args = WriteMemoryArgs {
            content,
            tenant: tenant.into(),
            tags: tags.unwrap_or_default(),
        };
        let guard = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("client mutex poisoned: {e}")))?;
        guard.write_memory(args).map_err(map_err)
    }

    /// Hybrid retrieval. Returns a [`SearchResult`] with a synthesized
    /// answer, individual hits, and citation ids.
    #[pyo3(signature = (query, tenant = "default", k = 5))]
    fn search(&self, query: String, tenant: &str, k: usize) -> PyResult<SearchResult> {
        let args = SearchArgs {
            query,
            tenant: tenant.into(),
            k,
        };
        let guard = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("client mutex poisoned: {e}")))?;
        let outcome = guard.search(args).map_err(map_err)?;
        Ok(SearchResult {
            answer: outcome.answer,
            hits: outcome
                .hits
                .into_iter()
                .map(|h| SearchHit {
                    memory_id: h.memory_id,
                    content: h.content,
                    tags: h.tags,
                    score: h.score,
                })
                .collect(),
            citations: outcome.citations,
        })
    }

    /// Record an outcome for the procedural compiler to consume.
    /// `artifacts_used` is a list of ULID-string artifact ids;
    /// `scores` is an optional dict of name → numeric score
    /// (e.g. `{"accuracy": 0.92, "latency_ms": 1850}`).
    #[pyo3(signature = (artifacts_used, success, episode = None, scores = None, error = None))]
    fn record_outcome(
        &self,
        artifacts_used: Vec<String>,
        success: bool,
        episode: Option<String>,
        scores: Option<HashMap<String, f32>>,
        error: Option<String>,
    ) -> PyResult<String> {
        let args = RecordOutcomeArgs {
            episode,
            artifacts_used,
            success,
            scores: scores.unwrap_or_default(),
            error,
        };
        let guard = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("client mutex poisoned: {e}")))?;
        guard.record_outcome(args).map_err(map_err)
    }

    fn __repr__(&self) -> String {
        "<mnesio.Client>".to_string()
    }
}

/// Structured search result. Iterable via `.hits`.
#[pyclass]
#[derive(Clone)]
pub struct SearchResult {
    /// Optional synthesized prose. `None` when the synthesizer chose
    /// not to produce one (e.g. relevance signal too weak).
    #[pyo3(get)]
    pub answer: Option<String>,
    /// Individual hits, ranked by hybrid score.
    #[pyo3(get)]
    pub hits: Vec<SearchHit>,
    /// Memory ids the synthesizer cited in its answer.
    #[pyo3(get)]
    pub citations: Vec<String>,
}

#[pymethods]
impl SearchResult {
    fn __repr__(&self) -> String {
        format!(
            "<mnesio.SearchResult hits={} citations={}>",
            self.hits.len(),
            self.citations.len()
        )
    }

    fn __len__(&self) -> usize {
        self.hits.len()
    }
}

/// One ranked hit from a search.
#[pyclass]
#[derive(Clone)]
pub struct SearchHit {
    #[pyo3(get)]
    pub memory_id: String,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub tags: Vec<String>,
    #[pyo3(get)]
    pub score: f32,
}

#[pymethods]
impl SearchHit {
    fn __repr__(&self) -> String {
        format!(
            "<mnesio.SearchHit memory_id={} score={:.3}>",
            self.memory_id, self.score
        )
    }
}

/// The `mnesio._mnesio` extension module that the Python package
/// re-exports under `mnesio.*`.
#[pymodule]
fn _mnesio(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Client>()?;
    m.add_class::<SearchResult>()?;
    m.add_class::<SearchHit>()?;

    // Expose a `MnesioError` alias for `RuntimeError` so callers can
    // `except mnesio.MnesioError:` if they want a single catch-all.
    let runtime_error = py.get_type_bound::<PyRuntimeError>();
    m.add("MnesioError", runtime_error)?;

    Ok(())
}
