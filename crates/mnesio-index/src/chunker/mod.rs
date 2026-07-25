//! Chunkers turn a long document into a sequence of shorter strings, each
//! of which becomes a [`mnesio_core::Memory`] sharing a common
//! [`mnesio_core::SourceRef`].
//!
//! The trait stays narrow on purpose — `chunk(&str) -> Vec<String>`.
//! Position metadata, source metadata, and embedding all happen at the
//! call site (the ingest pipeline in `mnesio-server`, today). That keeps
//! chunking strategies cleanly substitutable as we add more (paragraph,
//! sliding window, semantic-boundary via LLM).
//!
//! Today's implementations:
//!
//! - [`ParagraphChunker`] — splits on blank lines, the obvious default for
//!   structured prose (memos, reports, blog posts).
//!
//! Roadmap (later slices): `WindowChunker` for overlapping sliding
//! windows; `SemanticChunker` that asks an `LlmClient` to find natural
//! break points.

pub mod paragraph;

pub use paragraph::ParagraphChunker;

/// Turn a document into ordered chunks. Implementations must produce a
/// stable order: re-running on the same input must yield the same
/// `Vec<String>`, in the same order, so `Memory.position` stays meaningful
/// across ingests of the same document.
pub trait Chunker: Send + Sync {
    fn chunk(&self, text: &str) -> Vec<String>;
}
