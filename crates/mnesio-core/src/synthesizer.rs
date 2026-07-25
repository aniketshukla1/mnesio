//! Synthesizer seam — the layer that turns retrieval results into an answer.
//!
//! `Retriever` gives you ranked memories. `Synthesizer` composes an
//! **answer** from them: a structured set of excerpts with citations,
//! optionally accompanied by free-form prose. The split keeps the
//! production-shipping default ([`SnippetSynthesizer`] — deterministic,
//! extractive, no LLM) cleanly separated from heavier implementations
//! (`MmrCentroidSynthesizer`, `LlmSynthesizer`) we'll add later.
//!
//! The auditability story compounds here: extractive synthesizers can only
//! emit text that came from a real `Memory`, so every word in an answer
//! can be traced back to an immutable [`crate::types::MemoryRef`]. That's
//! the "you can see exactly how the agent reached its answer" promise made
//! concrete (long-form §8.2 differentiator #4).

use crate::types::MemoryRef;
use crate::MnesioError;
use async_trait::async_trait;
use serde::Serialize;

/// A retrieved memory passed to the synthesizer for excerpting.
///
/// The synthesizer doesn't talk to the event log itself; the host
/// (`mnesio-server`) resolves [`crate::traits::Hit`]s to `Passage`s so the
/// trait stays storage-agnostic. Future synthesizers that need richer
/// context (neighbour chunks, source metadata) get extended `Passage`
/// fields rather than coupling to a specific store.
#[derive(Debug, Clone)]
pub struct Passage {
    pub memory: MemoryRef,
    pub content: String,
    pub tags: Vec<String>,
    /// The retrieval score that surfaced this passage. Synthesizers can use
    /// it as a prior; the wire format echoes it through so UIs can show
    /// "answer derived from these top-N memories" with confidence.
    pub retrieval_score: f32,
}

/// One excerpt in a composed [`Answer`].
///
/// `segments` is a sequence of plain / highlighted chunks so a UI can
/// render the excerpt with query terms bolded without needing to know the
/// matching algorithm. The synthesizer decides which spans matter.
#[derive(Debug, Clone, Serialize)]
pub struct Excerpt {
    pub memory: MemoryRef,
    pub segments: Vec<ExcerptSegment>,
    pub retrieval_score: f32,
}

/// A run of text in an excerpt, either rendered plainly or highlighted as
/// matching the query.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ExcerptSegment {
    Plain { text: String },
    Highlight { text: String },
}

impl ExcerptSegment {
    pub fn text(&self) -> &str {
        match self {
            ExcerptSegment::Plain { text } | ExcerptSegment::Highlight { text } => text,
        }
    }
}

/// The synthesized answer returned to a caller.
///
/// `prose` is `None` for extractive synthesizers — they don't generate new
/// text, only select and highlight existing passages. LLM-backed
/// synthesizers populate `prose` with citation-bearing connective writing.
#[derive(Debug, Clone, Serialize)]
pub struct Answer {
    pub query: String,
    pub excerpts: Vec<Excerpt>,
    /// Distinct sources cited, in the order they first appear in
    /// `excerpts`. UIs render these as numbered citation chips.
    pub citations: Vec<MemoryRef>,
    /// Free-form prose composed across excerpts. `None` from extractive
    /// synthesizers; `Some` from LLM synthesizers.
    pub prose: Option<String>,
    pub provenance: SynthesisProvenance,
}

#[derive(Debug, Clone, Serialize)]
pub struct SynthesisProvenance {
    /// Name of the synthesizer (matches `Synthesizer::name`).
    pub synthesizer: String,
    /// Underlying model identifier when applicable (LLM-backed
    /// synthesizers). `None` for purely extractive / heuristic ones.
    pub model_id: Option<String>,
    pub elapsed_ms: u64,
}

/// Compose an [`Answer`] from a set of [`Passage`]s.
///
/// Implementations promise determinism unless they document otherwise.
/// `SnippetSynthesizer` and `MmrCentroidSynthesizer` are deterministic;
/// `LlmSynthesizer` is not.
#[async_trait]
pub trait Synthesizer: Send + Sync {
    /// Stable identifier — matches the `synthesizer` field in
    /// [`SynthesisProvenance`].
    fn name(&self) -> &str;

    async fn synthesize(&self, query: &str, passages: &[Passage]) -> Result<Answer, MnesioError>;
}
