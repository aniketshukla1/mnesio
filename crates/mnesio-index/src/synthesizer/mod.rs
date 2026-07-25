//! Synthesizer implementations.
//!
//! [`SnippetSynthesizer`] is the always-compiled, dependency-free reference
//! implementation — pure extractive synthesis. It highlights query-term
//! matches in each retrieved passage and returns the result as structured
//! excerpts that a UI can render with `<mark>`-style emphasis.
//!
//! Future implementations (`MmrCentroidSynthesizer` using the existing
//! `Embedder` to do centroid-based MMR re-ranking; `LlmSynthesizer` for
//! fluent prose with citations) plug into the same [`Synthesizer`] trait.

pub mod snippet;

pub use snippet::SnippetSynthesizer;
