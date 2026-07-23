//! # mneme-llm
//!
//! `LlmClient` implementations. The trait itself lives in
//! [`mneme_core::traits::LlmClient`]; this crate owns the concrete
//! backends so heavyweight deps (HTTP, ONNX, future things) don't
//! leak into `mneme-core`.
//!
//! Three implementations ship today:
//!
//! - [`FakeLlmClient`] — deterministic, dependency-free, always
//!   compiled. The Phase-1 [evolution worker][mneme-evolve] tests and
//!   the eventual Phase-2 procedural compiler tests run against it so
//!   the whole workspace's test suite stays fast and offline.
//! - [`OllamaLlmClient`] — real local backend talking to an Ollama
//!   instance over HTTP, behind the `ollama` feature flag. Default-on
//!   for production builds; CI / lean builds can opt out via
//!   `--no-default-features`.
//! - [`OpenAiCompatClient`] — OpenAI-compatible chat backend (OpenRouter
//!   / OpenAI / any `/v1/chat/completions` gateway), behind the `openai`
//!   feature. This is the frontier-class answerer + judge the published
//!   LoCoMo / LongMemEval numbers run through. The API key is read from
//!   the environment only.

pub mod fake;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "openai")]
pub mod openai;

pub use fake::FakeLlmClient;

#[cfg(feature = "ollama")]
pub use ollama::OllamaLlmClient;

#[cfg(feature = "openai")]
pub use openai::OpenAiCompatClient;
