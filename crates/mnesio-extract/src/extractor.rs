//! The [`Extractor`] seam and its LLM-backed implementation.

use crate::parse::parse_facts;
use crate::prompts;
use async_trait::async_trait;
use mnesio_core::{LlmClient, MnesioError};
use std::sync::Arc;

/// Pulls atomic, self-contained facts out of raw text. Behind a trait so
/// a future rule-based / spaCy-style / fine-tuned extractor can swap in
/// without touching the consolidator.
#[async_trait]
pub trait Extractor: Send + Sync {
    /// Extract zero or more atomic facts from `raw`. Returning an empty
    /// vec means "nothing worth remembering" and is a normal outcome,
    /// not an error.
    async fn extract(&self, raw: &str) -> Result<Vec<String>, MnesioError>;
}

/// LLM-backed extractor — the default. One `complete` call per raw turn.
pub struct LlmExtractor {
    llm: Arc<dyn LlmClient>,
}

impl LlmExtractor {
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl Extractor for LlmExtractor {
    async fn extract(&self, raw: &str) -> Result<Vec<String>, MnesioError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }
        let prompt = prompts::extract_facts(trimmed);
        let response = self.llm.complete(&prompt).await?;
        Ok(parse_facts(&response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_llm::FakeLlmClient;

    #[tokio::test]
    async fn llm_extractor_parses_facts() {
        let llm = Arc::new(
            FakeLlmClient::new()
                .with_default("FACT: Alice likes oat milk.\nFACT: Bob is in Berlin."),
        );
        let ex = LlmExtractor::new(llm);
        let facts = ex.extract("some raw conversation").await.unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0], "Alice likes oat milk.");
    }

    #[tokio::test]
    async fn llm_extractor_empty_input_skips_llm() {
        let llm = Arc::new(FakeLlmClient::new().with_default("FACT: should not happen"));
        let ex = LlmExtractor::new(llm.clone());
        let facts = ex.extract("   ").await.unwrap();
        assert!(facts.is_empty());
        assert_eq!(llm.call_count(), 0, "must not call the LLM on empty input");
    }

    #[tokio::test]
    async fn llm_extractor_none_response_is_empty() {
        let llm = Arc::new(FakeLlmClient::new().with_default("NONE"));
        let ex = LlmExtractor::new(llm);
        assert!(ex.extract("hi how are you").await.unwrap().is_empty());
    }
}
