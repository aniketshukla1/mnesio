//! Deterministic stub LLM for tests, CI, and Phase-1's worker harness.
//!
//! Three response-resolution strategies, checked in order:
//!
//! 1. **Exact match** — `with_response(prompt, response)`. Highest priority.
//!    Used when a test pins a specific prompt the worker is expected to
//!    construct (e.g. the canonical A-MEM "extract keywords from this
//!    memory" prompt).
//! 2. **Prefix match** — `with_prefix_match(prefix, response)`. Useful
//!    when many test prompts share a stable preamble but the trailing
//!    content varies (the trailing content being the memory under
//!    test). First-registered prefix wins; this keeps ordering
//!    deterministic.
//! 3. **Default** — `with_default(response)`. Fallback when nothing
//!    above matches. Defaults to `""` so tests fail loudly on
//!    unexpected prompts unless a default is configured.
//!
//! Also records every prompt the client ever saw in `call_log()`, so
//! tests can assert *what was asked* in addition to *what was returned*.

use async_trait::async_trait;
use mnesio_core::{LlmClient, MnesioError};
use std::sync::Mutex;

/// Deterministic LLM stub. See module docs for resolution order.
pub struct FakeLlmClient {
    exact: Vec<(String, String)>,
    prefix: Vec<(String, String)>,
    default: String,
    log: Mutex<Vec<String>>,
}

impl FakeLlmClient {
    pub fn new() -> Self {
        Self {
            exact: Vec::new(),
            prefix: Vec::new(),
            default: String::new(),
            log: Mutex::new(Vec::new()),
        }
    }

    /// Register an exact-prompt → response pair. If the same prompt is
    /// registered twice, the last one wins (typical builder semantics).
    pub fn with_response(mut self, prompt: impl Into<String>, response: impl Into<String>) -> Self {
        let p = prompt.into();
        let r = response.into();
        if let Some(slot) = self.exact.iter_mut().find(|(k, _)| k == &p) {
            slot.1 = r;
        } else {
            self.exact.push((p, r));
        }
        self
    }

    /// Register a prefix → response pair. Useful when many prompts
    /// share a stable preamble but the trailing content varies.
    /// First-registered prefix wins among multiple matches.
    pub fn with_prefix_match(
        mut self,
        prefix: impl Into<String>,
        response: impl Into<String>,
    ) -> Self {
        self.prefix.push((prefix.into(), response.into()));
        self
    }

    /// Fallback response when no exact / prefix rule matches.
    pub fn with_default(mut self, response: impl Into<String>) -> Self {
        self.default = response.into();
        self
    }

    /// Snapshot of every prompt this client has been asked to complete,
    /// in call order. Useful for asserting on what the system under
    /// test actually constructed.
    pub fn call_log(&self) -> Vec<String> {
        self.log.lock().expect("fake llm log poisoned").clone()
    }

    /// How many `complete` calls have been made so far.
    pub fn call_count(&self) -> usize {
        self.log.lock().expect("fake llm log poisoned").len()
    }
}

impl Default for FakeLlmClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmClient for FakeLlmClient {
    async fn complete(&self, prompt: &str) -> Result<String, MnesioError> {
        self.log
            .lock()
            .expect("fake llm log poisoned")
            .push(prompt.to_string());
        if let Some((_, r)) = self.exact.iter().find(|(k, _)| k == prompt) {
            return Ok(r.clone());
        }
        if let Some((_, r)) = self.prefix.iter().find(|(k, _)| prompt.starts_with(k)) {
            return Ok(r.clone());
        }
        Ok(self.default.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn exact_match_returns_configured_response() {
        let c = FakeLlmClient::new().with_response("hello", "world");
        assert_eq!(c.complete("hello").await.unwrap(), "world");
    }

    #[tokio::test]
    async fn prefix_match_handles_variable_suffix() {
        let c = FakeLlmClient::new().with_prefix_match("Extract keywords from:", "kw-a, kw-b");
        let response = c
            .complete("Extract keywords from: revenue grew 18% YoY")
            .await
            .unwrap();
        assert_eq!(response, "kw-a, kw-b");
    }

    #[tokio::test]
    async fn exact_match_wins_over_prefix() {
        let c = FakeLlmClient::new()
            .with_prefix_match("hello", "prefix-resp")
            .with_response("hello world", "exact-resp");
        assert_eq!(c.complete("hello world").await.unwrap(), "exact-resp");
    }

    #[tokio::test]
    async fn default_fires_when_no_rule_matches() {
        let c = FakeLlmClient::new().with_default("default-resp");
        assert_eq!(c.complete("anything").await.unwrap(), "default-resp");
    }

    #[tokio::test]
    async fn empty_default_returns_empty_string() {
        let c = FakeLlmClient::new();
        assert_eq!(c.complete("nothing matches").await.unwrap(), "");
    }

    #[tokio::test]
    async fn first_registered_prefix_wins() {
        let c = FakeLlmClient::new()
            .with_prefix_match("hel", "first")
            .with_prefix_match("hello", "second");
        // Both match but the first registered wins.
        assert_eq!(c.complete("hello world").await.unwrap(), "first");
    }

    #[tokio::test]
    async fn call_log_records_every_prompt_in_order() {
        let c = FakeLlmClient::new().with_default("ok");
        c.complete("first prompt").await.unwrap();
        c.complete("second prompt").await.unwrap();
        c.complete("third prompt").await.unwrap();
        assert_eq!(c.call_count(), 3);
        assert_eq!(
            c.call_log(),
            vec!["first prompt", "second prompt", "third prompt"]
        );
    }

    #[tokio::test]
    async fn re_registering_exact_replaces_response() {
        let c = FakeLlmClient::new()
            .with_response("k", "old")
            .with_response("k", "new");
        assert_eq!(c.complete("k").await.unwrap(), "new");
    }
}
