//! [`PolicyExecutor`] — "run this artifact against this input, return
//! what it produced."
//!
//! The shadow evaluator needs to compare what a candidate artifact
//! *does* against what the active artifact *did*. The trait abstracts
//! "doing": for a `SystemPrompt`, it means prepending the body to an
//! input and asking an LLM; for a `Heuristic` (future), it'd mean
//! matching the `when` clause and emitting the `then`; for a `Skill`
//! (further future), executing a code body.
//!
//! Slice B supports `SystemPrompt` only. Other kinds return an explicit
//! `MnesioError::Other` so the compile pipeline fails *visibly* if we
//! accidentally extend it to non-supported kinds before we've taught
//! the executor.

use async_trait::async_trait;
use mnesio_core::entity::{ArtifactKind, PolicyArtifact};
use mnesio_core::{LlmClient, MnesioError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Execute a policy artifact against a single input string. Returns the
/// model output. Implementations decide what "execute" means per
/// [`ArtifactKind`].
#[async_trait]
pub trait PolicyExecutor: Send + Sync {
    async fn execute(&self, artifact: &PolicyArtifact, input: &str) -> Result<String, MnesioError>;
}

/// Default executor: composes a `SystemPrompt` body with the input and
/// calls an [`LlmClient`]. Errors out for non-`SystemPrompt` kinds —
/// upcoming slices teach it more kinds.
pub struct LlmExecutor {
    llm: Arc<dyn LlmClient>,
}

impl LlmExecutor {
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl PolicyExecutor for LlmExecutor {
    async fn execute(&self, artifact: &PolicyArtifact, input: &str) -> Result<String, MnesioError> {
        let body = match &artifact.kind {
            ArtifactKind::SystemPrompt { body } => body,
            other => {
                return Err(MnesioError::Other(anyhow::anyhow!(
                    "LlmExecutor only supports SystemPrompt artifacts in Slice B (got {})",
                    kind_name(other)
                )));
            }
        };
        // Compose a minimal chat-style prompt. Real production paths
        // will format per LLM-specific role tokens; for the structured-
        // output prompts we issue this flat concatenation is fine.
        let prompt = format!("{body}\n\nUser: {input}\nAssistant:");
        self.llm.complete(&prompt).await
    }
}

/// Test-only executor that returns a canned response per
/// `(artifact_version, input)` pair. Deterministic, dependency-free.
///
/// Useful for shadow-eval tests that want to assert "candidate X passes
/// canary Y" without engaging a real LLM.
pub struct FakePolicyExecutor {
    /// (artifact_version, input) → response. Use `with_response` to
    /// register canned answers; falls back to `default_response` for
    /// any unmatched key.
    table: Mutex<HashMap<(u32, String), String>>,
    default_response: String,
    /// (artifact_version, input) calls, in arrival order. Tests inspect
    /// this to verify the executor was driven the expected way.
    pub calls: Mutex<Vec<(u32, String)>>,
}

impl FakePolicyExecutor {
    pub fn new() -> Self {
        Self {
            table: Mutex::new(HashMap::new()),
            default_response: String::new(),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Set the response returned for any input that doesn't have an
    /// explicit `with_response` entry. Defaults to empty string.
    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default_response = default.into();
        self
    }

    /// Register a canned response for `(version, input)`. Last write wins.
    pub fn with_response(
        self,
        version: u32,
        input: impl Into<String>,
        response: impl Into<String>,
    ) -> Self {
        self.table
            .lock()
            .unwrap()
            .insert((version, input.into()), response.into());
        self
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

impl Default for FakePolicyExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PolicyExecutor for FakePolicyExecutor {
    async fn execute(&self, artifact: &PolicyArtifact, input: &str) -> Result<String, MnesioError> {
        let key = (artifact.version, input.to_string());
        self.calls.lock().unwrap().push(key.clone());
        let response = self
            .table
            .lock()
            .unwrap()
            .get(&key)
            .cloned()
            .unwrap_or_else(|| self.default_response.clone());
        Ok(response)
    }
}

/// Friendly name for an [`ArtifactKind`] — used in error messages so the
/// operator sees "Heuristic" instead of `Heuristic { when: ..., then: ... }`.
fn kind_name(k: &ArtifactKind) -> &'static str {
    match k {
        ArtifactKind::SystemPrompt { .. } => "SystemPrompt",
        ArtifactKind::Heuristic { .. } => "Heuristic",
        ArtifactKind::Skill { .. } => "Skill",
        ArtifactKind::RetrievalRule { .. } => "RetrievalRule",
        ArtifactKind::Reflection { .. } => "Reflection",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::types::{new_id, BiTemporal, Scope};
    use mnesio_llm::FakeLlmClient;

    fn prompt_artifact(version: u32, body: &str) -> PolicyArtifact {
        PolicyArtifact {
            id: new_id(),
            version,
            scope: Scope::global("t"),
            kind: ArtifactKind::SystemPrompt { body: body.into() },
            canaries: Vec::new(),
            time: BiTemporal::now(),
        }
    }

    fn heuristic_artifact() -> PolicyArtifact {
        PolicyArtifact {
            id: new_id(),
            version: 1,
            scope: Scope::global("t"),
            kind: ArtifactKind::Heuristic {
                when: "w".into(),
                then: "t".into(),
            },
            canaries: Vec::new(),
            time: BiTemporal::now(),
        }
    }

    #[tokio::test]
    async fn llm_executor_concatenates_body_with_input() {
        let llm = Arc::new(FakeLlmClient::new().with_default("ok"));
        let ex = LlmExecutor::new(llm.clone());
        let out = ex
            .execute(&prompt_artifact(1, "Answer briefly."), "What is 2+2?")
            .await
            .unwrap();
        assert_eq!(out, "ok");
        // Verify the prompt actually contained both pieces.
        let log = llm.call_log();
        assert!(log[0].contains("Answer briefly."));
        assert!(log[0].contains("What is 2+2?"));
    }

    #[tokio::test]
    async fn llm_executor_rejects_non_system_prompt_kinds_explicitly() {
        let llm = Arc::new(FakeLlmClient::new());
        let ex = LlmExecutor::new(llm);
        let err = ex.execute(&heuristic_artifact(), "x").await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Heuristic"),
            "error must name the unsupported kind: {msg}"
        );
        assert!(
            msg.contains("Slice B"),
            "error must point at why it's unsupported: {msg}"
        );
    }

    #[tokio::test]
    async fn fake_executor_returns_canned_response_per_version_and_input() {
        let ex = FakePolicyExecutor::new()
            .with_default("default")
            .with_response(1, "sky?", "blue")
            .with_response(2, "sky?", "azure");
        let a1 = prompt_artifact(1, "x");
        let a2 = prompt_artifact(2, "x");
        assert_eq!(ex.execute(&a1, "sky?").await.unwrap(), "blue");
        assert_eq!(ex.execute(&a2, "sky?").await.unwrap(), "azure");
        assert_eq!(ex.execute(&a1, "grass?").await.unwrap(), "default");
    }

    #[tokio::test]
    async fn fake_executor_logs_every_call_in_order() {
        let ex = FakePolicyExecutor::new();
        ex.execute(&prompt_artifact(1, "x"), "a").await.unwrap();
        ex.execute(&prompt_artifact(1, "x"), "b").await.unwrap();
        ex.execute(&prompt_artifact(2, "x"), "a").await.unwrap();
        assert_eq!(ex.call_count(), 3);
        let calls = ex.calls.lock().unwrap();
        assert_eq!(calls[0], (1, "a".into()));
        assert_eq!(calls[2], (2, "a".into()));
    }
}
