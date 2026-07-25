//! [`Reflector`] — drives the LLM through reflect → propose.
//!
//! Given the active artifact and a batch of recent outcomes, asks the
//! LLM for failure patterns then for K concrete candidate revisions.
//! Returns a [`Proposal`] envelope ready for shadow evaluation.
//!
//! Behavior on failure modes:
//!
//! - LLM error during reflection → propagate as `MnesioError` (the
//!   compile loop logs + retries on the next tick).
//! - Reflection returns no findings → `Ok(None)`. There's nothing to
//!   propose; skip.
//! - LLM error during proposal → propagate.
//! - Parser returns zero candidates → `Ok(None)`. Same logic as above.
//! - Any individual candidate body shorter than [`MIN_CANDIDATE_LEN`]
//!   is dropped — almost-empty proposals never improve anything and
//!   waste eval budget.

use crate::parse::{parse_proposals, parse_reflection};
use crate::prompts::{proposal_prompt, reflection_prompt};
use crate::proposal::Proposal;
use mnesio_core::entity::{ArtifactKind, Outcome, PolicyArtifact};
#[cfg(test)]
use mnesio_core::types::new_id;
use mnesio_core::types::{ArtifactRef, BiTemporal};
use mnesio_core::{LlmClient, MnesioError};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Minimum candidate body length, in chars. Almost-empty proposals
/// (LLM said "yes." and nothing else) never improve anything.
pub const MIN_CANDIDATE_LEN: usize = 8;

/// Drives the reflect → propose phases. Owns the LLM client; holds no
/// other state — every `propose` call is independent.
pub struct Reflector {
    llm: Arc<dyn LlmClient>,
    /// How many candidate revisions to ask for per pass. The compiler's
    /// `candidates_per_artifact` is plumbed in here; the prompt clamps
    /// to `[1, 8]` defensively.
    pub k_candidates: usize,
}

impl Reflector {
    pub fn new(llm: Arc<dyn LlmClient>, k_candidates: usize) -> Self {
        Self { llm, k_candidates }
    }

    /// Run the full reflect → propose sequence. Returns `Ok(None)` for
    /// any of the "soft failure" branches documented at the module
    /// level; returns `Err` only for LLM-call failures.
    pub async fn propose(
        &self,
        active: &PolicyArtifact,
        outcomes: &[Outcome],
        rationale: impl Into<String>,
    ) -> Result<Option<Proposal>, MnesioError> {
        // Only SystemPrompt is supported in Slice B — bail visibly
        // rather than silently proposing nothing.
        if !matches!(active.kind, ArtifactKind::SystemPrompt { .. }) {
            return Err(MnesioError::Other(anyhow::anyhow!(
                "Reflector only supports SystemPrompt artifacts in Slice B"
            )));
        }

        // ---- Step 1: reflect ----
        let reflect_p = reflection_prompt(active, outcomes);
        let reflect_resp = self.llm.complete(&reflect_p).await?;
        let summary = parse_reflection(&reflect_resp);
        if summary.is_empty() {
            tracing::debug!("reflector: no findings — skipping proposal step");
            return Ok(None);
        }

        // ---- Step 2: propose ----
        let proposal_p = proposal_prompt(active, &summary.as_bullet_list(), self.k_candidates);
        let proposal_resp = self.llm.complete(&proposal_p).await?;
        let candidates: Vec<String> = parse_proposals(&proposal_resp)
            .into_iter()
            .filter(|body| body.len() >= MIN_CANDIDATE_LEN)
            .collect();
        if candidates.is_empty() {
            tracing::debug!("reflector: no usable candidates after parse — skipping");
            return Ok(None);
        }

        // Build PolicyArtifact candidates by cloning the active
        // artifact and swapping the body. Critical: each candidate
        // **reuses `active.id`** so a commit *replaces* the active
        // version under that id rather than creating a parallel
        // entry. The version field bumps by 1 to mark the lineage;
        // canaries carry over verbatim so the shadow evaluator
        // re-checks them.
        //
        // Multiple K candidates all sharing `(id=active.id, version=N+1)`
        // is intentional — only one wins the gate and lands in the
        // store's `active` map, replacing the (id=active.id, version=N)
        // entry. The losers are recorded in the `ProceduralProposed`
        // event for audit, then forgotten.
        let candidate_artifacts: Vec<PolicyArtifact> = candidates
            .into_iter()
            .map(|body| PolicyArtifact {
                id: active.id,
                version: active.version.saturating_add(1),
                scope: active.scope.clone(),
                kind: ArtifactKind::SystemPrompt { body },
                canaries: active.canaries.clone(),
                time: BiTemporal::now(),
            })
            .collect();

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Ok(Some(Proposal::new(
            active.scope.clone(),
            Some(ArtifactRef(active.id)),
            candidate_artifacts,
            rationale,
            now_ms,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::entity::JudgeSource;
    use mnesio_core::types::{EpisodeRef, Scope, TrajectoryRef};
    use mnesio_llm::FakeLlmClient;
    use std::collections::HashMap;

    fn active(body: &str) -> PolicyArtifact {
        PolicyArtifact {
            id: new_id(),
            version: 1,
            scope: Scope::global("t"),
            kind: ArtifactKind::SystemPrompt { body: body.into() },
            canaries: Vec::new(),
            time: BiTemporal::now(),
        }
    }

    fn outcome(success: bool) -> Outcome {
        let mut scores = HashMap::new();
        scores.insert("accuracy".into(), if success { 1.0 } else { 0.0 });
        Outcome {
            id: new_id(),
            episode: EpisodeRef(new_id()),
            artifacts_used: vec![ArtifactRef(new_id())],
            success: Some(success),
            scores,
            error: if success { None } else { Some("wrong".into()) },
            judge: JudgeSource::Environment,
            trajectory: TrajectoryRef(new_id()),
        }
    }

    /// FakeLlmClient that returns a canned reflection (with one
    /// finding) for the reflect prompt and 2 canned candidates for the
    /// propose prompt. Prefix-matched so the order doesn't matter.
    fn happy_llm() -> Arc<FakeLlmClient> {
        Arc::new(
            FakeLlmClient::new()
                .with_prefix_match(
                    "You are reviewing a recent batch",
                    "FINDING: the prompt is too terse for compound questions",
                )
                .with_prefix_match(
                    "You are revising a system prompt",
                    "--- CANDIDATE 1 ---\n\
                     You are a careful, helpful assistant. Answer in 1-3 sentences.\n\
                     --- CANDIDATE 2 ---\n\
                     You are a careful, helpful assistant. For compound questions, address each part.",
                ),
        )
    }

    #[tokio::test]
    async fn happy_path_produces_proposal_with_k_candidates() {
        let llm = happy_llm();
        let r = Reflector::new(llm, 2);
        let prop = r
            .propose(
                &active("Answer briefly."),
                &[outcome(false), outcome(false)],
                "test",
            )
            .await
            .unwrap()
            .expect("reflector should produce a proposal");
        assert_eq!(prop.candidates.len(), 2);
        assert!(prop.candidates[0].kind_is_system_prompt_containing("careful, helpful"));
        // base_version points at the active artifact's id.
        assert!(prop.base_version.is_some());
        // version bumped past active's.
        for c in &prop.candidates {
            assert_eq!(c.version, 2);
        }
    }

    #[tokio::test]
    async fn no_findings_returns_none_without_calling_propose_prompt() {
        let llm = Arc::new(FakeLlmClient::new().with_default("FINDING: none"));
        let r = Reflector::new(llm.clone(), 2);
        let p = r
            .propose(&active("body"), &[outcome(true)], "test")
            .await
            .unwrap();
        assert!(p.is_none(), "no findings → no proposal");
        // Only one LLM call (reflect), not two (no propose).
        assert_eq!(llm.call_count(), 1);
    }

    #[tokio::test]
    async fn no_parsed_candidates_returns_none() {
        let llm = Arc::new(
            FakeLlmClient::new()
                .with_prefix_match("You are reviewing", "FINDING: real one")
                .with_prefix_match("You are revising", "I can't help with that."),
        );
        let r = Reflector::new(llm, 2);
        let p = r
            .propose(&active("body"), &[outcome(false)], "test")
            .await
            .unwrap();
        assert!(p.is_none(), "parser returned zero candidates → no proposal");
    }

    #[tokio::test]
    async fn too_short_candidate_bodies_are_dropped() {
        let llm = Arc::new(
            FakeLlmClient::new()
                .with_prefix_match("You are reviewing", "FINDING: x")
                .with_prefix_match(
                    "You are revising",
                    "--- CANDIDATE 1 ---\n\
                     ok\n\
                     --- CANDIDATE 2 ---\n\
                     This one is long enough to survive the filter.",
                ),
        );
        let r = Reflector::new(llm, 2);
        let p = r
            .propose(&active("body"), &[outcome(false)], "test")
            .await
            .unwrap()
            .expect("should still produce — one candidate survives");
        assert_eq!(p.candidates.len(), 1, "the 2-char body must be dropped");
    }

    #[tokio::test]
    async fn non_system_prompt_kind_errors_explicitly() {
        let heuristic = PolicyArtifact {
            id: new_id(),
            version: 1,
            scope: Scope::global("t"),
            kind: ArtifactKind::Heuristic {
                when: "w".into(),
                then: "t".into(),
            },
            canaries: Vec::new(),
            time: BiTemporal::now(),
        };
        let r = Reflector::new(Arc::new(FakeLlmClient::new()), 2);
        let err = r
            .propose(&heuristic, &[outcome(false)], "test")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("SystemPrompt"));
    }

    // ----- helper trait so the test reads naturally -----

    trait SystemPromptCheck {
        fn kind_is_system_prompt_containing(&self, needle: &str) -> bool;
    }
    impl SystemPromptCheck for PolicyArtifact {
        fn kind_is_system_prompt_containing(&self, needle: &str) -> bool {
            matches!(&self.kind, ArtifactKind::SystemPrompt { body } if body.contains(needle))
        }
    }
}
