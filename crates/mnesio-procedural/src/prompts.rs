//! Prompt templates for the GEPA-style reflective loop.
//!
//! Two prompts, used in sequence:
//!
//! 1. [`reflection_prompt`] — given the current artifact and a batch of
//!    recent `Outcome`s, ask the LLM to identify failure patterns and
//!    suggest improvement directions. The output isn't actionable yet
//!    — it's diagnostic.
//! 2. [`proposal_prompt`] — given the reflection summary, ask the LLM
//!    to produce **K** concrete candidate revisions of the artifact
//!    body. Each candidate is a complete replacement (not a diff) so
//!    the parser stays simple.
//!
//! Both prompts demand strict output formats so the parsers in
//! [`crate::parse`] can extract structure without natural-language
//! ambiguity. The format is a small overhead in tokens but a large win
//! in reliability — real LLMs drift wildly when asked for free-form
//! suggestions.
//!
//! ## Anti-bypass discipline
//!
//! The proposal prompt explicitly tells the LLM what NOT to do: do not
//! delete canary-relevant behavior, do not weaken safety language, do
//! not propose changes to non-`SystemPrompt` artifacts (Slice B scope).
//! These are *hints* — the actual guarantee is the gate. The prompt
//! discipline just keeps the LLM from generating obviously-rejected
//! candidates and wasting eval budget.

use mnesio_core::entity::{ArtifactKind, Outcome, PolicyArtifact};

/// Build the reflection prompt. Embeds the current artifact body and
/// up to `max_outcomes` recent outcomes. The LLM is asked to return
/// findings in a strict bullet format.
pub fn reflection_prompt(artifact: &PolicyArtifact, outcomes: &[Outcome]) -> String {
    let body = artifact_body(artifact);
    let outcome_block = render_outcomes(outcomes, 8);
    format!(
        "You are reviewing a recent batch of agent outcomes under a fixed system prompt.\n\
         Your job is to identify failure patterns — recurring mistakes, missing instructions, \
         ambiguous wording. You are NOT proposing fixes yet; that's the next step.\n\
         \n\
         Current system prompt:\n\
         ---\n\
         {body}\n\
         ---\n\
         \n\
         Recent outcomes (success=true/false, score, error if any):\n\
         {outcome_block}\n\
         \n\
         Respond with one finding per line in this EXACT format:\n\
         FINDING: <one-sentence description of the pattern>\n\
         \n\
         Emit between 1 and 4 findings. If there are no clear patterns, emit:\n\
         FINDING: none"
    )
}

/// Build the proposal prompt. Embeds the current artifact body + the
/// reflection summary + an explicit count `k` of candidate revisions
/// to return.
///
/// Each candidate is a *complete* replacement for the system prompt
/// body. The parser splits on the `--- CANDIDATE N ---` sentinel so a
/// drifty LLM that includes prose before / between candidates can
/// still be parsed.
pub fn proposal_prompt(artifact: &PolicyArtifact, reflection_summary: &str, k: usize) -> String {
    let body = artifact_body(artifact);
    let k = k.clamp(1, 8);
    format!(
        "You are revising a system prompt to address the findings below. Produce {k} candidate \
         revisions. Each candidate must be a COMPLETE replacement for the current prompt, not a \
         diff.\n\
         \n\
         Hard constraints (the eval gate will reject anything that violates these):\n\
         - Do NOT remove safety language, refusals, or guardrails.\n\
         - Do NOT remove any explicit behaviour mentioned in the current prompt unless the \
           findings show it actively harms outcomes.\n\
         - Do keep the same general shape and length (within 50%).\n\
         \n\
         Current system prompt:\n\
         ---\n\
         {body}\n\
         ---\n\
         \n\
         Findings to address:\n\
         {reflection_summary}\n\
         \n\
         Respond with EXACTLY {k} candidates in this format, separated by the sentinel line. \
         Do not include any prose before, between, or after the candidates:\n\
         --- CANDIDATE 1 ---\n\
         <full revised system prompt body>\n\
         --- CANDIDATE 2 ---\n\
         <full revised system prompt body>\n\
         (and so on through CANDIDATE {k})"
    )
}

/// Extract the system-prompt body from an artifact. Returns an empty
/// string for non-`SystemPrompt` kinds — Slice B only supports prompts,
/// other kinds will land in later slices.
fn artifact_body(artifact: &PolicyArtifact) -> &str {
    match &artifact.kind {
        ArtifactKind::SystemPrompt { body } => body,
        _ => "",
    }
}

/// Render outcomes compactly. Truncates to `max` entries so a noisy
/// reflection batch doesn't blow the prompt budget.
fn render_outcomes(outcomes: &[Outcome], max: usize) -> String {
    if outcomes.is_empty() {
        return "(no outcomes in this batch)".into();
    }
    let mut s = String::new();
    for (i, o) in outcomes.iter().take(max).enumerate() {
        let success = match o.success {
            Some(true) => "success=true",
            Some(false) => "success=false",
            None => "success=?",
        };
        let score = o
            .scores
            .iter()
            .map(|(k, v)| format!("{k}={v:.2}"))
            .collect::<Vec<_>>()
            .join(", ");
        let err = o
            .error
            .as_deref()
            .map(|e| format!(" · error={e}"))
            .unwrap_or_default();
        s.push_str(&format!("  {i}. {success} · scores=[{score}]{err}\n"));
    }
    if outcomes.len() > max {
        s.push_str(&format!(
            "  ... ({} more truncated)\n",
            outcomes.len() - max
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::entity::JudgeSource;
    use mnesio_core::types::{new_id, ArtifactRef, BiTemporal, EpisodeRef, Scope, TrajectoryRef};
    use std::collections::HashMap;

    fn artifact(body: &str) -> PolicyArtifact {
        PolicyArtifact {
            id: new_id(),
            version: 1,
            scope: Scope::global("t"),
            kind: ArtifactKind::SystemPrompt { body: body.into() },
            canaries: Vec::new(),
            time: BiTemporal::now(),
        }
    }

    fn outcome(success: bool, error: Option<&str>) -> Outcome {
        let mut scores = HashMap::new();
        scores.insert("accuracy".into(), if success { 1.0 } else { 0.0 });
        Outcome {
            id: new_id(),
            episode: EpisodeRef(new_id()),
            artifacts_used: vec![ArtifactRef(new_id())],
            success: Some(success),
            scores,
            error: error.map(String::from),
            judge: JudgeSource::Environment,
            trajectory: TrajectoryRef(new_id()),
        }
    }

    #[test]
    fn reflection_prompt_embeds_artifact_and_outcomes() {
        let a = artifact("Always answer in one word.");
        let outs = vec![outcome(true, None), outcome(false, Some("too long"))];
        let p = reflection_prompt(&a, &outs);
        assert!(p.contains("Always answer in one word."));
        assert!(p.contains("FINDING:"), "must instruct on output format");
        assert!(p.contains("success=true"));
        assert!(p.contains("success=false"));
        assert!(p.contains("too long"));
    }

    #[test]
    fn reflection_prompt_with_empty_outcomes_says_so() {
        let p = reflection_prompt(&artifact("x"), &[]);
        assert!(p.contains("(no outcomes in this batch)"));
    }

    #[test]
    fn reflection_prompt_truncates_overlong_batches() {
        let outs: Vec<_> = (0..20).map(|_| outcome(true, None)).collect();
        let p = reflection_prompt(&artifact("x"), &outs);
        assert!(
            p.contains("more truncated"),
            "overlong batches must be truncated to keep prompt size bounded"
        );
    }

    #[test]
    fn proposal_prompt_demands_k_candidates_with_sentinels() {
        let p = proposal_prompt(&artifact("body"), "FINDING: too terse", 3);
        assert!(p.contains("EXACTLY 3"));
        assert!(p.contains("--- CANDIDATE 1 ---"));
        assert!(p.contains("--- CANDIDATE 2 ---"));
        assert!(p.contains("CANDIDATE 3"));
    }

    #[test]
    fn proposal_prompt_clamps_k_to_safe_range() {
        let p_high = proposal_prompt(&artifact("body"), "x", 999);
        // Clamp upper bound to 8 — anything more is silly.
        assert!(p_high.contains("EXACTLY 8"));
        let p_zero = proposal_prompt(&artifact("body"), "x", 0);
        // Clamp lower bound to 1 — proposing zero candidates is a no-op.
        assert!(p_zero.contains("EXACTLY 1"));
    }

    #[test]
    fn proposal_prompt_lists_anti_bypass_constraints() {
        let p = proposal_prompt(&artifact("body"), "x", 2);
        // These hints don't replace the gate but they save eval budget.
        assert!(p.contains("safety language"));
        assert!(p.contains("Do NOT remove"));
    }

    #[test]
    fn artifact_body_returns_empty_for_unsupported_kinds() {
        let a = PolicyArtifact {
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
        assert_eq!(artifact_body(&a), "");
    }
}
