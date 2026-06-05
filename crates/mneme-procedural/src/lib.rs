//! # mneme-procedural
//!
//! The procedural-memory compiler — **the wedge**. Turns batches of
//! [`mneme_core::Outcome`]s into improved, versioned
//! [`mneme_core::PolicyArtifact`]s via a GEPA-style (arXiv:2507.19457)
//! reflective loop: reflect → propose K candidates → shadow-evaluate →
//! Pareto-select → **gated commit**.
//!
//! ## The hard rule, encoded
//!
//! Hard Rule #1 (CLAUDE.md): *nothing procedural commits without passing
//! [`mneme_core::EvalReport::is_committable`].* This crate enforces it
//! mechanically:
//!
//! - [`mneme_core::EvalReport::is_committable`] is the non-bypassable
//!   strict baseline (canary 100%, safety probe, Δ ≥ 0).
//! - [`gate::EvalGates::evaluate`] **always** consults the baseline and
//!   emits [`gate::RejectReason::BaselineFailed`] if it trips, *before*
//!   evaluating any configurable thresholds. Setting every configurable
//!   knob to its weakest value still cannot bypass the baseline; a
//!   dedicated test holds that property at every commit.
//!
//! ## Slice progress
//!
//! - **Slice A — types + gate.** [`gate::EvalGates`], [`gate::Verdict`],
//!   [`gate::RejectReason`], [`proposal::Proposal`]. ✅
//! - **Slice B — reflective loop.** [`prompts`] + [`parse`] +
//!   [`executor::PolicyExecutor`] + [`judge::Judge`] +
//!   [`shadow::ShadowEvaluator`] + [`reflect::Reflector`]. The
//!   [`ProceduralCompiler::compile_with_inputs`] entry point wires them
//!   together — reflect → propose K → shadow-eval each → call the gate
//!   → return the first committable verdict. NO commit yet (that's
//!   Slice C). ✅ this slice.
//! - **Slice C — atomic active-version hot-swap.** Single-writer commit
//!   path that emits the `ProceduralCommitted` event, performs the
//!   pointer swap, and is reconstructible by log replay. Pending.
//! - **Slice D — ALFWorld-style eval suite.** Small, runnable harness
//!   demonstrating positive learning curve with zero safety-probe
//!   regression. The Phase-2 "done when" criterion. Pending.

pub mod curve;
pub mod eval;
pub mod executor;
pub mod gate;
pub mod judge;
pub mod parse;
pub mod prompts;
pub mod proposal;
pub mod reflect;
pub mod shadow;
pub mod store;
pub mod worker;

pub use curve::{LearningCurveCollector, LearningCurvePoint};
pub use eval::{BenchmarkTask, EvalSafetyProbe, EvalSuite, SuiteReport, TaskKind, TaskResult};
pub use executor::{FakePolicyExecutor, LlmExecutor, PolicyExecutor};
pub use gate::{EvalGates, RejectReason, Verdict};
pub use judge::{FakeJudge, Judge, JudgeVerdict};
pub use proposal::Proposal;
pub use reflect::Reflector;
pub use shadow::{ShadowEvaluator, ShadowInputs};
pub use store::ProceduralStore;
pub use worker::{spawn, EvalBinding, ProceduralWorker};

use mneme_core::entity::{Outcome, PolicyArtifact};
use mneme_core::event::EvalReport;
use mneme_core::MnemeError;
use std::sync::Arc;

/// What triggers a compile pass: a count- or time-based outcome batch.
#[derive(Debug, Clone)]
pub struct CompileTrigger {
    pub min_batch: usize,
    pub max_age_secs: u64,
}

impl Default for CompileTrigger {
    fn default() -> Self {
        Self {
            min_batch: 32,
            max_age_secs: 3600,
        }
    }
}

/// One row of compile output — the candidate, its shadow report, and
/// the gate's verdict on it. The compiler returns one per candidate so
/// the caller can show the *whole* eval matrix on the dashboard, not
/// just the winning row.
pub struct CandidateOutcome {
    pub candidate: PolicyArtifact,
    pub report: EvalReport,
    pub verdict: Verdict,
}

impl CandidateOutcome {
    pub fn committable(&self) -> bool {
        self.verdict.committable
    }
}

/// Full compile-pass result. Includes the upstream [`Proposal`] (for
/// audit), every candidate's outcome (so the dashboard can show what
/// was tried), and a convenience pointer to the winning row (the first
/// committable one — Pareto-selection across multiple committable
/// candidates lands in a follow-up slice).
pub struct CompileResult {
    pub proposal: Proposal,
    pub outcomes: Vec<CandidateOutcome>,
    /// Index into `outcomes` of the first committable candidate, if any.
    pub winner_idx: Option<usize>,
}

impl CompileResult {
    /// Convenience: did any candidate clear the gate?
    pub fn has_winner(&self) -> bool {
        self.winner_idx.is_some()
    }

    /// Borrow the winning candidate's outcome if one exists.
    pub fn winner(&self) -> Option<&CandidateOutcome> {
        self.winner_idx.and_then(|i| self.outcomes.get(i))
    }
}

/// The compiler. Composes a [`Reflector`] (LLM-driven candidate
/// generation) with a [`ShadowEvaluator`] (policy execution + judge
/// panel) and gates every candidate through [`EvalGates`].
///
/// Slice B builds the pipeline; Slice C will add the commit path that
/// emits `ProceduralCommitted` and atomically swaps the active version.
pub struct ProceduralCompiler {
    pub trigger: CompileTrigger,
    /// Configurable gate thresholds. Defaults to [`EvalGates::default`]
    /// — strict canaries, safety probe required, judge diversity = 2.
    pub gates: EvalGates,
    /// GEPA-style "propose K candidates" — number of revisions the
    /// reflector generates per pass.
    pub candidates_per_artifact: usize,
    reflector: Reflector,
    shadow: ShadowEvaluator,
}

impl ProceduralCompiler {
    /// Construct a compiler from its component pieces. The executor +
    /// judges build the shadow evaluator; the LLM client + k drive the
    /// reflector.
    pub fn new(
        llm: Arc<dyn mneme_core::LlmClient>,
        executor: Arc<dyn PolicyExecutor>,
        judges: Vec<Arc<dyn Judge>>,
        candidates_per_artifact: usize,
    ) -> Self {
        Self {
            trigger: CompileTrigger::default(),
            gates: EvalGates::default(),
            candidates_per_artifact,
            reflector: Reflector::new(llm, candidates_per_artifact),
            shadow: ShadowEvaluator::new(executor, judges),
        }
    }

    /// Override the gate thresholds. Builder-style for ergonomic
    /// chained construction.
    pub fn with_gates(mut self, gates: EvalGates) -> Self {
        self.gates = gates;
        self
    }

    /// Run one compile pass against a pre-assembled batch of inputs.
    /// This is the **end-to-end pipeline** the rest of Slice B builds
    /// up to: reflect → propose K → shadow-eval each → gate each →
    /// return all outcomes plus a pointer to the first committable
    /// one.
    ///
    /// Returns `Ok(None)` if the reflector decided there was nothing
    /// to propose (no findings / no parseable candidates). Returns
    /// `Err` only for LLM / executor errors — gate rejections are
    /// `Ok(Some(_))` with `winner_idx == None`.
    pub async fn compile_with_inputs(
        &self,
        active: &PolicyArtifact,
        outcomes: &[Outcome],
        shadow_inputs: &ShadowInputs,
        rationale: impl Into<String>,
    ) -> Result<Option<CompileResult>, MnemeError> {
        // ---- reflect → propose ----
        let proposal = match self.reflector.propose(active, outcomes, rationale).await? {
            Some(p) => p,
            None => {
                tracing::debug!("compile: reflector returned no proposal");
                return Ok(None);
            }
        };

        // ---- shadow-eval every candidate, gate every report ----
        let mut outcomes_vec: Vec<CandidateOutcome> = Vec::with_capacity(proposal.candidates.len());
        for candidate in &proposal.candidates {
            let report = self.shadow.evaluate(candidate, shadow_inputs).await?;
            let verdict = self.gates.evaluate(&report);
            outcomes_vec.push(CandidateOutcome {
                candidate: candidate.clone(),
                report,
                verdict,
            });
        }

        // ---- Pareto selection ----
        // Across the committable subset, pick the lexicographic winner
        // on (objective_delta, replay_success_rate). Both objectives
        // are higher-is-better. Deterministic tie-break: identical
        // (Δ, replay) pairs → lower index wins. This is the "P" in
        // GEPA — propose K, pick the Pareto-best.
        let winner_idx = pareto_winner_idx(&outcomes_vec);

        Ok(Some(CompileResult {
            proposal,
            outcomes: outcomes_vec,
            winner_idx,
        }))
    }

    /// Apply a `CompileResult` to the event log. The semantics depend
    /// on whether the result has a winner:
    ///
    /// - **winner present** → emit `ProceduralProposed` (with the
    ///   winning candidate placed at index 0 so `ProceduralStore` picks
    ///   it up) followed by `ProceduralCommitted` carrying the winner's
    ///   `EvalReport`. These two events together are the atomic hot-
    ///   swap: replay sees the proposal, caches the winner, then the
    ///   commit makes it active.
    /// - **no winner** → emit `ProceduralProposed` followed by
    ///   `ProceduralRejected` with the structured rejection reasons
    ///   joined into the reason string. The proposal is recorded for
    ///   audit, then explicitly rejected.
    ///
    /// Both paths emit `ProceduralProposed` so the audit log is
    /// complete: every attempt is recorded, win or lose. Replay
    /// reconstructs the active set + the full proposal history.
    ///
    /// Returns the `ProposalId` so the caller can correlate with their
    /// upstream tracking.
    pub async fn apply(
        &self,
        log: &dyn mneme_core::EventLog,
        result: CompileResult,
    ) -> Result<mneme_core::types::ProposalId, MnemeError> {
        let proposal_id = result.proposal.id;
        let (artifacts_ordered, decision) = match result.winner_idx {
            Some(idx) => {
                // Place the winner first so ProceduralStore.absorb picks
                // it up — Slice C carries no explicit winner index on
                // the wire, so "winner is index 0" is the convention.
                let mut artifacts = result.proposal.candidates.clone();
                if idx > 0 {
                    artifacts.swap(0, idx);
                }
                let winning = &result.outcomes[idx];
                (artifacts, ApplyDecision::Commit(winning.report.clone()))
            }
            None => {
                // No winner — keep candidate order as proposed; the
                // proposal event still records what was tried.
                let reason = render_rejection_reasons(&result.outcomes);
                (
                    result.proposal.candidates.clone(),
                    ApplyDecision::Reject(reason),
                )
            }
        };

        log.append(mneme_core::event::Event::ProceduralProposed {
            proposal: proposal_id,
            artifacts: artifacts_ordered,
        })
        .await?;

        match decision {
            ApplyDecision::Commit(report) => {
                log.append(mneme_core::event::Event::ProceduralCommitted {
                    proposal: proposal_id,
                    report,
                })
                .await?;
            }
            ApplyDecision::Reject(reason) => {
                log.append(mneme_core::event::Event::ProceduralRejected {
                    proposal: proposal_id,
                    reason,
                })
                .await?;
            }
        }
        Ok(proposal_id)
    }
}

enum ApplyDecision {
    Commit(mneme_core::event::EvalReport),
    Reject(String),
}

/// Pareto-front winner selection across the committable subset of a
/// `CompileResult`'s candidate outcomes. Uses lexicographic ordering
/// on `(objective_delta, replay_success_rate)` — both higher-is-better
/// — for a deterministic, easy-to-explain tie-break across the front.
///
/// Why lexicographic and not weighted sum? The compiler is trying to
/// improve the headline objective; replay is a tail-protection tie-
/// breaker, not a weighted partner. Weighted sums hide the operator
/// preference inside a coefficient; lex ordering surfaces it.
///
/// Why not dominance-only? In the rare case multiple candidates are
/// mutually non-dominated AND tied on the lex order (same Δ, same
/// replay), we want a single deterministic answer. Lex selects the
/// lowest index, matching GEPA's "propose K, pick one" contract.
///
/// Returns `None` if no candidate is committable.
fn pareto_winner_idx(outcomes: &[CandidateOutcome]) -> Option<usize> {
    outcomes
        .iter()
        .enumerate()
        .filter(|(_, o)| o.verdict.committable)
        .max_by(|(a_i, a), (b_i, b)| {
            // Higher objective_delta wins; ties go to higher
            // replay_success_rate; further ties go to lower index
            // (i.e. earlier-proposed candidate, since we reverse the
            // index comparison so `max_by` picks the smaller index).
            a.report
                .objective_delta
                .partial_cmp(&b.report.objective_delta)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.report
                        .replay_success_rate
                        .partial_cmp(&b.report.replay_success_rate)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| b_i.cmp(a_i))
        })
        .map(|(i, _)| i)
}

/// Format the per-candidate rejection reasons into a single line for
/// the `ProceduralRejected.reason` field. The structured `Verdict`
/// objects don't go on the wire individually — the dashboard re-derives
/// them by walking proposals and re-running the gate when it needs
/// details. The string form is the human-readable audit trail.
fn render_rejection_reasons(outcomes: &[CandidateOutcome]) -> String {
    if outcomes.is_empty() {
        return "no candidates evaluated".into();
    }
    let mut parts: Vec<String> = Vec::new();
    for (i, o) in outcomes.iter().enumerate() {
        let names: Vec<&'static str> = o
            .verdict
            .reasons
            .iter()
            .map(|r| match r {
                RejectReason::BaselineFailed => "baseline",
                RejectReason::NoCanaries => "no_canaries",
                RejectReason::CanariesFailing { .. } => "canaries",
                RejectReason::SafetyProbeFailed => "safety",
                RejectReason::ObjectiveRegression { .. } => "objective",
                RejectReason::ReplayRegression { .. } => "replay",
                RejectReason::InsufficientJudges { .. } => "judges",
            })
            .collect();
        parts.push(format!("c{i}=[{}]", names.join(",")));
    }
    parts.join(" ")
}

#[cfg(test)]
mod pareto_tests {
    use super::*;
    use mneme_core::entity::{ArtifactKind, PolicyArtifact};
    use mneme_core::event::EvalReport;
    use mneme_core::types::{new_id, BiTemporal, Scope};

    fn artifact() -> PolicyArtifact {
        PolicyArtifact {
            id: new_id(),
            version: 1,
            scope: Scope::global("t"),
            kind: ArtifactKind::SystemPrompt { body: "x".into() },
            canaries: vec![],
            time: BiTemporal::now(),
        }
    }

    fn outcome(committable: bool, delta: f32, replay: f32) -> CandidateOutcome {
        CandidateOutcome {
            candidate: artifact(),
            report: EvalReport {
                canaries_passed: 0,
                canaries_total: 0,
                replay_success_rate: replay,
                safety_probe_passed: true,
                objective_delta: delta,
                judges_consulted: 2,
            },
            verdict: Verdict {
                committable,
                reasons: vec![],
            },
        }
    }

    #[test]
    fn picks_highest_objective_delta() {
        let outcomes = vec![
            outcome(true, 0.1, 1.0),
            outcome(true, 0.3, 1.0),
            outcome(true, 0.2, 1.0),
        ];
        assert_eq!(pareto_winner_idx(&outcomes), Some(1));
    }

    #[test]
    fn breaks_delta_tie_with_replay_success_rate() {
        let outcomes = vec![
            outcome(true, 0.2, 0.96),
            outcome(true, 0.2, 0.99), // same Δ, higher replay → wins
            outcome(true, 0.2, 0.97),
        ];
        assert_eq!(pareto_winner_idx(&outcomes), Some(1));
    }

    #[test]
    fn breaks_full_tie_with_lower_index() {
        let outcomes = vec![
            outcome(true, 0.2, 0.99),
            outcome(true, 0.2, 0.99), // identical → earlier index wins
            outcome(true, 0.2, 0.99),
        ];
        assert_eq!(pareto_winner_idx(&outcomes), Some(0));
    }

    #[test]
    fn skips_non_committable_even_with_better_scores() {
        // Highest Δ but verdict.committable=false → cannot win.
        let outcomes = vec![
            outcome(false, 0.9, 1.0),
            outcome(true, 0.1, 0.98),
            outcome(true, 0.2, 0.97),
        ];
        assert_eq!(pareto_winner_idx(&outcomes), Some(2));
    }

    #[test]
    fn none_when_no_committable_candidates() {
        let outcomes = vec![outcome(false, 0.5, 1.0), outcome(false, 0.3, 1.0)];
        assert!(pareto_winner_idx(&outcomes).is_none());
    }

    #[test]
    fn empty_outcomes_is_none() {
        assert!(pareto_winner_idx(&[]).is_none());
    }
}
