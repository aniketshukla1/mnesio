//! [`ShadowEvaluator`] — turn a candidate artifact into an
//! [`EvalReport`] the gate can rule on.
//!
//! "Shadow" because the candidate never touches a real user / live
//! traffic — it runs against held-out inputs (canaries + replay
//! outcomes + safety probes) inside the compiler. Only after the gate
//! says yes does the active version get swapped (Slice C).
//!
//! ## What "running" an artifact means here
//!
//! For each input the [`PolicyExecutor`] produces an `actual` output.
//! Every judge in the panel then rules on `(input, expected, actual)`.
//! The shadow evaluator aggregates:
//!
//! - **Canaries** — every canary must be a *unanimous* pass (any
//!   judge dissenting → fail). Hard rule.
//! - **Replay** — success rate = fraction of replay inputs where the
//!   panel *majority* passed.
//! - **Safety probes** — every probe must be a *unanimous* pass. Same
//!   strictness as canaries; safety failures are the most expensive
//!   miss.
//! - **Objective Δ** — mean score on replay − baseline mean score on
//!   replay. The baseline is the active artifact run on the same
//!   inputs, also through the executor + judge panel.
//! - **Judges consulted** — number of *distinct* judge ids that
//!   contributed at least one verdict. Diversity gate input.
//!
//! ## Why majority for replay, unanimous for canary/safety
//!
//! Canaries and safety probes are explicit guardrails — the author
//! asserted "this must hold." Even one dissenting judge means the
//! candidate might be sliding off the guardrail; reject. Replay is
//! the long tail of "does it still mostly work" — judge noise on
//! individual replays is expected and shouldn't fail the gate by
//! itself. Majority across the panel filters that noise.

use crate::executor::PolicyExecutor;
use crate::judge::Judge;
use mnesio_core::entity::{Outcome, PolicyArtifact};
use mnesio_core::event::EvalReport;
use mnesio_core::MnesioError;
use std::collections::HashSet;
use std::sync::Arc;

/// Inputs the shadow evaluator needs to produce an [`EvalReport`].
/// Held separately so the same evaluator can run many candidates
/// against the same inputs (the GEPA "propose K, eval all" pattern)
/// without re-building everything.
pub struct ShadowInputs {
    /// Active artifact — the one being challenged. Drives the
    /// `objective_delta` baseline.
    pub baseline: PolicyArtifact,
    /// Held-out outcomes whose original inputs we replay. The `success`
    /// + `scores["objective"]` of the originals defines the baseline.
    pub replay: Vec<ReplayInput>,
    /// External safety probe set. Maintained outside the compiler —
    /// the compiler must never propose changes to it (report §10).
    pub safety_probes: Vec<SafetyProbe>,
}

/// One replay input. The `objective_baseline` is the score the active
/// artifact achieved on this input historically (extracted from the
/// `Outcome.scores` map under the agreed-upon key).
#[derive(Debug, Clone)]
pub struct ReplayInput {
    pub input: String,
    /// The expected/good answer hint passed to the judges.
    pub expected: String,
    /// Baseline objective score from the recorded `Outcome`. Used to
    /// compute Δ — candidate mean − baseline mean.
    pub objective_baseline: f32,
}

impl ReplayInput {
    /// Build a `ReplayInput` from an [`Outcome`], pulling the baseline
    /// score out of the `scores` map under `objective_key` (typically
    /// `"objective"` or `"accuracy"`). Returns `None` if the key isn't
    /// present — outcomes without a scored objective can't anchor a
    /// delta and the compiler should drop them.
    pub fn from_outcome(
        outcome: &Outcome,
        input: impl Into<String>,
        expected: impl Into<String>,
        objective_key: &str,
    ) -> Option<Self> {
        let score = *outcome.scores.get(objective_key)?;
        Some(Self {
            input: input.into(),
            expected: expected.into(),
            objective_baseline: score,
        })
    }
}

/// One safety probe — input the candidate must answer safely. The
/// judges score the `actual` against the `safe_expectation` description
/// (typically "refuses and explains why" or "answers without revealing
/// secrets").
#[derive(Debug, Clone)]
pub struct SafetyProbe {
    pub input: String,
    pub safe_expectation: String,
}

/// Pure-function shadow evaluator. Holds an [`Arc<dyn PolicyExecutor>`]
/// and a panel of judges; produces [`EvalReport`]s on demand.
pub struct ShadowEvaluator {
    executor: Arc<dyn PolicyExecutor>,
    judges: Vec<Arc<dyn Judge>>,
}

impl ShadowEvaluator {
    pub fn new(executor: Arc<dyn PolicyExecutor>, judges: Vec<Arc<dyn Judge>>) -> Self {
        Self { executor, judges }
    }

    /// Number of distinct judge ids in the panel. Surfaces into
    /// `EvalReport.judges_consulted` for the diversity gate.
    pub fn distinct_judge_ids(&self) -> usize {
        self.judges
            .iter()
            .map(|j| j.id().to_string())
            .collect::<HashSet<_>>()
            .len()
    }

    /// Run a candidate through the full shadow pipeline. The returned
    /// report is suitable for handing straight to
    /// [`crate::EvalGates::evaluate`].
    pub async fn evaluate(
        &self,
        candidate: &PolicyArtifact,
        inputs: &ShadowInputs,
    ) -> Result<EvalReport, MnesioError> {
        // -------- canaries --------
        let mut canaries_passed = 0u32;
        let canaries_total = candidate.canaries.len() as u32;
        for c in &candidate.canaries {
            if self.run_unanimous(candidate, &c.input, &c.expect).await? {
                canaries_passed += 1;
            }
        }

        // -------- replay --------
        let replay_total = inputs.replay.len();
        let mut replay_passes = 0usize;
        let mut candidate_objective_sum = 0.0f32;
        let mut baseline_objective_sum = 0.0f32;
        for r in &inputs.replay {
            let cand_out = self.executor.execute(candidate, &r.input).await?;
            let cand_score = self
                .panel_mean_score(&r.input, &r.expected, &cand_out)
                .await?;
            candidate_objective_sum += cand_score;
            baseline_objective_sum += r.objective_baseline;
            if self
                .panel_majority_pass(&r.input, &r.expected, &cand_out)
                .await?
            {
                replay_passes += 1;
            }
        }
        let replay_success_rate = if replay_total == 0 {
            // No replay = nothing to regress against. Treat as 1.0
            // (vacuously passing) rather than 0.0 — the canary +
            // safety floors still apply, so a no-replay run isn't a
            // get-out-of-jail-free card.
            1.0
        } else {
            replay_passes as f32 / replay_total as f32
        };
        let objective_delta = if replay_total == 0 {
            0.0
        } else {
            (candidate_objective_sum - baseline_objective_sum) / replay_total as f32
        };

        // -------- safety probes --------
        let mut safety_ok = true;
        for p in &inputs.safety_probes {
            if !self
                .run_unanimous(candidate, &p.input, &p.safe_expectation)
                .await?
            {
                safety_ok = false;
                // Keep iterating so we cover every probe — useful for
                // dashboard surfacing of which probe tripped, even
                // though we already know we'll reject. Cheap to do.
            }
        }

        Ok(EvalReport {
            canaries_passed,
            canaries_total,
            replay_success_rate,
            safety_probe_passed: safety_ok,
            objective_delta,
            judges_consulted: self.distinct_judge_ids() as u8,
        })
    }

    /// Run the executor + every judge on one input. True iff every
    /// judge in the panel ruled `passed`. Used for canaries + safety
    /// (strict gates).
    async fn run_unanimous(
        &self,
        candidate: &PolicyArtifact,
        input: &str,
        expected: &str,
    ) -> Result<bool, MnesioError> {
        let actual = self.executor.execute(candidate, input).await?;
        for j in &self.judges {
            let v = j.judge(input, expected, &actual).await?;
            if !v.passed {
                return Ok(false);
            }
        }
        // An empty panel is vacuously unanimous — but the diversity
        // gate will independently reject if `judges_consulted` is
        // below the threshold, so we don't need to second-guess it
        // here.
        Ok(true)
    }

    /// True iff the strict majority of the panel ruled `passed`. Used
    /// for replay (long-tail noise tolerance).
    async fn panel_majority_pass(
        &self,
        input: &str,
        expected: &str,
        actual: &str,
    ) -> Result<bool, MnesioError> {
        if self.judges.is_empty() {
            return Ok(false);
        }
        let mut passes = 0usize;
        for j in &self.judges {
            let v = j.judge(input, expected, actual).await?;
            if v.passed {
                passes += 1;
            }
        }
        Ok(passes * 2 > self.judges.len())
    }

    /// Mean numeric score across the panel. Used for the objective Δ.
    async fn panel_mean_score(
        &self,
        input: &str,
        expected: &str,
        actual: &str,
    ) -> Result<f32, MnesioError> {
        if self.judges.is_empty() {
            return Ok(0.0);
        }
        let mut sum = 0.0;
        for j in &self.judges {
            let v = j.judge(input, expected, actual).await?;
            sum += v.score;
        }
        Ok(sum / self.judges.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::FakePolicyExecutor;
    use crate::judge::{FakeJudge, JudgeVerdict};
    use mnesio_core::entity::{ArtifactKind, Canary};
    use mnesio_core::types::{new_id, BiTemporal, Scope};

    fn artifact(version: u32, canaries: Vec<Canary>) -> PolicyArtifact {
        PolicyArtifact {
            id: new_id(),
            version,
            scope: Scope::global("t"),
            kind: ArtifactKind::SystemPrompt {
                body: "ignored by FakePolicyExecutor".into(),
            },
            canaries,
            time: BiTemporal::now(),
        }
    }

    fn canary(input: &str, expect: &str) -> Canary {
        Canary {
            input: input.into(),
            expect: expect.into(),
        }
    }

    /// Minimal happy-path setup: 2 distinct judges, both always pass,
    /// executor returns the expected canary answer.
    fn happy_setup(candidate: &PolicyArtifact) -> ShadowEvaluator {
        let mut ex = FakePolicyExecutor::new().with_default("default");
        for c in &candidate.canaries {
            ex = ex.with_response(candidate.version, c.input.clone(), c.expect.clone());
        }
        ShadowEvaluator::new(
            Arc::new(ex),
            vec![
                Arc::new(FakeJudge::new("judge-a")),
                Arc::new(FakeJudge::new("judge-b")),
            ],
        )
    }

    #[tokio::test]
    async fn happy_path_produces_baseline_passing_report() {
        let cand = artifact(2, vec![canary("sky?", "blue"), canary("grass?", "green")]);
        let inputs = ShadowInputs {
            baseline: artifact(1, vec![]),
            replay: vec![],
            safety_probes: vec![],
        };
        let eval = happy_setup(&cand);
        let report = eval.evaluate(&cand, &inputs).await.unwrap();
        assert_eq!(report.canaries_passed, 2);
        assert_eq!(report.canaries_total, 2);
        assert!(report.safety_probe_passed);
        assert_eq!(report.replay_success_rate, 1.0);
        assert_eq!(report.objective_delta, 0.0);
        assert_eq!(report.judges_consulted, 2);
        assert!(report.is_committable(), "happy path must clear baseline");
    }

    #[tokio::test]
    async fn duplicate_judge_ids_count_as_one_distinct_judge() {
        // Diversity gate cares about UNIQUE ids — two `LlmJudge`s
        // wrapping the same model should not let an operator inflate
        // the count.
        let _cand = artifact(2, vec![]);
        let eval = ShadowEvaluator::new(
            Arc::new(FakePolicyExecutor::new()),
            vec![
                Arc::new(FakeJudge::new("same")),
                Arc::new(FakeJudge::new("same")),
                Arc::new(FakeJudge::new("different")),
            ],
        );
        assert_eq!(eval.distinct_judge_ids(), 2);
    }

    #[tokio::test]
    async fn one_dissenting_judge_fails_canary_strict_gate() {
        // Canaries demand unanimous panel agreement.
        let cand = artifact(2, vec![canary("sky?", "blue")]);
        let executor = FakePolicyExecutor::new().with_response(2, "sky?", "blue");
        let eval = ShadowEvaluator::new(
            Arc::new(executor),
            vec![
                Arc::new(FakeJudge::new("a")),
                Arc::new(FakeJudge::new("b").with_default(JudgeVerdict::fail("nope"))),
            ],
        );
        let inputs = ShadowInputs {
            baseline: artifact(1, vec![]),
            replay: vec![],
            safety_probes: vec![],
        };
        let report = eval.evaluate(&cand, &inputs).await.unwrap();
        assert_eq!(report.canaries_passed, 0, "one judge fails → canary fails");
        assert!(!report.is_committable());
    }

    #[tokio::test]
    async fn replay_uses_majority_not_unanimous() {
        // Three judges. Two pass, one fails → majority passes. Replay
        // success counts as passed.
        let cand = artifact(2, vec![]);
        let executor = FakePolicyExecutor::new().with_default("answer");
        let eval = ShadowEvaluator::new(
            Arc::new(executor),
            vec![
                Arc::new(FakeJudge::new("a")),
                Arc::new(FakeJudge::new("b")),
                Arc::new(FakeJudge::new("c").with_default(JudgeVerdict::fail("dissent"))),
            ],
        );
        let inputs = ShadowInputs {
            baseline: artifact(1, vec![]),
            replay: vec![ReplayInput {
                input: "anything".into(),
                expected: "anything".into(),
                objective_baseline: 0.5,
            }],
            safety_probes: vec![],
        };
        let report = eval.evaluate(&cand, &inputs).await.unwrap();
        assert_eq!(report.replay_success_rate, 1.0, "2/3 → majority pass");
    }

    #[tokio::test]
    async fn safety_probe_failure_is_strict() {
        let cand = artifact(2, vec![]);
        let executor = FakePolicyExecutor::new().with_default("unsafe answer");
        let eval = ShadowEvaluator::new(
            Arc::new(executor),
            vec![
                Arc::new(FakeJudge::new("a")),
                // One judge says the answer is unsafe — that's enough to fail.
                Arc::new(FakeJudge::new("b").with_verdict(
                    "dangerous q",
                    "unsafe answer",
                    JudgeVerdict::fail("unsafe"),
                )),
            ],
        );
        let inputs = ShadowInputs {
            baseline: artifact(1, vec![]),
            replay: vec![],
            safety_probes: vec![SafetyProbe {
                input: "dangerous q".into(),
                safe_expectation: "should refuse".into(),
            }],
        };
        let report = eval.evaluate(&cand, &inputs).await.unwrap();
        assert!(
            !report.safety_probe_passed,
            "any judge dissenting → probe fails"
        );
        assert!(!report.is_committable());
    }

    #[tokio::test]
    async fn objective_delta_reflects_candidate_minus_baseline_mean() {
        // Two replay inputs; baseline scored 0.5 each; candidate scores
        // 1.0 each (judges always pass) — Δ should be +0.5.
        let cand = artifact(2, vec![]);
        let executor = FakePolicyExecutor::new().with_default("good");
        let eval = ShadowEvaluator::new(Arc::new(executor), vec![Arc::new(FakeJudge::new("a"))]);
        let inputs = ShadowInputs {
            baseline: artifact(1, vec![]),
            replay: vec![
                ReplayInput {
                    input: "x".into(),
                    expected: "x".into(),
                    objective_baseline: 0.5,
                },
                ReplayInput {
                    input: "y".into(),
                    expected: "y".into(),
                    objective_baseline: 0.5,
                },
            ],
            safety_probes: vec![],
        };
        let report = eval.evaluate(&cand, &inputs).await.unwrap();
        assert!(
            (report.objective_delta - 0.5).abs() < 1e-6,
            "Δ = 1.0 − 0.5 = +0.5, got {}",
            report.objective_delta
        );
    }

    #[tokio::test]
    async fn empty_inputs_produce_vacuously_passing_report() {
        // No canaries, no replay, no safety probes — strictest interp
        // says everything-vacuous-counts, so the report passes baseline.
        // The diversity gate still applies (judges_consulted from the
        // panel size).
        let cand = artifact(2, vec![]);
        let eval = ShadowEvaluator::new(
            Arc::new(FakePolicyExecutor::new()),
            vec![Arc::new(FakeJudge::new("a")), Arc::new(FakeJudge::new("b"))],
        );
        let inputs = ShadowInputs {
            baseline: artifact(1, vec![]),
            replay: vec![],
            safety_probes: vec![],
        };
        let report = eval.evaluate(&cand, &inputs).await.unwrap();
        assert!(report.is_committable());
        assert_eq!(report.judges_consulted, 2);
    }
}
