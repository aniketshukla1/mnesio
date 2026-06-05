//! Counterfactual contribution scoring by leave-one-out (LOO) ablation.
//!
//! The one primitive is [`CounterfactualEvaluator::evaluate`]: it returns an
//! aggregate objective in `[0,1]` when a given set of memories is *masked*
//! (made invisible to retrieval). The baseline is `evaluate(&{})`. A memory's
//! contribution is `baseline − evaluate(&{memory})`:
//!
//! - **> 0** — removing it *drops* outcomes; load-bearing, keep.
//! - **≈ 0** — removing it changes nothing; dead weight, GC candidate.
//! - **< 0** — removing it *improves* outcomes; actively misleading, also a GC
//!   (and a Phase-14 "negative memory" suppression) candidate.

use async_trait::async_trait;
use mneme_core::types::MemoryRef;
use mneme_core::MnemeError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

/// The seam (Hard Rule #7). The real impl wires the retriever + executor over
/// a held-out outcome/task set and masks memories by id; [`FakeEvaluator`]
/// makes the engine's tests hermetic.
#[async_trait]
pub trait CounterfactualEvaluator: Send + Sync {
    /// Aggregate objective in `[0,1]` (recall@k, benchmark pass-rate, …) with
    /// `masked` memories excluded from retrieval. `evaluate(&{})` is the
    /// baseline against which contributions are measured.
    async fn evaluate(&self, masked: &HashSet<MemoryRef>) -> Result<f32, MnemeError>;
}

/// How contribution is attributed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreMode {
    /// Leave-one-out: mask exactly one memory at a time. Cheap (`O(corpus)`
    /// evals) but underestimates contribution under redundancy — two memories
    /// that cover the same fact each score ≈ 0 because the other still covers.
    LeaveOneOut,
    /// Greedy ablation: iteratively ablate the **least-valuable remaining**
    /// memory (the one whose removal leaves the highest score) and re-score,
    /// attributing each memory its *marginal* drop at the point it's removed.
    ///
    /// This recovers redundant-set contribution that LOO misses: once the
    /// first of a redundant pair is masked, the second becomes load-bearing
    /// and earns the set's full contribution at its removal step. It's a
    /// single greedy permutation of the Shapley value — exact on additive
    /// objectives, and on the *aggregate* of a redundant set, at the cost of
    /// `O(corpus²)` evals (still bounded by `max_candidates`, Hard Rule #6).
    GreedyAblation,
}

/// Bounds for one scoring pass. The engine never schedules itself; the caller
/// runs passes offline, off the write path (Hard Rules #5/#6).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalConfig {
    /// Hard cap on candidates scored per pass — the cascade bound (#6).
    pub max_candidates: usize,
    /// Contributions with magnitude `≤ epsilon` count as zero (float noise +
    /// "no measurable effect" tolerance). Drives `gc_candidates`.
    pub epsilon: f32,
    pub mode: ScoreMode,
}

impl Default for CausalConfig {
    fn default() -> Self {
        Self {
            max_candidates: 256,
            epsilon: 1e-4,
            mode: ScoreMode::LeaveOneOut,
        }
    }
}

/// One memory's measured contribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryContribution {
    pub memory: MemoryRef,
    /// `baseline − masked_score`. Positive = helpful, ~0 = inert, negative =
    /// harmful.
    pub contribution: f32,
    /// The objective when this memory alone was masked.
    pub masked_score: f32,
}

/// The result of one scoring pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionReport {
    /// `evaluate(&{})` — the objective with nothing masked.
    pub baseline_score: f32,
    pub mode: ScoreMode,
    /// One entry per scored candidate, in input order.
    pub scored: Vec<MemoryContribution>,
    /// How many candidates the caller supplied (may exceed `candidates_scored`
    /// when `max_candidates` clamps the pass).
    pub candidates_considered: usize,
    /// How many were actually scored this pass.
    pub candidates_scored: usize,
}

impl ContributionReport {
    /// Memories safe to retire: contribution at or below `epsilon` (inert *or*
    /// harmful). These are the *provable* zero/negative contributors — GC by
    /// measurement, not by age.
    pub fn gc_candidates(&self, epsilon: f32) -> Vec<MemoryRef> {
        self.scored
            .iter()
            .filter(|c| c.contribution <= epsilon)
            .map(|c| c.memory)
            .collect()
    }

    /// Load-bearing memories: contribution at or above `min_contribution`.
    pub fn high_contributors(&self, min_contribution: f32) -> Vec<&MemoryContribution> {
        self.scored
            .iter()
            .filter(|c| c.contribution >= min_contribution)
            .collect()
    }

    /// A copy of the scores sorted by contribution, highest first — for the
    /// dashboard panel ("what's actually carrying the corpus").
    pub fn ranked(&self) -> Vec<MemoryContribution> {
        let mut v = self.scored.clone();
        v.sort_by(|a, b| {
            b.contribution
                .partial_cmp(&a.contribution)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }
}

/// Leave-one-out contribution scorer.
pub struct ContributionScorer {
    cfg: CausalConfig,
}

impl ContributionScorer {
    pub fn new(cfg: CausalConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &CausalConfig {
        &self.cfg
    }

    /// Score `candidates` against `evaluator`, dispatching on the configured
    /// [`ScoreMode`]. The caller is responsible for scoping `candidates`
    /// (Hard Rule #3) — the scorer is scope-agnostic.
    pub async fn score(
        &self,
        evaluator: &dyn CounterfactualEvaluator,
        candidates: &[MemoryRef],
    ) -> Result<ContributionReport, MnemeError> {
        match self.cfg.mode {
            ScoreMode::LeaveOneOut => self.score_loo(evaluator, candidates).await,
            ScoreMode::GreedyAblation => self.score_greedy(evaluator, candidates).await,
        }
    }

    /// Leave-one-out: one baseline + one masked eval per candidate. Issues
    /// exactly `1 + min(candidates.len(), max_candidates)` evaluator calls.
    async fn score_loo(
        &self,
        evaluator: &dyn CounterfactualEvaluator,
        candidates: &[MemoryRef],
    ) -> Result<ContributionReport, MnemeError> {
        let baseline = evaluator.evaluate(&HashSet::new()).await?;
        let mut scored = Vec::new();
        for memory in candidates.iter().take(self.cfg.max_candidates) {
            let mut masked = HashSet::with_capacity(1);
            masked.insert(*memory);
            let masked_score = evaluator.evaluate(&masked).await?;
            scored.push(MemoryContribution {
                memory: *memory,
                contribution: baseline - masked_score,
                masked_score,
            });
        }
        let candidates_scored = scored.len();
        Ok(ContributionReport {
            baseline_score: baseline,
            mode: self.cfg.mode,
            scored,
            candidates_considered: candidates.len(),
            candidates_scored,
        })
    }

    /// Greedy ablation: repeatedly ablate the least-valuable remaining memory
    /// (highest score when added to the masked set) and attribute it the
    /// marginal drop at that step. Accumulates the masked set across steps so
    /// redundant memories earn the set's contribution once the cover is gone.
    ///
    /// `scored` is returned in **ablation order** (least-contribution first);
    /// `masked_score` is the cumulative score after that memory was ablated.
    /// Cost: `1 + N(N+1)/2` evals for `N = min(candidates, max_candidates)`.
    async fn score_greedy(
        &self,
        evaluator: &dyn CounterfactualEvaluator,
        candidates: &[MemoryRef],
    ) -> Result<ContributionReport, MnemeError> {
        let baseline = evaluator.evaluate(&HashSet::new()).await?;
        let mut remaining: Vec<MemoryRef> = candidates
            .iter()
            .take(self.cfg.max_candidates)
            .copied()
            .collect();
        let mut masked: HashSet<MemoryRef> = HashSet::with_capacity(remaining.len());
        let mut current = baseline;
        let mut scored: Vec<MemoryContribution> = Vec::with_capacity(remaining.len());

        while !remaining.is_empty() {
            // Pick the memory whose ablation leaves the highest score — i.e.
            // the least-valuable one to remove next.
            let mut best_idx = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for (i, m) in remaining.iter().enumerate() {
                masked.insert(*m);
                let trial = evaluator.evaluate(&masked).await?;
                masked.remove(m);
                if trial > best_score {
                    best_score = trial;
                    best_idx = i;
                }
            }
            let chosen = remaining.swap_remove(best_idx);
            let contribution = current - best_score;
            masked.insert(chosen);
            current = best_score;
            scored.push(MemoryContribution {
                memory: chosen,
                contribution,
                masked_score: best_score,
            });
        }
        let candidates_scored = scored.len();
        Ok(ContributionReport {
            baseline_score: baseline,
            mode: self.cfg.mode,
            scored,
            candidates_considered: candidates.len(),
            candidates_scored,
        })
    }
}

/// Deterministic, dependency-free evaluator for tests + offline modelling.
///
/// Models an **additive** objective: `evaluate(masked) = baseline − Σ
/// marginal(m)` clamped to `[0,1]`. Under this model LOO recovers each memory's
/// marginal exactly, which is what lets the tests assert the scorer's contract
/// precisely. Real corpora are non-additive (the redundancy gap), which is why
/// the production evaluator wires the actual retriever.
pub struct FakeEvaluator {
    baseline: f32,
    marginals: HashMap<MemoryRef, f32>,
    calls: AtomicUsize,
}

impl FakeEvaluator {
    pub fn new(baseline: f32) -> Self {
        Self {
            baseline,
            marginals: HashMap::new(),
            calls: AtomicUsize::new(0),
        }
    }

    /// Set how much the objective drops when `memory` is present and then
    /// masked. Positive = helpful memory; `0.0` = inert; negative = harmful.
    pub fn with_marginal(mut self, memory: MemoryRef, marginal: f32) -> Self {
        self.marginals.insert(memory, marginal);
        self
    }

    /// Number of `evaluate` calls so far — lets tests assert the cascade bound.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl CounterfactualEvaluator for FakeEvaluator {
    async fn evaluate(&self, masked: &HashSet<MemoryRef>) -> Result<f32, MnemeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let removed: f32 = masked.iter().filter_map(|m| self.marginals.get(m)).sum();
        Ok((self.baseline - removed).clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mneme_core::types::new_id;

    fn mref() -> MemoryRef {
        MemoryRef(new_id())
    }

    fn scorer() -> ContributionScorer {
        ContributionScorer::new(CausalConfig::default())
    }

    #[tokio::test]
    async fn loo_recovers_per_memory_marginals() {
        let (helpful, inert, harmful) = (mref(), mref(), mref());
        let eval = FakeEvaluator::new(0.8)
            .with_marginal(helpful, 0.4)
            .with_marginal(inert, 0.0)
            .with_marginal(harmful, -0.2);
        let report = scorer()
            .score(&eval, &[helpful, inert, harmful])
            .await
            .unwrap();

        assert!((report.baseline_score - 0.8).abs() < 1e-6);
        let by = |m: MemoryRef| {
            report
                .scored
                .iter()
                .find(|c| c.memory == m)
                .unwrap()
                .contribution
        };
        assert!((by(helpful) - 0.4).abs() < 1e-6, "helpful ≈ +0.4");
        assert!(by(inert).abs() < 1e-6, "inert ≈ 0");
        assert!((by(harmful) + 0.2).abs() < 1e-6, "harmful ≈ -0.2");
    }

    #[tokio::test]
    async fn gc_candidates_pick_inert_and_harmful_not_load_bearing() {
        let (helpful, inert, harmful) = (mref(), mref(), mref());
        let eval = FakeEvaluator::new(0.8)
            .with_marginal(helpful, 0.4)
            .with_marginal(inert, 0.0)
            .with_marginal(harmful, -0.2);
        let report = scorer()
            .score(&eval, &[helpful, inert, harmful])
            .await
            .unwrap();

        let gc = report.gc_candidates(scorer().config().epsilon);
        assert!(gc.contains(&inert), "inert is dead weight → GC");
        assert!(gc.contains(&harmful), "harmful improves on removal → GC");
        assert!(!gc.contains(&helpful), "load-bearing memory must be kept");

        let keep = report.high_contributors(0.3);
        assert_eq!(keep.len(), 1);
        assert_eq!(keep[0].memory, helpful);
    }

    #[tokio::test]
    async fn max_candidates_bounds_the_pass() {
        let cands: Vec<MemoryRef> = (0..5).map(|_| mref()).collect();
        let eval = FakeEvaluator::new(0.5);
        let cfg = CausalConfig {
            max_candidates: 2,
            ..CausalConfig::default()
        };
        let report = ContributionScorer::new(cfg)
            .score(&eval, &cands)
            .await
            .unwrap();

        assert_eq!(report.candidates_considered, 5);
        assert_eq!(report.candidates_scored, 2);
        // 1 baseline + 2 masked evals — the cascade bound (#6) held.
        assert_eq!(eval.calls(), 3);
    }

    // The two halves of the Phase-10 "done when", at the evaluator level:
    #[tokio::test]
    async fn done_when_zero_contributor_masking_does_not_move_the_score() {
        let inert = mref();
        let eval = FakeEvaluator::new(0.7).with_marginal(inert, 0.0);
        let baseline = eval.evaluate(&HashSet::new()).await.unwrap();
        let masked = eval.evaluate(&HashSet::from([inert])).await.unwrap();
        assert!(
            (baseline - masked).abs() < 1e-6,
            "pruning inert moves nothing"
        );
    }

    #[tokio::test]
    async fn done_when_high_contributor_removal_drops_outcomes() {
        let load_bearing = mref();
        let eval = FakeEvaluator::new(0.9).with_marginal(load_bearing, 0.5);
        let baseline = eval.evaluate(&HashSet::new()).await.unwrap();
        let masked = eval.evaluate(&HashSet::from([load_bearing])).await.unwrap();
        assert!(
            masked < baseline - 0.4,
            "removing it measurably drops the score"
        );
    }

    #[tokio::test]
    async fn ranked_orders_by_contribution_desc() {
        let (a, b, c) = (mref(), mref(), mref());
        let eval = FakeEvaluator::new(0.6)
            .with_marginal(a, 0.1)
            .with_marginal(b, 0.5)
            .with_marginal(c, 0.3);
        let report = scorer().score(&eval, &[a, b, c]).await.unwrap();
        let ranked = report.ranked();
        assert_eq!(ranked[0].memory, b);
        assert_eq!(ranked[1].memory, c);
        assert_eq!(ranked[2].memory, a);
    }

    // --- greedy ablation -------------------------------------------------

    /// A *redundant* capability: the objective gains `group_value` iff at least
    /// one group member is unmasked. Masking a strict subset loses nothing;
    /// masking the whole group loses `group_value`. This is the non-additive
    /// case LOO can't score (each member ≈ 0) but greedy ablation can.
    struct CoverageEvaluator {
        baseline: f32,
        group: HashSet<MemoryRef>,
        group_value: f32,
    }

    impl CoverageEvaluator {
        fn new(baseline: f32, group: &[MemoryRef], group_value: f32) -> Self {
            Self {
                baseline,
                group: group.iter().copied().collect(),
                group_value,
            }
        }
    }

    #[async_trait]
    impl CounterfactualEvaluator for CoverageEvaluator {
        async fn evaluate(&self, masked: &HashSet<MemoryRef>) -> Result<f32, MnemeError> {
            let cover_gone =
                !self.group.is_empty() && self.group.iter().all(|m| masked.contains(m));
            let lost = if cover_gone { self.group_value } else { 0.0 };
            Ok((self.baseline - lost).clamp(0.0, 1.0))
        }
    }

    fn greedy_scorer() -> ContributionScorer {
        ContributionScorer::new(CausalConfig {
            mode: ScoreMode::GreedyAblation,
            ..CausalConfig::default()
        })
    }

    #[tokio::test]
    async fn greedy_recovers_redundant_set_contribution_loo_misses() {
        let (a, b) = (mref(), mref());

        // LOO: each member scores ≈ 0 because the other still covers → the
        // set's real 0.4 contribution is missed entirely.
        let loo = scorer()
            .score(&CoverageEvaluator::new(0.8, &[a, b], 0.4), &[a, b])
            .await
            .unwrap();
        let loo_sum: f32 = loo.scored.iter().map(|c| c.contribution).sum();
        assert!(
            loo_sum.abs() < 1e-6,
            "LOO misses redundancy (≈0), got {loo_sum}"
        );

        // Greedy: recovers the full 0.4 (attributed to the last-removed member,
        // once the cover is gone).
        let greedy = greedy_scorer()
            .score(&CoverageEvaluator::new(0.8, &[a, b], 0.4), &[a, b])
            .await
            .unwrap();
        let greedy_sum: f32 = greedy.scored.iter().map(|c| c.contribution).sum();
        assert!(
            (greedy_sum - 0.4).abs() < 1e-6,
            "greedy recovers the set's 0.4, got {greedy_sum}"
        );
        assert_eq!(greedy.mode, ScoreMode::GreedyAblation);
    }

    #[tokio::test]
    async fn greedy_gc_keeps_one_of_a_redundant_pair() {
        let (a, b) = (mref(), mref());
        let report = greedy_scorer()
            .score(&CoverageEvaluator::new(0.8, &[a, b], 0.4), &[a, b])
            .await
            .unwrap();
        // Exactly one redundant copy is dead weight (its removal cost nothing
        // because the other still covered); the cover itself is kept. LOO would
        // wrongly GC *both* and lose the capability.
        let gc = report.gc_candidates(1e-4);
        assert_eq!(gc.len(), 1, "GC one copy, keep the cover; got {gc:?}");
        let kept = report.high_contributors(0.3);
        assert_eq!(kept.len(), 1, "the surviving cover carries the full 0.4");
    }

    #[tokio::test]
    async fn greedy_matches_loo_on_additive_objective() {
        // On an additive objective greedy and LOO agree — greedy is a strict
        // generalisation, not a different answer where LOO is already correct.
        let (helpful, inert, harmful) = (mref(), mref(), mref());
        let mk = || {
            FakeEvaluator::new(0.8)
                .with_marginal(helpful, 0.4)
                .with_marginal(inert, 0.0)
                .with_marginal(harmful, -0.2)
        };
        let report = greedy_scorer()
            .score(&mk(), &[helpful, inert, harmful])
            .await
            .unwrap();
        let by = |m: MemoryRef| {
            report
                .scored
                .iter()
                .find(|c| c.memory == m)
                .unwrap()
                .contribution
        };
        assert!((by(helpful) - 0.4).abs() < 1e-6, "helpful ≈ +0.4");
        assert!(by(inert).abs() < 1e-6, "inert ≈ 0");
        assert!((by(harmful) + 0.2).abs() < 1e-6, "harmful ≈ -0.2");
    }
}
