//! # mnesio-causal — counterfactual contribution scoring + GC (Phase 10)
//!
//! The replay dividend. mnesio's system of record is an append-only,
//! *replayable* log, so it can answer a question a storage-shaped memory
//! cannot: **"what would outcomes have been if this memory had never
//! existed?"** Re-run outcome evaluation with a memory *masked* and measure
//! the delta — that delta is the memory's **causal contribution**.
//!
//! With contribution in hand, garbage collection stops being a heuristic
//! time-decay guess (FadeMem) and becomes a *measurement*: a memory that
//! provably never moved an outcome is safe to retire; one whose removal drops
//! outcomes is load-bearing and must be kept.
//!
//! ## Shape
//!
//! - [`CounterfactualEvaluator`] — the one seam (Hard Rule #7). Given a set of
//!   masked memories it returns an aggregate objective in `[0,1]` (recall@k,
//!   benchmark pass-rate, …). The baseline is `evaluate(&{})`. The real impl
//!   wires the retriever + executor; [`FakeEvaluator`] makes tests hermetic.
//! - [`ContributionScorer`] — leave-one-out (LOO) ablation, bounded by
//!   [`CausalConfig`]. Produces a [`ContributionReport`].
//! - [`ContributionReport::gc_candidates`] / [`high_contributors`] — turn the
//!   report into a retire / keep decision.
//! - [`gc`] — apply a GC decision by *appending* [`Event::MemoryInvalidated`]
//!   (Hard Rule #2: invalidate, never delete; Hard Rule #4: it's an event).
//!
//! ## Hard-rule posture
//!
//! - **#5 (fast write path):** counterfactual replay is `O(corpus × eval)`. It
//!   runs *offline*, never on a write or the default search path.
//! - **#6 (bound the cascades):** [`CausalConfig::max_candidates`] caps each
//!   pass; the caller schedules passes, not the engine.
//! - **#3 (scope is a boundary):** the scorer operates on a candidate list the
//!   caller has already scoped; GC emits `MemoryInvalidated` in that scope.
//!
//! ## Redundancy: LOO vs greedy ablation
//!
//! [`ScoreMode::LeaveOneOut`] (cheap, `O(corpus)`) underestimates contribution
//! under **redundancy**: two memories each sufficient to answer a query both
//! score ~0 individually, yet removing both drops the score — the Shapley-vs-LOO
//! gap. [`ScoreMode::GreedyAblation`] closes it: it ablates the least-valuable
//! memory step by step, so once one of a redundant pair is masked the other
//! becomes load-bearing and earns the set's full contribution. Greedy is a
//! single Shapley permutation — exact on additive objectives and on the
//! aggregate of a redundant set — at `O(corpus²)` eval cost (still bounded by
//! `max_candidates`, Hard Rule #6). Pick LOO for cheap first-pass GC, greedy
//! when redundancy would otherwise hide load-bearing memories.

mod contribution;
mod evaluator;
mod gc;

pub use contribution::{
    CausalConfig, ContributionReport, ContributionScorer, CounterfactualEvaluator, FakeEvaluator,
    MemoryContribution, ScoreMode,
};
pub use evaluator::{RetrievalEvaluator, RetrievalTask};
pub use gc::{gc, GcDecision, GcOutcome, GC_REASON};
