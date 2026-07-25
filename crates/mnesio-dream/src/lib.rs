//! # mnesio-dream — negative memory + dreaming (Phase 14)
//!
//! Two extensions of the procedural compiler, both still behind the gate.
//!
//! 1. **Anti-memory** ([`suppress`]). Everyone learns what *to* retrieve;
//!    mnesio's outcome loop can learn what *not* to. A pattern of bad outcomes
//!    ("retrieving memory X for query-class Y hurt") is compiled into a gated
//!    suppression [`ArtifactKind::RetrievalRule`] — self-improving *negative
//!    space*. Crucially it's gated like any other artifact (Hard Rule #1): a
//!    suppression that would regress a canary is rejected, so anti-memory can
//!    never quietly blind the agent to something it needs.
//!
//! 2. **Dreaming** ([`dream`]). A bounded *offline* consolidation pass
//!    (sleep-time compute): replay the corpus, prune by **Phase-10
//!    counterfactual contribution** (provable dead/harmful weight, not a
//!    time-decay guess), and **re-anchor evolved notes to their `parent`** —
//!    the cascade-divergence fix flagged in CLAUDE.md "Known hard problems".
//!    It runs offline + bounded ([`DreamConfig`], Hard Rules #5/#6), never on
//!    the write path.
//!
//! ## Why only mnesio can do this
//!
//! Anti-memory needs a gated outcome loop (a storage-shaped memory has none —
//! it can only ever store the *positive*). Dreaming's prune-by-contribution
//! needs the replayable log (Phase 10); its re-anchoring needs bi-temporal
//! lineage (`parent` pointers). Both are substrate features competitors lack.
//!
//! ## Hard-rule posture
//!
//! - **#1 (gate):** [`SuppressionLearner::learn`] only emits a rule whose
//!   re-eval is `is_committable()`.
//! - **#2 / #4 (never overwrite, log is truth):** pruning appends
//!   `MemoryInvalidated`; re-anchoring appends `MemoryLinksUpdated`. Nothing is
//!   edited in place.
//! - **#5 / #6 (offline + bounded):** the dream pass is caller-scheduled and
//!   capped by [`DreamConfig`].
//! - **#7 (swappable seam):** [`SuppressionEvaluator`] is the trait; the fake
//!   keeps tests hermetic.

mod dream;
mod suppress;

pub use dream::{
    dream, DreamConfig, DreamOutcome, DreamPass, DreamReport, DriftedNote, ReanchorAction,
};
pub use suppress::{
    BadOutcome, FakeSuppressionEvaluator, SuppressConfig, SuppressionEvaluator, SuppressionLearner,
    SuppressionOutcome, SuppressionRule,
};
