//! # mneme-probe — self-falsifying memory (Phase 11)
//!
//! Memory with CI. A stored fact can carry a **re-checkable acceptance
//! probe** — an assertion that should still hold if the fact is still true.
//! A bounded worker re-runs probes and, on failure, *invalidates-and-
//! supersedes* the memory (Hard Rule #2: never overwrite — history is kept)
//! and can enqueue re-evolution. Storage-shaped competitors can show you a
//! belief; only a system with an eval substrate wired to invalidate-and-
//! supersede versioning can make a belief *falsify itself*.
//!
//! Two cooperating ideas:
//!
//! 1. **Acceptance probes** ([`Probe`], [`ProbeRunner`]). Re-evaluate a
//!    memory's claim. A `Refuted` outcome triggers [`falsify`], which appends
//!    the canonical supersede triple (`MemoryWritten` correction +
//!    `MemoryEvolved` lineage + `MemoryInvalidated`) — the exact shape the
//!    ingestion + evolution workers already emit, so every existing view
//!    (vector, BM25, graph, ACL) handles it unchanged.
//! 2. **Belief calibration** ([`belief`]). A per-memory confidence in
//!    `[0,1]`, *derived* by replaying corroborating/contradicting evidence
//!    from the bi-temporal chain — not stored as new state (Hard Rule #4: a
//!    view rebuildable from the log). Retrieval can then return
//!    "belief + confidence + why".
//!
//! ## Hard-rule posture
//!
//! - **#2 (never overwrite):** falsification supersedes; the refuted version
//!   stays in the log, reachable by time-travel.
//! - **#4 (log is the truth):** confidence is recomputed from events, never
//!   persisted as authoritative side state.
//! - **#5 (fast write path):** probes run in a bounded *offline* worker,
//!   never on a write or the default read path.
//! - **#6 (bound the cascades):** [`ProbeConfig::max_probes_per_pass`] caps
//!   each pass; the caller schedules passes, not the engine.
//! - **#7 (swappable seam):** [`Probe`] is the trait; the real impl wires an
//!   `LlmClient`/retriever, [`FakeProbe`] keeps tests hermetic.
//!
//! ## Known limitation (don't pretend it's solved)
//!
//! A probe is only as good as its check. A flaky probe could wrongly refute a
//! true memory; v1 mitigates with an explicit [`ProbeStatus::Inconclusive`]
//! outcome (never supersedes) and leaves multi-run quorum to
//! `TODO(phase-11)`.

mod belief;
mod falsify;
mod probe;

pub use belief::{belief_of, Belief, BeliefLedger, Evidence};
pub use falsify::{falsify, FalsifyOutcome, FALSIFY_REASON};
pub use probe::{
    FakeProbe, Probe, ProbeConfig, ProbeOutcome, ProbeReport, ProbeRunner, ProbeStatus,
    ProbeVerdict,
};
