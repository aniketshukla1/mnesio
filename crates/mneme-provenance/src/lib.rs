//! # mneme-provenance — regulator-grade provenance (Phase 15)
//!
//! The black-box recorder. Not a new engine — a *positioning capability* over
//! the substrate that already exists (append-only + bi-temporal log). It makes
//! three things first-class, each a pure read-path fold over the log.
//!
//! **Time-travel reconstruction** ([`Timeline::snapshot_as_of`]) replays the log
//! to reconstruct the *exact* live memory set "as the agent knew it" at any past
//! transaction time `T` (written ≤ T and not yet invalidated ≤ T). Re-answer a
//! query against that snapshot and you get the agent's belief at `T`, not
//! today's.
//!
//! **Provenance chains** ([`Timeline::provenance`]) trace every belief to its
//! source events + supersessions: the write that created it, the evolutions that
//! refined it, the invalidation that retired it, and the version that replaced
//! it — ordered by transaction time.
//!
//! **Verifiable erasure** ([`RedactionPolicy`]) redacts a crypto-shredded
//! subject from *both* live reads **and** historical replays — the same `forget`
//! that drops the key (Phase 8) blanks the content at every `T` — while the log
//! entries themselves stay (append-only, Hard Rule #2). The immutable record and
//! verifiable forget coexist: an auditor sees that an event happened and was
//! redacted, never the erased content.
//!
//! This is the combination no storage-shaped memory can offer: mutable storage
//! can't reconstruct a past state it overwrote, and can't prove erasure against
//! a log it doesn't keep.
//!
//! Hard-rule posture: every view here is derived by replay and nothing is
//! mutated (#2/#4) — `entry_count` is identical before and after a `forget`,
//! because erasure redacts the *projection*, never the log; the caller scopes
//! the entries it folds (#3). The tx-clock is the ULID `timestamp_ms()` carried
//! by every `LogEntry::id`, so "as of T" is exact and replay-deterministic.

mod timeline;

pub use timeline::{
    subject_passthrough, MemoryView, ProvenanceChain, ProvenanceLink, ProvenanceLinkKind,
    RedactionPolicy, Timeline,
};
