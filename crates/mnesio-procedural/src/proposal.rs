//! [`Proposal`] — the unit of work the compiler hands to the gate.
//!
//! A proposal bundles everything an operator (or the audit log) needs to
//! understand a single commit attempt:
//!
//! - **which** artifact version it would replace
//! - **what** the candidate replacement(s) look like
//! - **when** it was generated and **why**
//!
//! The [`mnesio_core::event::Event::ProceduralProposed`] event records the
//! `ProposalId` and the candidate artifacts; this struct is the in-memory
//! envelope that carries those alongside the metadata the compiler needs
//! for credit assignment, audit trails, and dashboard rendering.
//!
//! Proposals are intentionally cheap to clone — they're moved across the
//! reflect → propose → shadow-eval → commit pipeline and we want the
//! handoffs to stay readable.

use mnesio_core::entity::PolicyArtifact;
use mnesio_core::types::{new_id, ArtifactRef, ProposalId, Scope};
use serde::{Deserialize, Serialize};

/// One candidate revision under evaluation. Carries enough context that a
/// future commit (or a rejection audit) is self-explanatory without
/// joining against other tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    /// Stable ID — survives across reflect/eval/commit and matches the
    /// `proposal` field on the three `Procedural*` events.
    pub id: ProposalId,
    /// Scope the proposal applies to. Hard Rule #3: a proposal may only
    /// reference artifacts within its own scope; the compiler enforces
    /// scope alignment before handing the proposal to the gate.
    pub scope: Scope,
    /// The artifact version the proposal seeks to replace. `None` for a
    /// genuinely new artifact (no predecessor) — uncommon for the
    /// reflective loop, common for the first-write seed.
    pub base_version: Option<ArtifactRef>,
    /// One or more candidate artifacts. Multiple entries cover the
    /// GEPA-style "propose K candidates" step; the gate is run per
    /// candidate and the Pareto-best committable one wins.
    pub candidates: Vec<PolicyArtifact>,
    /// Human-readable reason this proposal exists. Surfaced on the
    /// dashboard's chain timeline alongside the verdict.
    pub rationale: String,
    /// Wall-clock timestamp (unix ms) the proposal was generated.
    /// Independent of `id` — `id` is monotonic but its embedded
    /// timestamp is the ULID's, which is fine but obscured by the
    /// public API.
    pub created_at_ms: u64,
}

impl Proposal {
    /// Construct a proposal with a fresh `ProposalId`. Call sites that
    /// need a specific id (replay, tests) should build the struct
    /// literally instead.
    pub fn new(
        scope: Scope,
        base_version: Option<ArtifactRef>,
        candidates: Vec<PolicyArtifact>,
        rationale: impl Into<String>,
        created_at_ms: u64,
    ) -> Self {
        Self {
            id: ProposalId(new_id()),
            scope,
            base_version,
            candidates,
            rationale: rationale.into(),
            created_at_ms,
        }
    }

    /// True iff this proposal is a genuinely new artifact rather than a
    /// rewrite of an existing one. Used by the audit log + dashboard
    /// to label commits as "first write" vs "version bump".
    pub fn is_first_write(&self) -> bool {
        self.base_version.is_none()
    }

    /// Number of candidate artifacts. The GEPA loop generates K of
    /// these; the gate picks the best committable one.
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::entity::ArtifactKind;
    use mnesio_core::types::{new_id, BiTemporal};

    fn make_artifact(body: &str) -> PolicyArtifact {
        PolicyArtifact {
            id: new_id(),
            version: 1,
            scope: Scope::global("t"),
            kind: ArtifactKind::SystemPrompt { body: body.into() },
            canaries: Vec::new(),
            time: BiTemporal::now(),
        }
    }

    #[test]
    fn new_assigns_fresh_proposal_id() {
        let a = Proposal::new(
            Scope::global("t"),
            None,
            vec![make_artifact("foo")],
            "test",
            42,
        );
        let b = Proposal::new(
            Scope::global("t"),
            None,
            vec![make_artifact("bar")],
            "test",
            42,
        );
        assert_ne!(a.id, b.id, "every proposal gets a distinct id");
    }

    #[test]
    fn first_write_detection_uses_base_version_presence() {
        let scope = Scope::global("t");
        let seed = Proposal::new(
            scope.clone(),
            None,
            vec![make_artifact("seed")],
            "first artifact",
            1,
        );
        assert!(seed.is_first_write());

        let bump = Proposal::new(
            scope,
            Some(ArtifactRef(new_id())),
            vec![make_artifact("revised")],
            "improved by reflective pass",
            2,
        );
        assert!(!bump.is_first_write());
    }

    #[test]
    fn candidate_count_reflects_candidates_vec() {
        let p = Proposal::new(
            Scope::global("t"),
            None,
            vec![make_artifact("a"), make_artifact("b"), make_artifact("c")],
            "K=3 from reflective loop",
            0,
        );
        assert_eq!(p.candidate_count(), 3);
    }

    #[test]
    fn serde_round_trips() {
        // Proposals don't go through bincode (they're in-memory only —
        // the events log the artifact list directly) but JSON round-trip
        // covers any future API surface.
        let p = Proposal::new(
            Scope::global("t"),
            Some(ArtifactRef(new_id())),
            vec![make_artifact("foo")],
            "rationale",
            123,
        );
        let json = serde_json::to_string(&p).unwrap();
        let p2: Proposal = serde_json::from_str(&json).unwrap();
        assert_eq!(p.id, p2.id);
        assert_eq!(p.candidate_count(), p2.candidate_count());
        assert_eq!(p.rationale, p2.rationale);
        assert_eq!(p.created_at_ms, p2.created_at_ms);
    }
}
