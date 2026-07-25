//! The append-only event log is the single system of record. Every index
//! (vector, BM25, graph, procedural) is a materialized view rebuildable by
//! replaying these events. See architecture report §4 and §8.

use crate::entity::{Memory, Outcome, PolicyArtifact, Source};
use crate::types::*;
use serde::{Deserialize, Serialize};

/// One immutable entry in the log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub id: Id,
    pub event: Event,
}

/// Everything that can happen in the system. Memory writes and outcome
/// records are commutative; procedural commits are NOT (single-writer).
// TODO(phase-0): `MemoryWritten(Memory)` makes this enum ~370 bytes; the next
// largest variant is ~160 bytes. Box `Memory` (and likely `Outcome`) to even
// out the variants. Deferred to its own slice so this one stays focused.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    MemoryWritten(Memory),
    /// Emitted by the async embedding worker once a `MemoryWritten` whose
    /// `embedding` was `None` has been embedded. The `model_id` lets a view
    /// (and the startup mismatch check) refuse to silently mix incompatible
    /// vector spaces — long-form design §4.5.
    MemoryEmbedded {
        id: MemoryRef,
        embedding: Vec<f32>,
        model_id: String,
    },
    /// Emitted by the evolution worker after its **note-construction**
    /// pass (A-MEM step P_s1). Updates the memory's derived structured
    /// fields without creating a new bi-temporal version — like
    /// `MemoryEmbedded`, this is a derived-state amendment, not lineage.
    MemoryNoteEnriched {
        id: MemoryRef,
        keywords: Vec<String>,
        tags: Vec<String>,
        context: String,
    },
    /// Emitted by the evolution worker after its **link-generation** pass
    /// (A-MEM step P_s2). Replaces the memory's `links` field with the
    /// LLM's selected related-memory refs. Also a derived-state amendment
    /// rather than lineage.
    MemoryLinksUpdated {
        id: MemoryRef,
        links: Vec<MemoryRef>,
    },
    MemoryEvolved {
        from: MemoryRef,
        to: MemoryRef,
        diff: ChangeSet,
    },
    MemoryInvalidated {
        id: MemoryRef,
        reason: String,
    },

    /// A raw, un-consolidated observation (a conversational turn, a note,
    /// a tool trace) entering the system. Appended fast on the write path
    /// (Hard Rule #5); the async **ingestion worker** (`mnesio-extract`)
    /// tails these, extracts atomic facts, and consolidates each into
    /// `MemoryWritten` (ADD), a supersede triple (UPDATE), or nothing
    /// (NOOP / below the admission floor). Keeping the raw turn in the log
    /// means the consolidation is fully replayable + auditable.
    ObservationRecorded {
        scope: Scope,
        content: String,
        /// Which actor/agent produced this (multi-agent attribution).
        /// `None` for single-agent / anonymous input.
        actor: Option<String>,
    },

    /// Grant agent `grantee` read access to memories owned by agent
    /// `owner` within `tenant`. Multi-agent ACL layer (P1#7): a finer
    /// boundary *inside* a `Scope`. An agent always reads its own
    /// memories + system-owned ("shared") memories; reading another
    /// agent's memories requires an explicit grant.
    AgentAccessGranted {
        tenant: String,
        owner: String,
        grantee: String,
    },
    /// Revoke a previously-granted cross-agent read.
    AgentAccessRevoked {
        tenant: String,
        owner: String,
        grantee: String,
    },

    /// A new source document was ingested. The chunks themselves arrive as
    /// separate `MemoryWritten` events with their `source` field pointing
    /// at this source's id.
    SourceIngested(Source),
    /// All chunks of a source are being invalidated together. Convenience
    /// over emitting N `MemoryInvalidated` events.
    SourceInvalidated {
        id: SourceRef,
        reason: String,
    },

    /// Set (or overwrite) a stable **profile** attribute for a scope —
    /// a long-term preference / trait / identity fact, distinct from
    /// episodic memories. Re-setting the same `attribute` supersedes the
    /// previous value; the `ProfileView` keeps the prior value as history
    /// (never overwrite — Hard Rule #2). `value = ""` clears it.
    ProfileSet {
        scope: Scope,
        /// Stable attribute key (e.g. `"diet"`, `"locale"`, `"timezone"`).
        attribute: String,
        value: String,
        /// Which actor/agent asserted this (multi-agent attribution).
        actor: Option<String>,
    },

    OutcomeRecorded(Outcome),

    ProceduralProposed {
        proposal: ProposalId,
        artifacts: Vec<PolicyArtifact>,
    },
    ProceduralCommitted {
        proposal: ProposalId,
        report: EvalReport,
    },
    ProceduralRejected {
        proposal: ProposalId,
        reason: String,
    },

    /// Snapshot of an evaluation-suite run against a committed artifact
    /// version. Emitted by the procedural worker after every successful
    /// commit, with optional baseline-seed points emitted at boot for
    /// version=1 artifacts that don't yet have a measurement.
    ///
    /// Treating curve points as log events (rather than ephemeral
    /// telemetry) is what makes Phase-2 "done when" claims auditable:
    /// the learning curve can be replayed exactly from the log, and a
    /// safety-probe regression is preserved as a permanent record
    /// rather than evaporating on process restart.
    LearningCurveRecorded {
        artifact: ArtifactRef,
        version: u32,
        /// Absolute suite score `[0.0, 1.0]`.
        benchmark_score: f32,
        /// Safety probe pass rate `[0.0, 1.0]`. Any value < 1.0 is the
        /// alignment-drift signal (report §10).
        safety_probe_pass_rate: f32,
        /// Gate's per-commit objective Δ — joined here so the curve
        /// can be plotted alongside the gate's relative signal.
        objective_delta: f32,
        /// Distinct judges that scored this version (diversity gate
        /// compliance).
        judges_consulted: u8,
    },
}

/// A structural diff produced by the evolution worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeSet {
    pub keywords_added: Vec<String>,
    pub keywords_removed: Vec<String>,
    pub tags_added: Vec<String>,
    pub tags_removed: Vec<String>,
    pub context_rewritten: bool,
}

/// The outcome of shadow-evaluating a procedural proposal before commit.
/// This is the regression guard LangMem omits (report §2, §10). The data
/// shape lives in `mnesio-core` because both event-log writers and the
/// procedural compiler need it; the *interpretation* (configurable
/// thresholds, structured rejection reasons) lives in `mnesio-procedural`
/// — see `mnesio_procedural::gate::EvalGates`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Canary inputs the proposal answered correctly under shadow eval.
    pub canaries_passed: u32,
    pub canaries_total: u32,
    /// Fraction of held-out replay outcomes the proposal reproduced
    /// successfully. Tail-protection: a proposal that ships canaries
    /// perfectly but silently breaks unrelated tasks is still a regression.
    pub replay_success_rate: f32,
    /// Externally-maintained safety probe (report §10 — the
    /// "Alignment Tipping Process" guard). A monotone-decreasing trend
    /// across versions is the hard-stop on commits; one bad probe is
    /// always a reject.
    pub safety_probe_passed: bool,
    /// Δ on the chosen objective vs. the active version. Strictly ≥ 0
    /// under default gates; tunable upward to require *improvement* not
    /// merely non-regression.
    pub objective_delta: f32,
    /// How many independent judges contributed to `safety_probe_passed`
    /// and the success scores. Single-judge probes are vulnerable to
    /// in-context reward hacking (report §10) so production gates demand
    /// 2+. Defaults to 0 on legacy events for serde back-compat — the
    /// gate treats 0 as "unknown" and rejects under any judge-diversity
    /// requirement, which is the safe direction.
    #[serde(default)]
    pub judges_consulted: u8,
}

impl EvalReport {
    /// Strict default gate — Hard Rule #1 in its most conservative form.
    /// All canaries pass, safety probe passes, objective is non-negative.
    ///
    /// This method is intentionally **not configurable**: it represents
    /// the invariant *every* procedural commit must satisfy. For
    /// configurable thresholds (judge diversity, replay rate floors,
    /// minimum improvement deltas) use
    /// [`mnesio_procedural::gate::EvalGates`] which composes additional
    /// rejection reasons on top of this baseline.
    pub fn is_committable(&self) -> bool {
        self.canaries_passed == self.canaries_total
            && self.safety_probe_passed
            && self.objective_delta >= 0.0
    }
}
