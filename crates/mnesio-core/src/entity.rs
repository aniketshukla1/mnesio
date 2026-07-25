//! The three entity families: [`Memory`] (knowledge), [`Outcome`] (feedback),
//! and [`PolicyArtifact`] (procedure). See architecture report §5.

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A unit of knowledge. An A-MEM-style note extended with a bi-temporal
/// stamp and an evolution lineage pointer.
///
/// When a longer document is ingested it's chunked into many `Memory`s that
/// share a [`SourceRef`] and carry a zero-based [`Memory::position`] within
/// their source. Standalone memories (not chunked from a document) leave
/// both fields `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: Id,
    pub scope: Scope,
    pub content: String,
    /// LLM-generated structured fields (A-MEM `K`, `G`, `X`).
    pub keywords: Vec<String>,
    pub tags: Vec<String>,
    pub context: String,
    /// Index into the embedding store; `None` until the async worker fills it.
    pub embedding: Option<Vec<f32>>,
    pub links: Vec<MemoryRef>,
    /// If this memory is an evolved version of another, points at the parent.
    pub parent: Option<MemoryRef>,
    pub evolution_count: u16,
    pub time: BiTemporal,
    pub provenance: Provenance,
    /// Source document this memory was chunked from (if any). Appended at
    /// the end of the struct so bincode-serialised pre-source logs still
    /// decode cleanly with `#[serde(default)]`.
    #[serde(default)]
    pub source: Option<SourceRef>,
    /// Zero-based position within the parent source. `None` when `source`
    /// is also `None`.
    #[serde(default)]
    pub position: Option<u32>,
}

/// A document a [`Memory`] was chunked from. Source identity persists even
/// when individual chunks evolve or get invalidated — a fact correction in
/// chunk 3 doesn't reopen the source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub id: Id,
    pub scope: Scope,
    pub title: String,
    /// Optional pointer to the original document (URL, file path, etc.).
    pub uri: Option<String>,
    /// Number of chunks the ingest emitted. Doesn't update on invalidation;
    /// "how many are still live" is derived from the chunks themselves.
    pub chunk_count: u32,
    pub time: BiTemporal,
    pub provenance: Provenance,
}

/// Where a memory came from — drives the trust scoring that protects the
/// procedural compiler from memory poisoning (report §10).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub source: String,
    /// 0.0 (untrusted) ..= 1.0 (fully trusted).
    pub trust: f32,
}

impl Default for Provenance {
    fn default() -> Self {
        Self {
            source: "unknown".into(),
            trust: 0.5,
        }
    }
}

/// Feedback the procedural compiler consumes. The `artifacts_used` field is
/// what makes credit assignment possible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub id: Id,
    pub episode: EpisodeRef,
    pub artifacts_used: Vec<ArtifactRef>,
    pub success: Option<bool>,
    /// Multi-objective scores (success rate, cost, latency, safety, ...).
    pub scores: HashMap<String, f32>,
    pub error: Option<String>,
    pub judge: JudgeSource,
    pub trajectory: TrajectoryRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JudgeSource {
    Environment,
    LlmJudge,
    Human,
    Mixed,
}

/// A versioned, scoped unit of procedure — the "how-to" the agent improves.
/// Modelled as a small ontology rather than a monolithic prompt (report §1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyArtifact {
    pub id: Id,
    pub version: u32,
    pub scope: Scope,
    pub kind: ArtifactKind,
    /// Canary inputs/outputs that any new version must still satisfy.
    pub canaries: Vec<Canary>,
    pub time: BiTemporal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactKind {
    SystemPrompt {
        body: String,
    },
    Heuristic {
        when: String,
        then: String,
    },
    Skill {
        signature: String,
        body: String,
        lang: String,
        preconditions: Vec<String>,
        postconditions: Vec<String>,
    },
    RetrievalRule {
        query_pattern: String,
        rewrite: String,
    },
    Reflection {
        episode: EpisodeRef,
        lesson: String,
    },
}

/// A single regression check carried by an artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Canary {
    pub input: String,
    pub expect: String,
}
