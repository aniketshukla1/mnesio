//! # mneme-extract
//!
//! **Ingestion intelligence** — the stage that turns a raw conversational
//! turn (or document) into *salient, deduplicated, conflict-resolved*
//! memories before anything is committed. This is the piece every
//! production memory layer (Mem0, Zep, MemMachine) builds its quality on
//! and the one mneme was missing: storing turns verbatim is a "junk
//! drawer"; the value is in deciding *what* to remember and *how it
//! relates to what you already know*.
//!
//! The pipeline is two stages, both behind traits so providers swap:
//!
//! 1. **Extract** ([`Extractor`]) — pull atomic facts out of raw text.
//!    One messy paragraph becomes several crisp, self-contained
//!    statements.
//! 2. **Consolidate** ([`Consolidator`]) — for each fact, decide against
//!    the memories you already hold:
//!    - **ADD** — genuinely new knowledge → write a new memory.
//!    - **UPDATE** — the fact *refines* or *contradicts* an existing
//!      memory → supersede it with a new bi-temporal version (Hard Rule
//!      #2: never overwrite, invalidate + re-version). Contradiction vs.
//!      refinement is tracked so the caller can audit *why* a memory
//!      changed.
//!    - **NOOP** — already represented → do nothing (dedup).
//!
//! ## Design choices
//!
//! - **Pure + caller-fed candidates.** [`Consolidator::consolidate`]
//!   takes the existing candidate memories as an argument rather than
//!   reaching into a store. That keeps the engine deterministic and
//!   testable offline; the async worker (host side) supplies candidates
//!   from the retriever + memory cache and applies the resulting
//!   [`ConsolidationAction`]s as events. This mirrors how
//!   `mneme-procedural` separates the pure compiler from the worker.
//! - **No event writes here.** The engine *plans*; it never appends to
//!   the log. That boundary keeps the write-path-fast rule (Hard Rule
//!   #5) the host's concern and this crate I/O-free.
//! - **Scope is carried through** so the host can assert every produced
//!   action stays inside the originating scope (Hard Rule #3).

pub mod consolidate;
pub mod decay;
pub mod extractor;
pub mod importance;
pub mod parse;
pub mod prompts;

pub use consolidate::{ConsolidateConfig, Consolidator};
pub use decay::{forgettable, DecayInput, DecayModel};
pub use extractor::{Extractor, LlmExtractor};
pub use importance::{
    heuristic_importance, novelty_vs, AdmissionPolicy, Importance, ImportanceWeights,
};
pub use parse::{Decision, UpdateReason};

use mneme_core::types::MemoryRef;

/// An existing memory presented to the consolidator as a dedup/conflict
/// candidate. Just the id + content — the engine needs nothing else to
/// reason about overlap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExistingMemory {
    pub id: MemoryRef,
    pub content: String,
}

impl ExistingMemory {
    pub fn new(id: MemoryRef, content: impl Into<String>) -> Self {
        Self {
            id,
            content: content.into(),
        }
    }
}

/// What the consolidator decided to do with one extracted fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsolidationAction {
    /// Net-new knowledge. The host should write a fresh
    /// [`mneme_core::Memory`] carrying `content` **using `id`** as its
    /// memory id. The id is assigned at plan time (a fresh ULID) so that
    /// a later fact in the *same batch* can NOOP/UPDATE against this one
    /// with a stable reference — the host must honour it.
    Add { id: MemoryRef, content: String },
    /// The fact supersedes `target`. The host should write a new
    /// bi-temporal version (with `content`, `parent = target`) and
    /// invalidate `target` — the same supersede-and-invalidate triple
    /// the evolution worker emits. `reason` records whether this was a
    /// factual contradiction or a refinement, for audit.
    Update {
        target: MemoryRef,
        content: String,
        reason: UpdateReason,
    },
    /// The fact is already represented by `duplicate_of` — drop it.
    Noop { duplicate_of: MemoryRef },
}

impl ConsolidationAction {
    /// True for actions that mutate state (`Add`/`Update`). Handy for
    /// metrics — "how many of N extracted facts actually changed memory?"
    pub fn is_write(&self) -> bool {
        !matches!(self, ConsolidationAction::Noop { .. })
    }
}

/// The full plan for one raw observation: the facts that were extracted
/// and the action chosen for each. Returned by
/// [`Consolidator::consolidate`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConsolidationPlan {
    pub actions: Vec<ConsolidationAction>,
}

impl ConsolidationPlan {
    pub fn adds(&self) -> usize {
        self.actions
            .iter()
            .filter(|a| matches!(a, ConsolidationAction::Add { .. }))
            .count()
    }
    pub fn updates(&self) -> usize {
        self.actions
            .iter()
            .filter(|a| matches!(a, ConsolidationAction::Update { .. }))
            .count()
    }
    pub fn noops(&self) -> usize {
        self.actions
            .iter()
            .filter(|a| matches!(a, ConsolidationAction::Noop { .. }))
            .count()
    }
}
