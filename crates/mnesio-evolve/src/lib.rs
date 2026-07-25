//! # mnesio-evolve
//!
//! Memory evolution — the *supporting* mechanism in the architecture
//! (long-form §2.3). When a memory is written, a bounded, async worker
//! retroactively re-tags / re-links semantically related memories
//! (A-MEM style, arXiv:2502.12110), but with the safety bounds the
//! long-form §3 insists on:
//!
//! - **Never overwrite** — invalidate + create a new bi-temporal version
//!   with a `parent` pointer.
//! - **`max_evolve_per_write`** — hard cap on how many neighbors one
//!   write can mutate; stops cascade fan-out.
//! - **Per-memory cooldown** — a memory can't be re-evolved until
//!   `cooldown_secs` has elapsed since its last evolution.
//! - **Lifetime evolution count** — `max_lifetime_evolutions` ceiling
//!   per memory; after that the memory is considered "stable" and the
//!   worker leaves it alone.
//! - **`min_change_threshold`** — a proposed evolution is only persisted
//!   when its structural delta meets this threshold; trivial tag-adds
//!   are dropped.
//! - **Scope isolation** — neighbor lookups are scope-filtered; no
//!   cross-tenant rewrites.
//! - **Self-exclusion** — the freshly-written memory cannot trigger
//!   evolution of itself.
//! - **Loop prevention** — memories that already carry a `parent` (i.e.
//!   are themselves the result of an earlier evolution) do not trigger
//!   a fresh evolution pass.
//!
//! The three A-MEM steps live in [`prompts`] (the LLM templates) and
//! [`parse`] (tolerant parsers). The [`worker`] module glues them
//! together with the bounded scheduler.

pub mod parse;
pub mod prompts;
pub mod worker;

#[cfg(test)]
mod worker_tests;

pub use worker::{spawn, EvolutionWorker};

/// Tunable bounds for the evolution worker (long-form §3, "hard caps").
#[derive(Debug, Clone)]
pub struct EvolveConfig {
    /// Maximum neighbors mutated by a single new-memory event. Caps the
    /// branching factor of any individual cascade.
    pub max_evolve_per_write: usize,
    /// Per-memory lifetime ceiling on how many times it may be evolved
    /// before the worker treats it as stable.
    pub max_lifetime_evolutions: u16,
    /// Minimum seconds between successive evolutions of the same
    /// memory. Prevents rapid-fire churn.
    pub cooldown_secs: u64,
    /// Minimum structural change for an evolution to be persisted at
    /// all — measured by [`parse::EvolutionChanges::total_additions`].
    /// Trivial proposals are dropped.
    pub min_change_threshold: usize,
}

impl Default for EvolveConfig {
    fn default() -> Self {
        Self {
            max_evolve_per_write: 3,
            max_lifetime_evolutions: 8,
            cooldown_secs: 300,
            min_change_threshold: 1,
        }
    }
}
