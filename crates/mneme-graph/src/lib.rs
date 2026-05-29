//! # mneme-graph
//!
//! Bi-temporal property graph store backed by **fjall**. Phase-4
//! deliverable: a [`MaterializedView`] over the event log that lets
//! callers ask *graph* questions —
//!
//! > "what memories does X cite, and what did X cite a week ago?"
//! > "give me the 2-hop neighbourhood of this memory inside tenant A."
//! > "is there a lineage path from this evolved note back to its original?"
//!
//! without scanning the log or post-filtering vector hits.
//!
//! ## Why bi-temporal
//!
//! Memory evolution invalidates the previous version and emits a new
//! one (Hard Rule #2). A flat "current" graph would lose that lineage
//! the moment the worker fires. Every edge stores `tx_from` and an
//! optional `tx_to`; every query takes an optional `as_of`. The graph
//! at `T` is the set of edges whose tx-interval contains `T`. Replay
//! the log up to `T` and you'd get the same graph — that's what makes
//! Hard Rule #4 (rebuild from log) hold.
//!
//! ## Why property graph (not RDF / triples)
//!
//! Each node carries the memory's [`Scope`], tags, and bi-temporal
//! stamp inline so traversals can scope-filter without a join. Each
//! edge carries a typed [`Relation`] (`Linked`, `EvolvedFrom`,
//! `EvolvedTo`, `ContainedIn`) plus an optional weight. Multi-relation
//! traversals are a single prefix scan.
//!
//! ## Hard rules respected
//!
//! - **#3 Scope is a security boundary.** Every traversal filters
//!   edges by the source node's scope using [`Scope::contains`]; a
//!   tenant-A query never returns a tenant-B node, even if a (buggy)
//!   `MemoryLinksUpdated` event slipped a cross-tenant link in.
//! - **#4 Single system of record.** The graph holds no state that
//!   isn't derivable from the event log. Drop the partition and
//!   replay; same graph.
//! - **#5 Write path stays fast.** `apply()` does a small bounded
//!   number of key writes per event; no LLM calls, no fan-out beyond
//!   the affected node's incident edges.
//!
//! ## What this crate *doesn't* do
//!
//! - It is not a Cypher / Gremlin engine. The query surface is
//!   intentionally narrow — neighbours, BFS, shortest-path. Pattern
//!   matching arrives only if a real use case shows up.
//! - It does not store memory *content* — only id, scope, bi-temporal
//!   stamp, and tags. The content lives in the event log; resolve it
//!   via `EventLog::read_from` on the returned ids.

pub mod record;
pub mod traversal;
pub mod view;

#[cfg(test)]
mod tests;

pub use record::{EdgeRecord, NodeRecord, Relation};
pub use traversal::{Hop, Path};
pub use view::{FjallGraphView, GraphStats, NodeDegree};
