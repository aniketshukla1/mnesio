//! [`FjallGraphView`] — the bi-temporal property graph as a
//! [`MaterializedView`] over the event log.
//!
//! Storage layout (fjall partitions):
//!
//! - `nodes` — `[node_id (16B)] -> NodeRecord`
//! - `edges_out` — `[src | rel | dst | tx_from] -> EdgeRecord`
//! - `edges_in`  — `[dst | rel | src | tx_from] -> EdgeRecord` (same value)
//! - `meta`      — `last_checkpoint -> Id`
//!
//! `edges_out` and `edges_in` hold identical `EdgeRecord` values; the
//! redundancy buys us one-prefix-scan in either direction without
//! maintaining a separate index. The cost is two writes per edge
//! mutation, which is fine at the write-path budget Phase 0 set
//! (Hard Rule #5).

use crate::record::{
    encode_edge_in_key, encode_edge_out_key, encode_node_key, in_prefix, in_prefix_rel, out_prefix,
    out_prefix_rel, prefix_upper_bound, EdgeRecord, NodeRecord, Relation,
};
use async_trait::async_trait;
use mneme_core::entity::{Memory, Source};
use mneme_core::event::{Event, LogEntry};
use mneme_core::traits::MaterializedView;
use mneme_core::types::{Id, MemoryRef, Scope};
use mneme_core::MnemeError;
use std::ops::Bound;
use std::path::Path;
use std::sync::Arc;
use time::OffsetDateTime;

/// The graph view. Cheap to `clone()` (everything is in an `Arc` /
/// fjall handle).
pub struct FjallGraphView {
    keyspace: fjall::Keyspace,
    nodes: fjall::PartitionHandle,
    edges_out: fjall::PartitionHandle,
    edges_in: fjall::PartitionHandle,
    meta: fjall::PartitionHandle,
}

const META_LAST_CHECKPOINT: &[u8] = b"last_checkpoint";

impl FjallGraphView {
    /// Open (or create) a graph store rooted at `path`. Idempotent —
    /// opening an existing store is a no-op apart from the partition
    /// handles.
    pub fn open(path: impl AsRef<Path>) -> Result<Arc<Self>, MnemeError> {
        let keyspace = fjall::Config::new(path)
            .open()
            .map_err(|e| MnemeError::Storage(e.to_string()))?;
        let nodes = keyspace
            .open_partition("graph_nodes", fjall::PartitionCreateOptions::default())
            .map_err(|e| MnemeError::Storage(e.to_string()))?;
        let edges_out = keyspace
            .open_partition("graph_edges_out", fjall::PartitionCreateOptions::default())
            .map_err(|e| MnemeError::Storage(e.to_string()))?;
        let edges_in = keyspace
            .open_partition("graph_edges_in", fjall::PartitionCreateOptions::default())
            .map_err(|e| MnemeError::Storage(e.to_string()))?;
        let meta = keyspace
            .open_partition("graph_meta", fjall::PartitionCreateOptions::default())
            .map_err(|e| MnemeError::Storage(e.to_string()))?;
        Ok(Arc::new(Self {
            keyspace,
            nodes,
            edges_out,
            edges_in,
            meta,
        }))
    }

    // --- internal helpers -------------------------------------------------

    fn put_node(&self, n: &NodeRecord) -> Result<(), MnemeError> {
        let key = encode_node_key(n.id);
        let bytes = bincode::serialize(n).map_err(|e| MnemeError::Storage(e.to_string()))?;
        self.nodes
            .insert(key, bytes)
            .map_err(|e| MnemeError::Storage(e.to_string()))?;
        Ok(())
    }

    fn get_node_raw(&self, id: MemoryRef) -> Result<Option<NodeRecord>, MnemeError> {
        let key = encode_node_key(id);
        match self
            .nodes
            .get(key)
            .map_err(|e| MnemeError::Storage(e.to_string()))?
        {
            Some(v) => {
                let n: NodeRecord =
                    bincode::deserialize(&v).map_err(|e| MnemeError::Storage(e.to_string()))?;
                Ok(Some(n))
            }
            None => Ok(None),
        }
    }

    fn put_edge(&self, e: &EdgeRecord) -> Result<(), MnemeError> {
        let k_out = encode_edge_out_key(e.src, e.relation, e.dst, e.tx_from);
        let k_in = encode_edge_in_key(e.src, e.relation, e.dst, e.tx_from);
        let bytes = bincode::serialize(e).map_err(|e| MnemeError::Storage(e.to_string()))?;
        self.edges_out
            .insert(k_out, bytes.clone())
            .map_err(|err| MnemeError::Storage(err.to_string()))?;
        self.edges_in
            .insert(k_in, bytes)
            .map_err(|err| MnemeError::Storage(err.to_string()))?;
        Ok(())
    }

    /// Re-stamp every live out-edge from `src` with `relation` as
    /// `tx_to = now`. Used by `MemoryLinksUpdated` to retire old
    /// Linked edges before writing the new generation.
    fn close_live_out_edges(
        &self,
        src: MemoryRef,
        relation: Relation,
        now: OffsetDateTime,
    ) -> Result<(), MnemeError> {
        let pref = out_prefix_rel(src, relation);
        let to_close = self.scan_edges(&self.edges_out, &pref, |e| e.tx_to.is_none())?;
        for mut e in to_close {
            e.tx_to = Some(now);
            self.put_edge(&e)?;
        }
        Ok(())
    }

    /// Stamp every live *associative* in/out-edge of `id` as closed.
    /// Used by `MemoryInvalidated`.
    ///
    /// Lineage edges (`EvolvedFrom` / `EvolvedTo`) are deliberately
    /// **left open**: "v2 evolved from v1" is immutable provenance and
    /// stays true forever, even once v1 is invalidated (Hard Rule #2 —
    /// never overwrite history). Tombstoning them would erase the
    /// lineage from the live graph, which is exactly the relationship
    /// the graph exists to preserve. `Linked` and `ContainedIn` edges
    /// *are* closed, since those represent current associations that a
    /// retired memory should no longer participate in.
    fn close_all_incident_edges(
        &self,
        id: MemoryRef,
        now: OffsetDateTime,
    ) -> Result<(), MnemeError> {
        let is_associative = |e: &EdgeRecord| {
            e.tx_to.is_none() && !matches!(e.relation, Relation::EvolvedFrom | Relation::EvolvedTo)
        };
        let out_pref = out_prefix(id);
        let out_open = self.scan_edges(&self.edges_out, &out_pref, is_associative)?;
        for mut e in out_open {
            e.tx_to = Some(now);
            self.put_edge(&e)?;
        }
        let in_pref = in_prefix(id);
        let in_open = self.scan_edges(&self.edges_in, &in_pref, is_associative)?;
        for mut e in in_open {
            e.tx_to = Some(now);
            self.put_edge(&e)?;
        }
        Ok(())
    }

    fn scan_edges(
        &self,
        part: &fjall::PartitionHandle,
        prefix: &[u8],
        keep: impl Fn(&EdgeRecord) -> bool,
    ) -> Result<Vec<EdgeRecord>, MnemeError> {
        let lower = prefix.to_vec();
        let bounds: (Bound<Vec<u8>>, Bound<Vec<u8>>) = match prefix_upper_bound(prefix) {
            Some(upper) => (Bound::Included(lower), Bound::Excluded(upper)),
            None => (Bound::Included(lower), Bound::Unbounded),
        };
        let mut out = Vec::new();
        for kv in part.range(bounds) {
            let (_, v) = kv.map_err(|e| MnemeError::Storage(e.to_string()))?;
            let edge: EdgeRecord =
                bincode::deserialize(&v).map_err(|e| MnemeError::Storage(e.to_string()))?;
            if keep(&edge) {
                out.push(edge);
            }
        }
        Ok(out)
    }

    /// Build the initial NodeRecord from a freshly-written Memory.
    fn node_from_memory(m: &Memory) -> NodeRecord {
        NodeRecord {
            id: MemoryRef(m.id),
            scope: m.scope.clone(),
            tags: m.tags.clone(),
            keywords: m.keywords.clone(),
            source: m.source,
            position: m.position,
            evolution_count: m.evolution_count,
            valid_from: m.time.valid_from,
            valid_to: m.time.valid_to,
            tx_from: m.time.tx_from,
            tx_to: m.time.tx_to,
            label: None,
            is_source: false,
        }
    }

    /// Build a node for a `Source` document so that chunk→source
    /// `ContainedIn` edges resolve to a real endpoint. The source's
    /// ULID is reused as the node id (the graph keys nodes by ULID
    /// regardless of entity kind). The title rides along as `label`.
    fn node_from_source(s: &Source) -> NodeRecord {
        NodeRecord {
            id: MemoryRef(s.id),
            scope: s.scope.clone(),
            tags: vec![],
            keywords: vec![],
            source: None,
            position: None,
            evolution_count: 0,
            valid_from: s.time.valid_from,
            valid_to: s.time.valid_to,
            tx_from: s.time.tx_from,
            tx_to: s.time.tx_to,
            label: Some(s.title.clone()),
            is_source: true,
        }
    }

    fn persist(&self) -> Result<(), MnemeError> {
        self.keyspace
            .persist(fjall::PersistMode::Buffer)
            .map_err(|e| MnemeError::Storage(e.to_string()))
    }

    fn record_checkpoint(&self, id: Id) -> Result<(), MnemeError> {
        self.meta
            .insert(META_LAST_CHECKPOINT, id.to_bytes())
            .map_err(|e| MnemeError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- public query API -------------------------------------------------

    /// Fetch a node by id. Filters by `as_of` (bi-temporal liveness)
    /// when provided, and by `scope` containment in every case.
    pub fn node(
        &self,
        id: MemoryRef,
        scope: &Scope,
        as_of: Option<OffsetDateTime>,
    ) -> Result<Option<NodeRecord>, MnemeError> {
        let Some(n) = self.get_node_raw(id)? else {
            return Ok(None);
        };
        if !scope.contains(&n.scope) {
            return Ok(None);
        }
        if let Some(t) = as_of {
            if !n.is_live_at(t) {
                return Ok(None);
            }
        }
        Ok(Some(n))
    }

    /// All edges flowing *out of* `src`. `relation = None` matches
    /// every relation type.
    ///
    /// The returned edges' endpoints are guaranteed to be scope-visible
    /// to the caller — cross-tenant edges (which shouldn't exist but
    /// are defended against here per Hard Rule #3) are filtered out.
    pub fn out_neighbors(
        &self,
        src: MemoryRef,
        relation: Option<Relation>,
        scope: &Scope,
        as_of: Option<OffsetDateTime>,
    ) -> Result<Vec<EdgeRecord>, MnemeError> {
        self.neighbors(&self.edges_out, src, relation, scope, as_of, true)
    }

    /// All edges flowing *into* `dst`.
    pub fn in_neighbors(
        &self,
        dst: MemoryRef,
        relation: Option<Relation>,
        scope: &Scope,
        as_of: Option<OffsetDateTime>,
    ) -> Result<Vec<EdgeRecord>, MnemeError> {
        self.neighbors(&self.edges_in, dst, relation, scope, as_of, false)
    }

    /// Shared body of `out_neighbors` / `in_neighbors`. `out` controls
    /// which side of the edge we scope-check (we already checked the
    /// origin via the caller; the other endpoint is the one we have
    /// to look up).
    fn neighbors(
        &self,
        part: &fjall::PartitionHandle,
        node: MemoryRef,
        relation: Option<Relation>,
        scope: &Scope,
        as_of: Option<OffsetDateTime>,
        out: bool,
    ) -> Result<Vec<EdgeRecord>, MnemeError> {
        // Cheap origin-side scope check — refuses to traverse if the
        // origin node is outside the caller's scope.
        if let Some(origin) = self.get_node_raw(node)? {
            if !scope.contains(&origin.scope) {
                return Ok(Vec::new());
            }
            if let Some(t) = as_of {
                if !origin.is_live_at(t) {
                    return Ok(Vec::new());
                }
            }
        } else {
            return Ok(Vec::new());
        }
        let prefix: Vec<u8> = match relation {
            Some(r) => {
                let p = if out {
                    out_prefix_rel(node, r)
                } else {
                    in_prefix_rel(node, r)
                };
                p.to_vec()
            }
            None => {
                let p = if out {
                    out_prefix(node)
                } else {
                    in_prefix(node)
                };
                p.to_vec()
            }
        };
        let edges = self.scan_edges(part, &prefix, |e| match as_of {
            Some(t) => e.is_live_at(t),
            None => e.tx_to.is_none(),
        })?;
        // Scope-filter the *other* endpoint. Same Hard Rule #3
        // defence as for the origin.
        let mut out_vec = Vec::with_capacity(edges.len());
        for e in edges {
            let other = if out { e.dst } else { e.src };
            if let Some(other_node) = self.get_node_raw(other)? {
                if !scope.contains(&other_node.scope) {
                    continue;
                }
                if let Some(t) = as_of {
                    if !other_node.is_live_at(t) {
                        continue;
                    }
                }
            } else {
                // Edge points at an unknown node — drop. This is the
                // safe default rather than surface a half-resolved
                // graph to the caller.
                continue;
            }
            out_vec.push(e);
        }
        Ok(out_vec)
    }

    /// Candidate starting nodes for a UI, ranked by live degree
    /// (out + in) so the most-connected node sorts first — a good
    /// default focus for a neighborhood view.
    ///
    /// Bounded work: scans at most `MAX_ROOT_SCAN` nodes and computes
    /// each one's degree via the same scope-filtered neighbour calls a
    /// real query would use. This is a dashboard convenience, not a
    /// hot path; for large stores prefer querying a known id directly.
    pub fn roots(
        &self,
        scope: &Scope,
        as_of: Option<OffsetDateTime>,
        limit: usize,
    ) -> Result<Vec<NodeDegree>, MnemeError> {
        const MAX_ROOT_SCAN: usize = 1_000;
        let mut out: Vec<NodeDegree> = Vec::new();
        for (scanned, kv) in self.nodes.iter().enumerate() {
            if scanned >= MAX_ROOT_SCAN {
                break;
            }
            let (_, v) = kv.map_err(|e| MnemeError::Storage(e.to_string()))?;
            let node: NodeRecord =
                bincode::deserialize(&v).map_err(|e| MnemeError::Storage(e.to_string()))?;
            if !scope.contains(&node.scope) {
                continue;
            }
            match as_of {
                Some(t) if !node.is_live_at(t) => continue,
                None if node.tx_to.is_some() => continue,
                _ => {}
            }
            let out_degree = self.out_neighbors(node.id, None, scope, as_of)?.len();
            let in_degree = self.in_neighbors(node.id, None, scope, as_of)?.len();
            out.push(NodeDegree {
                node,
                out_degree,
                in_degree,
            });
        }
        out.sort_by(|a, b| {
            (b.out_degree + b.in_degree)
                .cmp(&(a.out_degree + a.in_degree))
                // Stable tiebreak on id so the order is deterministic
                // across calls (Hard Rule #4 — replay-stable).
                .then_with(|| b.node.id.0.cmp(&a.node.id.0))
        });
        out.truncate(limit);
        Ok(out)
    }

    /// Total nodes + edges in the store. Useful for dashboard counters
    /// and the "size on disk" sanity check.
    pub fn stats(&self) -> Result<GraphStats, MnemeError> {
        let node_count = self.nodes.iter().filter_map(|kv| kv.ok()).count();
        let mut edge_count = 0usize;
        let mut live_edges = 0usize;
        for kv in self.edges_out.iter() {
            let (_, v) = kv.map_err(|e| MnemeError::Storage(e.to_string()))?;
            edge_count += 1;
            if let Ok(edge) = bincode::deserialize::<EdgeRecord>(&v) {
                if edge.tx_to.is_none() {
                    live_edges += 1;
                }
            }
        }
        Ok(GraphStats {
            node_count,
            edge_count,
            live_edge_count: live_edges,
        })
    }
}

/// A node plus its live degree, returned by [`FjallGraphView::roots`].
#[derive(Debug, Clone)]
pub struct NodeDegree {
    pub node: NodeRecord,
    pub out_degree: usize,
    pub in_degree: usize,
}

/// Per-store counters surfaced for dashboards / health checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphStats {
    pub node_count: usize,
    /// Total edges including tombstoned versions — the on-disk count.
    pub edge_count: usize,
    /// Edges that are still open (no `tx_to`). The dashboard plots
    /// `live_edge_count / edge_count` as a "graph health" ratio.
    pub live_edge_count: usize,
}

#[async_trait]
impl MaterializedView for FjallGraphView {
    fn name(&self) -> &str {
        "graph-view"
    }

    async fn apply(&self, entry: &LogEntry) -> Result<(), MnemeError> {
        let now = OffsetDateTime::now_utc();
        match &entry.event {
            Event::MemoryWritten(m) => {
                let n = Self::node_from_memory(m);
                self.put_node(&n)?;
                // Initial Linked edges + ContainedIn — both arrive
                // with `tx_from = now` so replaying the log yields
                // identical edges modulo wall-clock skew (which we
                // accept; the order is what matters for replay).
                for dst in &m.links {
                    let edge = EdgeRecord {
                        src: MemoryRef(m.id),
                        relation: Relation::Linked,
                        dst: *dst,
                        tx_from: now,
                        tx_to: None,
                        weight: None,
                    };
                    self.put_edge(&edge)?;
                }
                if let Some(src_doc) = m.source {
                    // The ContainedIn edge points at the source node
                    // (materialized by `SourceIngested`). If the source
                    // was never ingested as an event it stays a phantom
                    // endpoint and `neighbors()` filters it out — the
                    // edge is harmless either way.
                    let edge = EdgeRecord {
                        src: MemoryRef(m.id),
                        relation: Relation::ContainedIn,
                        dst: MemoryRef(src_doc.0),
                        tx_from: now,
                        tx_to: None,
                        weight: None,
                    };
                    self.put_edge(&edge)?;
                }
            }
            Event::MemoryLinksUpdated { id, links } => {
                // Hard Rule #2 — never overwrite. Close prior Linked
                // edges with `tx_to = now`, then open a new
                // generation. Replay produces the same tx-intervals
                // in the same order.
                self.close_live_out_edges(*id, Relation::Linked, now)?;
                for dst in links {
                    let edge = EdgeRecord {
                        src: *id,
                        relation: Relation::Linked,
                        dst: *dst,
                        tx_from: now,
                        tx_to: None,
                        weight: None,
                    };
                    self.put_edge(&edge)?;
                }
            }
            Event::MemoryNoteEnriched {
                id,
                keywords,
                tags,
                context: _,
            } => {
                // Mutates derived node properties without touching
                // lineage. If the node hasn't landed yet (event order
                // weirdness), we skip rather than synthesize a
                // half-empty node.
                if let Some(mut n) = self.get_node_raw(*id)? {
                    n.keywords = keywords.clone();
                    n.tags = tags.clone();
                    self.put_node(&n)?;
                }
            }
            Event::MemoryEvolved { from, to, .. } => {
                // Two lineage edges: `to -> from` (EvolvedFrom, so
                // "what did this evolve from?" is a prefix scan from
                // `to`) and `from -> to` (EvolvedTo).
                let e1 = EdgeRecord {
                    src: *to,
                    relation: Relation::EvolvedFrom,
                    dst: *from,
                    tx_from: now,
                    tx_to: None,
                    weight: None,
                };
                let e2 = EdgeRecord {
                    src: *from,
                    relation: Relation::EvolvedTo,
                    dst: *to,
                    tx_from: now,
                    tx_to: None,
                    weight: None,
                };
                self.put_edge(&e1)?;
                self.put_edge(&e2)?;
                // Mark the parent's valid_to so bi-temporal `as_of`
                // queries don't return both versions live at once.
                if let Some(mut parent) = self.get_node_raw(*from)? {
                    if parent.valid_to.is_none() {
                        parent.valid_to = Some(now);
                    }
                    self.put_node(&parent)?;
                }
            }
            Event::MemoryInvalidated { id, reason: _ } => {
                // Tombstone in tx-time (Hard Rule #2 — we don't drop
                // the row, we close the interval).
                if let Some(mut n) = self.get_node_raw(*id)? {
                    if n.tx_to.is_none() {
                        n.tx_to = Some(now);
                    }
                    self.put_node(&n)?;
                }
                self.close_all_incident_edges(*id, now)?;
            }
            Event::SourceIngested(src) => {
                // Materialize the document as a node so the chunks'
                // `ContainedIn` edges resolve to a real endpoint
                // instead of dangling on a phantom. Chunks may arrive
                // before or after this event; either order converges to
                // the same graph (Hard Rule #4).
                let n = Self::node_from_source(src);
                self.put_node(&n)?;
            }
            Event::SourceInvalidated { id, reason: _ } => {
                // Tombstone the source node + close its associative
                // edges (the chunks' `ContainedIn` edges point *in* to
                // it). Bi-temporal: we close the interval, never drop.
                let src_ref = MemoryRef(id.0);
                if let Some(mut n) = self.get_node_raw(src_ref)? {
                    if n.tx_to.is_none() {
                        n.tx_to = Some(now);
                    }
                    self.put_node(&n)?;
                }
                self.close_all_incident_edges(src_ref, now)?;
            }
            // Everything else doesn't shape the graph.
            _ => {}
        }
        self.record_checkpoint(entry.id)?;
        self.persist()?;
        Ok(())
    }

    async fn checkpoint(&self) -> Result<Option<Id>, MnemeError> {
        let Some(v) = self
            .meta
            .get(META_LAST_CHECKPOINT)
            .map_err(|e| MnemeError::Storage(e.to_string()))?
        else {
            return Ok(None);
        };
        if v.len() != 16 {
            return Err(MnemeError::Storage(format!(
                "graph checkpoint payload was {} bytes, expected 16",
                v.len()
            )));
        }
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&v);
        Ok(Some(Id::from_bytes(bytes)))
    }
}
