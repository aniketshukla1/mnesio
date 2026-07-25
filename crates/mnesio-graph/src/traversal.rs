//! Bounded BFS + shortest-path on top of [`FjallGraphView`].
//!
//! Both primitives respect:
//! - **Hard Rule #3** — every node visited is checked through the
//!   caller's [`Scope`].
//! - **Bi-temporality** — `as_of` flows through every neighbour
//!   fetch.
//! - **Cascade bounds (Hard Rule #6)** — `max_depth` is a hard cap,
//!   not a heuristic; the visited set additionally prevents loops.

use crate::record::{NodeRecord, Relation};
use crate::view::FjallGraphView;
use mnesio_core::types::{MemoryRef, Scope};
use mnesio_core::MnesioError;
use std::collections::{HashMap, HashSet, VecDeque};
use time::OffsetDateTime;

/// One step in a traversal result. The hop count is the BFS depth at
/// which we reached this node — the start node has `hop = 0`.
#[derive(Debug, Clone)]
pub struct Hop {
    pub node: NodeRecord,
    pub hop: u16,
}

/// A directed path between two nodes, oriented from `start` to `end`.
#[derive(Debug, Clone)]
pub struct Path {
    pub nodes: Vec<MemoryRef>,
    pub relations: Vec<Relation>,
}

impl Path {
    pub fn len(&self) -> usize {
        self.relations.len()
    }
    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }
}

impl FjallGraphView {
    /// Breadth-first traversal of the directed graph, walking out-
    /// edges only. Results are returned in BFS order — `start` first,
    /// then all 1-hop neighbours, etc. — which matches what callers
    /// usually want for "show me the neighbourhood of X".
    ///
    /// `max_depth = 0` returns just `start` (if it's in scope and
    /// live at `as_of`). `max_depth = u16::MAX` is still bounded
    /// because the visited set caps the work at the size of the
    /// in-scope subgraph.
    pub fn bfs(
        &self,
        start: MemoryRef,
        max_depth: u16,
        scope: &Scope,
        as_of: Option<OffsetDateTime>,
    ) -> Result<Vec<Hop>, MnesioError> {
        let Some(start_node) = self.node(start, scope, as_of)? else {
            return Ok(Vec::new());
        };
        let mut out: Vec<Hop> = vec![Hop {
            node: start_node,
            hop: 0,
        }];
        let mut visited: HashSet<MemoryRef> = HashSet::from([start]);
        let mut frontier: VecDeque<(MemoryRef, u16)> = VecDeque::from([(start, 0)]);
        while let Some((node, depth)) = frontier.pop_front() {
            if depth >= max_depth {
                continue;
            }
            // Every relation type — callers who want a specific
            // relation use `out_neighbors` directly.
            let edges = self.out_neighbors(node, None, scope, as_of)?;
            for e in edges {
                if !visited.insert(e.dst) {
                    continue;
                }
                if let Some(n) = self.node(e.dst, scope, as_of)? {
                    out.push(Hop {
                        node: n,
                        hop: depth + 1,
                    });
                    frontier.push_back((e.dst, depth + 1));
                }
            }
        }
        Ok(out)
    }

    /// Shortest directed path from `src` to `dst` through out-edges,
    /// or `None` if unreachable within `max_depth`. Ties are broken
    /// by edge-insertion order, which is itself stable across replays
    /// (Hard Rule #4).
    pub fn shortest_path(
        &self,
        src: MemoryRef,
        dst: MemoryRef,
        max_depth: u16,
        scope: &Scope,
        as_of: Option<OffsetDateTime>,
    ) -> Result<Option<Path>, MnesioError> {
        if src == dst {
            // Caller asked for a path to the source. Return the
            // degenerate zero-edge path.
            if self.node(src, scope, as_of)?.is_some() {
                return Ok(Some(Path {
                    nodes: vec![src],
                    relations: vec![],
                }));
            }
            return Ok(None);
        }
        if self.node(src, scope, as_of)?.is_none() {
            return Ok(None);
        }
        let mut parent: HashMap<MemoryRef, (MemoryRef, Relation)> = HashMap::new();
        let mut visited: HashSet<MemoryRef> = HashSet::from([src]);
        let mut frontier: VecDeque<(MemoryRef, u16)> = VecDeque::from([(src, 0)]);
        while let Some((node, depth)) = frontier.pop_front() {
            if depth >= max_depth {
                continue;
            }
            for e in self.out_neighbors(node, None, scope, as_of)? {
                if !visited.insert(e.dst) {
                    continue;
                }
                parent.insert(e.dst, (node, e.relation));
                if e.dst == dst {
                    // Walk parent map back to src.
                    let mut nodes = vec![dst];
                    let mut rels = vec![e.relation];
                    let mut cur = node;
                    while cur != src {
                        let (p, r) = parent
                            .get(&cur)
                            .copied()
                            .expect("BFS parent must exist for visited node");
                        nodes.push(cur);
                        rels.push(r);
                        cur = p;
                    }
                    nodes.push(src);
                    nodes.reverse();
                    rels.reverse();
                    return Ok(Some(Path {
                        nodes,
                        relations: rels,
                    }));
                }
                frontier.push_back((e.dst, depth + 1));
            }
        }
        Ok(None)
    }
}
