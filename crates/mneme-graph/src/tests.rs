//! End-to-end tests for the graph view. Touches real fjall (temp dir
//! per test) so we exercise persist + replay rather than just the
//! in-memory mock surface.

#![cfg(test)]

use crate::record::Relation;
use crate::view::FjallGraphView;
use mneme_core::entity::{Memory, Provenance};
use mneme_core::event::{ChangeSet, Event, LogEntry};
use mneme_core::traits::MaterializedView;
use mneme_core::types::{new_id, BiTemporal, MemoryRef, Scope, SourceRef};

fn temp_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("mneme-graph-test-{}", new_id()))
}

fn mem(tenant: &str) -> Memory {
    Memory {
        id: new_id(),
        scope: Scope::global(tenant),
        content: "x".into(),
        keywords: vec![],
        tags: vec![],
        context: String::new(),
        embedding: None,
        links: vec![],
        parent: None,
        evolution_count: 0,
        time: BiTemporal::now(),
        provenance: Provenance::default(),
        source: None,
        position: None,
    }
}

fn entry(event: Event) -> LogEntry {
    LogEntry {
        id: new_id(),
        event,
    }
}

#[tokio::test]
async fn memory_written_creates_node_and_initial_link_edges() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let mut a = mem("t");
    let b = mem("t");
    a.links = vec![MemoryRef(b.id)];
    let a_ref = MemoryRef(a.id);
    let b_ref = MemoryRef(b.id);
    v.apply(&entry(Event::MemoryWritten(b))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
    let n = v.node(a_ref, &Scope::global("t"), None).unwrap();
    assert!(n.is_some());
    let outs = v
        .out_neighbors(a_ref, Some(Relation::Linked), &Scope::global("t"), None)
        .unwrap();
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].dst, b_ref);
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn memory_links_updated_closes_old_and_opens_new_generation() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let mut a = mem("t");
    let b = mem("t");
    let c = mem("t");
    a.links = vec![MemoryRef(b.id)];
    let a_ref = MemoryRef(a.id);
    let b_ref = MemoryRef(b.id);
    let c_ref = MemoryRef(c.id);
    v.apply(&entry(Event::MemoryWritten(b))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(c))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();

    // Re-link a to c instead of b.
    v.apply(&entry(Event::MemoryLinksUpdated {
        id: a_ref,
        links: vec![c_ref],
    }))
    .await
    .unwrap();

    // Live view: a → c only.
    let live = v
        .out_neighbors(a_ref, Some(Relation::Linked), &Scope::global("t"), None)
        .unwrap();
    let live_dsts: Vec<_> = live.iter().map(|e| e.dst).collect();
    assert_eq!(live_dsts, vec![c_ref]);

    // Historical view (without as_of arg the old edge is closed) —
    // but if we feed an earlier timestamp the old edge should still
    // appear live. We use the b's tx_from for that.
    // (Both edges share roughly "now"; for the time-travel test we
    // rely on the dedicated test below.)
    let _ = b_ref;
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn cross_tenant_traversal_is_blocked_even_with_smuggled_edge() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let mut a = mem("a");
    let b = mem("b"); // different tenant
    a.links = vec![MemoryRef(b.id)]; // smuggled cross-tenant link
    let a_ref = MemoryRef(a.id);
    v.apply(&entry(Event::MemoryWritten(b))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();

    // Tenant-a caller asking for a's neighbours must NOT see b.
    let neighbours = v
        .out_neighbors(a_ref, None, &Scope::global("a"), None)
        .unwrap();
    assert!(
        neighbours.is_empty(),
        "scope filter must hide cross-tenant endpoint"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn memory_evolved_writes_both_lineage_directions() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let from = mem("t");
    let to = mem("t");
    let from_ref = MemoryRef(from.id);
    let to_ref = MemoryRef(to.id);
    v.apply(&entry(Event::MemoryWritten(from))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(to))).await.unwrap();
    v.apply(&entry(Event::MemoryEvolved {
        from: from_ref,
        to: to_ref,
        diff: ChangeSet {
            keywords_added: vec![],
            keywords_removed: vec![],
            tags_added: vec![],
            tags_removed: vec![],
            context_rewritten: false,
        },
    }))
    .await
    .unwrap();
    let lineage_back = v
        .out_neighbors(
            to_ref,
            Some(Relation::EvolvedFrom),
            &Scope::global("t"),
            None,
        )
        .unwrap();
    assert_eq!(lineage_back.len(), 1);
    assert_eq!(lineage_back[0].dst, from_ref);
    let lineage_forward = v
        .out_neighbors(
            from_ref,
            Some(Relation::EvolvedTo),
            &Scope::global("t"),
            None,
        )
        .unwrap();
    assert_eq!(lineage_forward.len(), 1);
    assert_eq!(lineage_forward[0].dst, to_ref);
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn memory_invalidated_tombstones_node_and_incident_edges() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let mut a = mem("t");
    let b = mem("t");
    a.links = vec![MemoryRef(b.id)];
    let a_ref = MemoryRef(a.id);
    let b_ref = MemoryRef(b.id);
    v.apply(&entry(Event::MemoryWritten(b))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
    v.apply(&entry(Event::MemoryInvalidated {
        id: a_ref,
        reason: "wrong".into(),
    }))
    .await
    .unwrap();

    // a is tombstoned — node lookup with no as_of still returns the
    // record (no time predicate), but it's marked tx_to=Some.
    let a_node = v.node(a_ref, &Scope::global("t"), None).unwrap().unwrap();
    assert!(
        a_node.tx_to.is_some(),
        "invalidated node must have tx_to set"
    );

    // Live out-neighbours from a (no as_of) must be empty.
    let lives = v
        .out_neighbors(a_ref, None, &Scope::global("t"), None)
        .unwrap();
    assert!(lives.is_empty(), "tombstoned edges should not appear live");

    // b is untouched.
    let b_node = v.node(b_ref, &Scope::global("t"), None).unwrap().unwrap();
    assert!(b_node.tx_to.is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn lineage_edges_survive_parent_invalidation() {
    // The demo's evolution chain: write parent, write child, evolve
    // parent→child, then invalidate the parent. Hard Rule #2 says the
    // "child evolved from parent" provenance must persist in the *live*
    // graph even though the parent is now tombstoned — only the
    // associative (Linked/ContainedIn) edges get closed.
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let mut parent = mem("t");
    let child = mem("t");
    let bystander = mem("t");
    // Parent also has an associative Linked edge — that one *should*
    // close on invalidation, unlike the lineage edges.
    parent.links = vec![MemoryRef(bystander.id)];
    let parent_ref = MemoryRef(parent.id);
    let child_ref = MemoryRef(child.id);
    let bystander_ref = MemoryRef(bystander.id);

    v.apply(&entry(Event::MemoryWritten(bystander)))
        .await
        .unwrap();
    v.apply(&entry(Event::MemoryWritten(parent))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(child))).await.unwrap();
    v.apply(&entry(Event::MemoryEvolved {
        from: parent_ref,
        to: child_ref,
        diff: ChangeSet {
            keywords_added: vec![],
            keywords_removed: vec![],
            tags_added: vec![],
            tags_removed: vec![],
            context_rewritten: false,
        },
    }))
    .await
    .unwrap();
    v.apply(&entry(Event::MemoryInvalidated {
        id: parent_ref,
        reason: "superseded by evolution".into(),
    }))
    .await
    .unwrap();

    // Live: the child still records that it evolved from the parent.
    let from_child = v
        .out_neighbors(
            child_ref,
            Some(Relation::EvolvedFrom),
            &Scope::global("t"),
            None,
        )
        .unwrap();
    assert_eq!(
        from_child.len(),
        1,
        "child→parent EvolvedFrom must survive parent invalidation"
    );
    assert_eq!(from_child[0].dst, parent_ref);

    // Live: the (tombstoned) parent still records the forward lineage.
    let to_child = v
        .out_neighbors(
            parent_ref,
            Some(Relation::EvolvedTo),
            &Scope::global("t"),
            None,
        )
        .unwrap();
    assert_eq!(
        to_child.len(),
        1,
        "parent→child EvolvedTo must survive parent invalidation"
    );

    // But the parent's associative Linked edge IS closed.
    let linked = v
        .out_neighbors(
            parent_ref,
            Some(Relation::Linked),
            &Scope::global("t"),
            None,
        )
        .unwrap();
    assert!(
        linked.is_empty(),
        "associative Linked edge should still tombstone on invalidation"
    );
    let _ = bystander_ref;
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn as_of_query_returns_historical_neighborhood() {
    use time::{Duration, OffsetDateTime};
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let mut a = mem("t");
    let b = mem("t");
    let c = mem("t");
    a.links = vec![MemoryRef(b.id)];
    let a_ref = MemoryRef(a.id);
    let b_ref = MemoryRef(b.id);
    let c_ref = MemoryRef(c.id);
    v.apply(&entry(Event::MemoryWritten(b))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(c))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();

    let t_after_first_gen = OffsetDateTime::now_utc();

    // Wait a tick (we manipulate tx-time by the wall clock here —
    // good enough for a single-process test).
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    v.apply(&entry(Event::MemoryLinksUpdated {
        id: a_ref,
        links: vec![c_ref],
    }))
    .await
    .unwrap();

    // Live: a → c. Historical at t_after_first_gen: a → b.
    let now = v
        .out_neighbors(a_ref, Some(Relation::Linked), &Scope::global("t"), None)
        .unwrap();
    let now_dsts: Vec<_> = now.iter().map(|e| e.dst).collect();
    assert_eq!(now_dsts, vec![c_ref]);

    let then = v
        .out_neighbors(
            a_ref,
            Some(Relation::Linked),
            &Scope::global("t"),
            Some(t_after_first_gen + Duration::milliseconds(1)),
        )
        .unwrap();
    let then_dsts: Vec<_> = then.iter().map(|e| e.dst).collect();
    assert_eq!(
        then_dsts,
        vec![b_ref],
        "as_of must reproduce the prior generation of Linked edges"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn bfs_respects_max_depth_bound() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    // Chain a -> b -> c -> d, all in tenant t.
    let mut a = mem("t");
    let mut b = mem("t");
    let mut c = mem("t");
    let d = mem("t");
    a.links = vec![MemoryRef(b.id)];
    b.links = vec![MemoryRef(c.id)];
    c.links = vec![MemoryRef(d.id)];
    let a_ref = MemoryRef(a.id);
    let b_ref = MemoryRef(b.id);
    let c_ref = MemoryRef(c.id);
    let d_ref = MemoryRef(d.id);
    // Write d first so subsequent links resolve to existing nodes.
    v.apply(&entry(Event::MemoryWritten(d))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(c))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(b))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();

    let hops0 = v.bfs(a_ref, 0, &Scope::global("t"), None).unwrap();
    assert_eq!(hops0.len(), 1);
    assert_eq!(hops0[0].node.id, a_ref);

    let hops2 = v.bfs(a_ref, 2, &Scope::global("t"), None).unwrap();
    let ids: Vec<_> = hops2.iter().map(|h| h.node.id).collect();
    assert!(ids.contains(&b_ref));
    assert!(ids.contains(&c_ref));
    assert!(!ids.contains(&d_ref), "depth-2 BFS must not reach d");

    let hops3 = v.bfs(a_ref, 3, &Scope::global("t"), None).unwrap();
    let ids: Vec<_> = hops3.iter().map(|h| h.node.id).collect();
    assert!(ids.contains(&d_ref), "depth-3 BFS must reach d");
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn bfs_visited_set_prevents_loops() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    // a <-> b cycle.
    let mut a = mem("t");
    let mut b = mem("t");
    a.links = vec![MemoryRef(b.id)];
    b.links = vec![MemoryRef(a.id)];
    let a_ref = MemoryRef(a.id);
    let b_ref = MemoryRef(b.id);
    v.apply(&entry(Event::MemoryWritten(b))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
    // Depth 100 must still terminate.
    let hops = v.bfs(a_ref, 100, &Scope::global("t"), None).unwrap();
    let ids: Vec<_> = hops.iter().map(|h| h.node.id).collect();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&a_ref));
    assert!(ids.contains(&b_ref));
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn shortest_path_returns_none_when_unreachable() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let a = mem("t");
    let b = mem("t"); // no edges
    let a_ref = MemoryRef(a.id);
    let b_ref = MemoryRef(b.id);
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(b))).await.unwrap();
    let p = v
        .shortest_path(a_ref, b_ref, 5, &Scope::global("t"), None)
        .unwrap();
    assert!(p.is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn shortest_path_finds_direct_link() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let mut a = mem("t");
    let b = mem("t");
    a.links = vec![MemoryRef(b.id)];
    let a_ref = MemoryRef(a.id);
    let b_ref = MemoryRef(b.id);
    v.apply(&entry(Event::MemoryWritten(b))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
    let p = v
        .shortest_path(a_ref, b_ref, 5, &Scope::global("t"), None)
        .unwrap()
        .expect("path exists");
    assert_eq!(p.len(), 1);
    assert_eq!(p.nodes, vec![a_ref, b_ref]);
    assert_eq!(p.relations, vec![Relation::Linked]);
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn shortest_path_self_is_zero_length() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let a = mem("t");
    let a_ref = MemoryRef(a.id);
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
    let p = v
        .shortest_path(a_ref, a_ref, 5, &Scope::global("t"), None)
        .unwrap()
        .expect("self path");
    assert!(p.is_empty());
    assert_eq!(p.nodes, vec![a_ref]);
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn contained_in_edge_links_chunk_to_source() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let mut chunk = mem("t");
    let src_id = new_id();
    chunk.source = Some(SourceRef(src_id));
    chunk.position = Some(0);
    let chunk_ref = MemoryRef(chunk.id);
    v.apply(&entry(Event::MemoryWritten(chunk))).await.unwrap();
    // The source itself never landed as MemoryWritten, so the
    // ContainedIn edge points at a phantom node. The neighbours call
    // must therefore filter the dst out (defensive), and the
    // scope-aware out_neighbors returns empty.
    let outs = v
        .out_neighbors(
            chunk_ref,
            Some(Relation::ContainedIn),
            &Scope::global("t"),
            None,
        )
        .unwrap();
    assert!(
        outs.is_empty(),
        "phantom source endpoint should be filtered"
    );
    std::fs::remove_dir_all(&dir).ok();
}

fn source(tenant: &str, title: &str) -> mneme_core::entity::Source {
    mneme_core::entity::Source {
        id: new_id(),
        scope: Scope::global(tenant),
        title: title.into(),
        uri: None,
        chunk_count: 0,
        time: BiTemporal::now(),
        provenance: Provenance::default(),
    }
}

#[tokio::test]
async fn contained_in_resolves_once_source_is_ingested() {
    // With a real source node present, the chunk's ContainedIn edge
    // resolves to it (no longer filtered as a phantom).
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let src = source("t", "Q3 Board Memo");
    let src_ref = MemoryRef(src.id);
    let mut chunk = mem("t");
    chunk.source = Some(SourceRef(src.id));
    chunk.position = Some(0);
    let chunk_ref = MemoryRef(chunk.id);

    v.apply(&entry(Event::SourceIngested(src))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(chunk))).await.unwrap();

    // The source node exists and is flagged.
    let src_node = v.node(src_ref, &Scope::global("t"), None).unwrap().unwrap();
    assert!(src_node.is_source);
    assert_eq!(src_node.label.as_deref(), Some("Q3 Board Memo"));

    // The chunk's ContainedIn edge now resolves to the source.
    let outs = v
        .out_neighbors(
            chunk_ref,
            Some(Relation::ContainedIn),
            &Scope::global("t"),
            None,
        )
        .unwrap();
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].dst, src_ref);

    // And the source sees the chunk as an in-neighbor.
    let ins = v
        .in_neighbors(
            src_ref,
            Some(Relation::ContainedIn),
            &Scope::global("t"),
            None,
        )
        .unwrap();
    assert_eq!(ins.len(), 1);
    assert_eq!(ins[0].src, chunk_ref);
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn source_invalidated_tombstones_source_node_and_contained_in() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let src = source("t", "Retracted Memo");
    let src_ref = MemoryRef(src.id);
    let mut chunk = mem("t");
    chunk.source = Some(SourceRef(src.id));
    let chunk_ref = MemoryRef(chunk.id);
    v.apply(&entry(Event::SourceIngested(src))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(chunk))).await.unwrap();
    v.apply(&entry(Event::SourceInvalidated {
        id: SourceRef(src_ref.0),
        reason: "retracted".into(),
    }))
    .await
    .unwrap();

    // Source node tombstoned.
    let n = v.node(src_ref, &Scope::global("t"), None).unwrap().unwrap();
    assert!(n.tx_to.is_some());
    // ContainedIn edges no longer live.
    let ins = v
        .in_neighbors(
            src_ref,
            Some(Relation::ContainedIn),
            &Scope::global("t"),
            None,
        )
        .unwrap();
    assert!(ins.is_empty());
    let _ = chunk_ref;
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn checkpoint_advances_on_every_apply() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    assert!(v.checkpoint().await.unwrap().is_none());
    let e = entry(Event::MemoryWritten(mem("t")));
    let last = e.id;
    v.apply(&e).await.unwrap();
    let cp = v.checkpoint().await.unwrap();
    assert_eq!(cp, Some(last));
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn replay_into_fresh_store_produces_equivalent_node_set() {
    // Hard Rule #4 — drop and rebuild.
    let dir1 = temp_dir();
    let dir2 = temp_dir();
    let mut a = mem("t");
    let b = mem("t");
    a.links = vec![MemoryRef(b.id)];
    let a_ref = MemoryRef(a.id);
    let b_ref = MemoryRef(b.id);
    let events = vec![
        entry(Event::MemoryWritten(b.clone())),
        entry(Event::MemoryWritten(a.clone())),
        entry(Event::MemoryLinksUpdated {
            id: a_ref,
            links: vec![b_ref], // re-affirm
        }),
    ];
    let v1 = FjallGraphView::open(&dir1).unwrap();
    let v2 = FjallGraphView::open(&dir2).unwrap();
    for e in &events {
        v1.apply(e).await.unwrap();
    }
    for e in &events {
        v2.apply(e).await.unwrap();
    }
    let s1 = v1.stats().unwrap();
    let s2 = v2.stats().unwrap();
    assert_eq!(s1.node_count, s2.node_count);
    assert_eq!(s1.edge_count, s2.edge_count);
    // Both should have the same node ids reachable.
    let n_a = v1.node(a_ref, &Scope::global("t"), None).unwrap().unwrap();
    let n_a2 = v2.node(a_ref, &Scope::global("t"), None).unwrap().unwrap();
    assert_eq!(n_a.scope, n_a2.scope);
    std::fs::remove_dir_all(&dir1).ok();
    std::fs::remove_dir_all(&dir2).ok();
}

#[tokio::test]
async fn note_enriched_updates_keywords_and_tags_only() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let a = mem("t");
    let a_ref = MemoryRef(a.id);
    let a_scope = a.scope.clone();
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
    v.apply(&entry(Event::MemoryNoteEnriched {
        id: a_ref,
        keywords: vec!["k1".into(), "k2".into()],
        tags: vec!["tag1".into()],
        context: "ignored by graph".into(),
    }))
    .await
    .unwrap();
    let n = v.node(a_ref, &Scope::global("t"), None).unwrap().unwrap();
    assert_eq!(n.keywords, vec!["k1".to_string(), "k2".to_string()]);
    assert_eq!(n.tags, vec!["tag1".to_string()]);
    // Scope is untouched.
    assert_eq!(n.scope, a_scope);
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn stats_counts_live_vs_total_edges() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let mut a = mem("t");
    let b = mem("t");
    let c = mem("t");
    a.links = vec![MemoryRef(b.id)];
    let a_ref = MemoryRef(a.id);
    let c_ref = MemoryRef(c.id);
    v.apply(&entry(Event::MemoryWritten(b))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(c))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
    v.apply(&entry(Event::MemoryLinksUpdated {
        id: a_ref,
        links: vec![c_ref],
    }))
    .await
    .unwrap();
    let s = v.stats().unwrap();
    assert_eq!(s.node_count, 3);
    assert_eq!(s.edge_count, 2, "old + new Linked edges both on disk");
    assert_eq!(s.live_edge_count, 1, "only the new Linked edge is open");
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn roots_ranks_by_degree_and_respects_scope() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    // `hub` is linked-to by two others and links out to one — degree 3.
    // `leaf` has no edges — degree 0. A different-tenant node must not
    // appear in tenant-t roots at all.
    let hub = mem("t");
    let leaf = mem("t");
    let other_tenant = mem("z");
    let hub_ref = MemoryRef(hub.id);
    let mut caller_a = mem("t");
    let mut caller_b = mem("t");
    let tail = mem("t");
    caller_a.links = vec![hub_ref];
    caller_b.links = vec![hub_ref];
    let mut hub = hub;
    hub.links = vec![MemoryRef(tail.id)];

    v.apply(&entry(Event::MemoryWritten(tail))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(hub))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(leaf))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(caller_a)))
        .await
        .unwrap();
    v.apply(&entry(Event::MemoryWritten(caller_b)))
        .await
        .unwrap();
    v.apply(&entry(Event::MemoryWritten(other_tenant)))
        .await
        .unwrap();

    let roots = v.roots(&Scope::global("t"), None, 10).unwrap();
    // Every root is in tenant t — the z-tenant node is filtered.
    assert!(roots.iter().all(|r| r.node.scope.tenant == "t"));
    // The hub (degree 3) sorts first.
    assert_eq!(roots[0].node.id, hub_ref);
    assert_eq!(roots[0].out_degree, 1);
    assert_eq!(roots[0].in_degree, 2);
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn roots_honors_limit() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    for _ in 0..5 {
        v.apply(&entry(Event::MemoryWritten(mem("t"))))
            .await
            .unwrap();
    }
    let roots = v.roots(&Scope::global("t"), None, 2).unwrap();
    assert_eq!(roots.len(), 2);
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn in_neighbors_returns_callers_to_a_node() {
    let dir = temp_dir();
    let v = FjallGraphView::open(&dir).unwrap();
    let target = mem("t");
    let target_ref = MemoryRef(target.id);
    let mut a = mem("t");
    let mut b = mem("t");
    a.links = vec![target_ref];
    b.links = vec![target_ref];
    let a_ref = MemoryRef(a.id);
    let b_ref = MemoryRef(b.id);
    v.apply(&entry(Event::MemoryWritten(target))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(a))).await.unwrap();
    v.apply(&entry(Event::MemoryWritten(b))).await.unwrap();
    let in_edges = v
        .in_neighbors(
            target_ref,
            Some(Relation::Linked),
            &Scope::global("t"),
            None,
        )
        .unwrap();
    let srcs: Vec<_> = in_edges.iter().map(|e| e.src).collect();
    assert_eq!(srcs.len(), 2);
    assert!(srcs.contains(&a_ref));
    assert!(srcs.contains(&b_ref));
    std::fs::remove_dir_all(&dir).ok();
}
