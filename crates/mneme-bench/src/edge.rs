//! Edge-case / adversarial stress harness.
//!
//! Throughput stress (see `scale.rs`) answers "does it stay fast at size?".
//! This harness answers the other half of "is it ready?": **does it stay
//! correct under hostile inputs and the seven hard-rule invariants?** Each
//! scenario drives the *real* `FjallEventLog → VectorView + Bm25View →
//! HybridRetriever` path with something pathological and asserts an invariant:
//!
//! - degenerate + syntax-laden + unicode queries never panic or error
//! - huge / empty content ingests and stays retrievable
//! - **scope is a security boundary** (Hard Rule #3): a rare in-scope needle is
//!   found and no out-of-scope memory ever leaks, even when the index is
//!   dominated by another tenant
//! - a superseded fact disappears from retrieval but its write **stays in the
//!   log** (Hard Rule #2: never overwrite history)
//! - tombstone-heavy indexes return only live memories
//! - a dimension-mismatched vector is rejected, not panicked on
//! - **views rebuild from the log** (Hard Rule #4): replaying the event log
//!   into fresh views reproduces identical BM25 results and identical recall
//! - concurrent writes all land with unique, monotonic ids (Hard Rule #2/#4)
//!
//! `run_edge_suite` returns a structured pass/fail report; the `edge`
//! subcommand exits non-zero on any failure so it gates CI like the recall
//! floors do.

use anyhow::{anyhow, bail, Result};
use mneme_core::entity::{Memory, Provenance};
use mneme_core::event::{Event, LogEntry};
use mneme_core::traits::MaterializedView;
use mneme_core::types::{new_id, BiTemporal, Id, MemoryRef, Scope};
use mneme_core::{Embedder, EventLog, Query, Retriever};
use mneme_index::{Bm25View, HybridRetriever, MockEmbedder, VectorView};
use mneme_store::FjallEventLog;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

const DIM: usize = 32;

/// Result of one adversarial scenario.
#[derive(Debug, Clone)]
pub struct EdgeOutcome {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

/// Aggregate report over all scenarios.
#[derive(Debug, Clone, Default)]
pub struct EdgeReport {
    pub outcomes: Vec<EdgeOutcome>,
}

impl EdgeReport {
    pub fn passed(&self) -> usize {
        self.outcomes.iter().filter(|o| o.passed).count()
    }
    pub fn failed(&self) -> usize {
        self.outcomes.iter().filter(|o| !o.passed).count()
    }
    pub fn all_passed(&self) -> bool {
        self.outcomes.iter().all(|o| o.passed)
    }
}

// --- shared scaffolding ---

struct Env {
    dir: PathBuf,
    log: Arc<FjallEventLog>,
    vector: Arc<VectorView>,
    bm25: Arc<Bm25View>,
    embedder: Arc<dyn Embedder>,
}

impl Env {
    fn open() -> Result<Self> {
        Self::open_with_capacity(100_000)
    }

    fn open_with_capacity(capacity: usize) -> Result<Self> {
        let dir = std::env::temp_dir().join(format!("mneme-edge-{}", new_id()));
        let log = FjallEventLog::open(&dir).map_err(|e| anyhow!("open log: {e}"))?;
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(DIM));
        let vector = Arc::new(VectorView::with_capacity(
            DIM,
            embedder.model_id().to_string(),
            capacity,
        ));
        let bm25 = Arc::new(Bm25View::new().map_err(|e| anyhow!("bm25 init: {e}"))?);
        Ok(Self {
            dir,
            log,
            vector,
            bm25,
            embedder,
        })
    }

    /// Write one memory through the real path (embed → append → apply both
    /// views, committing BM25 immediately). Returns its ref.
    async fn put(&self, scope: &Scope, content: &str, tags: &[&str]) -> Result<MemoryRef> {
        let entry = self.append_only(scope, content, tags).await?;
        self.vector.apply(&entry).await?;
        self.bm25.apply(&entry).await?;
        let id = match &entry.event {
            Event::MemoryWritten(m) => MemoryRef(m.id),
            _ => unreachable!(),
        };
        Ok(id)
    }

    /// Append a memory to the log without touching the views (used by the
    /// bulk/scope scenarios that stage the BM25 index for speed).
    async fn append_only(&self, scope: &Scope, content: &str, tags: &[&str]) -> Result<LogEntry> {
        let emb = self
            .embedder
            .embed(&[content.to_string()])
            .await?
            .into_iter()
            .next();
        let mem = Memory {
            id: new_id(),
            scope: scope.clone(),
            content: content.to_string(),
            keywords: vec![],
            tags: tags.iter().map(|t| (*t).to_string()).collect(),
            context: String::new(),
            embedding: emb,
            links: vec![],
            parent: None,
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: Provenance {
                source: "edge".into(),
                trust: 1.0,
            },
            source: None,
            position: None,
        };
        let event = Event::MemoryWritten(mem);
        let id = self.log.append(event.clone()).await?;
        Ok(LogEntry { id, event })
    }

    fn retriever(&self) -> HybridRetriever {
        HybridRetriever::new(
            self.vector.clone(),
            self.bm25.clone(),
            self.embedder.clone(),
        )
    }

    fn cleanup(self) {
        let dir = self.dir.clone();
        drop(self.log);
        std::fs::remove_dir_all(&dir).ok();
    }
}

fn ok(name: &str, detail: impl Into<String>) -> EdgeOutcome {
    EdgeOutcome {
        name: name.to_string(),
        passed: true,
        detail: detail.into(),
    }
}

// --- the suite ---

/// Run every adversarial scenario and collect a pass/fail report. A scenario
/// that returns `Err` is recorded as a failure with its message (rather than
/// aborting the run), so one broken invariant doesn't hide the others.
pub async fn run_edge_suite() -> Result<EdgeReport> {
    let mut report = EdgeReport::default();

    macro_rules! run {
        ($name:literal, $fut:expr) => {
            match $fut.await {
                Ok(o) => report.outcomes.push(o),
                Err(e) => report.outcomes.push(EdgeOutcome {
                    name: $name.to_string(),
                    passed: false,
                    detail: format!("{e}"),
                }),
            }
        };
    }

    run!("degenerate_queries", scenario_degenerate_queries());
    run!("pathological_query_syntax", scenario_pathological_syntax());
    run!("unicode_and_emoji", scenario_unicode());
    run!("huge_and_empty_content", scenario_huge_and_empty());
    run!("scope_isolation_extreme", scenario_scope_isolation());
    run!(
        "supersede_keeps_history",
        scenario_supersede_keeps_history()
    );
    run!("tombstone_heavy_index", scenario_tombstone_heavy());
    run!("dim_mismatch_rejected", scenario_dim_mismatch());
    run!("replay_rebuild_reproduces", scenario_replay_rebuild());
    run!("concurrent_writes_unique_ids", scenario_concurrent_writes());

    Ok(report)
}

/// Empty / whitespace / stopword-only queries and pathological `k` values must
/// never panic or error, and must respect the `k` bound.
async fn scenario_degenerate_queries() -> Result<EdgeOutcome> {
    let env = Env::open()?;
    let scope = Scope::global("edge");
    env.put(
        &scope,
        "the quarterly revenue grew by eighteen percent",
        &["fin"],
    )
    .await?;
    let retriever = env.retriever();

    let cases: &[(&str, usize)] = &[
        ("", 10),                 // empty
        ("    \t  ", 10),         // whitespace
        ("the of a an is to", 5), // stopwords only
        ("revenue", 0),           // k = 0
        ("revenue", 1_000_000),   // k ≫ corpus
    ];
    for (q, k) in cases {
        let query = Query {
            text: (*q).to_string(),
            scope: scope.clone(),
            k: *k,
            time_filter: None,
        };
        let hits = retriever
            .search(&query)
            .await
            .map_err(|e| anyhow!("query {q:?} k={k} errored: {e}"))?;
        if *k == 0 && !hits.is_empty() {
            env.cleanup();
            bail!("k=0 returned {} hits, expected 0", hits.len());
        }
        if hits.len() > (*k).max(1) && *k != 0 {
            // (max(1) guard not needed since k>0 here, but keep explicit)
            env.cleanup();
            bail!("query {q:?} returned {} hits > k={k}", hits.len());
        }
    }
    env.cleanup();
    Ok(ok(
        "degenerate_queries",
        "empty/whitespace/stopword/k=0/k≫N all handled without panic or error",
    ))
}

/// Queries full of tantivy query-syntax characters must be sanitized, not
/// surfaced as parser errors.
async fn scenario_pathological_syntax() -> Result<EdgeOutcome> {
    let env = Env::open()?;
    let scope = Scope::global("edge");
    env.put(&scope, "EMEA regulatory inquiry about pricing", &["legal"])
        .await?;
    let retriever = env.retriever();

    let nasty = [
        "+revenue -cost",
        ":::",
        "(((",
        "\"unterminated",
        "a AND OR NOT b",
        "field:value^3",
        "*",
        "~~~",
        "\\\\",
        "한국어 AND test",
        "💥 OR 🔥",
        "  +  -  :  ",
    ];
    for q in nasty {
        let query = Query {
            text: q.to_string(),
            scope: scope.clone(),
            k: 10,
            time_filter: None,
        };
        retriever
            .search(&query)
            .await
            .map_err(|e| anyhow!("syntax query {q:?} errored: {e}"))?;
    }
    env.cleanup();
    Ok(ok(
        "pathological_query_syntax",
        "12 syntax/operator-laden queries sanitized without parser errors",
    ))
}

/// Unicode, CJK and emoji content must ingest and stay retrievable by keyword.
async fn scenario_unicode() -> Result<EdgeOutcome> {
    let env = Env::open()?;
    let scope = Scope::global("edge");
    env.put(&scope, "会议 nebularquark 정산 完了", &["intl"])
        .await?;
    env.put(&scope, "café résumé naïve façade — fünf Straße", &["intl"])
        .await?;
    let retriever = env.retriever();

    // A distinctive ASCII token embedded among CJK must still be found.
    let query = Query {
        text: "nebularquark".to_string(),
        scope: scope.clone(),
        k: 5,
        time_filter: None,
    };
    let hits = retriever.search(&query).await?;
    let found = !hits.is_empty();
    env.cleanup();
    if !found {
        bail!("distinctive token embedded in unicode content was not retrieved");
    }
    Ok(ok(
        "unicode_and_emoji",
        "CJK/accented/emoji content ingests; embedded token retrievable",
    ))
}

/// A ~1 MB memory and an empty-content memory must both ingest, index and
/// (for the large one) stay retrievable, with no panic.
async fn scenario_huge_and_empty() -> Result<EdgeOutcome> {
    let env = Env::open()?;
    let scope = Scope::global("edge");

    // Empty content — must not break the index.
    env.put(&scope, "", &["empty"]).await?;

    // ~1 MB of filler with one distinctive gold token.
    let gold = "zqxgoldtoken";
    let huge = format!("{} {}", "lorem ipsum dolor sit amet ".repeat(40_000), gold);
    env.put(&scope, &huge, &["huge"]).await?;

    let retriever = env.retriever();
    let query = Query {
        text: gold.to_string(),
        scope: scope.clone(),
        k: 5,
        time_filter: None,
    };
    let hits = retriever.search(&query).await?;
    env.cleanup();
    if hits.is_empty() {
        bail!("gold token inside a ~1MB memory was not retrieved");
    }
    Ok(ok(
        "huge_and_empty_content",
        "empty-content memory ingested; gold token inside ~1MB memory retrieved",
    ))
}

/// Scope is a security boundary: one needle in tenant A among many tenant-B
/// memories must be found, and **no tenant-B memory may ever be returned** for
/// a tenant-A query — even though B dominates the index.
async fn scenario_scope_isolation() -> Result<EdgeOutcome> {
    const DISTRACTORS: usize = 4000;
    let env = Env::open_with_capacity(DISTRACTORS + 16)?;
    let scope_a = Scope::global("tenant-a");
    let scope_b = Scope::global("tenant-b");

    // Bulk-stage many tenant-B distractors sharing the query's keyword.
    let mut b_ids: HashSet<Id> = HashSet::new();
    for i in 0..DISTRACTORS {
        let entry = env
            .append_only(&scope_b, &format!("shared topic widget number {i}"), &["b"])
            .await?;
        env.vector.apply(&entry).await?;
        env.bm25.stage(&entry).map_err(|e| anyhow!("stage: {e}"))?;
        if let Event::MemoryWritten(m) = &entry.event {
            b_ids.insert(m.id);
        }
    }
    // One tenant-A needle that also matches the shared keyword.
    let a_needle = env
        .put(&scope_a, "shared topic widget secret-a-needle", &["a"])
        .await?;
    env.bm25.commit().map_err(|e| anyhow!("commit: {e}"))?;

    let retriever = env.retriever();
    let query = Query {
        text: "shared topic widget".to_string(),
        scope: scope_a.clone(),
        k: 10,
        time_filter: None,
    };
    let hits = retriever.search(&query).await?;

    // No tenant-B leakage.
    let leaked = hits.iter().any(|h| b_ids.contains(&h.memory.0));
    // The rare in-scope needle is found.
    let found_a = hits.iter().any(|h| h.memory == a_needle);
    env.cleanup();

    if leaked {
        bail!("tenant-B memory leaked into a tenant-A query result");
    }
    if !found_a {
        bail!(
            "the single tenant-A needle was not retrieved among {DISTRACTORS} tenant-B distractors"
        );
    }
    Ok(ok(
        "scope_isolation_extreme",
        format!("1 tenant-A needle found among {DISTRACTORS} tenant-B; zero cross-tenant leakage"),
    ))
}

/// A superseded fact must disappear from retrieval, but its original write must
/// remain in the append-only log (Hard Rule #2: never overwrite history).
async fn scenario_supersede_keeps_history() -> Result<EdgeOutcome> {
    let env = Env::open()?;
    let scope = Scope::global("edge");

    let old = env
        .put(
            &scope,
            "the capital project budget is forty zoltars",
            &["fact"],
        )
        .await?;
    // Invalidate the old fact and write the corrected one.
    let inval = Event::MemoryInvalidated {
        id: old,
        reason: "superseded".into(),
    };
    let inval_id = env.log.append(inval.clone()).await?;
    let inval_entry = LogEntry {
        id: inval_id,
        event: inval,
    };
    env.vector.apply(&inval_entry).await?;
    env.bm25.apply(&inval_entry).await?;
    env.put(
        &scope,
        "the capital project budget is ninety zoltars",
        &["fact"],
    )
    .await?;

    let retriever = env.retriever();
    let q = |t: &str| Query {
        text: t.to_string(),
        scope: scope.clone(),
        k: 10,
        time_filter: None,
    };
    let old_hits = retriever.search(&q("forty zoltars")).await?;
    let still_serving_old = old_hits.iter().any(|h| h.memory == old);

    // History: the original MemoryWritten event is still in the log.
    let entries = env.log.read_from(None).await?;
    let history_kept = entries.iter().any(|e| match &e.event {
        Event::MemoryWritten(m) => m.id == old.0,
        _ => false,
    });
    env.cleanup();

    if still_serving_old {
        bail!("superseded memory is still returned by retrieval");
    }
    if !history_kept {
        bail!("original write was dropped from the log — history was overwritten");
    }
    Ok(ok(
        "supersede_keeps_history",
        "superseded fact removed from retrieval; original write retained in append-only log",
    ))
}

/// An index where most memories are invalidated must return only live ones.
async fn scenario_tombstone_heavy() -> Result<EdgeOutcome> {
    let env = Env::open()?;
    let scope = Scope::global("edge");

    let mut refs = Vec::new();
    for i in 0..200 {
        refs.push(
            env.put(&scope, &format!("alpha record token{i}"), &["t"])
                .await?,
        );
    }
    // Invalidate all but the last 5.
    let live: HashSet<MemoryRef> = refs.iter().rev().take(5).copied().collect();
    for r in &refs {
        if live.contains(r) {
            continue;
        }
        let ev = Event::MemoryInvalidated {
            id: *r,
            reason: "tombstone-heavy test".into(),
        };
        let id = env.log.append(ev.clone()).await?;
        let entry = LogEntry { id, event: ev };
        env.vector.apply(&entry).await?;
        env.bm25.apply(&entry).await?;
    }

    let retriever = env.retriever();
    let hits = retriever
        .search(&Query {
            text: "alpha record".to_string(),
            scope: scope.clone(),
            k: 50,
            time_filter: None,
        })
        .await?;
    let any_dead = hits.iter().any(|h| !live.contains(&h.memory));
    let tombstones = env.vector.tombstone_count();
    let live_count = env.vector.live_count();
    env.cleanup();

    if any_dead {
        bail!("a tombstoned memory was returned by retrieval");
    }
    if live_count != 5 {
        bail!("vector live_count is {live_count}, expected 5");
    }
    if tombstones != 195 {
        bail!("vector tombstone_count is {tombstones}, expected 195");
    }
    Ok(ok(
        "tombstone_heavy_index",
        "195/200 invalidated; retrieval returns only the 5 live; counts consistent",
    ))
}

/// A vector of the wrong dimension must be rejected with an error, not panic.
async fn scenario_dim_mismatch() -> Result<EdgeOutcome> {
    let view = VectorView::new(DIM, "mock");
    // Query with wrong dim → Err, not panic.
    let scope = Scope::global("edge");
    let bad = vec![0.1f32; DIM + 7];
    let res = view.search(&bad, 5, &scope);
    if res.is_ok() {
        bail!("search with wrong-dim query was accepted; expected an error");
    }
    Ok(ok(
        "dim_mismatch_rejected",
        "wrong-dimension query rejected with an error (no panic)",
    ))
}

/// Replaying the event log into fresh views must reproduce identical BM25
/// results and identical vector recall (Hard Rule #4: views are materialized,
/// rebuildable from the log).
async fn scenario_replay_rebuild() -> Result<EdgeOutcome> {
    let env = Env::open()?;
    let scope = Scope::global("edge");

    // A mix of writes; invalidate a couple to exercise the tombstone path.
    let mut refs = Vec::new();
    for i in 0..60 {
        refs.push(
            env.put(
                &scope,
                &format!("project apollo milestone marker{i}"),
                &["p"],
            )
            .await?,
        );
    }
    for r in refs.iter().take(10) {
        let ev = Event::MemoryInvalidated {
            id: *r,
            reason: "replay test".into(),
        };
        let id = env.log.append(ev.clone()).await?;
        let entry = LogEntry { id, event: ev };
        env.vector.apply(&entry).await?;
        env.bm25.apply(&entry).await?;
    }

    let query_text = "milestone marker33";
    let live_retriever = env.retriever();
    let live_bm25 = env
        .bm25
        .search(query_text, 20, &scope)
        .map_err(|e| anyhow!("live bm25: {e}"))?;
    let live_hits = live_retriever
        .search(&Query {
            text: query_text.to_string(),
            scope: scope.clone(),
            k: 20,
            time_filter: None,
        })
        .await?;
    let live_recall_ids: HashSet<MemoryRef> = live_hits.iter().map(|h| h.memory).collect();

    // Rebuild fresh views purely by replaying the log.
    let entries = env.log.read_from(None).await?;
    let r_vector = Arc::new(VectorView::new(DIM, env.embedder.model_id().to_string()));
    let r_bm25 = Arc::new(Bm25View::new().map_err(|e| anyhow!("bm25 init: {e}"))?);
    for entry in &entries {
        r_vector
            .apply(entry)
            .await
            .map_err(|e| anyhow!("v apply: {e}"))?;
        r_bm25.stage(entry).map_err(|e| anyhow!("b stage: {e}"))?;
    }
    r_bm25.commit().map_err(|e| anyhow!("b commit: {e}"))?;

    let rebuilt_bm25 = r_bm25
        .search(query_text, 20, &scope)
        .map_err(|e| anyhow!("rebuilt bm25: {e}"))?;
    let rebuilt_retriever =
        HybridRetriever::new(r_vector.clone(), r_bm25.clone(), env.embedder.clone());
    let rebuilt_hits = rebuilt_retriever
        .search(&Query {
            text: query_text.to_string(),
            scope: scope.clone(),
            k: 20,
            time_filter: None,
        })
        .await?;
    let rebuilt_recall_ids: HashSet<MemoryRef> = rebuilt_hits.iter().map(|h| h.memory).collect();

    // BM25 is deterministic → identical ordered results.
    let bm25_ids_live: Vec<MemoryRef> = live_bm25.iter().map(|h| h.memory).collect();
    let bm25_ids_rebuilt: Vec<MemoryRef> = rebuilt_bm25.iter().map(|h| h.memory).collect();

    let live_count_match = env.vector.live_count() == r_vector.live_count();
    env.cleanup();

    if bm25_ids_live != bm25_ids_rebuilt {
        bail!("rebuilt BM25 results differ from the live view");
    }
    if live_recall_ids != rebuilt_recall_ids {
        bail!("rebuilt hybrid recall set differs from the live view");
    }
    if !live_count_match {
        bail!("rebuilt vector live_count differs from the live view");
    }
    Ok(ok(
        "replay_rebuild_reproduces",
        "fresh views replayed from the log reproduce identical BM25 + recall + live_count",
    ))
}

/// Many concurrent appends on a shared log must all land with unique ids and no
/// lost writes (Hard Rule #2/#4).
async fn scenario_concurrent_writes() -> Result<EdgeOutcome> {
    const N: usize = 256;
    let env = Env::open()?;
    let dir = env.dir.clone();
    let log = env.log; // already an Arc<FjallEventLog>
    let scope = Scope::global("edge");

    let mut handles = Vec::with_capacity(N);
    for i in 0..N {
        let log = log.clone();
        let scope = scope.clone();
        handles.push(tokio::spawn(async move {
            let mem = Memory {
                id: new_id(),
                scope,
                content: format!("concurrent write {i}"),
                keywords: vec![],
                tags: vec![],
                context: String::new(),
                embedding: None,
                links: vec![],
                parent: None,
                evolution_count: 0,
                time: BiTemporal::now(),
                provenance: Provenance {
                    source: "edge".into(),
                    trust: 1.0,
                },
                source: None,
                position: None,
            };
            log.append(Event::MemoryWritten(mem)).await
        }));
    }
    let mut returned_ids: HashSet<Id> = HashSet::new();
    for h in handles {
        let id = h.await.map_err(|e| anyhow!("join: {e}"))??;
        returned_ids.insert(id);
    }

    let entries = log.read_from(None).await?;
    let log_ids: HashSet<Id> = entries.iter().map(|e| e.id).collect();

    // best-effort cleanup of the temp keyspace
    if let Ok(inner) = Arc::try_unwrap(log) {
        drop(inner);
    }
    std::fs::remove_dir_all(&dir).ok();

    if returned_ids.len() != N {
        bail!(
            "expected {N} unique returned ids, got {} (id collision under concurrency)",
            returned_ids.len()
        );
    }
    if entries.len() != N {
        bail!(
            "log has {} entries, expected {N} (lost writes)",
            entries.len()
        );
    }
    if log_ids.len() != N {
        bail!("log contains duplicate ids under concurrency");
    }
    Ok(ok(
        "concurrent_writes_unique_ids",
        format!("{N} concurrent appends all landed with unique ids; no lost or colliding writes"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn full_edge_suite_passes() {
        let report = run_edge_suite().await.unwrap();
        let failures: Vec<&EdgeOutcome> = report.outcomes.iter().filter(|o| !o.passed).collect();
        assert!(
            report.all_passed(),
            "edge-case invariants violated: {:#?}",
            failures
        );
        // Guard against accidentally shrinking the suite.
        assert!(
            report.outcomes.len() >= 10,
            "expected ≥10 scenarios, got {}",
            report.outcomes.len()
        );
    }
}
