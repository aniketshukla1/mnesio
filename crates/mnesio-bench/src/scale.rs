//! Scale + load harness: ingest a large synthetic corpus through the *real*
//! storage → views → retriever path and measure how mnesio behaves at size.
//!
//! Reports, per N — and crucially it **separates the two phases** so the
//! numbers reflect mnesio's real architecture (Hard Rule #5: the write path is
//! the log append; embedding + indexing are async behind a bounded queue):
//! - **append path** — log-append throughput (mem/s) + p50/p95 latency. This
//!   is the user-facing write path the <5ms target governs.
//! - **index build** — vector(HNSW) + BM25 apply throughput + p50/p95. In the
//!   server this runs in the async embedding/index workers, *off* the write
//!   path; here we measure it separately so its cost is visible, not conflated
//!   with the append.
//! - **query latency** p50 / p95 / p99 over the labeled needle set
//! - **recall@k** against the generator's unambiguous gold tokens
//! - **index observability** (`VectorView` slot/tombstone/live counts)
//!
//! `mock` (32-dim, non-semantic) is for pushing N high cheaply; `fastembed`
//! (384-dim, real) is for an honest recall number on a smaller slice. Nothing
//! here is synthetic-in-the-metrics: every timing is a real operation against
//! the same code the server runs.

use crate::gen::GeneratedCorpus;
use anyhow::{anyhow, Result};
use mnesio_core::entity::{Memory, Provenance};
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::traits::MaterializedView;
use mnesio_core::types::{new_id, BiTemporal, MemoryRef, Scope};
use mnesio_core::{Embedder, EventLog, Query, Retriever};
use mnesio_index::{Bm25View, FastEmbedEmbedder, HybridRetriever, MockEmbedder, VectorView};
use mnesio_store::FjallEventLog;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Embedding batch size for ingest (amortizes model-call overhead under real
/// embedders; harmless for mock).
const EMBED_BATCH: usize = 256;

/// One row of scale results.
#[derive(Debug, Clone)]
pub struct ScaleReport {
    pub n: usize,
    pub embedder: String,
    pub k: usize,
    /// Total live memories actually ingested (base + evolution + contradiction).
    pub ingested: usize,
    pub needles: usize,

    // append path (the real <5ms write path: log append only)
    pub append_secs: f64,
    pub append_throughput_per_sec: f64,
    pub append_p50_ms: f64,
    pub append_p95_ms: f64,

    // index build (async in the server: vector HNSW + BM25 apply)
    pub index_secs: f64,
    pub index_throughput_per_sec: f64,
    pub index_p50_ms: f64,
    pub index_p95_ms: f64,
    /// One-time BM25 commit (segment flush) at the end of the bulk build,
    /// amortized out of the per-entry `index_p*` figures.
    pub index_commit_ms: f64,

    // query
    pub query_p50_ms: f64,
    pub query_p95_ms: f64,
    pub query_p99_ms: f64,
    pub recalled: usize,

    // index observability
    pub slot_count: usize,
    pub live_count: usize,
    pub tombstone_count: usize,
}

impl ScaleReport {
    pub fn recall(&self) -> f32 {
        if self.needles == 0 {
            0.0
        } else {
            self.recalled as f32 / self.needles as f32
        }
    }
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    // Nearest-rank on an already-sorted slice.
    let rank = (p / 100.0 * sorted_ms.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted_ms.len() - 1);
    sorted_ms[idx]
}

fn build_embedder(choice: &str) -> Result<Arc<dyn Embedder>> {
    match choice {
        "mock" => Ok(Arc::new(MockEmbedder::new(32))),
        "fastembed" => Ok(Arc::new(
            FastEmbedEmbedder::new().map_err(|e| anyhow!("fastembed init failed: {e}"))?,
        )),
        other => Err(anyhow!(
            "unknown embedder {other:?}; expected `mock` or `fastembed`"
        )),
    }
}

/// Run a single scale point: generate `n` memories (seed), ingest, query the
/// needles, and collect timings + recall + observability.
pub async fn run_scale_point(
    n: usize,
    seed: u64,
    k: usize,
    embedder_choice: &str,
) -> Result<ScaleReport> {
    let scope = Scope::global("scale");
    let embedder = build_embedder(embedder_choice)?;
    let corpus = GeneratedCorpus::generate(n, seed);

    let dir = std::env::temp_dir().join(format!("mnesio-scale-{}", new_id()));
    let log = FjallEventLog::open(&dir).map_err(|e| anyhow!("open log: {e}"))?;
    // Pre-size the HNSW for the known corpus so a 100k+ stress point doesn't
    // pay reallocation churn (the default hint is 100k).
    let vector = Arc::new(VectorView::with_capacity(
        embedder.dim(),
        embedder.model_id().to_string(),
        corpus.memories.len() + 16,
    ));
    let bm25 = Arc::new(Bm25View::new().map_err(|e| anyhow!("bm25 init: {e}"))?);

    // Map gold-bearing memories so recall can be scored by content.
    let mut content_by_id: HashMap<MemoryRef, String> = HashMap::new();
    let mut append_latencies_ms: Vec<f64> = Vec::with_capacity(corpus.memories.len());
    let mut index_latencies_ms: Vec<f64> = Vec::with_capacity(corpus.memories.len());

    let mems = &corpus.memories;

    // Phase 0 — EMBED: compute every embedding up front, in batches. In the
    // server this runs in the async embedding worker, *off* the write path
    // (Hard Rule #5); we deliberately keep it out of the append window below so
    // the append throughput reflects the real <5ms write path even under a slow
    // real (fastembed) embedder, not the embedding cost.
    let mut embeddings: Vec<Option<Vec<f32>>> = Vec::with_capacity(mems.len());
    let mut idx = 0;
    while idx < mems.len() {
        let end = (idx + EMBED_BATCH).min(mems.len());
        let batch_texts: Vec<String> = mems[idx..end].iter().map(|m| m.content.clone()).collect();
        let vectors = embedder
            .embed(&batch_texts)
            .await
            .map_err(|e| anyhow!("embed: {e}"))?;
        for offset in 0..(end - idx) {
            embeddings.push(vectors.get(offset).cloned());
        }
        idx = end;
    }

    // Phase 1 — APPEND: write every memory to the log. This is mnesio's real
    // user-facing write path (Hard Rule #5: <5ms target). Embeddings are
    // precomputed above, so this window is pure log-append. We retain the
    // resulting LogEntries to drive the index phase next.
    let mut entries: Vec<LogEntry> = Vec::with_capacity(mems.len());
    let append_start = Instant::now();
    for (gm, embedding) in mems.iter().zip(embeddings) {
        let mem = Memory {
            id: new_id(),
            scope: scope.clone(),
            content: gm.content.clone(),
            keywords: vec![],
            tags: gm.tags.clone(),
            context: String::new(),
            embedding,
            links: vec![],
            parent: None,
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: Provenance {
                source: "scale".into(),
                trust: 1.0,
            },
            source: None,
            position: None,
        };
        let mref = MemoryRef(mem.id);
        if gm.needle_for.is_some() {
            content_by_id.insert(mref, mem.content.clone());
        }
        let event = Event::MemoryWritten(mem);

        // Time the append-only critical path (the <5ms write path).
        let w = Instant::now();
        let id = log
            .append(event.clone())
            .await
            .map_err(|e| anyhow!("append: {e}"))?;
        append_latencies_ms.push(w.elapsed().as_secs_f64() * 1000.0);

        entries.push(LogEntry { id, event });
    }
    let append_secs = append_start.elapsed().as_secs_f64();

    // Phase 2 — INDEX BUILD (bulk replay-rebuild). Apply each entry to the
    // vector (HNSW insert) + BM25 (stage, no per-doc commit), then commit BM25
    // **once**. In the server this is the async embedding/index workers, *off*
    // the write path; timed separately so the build cost is visible, not blamed
    // on the append. The per-entry latency below is the add cost with the
    // segment-flush commit amortized out (committing per doc turns an O(N)
    // ingest into O(N) tantivy segment flushes — see `Bm25View::stage`); the
    // one-time commit is reported separately as `index_commit_ms`.
    let index_start = Instant::now();
    for entry in &entries {
        let w = Instant::now();
        vector
            .apply(entry)
            .await
            .map_err(|e| anyhow!("vector apply: {e}"))?;
        bm25.stage(entry).map_err(|e| anyhow!("bm25 stage: {e}"))?;
        index_latencies_ms.push(w.elapsed().as_secs_f64() * 1000.0);
    }
    let commit_start = Instant::now();
    bm25.commit().map_err(|e| anyhow!("bm25 commit: {e}"))?;
    let index_commit_ms = commit_start.elapsed().as_secs_f64() * 1000.0;
    let index_secs = index_start.elapsed().as_secs_f64();

    let retriever = HybridRetriever::new(vector.clone(), bm25, embedder.clone());

    // Query the needle set.
    let mut query_latencies_ms: Vec<f64> = Vec::with_capacity(corpus.needles.len());
    let mut recalled = 0usize;
    for needle in &corpus.needles {
        let query = Query {
            text: needle.query.clone(),
            scope: scope.clone(),
            k,
            time_filter: None,
        };
        let start = Instant::now();
        let hits = retriever
            .search(&query)
            .await
            .map_err(|e| anyhow!("search: {e}"))?;
        query_latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);

        let gold = needle.answer_substring.to_ascii_lowercase();
        let hit = hits.iter().any(|h| {
            content_by_id
                .get(&h.memory)
                .map(|c| c.to_ascii_lowercase().contains(&gold))
                .unwrap_or(false)
        });
        if hit {
            recalled += 1;
        }
    }

    append_latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    index_latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    query_latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let ingested = corpus.memories.len();
    let per_sec = |count: usize, secs: f64| if secs > 0.0 { count as f64 / secs } else { 0.0 };
    let report = ScaleReport {
        n,
        embedder: embedder_choice.to_string(),
        k,
        ingested,
        needles: corpus.needles.len(),
        append_secs,
        append_throughput_per_sec: per_sec(ingested, append_secs),
        append_p50_ms: percentile(&append_latencies_ms, 50.0),
        append_p95_ms: percentile(&append_latencies_ms, 95.0),
        index_secs,
        index_throughput_per_sec: per_sec(ingested, index_secs),
        index_p50_ms: percentile(&index_latencies_ms, 50.0),
        index_p95_ms: percentile(&index_latencies_ms, 95.0),
        index_commit_ms,
        query_p50_ms: percentile(&query_latencies_ms, 50.0),
        query_p95_ms: percentile(&query_latencies_ms, 95.0),
        query_p99_ms: percentile(&query_latencies_ms, 99.0),
        recalled,
        slot_count: vector.slot_count(),
        live_count: vector.live_count(),
        tombstone_count: vector.tombstone_count(),
    };

    drop(log);
    std::fs::remove_dir_all(&dir).ok();
    Ok(report)
}

/// CSV header for a scale sweep.
pub fn scale_csv_header() -> String {
    "n,embedder,k,ingested,needles,\
     append_secs,append_per_sec,append_p50_ms,append_p95_ms,\
     index_secs,index_per_sec,index_p50_ms,index_p95_ms,index_commit_ms,\
     query_p50_ms,query_p95_ms,query_p99_ms,recall,slot_count,live_count,tombstones"
        .to_string()
}

/// One CSV row.
pub fn scale_csv_row(r: &ScaleReport) -> String {
    format!(
        "{},{},{},{},{},\
         {:.3},{:.1},{:.4},{:.4},\
         {:.3},{:.1},{:.4},{:.4},{:.3},\
         {:.4},{:.4},{:.4},{:.4},{},{},{}",
        r.n,
        r.embedder,
        r.k,
        r.ingested,
        r.needles,
        r.append_secs,
        r.append_throughput_per_sec,
        r.append_p50_ms,
        r.append_p95_ms,
        r.index_secs,
        r.index_throughput_per_sec,
        r.index_p50_ms,
        r.index_p95_ms,
        r.index_commit_ms,
        r.query_p50_ms,
        r.query_p95_ms,
        r.query_p99_ms,
        r.recall(),
        r.slot_count,
        r.live_count,
        r.tombstone_count,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_nearest_rank() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&xs, 50.0), 5.0);
        assert_eq!(percentile(&xs, 95.0), 10.0);
        assert_eq!(percentile(&xs, 99.0), 10.0);
        assert_eq!(percentile(&xs, 100.0), 10.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
        assert_eq!(percentile(&[42.0], 50.0), 42.0);
    }

    #[tokio::test]
    async fn small_scale_point_runs_end_to_end_with_mock() {
        // A real ingest+query at modest N with the mock embedder: BM25 carries
        // recall, and the unambiguous gold tokens make it deterministic.
        let r = run_scale_point(400, 1, 10, "mock").await.unwrap();
        assert_eq!(r.ingested, r.slot_count, "every memory has a vector slot");
        assert_eq!(
            r.live_count, r.slot_count,
            "no invalidations in a scale ingest"
        );
        assert_eq!(r.tombstone_count, 0);
        assert!(r.needles > 0, "should have needles at N=400");
        assert!(r.append_throughput_per_sec > 0.0);
        assert!(r.index_throughput_per_sec > 0.0);
        // Gold tokens are exact + unique, so BM25 must recall ~all of them.
        assert!(
            r.recall() > 0.9,
            "exact unique gold tokens should be highly recalled; got {}",
            r.recall()
        );
        // Append should be much cheaper than the HNSW index apply — that's the
        // whole point of separating them (the <5ms write path vs async build).
        assert!(
            r.append_p50_ms <= r.index_p50_ms + 1e-9,
            "append p50 {} should be ≤ index p50 {}",
            r.append_p50_ms,
            r.index_p50_ms
        );
        // Latencies are real, non-negative, and ordered p50 ≤ p95 ≤ p99.
        assert!(r.query_p50_ms <= r.query_p95_ms + 1e-9);
        assert!(r.query_p95_ms <= r.query_p99_ms + 1e-9);
    }

    #[test]
    fn csv_row_matches_header_arity() {
        let header_cols = scale_csv_header().split(',').count();
        // A representative report.
        let r = ScaleReport {
            n: 1,
            embedder: "mock".into(),
            k: 10,
            ingested: 1,
            needles: 0,
            append_secs: 0.05,
            append_throughput_per_sec: 20.0,
            append_p50_ms: 0.05,
            append_p95_ms: 0.1,
            index_secs: 0.1,
            index_throughput_per_sec: 10.0,
            index_p50_ms: 0.1,
            index_p95_ms: 0.2,
            index_commit_ms: 0.5,
            query_p50_ms: 0.1,
            query_p95_ms: 0.2,
            query_p99_ms: 0.3,
            recalled: 0,
            slot_count: 1,
            live_count: 1,
            tombstone_count: 0,
        };
        let row_cols = scale_csv_row(&r).split(',').count();
        assert_eq!(header_cols, row_cols, "CSV header and row arity must match");
    }
}
