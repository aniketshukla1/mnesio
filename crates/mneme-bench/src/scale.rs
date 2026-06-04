//! Scale + load harness: ingest a large synthetic corpus through the *real*
//! storage → views → retriever path and measure how mneme behaves at size.
//!
//! Reports, per N:
//! - **write throughput** (memories/sec) + ingest wall time
//! - **per-write latency** p50 / p95 (append + vector apply + bm25 apply)
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
use mneme_core::entity::{Memory, Provenance};
use mneme_core::event::{Event, LogEntry};
use mneme_core::traits::MaterializedView;
use mneme_core::types::{new_id, BiTemporal, MemoryRef, Scope};
use mneme_core::{Embedder, EventLog, Query, Retriever};
use mneme_index::{Bm25View, FastEmbedEmbedder, HybridRetriever, MockEmbedder, VectorView};
use mneme_store::FjallEventLog;
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

    // ingest
    pub ingest_secs: f64,
    pub write_throughput_per_sec: f64,
    pub write_p50_ms: f64,
    pub write_p95_ms: f64,

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

    let dir = std::env::temp_dir().join(format!("mneme-scale-{}", new_id()));
    let log = FjallEventLog::open(&dir).map_err(|e| anyhow!("open log: {e}"))?;
    let vector = Arc::new(VectorView::new(
        embedder.dim(),
        embedder.model_id().to_string(),
    ));
    let bm25 = Arc::new(Bm25View::new().map_err(|e| anyhow!("bm25 init: {e}"))?);

    // Map gold-bearing memories so recall can be scored by content.
    let mut content_by_id: HashMap<MemoryRef, String> = HashMap::new();
    let mut write_latencies_ms: Vec<f64> = Vec::with_capacity(corpus.memories.len());

    let ingest_start = Instant::now();
    // Ingest in batches: embed the batch (one model call), then append + apply
    // per memory, timing the per-write path (append + both view applies).
    let mems = &corpus.memories;
    let mut idx = 0;
    while idx < mems.len() {
        let end = (idx + EMBED_BATCH).min(mems.len());
        let batch_texts: Vec<String> = mems[idx..end].iter().map(|m| m.content.clone()).collect();
        let vectors = embedder
            .embed(&batch_texts)
            .await
            .map_err(|e| anyhow!("embed: {e}"))?;
        for (offset, gm) in mems[idx..end].iter().enumerate() {
            let embedding = vectors.get(offset).cloned();
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

            // Time the per-write critical path.
            let w = Instant::now();
            let id = log
                .append(event.clone())
                .await
                .map_err(|e| anyhow!("append: {e}"))?;
            let entry = LogEntry { id, event };
            vector
                .apply(&entry)
                .await
                .map_err(|e| anyhow!("vector apply: {e}"))?;
            bm25.apply(&entry)
                .await
                .map_err(|e| anyhow!("bm25 apply: {e}"))?;
            write_latencies_ms.push(w.elapsed().as_secs_f64() * 1000.0);
        }
        idx = end;
    }
    let ingest_secs = ingest_start.elapsed().as_secs_f64();

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

    write_latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    query_latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let ingested = corpus.memories.len();
    let report = ScaleReport {
        n,
        embedder: embedder_choice.to_string(),
        k,
        ingested,
        needles: corpus.needles.len(),
        ingest_secs,
        write_throughput_per_sec: if ingest_secs > 0.0 {
            ingested as f64 / ingest_secs
        } else {
            0.0
        },
        write_p50_ms: percentile(&write_latencies_ms, 50.0),
        write_p95_ms: percentile(&write_latencies_ms, 95.0),
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
    "n,embedder,k,ingested,needles,ingest_secs,write_per_sec,write_p50_ms,write_p95_ms,\
     query_p50_ms,query_p95_ms,query_p99_ms,recall,slot_count,live_count,tombstones"
        .to_string()
}

/// One CSV row.
pub fn scale_csv_row(r: &ScaleReport) -> String {
    format!(
        "{},{},{},{},{},{:.3},{:.1},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{}",
        r.n,
        r.embedder,
        r.k,
        r.ingested,
        r.needles,
        r.ingest_secs,
        r.write_throughput_per_sec,
        r.write_p50_ms,
        r.write_p95_ms,
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
        assert!(r.write_throughput_per_sec > 0.0);
        // Gold tokens are exact + unique, so BM25 must recall ~all of them.
        assert!(
            r.recall() > 0.9,
            "exact unique gold tokens should be highly recalled; got {}",
            r.recall()
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
            ingest_secs: 0.1,
            write_throughput_per_sec: 10.0,
            write_p50_ms: 0.1,
            write_p95_ms: 0.2,
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
