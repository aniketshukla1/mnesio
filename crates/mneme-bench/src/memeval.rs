//! Memory-recall benchmarking — LOCOMO / LongMemEval style.
//!
//! Where `lib.rs` benchmarks the *procedural compiler* (does the agent
//! get better at a task?), this module benchmarks the *memory layer
//! itself*: ingest a haystack of memories, then ask questions and
//! measure whether the answer-bearing memory is **retrieved**.
//!
//! The headline metric is **recall@k**: for each question, does any of
//! the top-`k` retrieved memories contain the gold answer span? This is
//! the standard retrieval-quality proxy LOCOMO / LongMemEval report
//! alongside LLM-judged QA accuracy — and it needs no LLM, so it runs
//! fully offline and gates CI.
//!
//! The pipeline is the *real* one: `FjallEventLog` → `VectorView` +
//! `Bm25View` → `HybridRetriever` with RRF. Embedder is pluggable:
//! - `mock` (default) — offline, no downloads. Non-semantic, so recall
//!   leans on BM25 and the HNSW layer's internal randomness can flip a
//!   question that ranks right at the `k` cutoff. Use it for *smoke +
//!   CI availability*; set CI floors (`--min-recall`) with margin.
//! - `fastembed` — real semantic embeddings (downloads a model on first
//!   run). This is the configuration to quote a published number from.

use anyhow::{anyhow, bail, Context, Result};
use mneme_core::entity::{Memory, Provenance};
use mneme_core::event::{Event, LogEntry};
use mneme_core::traits::MaterializedView;
use mneme_core::types::{new_id, BiTemporal, MemoryRef, Scope};
use mneme_core::{Embedder, EventLog, Query, Retriever};
use mneme_index::{Bm25View, FastEmbedEmbedder, HybridRetriever, MockEmbedder, VectorView};
use mneme_store::FjallEventLog;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// 32-dim mock embedder — matches the server default. Non-semantic, so
/// recall under `mock` leans on the BM25 signal.
const MOCK_DIM: usize = 32;

/// A memory-recall suite, mirrored from the JSON files in `data/`.
#[derive(Debug, Deserialize, Serialize)]
pub struct MemEvalSuite {
    pub name: String,
    pub description: String,
    /// The haystack: memories to ingest before questioning.
    pub memories: Vec<MemItem>,
    pub questions: Vec<MemQuestion>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MemItem {
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MemQuestion {
    pub question: String,
    /// Case-insensitive substring that must appear in a retrieved
    /// memory for the question to count as recalled.
    pub answer_substring: String,
    /// `single-hop` | `multi-hop` | `temporal` | `open-domain` | …
    pub category: String,
}

/// Per-category recall tally.
#[derive(Debug, Clone)]
pub struct CategoryRecall {
    pub category: String,
    pub recalled: usize,
    pub total: usize,
}

impl CategoryRecall {
    pub fn rate(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.recalled as f32 / self.total as f32
        }
    }
}

/// Result of a full memory-eval run.
pub struct MemEvalReport {
    pub suite_name: String,
    pub embedder: String,
    pub k: usize,
    pub memory_count: usize,
    pub total_questions: usize,
    pub recalled: usize,
    pub per_category: Vec<CategoryRecall>,
    pub mean_latency_ms: f64,
}

impl MemEvalReport {
    /// Overall recall@k in `[0.0, 1.0]`.
    pub fn recall(&self) -> f32 {
        if self.total_questions == 0 {
            0.0
        } else {
            self.recalled as f32 / self.total_questions as f32
        }
    }
}

/// Parse a suite from JSON.
pub fn load_memeval_suite(json: &str) -> Result<MemEvalSuite> {
    let suite: MemEvalSuite = serde_json::from_str(json).context("parsing mem-eval suite JSON")?;
    if suite.memories.is_empty() {
        bail!("mem-eval suite {:?} has no memories", suite.name);
    }
    if suite.questions.is_empty() {
        bail!("mem-eval suite {:?} has no questions", suite.name);
    }
    Ok(suite)
}

fn build_embedder(choice: &str) -> Result<Arc<dyn Embedder>> {
    match choice {
        "mock" => Ok(Arc::new(MockEmbedder::new(MOCK_DIM))),
        "fastembed" => Ok(Arc::new(
            FastEmbedEmbedder::new().map_err(|e| anyhow!("fastembed init failed: {e}"))?,
        )),
        other => bail!("unknown embedder {other:?}; expected `mock` or `fastembed`"),
    }
}

/// Run a memory-recall benchmark end to end against the real pipeline.
pub async fn run_memeval(
    suite: &MemEvalSuite,
    k: usize,
    embedder_choice: &str,
) -> Result<MemEvalReport> {
    let scope = Scope::global("bench");
    let embedder = build_embedder(embedder_choice)?;

    // Real storage + views, in a throwaway temp keyspace.
    let dir = std::env::temp_dir().join(format!("mneme-memeval-{}", new_id()));
    let log = FjallEventLog::open(&dir).map_err(|e| anyhow!("open log: {e}"))?;
    let vector = Arc::new(VectorView::new(
        embedder.dim(),
        embedder.model_id().to_string(),
    ));
    let bm25 = Arc::new(Bm25View::new().map_err(|e| anyhow!("bm25 init: {e}"))?);

    // --- ingest the haystack ---
    // Embed inline so the vector view inserts on the synchronous path;
    // keep an id→content map for recall scoring.
    let mut content_by_id: HashMap<MemoryRef, String> = HashMap::new();
    for item in &suite.memories {
        let vectors = embedder
            .embed(std::slice::from_ref(&item.content))
            .await
            .map_err(|e| anyhow!("embed: {e}"))?;
        let embedding = vectors.into_iter().next();
        let mem = Memory {
            id: new_id(),
            scope: scope.clone(),
            content: item.content.clone(),
            keywords: vec![],
            tags: item.tags.clone(),
            context: String::new(),
            embedding,
            links: vec![],
            parent: None,
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: Provenance {
                source: "memeval".into(),
                trust: 1.0,
            },
            source: None,
            position: None,
        };
        content_by_id.insert(MemoryRef(mem.id), mem.content.clone());
        let event = Event::MemoryWritten(mem);
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
    }

    let retriever = HybridRetriever::new(vector, bm25, embedder.clone());

    // --- question loop ---
    let mut recalled = 0usize;
    let mut cat: HashMap<String, (usize, usize)> = HashMap::new();
    let mut total_latency = 0.0f64;
    for q in &suite.questions {
        let query = Query {
            text: q.question.clone(),
            scope: scope.clone(),
            k,
            time_filter: None,
        };
        let start = Instant::now();
        let hits = retriever
            .search(&query)
            .await
            .map_err(|e| anyhow!("search: {e}"))?;
        total_latency += start.elapsed().as_secs_f64() * 1000.0;

        let needle = q.answer_substring.to_ascii_lowercase();
        let hit = hits.iter().any(|h| {
            content_by_id
                .get(&h.memory)
                .map(|c| c.to_ascii_lowercase().contains(&needle))
                .unwrap_or(false)
        });
        if hit {
            recalled += 1;
        }
        let e = cat.entry(q.category.clone()).or_insert((0, 0));
        e.1 += 1;
        if hit {
            e.0 += 1;
        }
    }

    // Stable, sorted category order for deterministic reports.
    let mut per_category: Vec<CategoryRecall> = cat
        .into_iter()
        .map(|(category, (recalled, total))| CategoryRecall {
            category,
            recalled,
            total,
        })
        .collect();
    per_category.sort_by(|a, b| a.category.cmp(&b.category));

    let total_questions = suite.questions.len();
    let report = MemEvalReport {
        suite_name: suite.name.clone(),
        embedder: embedder_choice.to_string(),
        k,
        memory_count: suite.memories.len(),
        total_questions,
        recalled,
        per_category,
        mean_latency_ms: if total_questions > 0 {
            total_latency / total_questions as f64
        } else {
            0.0
        },
    };

    // Best-effort cleanup of the temp keyspace.
    drop(log);
    std::fs::remove_dir_all(&dir).ok();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = r#"{
        "name": "tiny",
        "description": "smoke",
        "memories": [
            {"content": "Alice was promoted to Staff Engineer in March 2024", "tags": ["career"]},
            {"content": "Bob relocated to the Berlin office last quarter", "tags": ["location"]},
            {"content": "The Q3 revenue grew 18 percent year over year", "tags": ["finance"]}
        ],
        "questions": [
            {"question": "what role was Alice promoted to?", "answer_substring": "Staff Engineer", "category": "single-hop"},
            {"question": "where did Bob relocate?", "answer_substring": "Berlin", "category": "single-hop"},
            {"question": "how much did Q3 revenue grow?", "answer_substring": "18 percent", "category": "single-hop"}
        ]
    }"#;

    #[tokio::test]
    async fn recall_on_tiny_suite_is_high_with_mock_embedder() {
        let suite = load_memeval_suite(TINY).unwrap();
        let report = run_memeval(&suite, 5, "mock").await.unwrap();
        assert_eq!(report.total_questions, 3);
        assert_eq!(report.memory_count, 3);
        // BM25 alone recalls keyword-overlapping answers; expect all 3.
        assert_eq!(report.recalled, 3, "recall@5 should find all answers");
        assert!((report.recall() - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn report_tracks_per_category() {
        let suite = load_memeval_suite(TINY).unwrap();
        let report = run_memeval(&suite, 5, "mock").await.unwrap();
        assert_eq!(report.per_category.len(), 1);
        assert_eq!(report.per_category[0].category, "single-hop");
        assert_eq!(report.per_category[0].total, 3);
    }

    #[test]
    fn load_rejects_empty_suite() {
        let bad = r#"{"name":"x","description":"","memories":[],"questions":[]}"#;
        assert!(load_memeval_suite(bad).is_err());
    }
}
