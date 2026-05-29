//! # mneme-index
//!
//! Materialized retrieval views over the event log: a vector index
//! (`hnsw_rs`, Phase 0) and a BM25 index (`tantivy`, Phase 0), unified by a
//! hybrid [`Retriever`]. The custom filtered-HNSW variant (report §7, Phase 3)
//! lands here later — keep this trait-shaped so it can be swapped in.
//!
//! STATUS: Phase 0 in progress. `VectorView`, `Bm25View`, `MockEmbedder`,
//! and `HybridRetriever` (RRF fusion with explainable per-signal breakdown)
//! are all in. Custom filtered HNSW (Phase 3) and persistent indexes are
//! the next bumps.

pub mod bm25;
pub mod chunker;
pub mod embedder;
pub mod partitioned_vector;
pub mod synthesizer;
pub mod vector;

pub use bm25::{Bm25Tier, Bm25View};
pub use chunker::{Chunker, ParagraphChunker};
#[cfg(feature = "fastembed")]
pub use embedder::FastEmbedEmbedder;
pub use embedder::MockEmbedder;
pub use partitioned_vector::{TenantPartitionedVectorView, TenantStats};
pub use synthesizer::SnippetSynthesizer;
pub use vector::VectorView;

use mneme_core::types::MemoryRef;
use mneme_core::{Embedder, Hit, MnemeError, Query, Retriever};
use std::collections::HashMap;
use std::sync::Arc;

/// Reciprocal-rank fusion constant — the classic RRF paper uses k=60 and the
/// hybrid-search literature has converged on that value. Smaller k weights
/// the top of each list more aggressively; larger k flattens the curve.
const RRF_K: f32 = 60.0;

/// Hybrid retriever fusing the vector and BM25 views via *weighted* reciprocal
/// rank fusion. Each result's `breakdown` carries the per-signal raw score
/// (not normalized) plus the fused RRF component, so downstream UIs and tests
/// can see exactly why a memory ranked where it did.
///
/// **Why weighted, not flat RRF?** Standard RRF treats every signal equally.
/// That assumption breaks when one signal is non-semantic — e.g. when the
/// embedder is a stub that hashes bytes into vectors (Phase 0's
/// `MockEmbedder`). A noise signal at rank 1 will out-score a real signal
/// at rank 2 in flat RRF, polluting the result. Per-signal weights let us
/// down-weight a known-noisy signal until the real one lands (FastEmbed in
/// the next slice).
///
/// Default weights:
/// - `w_bm25 = 1.0`
/// - `w_vector = 0.0` if the embedder reports a `mock-` model_id, else `1.0`
pub struct HybridRetriever {
    vector: Arc<VectorView>,
    bm25: Arc<Bm25View>,
    embedder: Arc<dyn Embedder>,
    /// Over-fetch factor — pull `over_fetch * k` from each view before fusing
    /// so a hit that ranks 8th in one signal but 1st in another still has a
    /// chance to land in the top-k. Phase 0 default.
    over_fetch: usize,
    w_vector: f32,
    w_bm25: f32,
}

impl HybridRetriever {
    pub fn new(vector: Arc<VectorView>, bm25: Arc<Bm25View>, embedder: Arc<dyn Embedder>) -> Self {
        // Detect mock embedders so we don't let hash noise dictate ranking.
        // The convention `model_id() == "mock-v*"` is set by `MockEmbedder`;
        // a real local/HTTP embedder uses its own model identifier.
        let w_vector = if embedder.model_id().starts_with("mock-") {
            0.0
        } else {
            1.0
        };
        Self {
            vector,
            bm25,
            embedder,
            over_fetch: 4,
            w_vector,
            w_bm25: 1.0,
        }
    }

    /// Override the default per-signal weights. Useful when the host wants
    /// to A/B different fusion strategies or pin behavior in a test.
    pub fn with_weights(mut self, w_vector: f32, w_bm25: f32) -> Self {
        self.w_vector = w_vector;
        self.w_bm25 = w_bm25;
        self
    }

    /// Direct access for callers that want the individual signal lists (the
    /// viz/UI wants all three to render the score breakdown side-by-side).
    pub fn vector(&self) -> &VectorView {
        &self.vector
    }

    pub fn bm25(&self) -> &Bm25View {
        &self.bm25
    }

    pub fn embedder(&self) -> &dyn Embedder {
        self.embedder.as_ref()
    }

    pub fn weights(&self) -> (f32, f32) {
        (self.w_vector, self.w_bm25)
    }
}

#[async_trait::async_trait]
impl Retriever for HybridRetriever {
    async fn search(&self, query: &Query) -> Result<Vec<Hit>, MnemeError> {
        if query.k == 0 {
            return Ok(Vec::new());
        }
        let over_k = query.k.saturating_mul(self.over_fetch).max(query.k);

        // Embed the textual query once for the vector view; BM25 consumes the
        // raw text directly.
        let embeddings = self
            .embedder
            .embed(std::slice::from_ref(&query.text))
            .await?;
        let query_vec = embeddings
            .first()
            .ok_or_else(|| MnemeError::Index("embedder returned no vectors".into()))?;

        let vector_hits = self.vector.search(query_vec, over_k, &query.scope)?;
        let bm25_hits = self.bm25.search(&query.text, over_k, &query.scope)?;

        // Weighted RRF fuses by *ranks*, scaled by per-signal weight. Rank-
        // based fusion is robust to wildly different score scales between
        // BM25 and L2/cosine; weighting on top of that lets us mute a
        // known-noisy signal (Phase-0 mock embedder) without re-engineering
        // the fusion. We keep the raw scores in the breakdown for visibility
        // even when a signal's weight is zero.
        let mut bucket: HashMap<MemoryRef, FusedHit> = HashMap::new();
        for (rank, h) in vector_hits.iter().enumerate() {
            let e = bucket.entry(h.memory).or_default();
            e.vector_score = h.score;
            e.rrf += self.w_vector / (RRF_K + (rank + 1) as f32);
        }
        for (rank, h) in bm25_hits.iter().enumerate() {
            let e = bucket.entry(h.memory).or_default();
            e.bm25_score = h.score;
            e.rrf += self.w_bm25 / (RRF_K + (rank + 1) as f32);
        }

        let mut hits: Vec<Hit> = bucket
            .into_iter()
            // A zero rrf means no weighted signal contributed — drop the
            // hit rather than letting noise from a 0-weighted signal show
            // up at the bottom of the list.
            .filter(|(_, f)| f.rrf > 0.0)
            .map(|(memory, f)| Hit {
                memory,
                score: f.rrf,
                breakdown: vec![
                    ("vector".to_string(), f.vector_score),
                    ("bm25".to_string(), f.bm25_score),
                    ("rrf".to_string(), f.rrf),
                ],
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(query.k);
        Ok(hits)
    }
}

#[derive(Default)]
struct FusedHit {
    vector_score: f32,
    bm25_score: f32,
    rrf: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mneme_core::entity::{Memory, Provenance};
    use mneme_core::event::{Event, LogEntry};
    use mneme_core::traits::MaterializedView;
    use mneme_core::types::{new_id, BiTemporal};
    use mneme_core::Scope;

    async fn build_corpus() -> (
        Arc<VectorView>,
        Arc<Bm25View>,
        Arc<MockEmbedder>,
        Scope,
        Vec<MemoryRef>,
    ) {
        let embedder = Arc::new(MockEmbedder::new(32));
        let vector = Arc::new(VectorView::new(
            embedder.dim(),
            embedder.model_id().to_string(),
        ));
        let bm25 = Arc::new(Bm25View::new().unwrap());
        let scope = Scope::global("test");

        let contents = [
            ("acme reported quarterly earnings", "earnings"),
            ("revenue up 15% year over year", "revenue"),
            ("supply chain stabilizing into Q4", "operations"),
            ("EMEA margins compressed this quarter", "margin"),
        ];
        let mut refs = Vec::new();
        for (text, tag) in contents {
            let emb = embedder.embed(&[text.to_string()]).await.unwrap();
            let m = Memory {
                id: new_id(),
                scope: scope.clone(),
                content: text.into(),
                keywords: vec![],
                tags: vec![tag.into()],
                context: String::new(),
                embedding: Some(emb.into_iter().next().unwrap()),
                links: vec![],
                parent: None,
                evolution_count: 0,
                time: BiTemporal::now(),
                provenance: Provenance::default(),
                source: None,
                position: None,
            };
            refs.push(MemoryRef(m.id));
            let entry = LogEntry {
                id: new_id(),
                event: Event::MemoryWritten(m),
            };
            vector.apply(&entry).await.unwrap();
            bm25.apply(&entry).await.unwrap();
        }
        (vector, bm25, embedder, scope, refs)
    }

    #[tokio::test]
    async fn hybrid_returns_bm25_keyword_match_on_top() {
        let (vector, bm25, embedder, scope, refs) = build_corpus().await;
        let retriever = HybridRetriever::new(vector, bm25, embedder);
        let hits = retriever
            .search(&Query {
                text: "revenue".into(),
                scope,
                k: 3,
                time_filter: None,
            })
            .await
            .unwrap();
        assert!(!hits.is_empty());
        let ids: Vec<_> = hits.iter().map(|h| h.memory).collect();
        assert!(
            ids.contains(&refs[1]),
            "revenue memory must appear in hybrid top-k; got {ids:?}"
        );
    }

    #[tokio::test]
    async fn breakdown_exposes_all_three_signals() {
        let (vector, bm25, embedder, scope, _refs) = build_corpus().await;
        let retriever = HybridRetriever::new(vector, bm25, embedder);
        let hits = retriever
            .search(&Query {
                text: "earnings".into(),
                scope,
                k: 2,
                time_filter: None,
            })
            .await
            .unwrap();
        let h = hits.first().expect("expected at least one hit");
        let names: Vec<_> = h.breakdown.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["vector", "bm25", "rrf"]);
    }

    #[tokio::test]
    async fn mock_embedder_zeroes_vector_weight() {
        let embedder = Arc::new(MockEmbedder::new(32));
        let vector = Arc::new(VectorView::new(
            embedder.dim(),
            embedder.model_id().to_string(),
        ));
        let bm25 = Arc::new(Bm25View::new().unwrap());
        let r = HybridRetriever::new(vector, bm25, embedder);
        let (wv, wb) = r.weights();
        assert_eq!(wv, 0.0, "mock embedder must zero out vector weight");
        assert_eq!(wb, 1.0);
    }

    #[tokio::test]
    async fn with_weights_overrides_default() {
        let embedder = Arc::new(MockEmbedder::new(32));
        let vector = Arc::new(VectorView::new(
            embedder.dim(),
            embedder.model_id().to_string(),
        ));
        let bm25 = Arc::new(Bm25View::new().unwrap());
        let r = HybridRetriever::new(vector, bm25, embedder).with_weights(0.7, 0.3);
        assert_eq!(r.weights(), (0.7, 0.3));
    }

    #[tokio::test]
    async fn zero_vector_weight_excludes_vector_only_hits() {
        // Set up: corpus with a strong BM25 match and a high-ranking vector
        // match for a "different" memory. With w_vector=0, only the BM25
        // match should land in hybrid results.
        let (vector, bm25, embedder, scope, refs) = build_corpus().await;
        let retriever = HybridRetriever::new(vector, bm25, embedder); // mock → w_vec=0

        let hits = retriever
            .search(&Query {
                text: "revenue".into(),
                scope,
                k: 5,
                time_filter: None,
            })
            .await
            .unwrap();

        // Every hit must have a non-zero BM25 contribution — vector-only
        // matches (the hash noise) must not appear.
        for h in &hits {
            let bm = h
                .breakdown
                .iter()
                .find(|(n, _)| n == "bm25")
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
            assert!(
                bm > 0.0,
                "with w_vector=0 every hybrid hit must have BM25 support; got {h:?}"
            );
        }
        // And the revenue memory must be the top hit, not noise.
        assert!(!hits.is_empty());
        assert_eq!(
            hits[0].memory, refs[1],
            "revenue memory must rank #1 with BM25-only fusion"
        );
    }

    #[tokio::test]
    async fn k_zero_returns_empty() {
        let (vector, bm25, embedder, scope, _) = build_corpus().await;
        let retriever = HybridRetriever::new(vector, bm25, embedder);
        let hits = retriever
            .search(&Query {
                text: "revenue".into(),
                scope,
                k: 0,
                time_filter: None,
            })
            .await
            .unwrap();
        assert!(hits.is_empty());
    }
}
