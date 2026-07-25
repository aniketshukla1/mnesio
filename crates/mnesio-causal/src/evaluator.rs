//! A real [`CounterfactualEvaluator`] over the [`Retriever`] seam.
//!
//! The objective is **recall@k** over a held-out task set: each task names a
//! query and the memories that *should* surface for it. Masking is done
//! *post-retrieval* — fetch `k + |masked| + over_fetch` hits, drop the masked
//! ids, then truncate to `k` — so we never have to teach the index about
//! exclusion. The over-fetch margin guarantees removing masked hits can't
//! starve the top-k of results that would otherwise have ranked in.
//!
//! Depends only on `mnesio-core` (the `Retriever` trait), so the production
//! `HybridRetriever` drops in as `Arc<dyn Retriever>` and tests use a tiny
//! fake — no `mnesio-index` coupling (Hard Rule #7).

use crate::contribution::CounterfactualEvaluator;
use async_trait::async_trait;
use mnesio_core::traits::{Query, Retriever};
use mnesio_core::types::MemoryRef;
use mnesio_core::{MnesioError, Scope};
use std::collections::HashSet;
use std::sync::Arc;

/// One held-out retrieval task: a query and the memories that are *relevant*
/// to it (the ground truth for recall@k).
#[derive(Debug, Clone)]
pub struct RetrievalTask {
    pub query: String,
    pub relevant: HashSet<MemoryRef>,
}

/// Recall@k evaluator. The baseline (`evaluate(&{})`) is the corpus's recall
/// with nothing masked; each LOO call measures the recall with one memory
/// hidden.
pub struct RetrievalEvaluator {
    retriever: Arc<dyn Retriever>,
    scope: Scope,
    tasks: Vec<RetrievalTask>,
    k: usize,
    over_fetch: usize,
}

impl RetrievalEvaluator {
    /// `over_fetch` defaults to 32 — generous enough that LOO masking never
    /// starves the top-k. Tune down to 0 only when the retriever returns a
    /// stable full ranking (e.g. tests).
    pub fn new(retriever: Arc<dyn Retriever>, scope: Scope, k: usize) -> Self {
        Self {
            retriever,
            scope,
            tasks: Vec::new(),
            k,
            over_fetch: 32,
        }
    }

    pub fn with_over_fetch(mut self, over_fetch: usize) -> Self {
        self.over_fetch = over_fetch;
        self
    }

    pub fn with_task(
        mut self,
        query: impl Into<String>,
        relevant: impl IntoIterator<Item = MemoryRef>,
    ) -> Self {
        self.tasks.push(RetrievalTask {
            query: query.into(),
            relevant: relevant.into_iter().collect(),
        });
        self
    }

    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// The memories worth scoring: the de-duplicated union of every task's
    /// relevant set, in first-seen order. The natural candidate list to hand
    /// [`crate::ContributionScorer::score`].
    pub fn candidate_universe(&self) -> Vec<MemoryRef> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for task in &self.tasks {
            for m in &task.relevant {
                if seen.insert(*m) {
                    out.push(*m);
                }
            }
        }
        out
    }

    /// Recall of a single task given the (already mask-filtered) top-k ids.
    fn task_recall(task: &RetrievalTask, topk: &[MemoryRef]) -> f32 {
        if task.relevant.is_empty() {
            return 0.0;
        }
        let found = topk.iter().filter(|m| task.relevant.contains(*m)).count();
        found as f32 / task.relevant.len() as f32
    }
}

#[async_trait]
impl CounterfactualEvaluator for RetrievalEvaluator {
    async fn evaluate(&self, masked: &HashSet<MemoryRef>) -> Result<f32, MnesioError> {
        if self.tasks.is_empty() {
            return Ok(0.0);
        }
        let fetch_k = self.k + masked.len() + self.over_fetch;
        let mut total = 0.0f32;
        for task in &self.tasks {
            let query = Query {
                text: task.query.clone(),
                scope: self.scope.clone(),
                k: fetch_k,
                time_filter: None,
            };
            let hits = self.retriever.search(&query).await?;
            let topk: Vec<MemoryRef> = hits
                .into_iter()
                .map(|h| h.memory)
                .filter(|m| !masked.contains(m))
                .take(self.k)
                .collect();
            total += Self::task_recall(task, &topk);
        }
        Ok(total / self.tasks.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contribution::{CausalConfig, ContributionScorer};
    use mnesio_core::traits::Hit;
    use mnesio_core::types::new_id;
    use std::collections::HashMap;

    /// Returns a fixed ranked list per query text, honoring `query.k`.
    struct FakeRetriever {
        table: HashMap<String, Vec<MemoryRef>>,
    }

    #[async_trait]
    impl Retriever for FakeRetriever {
        async fn search(&self, query: &Query) -> Result<Vec<Hit>, MnesioError> {
            let ranked = self.table.get(&query.text).cloned().unwrap_or_default();
            Ok(ranked
                .into_iter()
                .take(query.k)
                .enumerate()
                .map(|(i, memory)| Hit {
                    memory,
                    score: 1.0 / (i as f32 + 1.0),
                    breakdown: vec![],
                })
                .collect())
        }
    }

    fn mref() -> MemoryRef {
        MemoryRef(new_id())
    }

    fn evaluator(
        table: HashMap<String, Vec<MemoryRef>>,
    ) -> (RetrievalEvaluator, Arc<FakeRetriever>) {
        let r = Arc::new(FakeRetriever { table });
        (
            RetrievalEvaluator::new(r.clone(), Scope::global("t"), 3).with_over_fetch(0),
            r,
        )
    }

    #[tokio::test]
    async fn recall_at_k_over_the_retriever_seam() {
        let (m1, m2, m3) = (mref(), mref(), mref());
        let table = HashMap::from([("q".to_string(), vec![m1, m2, m3])]);
        let (eval, _) = evaluator(table);
        let eval = eval.with_task("q", [m1, m3]);
        let recall = eval.evaluate(&HashSet::new()).await.unwrap();
        assert!((recall - 1.0).abs() < 1e-6, "both relevant in top-3");
    }

    #[tokio::test]
    async fn masking_a_relevant_memory_drops_recall() {
        let (m1, m2, m3) = (mref(), mref(), mref());
        let table = HashMap::from([("q".to_string(), vec![m1, m2, m3])]);
        let (eval, _) = evaluator(table);
        let eval = eval.with_task("q", [m1, m3]);
        let masked = eval.evaluate(&HashSet::from([m1])).await.unwrap();
        assert!((masked - 0.5).abs() < 1e-6, "1 of 2 relevant remains → 0.5");
    }

    #[tokio::test]
    async fn masking_an_irrelevant_memory_leaves_recall_unchanged() {
        let (m1, m2, m3) = (mref(), mref(), mref());
        let table = HashMap::from([("q".to_string(), vec![m1, m2, m3])]);
        let (eval, _) = evaluator(table);
        let eval = eval.with_task("q", [m1, m3]);
        let masked = eval.evaluate(&HashSet::from([m2])).await.unwrap();
        assert!(
            (masked - 1.0).abs() < 1e-6,
            "masking irrelevant m2 changes nothing"
        );
    }

    #[tokio::test]
    async fn candidate_universe_is_deduped_union_of_relevant() {
        let (m1, m2, m3) = (mref(), mref(), mref());
        let table = HashMap::new();
        let (eval, _) = evaluator(table);
        let eval = eval.with_task("q1", [m1, m2]).with_task("q2", [m2, m3]);
        let universe = eval.candidate_universe();
        assert_eq!(universe.len(), 3);
        for m in [m1, m2, m3] {
            assert!(universe.contains(&m));
        }
    }

    // Slice B ↔ Slice A: contribution scoring over the *real Retriever seam*
    // reproduces the Phase-10 "done when" — the load-bearing memories score
    // positive, the dead-weight one scores ~0 and is the GC candidate.
    //
    // The candidate list is the *corpus* ([m1, m2, m3]) — what a production
    // pass hands in (all live memories in scope) — while relevance is the
    // separate ground truth ({m1, m3}). m2 is retrieved but never relevant,
    // so masking it leaves recall untouched: a provable GC candidate.
    #[tokio::test]
    async fn scoring_over_the_retriever_recovers_keep_and_gc_sets() {
        let (m1, m2, m3) = (mref(), mref(), mref());
        let table = HashMap::from([("q".to_string(), vec![m1, m2, m3])]);
        let (eval, _) = evaluator(table);
        let eval = eval.with_task("q", [m1, m3]); // m2 present in corpus but irrelevant

        let report = ContributionScorer::new(CausalConfig::default())
            .score(&eval, &[m1, m2, m3])
            .await
            .unwrap();

        let by = |m: MemoryRef| {
            report
                .scored
                .iter()
                .find(|c| c.memory == m)
                .unwrap()
                .contribution
        };
        assert!((by(m1) - 0.5).abs() < 1e-6, "m1 load-bearing");
        assert!((by(m3) - 0.5).abs() < 1e-6, "m3 load-bearing");
        assert!(by(m2).abs() < 1e-6, "m2 contributes nothing");

        let gc = report.gc_candidates(1e-4);
        assert_eq!(
            gc,
            vec![m2],
            "only the dead-weight memory is a GC candidate"
        );
    }

    #[tokio::test]
    async fn empty_task_set_scores_zero() {
        let (eval, _) = evaluator(HashMap::new());
        assert_eq!(eval.evaluate(&HashSet::new()).await.unwrap(), 0.0);
    }
}
