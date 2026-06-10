//! End-to-end **KV-cartridge accuracy-parity** eval — the Phase-12 "done when".
//!
//! The cartridge claim ("answer from a precompiled KV cache at equal accuracy,
//! much faster") had two proofs already: the token-identical *self-consistency
//! oracle* (generation from the cartridge == generation from the full prompt)
//! and single live factoids. What was missing is a **suite-level** number, the
//! way LOCOMO/LongMemEval report accuracy. This supplies it.
//!
//! For every question we answer twice through one [`KvBackend`] and compare:
//!
//! 1. **Cartridge** — compile the whole memory set into a KV blob **once**, then
//!    answer each query from that single blob.
//! 2. **Text-context (RAG baseline)** — for each query, retrieve the top-`k`
//!    memories through the *real* hybrid pipeline (`FjallEventLog` → vector +
//!    BM25 → RRF), compile *those* into a blob, and answer.
//!
//! Headlines: the **parity delta** (cartridge accuracy − text-context accuracy;
//! `≥ 0` means the cartridge doesn't lose accuracy) and the **speedup** (the
//! text-context path pays a fresh compile *per query*; the cartridge compiles
//! once and replays — the amortization the cartridge exists for). A final
//! **erasure check** drops the source memory of a correctly-answered question,
//! recompiles, and confirms the answer is gone — the recompile-from-shrunk-corpus
//! half of crypto-shred, demonstrated at suite level.
//!
//! Backend-parametric. The default [`FakeKvBackend`] is deterministic + offline
//! (so this gates CI), and its accuracy is a **mechanism demonstration, not a
//! published number** — exactly like [`crate::qaeval`]'s demo LLM. A real
//! generative backend (feature-gated) run through the same [`run_kveval`] yields
//! the publishable parity number; [`KvEvalReport::is_real`] flags which it is.

use crate::memeval::MemEvalSuite;
use anyhow::{anyhow, Result};
use mneme_core::entity::{Memory, Provenance};
use mneme_core::event::{Event, LogEntry};
use mneme_core::traits::MaterializedView;
use mneme_core::types::{new_id, BiTemporal, MemoryRef, Scope};
use mneme_core::{Embedder, EventLog, Query, Retriever};
use mneme_index::{Bm25View, HybridRetriever, MockEmbedder, VectorView};
use mneme_kv::KvBackend;
use mneme_store::FjallEventLog;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Offline mock embedder dim — matches the server/CI default.
const MOCK_DIM: usize = 32;

/// Result of a KV accuracy-parity run.
pub struct KvEvalReport {
    pub suite_name: String,
    /// Label for the KV backend (`fake`, `generative`, `qwen`, `candle`).
    /// Only non-`fake` runs are publishable numbers.
    pub backend: String,
    pub k: usize,
    pub member_count: usize,
    pub total: usize,
    pub cartridge_correct: usize,
    pub textctx_correct: usize,
    /// Mean per-query latency answering from the precompiled cartridge (µs).
    pub cartridge_us_mean: f64,
    /// Mean per-query latency for the text-context path: retrieve + compile a
    /// fresh per-query blob + answer (µs).
    pub textctx_us_mean: f64,
    /// True if dropping a correctly-answered question's source memory and
    /// recompiling made the cartridge unable to answer it (erasure-by-recompile).
    pub erasure_ok: bool,
}

impl KvEvalReport {
    pub fn cartridge_acc(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.cartridge_correct as f32 / self.total as f32
        }
    }

    pub fn textctx_acc(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.textctx_correct as f32 / self.total as f32
        }
    }

    /// Cartridge accuracy minus text-context accuracy. `≥ 0` = the cartridge
    /// does not lose accuracy versus per-query retrieval — the parity claim.
    pub fn parity_delta(&self) -> f32 {
        self.cartridge_acc() - self.textctx_acc()
    }

    /// How many times faster the cartridge answers than the text-context path
    /// (which recompiles per query). `0.0` if not measurable.
    pub fn speedup(&self) -> f64 {
        if self.cartridge_us_mean > 0.0 {
            self.textctx_us_mean / self.cartridge_us_mean
        } else {
            0.0
        }
    }

    /// True only for a real model backend — the [`FakeKvBackend`] number is a
    /// plumbing demonstration, never a published score.
    pub fn is_real(&self) -> bool {
        self.backend != "fake"
    }
}

/// Run the accuracy-parity eval over `suite` using `backend` for both the
/// cartridge and the text-context baseline (so the comparison is apples-to-apples
/// — the only difference is *when* compilation happens).
pub async fn run_kveval(
    suite: &MemEvalSuite,
    k: usize,
    backend: &dyn KvBackend,
    backend_label: &str,
) -> Result<KvEvalReport> {
    let scope = Scope::global("kveval");
    let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(MOCK_DIM));

    let dir = std::env::temp_dir().join(format!("mneme-kveval-{}", new_id()));
    let log = FjallEventLog::open(&dir).map_err(|e| anyhow!("open log: {e}"))?;
    let vector = Arc::new(VectorView::new(
        embedder.dim(),
        embedder.model_id().to_string(),
    ));
    let bm25 = Arc::new(Bm25View::new().map_err(|e| anyhow!("bm25 init: {e}"))?);

    // Memory contents in suite order — the index aligns with the cartridge's
    // member list, which the erasure check relies on.
    let contents: Vec<String> = suite.memories.iter().map(|m| m.content.clone()).collect();

    // Ingest the haystack through the real pipeline for the text-context path.
    let mut content_by_id: HashMap<MemoryRef, String> = HashMap::new();
    for item in &suite.memories {
        let embedding = embedder
            .embed(std::slice::from_ref(&item.content))
            .await
            .map_err(|e| anyhow!("embed: {e}"))?
            .into_iter()
            .next();
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
                source: "kveval".into(),
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
            .map_err(|e| anyhow!("v apply: {e}"))?;
        bm25.apply(&entry)
            .await
            .map_err(|e| anyhow!("b apply: {e}"))?;
    }
    let retriever = HybridRetriever::new(vector, bm25, embedder.clone());

    // Cartridge: compile the whole corpus ONCE; queries replay this blob.
    let cartridge_blob = backend.compile_blob(&contents).await;

    let mut cartridge_correct = 0usize;
    let mut textctx_correct = 0usize;
    let mut cartridge_us = 0f64;
    let mut textctx_us = 0f64;
    // (member index, question) of the first cartridge-correct question — the
    // target for the erasure check.
    let mut erase_target: Option<(usize, String, String)> = None;

    for q in &suite.questions {
        let gold = q.answer_substring.to_ascii_lowercase();

        // --- Cartridge path: answer from the single precompiled blob. ---
        let t0 = Instant::now();
        let a = backend.answer(&cartridge_blob, &q.question).await;
        cartridge_us += t0.elapsed().as_secs_f64() * 1e6;
        let cart_ok = a.answered && a.text.to_ascii_lowercase().contains(&gold);
        if cart_ok {
            cartridge_correct += 1;
            if erase_target.is_none() {
                if let Some(idx) = contents
                    .iter()
                    .position(|c| c.to_ascii_lowercase().contains(&gold))
                {
                    erase_target = Some((idx, q.question.clone(), gold.clone()));
                }
            }
        }

        // --- Text-context path: retrieve top-k, compile *those* per query. ---
        let t1 = Instant::now();
        let hits = retriever
            .search(&Query {
                text: q.question.clone(),
                scope: scope.clone(),
                k,
                time_filter: None,
            })
            .await
            .map_err(|e| anyhow!("search: {e}"))?;
        let ctx: Vec<String> = hits
            .iter()
            .filter_map(|h| content_by_id.get(&h.memory).cloned())
            .collect();
        let blob = backend.compile_blob(&ctx).await;
        let a2 = backend.answer(&blob, &q.question).await;
        textctx_us += t1.elapsed().as_secs_f64() * 1e6;
        if a2.answered && a2.text.to_ascii_lowercase().contains(&gold) {
            textctx_correct += 1;
        }
    }

    // Erasure check: drop the source memory of a correctly-answered question,
    // recompile, and confirm the cartridge can no longer answer it.
    let erasure_ok = match erase_target {
        Some((idx, question, gold)) => {
            let shrunk: Vec<String> = contents
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, c)| c.clone())
                .collect();
            let blob2 = backend.compile_blob(&shrunk).await;
            let a = backend.answer(&blob2, &question).await;
            !(a.answered && a.text.to_ascii_lowercase().contains(&gold))
        }
        // Nothing was answered correctly → erasure holds vacuously.
        None => true,
    };

    let total = suite.questions.len();
    drop(log);
    std::fs::remove_dir_all(&dir).ok();

    Ok(KvEvalReport {
        suite_name: suite.name.clone(),
        backend: backend_label.to_string(),
        k,
        member_count: contents.len(),
        total,
        cartridge_correct,
        textctx_correct,
        cartridge_us_mean: if total > 0 {
            cartridge_us / total as f64
        } else {
            0.0
        },
        textctx_us_mean: if total > 0 {
            textctx_us / total as f64
        } else {
            0.0
        },
        erasure_ok,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memeval::{MemItem, MemQuestion};
    use mneme_kv::FakeKvBackend;

    fn suite() -> MemEvalSuite {
        MemEvalSuite {
            name: "kv-tiny".into(),
            description: "smoke".into(),
            memories: vec![
                MemItem {
                    content: "Alice was promoted to Staff Engineer in March 2024".into(),
                    tags: vec![],
                },
                MemItem {
                    content: "Bob relocated to the Berlin office last quarter".into(),
                    tags: vec![],
                },
                MemItem {
                    content: "Carol leads the payments platform team".into(),
                    tags: vec![],
                },
            ],
            questions: vec![
                MemQuestion {
                    question: "what role was Alice promoted to?".into(),
                    answer_substring: "Staff Engineer".into(),
                    category: "single-hop".into(),
                },
                MemQuestion {
                    question: "where did Bob relocate?".into(),
                    answer_substring: "Berlin".into(),
                    category: "single-hop".into(),
                },
            ],
        }
    }

    #[tokio::test]
    async fn cartridge_parity_and_erasure_hold() {
        let backend = FakeKvBackend::new("fake-kv");
        let r = run_kveval(&suite(), 5, &backend, "fake").await.unwrap();
        assert_eq!(r.total, 2);
        // The cartridge sees every memory; the text-context path sees only the
        // top-k it retrieved — so the cartridge can never *lose* accuracy.
        assert!(
            r.parity_delta() >= 0.0,
            "cartridge accuracy {:.3} < text-context {:.3}",
            r.cartridge_acc(),
            r.textctx_acc()
        );
        assert!(r.cartridge_correct >= 1, "cartridge answers from its blob");
        assert!(
            r.erasure_ok,
            "dropping the source memory + recompiling removes the answer"
        );
        assert!(
            !r.is_real(),
            "the fake backend is a mechanism demo, not a published number"
        );
    }
}
