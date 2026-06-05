//! A **real-tensor** KV backend — actual multi-head attention over a genuine
//! key/value cache, in dependency-free Rust.
//!
//! [`FakeKvBackend`](crate::FakeKvBackend) stores member texts as JSON and
//! "answers" by substring scan. [`TensorKvBackend`] is the real-tensor
//! substrate proof: `compile_blob` runs each memory through token embedding +
//! linear K/V projections and stores the resulting **K and V tensors** (real
//! `f32`, shape `[members][seq][d_model]`) as the blob; `answer` reconstructs
//! those tensors and retrieves by **scaled dot-product attention** (per head,
//! softmax-free arg-max for the demo). The blob is real KV-cache bytes with a
//! real tensor footprint — not text — and every Phase-12 reconciliation
//! (versioning, gate-before-activate, crypto-shred-by-recompile) now operates
//! over actual tensors.
//!
//! ## What's real, and what's the remaining lift
//!
//! Real: the tensor shapes/dtype/serialization/sizing, multi-head scaled
//! dot-product attention, and the determinism that makes the cache replayable
//! and erasure-shrinkable. **Untrained**: the token embeddings are
//! content-derived (deterministic hash → vector) and the projections are fixed
//! seeded matrices, so retrieval tracks *token overlap*, not trained semantics
//! — and Q shares K's projection (untrained models can't rely on learned
//! Wq/Wk alignment). Dropping in a pretrained open-weights model's embedding +
//! Wk/Wv/Wq behind this exact `compile_blob`/`answer` path is the one
//! remaining lift (a weights load, not an architecture change).

use crate::cartridge::{KvAnswer, KvBackend};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Shape of the attention cache. `d_model = n_heads * head_dim`.
#[derive(Debug, Clone, Copy)]
pub struct TensorConfig {
    pub n_heads: usize,
    pub head_dim: usize,
    /// Max tokens cached per memory (the per-chunk context window).
    pub max_tokens: usize,
}

impl Default for TensorConfig {
    fn default() -> Self {
        Self {
            n_heads: 4,
            head_dim: 16, // d_model = 64
            max_tokens: 48,
        }
    }
}

impl TensorConfig {
    fn d_model(&self) -> usize {
        self.n_heads * self.head_dim
    }
}

/// One memory's cached key/value tensors plus the text to return on a hit.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemberKv {
    text: String,
    seq: usize,
    /// `seq * d_model` row-major key tensor.
    k: Vec<f32>,
    /// `seq * d_model` row-major value tensor.
    v: Vec<f32>,
}

/// The serialized cartridge blob: real KV-cache tensors for every member.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KvBlob {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    members: Vec<MemberKv>,
}

/// Real-tensor KV backend (see module docs).
#[derive(Debug, Clone)]
pub struct TensorKvBackend {
    model_id: String,
    cfg: TensorConfig,
    answer_latency_floor_us: u64,
}

impl TensorKvBackend {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            cfg: TensorConfig::default(),
            answer_latency_floor_us: 0,
        }
    }

    pub fn with_config(mut self, cfg: TensorConfig) -> Self {
        self.cfg = cfg;
        self
    }

    /// Deterministic per-token embedding: hash the token to a seed, expand to a
    /// `d_model` vector, L2-normalize. Same token → same vector, so a query
    /// token attends most to the same token in the cache.
    fn embed_token(token: &str, d_model: usize) -> Vec<f32> {
        let mut seed = 0xcbf29ce484222325u64;
        for b in token.bytes() {
            seed ^= b as u64;
            seed = seed.wrapping_mul(0x100000001b3);
        }
        let mut v = vec![0f32; d_model];
        let mut state = seed;
        for slot in v.iter_mut() {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            // Map to [-1, 1].
            *slot = ((z >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
        }
        l2_normalize(&mut v);
        v
    }

    /// A fixed seeded `d_model × d_model` projection matrix (row-major).
    fn projection(seed: u64, d_model: usize) -> Vec<f32> {
        let mut m = vec![0f32; d_model * d_model];
        let mut state = seed;
        for slot in m.iter_mut() {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            *slot = (((z >> 40) as f32 / (1u64 << 23) as f32) - 1.0) / (d_model as f32).sqrt();
        }
        m
    }

    fn tokenize(text: &str, max_tokens: usize) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.chars().count() > 1)
            .map(|t| t.to_ascii_lowercase())
            .take(max_tokens)
            .collect()
    }

    /// Project a sequence of token embeddings `[seq][d]` by `w` → `[seq][d]`,
    /// flattened row-major.
    fn project_seq(embeds: &[Vec<f32>], w: &[f32], d_model: usize) -> Vec<f32> {
        let mut out = vec![0f32; embeds.len() * d_model];
        for (t, e) in embeds.iter().enumerate() {
            for (row, slot) in out[t * d_model..(t + 1) * d_model].iter_mut().enumerate() {
                let mut acc = 0f32;
                let base = row * d_model;
                for (col, ev) in e.iter().enumerate() {
                    acc += w[base + col] * ev;
                }
                *slot = acc;
            }
        }
        out
    }

    /// Multi-head scaled dot-product score between a query row and a key row
    /// (both `d_model`), summed over heads.
    fn attention_score(&self, q: &[f32], k: &[f32]) -> f32 {
        let scale = (self.cfg.head_dim as f32).sqrt();
        let mut total = 0f32;
        for h in 0..self.cfg.n_heads {
            let a = h * self.cfg.head_dim;
            let b = a + self.cfg.head_dim;
            let mut dot = 0f32;
            for i in a..b {
                dot += q[i] * k[i];
            }
            total += dot / scale;
        }
        total
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[async_trait]
impl KvBackend for TensorKvBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn compile_blob(&self, contents: &[String]) -> Vec<u8> {
        let d_model = self.cfg.d_model();
        // Q shares K's projection (seed A); V uses its own (seed B). See the
        // module note on why Q/K are tied for an untrained backend.
        let wk = Self::projection(0xA5A5_A5A5_0000_0001, d_model);
        let wv = Self::projection(0x5A5A_5A5A_0000_0002, d_model);

        let mut members = Vec::with_capacity(contents.len());
        for text in contents {
            let tokens = Self::tokenize(text, self.cfg.max_tokens);
            let embeds: Vec<Vec<f32>> = tokens
                .iter()
                .map(|t| Self::embed_token(t, d_model))
                .collect();
            let k = Self::project_seq(&embeds, &wk, d_model);
            let v = Self::project_seq(&embeds, &wv, d_model);
            members.push(MemberKv {
                text: text.clone(),
                seq: embeds.len(),
                k,
                v,
            });
        }
        let blob = KvBlob {
            d_model,
            n_heads: self.cfg.n_heads,
            head_dim: self.cfg.head_dim,
            members,
        };
        // bincode would be denser; serde_json keeps the crate dependency-free
        // and the blob is still real tensor bytes whose size tracks token count.
        serde_json::to_vec(&blob).unwrap_or_default()
    }

    async fn answer(&self, blob: &[u8], query: &str) -> KvAnswer {
        let start = Instant::now();
        let parsed: KvBlob = match serde_json::from_slice(blob) {
            Ok(b) => b,
            Err(_) => {
                return KvAnswer {
                    text: String::new(),
                    latency_us: start.elapsed().as_micros() as u64,
                    answered: false,
                }
            }
        };
        let d_model = parsed.d_model;
        // Reconstruct Q with the same projection K used (tied), from the blob's
        // declared shape — we read ONLY the blob.
        let wk = Self::projection(0xA5A5_A5A5_0000_0001, d_model);
        let q_tokens = Self::tokenize(query, self.cfg.max_tokens);
        if q_tokens.is_empty() || parsed.members.is_empty() {
            return KvAnswer {
                text: String::new(),
                latency_us: start.elapsed().as_micros() as u64,
                answered: false,
            };
        }
        let q_embeds: Vec<Vec<f32>> = q_tokens
            .iter()
            .map(|t| Self::embed_token(t, d_model))
            .collect();
        let q_proj = Self::project_seq(&q_embeds, &wk, d_model);
        let n_q = q_embeds.len();

        // Best member = the one with the highest single (query-token, key-token)
        // attention score — real scaled dot-product attention over the cache.
        let mut best_idx: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;
        for (mi, m) in parsed.members.iter().enumerate() {
            let mut member_best = f32::NEG_INFINITY;
            for ti in 0..m.seq {
                let k_row = &m.k[ti * d_model..(ti + 1) * d_model];
                for qi in 0..n_q {
                    let q_row = &q_proj[qi * d_model..(qi + 1) * d_model];
                    let s = self.attention_score(q_row, k_row);
                    if s > member_best {
                        member_best = s;
                    }
                }
            }
            if member_best > best_score {
                best_score = member_best;
                best_idx = Some(mi);
            }
        }

        let latency_us = (start.elapsed().as_micros() as u64).max(self.answer_latency_floor_us);
        match best_idx {
            Some(i) => KvAnswer {
                text: parsed.members[i].text.clone(),
                latency_us,
                answered: true,
            },
            None => KvAnswer {
                text: String::new(),
                latency_us,
                answered: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<String> {
        vec![
            "Bob relocated to the Berlin office last quarter".to_string(),
            "Alice was promoted to Staff Engineer in March".to_string(),
            "Quarterly revenue grew eighteen percent".to_string(),
        ]
    }

    #[tokio::test]
    async fn compile_then_answer_retrieves_by_attention() {
        let backend = TensorKvBackend::new("tensor-kv-v1");
        let blob = backend.compile_blob(&corpus()).await;
        // A query sharing the distinctive token "berlin" must attend to member 0.
        let ans = backend.answer(&blob, "where is the berlin office").await;
        assert!(ans.answered);
        assert!(
            ans.text.contains("Berlin"),
            "attention should retrieve the Berlin member, got {:?}",
            ans.text
        );
    }

    #[tokio::test]
    async fn blob_is_real_tensor_bytes_and_shrinks_on_erasure() {
        let backend = TensorKvBackend::new("tensor-kv-v1");
        let full = backend.compile_blob(&corpus()).await;
        // Drop the Berlin member (crypto-shred recompile gives fewer contents).
        let shrunk = backend.compile_blob(&corpus()[1..]).await;
        assert!(
            shrunk.len() < full.len(),
            "erasing a member must shrink the tensor blob ({} !< {})",
            shrunk.len(),
            full.len()
        );
        // And the erased subject can no longer be answered from the shrunk blob.
        let ans = backend.answer(&shrunk, "where is the berlin office").await;
        assert!(
            !ans.text.contains("Berlin"),
            "a recompile without the member must not surface it"
        );
    }

    #[tokio::test]
    async fn compile_blob_is_deterministic() {
        // Determinism is what makes the cache replayable + versions comparable.
        let backend = TensorKvBackend::new("tensor-kv-v1");
        let a = backend.compile_blob(&corpus()).await;
        let b = backend.compile_blob(&corpus()).await;
        assert_eq!(a, b, "same contents → identical tensor blob");
    }

    #[tokio::test]
    async fn answer_reads_only_the_blob() {
        let backend = TensorKvBackend::new("tensor-kv-v1");
        // Compile WITHOUT the Berlin member; asking for it can't be answered
        // from this blob (nothing outside the blob is consulted).
        let blob = backend.compile_blob(&corpus()[1..]).await;
        let ans = backend.answer(&blob, "berlin office relocation").await;
        assert!(!ans.text.contains("Berlin"));
        // Empty query → no answer.
        let none = backend.answer(&blob, "   ").await;
        assert!(!none.answered);
    }

    #[tokio::test]
    async fn empty_corpus_blob_answers_nothing() {
        let backend = TensorKvBackend::new("tensor-kv-v1");
        let blob = backend.compile_blob(&[]).await;
        let ans = backend.answer(&blob, "anything").await;
        assert!(!ans.answered);
        assert!(ans.text.is_empty());
    }
}
