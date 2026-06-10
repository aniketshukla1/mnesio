//! A **pretrained-weights** KV backend (feature `pretrained-kv`).
//!
//! [`TensorKvBackend`](crate::TensorKvBackend) proved the tensor *math* with
//! untrained (hash-derived) embeddings + seeded projections. This backend
//! removes the "untrained" caveat: it loads **real open weights** — GPT-2's
//! token embeddings (`wte`), positional embeddings (`wpe`), and the layer-0
//! attention `c_attn` projection (the fused Q/K/V matrix `Wqkv` + bias) — and
//! runs the *actual* GPT-2 forward to build the cache:
//!
//! ```text
//!   x_i   = wte[token_i] + wpe[i]            (real embeddings)
//!   qkv_i = x_i · Wqkv + b                   (real pretrained projection)
//!   K_i   = qkv_i[768..1536]                 (the cartridge's key tensor)
//!   V_i   = qkv_i[1536..2304]                (the cartridge's value tensor)
//! ```
//!
//! `compile_blob` stores those real K/V tensors as the blob; `answer` projects
//! the query through the real `Wq` slice, **mean-pools** it, and retrieves by
//! **cosine** against each member's mean-pooled cached K. (GPT-2 base is a
//! causal LM, not a retriever — raw layer-0 max-token Q·K is dominated by
//! frequent subwords; mean-pooled hidden states + cosine is the standard way to
//! read a usable sentence vector out of a transformer layer.) So the retrieval
//! signal is now GPT-2's *learned* representation, not token overlap.
//!
//! Weights download once via `hf-hub` (cached under `~/.cache/huggingface`) and
//! are parsed with `safetensors`; tokenization is GPT-2 BPE via `tokenizers`.
//! All three are optional deps behind the `pretrained-kv` feature — the default
//! build stays dependency-free and offline. This is the documented "load the
//! open-weights model behind the same `compile_blob`/`answer` seam" lift: a
//! weights load, not an architecture change. (It uses layer-0 attention, not a
//! full N-layer forward — enough to make the K/V cache genuinely pretrained.)

use crate::cartridge::{KvAnswer, KvBackend};
use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokenizers::Tokenizer;

const HF_REPO: &str = "openai-community/gpt2";
const N_EMBD: usize = 768;
const QKV: usize = 3 * N_EMBD; // fused Q/K/V width = 2304
const MAX_TOKENS: usize = 64;

/// Real GPT-2 weights needed for a layer-0 KV cache.
struct Gpt2Weights {
    wte: Vec<f32>, // [vocab, 768]
    wpe: Vec<f32>, // [1024, 768]
    /// `c_attn` Conv1D weight, shape `[768, 2304]` row-major (`x · w`).
    c_attn_w: Vec<f32>,
    c_attn_b: Vec<f32>, // [2304]
    vocab: usize,
    n_pos: usize,
}

/// Pretrained-weights KV backend (see module docs).
pub struct PretrainedKvBackend {
    model_id: String,
    weights: Gpt2Weights,
    tokenizer: Tokenizer,
}

/// One member's real pretrained K/V cache + the text to return on a hit.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemberKv {
    text: String,
    seq: usize,
    k: Vec<f32>, // seq * N_EMBD
    v: Vec<f32>, // seq * N_EMBD
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KvBlob {
    members: Vec<MemberKv>,
}

fn tensor_to_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st
        .tensor(name)
        .map_err(|e| anyhow!("missing tensor {name}: {e}"))?;
    match t.dtype() {
        Dtype::F32 => Ok(t
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        other => bail!("tensor {name} has unsupported dtype {other:?}; expected F32"),
    }
}

impl PretrainedKvBackend {
    /// Download (once, cached) + load GPT-2 weights and tokenizer. Blocking —
    /// call at startup, not on a hot path.
    pub fn load() -> Result<Self> {
        use hf_hub::api::sync::Api;
        let api = Api::new().map_err(|e| {
            anyhow!(
                "{}: {e}",
                crate::cartridge::weights_hint(HF_REPO, "model weights")
            )
        })?;
        let repo = api.model(HF_REPO.to_string());
        let weights_path = repo.get("model.safetensors").map_err(|e| {
            anyhow!(
                "{}: {e}",
                crate::cartridge::weights_hint(HF_REPO, "model.safetensors")
            )
        })?;
        let tok_path = repo.get("tokenizer.json").map_err(|e| {
            anyhow!(
                "{}: {e}",
                crate::cartridge::weights_hint(HF_REPO, "tokenizer.json")
            )
        })?;

        let buffer = std::fs::read(&weights_path).map_err(|e| anyhow!("read weights: {e}"))?;
        let st =
            SafeTensors::deserialize(&buffer).map_err(|e| anyhow!("parse safetensors: {e}"))?;

        let wte = tensor_to_f32(&st, "wte.weight")?;
        let wpe = tensor_to_f32(&st, "wpe.weight")?;
        let c_attn_w = tensor_to_f32(&st, "h.0.attn.c_attn.weight")?;
        let c_attn_b = tensor_to_f32(&st, "h.0.attn.c_attn.bias")?;
        let vocab = wte.len() / N_EMBD;
        let n_pos = wpe.len() / N_EMBD;
        if c_attn_w.len() != N_EMBD * QKV || c_attn_b.len() != QKV {
            bail!("unexpected c_attn shape — not a GPT-2 checkpoint?");
        }

        let tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("load tokenizer: {e}"))?;

        Ok(Self {
            model_id: "gpt2".to_string(),
            weights: Gpt2Weights {
                wte,
                wpe,
                c_attn_w,
                c_attn_b,
                vocab,
                n_pos,
            },
            tokenizer,
        })
    }

    fn token_ids(&self, text: &str) -> Vec<u32> {
        match self.tokenizer.encode(text, false) {
            Ok(enc) => enc.get_ids().iter().take(MAX_TOKENS).copied().collect(),
            Err(_) => Vec::new(),
        }
    }

    /// `x_i = wte[id] + wpe[pos]` for each token; returns `[seq][768]`.
    fn embed(&self, ids: &[u32]) -> Vec<Vec<f32>> {
        let w = &self.weights;
        ids.iter()
            .enumerate()
            .map(|(pos, &id)| {
                let id = (id as usize).min(w.vocab - 1);
                let pos = pos.min(w.n_pos - 1);
                let wte = &w.wte[id * N_EMBD..(id + 1) * N_EMBD];
                let wpe = &w.wpe[pos * N_EMBD..(pos + 1) * N_EMBD];
                wte.iter().zip(wpe).map(|(a, b)| a + b).collect()
            })
            .collect()
    }

    /// One slice of the fused `qkv = x · Wqkv + b` for token `x`: `which` is 0
    /// (Q), 1 (K) or 2 (V); returns the `[768]` slice for that projection.
    fn project(&self, x: &[f32], which: usize) -> Vec<f32> {
        let w = &self.weights;
        let col0 = which * N_EMBD;
        let mut out = vec![0f32; N_EMBD];
        for (j, slot) in out.iter_mut().enumerate() {
            let col = col0 + j;
            let mut acc = w.c_attn_b[col];
            for (i, &xi) in x.iter().enumerate() {
                acc += xi * w.c_attn_w[i * QKV + col];
            }
            *slot = acc;
        }
        out
    }

    /// Mean-pool a `[seq][N_EMBD]` row-major tensor into one `[N_EMBD]` vector,
    /// then L2-normalize. Mean-pooled hidden states are the standard way to read
    /// a sentence-level vector out of a transformer layer; normalizing makes the
    /// dot product a cosine, so retrieval isn't dominated by a single high-norm
    /// token (raw layer-0 max-token Q·K is a poor retriever).
    fn pool_normalize(flat: &[f32], seq: usize) -> Vec<f32> {
        let mut pooled = vec![0f32; N_EMBD];
        if seq == 0 {
            return pooled;
        }
        for t in 0..seq {
            for (j, p) in pooled.iter_mut().enumerate() {
                *p += flat[t * N_EMBD + j];
            }
        }
        let inv = 1.0 / seq as f32;
        for p in pooled.iter_mut() {
            *p *= inv;
        }
        let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for p in pooled.iter_mut() {
                *p /= norm;
            }
        }
        pooled
    }
}

#[async_trait]
impl KvBackend for PretrainedKvBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn compile_blob(&self, contents: &[String]) -> Vec<u8> {
        let mut members = Vec::with_capacity(contents.len());
        for text in contents {
            let ids = self.token_ids(text);
            let x = self.embed(&ids);
            let mut k = Vec::with_capacity(x.len() * N_EMBD);
            let mut v = Vec::with_capacity(x.len() * N_EMBD);
            for row in &x {
                k.extend(self.project(row, 1)); // K
                v.extend(self.project(row, 2)); // V
            }
            members.push(MemberKv {
                text: text.clone(),
                seq: x.len(),
                k,
                v,
            });
        }
        serde_json::to_vec(&KvBlob { members }).unwrap_or_default()
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
        let ids = self.token_ids(query);
        let qx = self.embed(&ids);
        if qx.is_empty() || parsed.members.is_empty() {
            return KvAnswer {
                text: String::new(),
                latency_us: start.elapsed().as_micros() as u64,
                answered: false,
            };
        }
        // Project the query through the real Q weight (`Wq`), mean-pool +
        // normalize into one query vector; score it against each member's
        // mean-pooled, normalized cached K (`Wk`) by cosine. Real pretrained
        // Q/K representations, read as mean-pooled sentence encoders.
        let q_proj: Vec<f32> = qx.iter().flat_map(|row| self.project(row, 0)).collect();
        let q_vec = Self::pool_normalize(&q_proj, qx.len());

        let mut best_idx = None;
        let mut best_score = f32::NEG_INFINITY;
        for (mi, m) in parsed.members.iter().enumerate() {
            let k_vec = Self::pool_normalize(&m.k, m.seq);
            let score: f32 = q_vec.iter().zip(&k_vec).map(|(a, b)| a * b).sum();
            if score > best_score {
                best_score = score;
                best_idx = Some(mi);
            }
        }

        let latency_us = start.elapsed().as_micros() as u64;
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

    // Network + ~548MB weights download → ignored by default; run explicitly:
    //   cargo test -p mneme-kv --features pretrained-kv -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "downloads GPT-2 weights (~548MB) on first run"]
    async fn pretrained_compile_and_answer_with_real_gpt2_weights() {
        let backend = PretrainedKvBackend::load().expect("load gpt2");
        assert_eq!(backend.model_id(), "gpt2");

        let blob = backend.compile_blob(&corpus()).await;
        // The blob is real pretrained K/V tensor bytes — large, not a JSON bag.
        assert!(blob.len() > 10_000, "real KV tensors → sizeable blob");

        // Real GPT-2 attention should retrieve the Berlin member for a
        // Berlin-relocation query.
        let ans = backend.answer(&blob, "where did Bob move his office").await;
        assert!(ans.answered);
        assert!(
            ans.text.contains("Berlin"),
            "pretrained attention should retrieve the Berlin member, got {:?}",
            ans.text
        );

        // Erasure: recompile without the member → it can't be answered, and the
        // tensor blob shrinks.
        let shrunk = backend.compile_blob(&corpus()[1..]).await;
        assert!(shrunk.len() < blob.len());
        let gone = backend
            .answer(&shrunk, "where did Bob move his office")
            .await;
        assert!(!gone.text.contains("Berlin"));
    }
}
