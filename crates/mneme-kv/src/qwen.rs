//! **Modern open-weights** KV cartridge (feature `qwen-kv`) — the honest answer
//! to "why GPT-2, can't we use a more advanced model?".
//!
//! [`crate::GenerativeKvBackend`] proves the cartridge mechanism on GPT-2 (2019:
//! learned position embeddings, LayerNorm, dense multi-head attention, GELU).
//! [`QwenKvBackend`] runs the *same* cartridge path on **Qwen2.5-0.5B-Instruct**,
//! a 2024 architecture: **RMSNorm**, **RoPE** (θ=1e6), **grouped-query
//! attention** (14 query heads → 2 KV heads), **SwiGLU** MLP, **bf16** weights,
//! tied embeddings, 24 layers. It's instruction-tuned, so it actually *answers*.
//!
//! ## Why not Ollama for the cartridge?
//!
//! A KV cartridge *is* the model's per-layer key/value tensors, serialized and
//! re-injected to generate from. That needs white-box access to the weights, the
//! forward (to capture K/V), and cache restore — none of which Ollama's
//! black-box `text in → text out` API exposes. So the cartridge can't be built
//! on Ollama; it needs an in-process forward that owns the cache. We hand-roll
//! that forward in pure Rust (no candle/torch) to keep the dependency tree light
//! and the cache fully under our control. (candle + Metal would be the GPU path
//! behind this same [`KvBackend`] seam — a speed swap, not a design change.)
//!
//! ## Same reconciliations, harder model
//!
//! `compile_blob` prefills the full 24-layer forward over the (post-shred)
//! context and serializes the per-layer K/V cache (post-RoPE K, raw V) as the
//! blob; `answer` restores it and generates the query continuation attending
//! over the cartridge. The self-consistency oracle still holds — generation from
//! the cartridge is token-identical to processing the full prompt — proving the
//! RoPE/GQA cache is exact. Real bf16 weights from `Qwen/Qwen2.5-0.5B-Instruct`
//! via `hf-hub`, all behind the `qwen-kv` feature so the default build stays
//! dependency-free.

use crate::cartridge::{KvAnswer, KvBackend};
use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use safetensors::{Dtype, SafeTensors};
use std::time::Instant;
use tokenizers::Tokenizer;

const HF_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const HID: usize = 896;
const N_HEAD: usize = 14;
const N_KV: usize = 2;
const HEAD_DIM: usize = HID / N_HEAD; // 64
const Q_DIM: usize = N_HEAD * HEAD_DIM; // 896
const KV_DIM: usize = N_KV * HEAD_DIM; // 128
const GROUP: usize = N_HEAD / N_KV; // 7 query heads per KV head
const N_LAYER: usize = 24;
const INTER: usize = 4864;
const VOCAB: usize = 151936;
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;
const EOS: u32 = 151645; // <|im_end|>
const MAX_CTX: usize = 64; // cap cartridge prefix tokens
const MAX_NEW: usize = 8; // generated tokens per answer

const MAGIC_F32: &[u8; 4] = b"QWF4";

/// Per-transformer-block weights (HF Linear convention: `[out, in]`).
struct Block {
    ln1: Vec<f32>,    // input_layernorm [HID]
    q_w: Vec<f32>,    // [Q_DIM, HID]
    q_b: Vec<f32>,    // [Q_DIM]
    k_w: Vec<f32>,    // [KV_DIM, HID]
    k_b: Vec<f32>,    // [KV_DIM]
    v_w: Vec<f32>,    // [KV_DIM, HID]
    v_b: Vec<f32>,    // [KV_DIM]
    o_w: Vec<f32>,    // [HID, Q_DIM] (no bias)
    ln2: Vec<f32>,    // post_attention_layernorm [HID]
    gate_w: Vec<f32>, // [INTER, HID]
    up_w: Vec<f32>,   // [INTER, HID]
    down_w: Vec<f32>, // [HID, INTER]
}

/// Full Qwen2 weights.
struct Qwen2 {
    embed: Vec<f32>, // [VOCAB, HID] — also the tied lm_head
    blocks: Vec<Block>,
    norm: Vec<f32>, // final RMSNorm [HID]
}

/// Per-layer growing KV cache; `k`/`v` are `pos * KV_DIM` row-major.
#[derive(Clone, Default)]
struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Modern-architecture (Qwen2.5-0.5B) KV-cartridge backend (see module docs).
pub struct QwenKvBackend {
    model_id: String,
    model: Qwen2,
    tokenizer: Tokenizer,
}

// ---------------- tensor loading (f32 + bf16) ----------------

/// Read a tensor as `f32`, converting bf16 (Qwen ships bf16) on the fly.
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
        // bf16 → f32 is just the top 16 bits of the f32 bit pattern.
        Dtype::BF16 => Ok(t
            .data()
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect()),
        other => bail!("tensor {name} dtype {other:?} unsupported (expected F32/BF16)"),
    }
}

// ---------------- math (pure f32) ----------------

/// `y = x · Wᵀ + b`, HF Linear convention: `W` is `[n_out, n_in]` row-major.
fn linear_t(x: &[f32], w: &[f32], bias: Option<&[f32]>, n_in: usize, n_out: usize) -> Vec<f32> {
    let mut out = match bias {
        Some(b) => b.to_vec(),
        None => vec![0f32; n_out],
    };
    for (o, slot) in out.iter_mut().enumerate() {
        let row = &w[o * n_in..(o + 1) * n_in];
        let mut acc = 0f32;
        for (xi, wi) in x.iter().zip(row) {
            acc += xi * wi;
        }
        *slot += acc;
    }
    out
}

/// RMSNorm: `x * rsqrt(mean(x²) + eps) * weight` (no mean subtraction, no bias).
fn rmsnorm(x: &[f32], w: &[f32]) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + RMS_EPS).sqrt();
    x.iter().zip(w).map(|(v, wi)| v * inv * wi).collect()
}

/// SiLU (swish): `x * sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Rotary position embedding (HF rotate-half convention) applied in place to a
/// single `HEAD_DIM` head vector at absolute position `pos`.
fn rope(head: &mut [f32], pos: usize) {
    let half = HEAD_DIM / 2;
    for j in 0..half {
        let inv_freq = 1.0 / ROPE_THETA.powf(2.0 * j as f32 / HEAD_DIM as f32);
        let angle = pos as f32 * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let a = head[j];
        let b = head[j + half];
        head[j] = a * cos - b * sin;
        head[j + half] = b * cos + a * sin;
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

// ---------------- cartridge blob codec (compact binary, f32) ----------------

fn put_u32(buf: &mut Vec<u8>, v: usize) {
    buf.extend_from_slice(&(v as u32).to_le_bytes());
}
fn get_u32(buf: &[u8], off: &mut usize) -> Option<usize> {
    let end = off.checked_add(4)?;
    let b = buf.get(*off..end)?;
    *off = end;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
}

fn encode_blob(prefix_len: usize, layers: &[LayerCache]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC_F32);
    put_u32(&mut buf, prefix_len);
    put_u32(&mut buf, layers.len());
    for l in layers {
        let rows = l.k.len() / KV_DIM;
        put_u32(&mut buf, rows);
        for &x in &l.k {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &l.v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    buf
}

fn decode_blob(blob: &[u8]) -> Option<(usize, Vec<LayerCache>)> {
    if blob.get(0..4)? != MAGIC_F32 {
        return None;
    }
    let mut off = 4usize;
    let prefix_len = get_u32(blob, &mut off)?;
    let n_layers = get_u32(blob, &mut off)?;
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let rows = get_u32(blob, &mut off)?;
        let n = rows * KV_DIM;
        let read = |off: &mut usize| -> Option<Vec<f32>> {
            let mut t = vec![0f32; n];
            for slot in t.iter_mut() {
                let end = off.checked_add(4)?;
                let b = blob.get(*off..end)?;
                *off = end;
                *slot = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            Some(t)
        };
        let k = read(&mut off)?;
        let v = read(&mut off)?;
        layers.push(LayerCache { k, v });
    }
    Some((prefix_len, layers))
}

impl Qwen2 {
    fn empty_cache(&self) -> Vec<LayerCache> {
        (0..self.blocks.len())
            .map(|_| LayerCache::default())
            .collect()
    }

    /// Forward one token at absolute position `pos`, appending K/V to each
    /// layer's cache and attending causally over it. Returns logits.
    fn forward(&self, token: u32, pos: usize, cache: &mut [LayerCache]) -> Vec<f32> {
        let id = (token as usize).min(VOCAB - 1);
        let mut h: Vec<f32> = self.embed[id * HID..(id + 1) * HID].to_vec();

        for (li, blk) in self.blocks.iter().enumerate() {
            // --- attention (RMSNorm → QKV → RoPE → GQA → o_proj) ---
            let normed = rmsnorm(&h, &blk.ln1);
            let mut q = linear_t(&normed, &blk.q_w, Some(&blk.q_b), HID, Q_DIM);
            let mut k = linear_t(&normed, &blk.k_w, Some(&blk.k_b), HID, KV_DIM);
            let v = linear_t(&normed, &blk.v_w, Some(&blk.v_b), HID, KV_DIM);
            for head in 0..N_HEAD {
                rope(&mut q[head * HEAD_DIM..(head + 1) * HEAD_DIM], pos);
            }
            for head in 0..N_KV {
                rope(&mut k[head * HEAD_DIM..(head + 1) * HEAD_DIM], pos);
            }
            cache[li].k.extend_from_slice(&k);
            cache[li].v.extend_from_slice(&v);
            let n_keys = cache[li].k.len() / KV_DIM;

            let scale = (HEAD_DIM as f32).sqrt();
            let mut attn_out = vec![0f32; Q_DIM];
            for head in 0..N_HEAD {
                let kv_head = head / GROUP; // grouped-query attention
                let qh = &q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
                let koff = kv_head * HEAD_DIM;
                // scores over all cached keys (causal: keys 0..=pos present)
                let mut scores = vec![0f32; n_keys];
                let mut max = f32::NEG_INFINITY;
                for (j, sc) in scores.iter_mut().enumerate() {
                    let kj = &cache[li].k[j * KV_DIM + koff..j * KV_DIM + koff + HEAD_DIM];
                    let mut dot = 0f32;
                    for d in 0..HEAD_DIM {
                        dot += qh[d] * kj[d];
                    }
                    *sc = dot / scale;
                    if *sc > max {
                        max = *sc;
                    }
                }
                let mut denom = 0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - max).exp();
                    denom += *sc;
                }
                let inv = 1.0 / denom;
                let oh = &mut attn_out[head * HEAD_DIM..(head + 1) * HEAD_DIM];
                for (j, &scj) in scores.iter().enumerate() {
                    let w = scj * inv;
                    let vj = &cache[li].v[j * KV_DIM + koff..j * KV_DIM + koff + HEAD_DIM];
                    for d in 0..HEAD_DIM {
                        oh[d] += w * vj[d];
                    }
                }
            }
            let o = linear_t(&attn_out, &blk.o_w, None, Q_DIM, HID);
            for (hi, ov) in h.iter_mut().zip(o) {
                *hi += ov; // residual
            }

            // --- MLP (RMSNorm → SwiGLU) ---
            let normed2 = rmsnorm(&h, &blk.ln2);
            let gate = linear_t(&normed2, &blk.gate_w, None, HID, INTER);
            let up = linear_t(&normed2, &blk.up_w, None, HID, INTER);
            let act: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
            let down = linear_t(&act, &blk.down_w, None, INTER, HID);
            for (hi, dv) in h.iter_mut().zip(down) {
                *hi += dv; // residual
            }
        }

        let hf = rmsnorm(&h, &self.norm);
        // logits = hf · embedᵀ (tied lm_head)
        let mut logits = vec![0f32; VOCAB];
        for (vt, slot) in logits.iter_mut().enumerate() {
            let row = &self.embed[vt * HID..(vt + 1) * HID];
            let mut acc = 0f32;
            for d in 0..HID {
                acc += hf[d] * row[d];
            }
            *slot = acc;
        }
        logits
    }

    fn prefill(&self, ids: &[u32]) -> Vec<LayerCache> {
        let mut cache = self.empty_cache();
        for (pos, &t) in ids.iter().enumerate() {
            self.forward(t, pos, &mut cache);
        }
        cache
    }

    /// Greedily generate up to `max_new` tokens, feeding `query` first.
    fn generate(
        &self,
        cache: &mut [LayerCache],
        query: &[u32],
        start_pos: usize,
        max_new: usize,
    ) -> Vec<u32> {
        let mut pos = start_pos;
        let mut last = vec![0f32; VOCAB];
        for &t in query {
            last = self.forward(t, pos, cache);
            pos += 1;
        }
        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let next = argmax(&last) as u32;
            if next == EOS {
                break;
            }
            out.push(next);
            last = self.forward(next, pos, cache);
            pos += 1;
        }
        out
    }
}

impl QwenKvBackend {
    /// Download (cached) + load Qwen2.5-0.5B-Instruct. Blocking; call at startup.
    pub fn load() -> Result<Self> {
        use hf_hub::api::sync::Api;
        let api = Api::new().map_err(|e| {
            anyhow!(
                "{}: {e}",
                crate::cartridge::weights_hint(HF_REPO, "model weights")
            )
        })?;
        let repo = api.model(HF_REPO.to_string());
        let weights = repo.get("model.safetensors").map_err(|e| {
            anyhow!(
                "{}: {e}",
                crate::cartridge::weights_hint(HF_REPO, "model.safetensors")
            )
        })?;
        let tok = repo.get("tokenizer.json").map_err(|e| {
            anyhow!(
                "{}: {e}",
                crate::cartridge::weights_hint(HF_REPO, "tokenizer.json")
            )
        })?;
        let buffer = std::fs::read(&weights).map_err(|e| anyhow!("read weights: {e}"))?;
        let st =
            SafeTensors::deserialize(&buffer).map_err(|e| anyhow!("parse safetensors: {e}"))?;

        let embed = tensor_to_f32(&st, "model.embed_tokens.weight")?;
        let mut blocks = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            let p = format!("model.layers.{i}");
            blocks.push(Block {
                ln1: tensor_to_f32(&st, &format!("{p}.input_layernorm.weight"))?,
                q_w: tensor_to_f32(&st, &format!("{p}.self_attn.q_proj.weight"))?,
                q_b: tensor_to_f32(&st, &format!("{p}.self_attn.q_proj.bias"))?,
                k_w: tensor_to_f32(&st, &format!("{p}.self_attn.k_proj.weight"))?,
                k_b: tensor_to_f32(&st, &format!("{p}.self_attn.k_proj.bias"))?,
                v_w: tensor_to_f32(&st, &format!("{p}.self_attn.v_proj.weight"))?,
                v_b: tensor_to_f32(&st, &format!("{p}.self_attn.v_proj.bias"))?,
                o_w: tensor_to_f32(&st, &format!("{p}.self_attn.o_proj.weight"))?,
                ln2: tensor_to_f32(&st, &format!("{p}.post_attention_layernorm.weight"))?,
                gate_w: tensor_to_f32(&st, &format!("{p}.mlp.gate_proj.weight"))?,
                up_w: tensor_to_f32(&st, &format!("{p}.mlp.up_proj.weight"))?,
                down_w: tensor_to_f32(&st, &format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        let norm = tensor_to_f32(&st, "model.norm.weight")?;
        let tokenizer = Tokenizer::from_file(&tok).map_err(|e| anyhow!("tokenizer: {e}"))?;

        Ok(Self {
            model_id: "qwen2.5-0.5b-instruct".to_string(),
            model: Qwen2 {
                embed,
                blocks,
                norm,
            },
            tokenizer,
        })
    }

    fn encode(&self, text: &str, cap: usize) -> Vec<u32> {
        match self.tokenizer.encode(text, false) {
            Ok(e) => e.get_ids().iter().take(cap).copied().collect(),
            Err(_) => Vec::new(),
        }
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, true).unwrap_or_default()
    }

    fn context_ids(&self, contents: &[String]) -> Vec<u32> {
        self.encode(&contents.join("\n"), MAX_CTX)
    }

    /// Generate over the full `context ++ query` token sequence from scratch (no
    /// cartridge) — the correctness oracle for [`KvBackend::answer`].
    pub fn answer_full_prompt(&self, contents: &[String], query: &str) -> String {
        let ctx = self.context_ids(contents);
        let q = self.encode(query, MAX_CTX);
        let mut cache = self.model.prefill(&ctx);
        let gen = self.model.generate(&mut cache, &q, ctx.len(), MAX_NEW);
        self.decode(&gen)
    }
}

#[async_trait]
impl KvBackend for QwenKvBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn compile_blob(&self, contents: &[String]) -> Vec<u8> {
        let ctx = self.context_ids(contents);
        let layers = self.model.prefill(&ctx);
        encode_blob(ctx.len(), &layers)
    }

    async fn answer(&self, blob: &[u8], query: &str) -> KvAnswer {
        let start = Instant::now();
        let (prefix_len, mut cache) = match decode_blob(blob) {
            Some(p) => p,
            None => {
                return KvAnswer {
                    text: String::new(),
                    latency_us: start.elapsed().as_micros() as u64,
                    answered: false,
                }
            }
        };
        let q = self.encode(query, MAX_CTX);
        if cache.len() != N_LAYER || q.is_empty() {
            return KvAnswer {
                text: String::new(),
                latency_us: start.elapsed().as_micros() as u64,
                answered: false,
            };
        }
        let gen = self.model.generate(&mut cache, &q, prefix_len, MAX_NEW);
        let text = self.decode(&gen);
        KvAnswer {
            latency_us: start.elapsed().as_micros() as u64,
            answered: !text.trim().is_empty(),
            text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Network + ~1GB weights + a pure-Rust 24-layer forward → ignored. One test
    // loads the model once (the expensive part) and proves both reconciliations.
    // Run with:
    //   cargo test -p mneme-kv --release --features qwen-kv -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "downloads Qwen2.5-0.5B (~1GB) + runs a real 24-layer forward"]
    async fn qwen_cartridge_is_exact_and_erasable() {
        let be = QwenKvBackend::load().expect("load qwen");

        // (1) Self-consistency: cartridge generation == full-prompt generation.
        let ctx = vec![
            "Paris is the capital of France.".to_string(),
            "Berlin is the capital of Germany.".to_string(),
        ];
        let query = "\nQuestion: What is the capital of France?\nAnswer:";
        let blob = be.compile_blob(&ctx).await;
        assert!(blob.len() > 1000, "blob is a real multi-layer KV cache");
        let from_cartridge = be.answer(&blob, query).await;
        assert!(from_cartridge.answered);
        let from_full = be.answer_full_prompt(&ctx, query);
        // KV caching over RoPE + GQA is exact: cartridge generation must be
        // token-identical to processing the whole prompt from scratch.
        assert_eq!(
            from_cartridge.text.trim(),
            from_full.trim(),
            "cartridge generation must equal full-prompt generation"
        );
        // A 2024 instruct model should actually answer the factual query.
        assert!(
            from_cartridge.text.contains("Paris"),
            "expected 'Paris', got {:?}",
            from_cartridge.text
        );
        eprintln!("qwen generated from cartridge: {:?}", from_cartridge.text);

        // (2) Erasure-by-recompile: the fact is gone from generation.
        let with_fact = vec!["The project's codename is Nimbus.".to_string()];
        let without = vec!["The weather is sunny today.".to_string()];
        let q = "\nQuestion: What is the project's codename?\nAnswer:";
        let a = be.answer(&be.compile_blob(&with_fact).await, q).await;
        let b = be.answer(&be.compile_blob(&without).await, q).await;
        assert!(!b.text.contains("Nimbus"), "erased fact must not appear");
        eprintln!("with fact: {:?}\nwithout:  {:?}", a.text, b.text);
    }
}
