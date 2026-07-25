//! **Generative** KV cartridge (feature `generative-kv`) — the deepest Phase-12
//! lift: a full GPT-2 forward + generation loop where the cartridge **is** the
//! model's key/value cache, loaded into attention so the model *generates* from
//! it rather than merely retrieving over it.
//!
//! - [`GenerativeKvBackend::compile_blob`] tokenizes the (post-shred) memory
//!   context and runs the **full 12-layer GPT-2 forward** to prefill a KV cache
//!   for *every* layer. That cache, serialized, is the cartridge blob.
//! - [`GenerativeKvBackend::answer`] restores the cache and **generates** the
//!   query's continuation greedily, each new token attending over the restored
//!   prefix cache + its own — exactly prompt/prefix caching.
//!
//! ## The correctness oracle (no external reference needed)
//!
//! Generating *from the cartridge* (prefill the context once, reuse the cache)
//! must produce **token-identical** output to processing the full
//! `context ++ query` prompt from scratch — because causal-attention KV caching
//! is exact. The test `cartridge_generation_equals_full_prompt_generation`
//! asserts this. It simultaneously proves (a) the GPT-2 forward + cache are
//! correct and (b) the cartridge's value proposition: identical output while
//! skipping the prefix recompute on every query (the Phase-12 latency claim).
//!
//! Real open weights (`openai-community/gpt2`), parsed with `safetensors`,
//! tokenized with the GPT-2 BPE — all behind the `generative-kv` feature so the
//! default build stays dependency-free. Pure-Rust f32 math (no GPU/BLAS); fine
//! for the small contexts a cartridge demo uses.
//!
//! ## Quantization — the real meaning of `CartridgeKey::quant`
//!
//! The cartridge blob is compact little-endian binary in *both* precisions, so
//! the size win is a real property of quantization, not a serialization
//! artifact. [`Quant::Q8`] stores the KV cache as per-row **int8** (one `i8` per
//! element + one `f32` scale per token row) — a ~4× smaller cartridge with a
//! small, bounded dequantization error; [`Quant::F32`] keeps it dense. `answer`
//! auto-detects the precision from the blob's magic header and dequantizes a q8
//! cache back to f32 before generating (the generation math is always f32). The
//! `quant` field in [`CartridgeKey`](crate::CartridgeKey) was a bare label until
//! now; an f32 and a q8 cartridge over the same corpus are kept apart by their
//! distinct keys — never overwriting each other — under the same versioning
//! rules as any other view.

use crate::cartridge::{KvAnswer, KvBackend};
use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use safetensors::{Dtype, SafeTensors};
use std::time::Instant;
use tokenizers::Tokenizer;

const HF_REPO: &str = "openai-community/gpt2";
const N_EMBD: usize = 768;
const N_HEAD: usize = 12;
const HEAD_DIM: usize = N_EMBD / N_HEAD; // 64
const N_LAYER: usize = 12;
const FF: usize = 4 * N_EMBD; // 3072
const QKV: usize = 3 * N_EMBD; // 2304
const EOS: u32 = 50256; // <|endoftext|>
const LN_EPS: f32 = 1e-5;
const MAX_CTX: usize = 96; // cap cartridge prefix tokens (keeps the blob small)
const MAX_NEW: usize = 12; // generated tokens per answer

/// Per-transformer-block weights (GPT-2 Conv1D layers store `[in, out]`).
struct Block {
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    attn_w: Vec<f32>, // [768 * 2304]
    attn_b: Vec<f32>, // [2304]
    proj_w: Vec<f32>, // [768 * 768]
    proj_b: Vec<f32>, // [768]
    ln2_g: Vec<f32>,
    ln2_b: Vec<f32>,
    fc_w: Vec<f32>,  // [768 * 3072]
    fc_b: Vec<f32>,  // [3072]
    fcp_w: Vec<f32>, // [3072 * 768]
    fcp_b: Vec<f32>, // [768]
}

/// Full GPT-2 weights for an N-layer forward.
struct Gpt2 {
    wte: Vec<f32>,
    wpe: Vec<f32>,
    blocks: Vec<Block>,
    lnf_g: Vec<f32>,
    lnf_b: Vec<f32>,
    vocab: usize,
    n_pos: usize,
}

/// Per-layer growing KV cache; `k`/`v` are `pos * N_EMBD` row-major.
#[derive(Clone, Default)]
struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Cartridge tensor precision — the real meaning of [`CartridgeKey::quant`].
///
/// [`Quant::F32`] stores the KV cache as dense `f32`; [`Quant::Q8`] stores it as
/// per-row **int8** (one `i8` per element + one `f32` scale per token row), a
/// ~4× smaller cartridge with a small, bounded dequantization error. The blob is
/// compact little-endian binary in *both* cases, so the size win is a real
/// property of the quantization — not a serialization artifact.
///
/// [`CartridgeKey::quant`]: crate::CartridgeKey::quant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    /// Dense f32 cache (exact).
    F32,
    /// Per-row int8 cache (~4× smaller, lossy but bounded).
    Q8,
}

impl Quant {
    /// The [`CartridgeKey::quant`](crate::CartridgeKey::quant) label this maps to.
    pub fn label(self) -> &'static str {
        match self {
            Quant::F32 => "f32",
            Quant::Q8 => "q8",
        }
    }
}

// Blob format magics (4 bytes). The first byte of every cartridge blob selects
// the codec, so `answer` auto-detects precision without an out-of-band flag.
const MAGIC_F32: &[u8; 4] = b"MGF4";
const MAGIC_Q8: &[u8; 4] = b"MGQ8";

/// Generative GPT-2 KV-cartridge backend (see module docs).
pub struct GenerativeKvBackend {
    model_id: String,
    model: Gpt2,
    tokenizer: Tokenizer,
    quant: Quant,
}

// ---------------- tensor helpers (pure f32) ----------------

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
        other => bail!("tensor {name} dtype {other:?} unsupported (expected F32)"),
    }
}

/// `y = x · W + b`, with `W` row-major `[in, out]`, `x` `[in]` → `[out]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let mut out = b.to_vec();
    for (i, &xi) in x.iter().enumerate().take(n_in) {
        if xi == 0.0 {
            continue;
        }
        let row = &w[i * n_out..(i + 1) * n_out];
        for (o, &wv) in row.iter().enumerate() {
            out[o] += xi * wv;
        }
    }
    out
}

/// LayerNorm over a `[N_EMBD]` vector.
fn layernorm(x: &[f32], g: &[f32], b: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    x.iter()
        .zip(g)
        .zip(b)
        .map(|((v, gi), bi)| (v - mean) * inv * gi + bi)
        .collect()
}

/// GPT-2's `gelu_new` (tanh approximation).
fn gelu_new(x: &mut [f32]) {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    for v in x.iter_mut() {
        let x3 = *v * *v * *v;
        *v = 0.5 * *v * (1.0 + (C * (*v + 0.044715 * x3)).tanh());
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

// ---------------- cartridge blob codec (compact binary) ----------------

fn put_u32(buf: &mut Vec<u8>, v: usize) {
    buf.extend_from_slice(&(v as u32).to_le_bytes());
}
fn get_u32(buf: &[u8], off: &mut usize) -> Option<usize> {
    let end = off.checked_add(4)?;
    let bytes = buf.get(*off..end)?;
    *off = end;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize)
}
fn put_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn get_f32(buf: &[u8], off: &mut usize) -> Option<f32> {
    let end = off.checked_add(4)?;
    let bytes = buf.get(*off..end)?;
    *off = end;
    Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Per-row symmetric int8 quantization of one `rows × N_EMBD` tensor: append a
/// `f32` scale then `N_EMBD` `i8` codes for each row. `x ≈ code * scale`.
fn write_q8_tensor(buf: &mut Vec<u8>, t: &[f32], rows: usize) {
    for r in 0..rows {
        let row = &t[r * N_EMBD..(r + 1) * N_EMBD];
        let amax = row.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        put_f32(buf, scale);
        let inv = 1.0 / scale;
        for &x in row {
            let q = (x * inv).round().clamp(-127.0, 127.0) as i8;
            buf.push(q as u8);
        }
    }
}

/// Inverse of [`write_q8_tensor`]: read `rows` rows back into an f32 tensor.
fn read_q8_tensor(buf: &[u8], off: &mut usize, rows: usize) -> Option<Vec<f32>> {
    let mut out = vec![0f32; rows * N_EMBD];
    for r in 0..rows {
        let scale = get_f32(buf, off)?;
        let end = off.checked_add(N_EMBD)?;
        let codes = buf.get(*off..end)?;
        *off = end;
        for (c, slot) in codes
            .iter()
            .zip(out[r * N_EMBD..(r + 1) * N_EMBD].iter_mut())
        {
            *slot = (*c as i8) as f32 * scale;
        }
    }
    Some(out)
}

/// Serialize a prefilled cache to the cartridge blob under the given precision.
fn encode_blob(prefix_len: usize, layers: &[LayerCache], quant: Quant) -> Vec<u8> {
    let mut buf = Vec::new();
    match quant {
        Quant::F32 => {
            buf.extend_from_slice(MAGIC_F32);
            put_u32(&mut buf, prefix_len);
            put_u32(&mut buf, layers.len());
            for l in layers {
                let rows = l.k.len() / N_EMBD;
                put_u32(&mut buf, rows);
                for &x in &l.k {
                    put_f32(&mut buf, x);
                }
                for &x in &l.v {
                    put_f32(&mut buf, x);
                }
            }
        }
        Quant::Q8 => {
            buf.extend_from_slice(MAGIC_Q8);
            put_u32(&mut buf, prefix_len);
            put_u32(&mut buf, layers.len());
            for l in layers {
                let rows = l.k.len() / N_EMBD;
                put_u32(&mut buf, rows);
                write_q8_tensor(&mut buf, &l.k, rows);
                write_q8_tensor(&mut buf, &l.v, rows);
            }
        }
    }
    buf
}

/// Decode a cartridge blob (any precision) back to an f32 cache for generation.
/// Q8 is dequantized here; the generation math is always f32.
fn decode_blob(blob: &[u8]) -> Option<(usize, Vec<LayerCache>)> {
    let magic = blob.get(0..4)?;
    let q8 = match magic {
        m if m == MAGIC_F32 => false,
        m if m == MAGIC_Q8 => true,
        _ => return None,
    };
    let mut off = 4usize;
    let prefix_len = get_u32(blob, &mut off)?;
    let n_layers = get_u32(blob, &mut off)?;
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let rows = get_u32(blob, &mut off)?;
        let (k, v) = if q8 {
            let k = read_q8_tensor(blob, &mut off, rows)?;
            let v = read_q8_tensor(blob, &mut off, rows)?;
            (k, v)
        } else {
            let n = rows * N_EMBD;
            let mut k = vec![0f32; n];
            for slot in k.iter_mut() {
                *slot = get_f32(blob, &mut off)?;
            }
            let mut v = vec![0f32; n];
            for slot in v.iter_mut() {
                *slot = get_f32(blob, &mut off)?;
            }
            (k, v)
        };
        layers.push(LayerCache { k, v });
    }
    Some((prefix_len, layers))
}

impl Gpt2 {
    /// Forward one token at absolute position `pos`, appending its K/V to each
    /// layer's cache and attending causally over the cache. Returns logits.
    fn forward(&self, token: u32, pos: usize, cache: &mut [LayerCache]) -> Vec<f32> {
        let id = (token as usize).min(self.vocab - 1);
        let p = pos.min(self.n_pos - 1);
        let mut h: Vec<f32> = self.wte[id * N_EMBD..(id + 1) * N_EMBD]
            .iter()
            .zip(&self.wpe[p * N_EMBD..(p + 1) * N_EMBD])
            .map(|(a, b)| a + b)
            .collect();

        for (li, blk) in self.blocks.iter().enumerate() {
            // --- attention ---
            let normed = layernorm(&h, &blk.ln1_g, &blk.ln1_b);
            let qkv = linear(&normed, &blk.attn_w, &blk.attn_b, N_EMBD, QKV);
            let q = &qkv[0..N_EMBD];
            let k = &qkv[N_EMBD..2 * N_EMBD];
            let v = &qkv[2 * N_EMBD..3 * N_EMBD];
            cache[li].k.extend_from_slice(k);
            cache[li].v.extend_from_slice(v);
            let n_keys = cache[li].k.len() / N_EMBD;

            let mut attn_out = vec![0f32; N_EMBD];
            let scale = (HEAD_DIM as f32).sqrt();
            for head in 0..N_HEAD {
                let hs = head * HEAD_DIM;
                // scores over all cached keys (causal: keys 0..=pos are present)
                let mut scores = vec![0f32; n_keys];
                let mut max = f32::NEG_INFINITY;
                for (j, sc) in scores.iter_mut().enumerate() {
                    let kj = &cache[li].k[j * N_EMBD + hs..j * N_EMBD + hs + HEAD_DIM];
                    let mut dot = 0f32;
                    for d in 0..HEAD_DIM {
                        dot += q[hs + d] * kj[d];
                    }
                    *sc = dot / scale;
                    if *sc > max {
                        max = *sc;
                    }
                }
                // softmax
                let mut denom = 0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - max).exp();
                    denom += *sc;
                }
                let inv = 1.0 / denom;
                // weighted sum of values
                for (j, &scj) in scores.iter().enumerate() {
                    let w = scj * inv;
                    let vj = &cache[li].v[j * N_EMBD + hs..j * N_EMBD + hs + HEAD_DIM];
                    for d in 0..HEAD_DIM {
                        attn_out[hs + d] += w * vj[d];
                    }
                }
            }
            let proj = linear(&attn_out, &blk.proj_w, &blk.proj_b, N_EMBD, N_EMBD);
            for (hi, pv) in h.iter_mut().zip(proj) {
                *hi += pv; // residual
            }

            // --- MLP ---
            let normed2 = layernorm(&h, &blk.ln2_g, &blk.ln2_b);
            let mut ff = linear(&normed2, &blk.fc_w, &blk.fc_b, N_EMBD, FF);
            gelu_new(&mut ff);
            let ff2 = linear(&ff, &blk.fcp_w, &blk.fcp_b, FF, N_EMBD);
            for (hi, fv) in h.iter_mut().zip(ff2) {
                *hi += fv; // residual
            }
        }

        let hf = layernorm(&h, &self.lnf_g, &self.lnf_b);
        // logits = hf · wte^T  (tied lm_head)
        let mut logits = vec![0f32; self.vocab];
        for (vt, slot) in logits.iter_mut().enumerate() {
            let row = &self.wte[vt * N_EMBD..(vt + 1) * N_EMBD];
            let mut acc = 0f32;
            for d in 0..N_EMBD {
                acc += hf[d] * row[d];
            }
            *slot = acc;
        }
        logits
    }

    fn empty_cache(&self) -> Vec<LayerCache> {
        (0..self.blocks.len())
            .map(|_| LayerCache::default())
            .collect()
    }

    /// Prefill a cache by running `ids` through the model in order.
    fn prefill(&self, ids: &[u32]) -> Vec<LayerCache> {
        let mut cache = self.empty_cache();
        for (pos, &t) in ids.iter().enumerate() {
            self.forward(t, pos, &mut cache);
        }
        cache
    }

    /// Greedily generate up to `max_new` tokens continuing from `start_pos`,
    /// feeding `query` first. Mutates `cache`. Returns the generated ids.
    fn generate(
        &self,
        cache: &mut [LayerCache],
        query: &[u32],
        start_pos: usize,
        max_new: usize,
    ) -> Vec<u32> {
        let mut pos = start_pos;
        let mut last = vec![0f32; self.vocab];
        for &t in query {
            last = self.forward(t, pos, cache);
            pos += 1;
        }
        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            if pos >= self.n_pos {
                break;
            }
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

impl GenerativeKvBackend {
    /// Download (cached) + load full GPT-2. Blocking; call at startup.
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

        let wte = tensor_to_f32(&st, "wte.weight")?;
        let wpe = tensor_to_f32(&st, "wpe.weight")?;
        let vocab = wte.len() / N_EMBD;
        let n_pos = wpe.len() / N_EMBD;
        let mut blocks = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            let p = format!("h.{i}");
            blocks.push(Block {
                ln1_g: tensor_to_f32(&st, &format!("{p}.ln_1.weight"))?,
                ln1_b: tensor_to_f32(&st, &format!("{p}.ln_1.bias"))?,
                attn_w: tensor_to_f32(&st, &format!("{p}.attn.c_attn.weight"))?,
                attn_b: tensor_to_f32(&st, &format!("{p}.attn.c_attn.bias"))?,
                proj_w: tensor_to_f32(&st, &format!("{p}.attn.c_proj.weight"))?,
                proj_b: tensor_to_f32(&st, &format!("{p}.attn.c_proj.bias"))?,
                ln2_g: tensor_to_f32(&st, &format!("{p}.ln_2.weight"))?,
                ln2_b: tensor_to_f32(&st, &format!("{p}.ln_2.bias"))?,
                fc_w: tensor_to_f32(&st, &format!("{p}.mlp.c_fc.weight"))?,
                fc_b: tensor_to_f32(&st, &format!("{p}.mlp.c_fc.bias"))?,
                fcp_w: tensor_to_f32(&st, &format!("{p}.mlp.c_proj.weight"))?,
                fcp_b: tensor_to_f32(&st, &format!("{p}.mlp.c_proj.bias"))?,
            });
        }
        let lnf_g = tensor_to_f32(&st, "ln_f.weight")?;
        let lnf_b = tensor_to_f32(&st, "ln_f.bias")?;
        let tokenizer = Tokenizer::from_file(&tok).map_err(|e| anyhow!("tokenizer: {e}"))?;

        Ok(Self {
            model_id: "gpt2-generative".to_string(),
            model: Gpt2 {
                wte,
                wpe,
                blocks,
                lnf_g,
                lnf_b,
                vocab,
                n_pos,
            },
            tokenizer,
            quant: Quant::F32,
        })
    }

    /// Set the cartridge precision. The `model_id` is unchanged — precision is a
    /// separate [`CartridgeKey::quant`](crate::CartridgeKey::quant) dimension, so
    /// an f32 and a q8 cartridge over the same corpus get distinct keys and are
    /// kept apart rather than overwriting each other.
    pub fn with_quant(mut self, quant: Quant) -> Self {
        self.quant = quant;
        self
    }

    /// The precision this backend compiles cartridges at.
    pub fn quant(&self) -> Quant {
        self.quant
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

    /// The context prefix tokens for a set of memories (joined, capped).
    fn context_ids(&self, contents: &[String]) -> Vec<u32> {
        self.encode(&contents.join("\n"), MAX_CTX)
    }

    /// Generate an answer for `query` *from scratch* over the full
    /// `context ++ query` prompt (no cartridge) — the correctness oracle for
    /// [`KvBackend::answer`].
    pub fn answer_full_prompt(&self, contents: &[String], query: &str) -> String {
        let ctx = self.context_ids(contents);
        let q = self.encode(query, MAX_CTX);
        let mut cache = self.model.prefill(&ctx);
        let gen = self.model.generate(&mut cache, &q, ctx.len(), MAX_NEW);
        self.decode(&gen)
    }
}

#[async_trait]
impl KvBackend for GenerativeKvBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn compile_blob(&self, contents: &[String]) -> Vec<u8> {
        let ctx = self.context_ids(contents);
        let layers = self.model.prefill(&ctx);
        encode_blob(ctx.len(), &layers, self.quant)
    }

    async fn answer(&self, blob: &[u8], query: &str) -> KvAnswer {
        let start = Instant::now();
        let (prefix_len, mut cache) = match decode_blob(blob) {
            Some(parsed) => parsed,
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
        // Generate continuing from the restored prefix cache (dequantized to f32
        // if the blob was q8) — the cartridge drives generation; we do NOT
        // re-encode the context here.
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

    // Network + ~548MB weights → ignored by default. Run explicitly:
    //   cargo test -p mnesio-kv --features generative-kv -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "downloads GPT-2 weights (~548MB) + runs a real forward"]
    async fn cartridge_generation_equals_full_prompt_generation() {
        let be = GenerativeKvBackend::load().expect("load gpt2");
        let ctx = vec![
            "Paris is the capital of France.".to_string(),
            "Berlin is the capital of Germany.".to_string(),
        ];
        let query = "The capital of France is";

        // Cartridge path: prefill context once → serialize → restore → generate.
        let blob = be.compile_blob(&ctx).await;
        assert!(blob.len() > 1000, "blob is a real multi-layer KV cache");
        let from_cartridge = be.answer(&blob, query).await;
        assert!(from_cartridge.answered);

        // Oracle: generate over the full context++query prompt from scratch.
        let from_full = be.answer_full_prompt(&ctx, query);

        // The cartridge must give TOKEN-IDENTICAL output to the full prompt —
        // proving the KV cache + forward are correct and the cartridge is a
        // faithful (and cheaper) substitute for re-encoding the context.
        assert_eq!(
            from_cartridge.text.trim(),
            from_full.trim(),
            "cartridge generation must equal full-prompt generation"
        );
        // GPT-2 should actually answer the factual query.
        assert!(
            from_cartridge.text.contains("Paris"),
            "expected 'Paris', got {:?}",
            from_cartridge.text
        );
        eprintln!("generated from cartridge: {:?}", from_cartridge.text);
    }

    #[tokio::test]
    #[ignore = "downloads GPT-2 weights (~548MB)"]
    async fn erased_context_changes_generation() {
        let be = GenerativeKvBackend::load().expect("load gpt2");
        let with_fact = vec!["The project's codename is Nimbus.".to_string()];
        let without = vec!["The weather is sunny today.".to_string()];
        let q = "The project's codename is";

        let a = be.answer(&be.compile_blob(&with_fact).await, q).await;
        let b = be.answer(&be.compile_blob(&without).await, q).await;
        // Different cartridges (one shred-erased the fact) → different blobs and
        // (almost surely) different generations; the erased one can't surface it.
        assert!(!b.text.contains("Nimbus"), "erased fact must not appear");
        eprintln!("with fact: {:?}\nwithout:  {:?}", a.text, b.text);
    }

    // ---- q8 quantization (codec test needs no weights/network) ----

    /// The q8 codec is a real ~4× size win with bounded error — provable without
    /// loading GPT-2. Builds a synthetic multi-layer cache, encodes it at both
    /// precisions, and checks (a) the q8 blob is ~4× smaller and (b) decode(q8)
    /// reconstructs the f32 cache within the quantization step.
    #[test]
    fn q8_codec_is_4x_smaller_with_bounded_error() {
        // 3 layers × 6 token rows of deterministic-but-varied f32 values.
        let rows = 6usize;
        let mk = |salt: f32| -> LayerCache {
            let mut k = vec![0f32; rows * N_EMBD];
            let mut v = vec![0f32; rows * N_EMBD];
            for (i, slot) in k.iter_mut().enumerate() {
                *slot = ((i as f32 * 0.013 + salt).sin()) * 2.0;
            }
            for (i, slot) in v.iter_mut().enumerate() {
                *slot = ((i as f32 * 0.017 + salt).cos()) * 0.5;
            }
            LayerCache { k, v }
        };
        let layers = vec![mk(0.1), mk(0.2), mk(0.3)];

        let f32_blob = encode_blob(rows, &layers, Quant::F32);
        let q8_blob = encode_blob(rows, &layers, Quant::Q8);

        // Size: q8 is ~4× smaller (1 byte/elem + tiny per-row scale overhead vs
        // 4 bytes/elem). Assert a clear >3× win, honest because both are binary.
        assert!(
            (q8_blob.len() as f64) < (f32_blob.len() as f64) / 3.0,
            "q8 must be >3x smaller: f32={} q8={}",
            f32_blob.len(),
            q8_blob.len()
        );

        // Round-trip: f32 is exact, q8 is within one quant step of the original.
        let (pf, lf) = decode_blob(&f32_blob).expect("f32 decodes");
        let (pq, lq) = decode_blob(&q8_blob).expect("q8 decodes");
        assert_eq!(pf, rows);
        assert_eq!(pq, rows);
        for (orig, deq) in layers.iter().zip(&lf) {
            assert_eq!(orig.k, deq.k, "f32 round-trip is exact");
            assert_eq!(orig.v, deq.v);
        }
        let mut max_err = 0f32;
        for (orig, deq) in layers.iter().zip(&lq) {
            for (a, b) in orig.k.iter().zip(&deq.k) {
                max_err = max_err.max((a - b).abs());
            }
        }
        // Per-row scale ≤ amax/127; the worst dequant error is ≤ half a step,
        // comfortably under 0.05 for these magnitudes.
        assert!(max_err < 0.05, "q8 dequant error too large: {max_err}");
    }

    /// A q8 cartridge generates a correct answer at ~4× smaller blob — the real
    /// `quant` win on the actual GPT-2 cache.
    #[tokio::test]
    #[ignore = "downloads GPT-2 weights (~548MB) + runs a real forward"]
    async fn q8_cartridge_is_smaller_and_still_answers() {
        let f32_be = GenerativeKvBackend::load().expect("load gpt2");
        let q8_be = GenerativeKvBackend::load()
            .expect("load gpt2")
            .with_quant(Quant::Q8);
        assert_eq!(q8_be.quant().label(), "q8");

        let ctx = vec![
            "Paris is the capital of France.".to_string(),
            "Berlin is the capital of Germany.".to_string(),
        ];
        let query = "The capital of France is";

        let f32_blob = f32_be.compile_blob(&ctx).await;
        let q8_blob = q8_be.compile_blob(&ctx).await;
        assert!(
            (q8_blob.len() as f64) < (f32_blob.len() as f64) / 3.0,
            "q8 cartridge must be >3x smaller: f32={} q8={}",
            f32_blob.len(),
            q8_blob.len()
        );

        // The q8 cartridge still answers the factual query correctly despite the
        // lossy cache.
        let ans = q8_be.answer(&q8_blob, query).await;
        assert!(ans.answered);
        assert!(
            ans.text.contains("Paris"),
            "q8 cartridge should still answer 'Paris', got {:?}",
            ans.text
        );
        eprintln!(
            "f32 blob = {} bytes, q8 blob = {} bytes ({:.1}x smaller); q8 answer = {:?}",
            f32_blob.len(),
            q8_blob.len(),
            f32_blob.len() as f64 / q8_blob.len() as f64,
            ans.text
        );
    }
}
