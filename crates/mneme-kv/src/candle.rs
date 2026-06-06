//! **GPU-accelerated** KV cartridge (feature `candle-kv`) — the same cartridge
//! semantics as [`crate::QwenKvBackend`] (Qwen2.5-0.5B-Instruct) but with the
//! forward run on-device via [`candle_core`], so prefill is batched on the GPU
//! (Apple Metal with `--features metal`, CPU otherwise) instead of the pure-Rust
//! token-by-token loop.
//!
//! ## Why this exists / what it proves
//!
//! It's the GPU answer to "use a more advanced model, faster". The pure-Rust
//! `qwen-kv` backend is the **semantic oracle**: both must answer the same
//! factual query ("capital of France" → "Paris"). We do *not* assert
//! bit-identical tokens *across* backends — GPU and scalar-CPU f32 round
//! differently — but we *do* assert candle's **own** self-consistency
//! (cartridge generation == full-prompt generation, same device, same rounding),
//! which is the exact cartridge-correctness claim.
//!
//! The cartridge still owns the per-layer K/V cache ([`candle_core`]'s built-in
//! models keep theirs private), so the versioning / gate / crypto-shred
//! reconciliations are unchanged — this is a speed swap behind the
//! [`KvBackend`](crate::KvBackend) seam, not a redesign. The same code runs on
//! `Device::Cpu` or `Device::new_metal`, so the GPU-vs-CPU prefill latency is a
//! like-for-like measurement.

use crate::cartridge::{KvAnswer, KvBackend};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
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
const RMS_EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 1_000_000.0;
const EOS: u32 = 151645; // <|im_end|>
const MAX_CTX: usize = 64; // cap cartridge prefix tokens
const MAX_NEW: usize = 8; // generated tokens per answer
const MAX_POS: usize = MAX_CTX + MAX_NEW + 8; // RoPE table length

const MAGIC_F32: &[u8; 4] = b"CWF4";

/// Pick the fastest available device: Metal GPU when the `metal` feature is on
/// and a device is present, otherwise CPU. Never panics — always returns a
/// usable device.
pub fn best_device() -> Device {
    #[cfg(feature = "metal")]
    {
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
    }
    Device::Cpu
}

/// Human-readable label for the active device (for metrics / logs).
pub fn device_label(d: &Device) -> &'static str {
    if d.is_metal() {
        "metal"
    } else if d.is_cuda() {
        "cuda"
    } else {
        "cpu"
    }
}

/// One transformer block's weights as on-device tensors.
struct Block {
    ln1: Tensor,    // [HID]
    q_w: Tensor,    // [Q_DIM, HID]
    q_b: Tensor,    // [Q_DIM]
    k_w: Tensor,    // [KV_DIM, HID]
    k_b: Tensor,    // [KV_DIM]
    v_w: Tensor,    // [KV_DIM, HID]
    v_b: Tensor,    // [KV_DIM]
    o_w: Tensor,    // [HID, Q_DIM]
    ln2: Tensor,    // [HID]
    gate_w: Tensor, // [INTER, HID]
    up_w: Tensor,   // [INTER, HID]
    down_w: Tensor, // [HID, INTER]
}

/// Per-layer growing KV cache as on-device tensors, each `[n_kv, total, head_dim]`.
#[derive(Clone)]
struct LayerCache {
    k: Tensor,
    v: Tensor,
}

/// GPU-accelerated Qwen2 KV-cartridge backend (see module docs).
pub struct QwenCandleBackend {
    model_id: String,
    device: Device,
    embed: Tensor, // [VOCAB, HID] — also the tied lm_head
    blocks: Vec<Block>,
    norm: Tensor, // final RMSNorm [HID]
    cos: Tensor,  // [MAX_POS, HEAD_DIM/2]
    sin: Tensor,  // [MAX_POS, HEAD_DIM/2]
    tokenizer: Tokenizer,
}

/// `y = x · Wᵀ (+ b)` for `x [seq, in]`, `w [out, in]` → `[seq, out]`.
fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let y = x.matmul(&w.t()?)?;
    match b {
        Some(b) => Ok(y.broadcast_add(b)?),
        None => Ok(y),
    }
}

/// Repeat KV heads to match query heads (grouped-query attention).
/// `x [n_kv, t, d]` → `[n_kv*GROUP, t, d]`.
fn repeat_kv(x: &Tensor) -> Result<Tensor> {
    if GROUP == 1 {
        return Ok(x.clone());
    }
    let (nkv, t, d) = x.dims3()?;
    Ok(x.unsqueeze(1)?
        .expand((nkv, GROUP, t, d))?
        .reshape((nkv * GROUP, t, d))?)
}

impl QwenCandleBackend {
    /// Download (cached) + load Qwen2.5-0.5B-Instruct onto the best device.
    pub fn load() -> Result<Self> {
        Self::load_on(best_device())
    }

    /// Load onto a specific device (used to measure GPU vs CPU on identical code).
    pub fn load_on(device: Device) -> Result<Self> {
        use hf_hub::api::sync::Api;
        let api = Api::new().map_err(|e| anyhow!("hf-hub init: {e}"))?;
        let repo = api.model(HF_REPO.to_string());
        let weights = repo
            .get("model.safetensors")
            .map_err(|e| anyhow!("download weights: {e}"))?;
        let tok = repo
            .get("tokenizer.json")
            .map_err(|e| anyhow!("download tokenizer: {e}"))?;
        // VarBuilder converts bf16 → f32 on load; we keep the forward in f32.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)
                .map_err(|e| anyhow!("mmap safetensors: {e}"))?
        };

        let embed = vb.get((VOCAB, HID), "model.embed_tokens.weight")?;
        let mut blocks = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            let p = vb.pp(format!("model.layers.{i}"));
            let attn = p.pp("self_attn");
            let mlp = p.pp("mlp");
            blocks.push(Block {
                ln1: p.get(HID, "input_layernorm.weight")?,
                q_w: attn.get((Q_DIM, HID), "q_proj.weight")?,
                q_b: attn.get(Q_DIM, "q_proj.bias")?,
                k_w: attn.get((KV_DIM, HID), "k_proj.weight")?,
                k_b: attn.get(KV_DIM, "k_proj.bias")?,
                v_w: attn.get((KV_DIM, HID), "v_proj.weight")?,
                v_b: attn.get(KV_DIM, "v_proj.bias")?,
                o_w: attn.get((HID, Q_DIM), "o_proj.weight")?,
                ln2: p.get(HID, "post_attention_layernorm.weight")?,
                gate_w: mlp.get((INTER, HID), "gate_proj.weight")?,
                up_w: mlp.get((INTER, HID), "up_proj.weight")?,
                down_w: mlp.get((HID, INTER), "down_proj.weight")?,
            });
        }
        let norm = vb.get(HID, "model.norm.weight")?;
        let (cos, sin) = Self::rope_tables(&device)?;
        let tokenizer = Tokenizer::from_file(&tok).map_err(|e| anyhow!("tokenizer: {e}"))?;

        Ok(Self {
            model_id: "qwen2.5-0.5b-instruct-candle".to_string(),
            device,
            embed,
            blocks,
            norm,
            cos,
            sin,
            tokenizer,
        })
    }

    /// RoPE cos/sin tables, shape `[MAX_POS, HEAD_DIM/2]` (rotate-half layout).
    fn rope_tables(device: &Device) -> Result<(Tensor, Tensor)> {
        let half = HEAD_DIM / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| (1.0 / ROPE_THETA.powf(2.0 * j as f64 / HEAD_DIM as f64)) as f32)
            .collect();
        let mut cos = Vec::with_capacity(MAX_POS * half);
        let mut sin = Vec::with_capacity(MAX_POS * half);
        for p in 0..MAX_POS {
            for &f in &inv_freq {
                let a = p as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        let cos = Tensor::from_vec(cos, (MAX_POS, half), device)?;
        let sin = Tensor::from_vec(sin, (MAX_POS, half), device)?;
        Ok((cos, sin))
    }

    /// The active device's label (`metal` / `cuda` / `cpu`).
    pub fn device_label(&self) -> &'static str {
        device_label(&self.device)
    }

    fn empty_cache(&self) -> Vec<Option<LayerCache>> {
        (0..self.blocks.len()).map(|_| None).collect()
    }

    /// Forward `ids` (a batch of `seq` tokens) at absolute positions
    /// `start_pos..start_pos+seq`, growing each layer's KV cache and attending
    /// causally over it. Returns the **last token's** logits `[VOCAB]`.
    fn forward(
        &self,
        ids: &[u32],
        start_pos: usize,
        cache: &mut [Option<LayerCache>],
    ) -> Result<Tensor> {
        let seq = ids.len();
        let id_t = Tensor::from_vec(ids.to_vec(), (seq,), &self.device)?;
        let mut h = self.embed.index_select(&id_t, 0)?; // [seq, HID]

        // RoPE slices for these absolute positions.
        let cos = self.cos.narrow(0, start_pos, seq)?;
        let sin = self.sin.narrow(0, start_pos, seq)?;
        let scale = (HEAD_DIM as f64).sqrt();

        for (li, blk) in self.blocks.iter().enumerate() {
            // --- attention ---
            let normed = candle_nn::ops::rms_norm(&h.contiguous()?, &blk.ln1, RMS_EPS as f32)?;
            let q = linear(&normed, &blk.q_w, Some(&blk.q_b))?; // [seq, Q_DIM]
            let k = linear(&normed, &blk.k_w, Some(&blk.k_b))?; // [seq, KV_DIM]
            let v = linear(&normed, &blk.v_w, Some(&blk.v_b))?; // [seq, KV_DIM]

            // → [heads, seq, head_dim]
            let q = q
                .reshape((seq, N_HEAD, HEAD_DIM))?
                .transpose(0, 1)?
                .contiguous()?;
            let k = k
                .reshape((seq, N_KV, HEAD_DIM))?
                .transpose(0, 1)?
                .contiguous()?;
            let v = v
                .reshape((seq, N_KV, HEAD_DIM))?
                .transpose(0, 1)?
                .contiguous()?;

            // RoPE on Q and K (rotate-half), via a [1,h,seq,d] batch dim.
            let q = candle_nn::rotary_emb::rope(&q.unsqueeze(0)?, &cos, &sin)?.squeeze(0)?;
            let k = candle_nn::rotary_emb::rope(&k.unsqueeze(0)?, &cos, &sin)?.squeeze(0)?;

            // Append to cache → [n_kv, total, head_dim].
            let (k_all, v_all) = match &cache[li] {
                Some(c) => (Tensor::cat(&[&c.k, &k], 1)?, Tensor::cat(&[&c.v, &v], 1)?),
                None => (k.clone(), v.clone()),
            };
            cache[li] = Some(LayerCache {
                k: k_all.clone(),
                v: v_all.clone(),
            });
            let total = k_all.dim(1)?;

            // Grouped-query attention: repeat KV heads to N_HEAD.
            let k_rep = repeat_kv(&k_all)?; // [N_HEAD, total, hd]
            let v_rep = repeat_kv(&v_all)?;
            // scores [N_HEAD, seq, total]
            let scores = (q.matmul(&k_rep.transpose(1, 2)?.contiguous()?)? / scale)?;
            // causal mask: new row i (abs pos start_pos+i) may see cols 0..=start_pos+i.
            let mask = self.causal_mask(seq, total, start_pos)?;
            let scores = scores.broadcast_add(&mask)?;
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let ctx = probs.matmul(&v_rep)?; // [N_HEAD, seq, hd]
            let ctx = ctx.transpose(0, 1)?.reshape((seq, Q_DIM))?; // [seq, Q_DIM]
            let o = linear(&ctx, &blk.o_w, None)?;
            h = (h + o)?;

            // --- MLP (SwiGLU) ---
            let normed2 = candle_nn::ops::rms_norm(&h.contiguous()?, &blk.ln2, RMS_EPS as f32)?;
            let gate = candle_nn::ops::silu(&linear(&normed2, &blk.gate_w, None)?)?;
            let up = linear(&normed2, &blk.up_w, None)?;
            let act = (gate * up)?;
            let down = linear(&act, &blk.down_w, None)?;
            h = (h + down)?;
        }

        let h = candle_nn::ops::rms_norm(&h.contiguous()?, &self.norm, RMS_EPS as f32)?;
        let last = h.i(seq - 1)?; // [HID]
                                  // logits = last · embedᵀ  (tied lm_head)
        let logits = last.unsqueeze(0)?.matmul(&self.embed.t()?)?.squeeze(0)?;
        Ok(logits)
    }

    /// Additive causal mask `[seq, total]`: 0 where attendable, -inf otherwise.
    fn causal_mask(&self, seq: usize, total: usize, start_pos: usize) -> Result<Tensor> {
        let mut m = vec![0f32; seq * total];
        for i in 0..seq {
            let limit = start_pos + i; // last attendable column index
            for (j, slot) in m[i * total..(i + 1) * total].iter_mut().enumerate() {
                if j > limit {
                    *slot = f32::NEG_INFINITY;
                }
            }
        }
        Ok(Tensor::from_vec(m, (seq, total), &self.device)?)
    }

    fn prefill(&self, ids: &[u32]) -> Result<Vec<Option<LayerCache>>> {
        let mut cache = self.empty_cache();
        if !ids.is_empty() {
            self.forward(ids, 0, &mut cache)?;
        }
        Ok(cache)
    }

    /// Greedily generate up to `MAX_NEW` tokens, feeding `query` first.
    fn generate(
        &self,
        cache: &mut [Option<LayerCache>],
        query: &[u32],
        start_pos: usize,
    ) -> Result<Vec<u32>> {
        let mut pos = start_pos;
        let mut logits = self.forward(query, pos, cache)?;
        pos += query.len();
        let mut out = Vec::with_capacity(MAX_NEW);
        for _ in 0..MAX_NEW {
            if pos >= MAX_POS {
                break;
            }
            let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            if next == EOS {
                break;
            }
            out.push(next);
            logits = self.forward(&[next], pos, cache)?;
            pos += 1;
        }
        Ok(out)
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

    /// Generate over the full `context ++ query` sequence from scratch — the
    /// self-consistency oracle for [`KvBackend::answer`].
    pub fn answer_full_prompt(&self, contents: &[String], query: &str) -> Result<String> {
        let ctx = self.context_ids(contents);
        let q = self.encode(query, MAX_CTX);
        let mut cache = self.prefill(&ctx)?;
        let gen = self.generate(&mut cache, &q, ctx.len())?;
        Ok(self.decode(&gen))
    }

    /// Wall-clock cost of prefilling `contents` on this device (for the GPU-vs-CPU
    /// latency comparison).
    pub fn time_prefill(&self, contents: &[String]) -> Result<std::time::Duration> {
        let ctx = self.context_ids(contents);
        let start = Instant::now();
        let _ = self.prefill(&ctx)?;
        Ok(start.elapsed())
    }
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

/// Serialize the prefilled KV cache as the cartridge blob. Each layer's K/V is
/// pulled to host f32 in `[total, KV_DIM]` row-major (KV heads concatenated).
fn encode_blob(prefix_len: usize, cache: &[Option<LayerCache>]) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC_F32);
    put_u32(&mut buf, prefix_len);
    put_u32(&mut buf, cache.len());
    for layer in cache {
        let Some(c) = layer else {
            put_u32(&mut buf, 0);
            continue;
        };
        let total = c.k.dim(1)?;
        put_u32(&mut buf, total);
        // [n_kv, total, hd] → [total, n_kv, hd] → [total, KV_DIM]
        let k =
            c.k.transpose(0, 1)?
                .reshape((total, KV_DIM))?
                .to_vec2::<f32>()?;
        let v =
            c.v.transpose(0, 1)?
                .reshape((total, KV_DIM))?
                .to_vec2::<f32>()?;
        for row in k.iter().chain(v.iter()) {
            for &x in row {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    Ok(buf)
}

/// Inverse of [`encode_blob`]: restore the KV cache onto `device`.
fn decode_blob(blob: &[u8], device: &Device) -> Option<(usize, Vec<Option<LayerCache>>)> {
    if blob.get(0..4)? != MAGIC_F32 {
        return None;
    }
    let mut off = 4usize;
    let prefix_len = get_u32(blob, &mut off)?;
    let n_layers = get_u32(blob, &mut off)?;
    let mut cache = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let total = get_u32(blob, &mut off)?;
        if total == 0 {
            cache.push(None);
            continue;
        }
        let read = |off: &mut usize| -> Option<Tensor> {
            let n = total * KV_DIM;
            let mut flat = vec![0f32; n];
            for slot in flat.iter_mut() {
                let end = off.checked_add(4)?;
                let b = blob.get(*off..end)?;
                *off = end;
                *slot = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            // [total, KV_DIM] → [total, n_kv, hd] → [n_kv, total, hd]
            let t = Tensor::from_vec(flat, (total, KV_DIM), device).ok()?;
            t.reshape((total, N_KV, HEAD_DIM))
                .ok()?
                .transpose(0, 1)
                .ok()?
                .contiguous()
                .ok()
        };
        let k = read(&mut off)?;
        let v = read(&mut off)?;
        cache.push(Some(LayerCache { k, v }));
    }
    Some((prefix_len, cache))
}

#[async_trait]
impl KvBackend for QwenCandleBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn compile_blob(&self, contents: &[String]) -> Vec<u8> {
        let ctx = self.context_ids(contents);
        self.prefill(&ctx)
            .and_then(|c| encode_blob(ctx.len(), &c))
            .unwrap_or_default()
    }

    async fn answer(&self, blob: &[u8], query: &str) -> KvAnswer {
        let start = Instant::now();
        let (prefix_len, mut cache) = match decode_blob(blob, &self.device) {
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
        let text = match self.generate(&mut cache, &q, prefix_len) {
            Ok(g) => self.decode(&g),
            Err(_) => String::new(),
        };
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
    use candle_core::Tensor;

    /// De-risk smoke: a matmul on the best device. With `--features
    /// candle-kv,metal` on Apple Silicon this exercises the real Metal backend.
    #[test]
    fn device_matmul_smoke() {
        let dev = best_device();
        eprintln!("candle device: {}", device_label(&dev));
        let a = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), &dev).unwrap();
        let b = Tensor::from_vec(vec![1f32, 0., 0., 1., 1., 1.], (3, 2), &dev).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.dims(), &[2, 2]);
        let got = c.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(got[0], vec![4.0, 5.0]);
        assert_eq!(got[1], vec![10.0, 11.0]);
    }

    // Network + ~1GB weights + a real Qwen2 forward → ignored. Run with:
    //   cargo test -p mneme-kv --release --features candle-kv,metal -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "downloads Qwen2.5-0.5B (~1GB) + runs a real GPU forward"]
    async fn candle_cartridge_is_exact_erasable_and_fast() {
        let be = QwenCandleBackend::load().expect("load qwen (candle)");
        eprintln!("candle backend device: {}", be.device_label());

        // (1) Self-consistency: cartridge generation == full-prompt generation,
        // both on the same device (so rounding matches → token-identical).
        let ctx = vec![
            "Paris is the capital of France.".to_string(),
            "Berlin is the capital of Germany.".to_string(),
        ];
        let query = "\nQuestion: What is the capital of France?\nAnswer:";
        let blob = be.compile_blob(&ctx).await;
        assert!(blob.len() > 1000, "blob is a real multi-layer KV cache");
        let from_cartridge = be.answer(&blob, query).await;
        assert!(from_cartridge.answered);
        let from_full = be.answer_full_prompt(&ctx, query).expect("full prompt");
        assert_eq!(
            from_cartridge.text.trim(),
            from_full.trim(),
            "cartridge generation must equal full-prompt generation"
        );
        assert!(
            from_cartridge.text.contains("Paris"),
            "expected 'Paris', got {:?}",
            from_cartridge.text
        );
        eprintln!("candle cartridge answer: {:?}", from_cartridge.text);

        // (2) Erasure-by-recompile: the fact is gone from generation.
        let with_fact = vec!["The project's codename is Nimbus.".to_string()];
        let without = vec!["The weather is sunny today.".to_string()];
        let q = "\nQuestion: What is the project's codename?\nAnswer:";
        let _ = be.answer(&be.compile_blob(&with_fact).await, q).await;
        let b = be.answer(&be.compile_blob(&without).await, q).await;
        assert!(!b.text.contains("Nimbus"), "erased fact must not appear");

        // (3) GPU-vs-CPU prefill latency on identical code.
        let big: Vec<String> = (0..8)
            .map(|i| format!("Fact {i}: the quick brown fox jumps over the lazy dog."))
            .collect();
        let gpu = be.time_prefill(&big).expect("gpu prefill");
        let cpu_be = QwenCandleBackend::load_on(Device::Cpu).expect("load cpu");
        let cpu = cpu_be.time_prefill(&big).expect("cpu prefill");
        eprintln!(
            "prefill latency — {}: {:?}, cpu: {:?} ({:.1}x)",
            be.device_label(),
            gpu,
            cpu,
            cpu.as_secs_f64() / gpu.as_secs_f64().max(1e-9)
        );
    }
}
