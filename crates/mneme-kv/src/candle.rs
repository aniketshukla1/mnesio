//! **GPU-accelerated** KV cartridge (feature `candle-kv`) — the same cartridge
//! semantics as [`crate::QwenKvBackend`] but with the forward run on-device via
//! [`candle_core`], so prefill is batched on the GPU (Apple Metal with
//! `--features metal`, CPU otherwise) instead of the pure-Rust token-by-token
//! loop.
//!
//! ## Config-driven (any Qwen2-family size) + selectable precision
//!
//! The model dimensions are read from the repo's `config.json`, so the *same*
//! code loads Qwen2.5 **0.5B / 1.5B / 3B / 7B / …** — "use a bigger model" is a
//! repo string, not a rewrite. Precision is selectable: [`Precision::F32`]
//! (exact), [`Precision::F16`] (half, but narrow exponent), or
//! [`Precision::BF16`] (half, **f32's exponent range**).
//!
//! **Deep models need bf16, not f16.** Qwen ships bf16 and its activations use
//! that wide exponent range. In f16 (5-bit exponent, max ≈ 65504) a deep model's
//! values overflow to inf and generation collapses to garbage — the 0.5B happens
//! to stay in range, the 1.5B/28-layer does not. [`Precision::BF16`] keeps half
//! the memory of f32 *with* f32's range, so the 1.5B answers correctly (verified
//! live). Independently, the forward runs **mixed precision** as good hygiene:
//! weights + KV cache in the chosen dtype, but the residual stream / RMSNorm /
//! softmax / logits accumulate in **f32** (see [`QwenCandleBackend::forward`]).
//!
//! ## What it proves
//!
//! The pure-Rust `qwen-kv` backend is the **semantic oracle**: both must answer
//! the same factual query ("capital of France" → "Paris"). We do *not* assert
//! bit-identical tokens *across* backends — GPU and scalar-CPU f32 round
//! differently — but we *do* assert candle's **own** self-consistency (cartridge
//! generation == full-prompt generation, same device + dtype), the exact
//! cartridge-correctness claim. The cartridge still owns the per-layer K/V cache
//! ([`candle_core`]'s built-in models keep theirs private), so versioning / gate
//! / crypto-shred are unchanged — a speed swap behind the
//! [`KvBackend`](crate::KvBackend) seam. The same code runs on `Device::Cpu` or
//! `Device::new_metal`, so the GPU-vs-CPU prefill latency is like-for-like.

use crate::cartridge::{KvAnswer, KvBackend};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Default model — the smallest Qwen2.5 instruct checkpoint (ungated, ~1GB bf16).
pub const DEFAULT_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct";

const MAX_CTX: usize = 64; // cap cartridge prefix tokens
const MAX_NEW: usize = 8; // generated tokens per answer
const MAX_POS: usize = MAX_CTX + MAX_NEW + 8; // RoPE table length
const MAGIC: &[u8; 4] = b"CWK5";

/// Cartridge tensor precision for the on-device forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Full f32 (exact, larger, the correctness baseline).
    F32,
    /// Half f16 (5-bit exponent). Smallest, but its narrow exponent range
    /// (max ≈ 65504) overflows in *deep* Qwen models → use [`Precision::BF16`]
    /// for those. Fine for the 0.5B.
    F16,
    /// Brain-float bf16 (8-bit exponent — **f32's range**, 7-bit mantissa). The
    /// model's *native* dtype on disk, so it's the right half precision: half the
    /// memory of f32 with no exponent overflow, so even deep models (1.5B+) stay
    /// stable. Requires a bf16-capable device (Apple M-series Metal, CUDA).
    BF16,
}

impl Precision {
    fn dtype(self) -> DType {
        match self {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::BF16 => DType::BF16,
        }
    }
    /// The [`CartridgeKey::quant`](crate::CartridgeKey::quant) label this maps to.
    pub fn label(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
            Precision::BF16 => "bf16",
        }
    }

    /// Parse a precision label (`f32` / `f16` / `bf16`, case-insensitive).
    /// `None` for anything else, so callers can fall back to a default.
    pub fn from_label(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "float32" => Some(Precision::F32),
            "f16" | "fp16" | "float16" | "half" => Some(Precision::F16),
            "bf16" | "bfloat16" => Some(Precision::BF16),
            _ => None,
        }
    }
}

/// Architecture dimensions read from a model `config.json`. Despite the name,
/// this covers the **Qwen2 *and* Llama families** — both are RMSNorm + RoPE +
/// grouped-query attention + SwiGLU. The only structural difference the loader
/// cares about is the attention QKV **bias** (present in Qwen2, absent in
/// Llama), which is loaded optionally (see [`Block`]).
///
/// Assumes a **tied** `lm_head` (the embedding matrix doubles as the output
/// projection — true for Qwen2.5 and Llama-3.2) and standard RoPE.
// TODO(phase-12): untied `lm_head` checkpoints (Llama-2 / TinyLlama) and RoPE
// scaling (`rope_scaling`) are not yet handled — load `lm_head.weight` when
// present, and apply the scaling factor to the RoPE tables.
#[derive(Debug, Clone)]
struct QwenConfig {
    hid: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    n_layer: usize,
    inter: usize,
    vocab: usize,
    eps: f64,
    rope_theta: f64,
    eos: u32,
}

impl QwenConfig {
    fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    fn kv_dim(&self) -> usize {
        self.n_kv * self.head_dim
    }
    fn group(&self) -> usize {
        self.n_head / self.n_kv
    }
}

/// `eos_token_id` is a single int in Qwen2 configs but an **array** in Llama-3
/// configs (several end tokens). Accept either — we only need one stop token.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum EosTokenId {
    One(u32),
    Many(Vec<u32>),
}

impl EosTokenId {
    fn first(&self) -> u32 {
        match self {
            EosTokenId::One(x) => *x,
            EosTokenId::Many(v) => v.first().copied().unwrap_or(0),
        }
    }
}

#[derive(serde::Deserialize)]
struct RawConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    eos_token_id: EosTokenId,
}

impl From<RawConfig> for QwenConfig {
    fn from(r: RawConfig) -> Self {
        Self {
            head_dim: r.hidden_size / r.num_attention_heads,
            hid: r.hidden_size,
            n_head: r.num_attention_heads,
            n_kv: r.num_key_value_heads,
            n_layer: r.num_hidden_layers,
            inter: r.intermediate_size,
            vocab: r.vocab_size,
            eps: r.rms_norm_eps,
            rope_theta: r.rope_theta,
            eos: r.eos_token_id.first(),
        }
    }
}

/// Pick the fastest available device: Metal GPU when the `metal` feature is on
/// and a device is present, otherwise CPU. Never panics.
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
    ln1: Tensor,
    q_w: Tensor,
    /// Attention QKV biases — present in Qwen2, absent in Llama, hence optional.
    q_b: Option<Tensor>,
    k_w: Tensor,
    k_b: Option<Tensor>,
    v_w: Tensor,
    v_b: Option<Tensor>,
    o_w: Tensor,
    ln2: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
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
    dtype: DType,
    cfg: QwenConfig,
    embed: Tensor, // [vocab, hid] — also the tied lm_head
    blocks: Vec<Block>,
    norm: Tensor,
    cos: Tensor, // [MAX_POS, head_dim/2]
    sin: Tensor,
    tokenizer: Tokenizer,
}

/// Cast to `dt`, returning a cheap clone when already that dtype (so the f32
/// path pays nothing for the mixed-precision casts).
fn cast(t: &Tensor, dt: DType) -> Result<Tensor> {
    if t.dtype() == dt {
        Ok(t.clone())
    } else {
        Ok(t.to_dtype(dt)?)
    }
}

/// RMSNorm computed in **f32** regardless of the weight dtype: upcast the input
/// and (tiny `[hid]`) weight to f32, normalize, return f32. Stability for the
/// residual stream in mixed precision.
fn rms_norm_f32(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x = cast(&x.contiguous()?, DType::F32)?;
    let w = cast(weight, DType::F32)?;
    Ok(candle_nn::ops::rms_norm(&x, &w, eps)?)
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
/// `x [n_kv, t, d]` → `[n_kv*group, t, d]`.
fn repeat_kv(x: &Tensor, group: usize) -> Result<Tensor> {
    if group == 1 {
        return Ok(x.clone());
    }
    let (nkv, t, d) = x.dims3()?;
    Ok(x.unsqueeze(1)?
        .expand((nkv, group, t, d))?
        .reshape((nkv * group, t, d))?)
}

impl QwenCandleBackend {
    /// Load the default model ([`DEFAULT_REPO`]) at f32 onto the best device.
    pub fn load() -> Result<Self> {
        Self::load_repo(DEFAULT_REPO, best_device(), Precision::F32)
    }

    /// Load the default model at f32 onto a specific device (for GPU-vs-CPU).
    pub fn load_on(device: Device) -> Result<Self> {
        Self::load_repo(DEFAULT_REPO, device, Precision::F32)
    }

    /// Load the default model at f32 forced onto the CPU — the baseline half of
    /// the GPU-vs-CPU comparison, without the caller needing a `candle` `Device`.
    pub fn load_cpu() -> Result<Self> {
        Self::load_repo(DEFAULT_REPO, Device::Cpu, Precision::F32)
    }

    /// Load `repo` at `precision` onto the **best** device — the GPU-side
    /// constructor for callers (e.g. the server) that don't have a `candle`
    /// `Device` in scope.
    pub fn load_best(repo: &str, precision: Precision) -> Result<Self> {
        Self::load_repo(repo, best_device(), precision)
    }

    /// Load `repo` at `precision` forced onto the **CPU** — the matched baseline
    /// for a fair GPU-vs-CPU comparison on the same model + precision.
    pub fn load_cpu_repo(repo: &str, precision: Precision) -> Result<Self> {
        Self::load_repo(repo, Device::Cpu, precision)
    }

    /// Load any Qwen2-family `repo` at the given precision onto `device`. The
    /// architecture comes from the repo's `config.json`, so larger sizes load
    /// without code changes.
    pub fn load_repo(repo: &str, device: Device, precision: Precision) -> Result<Self> {
        use hf_hub::api::sync::Api;
        let api = Api::new().map_err(|e| {
            anyhow!(
                "{}: {e}",
                crate::cartridge::weights_hint(repo, "model weights")
            )
        })?;
        let r = api.model(repo.to_string());
        let cfg_path = r.get("config.json").map_err(|e| {
            anyhow!(
                "{}: {e}",
                crate::cartridge::weights_hint(repo, "config.json")
            )
        })?;
        let weights = r.get("model.safetensors").map_err(|e| {
            anyhow!(
                "{}: {e}",
                crate::cartridge::weights_hint(repo, "model.safetensors")
            )
        })?;
        let tok = r.get("tokenizer.json").map_err(|e| {
            anyhow!(
                "{}: {e}",
                crate::cartridge::weights_hint(repo, "tokenizer.json")
            )
        })?;

        let raw: RawConfig = serde_json::from_slice(&std::fs::read(&cfg_path)?)
            .map_err(|e| anyhow!("parse config.json: {e}"))?;
        let cfg = QwenConfig::from(raw);
        let dtype = precision.dtype();

        // VarBuilder converts the on-disk dtype (bf16) → the chosen compute dtype.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)
                .map_err(|e| anyhow!("mmap safetensors: {e}"))?
        };

        let embed = vb.get((cfg.vocab, cfg.hid), "model.embed_tokens.weight")?;
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        let (q_dim, kv_dim) = (cfg.q_dim(), cfg.kv_dim());
        for i in 0..cfg.n_layer {
            let p = vb.pp(format!("model.layers.{i}"));
            let attn = p.pp("self_attn");
            let mlp = p.pp("mlp");
            blocks.push(Block {
                ln1: p.get(cfg.hid, "input_layernorm.weight")?,
                q_w: attn.get((q_dim, cfg.hid), "q_proj.weight")?,
                q_b: attn.get(q_dim, "q_proj.bias").ok(),
                k_w: attn.get((kv_dim, cfg.hid), "k_proj.weight")?,
                k_b: attn.get(kv_dim, "k_proj.bias").ok(),
                v_w: attn.get((kv_dim, cfg.hid), "v_proj.weight")?,
                v_b: attn.get(kv_dim, "v_proj.bias").ok(),
                o_w: attn.get((cfg.hid, q_dim), "o_proj.weight")?,
                ln2: p.get(cfg.hid, "post_attention_layernorm.weight")?,
                gate_w: mlp.get((cfg.inter, cfg.hid), "gate_proj.weight")?,
                up_w: mlp.get((cfg.inter, cfg.hid), "up_proj.weight")?,
                down_w: mlp.get((cfg.hid, cfg.inter), "down_proj.weight")?,
            });
        }
        let norm = vb.get(cfg.hid, "model.norm.weight")?;
        let (cos, sin) = Self::rope_tables(&cfg, dtype, &device)?;
        let tokenizer = Tokenizer::from_file(&tok).map_err(|e| anyhow!("tokenizer: {e}"))?;

        let short = repo.rsplit('/').next().unwrap_or(repo).to_ascii_lowercase();
        Ok(Self {
            model_id: format!("{short}-candle-{}", precision.label()),
            device,
            dtype,
            cfg,
            embed,
            blocks,
            norm,
            cos,
            sin,
            tokenizer,
        })
    }

    /// RoPE cos/sin tables `[MAX_POS, head_dim/2]` (rotate-half), in `dtype`.
    fn rope_tables(cfg: &QwenConfig, dtype: DType, device: &Device) -> Result<(Tensor, Tensor)> {
        let half = cfg.head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| (1.0 / cfg.rope_theta.powf(2.0 * j as f64 / cfg.head_dim as f64)) as f32)
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
        let cos = Tensor::from_vec(cos, (MAX_POS, half), device)?.to_dtype(dtype)?;
        let sin = Tensor::from_vec(sin, (MAX_POS, half), device)?.to_dtype(dtype)?;
        Ok((cos, sin))
    }

    /// The active device's label (`metal` / `cuda` / `cpu`).
    pub fn device_label(&self) -> &'static str {
        device_label(&self.device)
    }

    /// The active precision label (`f32` / `f16` / `bf16`).
    pub fn precision_label(&self) -> &'static str {
        match self.dtype {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => "f32",
        }
    }

    fn empty_cache(&self) -> Vec<Option<LayerCache>> {
        (0..self.blocks.len()).map(|_| None).collect()
    }

    /// Forward `ids` at absolute positions `start_pos..start_pos+seq`, growing
    /// each layer's KV cache and attending causally over it. Returns the last
    /// token's logits `[vocab]`.
    ///
    /// **Mixed precision**: weights, KV cache, and matmuls run in the model dtype
    /// (`wd` — f16/bf16 save memory), but the **residual stream, RMSNorm,
    /// softmax, and final logits accumulate in f32** for numerical robustness.
    /// When `wd == f32` every cast is a cheap clone, so the f32 path is
    /// unchanged. (Note: the *deep-model* fix is choosing bf16 over f16 — f32
    /// accumulation alone doesn't rescue f16, because individual f16 *values*
    /// overflow its narrow exponent range; bf16 has f32's range. See the module
    /// docs.)
    fn forward(
        &self,
        ids: &[u32],
        start_pos: usize,
        cache: &mut [Option<LayerCache>],
    ) -> Result<Tensor> {
        let cfg = &self.cfg;
        let (q_dim, head_dim, n_head, n_kv) = (cfg.q_dim(), cfg.head_dim, cfg.n_head, cfg.n_kv);
        let wd = self.dtype; // weight / cache / matmul dtype
        let acc = DType::F32; // accumulation (residual / norm / softmax) dtype
        let eps = cfg.eps as f32;
        let seq = ids.len();
        let id_t = Tensor::from_vec(ids.to_vec(), (seq,), &self.device)?;
        // Residual stream lives in f32.
        let mut h = cast(&self.embed.index_select(&id_t, 0)?, acc)?; // [seq, hid]

        let cos = self.cos.narrow(0, start_pos, seq)?;
        let sin = self.sin.narrow(0, start_pos, seq)?;
        let scale = (head_dim as f64).sqrt();

        for (li, blk) in self.blocks.iter().enumerate() {
            // --- attention --- RMSNorm in f32, then downcast for the matmuls.
            let normed = rms_norm_f32(&h, &blk.ln1, eps)?;
            let nd = cast(&normed, wd)?;
            let q = linear(&nd, &blk.q_w, blk.q_b.as_ref())?;
            let k = linear(&nd, &blk.k_w, blk.k_b.as_ref())?;
            let v = linear(&nd, &blk.v_w, blk.v_b.as_ref())?;

            let q = q
                .reshape((seq, n_head, head_dim))?
                .transpose(0, 1)?
                .contiguous()?;
            let k = k
                .reshape((seq, n_kv, head_dim))?
                .transpose(0, 1)?
                .contiguous()?;
            let v = v
                .reshape((seq, n_kv, head_dim))?
                .transpose(0, 1)?
                .contiguous()?;

            let q = candle_nn::rotary_emb::rope(&q.unsqueeze(0)?, &cos, &sin)?.squeeze(0)?;
            let k = candle_nn::rotary_emb::rope(&k.unsqueeze(0)?, &cos, &sin)?.squeeze(0)?;

            let (k_all, v_all) = match &cache[li] {
                Some(c) => (Tensor::cat(&[&c.k, &k], 1)?, Tensor::cat(&[&c.v, &v], 1)?),
                None => (k.clone(), v.clone()),
            };
            cache[li] = Some(LayerCache {
                k: k_all.clone(),
                v: v_all.clone(),
            });
            let total = k_all.dim(1)?;

            let k_rep = repeat_kv(&k_all, cfg.group())?;
            let v_rep = repeat_kv(&v_all, cfg.group())?;
            // scores [n_head, seq, total]; mask + softmax in f32 for safety.
            let scores = (q.matmul(&k_rep.transpose(1, 2)?.contiguous()?)? / scale)?;
            let mask = self.causal_mask(seq, total, start_pos)?; // f32
            let scores = cast(&scores, acc)?.broadcast_add(&mask)?;
            let probs = cast(&candle_nn::ops::softmax_last_dim(&scores)?, wd)?;
            let ctx = probs.matmul(&v_rep)?; // [n_head, seq, head_dim]
            let ctx = ctx.transpose(0, 1)?.reshape((seq, q_dim))?;
            let o = cast(&linear(&ctx, &blk.o_w, None)?, acc)?;
            h = (h + o)?; // residual add in f32

            // --- MLP (SwiGLU) --- RMSNorm f32, matmuls wd, residual f32.
            let normed2 = rms_norm_f32(&h, &blk.ln2, eps)?;
            let nd2 = cast(&normed2, wd)?;
            let gate = candle_nn::ops::silu(&linear(&nd2, &blk.gate_w, None)?)?;
            let up = linear(&nd2, &blk.up_w, None)?;
            let act = (gate * up)?;
            let down = cast(&linear(&act, &blk.down_w, None)?, acc)?;
            h = (h + down)?; // residual add in f32
        }

        let h = rms_norm_f32(&h, &self.norm, eps)?; // f32
        let last = cast(&h.i(seq - 1)?, wd)?; // downcast for the tied lm_head matmul
        let logits = last.unsqueeze(0)?.matmul(&self.embed.t()?)?.squeeze(0)?;
        cast(&logits, acc)
    }

    /// Additive causal mask `[seq, total]` in **f32**: 0 where attendable, -inf
    /// otherwise (added to the f32 scores before softmax).
    fn causal_mask(&self, seq: usize, total: usize, start_pos: usize) -> Result<Tensor> {
        let mut m = vec![0f32; seq * total];
        for i in 0..seq {
            let limit = start_pos + i;
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
            if next == self.cfg.eos {
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

    /// Wall-clock cost of prefilling `contents` on this device (GPU-vs-CPU).
    pub fn time_prefill(&self, contents: &[String]) -> Result<std::time::Duration> {
        let ctx = self.context_ids(contents);
        let start = Instant::now();
        let _ = self.prefill(&ctx)?;
        Ok(start.elapsed())
    }

    /// Number of transformer layers (for metrics).
    pub fn n_layers(&self) -> usize {
        self.cfg.n_layer
    }
}

// ---------------- cartridge blob codec (compact binary, self-describing) ------

fn put_u32(buf: &mut Vec<u8>, v: usize) {
    buf.extend_from_slice(&(v as u32).to_le_bytes());
}
fn get_u32(buf: &[u8], off: &mut usize) -> Option<usize> {
    let end = off.checked_add(4)?;
    let b = buf.get(*off..end)?;
    *off = end;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
}

/// Serialize the prefilled KV cache (host f32, `[total, kv_dim]` per layer with
/// KV heads concatenated). The `kv_dim` is stored so decode is self-describing
/// across model sizes.
fn encode_blob(
    prefix_len: usize,
    kv_dim: usize,
    n_kv: usize,
    cache: &[Option<LayerCache>],
) -> Result<Vec<u8>> {
    let head_dim = kv_dim / n_kv;
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    put_u32(&mut buf, prefix_len);
    put_u32(&mut buf, cache.len());
    put_u32(&mut buf, kv_dim);
    put_u32(&mut buf, n_kv);
    for layer in cache {
        let Some(c) = layer else {
            put_u32(&mut buf, 0);
            continue;
        };
        let total = c.k.dim(1)?;
        put_u32(&mut buf, total);
        let to_rows = |t: &Tensor| -> Result<Vec<Vec<f32>>> {
            Ok(t.to_dtype(DType::F32)?
                .transpose(0, 1)?
                .reshape((total, n_kv * head_dim))?
                .to_vec2::<f32>()?)
        };
        let k = to_rows(&c.k)?;
        let v = to_rows(&c.v)?;
        for row in k.iter().chain(v.iter()) {
            for &x in row {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    Ok(buf)
}

/// Inverse of [`encode_blob`]: restore the KV cache onto `device` in `dtype`.
fn decode_blob(
    blob: &[u8],
    device: &Device,
    dtype: DType,
) -> Option<(usize, Vec<Option<LayerCache>>)> {
    if blob.get(0..4)? != MAGIC {
        return None;
    }
    let mut off = 4usize;
    let prefix_len = get_u32(blob, &mut off)?;
    let n_layers = get_u32(blob, &mut off)?;
    let kv_dim = get_u32(blob, &mut off)?;
    let n_kv = get_u32(blob, &mut off)?;
    if n_kv == 0 {
        return None;
    }
    let head_dim = kv_dim / n_kv;
    let mut cache = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let total = get_u32(blob, &mut off)?;
        if total == 0 {
            cache.push(None);
            continue;
        }
        let read = |off: &mut usize| -> Option<Tensor> {
            let n = total * kv_dim;
            let mut flat = vec![0f32; n];
            for slot in flat.iter_mut() {
                let end = off.checked_add(4)?;
                let b = blob.get(*off..end)?;
                *off = end;
                *slot = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            // [total, kv_dim] → [total, n_kv, head_dim] → [n_kv, total, head_dim]
            Tensor::from_vec(flat, (total, kv_dim), device)
                .ok()?
                .reshape((total, n_kv, head_dim))
                .ok()?
                .transpose(0, 1)
                .ok()?
                .contiguous()
                .ok()?
                .to_dtype(dtype)
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
        let (kv_dim, n_kv) = (self.cfg.kv_dim(), self.cfg.n_kv);
        self.prefill(&ctx)
            .and_then(|c| encode_blob(ctx.len(), kv_dim, n_kv, &c))
            .unwrap_or_default()
    }

    async fn answer(&self, blob: &[u8], query: &str) -> KvAnswer {
        let start = Instant::now();
        let (prefix_len, mut cache) = match decode_blob(blob, &self.device, self.dtype) {
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
        if cache.len() != self.cfg.n_layer || q.is_empty() {
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

    #[test]
    fn parses_llama_family_config_with_eos_array() {
        // A Llama-3-style config.json: `eos_token_id` is an *array*, and there
        // are no QKV-bias entries in config (bias lives in the weights and is
        // loaded optionally). No network or weights needed.
        let json = r#"{
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 16,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": [128001, 128008, 128009]
        }"#;
        let raw: RawConfig = serde_json::from_str(json).expect("llama config parses");
        let cfg = QwenConfig::from(raw);
        assert_eq!(cfg.hid, 2048);
        assert_eq!(cfg.head_dim, 64); // 2048 / 32
        assert_eq!(cfg.n_kv, 8);
        assert_eq!(cfg.group(), 4); // 32 / 8
        assert_eq!(cfg.eos, 128001); // first of the array
    }

    #[test]
    fn parses_qwen2_config_with_scalar_eos() {
        // Qwen2.5-0.5B shape: `eos_token_id` is a single int.
        let json = r#"{
            "hidden_size": 896,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "num_hidden_layers": 24,
            "intermediate_size": 4864,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "eos_token_id": 151645
        }"#;
        let raw: RawConfig = serde_json::from_str(json).expect("qwen config parses");
        let cfg = QwenConfig::from(raw);
        assert_eq!(cfg.eos, 151645);
        assert_eq!(cfg.n_head, 14);
        assert_eq!(cfg.head_dim, 64); // 896 / 14
    }

    // Network + ~1GB weights + a real Qwen2 forward → ignored. Run with:
    //   cargo test -p mneme-kv --release --features candle-kv,metal -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "downloads Qwen2.5-0.5B (~1GB) + runs a real GPU forward"]
    async fn candle_cartridge_is_exact_erasable_and_fast() {
        let be = QwenCandleBackend::load().expect("load qwen (candle)");
        eprintln!(
            "candle backend: {} / {} / {} layers",
            be.device_label(),
            be.precision_label(),
            be.n_layers()
        );

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

        // (2) Erasure-by-recompile.
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

    /// f16 precision: smaller/faster, must still answer correctly.
    #[tokio::test]
    #[ignore = "downloads Qwen2.5-0.5B (~1GB) + runs an f16 GPU forward"]
    async fn candle_f16_is_faster_and_still_answers() {
        let f32_be = QwenCandleBackend::load_repo(DEFAULT_REPO, best_device(), Precision::F32)
            .expect("load f32");
        let f16_be = QwenCandleBackend::load_repo(DEFAULT_REPO, best_device(), Precision::F16)
            .expect("load f16");
        assert_eq!(f16_be.precision_label(), "f16");

        let ctx = vec![
            "Paris is the capital of France.".to_string(),
            "Berlin is the capital of Germany.".to_string(),
        ];
        let query = "\nQuestion: What is the capital of France?\nAnswer:";
        let ans = f16_be.answer(&f16_be.compile_blob(&ctx).await, query).await;
        assert!(ans.answered);
        assert!(
            ans.text.contains("Paris"),
            "f16 should still answer 'Paris', got {:?}",
            ans.text
        );

        let big: Vec<String> = (0..8)
            .map(|i| format!("Fact {i}: the quick brown fox jumps over the lazy dog."))
            .collect();
        // Warm each backend first — the first call of a given dtype on Metal
        // pays one-time shader compilation, which would otherwise dwarf the
        // measurement. After warmup the comparison is fair.
        let _ = f32_be.time_prefill(&big);
        let _ = f16_be.time_prefill(&big);
        let t32 = f32_be.time_prefill(&big).expect("f32 prefill");
        let t16 = f16_be.time_prefill(&big).expect("f16 prefill");
        // At this tiny scale prefill is dispatch-bound, so we do NOT assert f16
        // is faster — its real win is memory (half the resident weights). We
        // only assert it answers correctly; the times are reported for context.
        eprintln!(
            "prefill (warm) — f32: {:?}, f16: {:?}; f16 answer: {:?}",
            t32, t16, ans.text
        );
    }

    /// A *larger* model (Qwen2.5-1.5B, 28 layers) loads + answers via the same
    /// config-driven path, **at bf16** — half precision with f32's exponent
    /// range (the model's native dtype), which keeps the deep model stable where
    /// f16 overflowed into garbage.
    #[tokio::test]
    #[ignore = "downloads Qwen2.5-1.5B (~3GB) + runs a real forward"]
    async fn candle_larger_model_loads_and_answers() {
        let be = QwenCandleBackend::load_repo(
            "Qwen/Qwen2.5-1.5B-Instruct",
            best_device(),
            Precision::BF16,
        )
        .expect("load qwen 1.5b");
        eprintln!(
            "candle backend: {} / {} / {} layers",
            be.device_label(),
            be.precision_label(),
            be.n_layers()
        );
        let ctx = vec!["Paris is the capital of France.".to_string()];
        let ans = be
            .answer(
                &be.compile_blob(&ctx).await,
                "\nQuestion: What is the capital of France?\nAnswer:",
            )
            .await;
        assert!(ans.answered);
        assert!(
            ans.text.contains("Paris"),
            "1.5B should answer 'Paris', got {:?}",
            ans.text
        );
        eprintln!("qwen 1.5b answer: {:?}", ans.text);
    }

    /// Real GPU-vs-CPU prefill speedup for the **1.5B**, the way you'd actually
    /// deploy it: **GPU bf16 vs CPU f32**. candle's CPU backend has no bf16
    /// matmul kernel, so f32 *is* the CPU baseline — and bf16 is the right GPU
    /// precision for a deep model (f16 overflows). Same model + identical code on
    /// `Device::new_metal` vs `Device::Cpu`; both warmed before timing. The
    /// precisions differ by necessity, so the number is labeled as such (not a
    /// same-precision microbenchmark — the 0.5B test covers f32-vs-f32). Captured
    /// for the README.
    #[tokio::test]
    #[ignore = "loads Qwen2.5-1.5B on GPU(bf16)+CPU(f32) (~9GB) + a slow CPU prefill"]
    async fn candle_larger_model_gpu_vs_cpu_speedup() {
        let repo = "Qwen/Qwen2.5-1.5B-Instruct";
        let gpu =
            QwenCandleBackend::load_repo(repo, best_device(), Precision::BF16).expect("gpu 1.5b");
        let cpu =
            QwenCandleBackend::load_repo(repo, Device::Cpu, Precision::F32).expect("cpu 1.5b");
        let big: Vec<String> = (0..8)
            .map(|i| format!("Fact {i}: the quick brown fox jumps over the lazy dog."))
            .collect();
        // Warm both (Metal shader compile / first-call costs) before timing.
        let _ = gpu.time_prefill(&big);
        let _ = cpu.time_prefill(&big);
        let g = gpu.time_prefill(&big).expect("gpu prefill");
        let c = cpu.time_prefill(&big).expect("cpu prefill");
        let ratio = c.as_secs_f64() / g.as_secs_f64().max(1e-9);
        eprintln!(
            "1.5B prefill — {} bf16: {:?}, cpu f32: {:?} ({:.1}x)",
            gpu.device_label(),
            g,
            c,
            ratio
        );
        assert!(g < c, "GPU prefill should beat CPU: gpu={g:?} cpu={c:?}");
    }
}
