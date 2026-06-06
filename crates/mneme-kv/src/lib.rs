//! # mneme-kv — gated KV cartridges (Phase 12, the moonshot)
//!
//! A KV "cartridge" is a reusable chunk of a transformer's key/value attention
//! state, compiled from a hot subset of memory so the model can attend to that
//! knowledge at inference without re-reading it as text. KV-cache memory is an
//! old idea; nobody productizes it as a *general memory layer* because the blob
//! is opaque, unversioned, unauditable, and un-erasable.
//!
//! mneme's move is to treat a cartridge as **just another materialized view of
//! the log** (Hard Rule #4): keyed by `(model_id, quant, rope_config)` + the
//! log prefix it was built from, and rebuildable by replay. That single reframe
//! dissolves all four blockers, and this crate proves each one *under test*:
//!
//! 1. **Versioning** — a cartridge is keyed by [`CartridgeKey`]; a model swap
//!    or a newer log prefix yields a new version, never an in-place mutation.
//! 2. **Gate before activation** — a compiled cartridge only goes active if it
//!    passes [`mneme_core::EvalReport::is_committable`] (Hard Rule #1). A blob
//!    that fails the gate is inert.
//! 3. **Audit** — the cartridge records exactly which memories it was compiled
//!    from + the key + the gate report; nothing about it is unexplainable.
//! 4. **Crypto-shred by recompile** — the cartridge is compiled from
//!    *key-sealed* memories ([`mneme_privacy::Keyring`]). Forget a subject
//!    (destroy their key) and recompile → the rebuilt tensor can no longer
//!    reconstruct the erased content (Hard Rule #2: the log is the truth, the
//!    cartridge is derived).
//!
//! ## What is and isn't real here
//!
//! The *substrate* is real and tested: the view semantics, the gate, the
//! versioning, and the erasure reconciliation. The **tensor backend is a seam**
//! ([`KvBackend`], Hard Rule #7). Two implementations ship:
//! - [`FakeKvBackend`] — deterministic, models the blob as the bag of
//!   (sealed-then-opened) texts and answers by lookup; fastest for exercising
//!   every reconciliation in tests.
//! - [`TensorKvBackend`] — a **real-tensor** backend: `compile_blob` runs each
//!   memory through token-embedding + linear K/V projections and stores the
//!   resulting **K/V tensors** (`f32`, `[members][seq][d_model]`) as the blob;
//!   `answer` retrieves by **multi-head scaled dot-product attention** over
//!   that cache. The blob is real KV-cache bytes whose size tracks token count,
//!   so erasure genuinely shrinks it. Still dependency-free and pure Rust.
//!
//! - [`PretrainedKvBackend`] *(feature `pretrained-kv`)* — removes the
//!   "untrained" caveat: it loads **real GPT-2 open weights** (token + position
//!   embeddings and the layer-0 `c_attn` Q/K/V projection), runs the actual
//!   forward to build the K/V cache, and retrieves over GPT-2's *learned*
//!   representation. Weights download once (cached) via `hf-hub`; all crypto/ML
//!   deps are feature-gated so the default build stays light + offline.
//!
//! - [`GenerativeKvBackend`] *(feature `generative-kv`)* — the deepest lift:
//!   the cartridge **is** GPT-2's key/value cache. `compile_blob` runs the
//!   **full 12-layer forward** over the (post-shred) context to prefill a KV
//!   cache for every layer; `answer` restores that cache and *generates* the
//!   query continuation, each new token attending over the cartridge. Proven by
//!   a self-consistency oracle: generating from the cartridge is **token-
//!   identical** to processing the full `context ++ query` prompt from scratch
//!   (KV caching is exact), so the cartridge is a faithful — and cheaper —
//!   substitute that skips the prefix recompute on every query. Erasure is
//!   reconciled the same way: shred a subject's key, recompile, and the rebuilt
//!   cache can no longer *generate* the erased fact. [`Quant::Q8`] makes the
//!   `quant` field of [`CartridgeKey`] real — per-row int8 shrinks the cartridge
//!   ~4× (live: 1.18 MB → 0.30 MB) while generating the same answer.
//!
//! - [`QwenKvBackend`] *(feature `qwen-kv`)* — the same cartridge path on a
//!   **2024 architecture**, the honest answer to "why GPT-2, can't we use a more
//!   advanced model?". Loads **Qwen2.5-0.5B-Instruct** (RMSNorm, RoPE,
//!   grouped-query attention, SwiGLU, bf16, 24 layers, instruction-tuned) and
//!   hand-rolls its forward in pure Rust so the cartridge owns the KV cache —
//!   which **Ollama's black-box API cannot expose**, the reason the cartridge
//!   can't be built on a remote text endpoint. Live: the cartridge answers
//!   "capital of France" → "Paris", token-identical to the full prompt (the
//!   RoPE/GQA cache is exact), and a shred-recompile drops the fact from
//!   generation.
//!
//! - [`QwenCandleBackend`] *(features `candle-kv` + `metal`)* — the **GPU**
//!   version of the same cartridge path, the Qwen2 forward ported to
//!   [`candle_core`] tensors on Apple Metal. Config-driven (architecture from
//!   the repo's `config.json`, so 0.5B/1.5B/3B/7B load with no code change) and
//!   precision-selectable ([`CandlePrecision::F16`] halves resident weights).
//!   The pure-Rust `qwen-kv` backend is its semantic oracle; we assert candle's
//!   *own* self-consistency (cartridge == full-prompt, same device + dtype), not
//!   bit-identity across CPU/GPU. Live on an M1 Pro: ~107× faster warm prefill
//!   than CPU on identical code, answers "Paris", erasure holds.
//!
//! That closes the last open piece of Phase 12: the cartridge is no longer a
//! retrieval index over a cache — it is the cache the model generates from, on
//! CPU or GPU, across model sizes and precisions.
//!
//! ## Hard-rule posture
//!
//! - **#1 (gate):** [`CartridgeStore::activate`] refuses a non-committable
//!   cartridge.
//! - **#2 / #4 (log is truth, never overwrite):** cartridges are derived +
//!   versioned; erasure is reconciled by recompiling from the (post-forget)
//!   log, never by editing a blob in place.
//! - **#5 (fast write path):** compilation is offline, on demand — never on a
//!   write.
//! - **#7 (swappable seam):** the tensor backend is a trait.

#[cfg(feature = "candle-kv")]
mod candle;
mod cartridge;
#[cfg(feature = "generative-kv")]
mod generative;
#[cfg(feature = "pretrained-kv")]
mod pretrained;
#[cfg(feature = "qwen-kv")]
mod qwen;
mod store;
mod tensor;

#[cfg(feature = "candle-kv")]
pub use candle::Precision as CandlePrecision;
#[cfg(feature = "candle-kv")]
pub use candle::{QwenCandleBackend, DEFAULT_REPO as CANDLE_DEFAULT_REPO};
pub use cartridge::{
    compile, Cartridge, CartridgeKey, CartridgeStatus, CompileError, FakeKvBackend, KvAnswer,
    KvBackend, SealedMemory,
};
#[cfg(feature = "generative-kv")]
pub use generative::{GenerativeKvBackend, Quant};
#[cfg(feature = "pretrained-kv")]
pub use pretrained::PretrainedKvBackend;
#[cfg(feature = "qwen-kv")]
pub use qwen::QwenKvBackend;
pub use store::{ActivateError, CartridgeStore};
pub use tensor::{TensorConfig, TensorKvBackend};
