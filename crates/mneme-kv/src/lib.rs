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
//! What's *not* yet done is the **pretrained weights**: `TensorKvBackend` uses
//! content-derived embeddings + fixed seeded projections (retrieval tracks
//! token overlap, not trained semantics; Q shares K's projection). Loading a
//! real open-weights model's embedding + Wq/Wk/Wv behind this same
//! `compile_blob`/`answer` path is the one remaining lift — a weights load, not
//! an architecture change. We claim the reconciliations that make KV memory
//! *shippable* are proven, and the tensor math is now real.
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

mod cartridge;
mod store;
mod tensor;

pub use cartridge::{
    compile, Cartridge, CartridgeKey, CartridgeStatus, CompileError, FakeKvBackend, KvAnswer,
    KvBackend, SealedMemory,
};
pub use store::{ActivateError, CartridgeStore};
pub use tensor::{TensorConfig, TensorKvBackend};
