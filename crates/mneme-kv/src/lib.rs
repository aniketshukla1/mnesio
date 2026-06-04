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
//! ([`KvBackend`], Hard Rule #7). [`FakeKvBackend`] is deterministic and
//! dependency-free — it models a KV blob as the bag of (sealed-then-opened)
//! memory texts and answers by lookup with a low simulated latency, which is
//! enough to exercise every reconciliation. A real backend over an
//! open-weights model's KV cache is `TODO(phase-12)` behind the same trait —
//! see the crate README. We do not claim the tensor half is done; we claim the
//! reconciliations that make KV memory *shippable* are proven.
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

pub use cartridge::{
    compile, Cartridge, CartridgeKey, CartridgeStatus, CompileError, FakeKvBackend, KvAnswer,
    KvBackend, SealedMemory,
};
pub use store::{ActivateError, CartridgeStore};
