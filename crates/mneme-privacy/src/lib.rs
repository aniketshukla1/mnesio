//! # mneme-privacy
//!
//! Two privacy capabilities that, together, let an **append-only,
//! replayable** memory log coexist with **PII minimisation** and the
//! **right to be forgotten** — the reconciliation no competitor with an
//! immutable audit log offers (see `COMPETITIVE.md`, P1#8).
//!
//! 1. **Redaction** ([`redact`]) — detect + mask PII in text *before* it
//!    is stored, so raw identifiers never enter the log in the first
//!    place. Trait-shaped ([`Redactor`]) so a heavier NER/ML detector can
//!    swap in for the built-in regex-free pattern scanner.
//!
//! 2. **Crypto-shredding** ([`keyring`]) — the GDPR-erasure trick for an
//!    append-only store: per-subject content is stored **encrypted under a
//!    subject-scoped key**. "Forgetting" a subject = **dropping that one
//!    key** ([`Keyring::forget`]). The ciphertext stays in the log
//!    (append-only, never rewritten — Hard Rule #2), but without the key
//!    it is permanently undecryptable: the subject is unreadable while the
//!    log's integrity + replayability are preserved.
//!
//! ## Why this shape
//!
//! - **Crypto is optional + behind a trait.** The default build ships no
//!   crypto dependency: [`keyring::XorCipher`] is a keystream cipher that is
//!   *correct* for the crypto-shred *protocol* (encrypt-on-write,
//!   drop-key-to-forget) and lets the whole flow be tested offline. The real,
//!   authenticated AEAD — [`keyring::ChaChaCipher`] (ChaCha20-Poly1305) — is a
//!   drop-in behind the [`keyring::Cipher`] trait, enabled with the `aead`
//!   feature (`--features aead`). The shred guarantee is identical either way
//!   — it comes from **key destruction**, not the cipher; the AEAD adds
//!   authenticated, tamper-resistant encryption for production.
//! - **Pure, no I/O.** Like `mneme-extract`, this crate plans + transforms;
//!   the host persists keys (in a KMS / secure store) and applies redaction
//!   on the write path.

pub mod keyring;
pub mod redact;

#[cfg(feature = "aead")]
pub use keyring::ChaChaCipher;
pub use keyring::{Cipher, Keyring, SealedBox, XorCipher};
pub use redact::{redact, PiiKind, RedactConfig, RedactionReport, Redactor, RegexlessRedactor};
