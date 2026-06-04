//! # mneme-exchange — certified skill exchange (Phase 13)
//!
//! A gated [`mneme_core::PolicyArtifact`] is a portable *unit of certified
//! competence*. This crate makes it shippable: export one as a signed
//! **certificate** (artifact + its canary suite + the issuer's `EvalReport`),
//! and let another instance *import* it — but only after **re-running the gate
//! locally** (Hard Rule #1). The shipped report is a hint, never an
//! authorization: the importer never trusts it, it re-derives its own.
//!
//! Why this is a moat: nobody else has a *gated unit of competence* to
//! certify. Without `is_committable()` an imported "skill" is just unverified
//! text. With it, competence becomes a transferable, re-verifiable artifact —
//! the basis for a trust-but-verify marketplace with network effects.
//!
//! ## The two halves
//!
//! 1. **Export** ([`export`]) — wrap an artifact + canaries + the issuer's
//!    `EvalReport` into a [`SkillCertificate`] and sign it ([`Signer`]).
//!    Refuses to certify an artifact whose own report isn't committable: you
//!    can't export competence you never had.
//! 2. **Import** ([`import`]) — *trust but verify*:
//!    - verify the signature (a tampered certificate is rejected outright);
//!    - **re-run the certificate's canaries locally** ([`CanaryRunner`]) to
//!      build a *fresh* `EvalReport` — the shipped report is ignored;
//!    - apply the importer's **own** [`mneme_procedural::EvalGates`]; the skill
//!      activates *only if B's gate passes*. B can be stricter than A.
//!
//! ## Hard-rule posture
//!
//! - **#1 (gate before activation):** import re-runs the gate on a locally-
//!   derived report; a non-committable result never activates. The shipped
//!   report cannot bypass this — it isn't consulted in the decision.
//! - **#7 (swappable seam):** [`Signer`] and [`CanaryRunner`] are traits;
//!   [`FakeSigner`] / [`FakeCanaryRunner`] keep tests hermetic. A real build
//!   wires ed25519 + the procedural executor.
//!
//! ## Known limitation (don't pretend it's solved)
//!
//! [`FakeSigner`] is a dependency-free keyed digest (FNV/SplitMix-style), not
//! a real asymmetric signature — enough to prove tamper-rejection + the
//! verify-before-activate flow under test. Production swaps in ed25519 behind
//! the same [`Signer`] trait; `TODO(phase-13)`.

mod certificate;
mod import;

pub use certificate::{
    export, ExportError, FakeSigner, SignatureError, SignedBytes, Signer, SkillCertificate,
};
pub use import::{
    import, CanaryRunner, FakeCanaryRunner, ImportError, ImportOutcome, LocalEvaluation,
};
