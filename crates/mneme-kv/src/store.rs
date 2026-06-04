//! [`CartridgeStore`] — the active-version registry for KV cartridges, with
//! gated activation (Hard Rule #1) and recompile-driven supersession.
//!
//! Mirrors the discipline of `mneme_procedural::ProceduralStore`: a compiled
//! cartridge is inert until it passes the gate; activation atomically swaps the
//! active version for a runtime; a newer version supersedes (never mutates) the
//! old one. Erasure is reconciled by recompiling from the post-forget log and
//! re-activating — the store never edits a blob in place.

use crate::cartridge::{Cartridge, CartridgeKey, CartridgeStatus};
use mneme_core::EvalReport;
use std::collections::HashMap;
use std::sync::RwLock;

/// Why activation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivateError {
    /// The gate (`EvalReport::is_committable`) said no. The cartridge is marked
    /// `Rejected` and stays inert (Hard Rule #1).
    GateFailed { reason: String },
}

impl std::fmt::Display for ActivateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivateError::GateFailed { reason } => {
                write!(f, "cartridge failed the gate: {reason}")
            }
        }
    }
}

impl std::error::Error for ActivateError {}

/// Registry of cartridges, indexed by *runtime* (model/quant/rope) so there's
/// exactly one active cartridge per runtime at a time. Keeps superseded
/// versions for audit/time-travel rather than dropping them.
#[derive(Default)]
pub struct CartridgeStore {
    inner: RwLock<Inner>,
}

#[derive(Default)]
struct Inner {
    /// `runtime-id → active cartridge`. The runtime id is
    /// `model_id|quant|rope_config` (log head excluded — a newer head is a new
    /// *version* of the same runtime's cartridge).
    active: HashMap<String, Cartridge>,
    /// Every cartridge ever activated or rejected, newest last — the audit log.
    history: Vec<Cartridge>,
}

fn runtime_id(key: &CartridgeKey) -> String {
    format!("{}|{}|{}", key.model_id, key.quant, key.rope_config)
}

impl CartridgeStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to activate `cartridge`, gated by `report`.
    ///
    /// On a committable report the cartridge is marked [`CartridgeStatus::Active`]
    /// and becomes the active version for its runtime, superseding any previous
    /// active cartridge there. On a non-committable report the cartridge is
    /// marked [`CartridgeStatus::Rejected`], recorded for audit, and the
    /// previously-active cartridge (if any) is left untouched — a failed gate
    /// can never degrade what's already serving (Hard Rule #1).
    pub fn activate(
        &self,
        mut cartridge: Cartridge,
        report: &EvalReport,
    ) -> Result<Cartridge, ActivateError> {
        let mut inner = self.inner.write().unwrap();
        if !report.is_committable() {
            cartridge.status = CartridgeStatus::Rejected;
            inner.history.push(cartridge.clone());
            return Err(ActivateError::GateFailed {
                reason: gate_reason(report),
            });
        }
        cartridge.status = CartridgeStatus::Active;
        let rt = runtime_id(&cartridge.key);
        // Demote the prior active version to Rejected-in-history (superseded).
        if let Some(mut prev) = inner.active.remove(&rt) {
            prev.status = CartridgeStatus::Rejected;
            inner.history.push(prev);
        }
        inner.active.insert(rt, cartridge.clone());
        inner.history.push(cartridge.clone());
        Ok(cartridge)
    }

    /// The active cartridge for a runtime, if any.
    pub fn active_for(&self, key: &CartridgeKey) -> Option<Cartridge> {
        self.inner
            .read()
            .unwrap()
            .active
            .get(&runtime_id(key))
            .cloned()
    }

    /// Number of currently-active cartridges (one per runtime).
    pub fn active_count(&self) -> usize {
        self.inner.read().unwrap().active.len()
    }

    /// Total cartridges ever seen (active + rejected + superseded) — the audit
    /// trail length.
    pub fn history_len(&self) -> usize {
        self.inner.read().unwrap().history.len()
    }

    /// Next version number to use for a runtime (1 + highest seen). Lets the
    /// caller produce monotonically-versioned cartridges across recompiles.
    pub fn next_version(&self, key: &CartridgeKey) -> u32 {
        let inner = self.inner.read().unwrap();
        let rt = runtime_id(key);
        let max = inner
            .history
            .iter()
            .filter(|c| runtime_id(&c.key) == rt)
            .map(|c| c.version)
            .max()
            .unwrap_or(0);
        max + 1
    }
}

/// One-line reason a report failed the gate (for `ActivateError` + dashboards).
fn gate_reason(r: &EvalReport) -> String {
    let mut reasons = Vec::new();
    if r.canaries_passed != r.canaries_total {
        reasons.push(format!(
            "canaries {}/{}",
            r.canaries_passed, r.canaries_total
        ));
    }
    if !r.safety_probe_passed {
        reasons.push("safety probe failed".to_string());
    }
    if r.objective_delta < 0.0 {
        reasons.push(format!("objective Δ {:.3} < 0", r.objective_delta));
    }
    if reasons.is_empty() {
        "non-committable".to_string()
    } else {
        reasons.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cartridge::{compile, CartridgeKey, FakeKvBackend, KvBackend, SealedMemory};
    use mneme_core::types::{new_id, MemoryRef};
    use mneme_privacy::Keyring;

    fn mref() -> MemoryRef {
        MemoryRef(new_id())
    }

    fn key() -> CartridgeKey {
        CartridgeKey::new("fake-llm-v1", "q8", "rope-default")
    }

    fn pass() -> EvalReport {
        EvalReport {
            canaries_passed: 3,
            canaries_total: 3,
            replay_success_rate: 1.0,
            safety_probe_passed: true,
            objective_delta: 0.1,
            judges_consulted: 2,
        }
    }

    fn fail_safety() -> EvalReport {
        EvalReport {
            safety_probe_passed: false,
            ..pass()
        }
    }

    async fn make_cartridge(kr: &Keyring, version: u32, texts: &[(&str, &str)]) -> Cartridge {
        let backend = FakeKvBackend::new("fake-llm-v1");
        let members: Vec<SealedMemory> = texts
            .iter()
            .map(|(subject, text)| SealedMemory {
                id: mref(),
                sealed: kr.seal(subject, text.as_bytes()).unwrap(),
            })
            .collect();
        compile(&backend, kr, key(), version, &members)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn committable_cartridge_activates() {
        let kr = Keyring::new();
        let store = CartridgeStore::new();
        let c = make_cartridge(&kr, 1, &[("alice", "alice likes tea")]).await;
        let activated = store.activate(c, &pass()).unwrap();
        assert_eq!(activated.status, CartridgeStatus::Active);
        assert_eq!(store.active_count(), 1);
        assert!(store.active_for(&key()).is_some());
    }

    #[tokio::test]
    async fn non_committable_cartridge_is_refused_and_inert() {
        // Hard Rule #1: a blob that fails the gate never serves.
        let kr = Keyring::new();
        let store = CartridgeStore::new();
        let c = make_cartridge(&kr, 1, &[("alice", "alice likes tea")]).await;
        let err = store.activate(c, &fail_safety()).unwrap_err();
        assert!(matches!(err, ActivateError::GateFailed { .. }));
        assert_eq!(store.active_count(), 0, "nothing active");
        assert_eq!(store.history_len(), 1, "rejection recorded for audit");
    }

    #[tokio::test]
    async fn failed_gate_does_not_disturb_the_serving_cartridge() {
        let kr = Keyring::new();
        let store = CartridgeStore::new();
        let good = make_cartridge(&kr, 1, &[("alice", "alice likes tea")]).await;
        store.activate(good, &pass()).unwrap();
        let bad = make_cartridge(&kr, 2, &[("alice", "alice likes tea")]).await;
        let _ = store.activate(bad, &fail_safety());
        // Still v1 serving.
        assert_eq!(store.active_for(&key()).unwrap().version, 1);
    }

    #[tokio::test]
    async fn newer_version_supersedes_older() {
        let kr = Keyring::new();
        let store = CartridgeStore::new();
        let v1 = make_cartridge(&kr, 1, &[("alice", "alice likes tea")]).await;
        store.activate(v1, &pass()).unwrap();
        let v2 = make_cartridge(
            &kr,
            store.next_version(&key()),
            &[("alice", "alice likes tea")],
        )
        .await;
        assert_eq!(v2.version, 2);
        store.activate(v2, &pass()).unwrap();
        assert_eq!(store.active_count(), 1, "still one active per runtime");
        assert_eq!(store.active_for(&key()).unwrap().version, 2);
    }

    #[tokio::test]
    async fn crypto_shred_by_recompile_then_reactivate() {
        // The headline Phase-12 reconciliation, end to end through the store.
        let kr = Keyring::new();
        let backend = FakeKvBackend::new("fake-llm-v1");
        let store = CartridgeStore::new();

        let members = vec![
            SealedMemory {
                id: mref(),
                sealed: kr.seal("alice", b"alice account number is 12345").unwrap(),
            },
            SealedMemory {
                id: mref(),
                sealed: kr.seal("bob", b"bob prefers aisle seats").unwrap(),
            },
        ];

        // v1: both subjects present, gated active, answers about alice.
        let v1 = compile(&backend, &kr, key(), 1, &members).await.unwrap();
        let v1 = store.activate(v1, &pass()).unwrap();
        assert!(
            backend
                .answer(&v1.blob, "alice account number")
                .await
                .answered
        );

        // Forget alice (destroy key), recompile, re-activate.
        kr.forget("alice");
        let v2 = compile(&backend, &kr, key(), store.next_version(&key()), &members)
            .await
            .unwrap();
        store.activate(v2, &pass()).unwrap();

        // The active cartridge can no longer reconstruct alice's content…
        let active = store.active_for(&key()).unwrap();
        assert_eq!(active.version, 2);
        assert!(
            !backend
                .answer(&active.blob, "alice account number")
                .await
                .answered,
            "forgotten subject is unrecoverable from the recompiled cartridge"
        );
        // …while bob is unaffected.
        assert!(
            backend
                .answer(&active.blob, "bob aisle seats")
                .await
                .answered
        );
    }
}
