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
///
/// At scale (many runtimes, large blobs) two things must stay bounded: the
/// total bytes held by *active* cartridges, and the audit `history`. Both are
/// off by default ([`new`](Self::new) = unbounded, preserving the original
/// behavior); opt in with [`with_budget`](Self::with_budget) /
/// [`with_max_history`](Self::with_max_history).
#[derive(Default)]
pub struct CartridgeStore {
    inner: RwLock<Inner>,
    /// Cap on total bytes held by *active* cartridge blobs across all runtimes.
    /// When an activation pushes the total over budget, least-recently-used
    /// active runtimes are evicted (never the one just activated) until back
    /// under budget. `None` = unbounded.
    max_active_bytes: Option<usize>,
    /// Cap on the audit `history` length (ring buffer of the most recent
    /// entries). `None` = keep everything.
    max_history: Option<usize>,
}

#[derive(Default)]
struct Inner {
    /// `runtime-id → active cartridge`. The runtime id is
    /// `model_id|quant|rope_config` (log head excluded — a newer head is a new
    /// *version* of the same runtime's cartridge).
    active: HashMap<String, Cartridge>,
    /// Every cartridge ever activated, rejected, or evicted, newest last — the
    /// audit log (bounded by `max_history` if set).
    history: Vec<Cartridge>,
    /// Running total of bytes held by `active` blobs (kept in sync on every
    /// insert/demote/evict so the budget check is O(1)).
    active_bytes: usize,
    /// Monotonic access clock — bumped on activate and on serve (`active_for`),
    /// so the smallest `last_used` tick is the least-recently-used runtime.
    clock: u64,
    /// `runtime-id → last access tick`. Drives LRU eviction.
    last_used: HashMap<String, u64>,
    /// `runtime-id → highest version ever seen`. Tracked separately from
    /// `history` so version monotonicity survives history pruning.
    max_version: HashMap<String, u32>,
}

fn runtime_id(key: &CartridgeKey) -> String {
    format!("{}|{}|{}", key.model_id, key.quant, key.rope_config)
}

/// Push a cartridge into the audit history, updating the per-runtime
/// `max_version` and enforcing the optional ring-buffer cap.
fn push_history(inner: &mut Inner, c: Cartridge, max_history: Option<usize>) {
    let rt = runtime_id(&c.key);
    let slot = inner.max_version.entry(rt).or_insert(0);
    *slot = (*slot).max(c.version);
    inner.history.push(c);
    if let Some(h) = max_history {
        if h == 0 {
            inner.history.clear();
        } else if inner.history.len() > h {
            let excess = inner.history.len() - h;
            inner.history.drain(0..excess);
        }
    }
}

/// Evict least-recently-used active runtimes (never `keep_rt`) until the active
/// byte total is within `budget`. Evicted cartridges are marked
/// [`CartridgeStatus::Evicted`] and preserved in history.
fn evict_until_under(inner: &mut Inner, budget: usize, keep_rt: &str, max_history: Option<usize>) {
    while inner.active_bytes > budget && inner.active.len() > 1 {
        let victim = inner
            .active
            .keys()
            .filter(|k| k.as_str() != keep_rt)
            .min_by_key(|k| inner.last_used.get(*k).copied().unwrap_or(0))
            .cloned();
        let Some(victim) = victim else { break };
        if let Some(mut c) = inner.active.remove(&victim) {
            inner.active_bytes = inner.active_bytes.saturating_sub(c.blob.len());
            inner.last_used.remove(&victim);
            c.status = CartridgeStatus::Evicted;
            push_history(inner, c, max_history);
        }
    }
}

impl CartridgeStore {
    /// Unbounded store (no byte budget, no history cap) — the default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store that caps total *active* blob bytes at `max_active_bytes`,
    /// LRU-evicting active runtimes that push the total over budget.
    pub fn with_budget(max_active_bytes: usize) -> Self {
        Self {
            max_active_bytes: Some(max_active_bytes),
            ..Self::default()
        }
    }

    /// Cap the audit `history` at the `max_history` most-recent entries
    /// (ring buffer). Chainable with [`with_budget`](Self::with_budget).
    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history = Some(max_history);
        self
    }

    /// Try to activate `cartridge`, gated by `report`.
    ///
    /// On a committable report the cartridge is marked [`CartridgeStatus::Active`]
    /// and becomes the active version for its runtime, superseding any previous
    /// active cartridge there. On a non-committable report the cartridge is
    /// marked [`CartridgeStatus::Rejected`], recorded for audit, and the
    /// previously-active cartridge (if any) is left untouched — a failed gate
    /// can never degrade what's already serving (Hard Rule #1).
    ///
    /// After a successful activation, if a byte budget is set and exceeded,
    /// least-recently-used active runtimes (never the one just activated) are
    /// evicted until the total is back under budget.
    pub fn activate(
        &self,
        mut cartridge: Cartridge,
        report: &EvalReport,
    ) -> Result<Cartridge, ActivateError> {
        let mut inner = self.inner.write().unwrap();
        if !report.is_committable() {
            cartridge.status = CartridgeStatus::Rejected;
            push_history(&mut inner, cartridge.clone(), self.max_history);
            return Err(ActivateError::GateFailed {
                reason: gate_reason(report),
            });
        }
        cartridge.status = CartridgeStatus::Active;
        let rt = runtime_id(&cartridge.key);
        let new_bytes = cartridge.blob.len();
        // Demote the prior active version to Rejected-in-history (superseded).
        if let Some(mut prev) = inner.active.remove(&rt) {
            inner.active_bytes = inner.active_bytes.saturating_sub(prev.blob.len());
            prev.status = CartridgeStatus::Rejected;
            push_history(&mut inner, prev, self.max_history);
        }
        inner.active_bytes += new_bytes;
        inner.active.insert(rt.clone(), cartridge.clone());
        inner.clock += 1;
        let tick = inner.clock;
        inner.last_used.insert(rt.clone(), tick);
        push_history(&mut inner, cartridge.clone(), self.max_history);
        if let Some(budget) = self.max_active_bytes {
            evict_until_under(&mut inner, budget, &rt, self.max_history);
        }
        Ok(cartridge)
    }

    /// The active cartridge for a runtime, if any. Serving a cartridge counts
    /// as an access — it bumps the runtime's LRU recency so a hot runtime is
    /// not evicted in favor of a cold one.
    pub fn active_for(&self, key: &CartridgeKey) -> Option<Cartridge> {
        let mut inner = self.inner.write().unwrap();
        let rt = runtime_id(key);
        let hit = inner.active.get(&rt).cloned();
        if hit.is_some() {
            inner.clock += 1;
            let tick = inner.clock;
            inner.last_used.insert(rt, tick);
        }
        hit
    }

    /// Number of currently-active cartridges (one per runtime).
    pub fn active_count(&self) -> usize {
        self.inner.read().unwrap().active.len()
    }

    /// Total bytes held by active cartridge blobs (what the budget bounds).
    pub fn active_bytes(&self) -> usize {
        self.inner.read().unwrap().active_bytes
    }

    /// Total cartridges currently retained in the audit trail (active +
    /// rejected + superseded + evicted), bounded by `max_history` if set.
    pub fn history_len(&self) -> usize {
        self.inner.read().unwrap().history.len()
    }

    /// Next version number to use for a runtime (1 + highest ever seen).
    /// Tracked independently of `history` so it stays monotonic even when the
    /// history ring buffer has pruned old versions.
    pub fn next_version(&self, key: &CartridgeKey) -> u32 {
        let inner = self.inner.read().unwrap();
        inner
            .max_version
            .get(&runtime_id(key))
            .copied()
            .unwrap_or(0)
            + 1
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

    fn key_for(model: &str) -> CartridgeKey {
        CartridgeKey::new(model, "q8", "rope-default")
    }

    async fn make_for(
        kr: &Keyring,
        model: &str,
        version: u32,
        texts: &[(&str, &str)],
    ) -> Cartridge {
        let backend = FakeKvBackend::new(model);
        let members: Vec<SealedMemory> = texts
            .iter()
            .map(|(subject, text)| SealedMemory {
                id: mref(),
                sealed: kr.seal(subject, text.as_bytes()).unwrap(),
            })
            .collect();
        compile(&backend, kr, key_for(model), version, &members)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn budget_evicts_least_recently_used() {
        let kr = Keyring::new();
        // Each blob is the JSON of one big text, so two fit a budget but three
        // don't — forcing exactly one eviction.
        let big = "x".repeat(400);
        let a = make_for(&kr, "model-a", 1, &[("a", &big)]).await;
        let b = make_for(&kr, "model-b", 1, &[("b", &big)]).await;
        let c = make_for(&kr, "model-c", 1, &[("c", &big)]).await;
        let one = a.blob.len();
        let store = CartridgeStore::with_budget(one * 2 + 10);

        store.activate(a, &pass()).unwrap();
        store.activate(b, &pass()).unwrap();
        // Serve A so B becomes the least-recently-used runtime.
        assert!(store.active_for(&key_for("model-a")).is_some());
        // C overflows the budget → LRU eviction drops B, keeps A + C.
        store.activate(c, &pass()).unwrap();

        assert_eq!(store.active_count(), 2);
        assert!(
            store.active_for(&key_for("model-a")).is_some(),
            "A kept — recently served"
        );
        assert!(
            store.active_for(&key_for("model-c")).is_some(),
            "C kept — just activated, never evicted"
        );
        assert!(
            store.active_for(&key_for("model-b")).is_none(),
            "B evicted — least recently used"
        );
        assert!(store.active_bytes() <= one * 2 + 10, "back under budget");
    }

    #[tokio::test]
    async fn history_cap_bounds_audit_and_keeps_versions_monotonic() {
        let kr = Keyring::new();
        let store = CartridgeStore::with_budget(usize::MAX).with_max_history(3);
        for v in 1..=6u32 {
            let c = make_for(&kr, "model-x", v, &[("x", "x likes tea")]).await;
            store.activate(c, &pass()).unwrap();
        }
        assert!(
            store.history_len() <= 3,
            "history bounded to 3, got {}",
            store.history_len()
        );
        // Latest version still serves despite history pruning…
        assert_eq!(store.active_for(&key_for("model-x")).unwrap().version, 6);
        // …and the version counter is monotonic across the prune.
        assert_eq!(store.next_version(&key_for("model-x")), 7);
    }

    #[tokio::test]
    async fn just_activated_is_never_evicted_even_if_over_budget() {
        let kr = Keyring::new();
        let big = "y".repeat(1000);
        // Budget smaller than a single blob: we still can't evict the only
        // (just-activated) cartridge — better to be over budget than serve nothing.
        let store = CartridgeStore::with_budget(10);
        let c = make_for(&kr, "model-z", 1, &[("z", &big)]).await;
        store.activate(c, &pass()).unwrap();
        assert_eq!(store.active_count(), 1);
        assert!(store.active_for(&key_for("model-z")).is_some());
    }
}
