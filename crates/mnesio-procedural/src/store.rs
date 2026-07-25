//! [`ProceduralStore`] — the active-version registry for
//! [`PolicyArtifact`]s.
//!
//! Hard Rule #4: the event log is the single system of record. The
//! store is derived state — a `HashMap<Id, PolicyArtifact>` keyed by
//! artifact id that always holds the **latest committed version** of
//! every artifact. It is rebuilt on startup by replaying the log, and
//! kept current by absorbing each new event as it lands.
//!
//! ## What counts as "active"
//!
//! For a given artifact id, the active version is the artifact carried
//! by the most recent `ProceduralCommitted` whose proposal listed it.
//! In the GEPA-style flow that's exactly one candidate per commit (the
//! Pareto winner) — the proposal carries K candidates but only the
//! winner ships.
//!
//! `ProceduralProposed` events do NOT update active state — proposals
//! are speculative until the gate verdict comes through as
//! `ProceduralCommitted` or `ProceduralRejected`. Replay applies
//! commits, ignores proposals + rejections (those exist for the audit
//! log + dashboard only).
//!
//! ## Concurrency
//!
//! The store is `Send + Sync` via an internal `RwLock`. Many readers
//! (snapshot endpoints, the executor pipeline) can read concurrently;
//! writers (the procedural worker's commit path) serialize. The
//! single-writer invariant is enforced one level up — the worker is
//! the only thing that mutates active state.

use mnesio_core::entity::PolicyArtifact;
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::types::{ArtifactRef, ProposalId, Scope};
use mnesio_core::{EventLog, MnesioError};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Active-version registry. Cheap to clone (it's an `Arc` over the
/// inner state) so the worker, the dashboard handler, and tests can
/// share one instance.
#[derive(Clone, Default)]
pub struct ProceduralStore {
    inner: Arc<RwLock<StoreInner>>,
}

#[derive(Default)]
struct StoreInner {
    /// Active version per artifact id.
    active: HashMap<ArtifactRef, PolicyArtifact>,
    /// `ProposalId → ArtifactRef` map. Lets `absorb` find which active
    /// version a `ProceduralCommitted` refers to without re-walking the
    /// log: `ProceduralProposed` records `(ProposalId, [candidates])`
    /// up front, so when the committed event arrives the worker can
    /// look up which proposal was selected and which candidate from
    /// that proposal won.
    ///
    /// Stored as `(artifact_ref, candidate_idx)` — the worker picks
    /// the candidate when emitting the commit.
    proposal_winners: HashMap<ProposalId, (ArtifactRef, PolicyArtifact)>,
}

impl ProceduralStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replay the entire log to rebuild active state. Idempotent —
    /// events that don't change procedural state are no-ops.
    pub async fn replay(&self, log: &dyn EventLog) -> Result<(), MnesioError> {
        let entries = log.read_from(None).await?;
        for entry in &entries {
            self.absorb(entry).await;
        }
        Ok(())
    }

    /// Update the store from a single log entry. Public so the worker
    /// can drive it directly during the tail loop, and so tests can
    /// fabricate state without going through `replay`.
    pub async fn absorb(&self, entry: &LogEntry) {
        match &entry.event {
            // Proposal event carries the full candidate list — stash
            // it so a follow-up `ProceduralCommitted` can pick the
            // winner without re-walking the log. We can't decide the
            // winner from the proposal alone (the gate runs *after*
            // proposal emission), so we keep ALL candidates indexed by
            // artifact-id-then-position and let the committed event
            // disambiguate.
            //
            // For simplicity (Slice C) we cache only the first
            // candidate — the worker's `commit` path enforces "winner
            // is always index 0" by re-ordering before emit. A future
            // slice with true Pareto selection will need to carry the
            // winner index in the committed event itself.
            Event::ProceduralProposed {
                proposal,
                artifacts,
            } => {
                if let Some(first) = artifacts.first() {
                    let aref = ArtifactRef(first.id);
                    self.inner
                        .write()
                        .await
                        .proposal_winners
                        .insert(*proposal, (aref, first.clone()));
                }
            }
            Event::ProceduralCommitted { proposal, .. } => {
                let mut g = self.inner.write().await;
                if let Some((aref, artifact)) = g.proposal_winners.remove(proposal) {
                    g.active.insert(aref, artifact);
                }
            }
            Event::ProceduralRejected { proposal, .. } => {
                // Drop the cached candidate(s) — no commit landed.
                self.inner.write().await.proposal_winners.remove(proposal);
            }
            _ => {}
        }
    }

    /// Snapshot of all active artifacts. Cheap clone — used by the
    /// dashboard handler and the worker's "pick something to compile"
    /// pass.
    pub async fn all(&self) -> Vec<PolicyArtifact> {
        self.inner.read().await.active.values().cloned().collect()
    }

    /// All active artifacts in the given scope. Hard Rule #3 — procedural
    /// learning is scope-isolated.
    pub async fn all_in_scope(&self, scope: &Scope) -> Vec<PolicyArtifact> {
        self.inner
            .read()
            .await
            .active
            .values()
            .filter(|a| a.scope == *scope)
            .cloned()
            .collect()
    }

    /// Active version for a specific artifact id, if any. Returns
    /// `None` when the id was never seeded or is currently mid-rewrite
    /// (the proposal landed but the commit hasn't yet).
    pub async fn get(&self, aref: ArtifactRef) -> Option<PolicyArtifact> {
        self.inner.read().await.active.get(&aref).cloned()
    }

    /// Number of distinct active artifacts. Cheap.
    pub async fn len(&self) -> usize {
        self.inner.read().await.active.len()
    }

    /// True iff `len() == 0` — exposed for clippy's
    /// `len-without-is-empty` lint and because some callers prefer the
    /// negative form.
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::entity::ArtifactKind;
    use mnesio_core::event::EvalReport;
    use mnesio_core::types::{new_id, BiTemporal, Id};
    use std::sync::Mutex;

    /// In-process log just for these tests — same pattern as evolve's
    /// worker_tests. Not exported; the production log is fjall.
    struct MemoryLog {
        entries: Mutex<Vec<LogEntry>>,
    }
    impl MemoryLog {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                entries: Mutex::new(Vec::new()),
            })
        }
    }
    #[async_trait::async_trait]
    impl EventLog for MemoryLog {
        async fn append(&self, event: Event) -> Result<Id, MnesioError> {
            let id = new_id();
            self.entries.lock().unwrap().push(LogEntry { id, event });
            Ok(id)
        }
        async fn read_from(&self, after: Option<Id>) -> Result<Vec<LogEntry>, MnesioError> {
            let g = self.entries.lock().unwrap();
            Ok(match after {
                None => g.clone(),
                Some(id) => g.iter().filter(|e| e.id > id).cloned().collect(),
            })
        }
    }

    fn artifact(version: u32, body: &str) -> PolicyArtifact {
        PolicyArtifact {
            id: new_id(),
            version,
            scope: Scope::global("t"),
            kind: ArtifactKind::SystemPrompt { body: body.into() },
            canaries: Vec::new(),
            time: BiTemporal::now(),
        }
    }

    fn good_report() -> EvalReport {
        EvalReport {
            canaries_passed: 0,
            canaries_total: 0,
            replay_success_rate: 1.0,
            safety_probe_passed: true,
            objective_delta: 0.0,
            judges_consulted: 2,
        }
    }

    #[tokio::test]
    async fn proposal_then_commit_makes_artifact_active() {
        let log = MemoryLog::new();
        let store = ProceduralStore::new();
        let a = artifact(2, "improved body");
        let prop_id = ProposalId(new_id());

        log.append(Event::ProceduralProposed {
            proposal: prop_id,
            artifacts: vec![a.clone()],
        })
        .await
        .unwrap();
        log.append(Event::ProceduralCommitted {
            proposal: prop_id,
            report: good_report(),
        })
        .await
        .unwrap();

        store.replay(log.as_ref()).await.unwrap();
        let active = store.get(ArtifactRef(a.id)).await;
        assert!(
            active.is_some(),
            "committed proposal must produce active version"
        );
        assert_eq!(active.unwrap().version, 2);
    }

    #[tokio::test]
    async fn proposal_then_reject_leaves_no_active_version() {
        let log = MemoryLog::new();
        let store = ProceduralStore::new();
        let a = artifact(2, "bad body");
        let prop_id = ProposalId(new_id());

        log.append(Event::ProceduralProposed {
            proposal: prop_id,
            artifacts: vec![a.clone()],
        })
        .await
        .unwrap();
        log.append(Event::ProceduralRejected {
            proposal: prop_id,
            reason: "safety probe failed".into(),
        })
        .await
        .unwrap();

        store.replay(log.as_ref()).await.unwrap();
        assert!(store.is_empty().await);
        assert!(store.get(ArtifactRef(a.id)).await.is_none());
    }

    #[tokio::test]
    async fn later_commit_supersedes_earlier_active_version() {
        let log = MemoryLog::new();
        let store = ProceduralStore::new();
        let mut a = artifact(1, "v1");
        let aref = ArtifactRef(a.id);
        let p1 = ProposalId(new_id());
        let p2 = ProposalId(new_id());

        // v1 commit
        log.append(Event::ProceduralProposed {
            proposal: p1,
            artifacts: vec![a.clone()],
        })
        .await
        .unwrap();
        log.append(Event::ProceduralCommitted {
            proposal: p1,
            report: good_report(),
        })
        .await
        .unwrap();

        // v2 commit — same artifact id, bumped version
        a.version = 2;
        a.kind = ArtifactKind::SystemPrompt { body: "v2".into() };
        log.append(Event::ProceduralProposed {
            proposal: p2,
            artifacts: vec![a.clone()],
        })
        .await
        .unwrap();
        log.append(Event::ProceduralCommitted {
            proposal: p2,
            report: good_report(),
        })
        .await
        .unwrap();

        store.replay(log.as_ref()).await.unwrap();
        let active = store.get(aref).await.unwrap();
        assert_eq!(active.version, 2);
        match active.kind {
            ArtifactKind::SystemPrompt { body } => assert_eq!(body, "v2"),
            _ => panic!("wrong kind"),
        }
    }

    #[tokio::test]
    async fn all_in_scope_filters_by_scope() {
        let log = MemoryLog::new();
        let store = ProceduralStore::new();
        let mut a1 = artifact(1, "t1");
        a1.scope = Scope::global("tenant1");
        let mut a2 = artifact(1, "t2");
        a2.scope = Scope::global("tenant2");

        for a in [&a1, &a2] {
            let p = ProposalId(new_id());
            log.append(Event::ProceduralProposed {
                proposal: p,
                artifacts: vec![a.clone()],
            })
            .await
            .unwrap();
            log.append(Event::ProceduralCommitted {
                proposal: p,
                report: good_report(),
            })
            .await
            .unwrap();
        }
        store.replay(log.as_ref()).await.unwrap();
        assert_eq!(store.all_in_scope(&Scope::global("tenant1")).await.len(), 1);
        assert_eq!(store.all_in_scope(&Scope::global("tenant2")).await.len(), 1);
        assert_eq!(store.all().await.len(), 2);
    }

    #[tokio::test]
    async fn replay_is_idempotent() {
        let log = MemoryLog::new();
        let store = ProceduralStore::new();
        let a = artifact(1, "body");
        let p = ProposalId(new_id());
        log.append(Event::ProceduralProposed {
            proposal: p,
            artifacts: vec![a.clone()],
        })
        .await
        .unwrap();
        log.append(Event::ProceduralCommitted {
            proposal: p,
            report: good_report(),
        })
        .await
        .unwrap();

        store.replay(log.as_ref()).await.unwrap();
        store.replay(log.as_ref()).await.unwrap();
        assert_eq!(store.len().await, 1, "second replay must not double-count");
    }

    #[tokio::test]
    async fn unrelated_events_do_not_pollute_active_state() {
        // The store must ignore memory events entirely — they belong
        // to a different materialised view.
        let log = MemoryLog::new();
        let store = ProceduralStore::new();
        // Just append one memory event and one source-invalidated;
        // store should stay empty.
        use mnesio_core::entity::{Memory, Provenance};
        use mnesio_core::types::{MemoryRef, SourceRef};
        let mem = Memory {
            id: new_id(),
            scope: Scope::global("t"),
            content: "x".into(),
            keywords: vec![],
            tags: vec![],
            context: String::new(),
            embedding: None,
            links: vec![],
            parent: None,
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: Provenance::default(),
            source: None,
            position: None,
        };
        log.append(Event::MemoryWritten(mem)).await.unwrap();
        log.append(Event::SourceInvalidated {
            id: SourceRef(new_id()),
            reason: "x".into(),
        })
        .await
        .unwrap();
        let _ = MemoryRef(new_id());
        store.replay(log.as_ref()).await.unwrap();
        assert!(store.is_empty().await);
    }
}
