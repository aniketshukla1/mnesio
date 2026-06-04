//! Falsification: turn a refuted probe outcome into an *invalidate-and-
//! supersede* on the log (Hard Rule #2 — never overwrite; the refuted version
//! stays, reachable by time-travel).
//!
//! Emits the **same canonical triple** the ingestion + evolution workers use,
//! so every existing materialized view (vector, BM25, graph, ACL) handles a
//! falsification with no special-casing:
//!
//! 1. `MemoryWritten(correction)` — a new bi-temporal version, `parent` =
//!    refuted memory, carrying the probe's correction text.
//! 2. `MemoryEvolved { from: refuted, to: correction }` — the lineage edge.
//! 3. `MemoryInvalidated { id: refuted, reason }` — soft-delete the old
//!    version, stamped [`FALSIFY_REASON`] so audits can tell probe-driven
//!    supersession apart from contradiction / GC.

use crate::probe::{ProbeOutcome, ProbeStatus};
use mneme_core::entity::{Memory, Provenance};
use mneme_core::event::{ChangeSet, Event};
use mneme_core::traits::EventLog;
use mneme_core::types::{new_id, BiTemporal, MemoryRef};
use mneme_core::{MnemeError, Scope};

/// Reason stamped on every probe-driven invalidation.
pub const FALSIFY_REASON: &str = "self-falsified: acceptance probe refuted the claim";

/// What a falsification pass did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FalsifyOutcome {
    /// Memories superseded (one supersede triple each).
    pub superseded: usize,
}

/// Apply falsification for every `Refuted` outcome in `outcomes`.
///
/// Non-`Refuted` outcomes are skipped (an `Inconclusive` probe must never
/// destroy a memory). Each refutation lands in `scope`; the correction's
/// content is the verdict's `correction` if present, else a generic note that
/// embeds the probe's reason. Returns how many memories were superseded.
pub async fn falsify(
    log: &dyn EventLog,
    scope: &Scope,
    outcomes: &[ProbeOutcome],
) -> Result<FalsifyOutcome, MnemeError> {
    let mut superseded = 0;
    for outcome in outcomes {
        if outcome.verdict.status != ProbeStatus::Refuted {
            continue;
        }
        supersede_one(log, scope, outcome).await?;
        superseded += 1;
    }
    Ok(FalsifyOutcome { superseded })
}

async fn supersede_one(
    log: &dyn EventLog,
    scope: &Scope,
    outcome: &ProbeOutcome,
) -> Result<(), MnemeError> {
    let target = outcome.memory;
    let content = outcome.verdict.correction.clone().unwrap_or_else(|| {
        format!(
            "[superseded by acceptance probe] {}",
            outcome.verdict.reason
        )
    });

    let correction = Memory {
        id: new_id(),
        scope: scope.clone(),
        content,
        keywords: vec![],
        tags: vec![],
        context: String::new(),
        embedding: None,
        links: vec![],
        parent: Some(target),
        evolution_count: 0,
        time: BiTemporal::now(),
        provenance: Provenance {
            source: "probe".to_string(),
            // A self-correction is moderately trusted: it came from a direct
            // re-check, but the original claim was wrong, so not maxed out.
            trust: 0.7,
        },
        source: None,
        position: None,
    };
    let correction_ref = MemoryRef(correction.id);

    log.append(Event::MemoryWritten(correction)).await?;
    log.append(Event::MemoryEvolved {
        from: target,
        to: correction_ref,
        diff: ChangeSet {
            keywords_added: vec![],
            keywords_removed: vec![],
            tags_added: vec![],
            tags_removed: vec![],
            context_rewritten: true,
        },
    })
    .await?;
    log.append(Event::MemoryInvalidated {
        id: target,
        reason: FALSIFY_REASON.to_string(),
    })
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{ProbeOutcome, ProbeVerdict};
    use async_trait::async_trait;
    use mneme_core::event::LogEntry;
    use mneme_core::types::new_id;
    use std::sync::Mutex;

    #[derive(Default)]
    struct FakeLog {
        events: Mutex<Vec<Event>>,
    }

    #[async_trait]
    impl EventLog for FakeLog {
        async fn append(&self, event: Event) -> Result<mneme_core::Id, MnemeError> {
            self.events.lock().unwrap().push(event);
            Ok(new_id())
        }
        async fn read_from(
            &self,
            _after: Option<mneme_core::Id>,
        ) -> Result<Vec<LogEntry>, MnemeError> {
            Ok(vec![])
        }
    }

    impl FakeLog {
        fn events(&self) -> Vec<Event> {
            self.events.lock().unwrap().clone()
        }
    }

    fn mref() -> MemoryRef {
        MemoryRef(new_id())
    }

    fn refuted(memory: MemoryRef, correction: Option<&str>) -> ProbeOutcome {
        ProbeOutcome {
            memory,
            verdict: ProbeVerdict::refuted("stale", correction.map(|s| s.to_string())),
        }
    }

    #[tokio::test]
    async fn refuted_memory_emits_the_supersede_triple_history_kept() {
        let log = FakeLog::default();
        let scope = Scope::global("acme");
        let target = mref();
        let out = falsify(&log, &scope, &[refuted(target, Some("corrected fact"))])
            .await
            .unwrap();
        assert_eq!(out.superseded, 1);

        let events = log.events();
        assert_eq!(events.len(), 3, "written + evolved + invalidated");

        // 1. correction written with parent = target, content = correction.
        let (correction_ref, parent, content) = match &events[0] {
            Event::MemoryWritten(m) => (MemoryRef(m.id), m.parent, m.content.clone()),
            other => panic!("expected MemoryWritten, got {other:?}"),
        };
        assert_eq!(parent, Some(target), "child points at the refuted parent");
        assert_eq!(content, "corrected fact");

        // 2. lineage edge target → correction.
        match &events[1] {
            Event::MemoryEvolved { from, to, .. } => {
                assert_eq!(*from, target);
                assert_eq!(*to, correction_ref);
            }
            other => panic!("expected MemoryEvolved, got {other:?}"),
        }

        // 3. old version invalidated (NOT deleted) with the falsify reason.
        match &events[2] {
            Event::MemoryInvalidated { id, reason } => {
                assert_eq!(*id, target);
                assert_eq!(reason, FALSIFY_REASON);
            }
            other => panic!("expected MemoryInvalidated, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn correction_falls_back_to_reason_when_absent() {
        let log = FakeLog::default();
        let target = mref();
        falsify(&log, &Scope::global("t"), &[refuted(target, None)])
            .await
            .unwrap();
        match &log.events()[0] {
            Event::MemoryWritten(m) => {
                assert!(m.content.contains("superseded by acceptance probe"));
            }
            other => panic!("expected MemoryWritten, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn held_and_inconclusive_never_supersede() {
        let log = FakeLog::default();
        let held = ProbeOutcome {
            memory: mref(),
            verdict: ProbeVerdict::held("ok"),
        };
        let inc = ProbeOutcome {
            memory: mref(),
            verdict: ProbeVerdict::inconclusive("dunno"),
        };
        let out = falsify(&log, &Scope::global("t"), &[held, inc])
            .await
            .unwrap();
        assert_eq!(out.superseded, 0);
        assert!(
            log.events().is_empty(),
            "no events for non-refuted outcomes"
        );
    }
}
