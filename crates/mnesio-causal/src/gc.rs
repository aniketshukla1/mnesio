//! Garbage collection by *measurement*. A [`ContributionReport`] names the
//! memories that provably never moved an outcome; GC retires them — but never
//! deletes (Hard Rule #2). Retirement is an appended
//! [`Event::MemoryInvalidated`] with no replacement, so the log stays
//! append-only (#4) and the history is fully reconstructable.

use crate::contribution::ContributionReport;
use mnesio_core::traits::EventLog;
use mnesio_core::types::MemoryRef;
use mnesio_core::{Event, MnesioError};

/// The reason string stamped on every causal-GC invalidation, so audits +
/// the dashboard can tell measured retirement apart from contradiction /
/// supersession.
pub const GC_REASON: &str = "causal-gc: counterfactual contribution ≤ epsilon";

/// Which memories to retire this pass.
#[derive(Debug, Clone, Default)]
pub struct GcDecision {
    pub retire: Vec<MemoryRef>,
}

impl GcDecision {
    /// Build a decision from a scored report: retire everything at or below
    /// `epsilon` (inert + harmful).
    pub fn from_report(report: &ContributionReport, epsilon: f32) -> Self {
        Self {
            retire: report.gc_candidates(epsilon),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.retire.is_empty()
    }
}

/// What a GC pass did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GcOutcome {
    pub retired: usize,
}

/// Apply a GC decision by appending one [`Event::MemoryInvalidated`] per
/// retired memory.
///
/// The event carries `{ id, reason }`; the memory's `Scope` is intrinsic to
/// it, so it isn't repeated here. The caller is still responsible for having
/// scoped the report's candidates (Hard Rule #3) — GC will faithfully retire
/// whatever it's handed. Idempotency is *not* claimed: invalidating an
/// already-invalidated memory is a harmless no-op at the view layer, but
/// callers running repeated passes should re-score against live memories
/// rather than replaying a stale decision.
pub async fn gc(log: &dyn EventLog, decision: &GcDecision) -> Result<GcOutcome, MnesioError> {
    for memory in &decision.retire {
        log.append(Event::MemoryInvalidated {
            id: *memory,
            reason: GC_REASON.to_string(),
        })
        .await?;
    }
    Ok(GcOutcome {
        retired: decision.retire.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contribution::{CausalConfig, ContributionScorer, FakeEvaluator};
    use async_trait::async_trait;
    use mnesio_core::event::LogEntry;
    use mnesio_core::types::new_id;
    use std::sync::Mutex;

    /// Capture-only event log: records appended events for assertions.
    /// `read_from` is not exercised by these tests.
    #[derive(Default)]
    struct FakeLog {
        events: Mutex<Vec<Event>>,
    }

    #[async_trait]
    impl EventLog for FakeLog {
        async fn append(&self, event: Event) -> Result<mnesio_core::Id, MnesioError> {
            self.events.lock().unwrap().push(event);
            Ok(new_id())
        }
        async fn read_from(
            &self,
            _after: Option<mnesio_core::Id>,
        ) -> Result<Vec<LogEntry>, MnesioError> {
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

    #[tokio::test]
    async fn gc_appends_one_invalidation_per_retired_memory() {
        let log = FakeLog::default();
        let (m1, m2) = (mref(), mref());
        let decision = GcDecision {
            retire: vec![m1, m2],
        };

        let outcome = gc(&log, &decision).await.unwrap();
        assert_eq!(outcome.retired, 2);

        let events = log.events();
        assert_eq!(events.len(), 2);
        for ev in &events {
            match ev {
                Event::MemoryInvalidated { reason, .. } => {
                    assert_eq!(reason, GC_REASON, "stamped with the causal-GC reason");
                }
                other => panic!("expected MemoryInvalidated, got {other:?}"),
            }
        }
        let invalidated: Vec<MemoryRef> = events
            .iter()
            .filter_map(|e| match e {
                Event::MemoryInvalidated { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert!(invalidated.contains(&m1));
        assert!(invalidated.contains(&m2));
    }

    #[tokio::test]
    async fn empty_decision_is_a_noop() {
        let log = FakeLog::default();
        let outcome = gc(&log, &GcDecision::default()).await.unwrap();
        assert_eq!(outcome.retired, 0);
        assert!(log.events().is_empty());
    }

    #[tokio::test]
    async fn from_report_retires_only_low_contributors_then_gc_emits_them() {
        let (helpful, inert) = (mref(), mref());
        let eval = FakeEvaluator::new(0.8)
            .with_marginal(helpful, 0.5)
            .with_marginal(inert, 0.0);
        let report = ContributionScorer::new(CausalConfig::default())
            .score(&eval, &[helpful, inert])
            .await
            .unwrap();

        let decision = GcDecision::from_report(&report, 1e-4);
        assert_eq!(
            decision.retire,
            vec![inert],
            "only the inert memory retires"
        );

        let log = FakeLog::default();
        let outcome = gc(&log, &decision).await.unwrap();
        assert_eq!(outcome.retired, 1);
        match &log.events()[0] {
            Event::MemoryInvalidated { id, .. } => assert_eq!(*id, inert),
            other => panic!("expected MemoryInvalidated, got {other:?}"),
        }
    }
}
