//! [`LearningCurveCollector`] — the Phase 2 "done when" instrument.
//!
//! CLAUDE.md: *Phase 2 done when: positive learning curve on an
//! ALFWorld-style suite with no safety-probe regression.* This module
//! is what makes that statement falsifiable.
//!
//! Records one [`LearningCurvePoint`] per committed artifact version,
//! capturing the absolute [`EvalSuite`][crate::eval::EvalSuite] score
//! and the safety probe pass rate. The dashboard plots these as a line
//! chart: if the benchmark line trends up *and* the safety line stays
//! at 1.0, the wedge is real.
//!
//! ## Where points come from
//!
//! Pushed by the [`crate::worker::ProceduralWorker`] after every
//! successful commit. The worker:
//! 1. lands the commit through `ProceduralCompiler::apply`
//! 2. re-fetches the now-active artifact from the [`crate::ProceduralStore`]
//! 3. runs the [`crate::eval::EvalSuite`] against it
//! 4. records a [`LearningCurvePoint`] here
//!
//! Replay-from-log IS supported as of Slice E: every recorded point
//! is also emitted as a [`mnesio_core::event::Event::LearningCurveRecorded`]
//! event, so the collector can rebuild its history exactly by replaying
//! the log. Restart-survives — a safety-probe regression that happened
//! a week ago is still in the curve. The in-memory cache stays as a
//! cheap accessor for the dashboard; the log is the truth.
//!
//! ## Concurrency
//!
//! Internal `RwLock` — same single-writer pattern as
//! [`crate::ProceduralStore`]. The worker is the only thing that
//! mutates; many dashboard readers can pull `points()` concurrently.

use mnesio_core::event::{Event, LogEntry};
use mnesio_core::types::ArtifactRef;
use mnesio_core::{EventLog, MnesioError};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// One point on the learning curve. Snapshot of the suite + gate
/// signal at the moment a commit landed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearningCurvePoint {
    pub artifact_id: ArtifactRef,
    pub version: u32,
    /// Wall-clock (unix ms) of the commit.
    pub timestamp_ms: u64,
    /// Absolute suite score `[0.0, 1.0]` — the curve's Y axis.
    pub benchmark_score: f32,
    /// Safety probe pass rate `[0.0, 1.0]` — must stay 1.0 throughout.
    /// Any dip is the alignment-drift signal that should halt
    /// learning per report §10.
    pub safety_probe_pass_rate: f32,
    /// Gate's per-commit objective Δ — useful overlay so the operator
    /// can see whether the absolute score is rising because the *gate*
    /// is rewarding it, or for unrelated reasons.
    pub objective_delta: f32,
    /// How many distinct judges signed off on this commit — diversity
    /// gate compliance indicator.
    pub judges_consulted: u8,
}

/// Append-only collector of learning curve points. Cloneable handle
/// over an internal `Arc<RwLock<…>>`.
#[derive(Clone, Default)]
pub struct LearningCurveCollector {
    inner: Arc<RwLock<Vec<LearningCurvePoint>>>,
}

impl LearningCurveCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a new point. Ordered by arrival — the dashboard plots
    /// chronologically.
    pub async fn record(&self, point: LearningCurvePoint) {
        self.inner.write().await.push(point);
    }

    /// Replay the entire log to rebuild the curve. Idempotent — a
    /// second `replay` against the same log will append the same
    /// points again, which is intentional: the in-memory state is
    /// derived, the log is the truth. Callers that want a clean
    /// rebuild should construct a fresh `LearningCurveCollector`.
    pub async fn replay(&self, log: &dyn EventLog) -> Result<(), MnesioError> {
        let entries = log.read_from(None).await?;
        for entry in &entries {
            self.absorb(entry).await;
        }
        Ok(())
    }

    /// Update the collector from a single log entry. Public so the
    /// worker can drive it directly during the tail loop. Non-curve
    /// events are no-ops.
    pub async fn absorb(&self, entry: &LogEntry) {
        if let Event::LearningCurveRecorded {
            artifact,
            version,
            benchmark_score,
            safety_probe_pass_rate,
            objective_delta,
            judges_consulted,
        } = &entry.event
        {
            self.record(LearningCurvePoint {
                artifact_id: *artifact,
                version: *version,
                timestamp_ms: entry.id.timestamp_ms(),
                benchmark_score: *benchmark_score,
                safety_probe_pass_rate: *safety_probe_pass_rate,
                objective_delta: *objective_delta,
                judges_consulted: *judges_consulted,
            })
            .await;
        }
    }

    /// Snapshot of every recorded point in arrival order.
    pub async fn points(&self) -> Vec<LearningCurvePoint> {
        self.inner.read().await.clone()
    }

    /// All points for a specific artifact, in arrival order. Used by
    /// the dashboard when filtering per-artifact curves.
    pub async fn points_for(&self, aref: ArtifactRef) -> Vec<LearningCurvePoint> {
        self.inner
            .read()
            .await
            .iter()
            .filter(|p| p.artifact_id == aref)
            .cloned()
            .collect()
    }

    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// True iff every recorded point has `safety_probe_pass_rate ==
    /// 1.0`. The single-bit alignment indicator the report (§10) asks
    /// for — monitoring tools that want to halt learning on any
    /// regression read this.
    pub async fn safety_clean(&self) -> bool {
        self.inner
            .read()
            .await
            .iter()
            .all(|p| p.safety_probe_pass_rate >= 1.0 - 1e-6)
    }

    /// True iff `benchmark_score` is monotone non-decreasing across
    /// every recorded point for a given artifact. The strictest
    /// version of "the curve is going up." Dashboards usually want
    /// the *trend* (last N points up) — this method answers the
    /// stricter, easier-to-falsify question.
    pub async fn strictly_non_decreasing(&self, aref: ArtifactRef) -> bool {
        let points = self.points_for(aref).await;
        let mut prev = -1.0f32;
        for p in &points {
            if p.benchmark_score < prev - 1e-6 {
                return false;
            }
            prev = p.benchmark_score;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::types::new_id;

    fn point(version: u32, score: f32, safety: f32, aref: ArtifactRef) -> LearningCurvePoint {
        LearningCurvePoint {
            artifact_id: aref,
            version,
            timestamp_ms: 1_000_000 + version as u64 * 1000,
            benchmark_score: score,
            safety_probe_pass_rate: safety,
            objective_delta: 0.0,
            judges_consulted: 2,
        }
    }

    #[tokio::test]
    async fn record_and_read_back_in_order() {
        let c = LearningCurveCollector::new();
        let a = ArtifactRef(new_id());
        for v in 1..=3 {
            c.record(point(v, v as f32 / 3.0, 1.0, a)).await;
        }
        let pts = c.points().await;
        assert_eq!(pts.len(), 3);
        assert_eq!(pts[0].version, 1);
        assert_eq!(pts[2].version, 3);
    }

    #[tokio::test]
    async fn points_for_filters_by_artifact() {
        let c = LearningCurveCollector::new();
        let a1 = ArtifactRef(new_id());
        let a2 = ArtifactRef(new_id());
        c.record(point(1, 0.5, 1.0, a1)).await;
        c.record(point(1, 0.6, 1.0, a2)).await;
        c.record(point(2, 0.7, 1.0, a1)).await;
        assert_eq!(c.points_for(a1).await.len(), 2);
        assert_eq!(c.points_for(a2).await.len(), 1);
    }

    #[tokio::test]
    async fn safety_clean_detects_any_dip() {
        let c = LearningCurveCollector::new();
        let a = ArtifactRef(new_id());
        c.record(point(1, 0.5, 1.0, a)).await;
        c.record(point(2, 0.6, 1.0, a)).await;
        assert!(c.safety_clean().await);
        c.record(point(3, 0.7, 0.5, a)).await; // ← regression
        assert!(
            !c.safety_clean().await,
            "any dip below 1.0 must trip the flag"
        );
    }

    #[tokio::test]
    async fn strictly_non_decreasing_rejects_dropoff() {
        let c = LearningCurveCollector::new();
        let a = ArtifactRef(new_id());
        c.record(point(1, 0.3, 1.0, a)).await;
        c.record(point(2, 0.6, 1.0, a)).await;
        c.record(point(3, 0.6, 1.0, a)).await; // plateau OK
        assert!(c.strictly_non_decreasing(a).await);
        c.record(point(4, 0.5, 1.0, a)).await; // dropoff!
        assert!(!c.strictly_non_decreasing(a).await);
    }

    #[tokio::test]
    async fn strictly_non_decreasing_is_per_artifact() {
        // A regression on artifact 2 must NOT count against artifact 1.
        let c = LearningCurveCollector::new();
        let a1 = ArtifactRef(new_id());
        let a2 = ArtifactRef(new_id());
        c.record(point(1, 0.5, 1.0, a1)).await;
        c.record(point(2, 0.7, 1.0, a1)).await;
        c.record(point(1, 0.5, 1.0, a2)).await;
        c.record(point(2, 0.2, 1.0, a2)).await;
        assert!(c.strictly_non_decreasing(a1).await);
        assert!(!c.strictly_non_decreasing(a2).await);
    }

    #[tokio::test]
    async fn empty_collector_satisfies_both_predicates_vacuously() {
        let c = LearningCurveCollector::new();
        assert!(c.is_empty().await);
        assert!(c.safety_clean().await);
        assert!(c.strictly_non_decreasing(ArtifactRef(new_id())).await);
    }

    // ---- replay-from-log ----

    use mnesio_core::event::LogEntry;
    use mnesio_core::types::Id;
    use std::sync::Mutex;

    /// Minimal in-process log so the replay test stays
    /// dependency-free.
    struct MemoryLog {
        entries: Mutex<Vec<LogEntry>>,
    }

    #[async_trait::async_trait]
    impl mnesio_core::EventLog for MemoryLog {
        async fn append(
            &self,
            event: mnesio_core::event::Event,
        ) -> Result<Id, mnesio_core::MnesioError> {
            let id = new_id();
            self.entries.lock().unwrap().push(LogEntry { id, event });
            Ok(id)
        }

        async fn read_from(
            &self,
            after: Option<Id>,
        ) -> Result<Vec<LogEntry>, mnesio_core::MnesioError> {
            let g = self.entries.lock().unwrap();
            Ok(match after {
                None => g.clone(),
                Some(id) => g.iter().filter(|e| e.id > id).cloned().collect(),
            })
        }
    }

    #[tokio::test]
    async fn replay_rebuilds_curve_from_log_events() {
        let log = MemoryLog {
            entries: Mutex::new(Vec::new()),
        };
        let aref = ArtifactRef(new_id());

        // Append three curve events directly to the log (no worker
        // involved — this isolates the replay logic).
        for v in 1..=3 {
            log.append(mnesio_core::event::Event::LearningCurveRecorded {
                artifact: aref,
                version: v,
                benchmark_score: v as f32 / 3.0,
                safety_probe_pass_rate: 1.0,
                objective_delta: 0.0,
                judges_consulted: 2,
            })
            .await
            .unwrap();
        }

        // Fresh collector + replay → curve should be exactly what we
        // appended, in the right order.
        let c = LearningCurveCollector::new();
        c.replay(&log).await.unwrap();
        let pts = c.points().await;
        assert_eq!(pts.len(), 3);
        assert_eq!(pts[0].version, 1);
        assert_eq!(pts[2].version, 3);
        assert!((pts[2].benchmark_score - 1.0).abs() < 1e-6);
        assert!(c.safety_clean().await);
        assert!(c.strictly_non_decreasing(aref).await);
    }

    #[tokio::test]
    async fn replay_ignores_non_curve_events() {
        // Throw an unrelated event at the log and confirm replay
        // doesn't add anything.
        let log = MemoryLog {
            entries: Mutex::new(Vec::new()),
        };
        log.append(mnesio_core::event::Event::MemoryInvalidated {
            id: mnesio_core::types::MemoryRef(new_id()),
            reason: "unrelated".into(),
        })
        .await
        .unwrap();
        let c = LearningCurveCollector::new();
        c.replay(&log).await.unwrap();
        assert!(c.is_empty().await);
    }
}
