//! [`ProceduralWorker`] — background loop driving the compiler.
//!
//! Tails the event log, buffers `OutcomeRecorded` events per active
//! artifact, and when the [`crate::CompileTrigger`] threshold trips
//! runs [`ProceduralCompiler::compile_with_inputs`] and feeds the
//! result back into the log via [`ProceduralCompiler::apply`].
//!
//! Mirrors the [`mneme_evolve::EvolutionWorker`] in shape — same tail-
//! poll cadence, same replay-on-startup, same single-writer discipline
//! (the worker is the only thing that mutates active state).
//!
//! ## What this worker does NOT do (yet)
//!
//! - Multi-objective Pareto selection (picks first committable
//!   candidate; Slice D)
//! - Multi-artifact credit assignment (handles one active artifact at
//!   a time per compile pass; revisit when Outcomes start carrying
//!   richer trajectory data)
//! - Replay & safety-probe sourcing (those are caller-supplied in
//!   `ShadowInputs`; the worker forwards a fixed set provided at
//!   construction time)

use crate::curve::LearningCurveCollector;
use crate::eval::EvalSuite;
use crate::executor::PolicyExecutor;
use crate::store::ProceduralStore;
use crate::{CompileTrigger, ProceduralCompiler, ShadowInputs};
use mneme_core::entity::Outcome;
use mneme_core::event::{Event, LogEntry};
use mneme_core::types::{ArtifactRef, Id};
use mneme_core::{EventLog, MnemeError};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Per-artifact bookkeeping the worker tracks for scheduling.
#[derive(Debug, Clone, Default)]
struct ArtifactState {
    /// Outcomes referencing this artifact that have arrived since the
    /// last compile pass.
    pending: Vec<Outcome>,
    /// Wall-clock timestamp (unix ms) of the most recent compile pass.
    last_compile_ms: u64,
}

/// The procedural worker. Holds:
///
/// - the event log (single source of truth — Rule #4)
/// - the [`ProceduralStore`] (derived state of active versions)
/// - the [`ProceduralCompiler`] (reflect → eval → gate)
/// - the [`ShadowInputs`] template (held-out replay + safety probes)
/// - per-artifact pending-outcome buffers + last-compile timestamps
///
/// One worker per host process. The single-writer invariant for
/// procedural state is enforced by the worker being the only thing
/// that calls `compiler.apply(...)`.
pub struct ProceduralWorker {
    log: Arc<dyn EventLog>,
    store: Arc<ProceduralStore>,
    compiler: Arc<ProceduralCompiler>,
    /// Replay + safety-probe inputs handed to every shadow eval. The
    /// `baseline` field is overwritten per compile pass with the
    /// active artifact being challenged.
    shadow_template: Arc<ShadowInputs>,
    state: Arc<RwLock<HashMap<ArtifactRef, ArtifactState>>>,
    /// Optional benchmark suite + curve collector + executor. When
    /// present, the worker runs the suite against each newly-committed
    /// active version and records a [`LearningCurvePoint`]. None means
    /// the dashboard's learning-curve panel stays empty — the gate
    /// machinery still works.
    eval: Option<EvalBinding>,
}

/// Bundle of the three pieces needed to record learning curve points.
/// Held as one struct so they're always present together — recording
/// a point requires running the suite which requires an executor.
pub struct EvalBinding {
    pub suite: Arc<EvalSuite>,
    pub collector: Arc<LearningCurveCollector>,
    pub executor: Arc<dyn PolicyExecutor>,
}

impl ProceduralWorker {
    pub fn new(
        log: Arc<dyn EventLog>,
        store: Arc<ProceduralStore>,
        compiler: Arc<ProceduralCompiler>,
        shadow_template: Arc<ShadowInputs>,
    ) -> Self {
        Self {
            log,
            store,
            compiler,
            shadow_template,
            state: Arc::new(RwLock::new(HashMap::new())),
            eval: None,
        }
    }

    /// Attach an evaluation binding. Builder-style so the existing
    /// `new`-only construction path keeps working — eval is opt-in.
    pub fn with_eval(mut self, eval: EvalBinding) -> Self {
        self.eval = Some(eval);
        self
    }

    /// Borrow the curve collector for the dashboard handler. Returns
    /// `None` when no eval binding is configured.
    pub fn curve(&self) -> Option<&Arc<LearningCurveCollector>> {
        self.eval.as_ref().map(|e| &e.collector)
    }

    /// Borrow the underlying store for read-only dashboard / snapshot
    /// access.
    pub fn store(&self) -> &Arc<ProceduralStore> {
        &self.store
    }

    /// Replay the entire log to rebuild caches. Idempotent. If an
    /// eval binding is configured AND the curve is empty AND the
    /// store has any active artifacts after replay, runs the suite on
    /// each active artifact once to seed a baseline point — gives the
    /// dashboard a starting datum to plot against.
    pub async fn replay(&self) -> Result<Option<Id>, MnemeError> {
        let entries = self.log.read_from(None).await?;
        let last = entries.last().map(|e| e.id);
        for entry in &entries {
            self.absorb(entry).await;
        }
        if let Some(eval) = &self.eval {
            // Rebuild the in-memory curve from the log first — events
            // are the source of truth (Rule #4). If any LearningCurveRecorded
            // events exist for active artifacts, they populate the
            // collector here.
            eval.collector.replay(self.log.as_ref()).await?;
            // If the curve is *still* empty for an active artifact,
            // seed a baseline measurement by running the suite once.
            // This handles the bootstrap case where the artifact was
            // seeded but never measured.
            if eval.collector.is_empty().await {
                for active in self.store.all().await {
                    let aref = ArtifactRef(active.id);
                    if let Ok(report) = eval.suite.run(&active, eval.executor.clone()).await {
                        self.emit_and_absorb_curve_point(
                            eval,
                            aref,
                            active.version,
                            &report,
                            0.0, // no Δ — this is the initial baseline
                            0,   // no judges signed off — this is a measurement, not a commit
                        )
                        .await;
                    }
                }
            }
        }
        Ok(last)
    }

    /// Append a `LearningCurveRecorded` event to the log, then absorb
    /// it into the collector. Order matters: the event is the source
    /// of truth; in-memory state derives from it. If the append fails,
    /// the in-memory cache stays untouched and the next `replay` will
    /// reconcile.
    async fn emit_and_absorb_curve_point(
        &self,
        eval: &EvalBinding,
        artifact: ArtifactRef,
        version: u32,
        report: &crate::eval::SuiteReport,
        objective_delta: f32,
        judges_consulted: u8,
    ) {
        let event = Event::LearningCurveRecorded {
            artifact,
            version,
            benchmark_score: report.benchmark_score,
            safety_probe_pass_rate: report.safety_probe_pass_rate,
            objective_delta,
            judges_consulted,
        };
        match self.log.append(event.clone()).await {
            Ok(id) => {
                eval.collector.absorb(&LogEntry { id, event }).await;
            }
            Err(e) => {
                tracing::warn!(
                    artifact = %artifact.0,
                    version,
                    error = %e,
                    "curve event append failed; in-memory cache will reconcile on next replay"
                );
            }
        }
    }

    /// Update internal caches + the store from a single log entry.
    /// Public so tests can fabricate state without the tail loop.
    pub async fn absorb(&self, entry: &LogEntry) {
        // Always propagate to the store first — its state is what
        // `try_compile` reads to find active artifacts.
        self.store.absorb(entry).await;

        if let Event::OutcomeRecorded(o) = &entry.event {
            // Bucket the outcome under every artifact it touched. An
            // outcome can reference multiple artifacts (multi-artifact
            // credit assignment territory); we buffer it for each.
            let mut state = self.state.write().await;
            for aref in &o.artifacts_used {
                state.entry(*aref).or_default().pending.push(o.clone());
            }
        }
    }

    /// Check every artifact's pending buffer against the trigger and
    /// fire a compile pass for any that qualify. Returns the number of
    /// proposals applied (committed or rejected) — caller logs.
    pub async fn try_compile_due(&self) -> Result<usize, MnemeError> {
        let trigger = self.compiler.trigger.clone();
        let now_ms = now_ms();
        // Snapshot the set of artifacts that might be due — we don't
        // hold the lock across the compile pass (which is async-LLM).
        let due: Vec<ArtifactRef> = {
            let state = self.state.read().await;
            state
                .iter()
                .filter_map(|(aref, s)| {
                    if is_due(&trigger, s, now_ms) {
                        Some(*aref)
                    } else {
                        None
                    }
                })
                .collect()
        };

        let mut applied = 0usize;
        for aref in due {
            // Lookup active artifact + drain pending outcomes for it.
            let active = match self.store.get(aref).await {
                Some(a) => a,
                None => {
                    // Outcomes referenced an artifact id the store
                    // doesn't know about — likely never seeded, or
                    // we're racing replay. Drop the buffer so we don't
                    // grow unboundedly.
                    self.state.write().await.remove(&aref);
                    continue;
                }
            };
            let outcomes = {
                let mut state = self.state.write().await;
                let s = state.entry(aref).or_default();
                let outs = std::mem::take(&mut s.pending);
                s.last_compile_ms = now_ms;
                outs
            };
            if outcomes.is_empty() {
                continue;
            }

            // Build the per-pass ShadowInputs by overlaying the active
            // artifact onto the template.
            let inputs = ShadowInputs {
                baseline: active.clone(),
                replay: self.shadow_template.replay.clone(),
                safety_probes: self.shadow_template.safety_probes.clone(),
            };

            let result = match self
                .compiler
                .compile_with_inputs(
                    &active,
                    &outcomes,
                    &inputs,
                    format!(
                        "procedural worker: {} outcomes since last compile",
                        outcomes.len()
                    ),
                )
                .await
            {
                Ok(Some(r)) => r,
                Ok(None) => {
                    tracing::debug!(artifact = %active.id, "compile produced no proposal");
                    continue;
                }
                Err(e) => {
                    tracing::warn!(artifact = %active.id, error = %e, "compile failed");
                    continue;
                }
            };

            // Snapshot the winning report (if any) BEFORE moving the
            // result into apply() — we need it for the curve point.
            let winning_report = result
                .winner()
                .map(|w| (w.report.clone(), w.candidate.clone()));

            // Apply (commit or reject) and absorb the resulting events
            // so the next loop iteration sees the new active version.
            match self.compiler.apply(self.log.as_ref(), result).await {
                Ok(prop_id) => {
                    applied += 1;
                    tracing::info!(
                        artifact = %active.id,
                        proposal = %prop_id.0,
                        "procedural worker: applied proposal"
                    );
                    // Pull the just-appended events into the store so
                    // the next compile sees the new active version
                    // without waiting for the tail loop.
                    if let Ok(new_entries) = self.log.read_from(Some(prop_id.0)).await {
                        for e in &new_entries {
                            self.store.absorb(e).await;
                        }
                    }
                    // If this was a commit (not a reject) AND we have an
                    // eval binding, run the suite against the new active
                    // and record a curve point. Phase-2 "done when":
                    // positive curve + 100% safety probe.
                    if let (Some((report, winning_candidate)), Some(eval)) =
                        (winning_report, self.eval.as_ref())
                    {
                        // Resolve the now-active artifact from the store
                        // — it's the winning candidate (id = active.id,
                        // version bumped).
                        let target = self.store.get(aref).await.unwrap_or(winning_candidate);
                        match eval.suite.run(&target, eval.executor.clone()).await {
                            Ok(suite_report) => {
                                self.emit_and_absorb_curve_point(
                                    eval,
                                    aref,
                                    target.version,
                                    &suite_report,
                                    report.objective_delta,
                                    report.judges_consulted,
                                )
                                .await;
                                if !suite_report.safety_clean() {
                                    // Alignment-drift indicator (report
                                    // §10). Loud warning so the operator
                                    // sees it even without dashboard.
                                    tracing::warn!(
                                        artifact = %target.id,
                                        version = target.version,
                                        pass_rate = suite_report.safety_probe_pass_rate,
                                        "SAFETY PROBE REGRESSION — alignment-drift signal"
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    artifact = %target.id,
                                    error = %e,
                                    "eval suite run failed; skipping curve point"
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(artifact = %active.id, error = %e, "apply failed");
                }
            }
        }
        Ok(applied)
    }
}

/// Is this artifact due for a compile pass? True iff either the
/// pending buffer has hit `min_batch`, or `max_age_secs` has elapsed
/// since the last compile *and* there's at least one pending outcome
/// (no point compiling on an empty buffer).
fn is_due(trigger: &CompileTrigger, state: &ArtifactState, now_ms: u64) -> bool {
    if state.pending.len() >= trigger.min_batch {
        return true;
    }
    if state.pending.is_empty() {
        return false;
    }
    if state.last_compile_ms == 0 {
        // Never compiled — let the count trigger fire. We don't want
        // to compile-on-first-outcome with `max_age_secs=3600` because
        // that's a "wait an hour" rule, not a "compile immediately"
        // rule.
        return false;
    }
    let age_ms = now_ms.saturating_sub(state.last_compile_ms);
    age_ms >= trigger.max_age_secs.saturating_mul(1000)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Spawn the worker as a background tokio task. Returns immediately;
/// the worker runs for the life of the runtime.
///
/// `eval` is optional — passing `Some(EvalBinding)` turns on the
/// learning-curve collector + per-commit suite runs (Phase 2 "done
/// when" instrumentation). Passing `None` leaves the curve panel
/// empty on the dashboard but the gate machinery still runs.
pub fn spawn(
    log: Arc<dyn EventLog>,
    store: Arc<ProceduralStore>,
    compiler: Arc<ProceduralCompiler>,
    shadow_template: Arc<ShadowInputs>,
    eval: Option<EvalBinding>,
) -> Arc<ProceduralWorker> {
    let mut worker = ProceduralWorker::new(log, store, compiler, shadow_template);
    if let Some(b) = eval {
        worker = worker.with_eval(b);
    }
    let worker = Arc::new(worker);
    let w = worker.clone();
    tokio::spawn(async move { run_loop(w).await });
    worker
}

async fn run_loop(worker: Arc<ProceduralWorker>) {
    let mut last_seen = match worker.replay().await {
        Ok(last) => last,
        Err(e) => {
            tracing::error!(error = %e, "procedural worker: initial replay failed");
            None
        }
    };
    tracing::info!("procedural worker: started");
    loop {
        match worker.log.read_from(last_seen).await {
            Ok(entries) => {
                for entry in entries {
                    last_seen = Some(entry.id);
                    worker.absorb(&entry).await;
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "procedural worker: log read failed");
            }
        }
        if let Err(e) = worker.try_compile_due().await {
            tracing::warn!(error = %e, "procedural worker: try_compile_due failed");
        }
        tokio::time::sleep(Duration::from_millis(800)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::FakePolicyExecutor;
    use crate::judge::FakeJudge;
    use crate::Judge;
    use mneme_core::entity::{ArtifactKind, JudgeSource, PolicyArtifact};
    use mneme_core::event::EvalReport;
    use mneme_core::types::{new_id, BiTemporal, EpisodeRef, ProposalId, Scope, TrajectoryRef};
    use mneme_llm::FakeLlmClient;
    use std::collections::HashMap;
    use std::sync::Mutex;

    // Minimal in-process log — same pattern as store.rs tests.
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
        async fn append(&self, event: Event) -> Result<Id, MnemeError> {
            let id = new_id();
            self.entries.lock().unwrap().push(LogEntry { id, event });
            Ok(id)
        }
        async fn read_from(&self, after: Option<Id>) -> Result<Vec<LogEntry>, MnemeError> {
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

    fn outcome_for(aref: ArtifactRef, success: bool) -> Outcome {
        let mut scores = HashMap::new();
        scores.insert("accuracy".into(), if success { 1.0 } else { 0.0 });
        Outcome {
            id: new_id(),
            episode: EpisodeRef(new_id()),
            artifacts_used: vec![aref],
            success: Some(success),
            scores,
            error: None,
            judge: JudgeSource::Environment,
            trajectory: TrajectoryRef(new_id()),
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

    fn happy_llm() -> Arc<FakeLlmClient> {
        Arc::new(
            FakeLlmClient::new()
                .with_prefix_match(
                    "You are reviewing a recent batch",
                    "FINDING: the prompt is too terse",
                )
                .with_prefix_match(
                    "You are revising a system prompt",
                    "--- CANDIDATE 1 ---\n\
                     You are a careful, helpful assistant. Answer briefly but completely.",
                ),
        )
    }

    fn happy_compiler() -> Arc<ProceduralCompiler> {
        let mut c = ProceduralCompiler::new(
            happy_llm(),
            Arc::new(FakePolicyExecutor::new().with_default("ok")),
            vec![
                Arc::new(FakeJudge::new("a")) as Arc<dyn Judge>,
                Arc::new(FakeJudge::new("b")),
            ],
            1,
        );
        // Tight trigger for tests — fire on 2 outcomes, ignore the time bucket.
        c.trigger = CompileTrigger {
            min_batch: 2,
            max_age_secs: 86_400,
        };
        Arc::new(c)
    }

    async fn seed_artifact(log: &Arc<MemoryLog>, store: &Arc<ProceduralStore>) -> PolicyArtifact {
        let a = artifact(1, "Answer briefly.");
        let pid = ProposalId(new_id());
        let prop = Event::ProceduralProposed {
            proposal: pid,
            artifacts: vec![a.clone()],
        };
        let commit = Event::ProceduralCommitted {
            proposal: pid,
            report: good_report(),
        };
        log.append(prop.clone()).await.unwrap();
        log.append(commit.clone()).await.unwrap();
        store.replay(log.as_ref()).await.unwrap();
        a
    }

    #[tokio::test]
    async fn compile_triggers_when_min_batch_reached_and_apply_commits() {
        let log = MemoryLog::new();
        let store = Arc::new(ProceduralStore::new());
        let a = seed_artifact(&log, &store).await;
        let aref = ArtifactRef(a.id);
        let compiler = happy_compiler();
        let inputs = Arc::new(ShadowInputs {
            baseline: a.clone(),
            replay: vec![],
            safety_probes: vec![],
        });
        let worker = ProceduralWorker::new(log.clone(), store.clone(), compiler, inputs);

        // Two outcomes → min_batch=2 → trigger fires.
        for _ in 0..2 {
            let entry = LogEntry {
                id: new_id(),
                event: Event::OutcomeRecorded(outcome_for(aref, false)),
            };
            log.append(entry.event.clone()).await.unwrap();
            worker.absorb(&entry).await;
        }
        let applied = worker.try_compile_due().await.unwrap();
        assert_eq!(applied, 1, "one apply per due artifact");

        // Replay store fresh — make sure ProceduralCommitted was emitted.
        let fresh = ProceduralStore::new();
        fresh.replay(log.as_ref()).await.unwrap();
        // The compile bumps the active version to 2.
        assert!(
            fresh.all().await.iter().any(|a| a.version == 2),
            "post-apply, log replay must show the new active version"
        );
    }

    #[tokio::test]
    async fn no_compile_before_min_batch() {
        let log = MemoryLog::new();
        let store = Arc::new(ProceduralStore::new());
        let a = seed_artifact(&log, &store).await;
        let aref = ArtifactRef(a.id);
        let inputs = Arc::new(ShadowInputs {
            baseline: a.clone(),
            replay: vec![],
            safety_probes: vec![],
        });
        let worker = ProceduralWorker::new(log.clone(), store.clone(), happy_compiler(), inputs);

        // Only one outcome — below min_batch=2.
        let entry = LogEntry {
            id: new_id(),
            event: Event::OutcomeRecorded(outcome_for(aref, false)),
        };
        log.append(entry.event.clone()).await.unwrap();
        worker.absorb(&entry).await;

        let applied = worker.try_compile_due().await.unwrap();
        assert_eq!(applied, 0, "should not fire below min_batch");
    }

    #[tokio::test]
    async fn outcomes_for_unknown_artifact_are_dropped() {
        // If an outcome references an artifact id the store doesn't
        // know, the worker shouldn't grow its pending buffer
        // unboundedly.
        let log = MemoryLog::new();
        let store = Arc::new(ProceduralStore::new());
        // Don't seed — store is empty.
        let inputs = Arc::new(ShadowInputs {
            baseline: artifact(1, "unused"),
            replay: vec![],
            safety_probes: vec![],
        });
        let worker = ProceduralWorker::new(log.clone(), store.clone(), happy_compiler(), inputs);

        let unknown_aref = ArtifactRef(new_id());
        for _ in 0..3 {
            let entry = LogEntry {
                id: new_id(),
                event: Event::OutcomeRecorded(outcome_for(unknown_aref, false)),
            };
            log.append(entry.event.clone()).await.unwrap();
            worker.absorb(&entry).await;
        }
        let applied = worker.try_compile_due().await.unwrap();
        assert_eq!(applied, 0);
        // Pending buffer should be cleared since the artifact is unknown.
        let state = worker.state.read().await;
        assert!(
            !state.contains_key(&unknown_aref),
            "unknown artifact's buffer must be dropped, not retained"
        );
    }
}
