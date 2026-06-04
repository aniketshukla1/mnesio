//! Phase 11 — live self-falsifying-memory endpoint.
//!
//! `GET /api/probe/metrics` runs an on-demand acceptance-probe pass over the
//! *live* corpus and reports, per memory, whether its claim still holds
//! (`Held`), no longer holds (`Refuted`), or couldn't be decided
//! (`Inconclusive`) — plus each memory's calibrated belief (confidence + why).
//!
//! By default the pass is a **dry run**: a GET is side-effect-free, it only
//! previews what *would* be falsified. With `?apply=true` it performs the
//! falsification — appending the canonical supersede triple for every refuted
//! memory ([`mneme_probe::falsify`]) so the refuted version is invalidated and
//! a correction takes its place, with history kept (Hard Rule #2). That's the
//! Phase-11 "done when": a fact whose probe fails auto-supersedes with no
//! human in the loop.
//!
//! Probe rules are matched against memory *content* (id-agnostic, replayable).
//! The pass is bounded ([`ProbeConfig`], #6) and runs only when the endpoint
//! is hit, so it stays off the write path (#5).

use crate::viz::AppState;
use axum::extract::{Query as AxumQuery, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use mneme_core::event::Event;
use mneme_core::traits::MaterializedView;
use mneme_core::types::MemoryRef;
use mneme_core::Scope;
use mneme_probe::{
    belief_of, Evidence, FakeProbe, Probe, ProbeConfig, ProbeRunner, ProbeStatus, ProbeVerdict,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Cap on memories probed per pass — bounds the on-demand endpoint (#6).
const PROBE_MAX: usize = 128;
/// Confidence at/below which a memory is flagged "doubted".
const DOUBT_FLOOR: f32 = 0.35;

/// A demo acceptance probe: if a live memory's content matches `stale_marker`
/// (case-insensitive), the claim is treated as no longer holding, and
/// `correction` is the replacement fact to record on supersession.
struct DemoProbeRule {
    stale_marker: &'static str,
    correction: &'static str,
}

/// Probe rules over the demo corpus (see `demo.rs`). Each targets a concrete,
/// dated claim that a later re-check would find stale.
const DEMO_RULES: &[DemoProbeRule] = &[
    DemoProbeRule {
        // "Lead times are back within 6 to 8 weeks" — a re-check finds the
        // window has since widened.
        stale_marker: "6 to 8 weeks",
        correction: "Supply chain re-check (probe): component lead times have widened to 10–12 weeks; the earlier 6–8 week claim no longer holds.",
    },
    DemoProbeRule {
        // "Renewal pipeline for FY26 is already 65% covered" — re-check finds
        // the figure moved.
        stale_marker: "65% covered",
        correction: "Renewal re-check (probe): FY26 pipeline coverage is now 48%, not 65% — the earlier figure is stale.",
    },
];

/// A content-driven probe backed by [`DEMO_RULES`]. Built per request from the
/// crate's [`FakeProbe`] so the engine's tested logic is what actually runs.
fn demo_probe() -> impl Probe {
    let mut p = FakeProbe::new();
    for rule in DEMO_RULES {
        p = p.refute_on(rule.stale_marker);
    }
    p
}

/// Correction text for a refuted memory, by matching its content to a rule.
fn correction_for(content: &str) -> Option<String> {
    let lc = content.to_ascii_lowercase();
    DEMO_RULES
        .iter()
        .find(|r| lc.contains(&r.stale_marker.to_ascii_lowercase()))
        .map(|r| r.correction.to_string())
}

#[derive(Debug, Deserialize, Default)]
pub struct ProbeParams {
    /// When `true`, actually falsify refuted memories (append the supersede
    /// triple). Default `false` — a dry-run preview.
    #[serde(default)]
    pub apply: bool,
}

/// `GET /api/probe/metrics[?apply=true]`.
pub async fn probe_metrics(
    State(state): State<Arc<AppState>>,
    AxumQuery(params): AxumQuery<ProbeParams>,
) -> Response {
    let scope = Scope::global(&state.default_tenant);

    // Reconstruct the live corpus (written − invalidated) with content + the
    // provenance trust we'll use as each belief's prior (Hard Rule #4).
    let entries = match state.log.read_from(None).await {
        Ok(e) => e,
        Err(e) => {
            return Json(ProbeReportDto::disabled(format!("log read failed: {e}"))).into_response()
        }
    };
    struct Row {
        content: String,
        prior: f32,
    }
    let mut rows: HashMap<MemoryRef, Row> = HashMap::new();
    let mut order: Vec<MemoryRef> = Vec::new();
    for entry in &entries {
        match &entry.event {
            Event::MemoryWritten(m) if m.scope.tenant == scope.tenant => {
                let r = MemoryRef(m.id);
                if rows
                    .insert(
                        r,
                        Row {
                            content: m.content.clone(),
                            prior: m.provenance.trust,
                        },
                    )
                    .is_none()
                {
                    order.push(r);
                }
            }
            Event::MemoryInvalidated { id, .. } => {
                rows.remove(id);
            }
            _ => {}
        }
    }
    let live: Vec<MemoryRef> = order.into_iter().filter(|r| rows.contains_key(r)).collect();
    if live.is_empty() {
        return Json(ProbeReportDto::disabled(
            "no live memories in scope yet — try again once the demo corpus has streamed in"
                .to_string(),
        ))
        .into_response();
    }

    // Run the bounded probe pass over the live corpus.
    let candidates: Vec<(MemoryRef, String)> =
        live.iter().map(|r| (*r, rows[r].content.clone())).collect();
    let probe = demo_probe();
    let runner = ProbeRunner::new(ProbeConfig {
        max_probes_per_pass: PROBE_MAX,
    });
    let report = match runner.run(&probe, &candidates).await {
        Ok(r) => r,
        Err(e) => {
            return Json(ProbeReportDto::disabled(format!("probe pass failed: {e}")))
                .into_response()
        }
    };

    let snippet = |c: &str| -> String {
        let mut s: String = c.chars().take(110).collect();
        if c.chars().count() > 110 {
            s.push('…');
        }
        s
    };
    // Belief for one memory: prior from provenance, ProbePassed/Failed from
    // this pass's verdict. (A fuller calibration would also fold corroboration
    // from the graph; this is the probe-driven slice.)
    let belief_dto = |mem: &MemoryRef, status: ProbeStatus| -> BeliefDto {
        let prior = rows.get(mem).map(|r| r.prior).unwrap_or(0.5);
        let ev = match status {
            ProbeStatus::Held => vec![Evidence::ProbePassed],
            ProbeStatus::Refuted => vec![Evidence::ProbeFailed],
            ProbeStatus::Inconclusive => vec![],
        };
        let b = belief_of(prior, &ev);
        BeliefDto {
            confidence: b.confidence,
            prior: b.prior,
            doubted: b.is_doubted(DOUBT_FLOOR),
            why: b.rationale(),
        }
    };

    let mut outcomes_dto: Vec<ProbeOutcomeDto> = report
        .outcomes
        .iter()
        .map(|o| {
            let content = rows
                .get(&o.memory)
                .map(|r| r.content.as_str())
                .unwrap_or("");
            ProbeOutcomeDto {
                memory_id: o.memory.0.to_string(),
                status: status_str(o.verdict.status),
                reason: o.verdict.reason.clone(),
                snippet: snippet(content),
                belief: belief_dto(&o.memory, o.verdict.status),
            }
        })
        .collect();
    // Surface refuted first, then inconclusive, then held — most interesting
    // on top for the dashboard.
    outcomes_dto.sort_by_key(|o| match o.status.as_str() {
        "Refuted" => 0,
        "Inconclusive" => 1,
        _ => 2,
    });

    // One worked "done when" example: the first refuted memory, with the
    // correction that supersedes it and the before/after belief.
    let example = report.refuted().first().map(|o| {
        let content = rows
            .get(&o.memory)
            .map(|r| r.content.as_str())
            .unwrap_or("");
        let prior = rows.get(&o.memory).map(|r| r.prior).unwrap_or(0.5);
        let before = belief_of(prior, &[]);
        let after = belief_of(prior, &[Evidence::ProbeFailed]);
        FalsifyExampleDto {
            memory_id: o.memory.0.to_string(),
            stale_claim: snippet(content),
            correction: correction_for(content).unwrap_or_else(|| o.verdict.reason.clone()),
            confidence_before: before.confidence,
            confidence_after: after.confidence,
        }
    });

    // Apply path: actually falsify (append supersede triples) + fan to views.
    let mut superseded = 0usize;
    if params.apply {
        for o in report.refuted() {
            let content = rows
                .get(&o.memory)
                .map(|r| r.content.clone())
                .unwrap_or_default();
            let verdict = ProbeVerdict::refuted(o.verdict.reason.clone(), correction_for(&content));
            let one = vec![mneme_probe::ProbeOutcome {
                memory: o.memory,
                verdict,
            }];
            match mneme_probe::falsify(state.log.as_ref(), &scope, &one).await {
                Ok(out) => {
                    superseded += out.superseded;
                }
                Err(e) => {
                    return Json(ProbeReportDto::disabled(format!("falsify failed: {e}")))
                        .into_response()
                }
            }
        }
        // Catch the views up to the new head so search reflects the
        // supersession immediately (bm25 has no tailer; vector is also driven
        // by the embedding worker but apply() is idempotent here).
        if superseded > 0 {
            fan_new_entries(&state).await;
        }
    }

    let payload = ProbeReportDto {
        enabled: true,
        note: None,
        applied: params.apply,
        live_memories: live.len(),
        probed: report.candidates_probed,
        held: report.held_count(),
        refuted: report.refuted_count(),
        inconclusive: report.inconclusive_count(),
        superseded,
        doubt_floor: DOUBT_FLOOR,
        falsify_example: example,
        outcomes: outcomes_dto,
    };
    Json(payload).into_response()
}

/// Re-read the log tail and fan any not-yet-applied entries to the in-memory
/// retrieval views, so a freshly-applied supersession is searchable at once.
/// Cheap at demo scale (full replay); a production build would track a head.
async fn fan_new_entries(state: &Arc<AppState>) {
    let entries = match state.log.read_from(None).await {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(error = %e, "probe: tail read for view fan-out failed");
            return;
        }
    };
    for entry in &entries {
        apply_quiet(state.vector.apply(entry).await, "vector");
        apply_quiet(state.bm25.apply(entry).await, "bm25");
    }
}

fn apply_quiet(res: Result<(), mneme_core::MnemeError>, view: &str) {
    if let Err(e) = res {
        tracing::warn!(error = %e, view, "probe: view apply failed");
    }
}

fn status_str(s: ProbeStatus) -> String {
    match s {
        ProbeStatus::Held => "Held",
        ProbeStatus::Refuted => "Refuted",
        ProbeStatus::Inconclusive => "Inconclusive",
    }
    .to_string()
}

#[derive(Serialize)]
pub struct ProbeReportDto {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Whether this call performed falsification (`?apply=true`) or previewed.
    pub applied: bool,
    pub live_memories: usize,
    pub probed: usize,
    pub held: usize,
    pub refuted: usize,
    pub inconclusive: usize,
    /// Memories actually superseded this call (0 on a dry run).
    pub superseded: usize,
    pub doubt_floor: f32,
    /// One worked falsification example for the dashboard's "done when" card.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub falsify_example: Option<FalsifyExampleDto>,
    pub outcomes: Vec<ProbeOutcomeDto>,
}

impl ProbeReportDto {
    fn disabled(note: String) -> Self {
        Self {
            enabled: false,
            note: Some(note),
            applied: false,
            live_memories: 0,
            probed: 0,
            held: 0,
            refuted: 0,
            inconclusive: 0,
            superseded: 0,
            doubt_floor: DOUBT_FLOOR,
            falsify_example: None,
            outcomes: vec![],
        }
    }
}

#[derive(Serialize)]
pub struct ProbeOutcomeDto {
    pub memory_id: String,
    /// "Held" | "Refuted" | "Inconclusive".
    pub status: String,
    pub reason: String,
    pub snippet: String,
    pub belief: BeliefDto,
}

#[derive(Serialize)]
pub struct BeliefDto {
    pub confidence: f32,
    pub prior: f32,
    pub doubted: bool,
    pub why: String,
}

#[derive(Serialize)]
pub struct FalsifyExampleDto {
    pub memory_id: String,
    pub stale_claim: String,
    pub correction: String,
    pub confidence_before: f32,
    pub confidence_after: f32,
}
