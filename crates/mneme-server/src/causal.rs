//! Phase 10 — live counterfactual contribution endpoint.
//!
//! `GET /api/causal/metrics` runs an on-demand [`mneme_causal`] contribution
//! pass over the *live* corpus using the real [`HybridRetriever`], and reports
//! each memory's measured causal contribution to recall@k — the "replay
//! dividend" made visible.
//!
//! It is strictly **read-only**: it scores and reports, it never appends a
//! `MemoryInvalidated` (GC is a deliberate, separate action — never a side
//! effect of a GET). The scoring pass is bounded ([`CausalConfig`]) and runs
//! only when this endpoint is hit, so it stays off the write path (Hard
//! Rule #5) and respects the cascade bound (#6).
//!
//! Ground truth is resolved from memory *content* (substring match), not from
//! hardcoded ULIDs, so the demo is fully replayable and id-agnostic.

use crate::viz::AppState;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use mneme_causal::{CausalConfig, ContributionScorer, RetrievalEvaluator};
use mneme_core::event::Event;
use mneme_core::types::MemoryRef;
use mneme_core::{Retriever, Scope};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;

/// recall@k cutoff for the contribution objective.
const CAUSAL_K: usize = 5;
/// Cap on memories scored per pass — bounds cost of the on-demand endpoint.
const CAUSAL_MAX_CANDIDATES: usize = 64;
/// How many top-ranked contributions to surface in the response.
const TOP_N: usize = 12;

/// A held-out probe: a natural-language query plus the content substrings that
/// mark a live memory as a *gold* answer for it. Resolving gold ids from
/// content keeps the demo replayable (no ULIDs baked into source).
struct CausalProbe {
    query: &'static str,
    /// A memory is relevant iff its content contains *all* of these
    /// (case-insensitive).
    gold_substrings: &'static [&'static str],
}

/// Probes over the demo corpus (see `demo.rs`). Each targets a small, distinct
/// relevant set so leave-one-out masking produces a clear signal.
const DEMO_PROBES: &[CausalProbe] = &[
    CausalProbe {
        query: "what was Acme Q3 revenue?",
        gold_substrings: &["acme", "revenue"],
    },
    CausalProbe {
        query: "how did Widget Inc perform in EMEA?",
        gold_substrings: &["widget inc", "emea"],
    },
    CausalProbe {
        query: "what is the supply chain status?",
        gold_substrings: &["supply chain"],
    },
    CausalProbe {
        query: "competitor market share movement?",
        gold_substrings: &["competitor", "market share"],
    },
    CausalProbe {
        query: "what is the FY26 revenue growth guidance?",
        gold_substrings: &["fy26", "growth"],
    },
];

/// `GET /api/causal/metrics`.
pub async fn causal_metrics(State(state): State<Arc<AppState>>) -> Response {
    let scope = Scope::global(&state.default_tenant);

    // Reconstruct the live corpus for this tenant from the log (Hard Rule #4:
    // a view derivable by replay). live = written − invalidated.
    let entries = match state.log.read_from(None).await {
        Ok(e) => e,
        Err(e) => {
            return Json(CausalReport::disabled(format!("log read failed: {e}"))).into_response();
        }
    };
    let mut content: HashMap<MemoryRef, String> = HashMap::new();
    let mut order: Vec<MemoryRef> = Vec::new();
    for entry in &entries {
        match &entry.event {
            Event::MemoryWritten(m) if m.scope.tenant == scope.tenant => {
                let r = MemoryRef(m.id);
                if content.insert(r, m.content.clone()).is_none() {
                    order.push(r);
                }
            }
            Event::MemoryInvalidated { id, .. } => {
                content.remove(id);
            }
            _ => {}
        }
    }
    let live: Vec<MemoryRef> = order
        .into_iter()
        .filter(|r| content.contains_key(r))
        .collect();

    if live.is_empty() {
        return Json(CausalReport::disabled(
            "no live memories in scope yet — try again once the demo corpus has streamed in"
                .to_string(),
        ))
        .into_response();
    }

    // Build the held-out task set: resolve each probe's gold set by content
    // match against the live corpus. Drop probes with no resolvable gold.
    let retriever_dyn: Arc<dyn Retriever> = state.retriever.clone();
    let mut evaluator = RetrievalEvaluator::new(retriever_dyn, scope, CAUSAL_K);
    let mut active_probes = 0usize;
    for probe in DEMO_PROBES {
        let gold: Vec<MemoryRef> = live
            .iter()
            .copied()
            .filter(|r| {
                let c = content[r].to_ascii_lowercase();
                probe.gold_substrings.iter().all(|s| c.contains(s))
            })
            .collect();
        if !gold.is_empty() {
            evaluator = evaluator.with_task(probe.query, gold);
            active_probes += 1;
        }
    }

    if active_probes == 0 {
        return Json(CausalReport::disabled(
            "no probes resolved a gold memory in the live corpus".to_string(),
        ))
        .into_response();
    }

    // Score every live memory's contribution (bounded). The candidate list is
    // the corpus itself — exactly what a production GC pass scores.
    let cfg = CausalConfig {
        max_candidates: CAUSAL_MAX_CANDIDATES,
        ..CausalConfig::default()
    };
    let report = match ContributionScorer::new(cfg).score(&evaluator, &live).await {
        Ok(r) => r,
        Err(e) => {
            return Json(CausalReport::disabled(format!("scoring failed: {e}"))).into_response();
        }
    };

    let snippet = |r: &MemoryRef| -> String {
        let c = content.get(r).map(String::as_str).unwrap_or("");
        let mut s: String = c.chars().take(110).collect();
        if c.chars().count() > 110 {
            s.push('…');
        }
        s
    };

    let ranked = report.ranked();
    let top: Vec<ContributionDto> = ranked
        .iter()
        .take(TOP_N)
        .map(|c| ContributionDto {
            memory_id: c.memory.0.to_string(),
            contribution: c.contribution,
            masked_recall: c.masked_score,
            snippet: snippet(&c.memory),
        })
        .collect();

    let gc = report.gc_candidates(CausalConfig::default().epsilon);

    // The two halves of the Phase-10 "done when", drawn straight from the
    // report: the top contributor (removing it drops recall) and a
    // zero-contributor (pruning it moves nothing).
    let high = ranked.first().map(|c| DoneWhenDto {
        memory_id: c.memory.0.to_string(),
        snippet: snippet(&c.memory),
        baseline_recall: report.baseline_score,
        masked_recall: c.masked_score,
        delta: c.contribution,
    });
    let zero = ranked
        .iter()
        .rev()
        .find(|c| c.contribution <= CausalConfig::default().epsilon)
        .map(|c| DoneWhenDto {
            memory_id: c.memory.0.to_string(),
            snippet: snippet(&c.memory),
            baseline_recall: report.baseline_score,
            masked_recall: c.masked_score,
            delta: c.contribution,
        });

    let payload = CausalReport {
        enabled: true,
        note: None,
        k: CAUSAL_K,
        probes: active_probes,
        baseline_recall: report.baseline_score,
        candidates_scored: report.candidates_scored,
        candidates_considered: report.candidates_considered,
        gc_candidate_count: gc.len(),
        top_contributors: top,
        done_when_high_contributor: high,
        done_when_zero_contributor: zero,
    };
    Json(payload).into_response()
}

#[derive(Serialize)]
pub struct CausalReport {
    pub enabled: bool,
    /// Set when `enabled == false` to explain why.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub k: usize,
    pub probes: usize,
    /// recall@k with nothing masked — the baseline the deltas are measured
    /// against.
    pub baseline_recall: f32,
    pub candidates_scored: usize,
    pub candidates_considered: usize,
    /// Memories whose contribution is ≤ epsilon (provable GC candidates).
    pub gc_candidate_count: usize,
    pub top_contributors: Vec<ContributionDto>,
    /// Phase-10 "done when" #1: removing this drops recall.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_when_high_contributor: Option<DoneWhenDto>,
    /// Phase-10 "done when" #2: pruning this moves recall by ~0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_when_zero_contributor: Option<DoneWhenDto>,
}

impl CausalReport {
    fn disabled(note: String) -> Self {
        Self {
            enabled: false,
            note: Some(note),
            k: CAUSAL_K,
            probes: 0,
            baseline_recall: 0.0,
            candidates_scored: 0,
            candidates_considered: 0,
            gc_candidate_count: 0,
            top_contributors: vec![],
            done_when_high_contributor: None,
            done_when_zero_contributor: None,
        }
    }
}

#[derive(Serialize)]
pub struct ContributionDto {
    pub memory_id: String,
    /// baseline_recall − masked_recall. Positive = load-bearing.
    pub contribution: f32,
    /// recall@k with this memory masked.
    pub masked_recall: f32,
    pub snippet: String,
}

#[derive(Serialize)]
pub struct DoneWhenDto {
    pub memory_id: String,
    pub snippet: String,
    pub baseline_recall: f32,
    pub masked_recall: f32,
    /// baseline − masked. ≈0 for the zero-contributor, >0 for the high one.
    pub delta: f32,
}
