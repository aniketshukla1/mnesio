//! Phase 14 — live negative-memory + dreaming endpoint.
//!
//! `GET /api/dream/metrics` demonstrates, in one read-only pass over the live
//! corpus, both Phase-14 halves:
//!
//! - **Anti-memory** — learn a gated suppression rule from a (modelled) bad
//!   outcome: a beneficial, canary-safe suppression *commits*, while a
//!   canary-breaking suppression is *rejected* by the gate (Hard Rule #1).
//! - **Dreaming** — score the live corpus's counterfactual contribution via
//!   the real [`HybridRetriever`] (Phase 10), then plan a bounded dream pass:
//!   prune the provable dead/harmful weight and re-anchor drifted evolution
//!   notes to their parent, reporting the expected next-generation lift.
//!
//! Read-only: the learner / evaluators are ephemeral per request and the dream
//! pass is only *planned* (not applied), so a GET appends nothing. A real
//! scheduled worker would call `mnesio_dream::dream(...)` to apply the plan.

use crate::viz::AppState;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use mnesio_causal::{CausalConfig, ContributionScorer, RetrievalEvaluator};
use mnesio_core::event::Event;
use mnesio_core::types::MemoryRef;
use mnesio_core::{Retriever, Scope};
use mnesio_dream::{
    BadOutcome, DreamConfig, DreamPass, DriftedNote, FakeSuppressionEvaluator, SuppressConfig,
    SuppressionLearner,
};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;

const DREAM_K: usize = 5;
const CAUSAL_MAX: usize = 64;
/// Demo query-class for the anti-memory half.
const SUPPRESS_CLASS: &str = "pricing";

/// Held-out probes (mirror the causal endpoint) so the contribution scorer has
/// real recall@k signal over the demo corpus.
const PROBES: &[(&str, &[&str])] = &[
    ("what was Acme Q3 revenue?", &["acme", "revenue"]),
    (
        "how did Widget Inc perform in EMEA?",
        &["widget inc", "emea"],
    ),
    ("what is the supply chain status?", &["supply chain"]),
    (
        "competitor market share movement?",
        &["competitor", "market share"],
    ),
];

/// `GET /api/dream/metrics`.
pub async fn dream_metrics(State(state): State<Arc<AppState>>) -> Response {
    let scope = Scope::global(&state.default_tenant);

    // Reconstruct the live corpus + the evolution lineage (parent/child +
    // evolution_count) from the log (Hard Rule #4).
    let entries = match state.log.read_from(None).await {
        Ok(e) => e,
        Err(e) => {
            return Json(DreamReportDto::disabled(format!("log read failed: {e}"))).into_response()
        }
    };
    struct Row {
        content: String,
        parent: Option<MemoryRef>,
        evolution_count: u16,
        links: Vec<MemoryRef>,
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
                            parent: m.parent,
                            evolution_count: m.evolution_count,
                            links: m.links.clone(),
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
        return Json(DreamReportDto::disabled(
            "no live memories in scope yet — wait for the demo corpus to stream in".to_string(),
        ))
        .into_response();
    }

    // ---------------- anti-memory half ----------------
    // Pick a real live memory to model as "misleading for the pricing class".
    let target = live[0];
    let learner = SuppressionLearner::new(SuppressConfig::default());

    // (a) a beneficial, canary-safe suppression → should COMMIT.
    let good_eval = FakeSuppressionEvaluator::new().beneficial(SUPPRESS_CLASS, target, 0.2);
    let good = learner
        .learn(
            &good_eval,
            &[BadOutcome {
                query_class: SUPPRESS_CLASS.into(),
                memory: target,
            }],
        )
        .await
        .unwrap_or_default();
    let suppress_committed = good.first().map(|o| o.committed()).unwrap_or(false);

    // (b) a canary-breaking suppression → gate must REJECT.
    let bad_eval = FakeSuppressionEvaluator::new().breaks_canary(SUPPRESS_CLASS, target);
    let bad = learner
        .learn(
            &bad_eval,
            &[BadOutcome {
                query_class: SUPPRESS_CLASS.into(),
                memory: target,
            }],
        )
        .await
        .unwrap_or_default();
    let canary_breaker_rejected = bad.first().map(|o| !o.committed()).unwrap_or(false);

    // ---------------- dreaming half ----------------
    // Score live contribution via the real retriever (Phase 10).
    let retriever_dyn: Arc<dyn Retriever> = state.retriever.clone();
    let mut evaluator = RetrievalEvaluator::new(retriever_dyn, scope.clone(), DREAM_K);
    let mut probes_active = 0usize;
    for (q, golds) in PROBES {
        let gold: Vec<MemoryRef> = live
            .iter()
            .copied()
            .filter(|r| {
                let c = rows[r].content.to_ascii_lowercase();
                golds.iter().all(|g| c.contains(g))
            })
            .collect();
        if !gold.is_empty() {
            evaluator = evaluator.with_task(*q, gold);
            probes_active += 1;
        }
    }

    let contributions = match ContributionScorer::new(CausalConfig {
        max_candidates: CAUSAL_MAX,
        ..CausalConfig::default()
    })
    .score(&evaluator, &live)
    .await
    {
        Ok(r) => r,
        Err(e) => {
            return Json(DreamReportDto::disabled(format!(
                "contribution scoring failed: {e}"
            )))
            .into_response()
        }
    };

    // Demo-tuned dream config:
    // - `max_prune` is capped low so the pass prunes an *illustrative sample*
    //   of dead/inert weight rather than the whole tail. (Under a 4-probe demo
    //   suite most memories aren't gold for any probe, so the inert set is
    //   large; a low cap keeps the demo honest about pruning a slice.)
    // - `drift_threshold = 1` matches the demo's evolution children (which have
    //   evolution_count = 1) so re-anchoring is visible on real lineage.
    let cfg = DreamConfig {
        max_prune: 5,
        drift_threshold: 1,
        ..DreamConfig::default()
    };
    // Drifted notes: live memories with a parent (the demo's evolution chains).
    let drifted: Vec<DriftedNote> = live
        .iter()
        .filter_map(|r| {
            let row = &rows[r];
            row.parent.map(|parent| DriftedNote {
                child: *r,
                parent,
                evolution_count: row.evolution_count,
                current_links: row.links.clone(),
            })
        })
        .collect();
    let drifted_total = drifted.len();

    let plan = DreamPass::new(cfg).plan(&contributions, &drifted);

    let snippet = |r: &MemoryRef| -> String {
        let c = rows.get(r).map(|x| x.content.as_str()).unwrap_or("");
        let mut s: String = c.chars().take(90).collect();
        if c.chars().count() > 90 {
            s.push('…');
        }
        s
    };
    let prune_examples: Vec<String> = plan.prune.iter().take(5).map(&snippet).collect();
    let reanchor_examples: Vec<String> = plan
        .reanchor
        .iter()
        .take(5)
        .map(|a| snippet(&a.child))
        .collect();

    // The Phase-14 "done when", both halves: anti-memory gates correctly
    // (beneficial suppression commits, canary-breaker rejected), AND the dream
    // pass does real consolidation work (prunes provable dead/harmful weight
    // and re-anchors at least one drifted note — the cascade-divergence fix).
    let done_when = suppress_committed
        && canary_breaker_rejected
        && plan.pruned_count() > 0
        && plan.reanchored_count() > 0;

    let payload = DreamReportDto {
        enabled: true,
        note: None,
        live_memories: live.len(),
        // anti-memory
        suppress_class: SUPPRESS_CLASS.to_string(),
        suppress_target: target.0.to_string(),
        suppress_committed,
        canary_breaker_rejected,
        // dreaming
        probes: probes_active,
        baseline_recall: plan.baseline_score,
        candidates_scored: plan.candidates_considered,
        pruned: plan.pruned_count(),
        generation_delta: plan.generation_delta,
        drifted_total,
        reanchored: plan.reanchored_count(),
        prune_examples,
        reanchor_examples,
        done_when,
    };
    Json(payload).into_response()
}

#[derive(Serialize)]
pub struct DreamReportDto {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub live_memories: usize,

    // anti-memory half
    pub suppress_class: String,
    pub suppress_target: String,
    pub suppress_committed: bool,
    pub canary_breaker_rejected: bool,

    // dreaming half
    pub probes: usize,
    pub baseline_recall: f32,
    pub candidates_scored: usize,
    pub pruned: usize,
    /// Σ max(0, −contribution) over pruned — expected next-generation lift.
    pub generation_delta: f32,
    pub drifted_total: usize,
    pub reanchored: usize,
    pub prune_examples: Vec<String>,
    pub reanchor_examples: Vec<String>,

    /// Phase-14 "done when" (anti-memory half): beneficial suppression commits,
    /// canary-breaker rejected.
    pub done_when: bool,
}

impl DreamReportDto {
    fn disabled(note: String) -> Self {
        Self {
            enabled: false,
            note: Some(note),
            live_memories: 0,
            suppress_class: SUPPRESS_CLASS.to_string(),
            suppress_target: String::new(),
            suppress_committed: false,
            canary_breaker_rejected: false,
            probes: 0,
            baseline_recall: 0.0,
            candidates_scored: 0,
            pruned: 0,
            generation_delta: 0.0,
            drifted_total: 0,
            reanchored: 0,
            prune_examples: vec![],
            reanchor_examples: vec![],
            done_when: false,
        }
    }
}
