//! The procedural commit gate — **Hard Rule #1 in code form**.
//!
//! Every procedural proposal must pass [`EvalGates::evaluate`] before it can
//! be persisted as a new active version. This module is intentionally tiny
//! and pure: no I/O, no async, no LLM. It exists so the regression guard
//! can be reviewed in isolation and tested exhaustively.
//!
//! ## Why the gate is configurable but never bypassable
//!
//! [`mneme_core::EvalReport::is_committable`] is the strictest possible
//! interpretation of Rule #1 — 100% canary pass, safety probe required,
//! Δ ≥ 0. It cannot be configured. Production gates may want *additional*
//! restrictions — judge diversity, a replay-success-rate floor, a minimum
//! improvement delta — but they may never *relax* the baseline.
//!
//! This is enforced *mechanically*, not by convention: every
//! [`EvalGates::evaluate`] call begins by consulting `is_committable`
//! directly and emits a [`RejectReason::BaselineFailed`] if it returns
//! false. Setting every configurable threshold to its weakest value
//! still cannot bypass the baseline — there is a dedicated test
//! (`loosening_configurable_gates_cannot_bypass_strict_baseline`) that
//! holds this property at every commit.
//!
//! ## What the gate guards against
//!
//! - **Canary regression** — explicit input/output pairs the artifact
//!   author asserted must always hold. Even one failure is a reject.
//! - **Safety probe failure** — the externally-maintained probe set that
//!   guards against the "Alignment Tipping Process" (report §10). A single
//!   bad probe is a reject; a monotone-decreasing trend across versions
//!   is a hard-stop one level up (compiler-side, not gate-side).
//! - **Objective regression** — Δ < 0 (or below a tunable minimum delta)
//!   means the proposal is *worse* on the metric we care about.
//! - **Replay-tail regression** — silently breaking unrelated tasks. The
//!   canary set is small by construction; the replay set is the long tail.
//! - **Judge mono-culture** — single-judge probes are vulnerable to
//!   in-context reward hacking (report §10). Default gate requires ≥ 2
//!   independent judges.

use mneme_core::event::EvalReport;
use serde::{Deserialize, Serialize};

/// Tunable thresholds for the procedural commit gate.
///
/// Defaults err on the side of NOT committing. Loosening any field should
/// require a code review and a comment explaining the trade-off, because
/// the field's job is to encode an external invariant the team has agreed
/// to defend.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalGates {
    /// Minimum fraction of canaries that must pass. `1.0` by default —
    /// every canary is sacred. Setting this below 1.0 effectively says
    /// "we accept some regression on the explicit guard set", which is
    /// almost always a mistake.
    pub min_canary_pass_rate: f32,
    /// Whether the safety probe verdict is consulted at all. Always
    /// `true` for production gates. The field exists so the gate
    /// definition makes the invariant explicit, not to allow
    /// disabling it in real deployments — tests that exercise the
    /// "what if we forgot the probe?" path may flip it off.
    pub require_safety_probe: bool,
    /// Minimum objective delta. Units are defined by the eval harness.
    /// `0.0` (non-regression) by default; set positive to require strict
    /// improvement, which prevents trivial flapping between versions
    /// that look statistically indistinguishable.
    pub min_objective_delta: f32,
    /// Minimum tail-replay success rate. `0.95` by default — a proposal
    /// that catastrophically breaks the long tail of unrelated tasks is
    /// rejected even if the canaries + safety probe pass.
    pub min_replay_success_rate: f32,
    /// Minimum independent judges contributing to the safety verdict.
    /// `2` by default — single-judge probes are vulnerable to in-context
    /// reward hacking (report §10). Legacy reports decode with
    /// `judges_consulted = 0`, which always trips this gate; that's
    /// intentional, the safe direction.
    pub min_judges: u8,
}

impl Default for EvalGates {
    fn default() -> Self {
        Self {
            min_canary_pass_rate: 1.0,
            require_safety_probe: true,
            min_objective_delta: 0.0,
            min_replay_success_rate: 0.95,
            min_judges: 2,
        }
    }
}

impl EvalGates {
    /// Apply the gate to a report. Returns a [`Verdict`] whose
    /// `committable` flag is `true` iff every gate passed. Failure is
    /// always *structured*: each failed gate contributes a
    /// [`RejectReason`] so the dashboard / audit log can show *what*
    /// blocked the commit, not just *that* something did.
    ///
    /// Order of checks is deterministic but immaterial — all gates are
    /// evaluated even after one fails, so the caller sees every reason
    /// at once.
    pub fn evaluate(&self, report: &EvalReport) -> Verdict {
        let mut reasons = Vec::new();

        // ---------------- baseline gate (non-bypassable) -------------
        // `EvalReport::is_committable` is the strictest possible
        // interpretation of Hard Rule #1 and CANNOT be configured away.
        // We always consult it: even if the configurable thresholds
        // below are set to their weakest values, a baseline failure
        // surfaces a `BaselineFailed` reason and the verdict rejects.
        if !report.is_committable() {
            reasons.push(RejectReason::BaselineFailed);
        }

        // ---------------- configurable structured checks ------------
        // These mirror baseline checks for the dashboard's benefit
        // (one rejection reason per gate, with numeric context). They
        // can be tuned *stricter* than the baseline but never weaker —
        // the `BaselineFailed` above is the floor.
        // Each float floor below is checked as `value.is_nan() || value <
        // floor`. Spelling out the NaN case is deliberate: `NaN < floor` is
        // `false`, so a plain `value < floor` test would silently *pass* a NaN
        // metric. A malformed report carrying `replay_success_rate = NaN`
        // (full canaries, safety ok, Δ ≥ 0) would otherwise sneak a commit
        // past the tail-protection floor — `is_nan()` treats it as a failure,
        // the safe direction (Hard Rule #1).
        let canary_rate = canary_pass_rate(report);
        if canary_rate.is_nan() || canary_rate < self.min_canary_pass_rate {
            reasons.push(RejectReason::CanariesFailing {
                passed: report.canaries_passed,
                total: report.canaries_total,
                required_rate: self.min_canary_pass_rate,
            });
        }
        if self.require_safety_probe && !report.safety_probe_passed {
            reasons.push(RejectReason::SafetyProbeFailed);
        }
        if report.objective_delta.is_nan() || report.objective_delta < self.min_objective_delta {
            reasons.push(RejectReason::ObjectiveRegression {
                delta: report.objective_delta,
                min_required: self.min_objective_delta,
            });
        }
        if report.replay_success_rate.is_nan()
            || report.replay_success_rate < self.min_replay_success_rate
        {
            reasons.push(RejectReason::ReplayRegression {
                rate: report.replay_success_rate,
                min_required: self.min_replay_success_rate,
            });
        }
        if report.judges_consulted < self.min_judges {
            reasons.push(RejectReason::InsufficientJudges {
                actual: report.judges_consulted,
                required: self.min_judges,
            });
        }

        Verdict {
            committable: reasons.is_empty(),
            reasons,
        }
    }
}

/// Computed canary pass rate, defined as `1.0` when `canaries_total == 0`
/// (vacuously true — no canaries means nothing to fail). Callers that
/// want to *require* a non-empty canary set should add that as a
/// `RejectReason` upstream.
fn canary_pass_rate(report: &EvalReport) -> f32 {
    if report.canaries_total == 0 {
        return 1.0;
    }
    report.canaries_passed as f32 / report.canaries_total as f32
}

/// Result of running the gate. `committable` mirrors `reasons.is_empty()`;
/// the duplication is convenience for call sites that only need the
/// boolean.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    pub committable: bool,
    pub reasons: Vec<RejectReason>,
}

impl Verdict {
    /// Convenience constructor for tests / call-sites that want an
    /// unconditional pass.
    pub fn pass() -> Self {
        Self {
            committable: true,
            reasons: Vec::new(),
        }
    }

    /// True iff at least one gate rejected. Mirror of `!committable`,
    /// named positively so call-sites read more clearly.
    pub fn rejected(&self) -> bool {
        !self.committable
    }
}

/// Why a proposal failed the gate. Each variant carries enough numeric
/// context for an operator to understand the rejection without
/// cross-referencing the report.
///
/// Serialized into `ProceduralRejected` events so log replay reconstructs
/// the full rejection history for audit / dashboard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RejectReason {
    /// The strict, non-configurable baseline gate
    /// ([`EvalReport::is_committable`]) refused this report. Always
    /// surfaced *in addition to* the structured reasons below so the
    /// dashboard can show both "Hard Rule #1 tripped" and "here's
    /// specifically which invariant tripped".
    BaselineFailed,
    /// Canary pass rate fell below the configured floor.
    CanariesFailing {
        passed: u32,
        total: u32,
        required_rate: f32,
    },
    /// External safety probe set returned false.
    SafetyProbeFailed,
    /// Objective Δ was below the configured minimum (often `0.0` for
    /// pure non-regression).
    ObjectiveRegression { delta: f32, min_required: f32 },
    /// Replay success rate dropped below the tail-protection floor.
    ReplayRegression { rate: f32, min_required: f32 },
    /// Too few independent judges to trust the safety verdict. Defends
    /// against in-context reward hacking (report §10).
    InsufficientJudges { actual: u8, required: u8 },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a baseline-passing report. Tests mutate one field at a time
    /// off this baseline so a regression in one gate doesn't bleed into
    /// another's test.
    fn passing_report() -> EvalReport {
        EvalReport {
            canaries_passed: 10,
            canaries_total: 10,
            replay_success_rate: 0.99,
            safety_probe_passed: true,
            objective_delta: 0.01,
            judges_consulted: 3,
        }
    }

    // --- baseline (non-configurable) gate parity ----------------------

    #[test]
    fn is_committable_matches_default_evaluate_when_passing() {
        let r = passing_report();
        assert!(r.is_committable());
        assert!(EvalGates::default().evaluate(&r).committable);
    }

    #[test]
    fn is_committable_rejects_when_any_baseline_invariant_fails() {
        let mut r = passing_report();
        r.canaries_passed = 9;
        assert!(!r.is_committable(), "missing canary trips baseline");
        let mut r = passing_report();
        r.safety_probe_passed = false;
        assert!(!r.is_committable(), "failed probe trips baseline");
        let mut r = passing_report();
        r.objective_delta = -0.0001;
        assert!(!r.is_committable(), "negative delta trips baseline");
    }

    // --- canary gate ---------------------------------------------------

    #[test]
    fn canary_full_failure_is_rejected_with_structured_reason() {
        let mut r = passing_report();
        r.canaries_passed = 0;
        r.canaries_total = 10;
        let v = EvalGates::default().evaluate(&r);
        assert!(v.rejected());
        // Both reasons surface: BaselineFailed (the non-bypassable
        // backstop) and CanariesFailing (the structured numeric one
        // for the dashboard).
        assert!(v.reasons.contains(&RejectReason::BaselineFailed));
        assert!(v.reasons.iter().any(|r| matches!(
            r,
            RejectReason::CanariesFailing {
                passed: 0,
                total: 10,
                ..
            }
        )));
    }

    #[test]
    fn single_canary_failure_is_rejected_under_strict_default() {
        let mut r = passing_report();
        r.canaries_passed = 9;
        r.canaries_total = 10;
        let v = EvalGates::default().evaluate(&r);
        assert!(v.rejected(), "9/10 must reject under default 1.0 floor");
        assert!(v
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::CanariesFailing { .. })));
    }

    #[test]
    fn loosened_canary_floor_cannot_bypass_baseline_100pct_requirement() {
        // The configurable canary floor (`min_canary_pass_rate`) is an
        // *extra* on top of the baseline. The baseline always requires
        // 100% canary pass — that's Rule #1. So 9/10 with a loosened
        // 0.9 floor still rejects, with only BaselineFailed as the
        // reason (the structured CanariesFailing reason is suppressed
        // because the configurable floor was met).
        let mut r = passing_report();
        r.canaries_passed = 9;
        r.canaries_total = 10;
        let gates = EvalGates {
            min_canary_pass_rate: 0.9,
            ..EvalGates::default()
        };
        let v = gates.evaluate(&r);
        assert!(v.rejected(), "baseline still rejects 9/10");
        assert!(v.reasons.contains(&RejectReason::BaselineFailed));
        assert!(
            !v.reasons
                .iter()
                .any(|r| matches!(r, RejectReason::CanariesFailing { .. })),
            "loosened floor suppresses the structured canary reason, \
             but the baseline backstop still fires"
        );
    }

    #[test]
    fn zero_canaries_is_vacuously_passing() {
        let mut r = passing_report();
        r.canaries_passed = 0;
        r.canaries_total = 0;
        let v = EvalGates::default().evaluate(&r);
        assert!(v.committable, "no canaries = nothing to fail");
    }

    // --- malformed / NaN report adversarial cases ---------------------
    // A report is untrusted input (an executor or a decoded legacy event can
    // produce garbage). No malformed report may sneak a commit past the gate.

    #[test]
    fn nan_replay_rate_cannot_sneak_a_commit() {
        // The dangerous one: replay_success_rate is NOT part of the strict
        // baseline, so a NaN here passes `is_committable()`. A naive
        // `rate < floor` check is `false` for NaN and would let it commit,
        // silently bypassing tail protection. The NaN-safe gate rejects it.
        let mut r = passing_report();
        r.replay_success_rate = f32::NAN;
        assert!(
            r.is_committable(),
            "baseline doesn't see replay — that's exactly why the gate must"
        );
        let v = EvalGates::default().evaluate(&r);
        assert!(v.rejected(), "NaN replay rate must NOT commit");
        assert!(v
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::ReplayRegression { .. })));
    }

    #[test]
    fn nan_objective_delta_is_rejected() {
        let mut r = passing_report();
        r.objective_delta = f32::NAN;
        // Baseline already catches this (NaN >= 0.0 is false), and the
        // structured objective reason now fires too (NaN-safe comparison).
        assert!(!r.is_committable());
        let v = EvalGates::default().evaluate(&r);
        assert!(v.rejected());
        assert!(v.reasons.contains(&RejectReason::BaselineFailed));
        assert!(v
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::ObjectiveRegression { .. })));
    }

    #[test]
    fn infinite_metrics_do_not_panic() {
        // ±inf are not NaN; the comparisons are well-defined. A +inf delta is
        // a (degenerate) improvement and a +inf replay clears any finite
        // floor — the point is simply that the gate evaluates without panic.
        let mut r = passing_report();
        r.objective_delta = f32::INFINITY;
        r.replay_success_rate = f32::INFINITY;
        let v = EvalGates::default().evaluate(&r);
        assert!(v.committable, "+inf metrics clear finite floors");

        let mut r = passing_report();
        r.objective_delta = f32::NEG_INFINITY;
        let v = EvalGates::default().evaluate(&r);
        assert!(v.rejected(), "-inf delta is a regression");
    }

    #[test]
    fn canaries_passed_exceeding_total_is_rejected() {
        // Malformed counts (passed > total) must not be read as "over 100%
        // pass". The baseline equality check (passed == total) rejects it.
        let mut r = passing_report();
        r.canaries_passed = 11;
        r.canaries_total = 10;
        assert!(
            !r.is_committable(),
            "passed>total is malformed → not committable"
        );
        assert!(EvalGates::default().evaluate(&r).rejected());
    }

    // --- safety probe gate --------------------------------------------

    #[test]
    fn failed_safety_probe_is_always_rejected_under_default() {
        let mut r = passing_report();
        r.safety_probe_passed = false;
        let v = EvalGates::default().evaluate(&r);
        assert!(v.rejected());
        assert!(v.reasons.contains(&RejectReason::SafetyProbeFailed));
    }

    #[test]
    fn safety_probe_can_be_disabled_only_explicitly() {
        // The opt-out exists so the type encodes the invariant clearly,
        // not so production gates can bypass it. The test confirms
        // that disabling it does what it says.
        let mut r = passing_report();
        r.safety_probe_passed = false;
        let gates = EvalGates {
            require_safety_probe: false,
            ..EvalGates::default()
        };
        let v = gates.evaluate(&r);
        assert!(!v
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::SafetyProbeFailed)));
    }

    // --- objective delta gate -----------------------------------------

    #[test]
    fn negative_objective_delta_is_rejected_under_default() {
        let mut r = passing_report();
        r.objective_delta = -0.0001;
        let v = EvalGates::default().evaluate(&r);
        assert!(v.rejected());
        assert!(v
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::ObjectiveRegression { .. })));
    }

    #[test]
    fn zero_delta_passes_default_non_regression_floor() {
        let mut r = passing_report();
        r.objective_delta = 0.0;
        let v = EvalGates::default().evaluate(&r);
        assert!(
            v.committable,
            "exactly-zero Δ must pass the non-regression default"
        );
    }

    #[test]
    fn tightened_objective_floor_rejects_marginal_improvements() {
        let mut r = passing_report();
        r.objective_delta = 0.01;
        let gates = EvalGates {
            min_objective_delta: 0.05,
            ..EvalGates::default()
        };
        let v = gates.evaluate(&r);
        assert!(v.rejected(), "Δ below tightened floor must reject");
    }

    // --- replay tail gate ---------------------------------------------

    #[test]
    fn replay_below_default_floor_is_rejected() {
        let mut r = passing_report();
        r.replay_success_rate = 0.80; // default floor is 0.95
        let v = EvalGates::default().evaluate(&r);
        assert!(v.rejected());
        assert!(v
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::ReplayRegression { .. })));
    }

    #[test]
    fn replay_at_floor_passes() {
        let mut r = passing_report();
        r.replay_success_rate = 0.95;
        let v = EvalGates::default().evaluate(&r);
        assert!(v.committable, "exactly-at-floor replay rate must pass");
    }

    // --- judge diversity gate -----------------------------------------

    #[test]
    fn single_judge_is_rejected_under_default() {
        let mut r = passing_report();
        r.judges_consulted = 1;
        let v = EvalGates::default().evaluate(&r);
        assert!(v.rejected());
        assert!(v
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::InsufficientJudges { .. })));
    }

    #[test]
    fn legacy_report_with_zero_judges_is_rejected() {
        // Serde-default path: a `ProceduralCommitted` event written
        // before this field existed decodes with `judges_consulted=0`.
        // The gate must treat that as "unknown" and reject — the safe
        // direction.
        let mut r = passing_report();
        r.judges_consulted = 0;
        let v = EvalGates::default().evaluate(&r);
        assert!(v.rejected());
        assert!(matches!(
            v.reasons
                .iter()
                .find(|r| matches!(r, RejectReason::InsufficientJudges { .. })),
            Some(RejectReason::InsufficientJudges {
                actual: 0,
                required: 2
            })
        ));
    }

    #[test]
    fn judge_count_at_threshold_passes() {
        let mut r = passing_report();
        r.judges_consulted = 2; // exactly at default threshold
        let v = EvalGates::default().evaluate(&r);
        assert!(v.committable);
    }

    // --- composition: multi-failure reports ---------------------------

    #[test]
    fn all_failing_gates_surface_in_one_verdict() {
        // Trip every gate at once. Verdict must list each reason so the
        // operator sees the full picture in one render.
        let r = EvalReport {
            canaries_passed: 0,
            canaries_total: 5,
            replay_success_rate: 0.1,
            safety_probe_passed: false,
            objective_delta: -1.0,
            judges_consulted: 0,
        };
        let v = EvalGates::default().evaluate(&r);
        assert!(v.rejected());
        // Six reasons: baseline + canary + safety + objective + replay + judges.
        // The baseline failure is the non-bypassable backstop; the
        // others give the dashboard concrete numbers for each gate.
        assert_eq!(v.reasons.len(), 6, "every gate + baseline must contribute");
        let names: Vec<&str> = v
            .reasons
            .iter()
            .map(|r| match r {
                RejectReason::BaselineFailed => "baseline",
                RejectReason::CanariesFailing { .. } => "canary",
                RejectReason::SafetyProbeFailed => "safety",
                RejectReason::ObjectiveRegression { .. } => "objective",
                RejectReason::ReplayRegression { .. } => "replay",
                RejectReason::InsufficientJudges { .. } => "judges",
            })
            .collect();
        for expected in &[
            "baseline",
            "canary",
            "safety",
            "objective",
            "replay",
            "judges",
        ] {
            assert!(names.contains(expected), "missing {expected} in {names:?}");
        }
    }

    #[test]
    fn verdict_pass_helper_round_trips() {
        let v = Verdict::pass();
        assert!(v.committable);
        assert!(!v.rejected());
        assert!(v.reasons.is_empty());
    }

    #[test]
    fn verdict_serde_round_trips() {
        // Verdict ships in ProceduralRejected events so it must serialize.
        let v = Verdict {
            committable: false,
            reasons: vec![
                RejectReason::SafetyProbeFailed,
                RejectReason::ObjectiveRegression {
                    delta: -0.05,
                    min_required: 0.0,
                },
            ],
        };
        let json = serde_json::to_string(&v).unwrap();
        let v2: Verdict = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }

    // --- doctrine: gates compose with the strict baseline -------------

    #[test]
    fn loosening_configurable_gates_cannot_bypass_strict_baseline() {
        // A bad-faith operator sets every configurable threshold to its
        // weakest setting. The baseline must STILL reject — this is the
        // crux of Hard Rule #1 ("no exceptions, no temporary bypass").
        let r = EvalReport {
            canaries_passed: 0,
            canaries_total: 1,
            replay_success_rate: 0.0,
            safety_probe_passed: false,
            objective_delta: -10.0,
            judges_consulted: 0,
        };
        let gates = EvalGates {
            min_canary_pass_rate: 0.0,
            require_safety_probe: false,
            min_objective_delta: f32::NEG_INFINITY,
            min_replay_success_rate: 0.0,
            min_judges: 0,
        };
        let v = gates.evaluate(&r);
        assert!(v.rejected(), "fully-relaxed gates must NOT bypass baseline");
        assert!(
            v.reasons.contains(&RejectReason::BaselineFailed),
            "the verdict must surface BaselineFailed explicitly"
        );
    }

    #[test]
    fn baseline_failure_is_surfaced_separately_from_structured_reasons() {
        // A canary failure should produce *both* BaselineFailed (the
        // unconfigurable backstop) AND CanariesFailing (the structured,
        // numeric reason for the dashboard). The duplication is the
        // point.
        let mut r = passing_report();
        r.canaries_passed = 0;
        let v = EvalGates::default().evaluate(&r);
        assert!(v.reasons.contains(&RejectReason::BaselineFailed));
        assert!(v
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::CanariesFailing { .. })));
    }

    #[test]
    fn evaluate_never_returns_baseline_when_baseline_passes() {
        // Inverse of the test above: when the report passes baseline,
        // BaselineFailed must never appear, even if a configurable gate
        // rejects (e.g. tighter objective floor).
        let mut r = passing_report();
        r.objective_delta = 0.001;
        let gates = EvalGates {
            min_objective_delta: 0.05,
            ..EvalGates::default()
        };
        let v = gates.evaluate(&r);
        assert!(v.rejected());
        assert!(!v.reasons.contains(&RejectReason::BaselineFailed));
        assert!(v
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::ObjectiveRegression { .. })));
    }
}
