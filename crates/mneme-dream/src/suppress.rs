//! Anti-memory: learn gated suppression rules from bad outcomes.
//!
//! A [`BadOutcome`] records "retrieving `memory` for `query_class` hurt". The
//! [`SuppressionLearner`] proposes a suppression [`SuppressionRule`] (a
//! `RetrievalRule` artifact that rewrites a query-class to *exclude* the
//! offending memory), re-evaluates it through a [`SuppressionEvaluator`], and
//! commits it **only if the re-eval is committable** (Hard Rule #1). A
//! suppression that would regress a canary is rejected — anti-memory can't
//! blind the agent to something it still needs.

use async_trait::async_trait;
use mneme_core::entity::ArtifactKind;
use mneme_core::event::EvalReport;
use mneme_core::types::MemoryRef;
use mneme_core::MnemeError;
use mneme_procedural::{EvalGates, RejectReason};
use serde::{Deserialize, Serialize};

/// One observed bad outcome: surfacing `memory` for queries in `query_class`
/// led to a worse result. The unit anti-memory learns from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BadOutcome {
    /// A coarse query category (e.g. `"pricing"`, `"q3-earnings"`). Suppression
    /// is scoped to a class, never a single phrasing.
    pub query_class: String,
    /// The memory whose retrieval, for this class, hurt outcomes.
    pub memory: MemoryRef,
}

/// A proposed/committed suppression: for `query_class`, drop `memory` from the
/// candidate set. Rendered as a `RetrievalRule` artifact so it rides the same
/// store + injection path as every other policy artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuppressionRule {
    pub query_class: String,
    pub memory: MemoryRef,
}

impl SuppressionRule {
    /// Render as the artifact kind the procedural store understands. The
    /// `rewrite` encodes the exclusion directive a retriever applies.
    pub fn to_artifact_kind(&self) -> ArtifactKind {
        ArtifactKind::RetrievalRule {
            query_pattern: self.query_class.clone(),
            rewrite: format!("-exclude:{}", self.memory.0),
        }
    }
}

/// Re-evaluate a candidate suppression: what does the gate-relevant report
/// look like if we apply this rule? The seam (Hard Rule #7). A real impl
/// re-runs canaries + the objective with the memory excluded for the class;
/// [`FakeSuppressionEvaluator`] models it deterministically.
#[async_trait]
pub trait SuppressionEvaluator: Send + Sync {
    async fn evaluate(&self, rule: &SuppressionRule) -> Result<EvalReport, MnemeError>;
}

/// Bounds for a learning pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressConfig {
    /// Max suppression rules committed per pass — cascade bound (#6).
    pub max_rules_per_pass: usize,
    /// Gate the learner applies to each candidate's re-eval.
    #[serde(skip, default)]
    pub gates: EvalGates,
}

impl Default for SuppressConfig {
    fn default() -> Self {
        Self {
            max_rules_per_pass: 16,
            gates: EvalGates::default(),
        }
    }
}

/// The result of evaluating one candidate suppression.
#[derive(Debug, Clone)]
pub enum SuppressionOutcome {
    /// The re-eval passed the gate; the rule is cleared to commit.
    Committed {
        rule: SuppressionRule,
        report: EvalReport,
    },
    /// The re-eval failed the gate; the rule is refused (e.g. it would regress
    /// a canary). Carries the structured reasons.
    Rejected {
        rule: SuppressionRule,
        report: EvalReport,
        reasons: Vec<RejectReason>,
    },
}

impl SuppressionOutcome {
    pub fn committed(&self) -> bool {
        matches!(self, SuppressionOutcome::Committed { .. })
    }
    pub fn rule(&self) -> &SuppressionRule {
        match self {
            SuppressionOutcome::Committed { rule, .. }
            | SuppressionOutcome::Rejected { rule, .. } => rule,
        }
    }
}

/// Learns gated suppression rules from bad outcomes.
pub struct SuppressionLearner {
    cfg: SuppressConfig,
}

impl SuppressionLearner {
    pub fn new(cfg: SuppressConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &SuppressConfig {
        &self.cfg
    }

    /// For each distinct `(query_class, memory)` in `bad_outcomes` (deduped,
    /// up to the cascade bound), propose a suppression, re-evaluate it, and
    /// gate it. Returns one [`SuppressionOutcome`] per evaluated candidate —
    /// committed *only* when the re-eval is committable (Hard Rule #1).
    pub async fn learn(
        &self,
        evaluator: &dyn SuppressionEvaluator,
        bad_outcomes: &[BadOutcome],
    ) -> Result<Vec<SuppressionOutcome>, MnemeError> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for bad in bad_outcomes {
            let key = (bad.query_class.clone(), bad.memory);
            if !seen.insert(key) {
                continue; // dedup repeated complaints
            }
            if out.len() >= self.cfg.max_rules_per_pass {
                break;
            }
            let rule = SuppressionRule {
                query_class: bad.query_class.clone(),
                memory: bad.memory,
            };
            let report = evaluator.evaluate(&rule).await?;
            let verdict = self.cfg.gates.evaluate(&report);
            if verdict.committable {
                out.push(SuppressionOutcome::Committed { rule, report });
            } else {
                out.push(SuppressionOutcome::Rejected {
                    rule,
                    report,
                    reasons: verdict.reasons,
                });
            }
        }
        Ok(out)
    }
}

/// Deterministic, dependency-free [`SuppressionEvaluator`] for tests + demos.
///
/// Keyed by `(query_class, memory.0-string)` → the report that applying the
/// rule would produce. Default for any unconfigured rule is a clean,
/// committable report (a harmless suppression). Configure a specific rule to
/// model "suppressing this would break a canary" or "this suppression lifts
/// the objective".
pub struct FakeSuppressionEvaluator {
    reports: std::collections::HashMap<(String, String), EvalReport>,
    default: EvalReport,
}

impl FakeSuppressionEvaluator {
    pub fn new() -> Self {
        Self {
            reports: std::collections::HashMap::new(),
            default: committable_report(0.0),
        }
    }

    /// Set the re-eval report for a specific rule.
    pub fn with_report(mut self, query_class: &str, memory: MemoryRef, report: EvalReport) -> Self {
        self.reports
            .insert((query_class.to_string(), memory.0.to_string()), report);
        self
    }

    /// Convenience: model a beneficial suppression (objective lifts by `delta`,
    /// canaries intact).
    pub fn beneficial(self, query_class: &str, memory: MemoryRef, delta: f32) -> Self {
        self.with_report(query_class, memory, committable_report(delta))
    }

    /// Convenience: model a harmful suppression that breaks a canary.
    pub fn breaks_canary(self, query_class: &str, memory: MemoryRef) -> Self {
        self.with_report(query_class, memory, canary_breaking_report())
    }
}

impl Default for FakeSuppressionEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SuppressionEvaluator for FakeSuppressionEvaluator {
    async fn evaluate(&self, rule: &SuppressionRule) -> Result<EvalReport, MnemeError> {
        let key = (rule.query_class.clone(), rule.memory.0.to_string());
        Ok(self
            .reports
            .get(&key)
            .cloned()
            .unwrap_or_else(|| self.default.clone()))
    }
}

/// A clean, committable report with the given objective delta.
fn committable_report(objective_delta: f32) -> EvalReport {
    EvalReport {
        canaries_passed: 3,
        canaries_total: 3,
        replay_success_rate: 1.0,
        safety_probe_passed: true,
        objective_delta,
        judges_consulted: 2,
    }
}

/// A report where applying the suppression broke a canary — the gate must
/// reject it.
fn canary_breaking_report() -> EvalReport {
    EvalReport {
        canaries_passed: 2,
        canaries_total: 3,
        replay_success_rate: 1.0,
        safety_probe_passed: true,
        objective_delta: 0.1,
        judges_consulted: 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mneme_core::types::new_id;

    fn mref() -> MemoryRef {
        MemoryRef(new_id())
    }

    fn learner() -> SuppressionLearner {
        SuppressionLearner::new(SuppressConfig::default())
    }

    #[tokio::test]
    async fn beneficial_suppression_is_committed() {
        // The Phase-14 anti-memory "done when": a misleading memory is
        // suppressed for a query-class without regressing canaries.
        let bad = mref();
        let eval = FakeSuppressionEvaluator::new().beneficial("pricing", bad, 0.2);
        let outcomes = learner()
            .learn(
                &eval,
                &[BadOutcome {
                    query_class: "pricing".into(),
                    memory: bad,
                }],
            )
            .await
            .unwrap();
        assert_eq!(outcomes.len(), 1);
        assert!(
            outcomes[0].committed(),
            "beneficial, canary-safe → committed"
        );
        assert_eq!(outcomes[0].rule().query_class, "pricing");
    }

    #[tokio::test]
    async fn canary_breaking_suppression_is_rejected() {
        // Gate stops anti-memory from blinding the agent (Hard Rule #1).
        let bad = mref();
        let eval = FakeSuppressionEvaluator::new().breaks_canary("pricing", bad);
        let outcomes = learner()
            .learn(
                &eval,
                &[BadOutcome {
                    query_class: "pricing".into(),
                    memory: bad,
                }],
            )
            .await
            .unwrap();
        assert!(
            !outcomes[0].committed(),
            "a canary-breaking suppression must be refused"
        );
        if let SuppressionOutcome::Rejected { reasons, .. } = &outcomes[0] {
            assert!(reasons.contains(&RejectReason::BaselineFailed));
            assert!(reasons
                .iter()
                .any(|r| matches!(r, RejectReason::CanariesFailing { .. })));
        } else {
            panic!("expected Rejected");
        }
    }

    #[tokio::test]
    async fn duplicate_complaints_are_deduped() {
        let bad = mref();
        let eval = FakeSuppressionEvaluator::new().beneficial("pricing", bad, 0.2);
        let outcomes = learner()
            .learn(
                &eval,
                &[
                    BadOutcome {
                        query_class: "pricing".into(),
                        memory: bad,
                    },
                    BadOutcome {
                        query_class: "pricing".into(),
                        memory: bad,
                    },
                ],
            )
            .await
            .unwrap();
        assert_eq!(
            outcomes.len(),
            1,
            "same (class, memory) collapses to one rule"
        );
    }

    #[tokio::test]
    async fn max_rules_per_pass_bounds_the_pass() {
        let cfg = SuppressConfig {
            max_rules_per_pass: 2,
            ..SuppressConfig::default()
        };
        let bads: Vec<BadOutcome> = (0..5)
            .map(|i| BadOutcome {
                query_class: format!("class-{i}"),
                memory: mref(),
            })
            .collect();
        let eval = FakeSuppressionEvaluator::new(); // default committable
        let outcomes = SuppressionLearner::new(cfg)
            .learn(&eval, &bads)
            .await
            .unwrap();
        assert_eq!(outcomes.len(), 2, "cascade bound (#6) held");
    }

    #[test]
    fn rule_renders_as_exclusion_retrieval_rule() {
        let m = mref();
        let rule = SuppressionRule {
            query_class: "pricing".into(),
            memory: m,
        };
        match rule.to_artifact_kind() {
            ArtifactKind::RetrievalRule {
                query_pattern,
                rewrite,
            } => {
                assert_eq!(query_pattern, "pricing");
                assert!(rewrite.contains("-exclude:"));
                assert!(rewrite.contains(&m.0.to_string()));
            }
            other => panic!("expected RetrievalRule, got {other:?}"),
        }
    }
}
