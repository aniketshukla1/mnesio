//! Import: *trust but verify*.
//!
//! Importing a [`SkillCertificate`] never trusts the shipped report. The
//! importer:
//!
//! 1. verifies the signature (tampered → reject);
//! 2. **re-runs the certificate's canaries locally** ([`CanaryRunner`]) to
//!    build a *fresh* [`EvalReport`] reflecting how the artifact behaves on
//!    *this* instance;
//! 3. applies the importer's own [`mnesio_procedural::EvalGates`] to that fresh
//!    report — the skill activates *only if B's gate passes* (Hard Rule #1).
//!
//! B can run a stricter gate than A; an A-passing certificate can still be
//! rejected by B. And because step 3 gates the *locally-derived* report, a
//! forged "committable" `issuer_report` buys an attacker nothing.

use crate::certificate::{SignatureError, Signer, SkillCertificate};
use async_trait::async_trait;
use mnesio_core::entity::{Canary, PolicyArtifact};
use mnesio_core::event::EvalReport;
use mnesio_core::MnesioError;
use mnesio_procedural::{EvalGates, RejectReason, Verdict};

/// Re-runs a certificate's canaries on the importing instance. The seam that
/// lets B form its *own* opinion of the artifact (Hard Rule #7).
#[async_trait]
pub trait CanaryRunner: Send + Sync {
    /// Run `canaries` against `artifact` locally; return how many passed.
    /// (The full safety/replay/judge signals are supplied by the caller via
    /// [`LocalEvaluation`]; this seam owns just the canary re-execution, which
    /// is the part that depends on the local executor.)
    async fn run_canaries(
        &self,
        artifact: &PolicyArtifact,
        canaries: &[Canary],
    ) -> Result<u32, MnesioError>;
}

/// The importer's locally-measured signals other than canaries, folded into
/// the fresh report. In a full build these come from B's own shadow-eval
/// (safety probe, tail replay, judge panel); here they're supplied explicitly
/// so the import logic + gate are what's under test.
#[derive(Debug, Clone)]
pub struct LocalEvaluation {
    pub safety_probe_passed: bool,
    pub replay_success_rate: f32,
    pub objective_delta: f32,
    pub judges_consulted: u8,
}

impl Default for LocalEvaluation {
    /// A clean local evaluation: everything an importer would see for a skill
    /// that behaves well on its corpus. Canary count is filled in by `import`.
    fn default() -> Self {
        Self {
            safety_probe_passed: true,
            replay_success_rate: 1.0,
            objective_delta: 0.0,
            judges_consulted: 2,
        }
    }
}

/// What happened on import.
///
/// Not `PartialEq`: it carries `PolicyArtifact` / `EvalReport`, which don't
/// implement it. Tests pattern-match on the variant instead.
#[derive(Debug, Clone)]
pub enum ImportOutcome {
    /// The certificate verified, the local re-eval passed B's gate, and the
    /// artifact is cleared to activate. Carries the *locally-derived* report
    /// (not the issuer's) for audit.
    Activated {
        artifact: Box<PolicyArtifact>,
        local_report: EvalReport,
    },
    /// The certificate verified but B's gate rejected the local re-eval — the
    /// artifact must NOT activate. Carries the structured reasons + the local
    /// report so the dashboard can show why.
    Rejected {
        local_report: EvalReport,
        reasons: Vec<RejectReason>,
    },
}

impl ImportOutcome {
    pub fn activated(&self) -> bool {
        matches!(self, ImportOutcome::Activated { .. })
    }
}

/// Why an import couldn't even be evaluated (vs. evaluated-and-rejected).
#[derive(Debug, Clone, PartialEq)]
pub enum ImportError {
    /// Signature verification failed — tampered or untrusted issuer key.
    Signature(SignatureError),
    /// The local canary runner errored.
    Runner(String),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::Signature(e) => write!(f, "{e}"),
            ImportError::Runner(e) => write!(f, "local canary runner failed: {e}"),
        }
    }
}

impl std::error::Error for ImportError {}

/// Import a certificate under instance B's `gates`, using `signer` to verify
/// and `runner` to re-run canaries locally.
///
/// Returns `Err` only when the certificate can't be evaluated at all (bad
/// signature, runner failure). A verified-but-gate-failing import is a
/// successful *evaluation* with an [`ImportOutcome::Rejected`] — the caller
/// learns exactly why, and nothing activates.
pub async fn import(
    cert: &SkillCertificate,
    signer: &dyn Signer,
    runner: &dyn CanaryRunner,
    local: LocalEvaluation,
    gates: &EvalGates,
) -> Result<ImportOutcome, ImportError> {
    // 1. Trust nothing unsigned/tampered.
    cert.verify(signer).map_err(ImportError::Signature)?;

    // 2. Re-derive a LOCAL report by re-running the canaries here. The
    //    issuer's report is never read into this — it's a claim, not evidence.
    let canaries_total = cert.canaries.len() as u32;
    let canaries_passed = runner
        .run_canaries(&cert.artifact, &cert.canaries)
        .await
        .map_err(|e| ImportError::Runner(e.to_string()))?;

    let local_report = EvalReport {
        canaries_passed,
        canaries_total,
        replay_success_rate: local.replay_success_rate,
        safety_probe_passed: local.safety_probe_passed,
        objective_delta: local.objective_delta,
        judges_consulted: local.judges_consulted,
    };

    // 3. Apply B's own gate to B's own report (Hard Rule #1).
    let verdict: Verdict = gates.evaluate(&local_report);
    if verdict.committable {
        Ok(ImportOutcome::Activated {
            artifact: Box::new(cert.artifact.clone()),
            local_report,
        })
    } else {
        Ok(ImportOutcome::Rejected {
            local_report,
            reasons: verdict.reasons,
        })
    }
}

/// Deterministic, dependency-free [`CanaryRunner`] for tests + offline demos.
///
/// Re-runs each canary by a simple rule: a canary "passes" iff its `expect`
/// substring appears in the artifact's rendered text (system-prompt body,
/// heuristic, skill body, …). Models "does this artifact still satisfy the
/// guard on *our* instance" without needing a real executor. A `fault` knob
/// forces N canaries to fail, to model an artifact that under-performs on B.
pub struct FakeCanaryRunner {
    /// Force the first `force_failures` canaries to count as failed, modelling
    /// an artifact that doesn't reproduce on instance B.
    force_failures: u32,
}

impl FakeCanaryRunner {
    pub fn new() -> Self {
        Self { force_failures: 0 }
    }

    /// Model an artifact that fails `n` of its canaries on this instance.
    pub fn with_forced_failures(n: u32) -> Self {
        Self { force_failures: n }
    }

    fn rendered(artifact: &PolicyArtifact) -> String {
        use mnesio_core::entity::ArtifactKind::*;
        match &artifact.kind {
            SystemPrompt { body } => body.clone(),
            Heuristic { when, then } => format!("{when} {then}"),
            Skill {
                signature, body, ..
            } => format!("{signature} {body}"),
            RetrievalRule {
                query_pattern,
                rewrite,
            } => format!("{query_pattern} {rewrite}"),
            Reflection { lesson, .. } => lesson.clone(),
        }
    }
}

impl Default for FakeCanaryRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CanaryRunner for FakeCanaryRunner {
    async fn run_canaries(
        &self,
        artifact: &PolicyArtifact,
        canaries: &[Canary],
    ) -> Result<u32, MnesioError> {
        let text = Self::rendered(artifact).to_ascii_lowercase();
        let mut passed = 0u32;
        for (i, c) in canaries.iter().enumerate() {
            let forced_fail = (i as u32) < self.force_failures;
            let expect = c.expect.to_ascii_lowercase();
            if !forced_fail && (expect.is_empty() || text.contains(&expect)) {
                passed += 1;
            }
        }
        Ok(passed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::{export, FakeSigner};
    use mnesio_core::entity::{ArtifactKind, Canary};
    use mnesio_core::types::{new_id, BiTemporal, Scope};

    fn artifact() -> PolicyArtifact {
        PolicyArtifact {
            id: new_id(),
            version: 3,
            scope: Scope::global("alice-corp"),
            kind: ArtifactKind::SystemPrompt {
                // Body satisfies the canary's `expect = "cite"`.
                body: "Always cite your sources before answering.".into(),
            },
            canaries: vec![],
            time: BiTemporal::now(),
        }
    }

    fn canaries() -> Vec<Canary> {
        vec![
            Canary {
                input: "q1".into(),
                expect: "cite".into(),
            },
            Canary {
                input: "q2".into(),
                expect: "cite".into(),
            },
        ]
    }

    fn committable() -> EvalReport {
        EvalReport {
            canaries_passed: 2,
            canaries_total: 2,
            replay_success_rate: 1.0,
            safety_probe_passed: true,
            objective_delta: 0.2,
            judges_consulted: 2,
        }
    }

    fn cert() -> (FakeSigner, SkillCertificate) {
        let signer = FakeSigner::new();
        let c = export(&signer, "alice-corp", artifact(), canaries(), committable()).unwrap();
        (signer, c)
    }

    // The Phase-13 "done when": certified on A imports to B and activates only
    // after B re-runs ITS OWN gate.
    #[tokio::test]
    async fn certified_skill_activates_on_b_after_b_regate() {
        let (signer, c) = cert();
        let out = import(
            &c,
            &signer,
            &FakeCanaryRunner::new(),
            LocalEvaluation::default(),
            &EvalGates::default(),
        )
        .await
        .unwrap();
        assert!(out.activated(), "clean re-eval under B's gate → activate");
        if let ImportOutcome::Activated { local_report, .. } = out {
            // The activation rode on a LOCALLY-derived report (2/2 canaries
            // re-run here), not the shipped one.
            assert_eq!(local_report.canaries_passed, 2);
            assert_eq!(local_report.canaries_total, 2);
        }
    }

    #[tokio::test]
    async fn tampered_certificate_is_rejected_before_any_eval() {
        let (signer, mut c) = cert();
        c.artifact.kind = ArtifactKind::SystemPrompt {
            body: "Never cite anything.".into(),
        };
        let err = import(
            &c,
            &signer,
            &FakeCanaryRunner::new(),
            LocalEvaluation::default(),
            &EvalGates::default(),
        )
        .await
        .unwrap_err();
        assert_eq!(err, ImportError::Signature(SignatureError::Invalid));
    }

    #[tokio::test]
    async fn b_gate_rejects_when_canaries_fail_locally() {
        // The artifact passed on A, but on B a canary doesn't reproduce.
        let (signer, c) = cert();
        let out = import(
            &c,
            &signer,
            &FakeCanaryRunner::with_forced_failures(1), // 1 of 2 canaries fails on B
            LocalEvaluation::default(),
            &EvalGates::default(),
        )
        .await
        .unwrap();
        assert!(
            !out.activated(),
            "a canary failing locally must block activation"
        );
        if let ImportOutcome::Rejected {
            local_report,
            reasons,
        } = out
        {
            assert_eq!(local_report.canaries_passed, 1);
            assert!(reasons.contains(&RejectReason::BaselineFailed));
        } else {
            panic!("expected Rejected");
        }
    }

    #[tokio::test]
    async fn stricter_b_gate_can_reject_an_a_passing_cert() {
        // A's report had objective_delta 0.2 and was committable. B demands
        // strict improvement ≥ 0.5 — its local eval (delta 0.0 default) fails.
        let (signer, c) = cert();
        let strict = EvalGates {
            min_objective_delta: 0.5,
            ..EvalGates::default()
        };
        let out = import(
            &c,
            &signer,
            &FakeCanaryRunner::new(),
            LocalEvaluation::default(),
            &strict,
        )
        .await
        .unwrap();
        assert!(!out.activated(), "B's stricter gate rejects");
        if let ImportOutcome::Rejected { reasons, .. } = out {
            assert!(reasons
                .iter()
                .any(|r| matches!(r, RejectReason::ObjectiveRegression { .. })));
        }
    }

    #[tokio::test]
    async fn forged_committable_issuer_report_buys_nothing() {
        // Even if the shipped report claims perfection, B gates on its OWN
        // re-eval. Model a local safety-probe failure: import must reject
        // despite the issuer_report being committable.
        let (signer, c) = cert();
        let local = LocalEvaluation {
            safety_probe_passed: false, // B's own probe trips
            ..LocalEvaluation::default()
        };
        let out = import(
            &c,
            &signer,
            &FakeCanaryRunner::new(),
            local,
            &EvalGates::default(),
        )
        .await
        .unwrap();
        assert!(
            !out.activated(),
            "shipped report is ignored; B's local safety failure blocks activation"
        );
        if let ImportOutcome::Rejected { reasons, .. } = out {
            assert!(reasons.contains(&RejectReason::SafetyProbeFailed));
        }
    }
}
