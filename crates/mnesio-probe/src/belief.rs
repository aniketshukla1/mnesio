//! Belief calibration: a per-memory confidence in `[0,1]`, *derived* from
//! evidence rather than stored as authoritative state (Hard Rule #4).
//!
//! Each memory starts at a prior (its provenance trust, say `0.5`). Every
//! piece of evidence nudges it: corroboration pushes confidence up,
//! contradiction pushes it down, a passing probe corroborates, a failing
//! probe refutes. The update is a bounded multiplicative step toward 1.0 (for
//! support) or 0.0 (for doubt), so confidence stays in `[0,1]` and no single
//! observation can slam it to a hard edge — calibration, not a latch.
//!
//! Because it's a pure fold over an evidence list, the host can rebuild every
//! memory's confidence by replaying the log: collect `Corroborated` from
//! later memories that agree, `Contradicted` from supersession chains, and
//! probe outcomes from the probe worker, then fold. Nothing here persists.

use serde::{Deserialize, Serialize};

/// One piece of evidence bearing on a memory's truth. The `weight` (in
/// `(0,1]`) scales how hard this nudges confidence — e.g. a trusted source's
/// corroboration weighs more than a passing mention.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Evidence {
    /// Independent support for the claim.
    Corroborated { weight: f32 },
    /// Evidence against the claim (a contradicting memory, a failed check).
    Contradicted { weight: f32 },
    /// An acceptance probe re-ran and held.
    ProbePassed,
    /// An acceptance probe re-ran and failed.
    ProbeFailed,
}

impl Evidence {
    /// Signed, clamped nudge magnitude in `[-1, 1]`. Positive = supports,
    /// negative = doubts. Probe outcomes carry a fixed, deliberately strong
    /// weight (a probe is a *direct* re-test, not circumstantial).
    fn signed_weight(self) -> f32 {
        match self {
            Evidence::Corroborated { weight } => weight.clamp(0.0, 1.0),
            Evidence::Contradicted { weight } => -weight.clamp(0.0, 1.0),
            Evidence::ProbePassed => 0.9,
            Evidence::ProbeFailed => -0.9,
        }
    }
}

/// The calibrated belief in a memory: a confidence plus the tallies that
/// produced it, so retrieval can answer "belief + confidence + why".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Belief {
    /// Calibrated confidence in `[0,1]`.
    pub confidence: f32,
    /// Prior the fold started from.
    pub prior: f32,
    pub corroborations: u32,
    pub contradictions: u32,
    pub probes_passed: u32,
    pub probes_failed: u32,
}

impl Belief {
    /// A one-line, human-readable "why" for the dashboard / API.
    pub fn rationale(&self) -> String {
        format!(
            "prior {:.2} → {:.2} from {} corroboration(s), {} contradiction(s), {} probe pass(es), {} probe fail(s)",
            self.prior,
            self.confidence,
            self.corroborations,
            self.contradictions,
            self.probes_passed,
            self.probes_failed,
        )
    }

    /// True once confidence has fallen at/below `floor` — a candidate the host
    /// may choose to re-evaluate or suppress.
    pub fn is_doubted(&self, floor: f32) -> bool {
        self.confidence <= floor
    }
}

/// Fold a prior + an evidence stream into a calibrated [`Belief`].
///
/// Update rule per item with signed weight `w`:
/// - `w > 0`: `c ← c + w·(1 − c)` (asymptote toward 1.0)
/// - `w < 0`: `c ← c + w·c`       (asymptote toward 0.0; `w` is negative)
///
/// This keeps `c` in `[0,1]` for any sequence and any per-step `|w| ≤ 1`, and
/// is order-robust enough for calibration without pretending to be a rigorous
/// Bayesian posterior.
pub fn belief_of(prior: f32, evidence: &[Evidence]) -> Belief {
    let mut c = prior.clamp(0.0, 1.0);
    let mut b = Belief {
        confidence: c,
        prior: c,
        corroborations: 0,
        contradictions: 0,
        probes_passed: 0,
        probes_failed: 0,
    };
    for ev in evidence {
        match ev {
            Evidence::Corroborated { .. } => b.corroborations += 1,
            Evidence::Contradicted { .. } => b.contradictions += 1,
            Evidence::ProbePassed => b.probes_passed += 1,
            Evidence::ProbeFailed => b.probes_failed += 1,
        }
        let w = ev.signed_weight();
        if w >= 0.0 {
            c += w * (1.0 - c);
        } else {
            c += w * c;
        }
        c = c.clamp(0.0, 1.0);
    }
    b.confidence = c;
    b
}

/// A tiny accumulator so callers can build an evidence stream imperatively
/// (e.g. while iterating the log) and fold once.
#[derive(Debug, Default, Clone)]
pub struct BeliefLedger {
    prior: Option<f32>,
    evidence: Vec<Evidence>,
}

impl BeliefLedger {
    pub fn new(prior: f32) -> Self {
        Self {
            prior: Some(prior),
            evidence: Vec::new(),
        }
    }

    pub fn push(&mut self, ev: Evidence) -> &mut Self {
        self.evidence.push(ev);
        self
    }

    pub fn corroborate(&mut self, weight: f32) -> &mut Self {
        self.push(Evidence::Corroborated { weight })
    }

    pub fn contradict(&mut self, weight: f32) -> &mut Self {
        self.push(Evidence::Contradicted { weight })
    }

    pub fn finish(&self) -> Belief {
        belief_of(self.prior.unwrap_or(0.5), &self.evidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_evidence_returns_prior() {
        let b = belief_of(0.5, &[]);
        assert!((b.confidence - 0.5).abs() < 1e-6);
        assert_eq!(b.prior, 0.5);
    }

    #[test]
    fn corroboration_raises_contradiction_lowers() {
        let up = belief_of(0.5, &[Evidence::Corroborated { weight: 0.5 }]);
        assert!(up.confidence > 0.5, "corroboration raises confidence");
        let down = belief_of(0.5, &[Evidence::Contradicted { weight: 0.5 }]);
        assert!(down.confidence < 0.5, "contradiction lowers confidence");
    }

    #[test]
    fn confidence_stays_in_unit_interval_under_extremes() {
        let many_up = vec![Evidence::Corroborated { weight: 1.0 }; 50];
        let b = belief_of(0.5, &many_up);
        assert!(b.confidence <= 1.0 && b.confidence > 0.99);
        let many_down = vec![Evidence::Contradicted { weight: 1.0 }; 50];
        let b = belief_of(0.5, &many_down);
        assert!(b.confidence >= 0.0 && b.confidence < 0.01);
    }

    #[test]
    fn failed_probe_drives_strong_doubt() {
        // The Phase-11 calibration half: a probe failure tanks confidence.
        let before = belief_of(0.7, &[Evidence::Corroborated { weight: 0.3 }]);
        let after = belief_of(
            0.7,
            &[
                Evidence::Corroborated { weight: 0.3 },
                Evidence::ProbeFailed,
            ],
        );
        assert!(
            after.confidence < before.confidence,
            "a failed probe must lower confidence"
        );
        assert!(after.is_doubted(0.2), "strong refutation → doubted");
        assert_eq!(after.probes_failed, 1);
    }

    #[test]
    fn passing_probe_corroborates() {
        let b = belief_of(0.5, &[Evidence::ProbePassed]);
        assert!(b.confidence > 0.5);
        assert_eq!(b.probes_passed, 1);
    }

    #[test]
    fn ledger_builds_same_result_as_belief_of() {
        let mut l = BeliefLedger::new(0.5);
        l.corroborate(0.4).contradict(0.2);
        let folded = belief_of(
            0.5,
            &[
                Evidence::Corroborated { weight: 0.4 },
                Evidence::Contradicted { weight: 0.2 },
            ],
        );
        assert_eq!(l.finish(), folded);
    }

    #[test]
    fn rationale_mentions_counts() {
        let b = belief_of(0.5, &[Evidence::ProbeFailed]);
        let r = b.rationale();
        assert!(r.contains("probe fail"), "rationale: {r}");
    }
}
