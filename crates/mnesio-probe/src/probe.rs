//! Acceptance probes — the "CI" half of self-falsifying memory.
//!
//! A [`Probe`] re-checks a single memory's claim and returns a
//! [`ProbeVerdict`]. The [`ProbeRunner`] applies a probe to a batch of
//! candidates (bounded by [`ProbeConfig`]) and produces a [`ProbeReport`]
//! naming which memories were *refuted* — the inputs to [`crate::falsify`].
//!
//! The runner is pure orchestration over the [`Probe`] seam (Hard Rule #7):
//! the production probe wires an `LlmClient` / retriever; [`FakeProbe`] keeps
//! tests hermetic and deterministic.

use async_trait::async_trait;
use mnesio_core::types::MemoryRef;
use mnesio_core::MnesioError;
use serde::{Deserialize, Serialize};

/// What a single probe re-check concluded about a memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbeStatus {
    /// The claim still holds. Corroborates (raises confidence).
    Held,
    /// The claim no longer holds. Triggers invalidate-and-supersede.
    Refuted,
    /// The probe could not decide (no signal, transient error). NEVER
    /// supersedes — a flaky probe must not destroy a true memory.
    Inconclusive,
}

/// A probe's verdict on one memory: status + a human-readable reason and the
/// correction text to write when superseding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeVerdict {
    pub status: ProbeStatus,
    /// Why the probe concluded what it did (shown in the falsification chain).
    pub reason: String,
    /// Replacement content to record as the superseding memory when
    /// `Refuted`. `None` falls back to a generic correction note.
    pub correction: Option<String>,
}

impl ProbeVerdict {
    pub fn held(reason: impl Into<String>) -> Self {
        Self {
            status: ProbeStatus::Held,
            reason: reason.into(),
            correction: None,
        }
    }
    pub fn refuted(reason: impl Into<String>, correction: Option<String>) -> Self {
        Self {
            status: ProbeStatus::Refuted,
            reason: reason.into(),
            correction,
        }
    }
    pub fn inconclusive(reason: impl Into<String>) -> Self {
        Self {
            status: ProbeStatus::Inconclusive,
            reason: reason.into(),
            correction: None,
        }
    }
}

/// The probe seam. Given a memory's id + content, re-check its claim.
#[async_trait]
pub trait Probe: Send + Sync {
    async fn check(&self, memory: MemoryRef, content: &str) -> Result<ProbeVerdict, MnesioError>;
}

/// One scored candidate, pairing a memory with the verdict the probe returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeOutcome {
    pub memory: MemoryRef,
    pub verdict: ProbeVerdict,
}

/// Bounds for one probe pass. The engine never schedules itself; the caller
/// runs passes offline, off the write path (Hard Rules #5/#6).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeConfig {
    /// Hard cap on memories probed per pass — the cascade bound (#6).
    pub max_probes_per_pass: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            max_probes_per_pass: 128,
        }
    }
}

/// Aggregate result of one [`ProbeRunner::run`] pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeReport {
    pub outcomes: Vec<ProbeOutcome>,
    pub candidates_considered: usize,
    pub candidates_probed: usize,
}

impl ProbeReport {
    /// Memories the probe refuted — the inputs to falsification.
    pub fn refuted(&self) -> Vec<&ProbeOutcome> {
        self.outcomes
            .iter()
            .filter(|o| o.verdict.status == ProbeStatus::Refuted)
            .collect()
    }

    pub fn held_count(&self) -> usize {
        self.count(ProbeStatus::Held)
    }
    pub fn refuted_count(&self) -> usize {
        self.count(ProbeStatus::Refuted)
    }
    pub fn inconclusive_count(&self) -> usize {
        self.count(ProbeStatus::Inconclusive)
    }

    fn count(&self, status: ProbeStatus) -> usize {
        self.outcomes
            .iter()
            .filter(|o| o.verdict.status == status)
            .count()
    }
}

/// Applies a [`Probe`] to a bounded batch of `(memory, content)` candidates.
pub struct ProbeRunner {
    cfg: ProbeConfig,
}

impl ProbeRunner {
    pub fn new(cfg: ProbeConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &ProbeConfig {
        &self.cfg
    }

    /// Probe up to `max_probes_per_pass` candidates, in order. The caller is
    /// responsible for scoping candidates (Hard Rule #3).
    pub async fn run(
        &self,
        probe: &dyn Probe,
        candidates: &[(MemoryRef, String)],
    ) -> Result<ProbeReport, MnesioError> {
        let mut outcomes = Vec::new();
        for (memory, content) in candidates.iter().take(self.cfg.max_probes_per_pass) {
            let verdict = probe.check(*memory, content).await?;
            outcomes.push(ProbeOutcome {
                memory: *memory,
                verdict,
            });
        }
        let candidates_probed = outcomes.len();
        Ok(ProbeReport {
            outcomes,
            candidates_considered: candidates.len(),
            candidates_probed,
        })
    }
}

/// Deterministic, dependency-free probe for tests + offline demos.
///
/// Refutes a memory iff its content contains any configured `refute_marker`
/// (case-insensitive); otherwise `Held`. An optional `inconclusive_marker`
/// forces the third path. Models "the world changed and the claim no longer
/// checks out" without needing an LLM.
pub struct FakeProbe {
    refute_markers: Vec<String>,
    inconclusive_markers: Vec<String>,
    correction: Option<String>,
}

impl FakeProbe {
    pub fn new() -> Self {
        Self {
            refute_markers: Vec::new(),
            inconclusive_markers: Vec::new(),
            correction: None,
        }
    }

    /// A content substring (case-insensitive) that makes the probe refute.
    pub fn refute_on(mut self, marker: impl Into<String>) -> Self {
        self.refute_markers.push(marker.into());
        self
    }

    pub fn inconclusive_on(mut self, marker: impl Into<String>) -> Self {
        self.inconclusive_markers.push(marker.into());
        self
    }

    /// Correction text to attach to refuted verdicts.
    pub fn with_correction(mut self, correction: impl Into<String>) -> Self {
        self.correction = Some(correction.into());
        self
    }
}

impl Default for FakeProbe {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Probe for FakeProbe {
    async fn check(&self, _memory: MemoryRef, content: &str) -> Result<ProbeVerdict, MnesioError> {
        let lc = content.to_ascii_lowercase();
        if self
            .inconclusive_markers
            .iter()
            .any(|m| lc.contains(&m.to_ascii_lowercase()))
        {
            return Ok(ProbeVerdict::inconclusive("probe could not decide"));
        }
        if let Some(marker) = self
            .refute_markers
            .iter()
            .find(|m| lc.contains(&m.to_ascii_lowercase()))
        {
            return Ok(ProbeVerdict::refuted(
                format!("probe refuted: claim mentions stale marker {marker:?}"),
                self.correction.clone(),
            ));
        }
        Ok(ProbeVerdict::held("probe re-check passed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::types::new_id;

    fn mref() -> MemoryRef {
        MemoryRef(new_id())
    }

    fn runner() -> ProbeRunner {
        ProbeRunner::new(ProbeConfig::default())
    }

    #[tokio::test]
    async fn fake_probe_holds_refutes_and_is_inconclusive() {
        let probe = FakeProbe::new()
            .refute_on("deprecated")
            .inconclusive_on("maybe");
        let held = probe.check(mref(), "the API is stable").await.unwrap();
        assert_eq!(held.status, ProbeStatus::Held);
        let refuted = probe
            .check(mref(), "this endpoint is DEPRECATED now")
            .await
            .unwrap();
        assert_eq!(refuted.status, ProbeStatus::Refuted);
        let inc = probe.check(mref(), "maybe true").await.unwrap();
        assert_eq!(inc.status, ProbeStatus::Inconclusive);
    }

    #[tokio::test]
    async fn runner_partitions_outcomes_by_status() {
        let (m1, m2, m3) = (mref(), mref(), mref());
        let probe = FakeProbe::new().refute_on("stale").inconclusive_on("dunno");
        let cands = vec![
            (m1, "fresh fact".to_string()),
            (m2, "this is stale".to_string()),
            (m3, "dunno about this".to_string()),
        ];
        let report = runner().run(&probe, &cands).await.unwrap();
        assert_eq!(report.held_count(), 1);
        assert_eq!(report.refuted_count(), 1);
        assert_eq!(report.inconclusive_count(), 1);
        let refuted = report.refuted();
        assert_eq!(refuted.len(), 1);
        assert_eq!(refuted[0].memory, m2);
    }

    #[tokio::test]
    async fn max_probes_per_pass_bounds_the_pass() {
        let probe = FakeProbe::new();
        let cands: Vec<(MemoryRef, String)> =
            (0..5).map(|i| (mref(), format!("fact {i}"))).collect();
        let cfg = ProbeConfig {
            max_probes_per_pass: 2,
        };
        let report = ProbeRunner::new(cfg).run(&probe, &cands).await.unwrap();
        assert_eq!(report.candidates_considered, 5);
        assert_eq!(report.candidates_probed, 2);
    }

    #[tokio::test]
    async fn refuted_carries_correction_text() {
        let probe = FakeProbe::new()
            .refute_on("old price")
            .with_correction("price is now $20");
        let v = probe.check(mref(), "the old price was $10").await.unwrap();
        assert_eq!(v.status, ProbeStatus::Refuted);
        assert_eq!(v.correction.as_deref(), Some("price is now $20"));
    }
}
