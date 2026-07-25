//! Admission control — *should this even become a long-term memory?*
//!
//! Storing everything is the "junk drawer" failure mode the 2026 memory
//! literature keeps flagging. A-MAC (Adaptive Memory Admission Control)
//! scores a candidate along five dimensions before letting it into
//! long-term storage; we mirror that with an explainable, dependency-free
//! heuristic scorer plus a threshold policy. The host runs this on
//! `ConsolidationAction::Add` content before committing.
//!
//! All five sub-scores and the composite are in `[0.0, 1.0]`.

/// Five-dimensional importance, A-MAC style.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Importance {
    /// How actionable / durable the fact is (vs. transient chatter).
    pub utility: f32,
    /// How much we trust it (maps from `Provenance::trust` when known).
    pub confidence: f32,
    /// How *new* it is relative to what we already hold. A near-duplicate
    /// scores ~0; a wholly new fact scores ~1.
    pub novelty: f32,
    /// Recency of the observation. Fresh writes are 1.0; the host can
    /// pass a decayed value when re-scoring old memories.
    pub recency: f32,
    /// Prior for the *kind* of fact — identity/preferences outrank
    /// one-off events.
    pub type_prior: f32,
}

impl Importance {
    /// Weighted composite in `[0.0, 1.0]`.
    pub fn composite(&self, w: &ImportanceWeights) -> f32 {
        let total = w.utility + w.confidence + w.novelty + w.recency + w.type_prior;
        if total <= 0.0 {
            return 0.0;
        }
        let s = self.utility * w.utility
            + self.confidence * w.confidence
            + self.novelty * w.novelty
            + self.recency * w.recency
            + self.type_prior * w.type_prior;
        (s / total).clamp(0.0, 1.0)
    }
}

/// Per-dimension weights for the composite. Default leans on novelty +
/// utility (don't re-store what you know; do store what's useful).
#[derive(Debug, Clone, Copy)]
pub struct ImportanceWeights {
    pub utility: f32,
    pub confidence: f32,
    pub novelty: f32,
    pub recency: f32,
    pub type_prior: f32,
}

impl Default for ImportanceWeights {
    fn default() -> Self {
        Self {
            utility: 0.30,
            confidence: 0.15,
            novelty: 0.35,
            recency: 0.10,
            type_prior: 0.10,
        }
    }
}

/// Threshold policy: admit a fact iff its composite ≥ `min_composite`.
#[derive(Debug, Clone)]
pub struct AdmissionPolicy {
    pub min_composite: f32,
    pub weights: ImportanceWeights,
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self {
            // A floor that culls obvious noise — greetings, exact/near
            // duplicates — without filtering genuine (even short) novel
            // facts. Tuned so a zero-novelty duplicate lands below it
            // while any novel fact clears it; ranking is the retrieval +
            // decay layers' job, not admission's.
            min_composite: 0.5,
            weights: ImportanceWeights::default(),
        }
    }
}

impl AdmissionPolicy {
    /// True if `imp` clears the bar.
    pub fn admit(&self, imp: &Importance) -> bool {
        imp.composite(&self.weights) >= self.min_composite
    }
}

/// Novelty of `fact` against existing `candidates`: `1 - max Jaccard
/// token overlap`. Cheap, embedding-free, good enough to catch
/// paraphrase-level duplication that the LLM consolidation step might
/// also catch — but here without an LLM call. Empty candidates → 1.0
/// (everything is novel when you know nothing).
pub fn novelty_vs(fact: &str, candidates: &[&str]) -> f32 {
    let f = token_set(fact);
    if f.is_empty() {
        return 0.0;
    }
    let mut max_sim = 0.0f32;
    for c in candidates {
        let cs = token_set(c);
        if cs.is_empty() {
            continue;
        }
        let inter = f.iter().filter(|t| cs.contains(*t)).count() as f32;
        let union = f.union(&cs).count() as f32;
        let jaccard = if union > 0.0 { inter / union } else { 0.0 };
        if jaccard > max_sim {
            max_sim = jaccard;
        }
    }
    (1.0 - max_sim).clamp(0.0, 1.0)
}

/// A cheap, explainable heuristic importance for a freshly-extracted
/// fact. `trust` comes from the source's [`mnesio_core::Provenance`];
/// `type_prior` lets the caller bias by fact kind (default 0.5).
pub fn heuristic_importance(
    fact: &str,
    candidates: &[&str],
    trust: f32,
    type_prior: f32,
) -> Importance {
    let tokens = token_set(fact);
    let n_tokens = tokens.len();
    // Utility heuristic: extremely short or filler-only facts are low
    // utility; facts carrying a number or proper-noun-ish capitalised
    // token tend to be durable. Bounded to [0.2, 0.95].
    let has_digit = fact.chars().any(|c| c.is_ascii_digit());
    let has_capitalised = fact
        .split_whitespace()
        .skip(1) // ignore sentence-initial capital
        .any(|w| w.chars().next().is_some_and(|c| c.is_ascii_uppercase()));
    let mut utility: f32 = 0.4;
    if n_tokens >= 4 {
        utility += 0.2;
    }
    if has_digit {
        utility += 0.15;
    }
    if has_capitalised {
        utility += 0.15;
    }
    utility = utility.clamp(0.2, 0.95);

    Importance {
        utility,
        confidence: trust.clamp(0.0, 1.0),
        novelty: novelty_vs(fact, candidates),
        recency: 1.0,
        type_prior: type_prior.clamp(0.0, 1.0),
    }
}

/// Lowercased word set, punctuation-stripped, stop-word-free-ish. Kept
/// local + minimal so this module has no deps.
fn token_set(s: &str) -> std::collections::HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .map(|t| t.trim().to_ascii_lowercase())
        .filter(|t| t.len() > 2 && !is_stop(t))
        .collect()
}

fn is_stop(t: &str) -> bool {
    matches!(
        t,
        "the" | "and" | "for" | "are" | "was" | "has" | "have" | "with" | "that" | "this" | "not"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_is_weighted_average_in_unit_range() {
        let imp = Importance {
            utility: 1.0,
            confidence: 1.0,
            novelty: 1.0,
            recency: 1.0,
            type_prior: 1.0,
        };
        let c = imp.composite(&ImportanceWeights::default());
        assert!((c - 1.0).abs() < 1e-6);

        let zero = Importance {
            utility: 0.0,
            confidence: 0.0,
            novelty: 0.0,
            recency: 0.0,
            type_prior: 0.0,
        };
        assert_eq!(zero.composite(&ImportanceWeights::default()), 0.0);
    }

    #[test]
    fn novelty_is_one_without_candidates() {
        assert_eq!(novelty_vs("Alice likes oat milk", &[]), 1.0);
    }

    #[test]
    fn novelty_drops_for_near_duplicate() {
        let n = novelty_vs(
            "Alice likes oat milk flat whites",
            &["Alice likes oat milk flat whites"],
        );
        assert!(n < 0.1, "identical fact should be ~0 novelty, got {n}");
    }

    #[test]
    fn novelty_high_for_unrelated() {
        let n = novelty_vs(
            "Quarterly revenue grew eighteen percent",
            &["Alice prefers oat milk"],
        );
        assert!(n > 0.8, "unrelated fact should be high novelty, got {n}");
    }

    #[test]
    fn admission_rejects_exact_duplicate_low_novelty() {
        let policy = AdmissionPolicy::default();
        // A duplicate, otherwise reasonable fact: novelty ~0 drags the
        // composite under the floor.
        let dup = heuristic_importance("Bob lives in Berlin", &["Bob lives in Berlin"], 0.7, 0.5);
        assert!(!policy.admit(&dup), "exact duplicate should be rejected");
    }

    #[test]
    fn admission_accepts_novel_useful_fact() {
        let policy = AdmissionPolicy::default();
        let imp = heuristic_importance(
            "Acme Q3 revenue grew 18% to $284M",
            &["Alice prefers oat milk"],
            0.9,
            0.6,
        );
        assert!(policy.admit(&imp), "novel useful fact should be admitted");
    }

    #[test]
    fn admission_rejects_low_signal_chatter() {
        let policy = AdmissionPolicy::default();
        // Short, no candidates so novelty=1, but utility floor + low
        // type_prior keeps a one-word "ok" out... actually with novelty
        // 1.0 weighted 0.35 it can pass; verify utility floor behavior.
        let imp = heuristic_importance("ok", &["something"], 0.5, 0.1);
        // "ok" has <3 chars after tokenisation → empty token set →
        // novelty 0.0, utility floor 0.2 → composite low → rejected.
        assert!(!policy.admit(&imp));
    }
}
