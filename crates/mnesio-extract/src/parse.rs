//! Tolerant parsers for the extraction + decision responses.
//!
//! As in `mnesio-evolve::parse`, these accept noisy output and degrade to
//! a safe default (empty extraction, or `Decision::Add`) rather than
//! erroring — the worst a misparse can do is write a memory that a later
//! consolidation pass dedups, never lose or corrupt data.

use serde::{Deserialize, Serialize};

/// Why an `UPDATE` decision supersedes an existing memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateReason {
    /// The new fact factually conflicts with the old one (e.g. a revised
    /// figure). The old version is wrong-as-of-now.
    Contradiction,
    /// The new fact refines/extends the old one without contradicting it
    /// (e.g. adds a detail). The old version was incomplete.
    Refinement,
}

/// The consolidator's decision for a single fact, parsed from a
/// [`crate::prompts::decide_action`] response. Indices are **0-based**
/// here (the prompt is 1-based; we subtract on parse).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// New knowledge.
    Add,
    /// Already captured by candidate at this index.
    Noop(usize),
    /// Supersede the candidate at this index, for the given reason.
    Update(usize, UpdateReason),
}

/// Parse the `FACT:`-prefixed lines from an extraction response.
/// `NONE`/empty → no facts. Lines without the prefix are ignored (the
/// model sometimes adds a preamble).
pub fn parse_facts(response: &str) -> Vec<String> {
    let r = response.trim();
    if r.is_empty() || r.eq_ignore_ascii_case("none") {
        return Vec::new();
    }
    let mut out = Vec::new();
    for line in r.lines() {
        let line = strip_bullet(line.trim());
        let upper = line.to_ascii_uppercase();
        if let Some(_rest) = upper.strip_prefix("FACT:") {
            let value = line["FACT:".len()..].trim();
            if !value.is_empty() && !value.eq_ignore_ascii_case("none") {
                out.push(value.to_string());
            }
        }
    }
    out
}

/// Parse a decision line. Unrecognised input defaults to
/// [`Decision::Add`] — the safe choice (a spurious add is dedup-able; a
/// spurious noop silently drops knowledge).
///
/// `candidate_count` bounds the referenced index; an out-of-range index
/// (model hallucinated a memory number) also falls back to `Add`.
pub fn parse_decision(response: &str, candidate_count: usize) -> Decision {
    let upper = response.trim().to_ascii_uppercase();
    // Find the DECISION: marker anywhere (model may add a preamble).
    let after = match upper.find("DECISION:") {
        Some(idx) => upper[idx + "DECISION:".len()..].trim(),
        None => upper.trim(),
    };

    let first_index = |s: &str| -> Option<usize> {
        let mut cur = String::new();
        for c in s.chars().chain(std::iter::once(' ')) {
            if c.is_ascii_digit() {
                cur.push(c);
            } else if !cur.is_empty() {
                return cur.parse::<usize>().ok();
            }
        }
        None
    };

    // Resolve a 1-based candidate number to a valid 0-based index.
    let resolve = |s: &str| -> Option<usize> {
        let n = first_index(s)?;
        if n >= 1 && n <= candidate_count {
            Some(n - 1)
        } else {
            None
        }
    };

    if after.starts_with("NOOP") {
        if let Some(i) = resolve(after) {
            return Decision::Noop(i);
        }
        // NOOP with no/invalid target is meaningless — treat as Add so
        // we don't silently drop the fact.
        return Decision::Add;
    }
    if after.starts_with("UPDATE") {
        if let Some(i) = resolve(after) {
            let reason = if after.contains("CONTRADICT") {
                UpdateReason::Contradiction
            } else {
                UpdateReason::Refinement
            };
            return Decision::Update(i, reason);
        }
        return Decision::Add;
    }
    // ADD or anything unrecognised.
    Decision::Add
}

/// Drop leading bullets / numbering the model likes to add.
fn strip_bullet(s: &str) -> &str {
    let s = s.trim_start_matches(['-', '*', '•', '·', '–', '—', '|']);
    let s = s.trim_start();
    if let Some(rest) = s.strip_prefix(|c: char| c.is_ascii_digit()) {
        let trimmed = rest.trim_start_matches(|c: char| c.is_ascii_digit());
        if let Some(rest) = trimmed.strip_prefix(['.', ')', ':']) {
            return rest.trim_start();
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_facts_extracts_prefixed_lines() {
        let r = "FACT: Alice prefers oat milk.\n\
                 FACT: Alice is allergic to peanuts.";
        let f = parse_facts(r);
        assert_eq!(
            f,
            vec![
                "Alice prefers oat milk.".to_string(),
                "Alice is allergic to peanuts.".to_string()
            ]
        );
    }

    #[test]
    fn parse_facts_ignores_preamble_and_bullets() {
        let r = "Here are the facts:\n- FACT: X happened\n2. FACT: Y happened\nrandom line";
        let f = parse_facts(r);
        assert_eq!(f, vec!["X happened".to_string(), "Y happened".to_string()]);
    }

    #[test]
    fn parse_facts_none_and_empty() {
        assert!(parse_facts("NONE").is_empty());
        assert!(parse_facts("none").is_empty());
        assert!(parse_facts("").is_empty());
        assert!(parse_facts("FACT: none").is_empty());
    }

    #[test]
    fn parse_decision_add() {
        assert_eq!(parse_decision("DECISION: ADD", 3), Decision::Add);
        assert_eq!(parse_decision("decision: add", 0), Decision::Add);
    }

    #[test]
    fn parse_decision_noop_resolves_index() {
        assert_eq!(parse_decision("DECISION: NOOP 2", 3), Decision::Noop(1));
    }

    #[test]
    fn parse_decision_update_with_reason() {
        assert_eq!(
            parse_decision("DECISION: UPDATE 1 CONTRADICTION", 3),
            Decision::Update(0, UpdateReason::Contradiction)
        );
        assert_eq!(
            parse_decision("DECISION: UPDATE 3 REFINEMENT", 3),
            Decision::Update(2, UpdateReason::Refinement)
        );
    }

    #[test]
    fn parse_decision_update_defaults_reason_to_refinement() {
        // No reason token → refinement (the conservative read: extend,
        // don't declare a contradiction).
        assert_eq!(
            parse_decision("DECISION: UPDATE 2", 3),
            Decision::Update(1, UpdateReason::Refinement)
        );
    }

    #[test]
    fn parse_decision_out_of_range_falls_back_to_add() {
        // n=9 but only 3 candidates → can't trust it, add instead of
        // silently dropping or mis-targeting.
        assert_eq!(parse_decision("DECISION: NOOP 9", 3), Decision::Add);
        assert_eq!(
            parse_decision("DECISION: UPDATE 9 CONTRADICTION", 3),
            Decision::Add
        );
    }

    #[test]
    fn parse_decision_noop_without_target_is_add() {
        assert_eq!(parse_decision("DECISION: NOOP", 3), Decision::Add);
    }

    #[test]
    fn parse_decision_tolerates_preamble() {
        assert_eq!(
            parse_decision("Sure! DECISION: NOOP 1", 2),
            Decision::Noop(0)
        );
    }

    #[test]
    fn parse_decision_garbage_is_add() {
        assert_eq!(parse_decision("I think maybe?", 2), Decision::Add);
        assert_eq!(parse_decision("", 2), Decision::Add);
    }
}
