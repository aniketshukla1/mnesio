//! Tolerant parsers for the reflection + proposal outputs.
//!
//! Same design philosophy as `mnesio-evolve::parse`: a real local LLM
//! sometimes drifts off the requested format. Empty / malformed output
//! becomes an empty extraction (no findings, no candidates) rather than
//! an error, and the compile loop treats that as a no-op rather than
//! retrying. Retrying a stuck LLM rarely helps; it just burns budget.

/// Findings emitted by the reflection step. Order-preserving for the
/// dashboard timeline.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReflectionSummary {
    pub findings: Vec<String>,
}

impl ReflectionSummary {
    /// True iff the reflection produced no actionable findings — either
    /// the LLM explicitly said "none" or it returned garbage we couldn't
    /// parse. Either way the proposal step should skip.
    pub fn is_empty(&self) -> bool {
        self.findings.is_empty()
    }

    /// Compact render for re-embedding into the proposal prompt.
    pub fn as_bullet_list(&self) -> String {
        if self.findings.is_empty() {
            return "(no findings)".into();
        }
        self.findings
            .iter()
            .map(|f| format!("- {f}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Parse a reflection-step response. Looks for lines beginning with
/// `FINDING:` (case-insensitive); strips the prefix and keeps the rest.
/// A finding of `none` (any case) is treated as the explicit empty
/// signal — discarded entirely.
pub fn parse_reflection(response: &str) -> ReflectionSummary {
    let mut findings = Vec::new();
    for raw in response.lines() {
        let line = raw.trim();
        let upper = line.to_ascii_uppercase();
        if let Some(rest) = upper.strip_prefix("FINDING:") {
            let value = line[line.len() - rest.len()..].trim();
            if value.is_empty() {
                continue;
            }
            if value.eq_ignore_ascii_case("none") {
                // Explicit "no findings" sentinel — discard so the
                // summary is empty and the compile loop skips.
                continue;
            }
            findings.push(value.to_string());
        }
    }
    ReflectionSummary { findings }
}

/// Parse a proposal-step response into a vector of candidate bodies.
/// Splits on the `--- CANDIDATE N ---` sentinel; each non-empty
/// section becomes one candidate. The leading section before the first
/// sentinel (often LLM preamble) is discarded.
///
/// Returns an empty vec when no sentinels are found — caller treats
/// that as "LLM refused / drifted; skip this proposal pass."
pub fn parse_proposals(response: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current: Option<String> = None;
    for raw in response.lines() {
        let line = raw.trim_end();
        if is_candidate_sentinel(line) {
            if let Some(body) = current.take() {
                let trimmed = body.trim().to_string();
                if !trimmed.is_empty() {
                    out.push(trimmed);
                }
            }
            current = Some(String::new());
            continue;
        }
        if let Some(body) = current.as_mut() {
            body.push_str(line);
            body.push('\n');
        }
    }
    if let Some(body) = current.take() {
        let trimmed = body.trim().to_string();
        if !trimmed.is_empty() {
            out.push(trimmed);
        }
    }
    out
}

/// Identify the `--- CANDIDATE N ---` sentinel tolerantly: any line
/// that contains `CANDIDATE` between dashes counts. This protects
/// against drift like `=== CANDIDATE 1 ===` or `--- candidate 1 ---`.
fn is_candidate_sentinel(line: &str) -> bool {
    let stripped: String = line
        .chars()
        .filter(|c| !matches!(*c, '-' | '=' | '*' | '#' | ' ' | '\t'))
        .collect();
    let upper = stripped.to_ascii_uppercase();
    if !upper.starts_with("CANDIDATE") {
        return false;
    }
    // Must be followed by a digit. Avoids spurious matches on prose like
    // "CANDIDATE: foo" inside a body that happens to start with the word.
    upper
        .chars()
        .skip("CANDIDATE".len())
        .any(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- reflection parser ----------

    #[test]
    fn parse_reflection_extracts_each_finding_line() {
        let r = "FINDING: prompt is too terse for compound questions\n\
                 FINDING: missing example for citation format\n\
                 FINDING: tone drifts when input is long";
        let s = parse_reflection(r);
        assert_eq!(s.findings.len(), 3);
        assert!(s.findings[0].contains("too terse"));
        assert!(!s.is_empty());
    }

    #[test]
    fn parse_reflection_tolerates_mixed_case_prefix() {
        let r = "finding: lowercase ok\nFiNdInG: weird case still ok";
        let s = parse_reflection(r);
        assert_eq!(s.findings.len(), 2);
    }

    #[test]
    fn parse_reflection_treats_none_sentinel_as_empty() {
        let s = parse_reflection("FINDING: none");
        assert!(s.is_empty());
    }

    #[test]
    fn parse_reflection_ignores_lines_without_prefix() {
        let r = "Here are my findings:\n\
                 FINDING: real one\n\
                 More commentary here";
        let s = parse_reflection(r);
        assert_eq!(s.findings.len(), 1);
        assert_eq!(s.findings[0], "real one");
    }

    #[test]
    fn parse_reflection_empty_input_returns_empty() {
        assert!(parse_reflection("").is_empty());
        assert!(parse_reflection("complete garbage no findings here").is_empty());
    }

    #[test]
    fn reflection_summary_bullet_list_renders_for_proposal_prompt() {
        let s = ReflectionSummary {
            findings: vec!["a".into(), "b".into()],
        };
        assert_eq!(s.as_bullet_list(), "- a\n- b");
        let empty = ReflectionSummary::default();
        assert_eq!(empty.as_bullet_list(), "(no findings)");
    }

    // ---------- proposal parser ----------

    #[test]
    fn parse_proposals_extracts_each_candidate_body() {
        let r = "--- CANDIDATE 1 ---\n\
                 You are a helpful assistant.\n\
                 Answer concisely.\n\
                 --- CANDIDATE 2 ---\n\
                 You are a helpful, careful assistant.\n\
                 Answer in 1-3 sentences.";
        let cands = parse_proposals(r);
        assert_eq!(cands.len(), 2);
        assert!(cands[0].contains("Answer concisely"));
        assert!(cands[1].contains("1-3 sentences"));
    }

    #[test]
    fn parse_proposals_discards_preamble_before_first_sentinel() {
        let r = "Sure! Here are 2 candidates as requested:\n\
                 --- CANDIDATE 1 ---\n\
                 body one\n\
                 --- CANDIDATE 2 ---\n\
                 body two";
        let cands = parse_proposals(r);
        assert_eq!(cands.len(), 2);
        assert_eq!(cands[0], "body one");
        assert_eq!(cands[1], "body two");
    }

    #[test]
    fn parse_proposals_tolerates_sentinel_drift() {
        // === CANDIDATE 1 === should still match.
        let r = "=== CANDIDATE 1 ===\nfoo\n=== CANDIDATE 2 ===\nbar";
        let cands = parse_proposals(r);
        assert_eq!(cands.len(), 2);
        assert_eq!(cands[0], "foo");
        assert_eq!(cands[1], "bar");
    }

    #[test]
    fn parse_proposals_returns_empty_when_no_sentinel_found() {
        let cands = parse_proposals("Sorry, I can't help with that.");
        assert!(cands.is_empty());
    }

    #[test]
    fn parse_proposals_drops_empty_candidate_bodies() {
        // A drifty LLM might emit a sentinel with nothing under it.
        let r = "--- CANDIDATE 1 ---\nfoo\n--- CANDIDATE 2 ---\n--- CANDIDATE 3 ---\nbar";
        let cands = parse_proposals(r);
        assert_eq!(cands.len(), 2);
        assert_eq!(cands[0], "foo");
        assert_eq!(cands[1], "bar");
    }

    #[test]
    fn sentinel_requires_digit_to_avoid_prose_match() {
        // The word "CANDIDATE" without a number isn't a sentinel.
        assert!(!is_candidate_sentinel("--- CANDIDATE ---"));
        assert!(!is_candidate_sentinel("the best candidate is foo"));
        assert!(is_candidate_sentinel("--- CANDIDATE 1 ---"));
    }

    #[test]
    fn parse_proposals_preserves_multiline_bodies() {
        let r = "--- CANDIDATE 1 ---\n\
                 Line one.\n\
                 Line two.\n\
                 Line three.\n\
                 --- CANDIDATE 2 ---\n\
                 Other body.";
        let cands = parse_proposals(r);
        assert_eq!(cands.len(), 2);
        assert!(cands[0].contains("Line one."));
        assert!(cands[0].contains("Line three."));
    }
}
