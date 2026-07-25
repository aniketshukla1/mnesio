//! Tolerant parsers for the three A-MEM prompt responses.
//!
//! These deliberately accept slightly noisy output (extra whitespace,
//! markdown bullets, mixed casing) so a real-world local LLM that
//! doesn't follow the format perfectly still produces something
//! usable. When parsing fails we return *empty* extraction rather
//! than errors — the worker treats "nothing extracted" as a no-op
//! rather than retrying.

/// Result of parsing a [`crate::prompts::note_construction`] response.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NoteFields {
    pub keywords: Vec<String>,
    pub tags: Vec<String>,
    pub context: String,
}

/// Parse a note-construction LLM response. Looks for three lines
/// prefixed `KEYWORDS:`, `TAGS:`, `CONTEXT:` (case-insensitive).
/// Missing lines become empty fields.
pub fn parse_note(response: &str) -> NoteFields {
    let mut nf = NoteFields::default();
    for line in response.lines() {
        let line = strip_bullet(line.trim());
        let upper = line.to_ascii_uppercase();
        if let Some(rest) = upper.strip_prefix("KEYWORDS:") {
            nf.keywords = split_csv_lower(&line[line.len() - rest.len()..]);
        } else if let Some(rest) = upper.strip_prefix("TAGS:") {
            nf.tags = split_csv_lower(&line[line.len() - rest.len()..]);
        } else if let Some(rest) = upper.strip_prefix("CONTEXT:") {
            nf.context = line[line.len() - rest.len()..].trim().to_string();
        }
    }
    nf
}

/// Parse a [`crate::prompts::link_generation`] response into the
/// 1-based candidate indices the LLM selected. `NONE`, empty, or
/// unparseable output → empty vec.
pub fn parse_link_selection(response: &str) -> Vec<usize> {
    let r = response.trim();
    if r.is_empty() {
        return Vec::new();
    }
    if r.to_ascii_uppercase().starts_with("NONE") {
        return Vec::new();
    }
    // Walk character by character collecting integer runs. Robust to
    // commas, spaces, bullets, parenthetical "(none)" style, etc.
    let mut out = Vec::new();
    let mut current = String::new();
    for c in r.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_digit() {
            current.push(c);
        } else if !current.is_empty() {
            if let Ok(n) = current.parse::<usize>() {
                if n >= 1 && !out.contains(&n) {
                    out.push(n);
                }
            }
            current.clear();
        }
    }
    out
}

/// Result of parsing a [`crate::prompts::evolution_proposal`] response.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvolutionChanges {
    pub tags_add: Vec<String>,
    pub keywords_add: Vec<String>,
}

impl EvolutionChanges {
    /// Number of distinct additions across both fields. The worker
    /// uses this against [`crate::EvolveConfig::min_change_threshold`]
    /// to decide whether the change is worth persisting.
    pub fn total_additions(&self) -> usize {
        self.tags_add.len() + self.keywords_add.len()
    }

    /// Did the LLM signal "no changes" by returning NONE / empty?
    pub fn is_empty(&self) -> bool {
        self.total_additions() == 0
    }
}

/// Parse an evolution-proposal LLM response. `NONE` short-circuits;
/// otherwise extract `TAGS_ADD:` and `KEYWORDS_ADD:` lines. Removals
/// are intentionally not parsed even if the model emits them — see
/// the note in [`crate::prompts::evolution_proposal`].
pub fn parse_evolution(response: &str) -> EvolutionChanges {
    let mut changes = EvolutionChanges::default();
    let r = response.trim();
    if r.is_empty() || r.to_ascii_uppercase().starts_with("NONE") {
        return changes;
    }
    for line in r.lines() {
        let line = strip_bullet(line.trim());
        let upper = line.to_ascii_uppercase();
        if let Some(_rest) = upper.strip_prefix("TAGS_ADD:") {
            let value_start = "TAGS_ADD:".len();
            changes.tags_add = split_csv_lower(&line[value_start..]);
        } else if let Some(_rest) = upper.strip_prefix("KEYWORDS_ADD:") {
            let value_start = "KEYWORDS_ADD:".len();
            changes.keywords_add = split_csv_lower(&line[value_start..]);
        }
    }
    changes
}

/// Split a CSV-ish string on commas, trim each piece, drop empties,
/// lowercase. Tolerant of stray "and" / "or" connectors and brackets.
fn split_csv_lower(s: &str) -> Vec<String> {
    let cleaned: String = s
        .chars()
        .map(|c| match c {
            '[' | ']' | '(' | ')' | '"' | '\'' | '`' => ' ',
            _ => c,
        })
        .collect();
    let mut out: Vec<String> = cleaned
        .split([',', ';', '\n'])
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .filter(|t| !is_filler(t))
        .map(|t| t.to_ascii_lowercase())
        .collect();
    // Stable dedupe — keep first occurrence.
    let mut seen = std::collections::HashSet::new();
    out.retain(|t| seen.insert(t.clone()));
    out
}

/// Drop common bullet prefixes ("-", "*", "1.", "•") that the LLM
/// likes to add when responding in list form.
fn strip_bullet(s: &str) -> &str {
    let s = s.trim_start_matches(['-', '*', '•', '·', '–', '—', '|']);
    let s = s.trim_start();
    // Numbered bullets like "1." or "1)" — strip just the leading
    // number+separator, not the rest of the line.
    if let Some(rest) = s.strip_prefix(|c: char| c.is_ascii_digit()) {
        let trimmed = rest.trim_start_matches(|c: char| c.is_ascii_digit());
        if let Some(rest) = trimmed.strip_prefix(['.', ')', ':']) {
            return rest.trim_start();
        }
    }
    s
}

/// Filter out common filler words the LLM sometimes splits as
/// separate "items" (e.g. "earnings, and revenue").
fn is_filler(s: &str) -> bool {
    matches!(s.to_ascii_lowercase().as_str(), "and" | "or" | "none")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_note_extracts_all_three_fields() {
        let r = "KEYWORDS: revenue, growth, q3\n\
                 TAGS: earnings, finance\n\
                 CONTEXT: A revenue growth report for Q3.";
        let nf = parse_note(r);
        assert_eq!(nf.keywords, vec!["revenue", "growth", "q3"]);
        assert_eq!(nf.tags, vec!["earnings", "finance"]);
        assert_eq!(nf.context, "A revenue growth report for Q3.");
    }

    #[test]
    fn parse_note_tolerates_case_variants() {
        let r = "keywords: a, b\nTAGS: c\ncontext: d";
        let nf = parse_note(r);
        assert_eq!(nf.keywords, vec!["a", "b"]);
        assert_eq!(nf.tags, vec!["c"]);
        assert_eq!(nf.context, "d");
    }

    #[test]
    fn parse_note_handles_bullet_prefixes() {
        let r = "- KEYWORDS: a, b\n* TAGS: c";
        let nf = parse_note(r);
        assert_eq!(nf.keywords, vec!["a", "b"]);
        assert_eq!(nf.tags, vec!["c"]);
    }

    #[test]
    fn parse_note_returns_empty_when_unrecognised() {
        assert_eq!(parse_note(""), NoteFields::default());
        assert_eq!(parse_note("garbage in"), NoteFields::default());
    }

    #[test]
    fn parse_note_drops_filler_words() {
        let r = "KEYWORDS: revenue, and, growth";
        let nf = parse_note(r);
        assert_eq!(nf.keywords, vec!["revenue", "growth"]);
    }

    #[test]
    fn parse_link_selection_handles_csv_and_spaces() {
        assert_eq!(parse_link_selection("1, 3"), vec![1, 3]);
        assert_eq!(parse_link_selection("2"), vec![2]);
        assert_eq!(parse_link_selection("1,2,3"), vec![1, 2, 3]);
        assert_eq!(parse_link_selection("[1, 3]"), vec![1, 3]);
    }

    #[test]
    fn parse_link_selection_handles_none_and_empty() {
        assert!(parse_link_selection("NONE").is_empty());
        assert!(parse_link_selection("none").is_empty());
        assert!(parse_link_selection("").is_empty());
        assert!(parse_link_selection("(none)").is_empty());
    }

    #[test]
    fn parse_link_selection_dedupes() {
        assert_eq!(parse_link_selection("1, 1, 2, 2, 3"), vec![1, 2, 3]);
    }

    #[test]
    fn parse_evolution_extracts_additions() {
        let r = "TAGS_ADD: correction, revised\nKEYWORDS_ADD: audit";
        let ch = parse_evolution(r);
        assert_eq!(ch.tags_add, vec!["correction", "revised"]);
        assert_eq!(ch.keywords_add, vec!["audit"]);
        assert_eq!(ch.total_additions(), 3);
        assert!(!ch.is_empty());
    }

    #[test]
    fn parse_evolution_handles_none() {
        assert!(parse_evolution("NONE").is_empty());
        assert!(parse_evolution("none\nthe memory needs no changes").is_empty());
    }

    #[test]
    fn parse_evolution_empty_additions_is_empty() {
        let ch = parse_evolution("TAGS_ADD:\nKEYWORDS_ADD:");
        assert!(ch.is_empty());
    }

    #[test]
    fn split_csv_lower_dedupes_and_strips_brackets() {
        assert_eq!(split_csv_lower("[a, b, a, c]"), vec!["a", "b", "c"]);
    }
}
