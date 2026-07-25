//! Prompt templates for the A-MEM three-step pipeline.
//!
//! Kept as pure functions returning `String` so they're trivially
//! testable and easy to A/B without touching the worker logic.
//! Output format is strict line-prefix style — easy for both a
//! sloppy local LLM and the deterministic `FakeLlmClient` to produce,
//! and easy for `parse.rs` to validate.

use mnesio_core::types::MemoryRef;

/// **Step P_s1 — note construction.** Asks the LLM to extract three
/// structured fields (`KEYWORDS`, `TAGS`, `CONTEXT`) from a raw memory's
/// content. Combined into a single LLM call (vs. three separate ones)
/// to keep evolution-worker latency bounded.
pub fn note_construction(content: &str) -> String {
    format!(
        "Read the following memory and extract three structured fields.\n\
         \n\
         Memory:\n{content}\n\
         \n\
         Respond with EXACTLY three lines, in this order:\n\
         KEYWORDS: 3 to 5 comma-separated keywords\n\
         TAGS: 1 to 3 comma-separated lowercase topical tags\n\
         CONTEXT: one short sentence describing what this memory is about\n"
    )
}

/// **Step P_s2 — link generation.** Asks the LLM to pick which
/// candidate memories (out of an already-retrieved top-k) have a
/// meaningful relationship to the new memory. Candidates are presented
/// with stable 1-based numbering; the LLM responds with a comma-
/// separated list of those numbers (or NONE).
///
/// The opaque [`MemoryRef`]s are kept out of the prompt — they're
/// long opaque ULIDs and just inflate the token count. The worker
/// reconciles position → ref after parsing.
pub fn link_generation(new_content: &str, candidates: &[Candidate<'_>]) -> String {
    let mut s = String::with_capacity(256 + new_content.len() + candidates.len() * 200);
    s.push_str("A new memory was just recorded:\n");
    s.push_str(new_content);
    s.push_str("\n\nHere are candidate memories that may be related:\n");
    for (i, c) in candidates.iter().enumerate() {
        s.push_str(&format!("{}. {}\n", i + 1, c.content));
    }
    s.push_str(
        "\nList the numbers of candidates that have a meaningful relationship to the new \
         memory (same topic, supersedes, supports, contradicts, etc.). \
         Respond with a comma-separated list of numbers, e.g. \"1, 3\", or NONE.\n\
         Response: ",
    );
    s
}

/// A single neighbor candidate fed into [`link_generation`] and
/// [`evolution_proposal`]. Keeping it borrow-only avoids cloning the
/// memory's text content on every prompt build.
pub struct Candidate<'a> {
    pub memory: MemoryRef,
    pub content: &'a str,
    pub tags: &'a [String],
    pub keywords: &'a [String],
}

/// **Step P_s3 — evolution proposal.** For a single neighbor, asks the
/// LLM to propose **additive** updates to the neighbor's tags +
/// keywords that incorporate the relationship to the new memory.
///
/// Deliberately additive-only (no removals) to keep cascades bounded:
/// shrinking a neighbor's tag set destabilises retrieval and is a
/// common A-MEM divergence mode. The corresponding hard cap that
/// stops compounding is in [`crate::EvolveConfig`].
pub fn evolution_proposal(neighbor: &Candidate<'_>, new_memory_content: &str) -> String {
    format!(
        "An existing memory and its current annotations:\n\
         CONTENT: {nbr_content}\n\
         TAGS: {tags}\n\
         KEYWORDS: {keywords}\n\
         \n\
         A newly-recorded related memory:\n{new_memory_content}\n\
         \n\
         Considering the relationship, propose ONLY ADDITIVE changes to the \
         existing memory's tags and keywords — new tags/keywords that reflect \
         the relationship. Do NOT remove anything. If no meaningful additions \
         are warranted, respond with NONE.\n\
         \n\
         Format:\n\
         TAGS_ADD: comma-separated lowercase tags (or empty)\n\
         KEYWORDS_ADD: comma-separated lowercase keywords (or empty)\n\
         \n\
         Or, if no changes:\n\
         NONE\n",
        nbr_content = neighbor.content,
        tags = neighbor.tags.join(", "),
        keywords = neighbor.keywords.join(", "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::new_id;

    #[test]
    fn note_construction_includes_content_and_expected_keys() {
        let p = note_construction("Revenue grew 18% YoY");
        assert!(p.contains("Revenue grew 18% YoY"));
        assert!(p.contains("KEYWORDS:"));
        assert!(p.contains("TAGS:"));
        assert!(p.contains("CONTEXT:"));
    }

    #[test]
    fn link_generation_numbers_candidates_1_based() {
        let cands = [
            Candidate {
                memory: MemoryRef(new_id()),
                content: "first",
                tags: &[],
                keywords: &[],
            },
            Candidate {
                memory: MemoryRef(new_id()),
                content: "second",
                tags: &[],
                keywords: &[],
            },
        ];
        let p = link_generation("the new memory", &cands);
        assert!(p.contains("1. first"));
        assert!(p.contains("2. second"));
        assert!(p.contains("NONE"));
    }

    #[test]
    fn evolution_proposal_shows_existing_tags_and_keywords() {
        let cand = Candidate {
            memory: MemoryRef(new_id()),
            content: "neighbor body",
            tags: &["earnings".to_string(), "q3".to_string()],
            keywords: &["revenue".to_string()],
        };
        let p = evolution_proposal(&cand, "a related new memory");
        assert!(p.contains("neighbor body"));
        assert!(p.contains("earnings, q3"));
        assert!(p.contains("revenue"));
        assert!(p.contains("ADDITIVE"));
        assert!(p.contains("TAGS_ADD:"));
        assert!(p.contains("KEYWORDS_ADD:"));
    }
}
