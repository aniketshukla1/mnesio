//! Prompt templates for extraction + consolidation.
//!
//! Same philosophy as `mneme-evolve::prompts`: pure functions returning
//! `String`, strict line-prefix output format that both a sloppy local
//! LLM and the deterministic `FakeLlmClient` can produce and `parse.rs`
//! can validate.

/// **Extraction.** Turn a raw turn/paragraph into atomic, self-contained
/// facts — each one a standalone statement that makes sense without the
/// surrounding conversation (so it can be retrieved in isolation later).
pub fn extract_facts(raw: &str) -> String {
    format!(
        "Extract the durable, atomic facts worth remembering from the text below. \
         Each fact must be a single self-contained statement understandable on its \
         own (resolve pronouns, include the subject). Ignore pleasantries, questions, \
         and transient chatter.\n\
         \n\
         Text:\n{raw}\n\
         \n\
         Respond with one fact per line, each prefixed `FACT:`. If there is nothing \
         worth remembering, respond with exactly `NONE`.\n\
         Response:\n"
    )
}

/// **Consolidation decision.** Given one freshly-extracted `fact` and a
/// numbered list of existing candidate memories, ask the LLM what to do.
///
/// The response grammar (one line):
/// - `DECISION: ADD` — new knowledge.
/// - `DECISION: NOOP <n>` — already captured by candidate n (dedup).
/// - `DECISION: UPDATE <n> CONTRADICTION` — fact conflicts with n.
/// - `DECISION: UPDATE <n> REFINEMENT` — fact refines/extends n.
pub fn decide_action(fact: &str, candidates: &[&str]) -> String {
    let mut s = String::with_capacity(256 + fact.len() + candidates.len() * 120);
    s.push_str("A new candidate fact has been extracted:\n");
    s.push_str(fact);
    if candidates.is_empty() {
        s.push_str(
            "\n\nThere are no existing memories to compare against.\n\
             Respond with exactly: DECISION: ADD\n\
             Response: ",
        );
        return s;
    }
    s.push_str("\n\nExisting memories that may overlap:\n");
    for (i, c) in candidates.iter().enumerate() {
        s.push_str(&format!("{}. {}\n", i + 1, c));
    }
    s.push_str(
        "\nDecide how the new fact relates to the existing memories. Respond with \
         EXACTLY ONE line in one of these forms:\n\
         DECISION: ADD                      (the fact is new knowledge)\n\
         DECISION: NOOP <n>                 (already captured by memory n)\n\
         DECISION: UPDATE <n> CONTRADICTION (the fact conflicts with memory n)\n\
         DECISION: UPDATE <n> REFINEMENT    (the fact refines/extends memory n)\n\
         Response: ",
    );
    s
}
