//! Tiny content-derived `LlmClient` for the demo.
//!
//! [`mneme_llm::FakeLlmClient`] is deterministic but returns identical
//! responses for every prompt — fine for unit tests, boring for the
//! demo dashboard (every memory ends up with the same canned tags).
//! `DemoLlmClient` returns *content-derived* answers without requiring
//! a real LLM: the enrichment for each memory varies because the
//! prompt's memory content varies.
//!
//! What it produces:
//!
//! - **Note construction** — keywords from the first ~5 distinct
//!   alphabetic words of the memory; tag `"auto-enriched"`; context
//!   is the first 80 characters of the memory.
//! - **Link generation** — picks candidate #1 if any exist.
//! - **Evolution proposal** — proposes a single `TAGS_ADD: related`
//!   addition, which produces a minimal but visible evolution chain
//!   in the dashboard.
//!
//! For *real* enrichment quality, set `MNEME_EVOLVE_LLM=ollama` and
//! point [`OllamaLlmClient`][mneme_llm::OllamaLlmClient] at a running
//! local model.

use async_trait::async_trait;
use mneme_core::{LlmClient, MnemeError};

/// Demo-friendly stub LLM. Returns content-derived (not random)
/// responses so the dashboard's enrichment events look distinct per
/// memory even without a real backend.
pub struct DemoLlmClient;

impl DemoLlmClient {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DemoLlmClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmClient for DemoLlmClient {
    async fn complete(&self, prompt: &str) -> Result<String, MnemeError> {
        if prompt.starts_with("Read the following memory") {
            return Ok(note_response(prompt));
        }
        if prompt.starts_with("A new memory was just recorded") {
            // Always pick candidate #1 if the prompt is a link-generation
            // request. Worker maps that back to the top-scoring neighbour.
            return Ok("1".to_string());
        }
        if prompt.starts_with("An existing memory and its current annotations") {
            // Minimal additive evolution. Just enough to produce a visible
            // evolved-memory chain in the dashboard without runaway.
            return Ok("TAGS_ADD: related\nKEYWORDS_ADD: linked".to_string());
        }
        // Phase-7 ingestion prompts.
        if prompt.starts_with("Extract the durable, atomic facts") {
            return Ok(extract_response(prompt));
        }
        if prompt.starts_with("A new candidate fact has been extracted:") {
            return Ok(decide_response(prompt));
        }
        // Phase-2 procedural compiler prompts — reflection + proposal.
        // Returning real-looking but content-derived responses keeps the
        // dashboard's improvement curve moving without requiring Ollama.
        if prompt.starts_with("You are reviewing a recent batch") {
            return Ok(reflection_response(prompt));
        }
        if prompt.starts_with("You are revising a system prompt") {
            return Ok(proposal_response(prompt));
        }
        // Fallback — worker handles "" as no-op.
        Ok(String::new())
    }
}

/// Build a `note_construction` response from the memory content the
/// prompt embeds. Returns format identical to what
/// `mneme_evolve::prompts::note_construction` asks for.
fn note_response(prompt: &str) -> String {
    let content = extract_memory_content(prompt);
    let keywords = top_words(&content, 5);
    let context = content
        .chars()
        .take(80)
        .collect::<String>()
        .trim()
        .to_string();
    format!(
        "KEYWORDS: {kws}\nTAGS: auto-enriched\nCONTEXT: {ctx}",
        kws = keywords.join(", "),
        ctx = context
    )
}

/// Procedural reflection response. Counts the number of `success=false`
/// outcomes embedded in the prompt and emits one or two findings — just
/// enough to keep the reflective loop unblocked. Always returns at
/// least one finding so the proposal step actually fires (unless the
/// batch contained literally zero failures, in which case `none`).
fn reflection_response(prompt: &str) -> String {
    let failures = prompt.matches("success=false").count();
    if failures == 0 {
        return "FINDING: none".into();
    }
    let mut findings = vec![format!(
        "FINDING: {failures} recent outcomes failed — the prompt may be too terse"
    )];
    if failures >= 2 {
        findings.push("FINDING: compound questions appear to be a recurring failure mode".into());
    }
    findings.join("\n")
}

/// Procedural proposal response. **Content-aware** — looks at what the
/// active body already says and proposes a candidate that adds the
/// *next* missing improvement, so the learning curve actually climbs
/// across iterations instead of plateauing on the first signal.
///
/// The signal progression mirrors common prompt-engineering wisdom:
/// 1. start by adding "address each part" (helps multi-step reasoning)
/// 2. once present, add "be precise / uncertain → say so" (helps epistemics)
/// 3. once both present, polish phrasing slightly to keep the loop alive
///
/// Always emits exactly 2 candidates: the "primary" (the next missing
/// signal) and a "stylistic alternative". The compiler picks the first
/// committable one — so the primary effectively drives the curve.
fn proposal_response(prompt: &str) -> String {
    let body = extract_system_prompt_body(prompt);
    let body_lower = body.to_ascii_lowercase();
    let has_reasoning = body_lower.contains("address each")
        || body_lower.contains("each part")
        || body_lower.contains("multiple parts");
    let has_epistemics = body_lower.contains("be precise")
        || body_lower.contains("uncertain")
        || body_lower.contains("say so rather than guessing");

    let primary = if !has_reasoning {
        // Stage 1: teach the prompt about compound questions.
        format!(
            "{body} For any question with multiple parts, address each one explicitly and \
             walk through your reasoning step by step."
        )
    } else if !has_epistemics {
        // Stage 2: teach honest uncertainty.
        format!(
            "{body} Be precise. When you are uncertain about a factual claim, say so rather \
             than guessing — \"I'm uncertain\" is always a valid answer."
        )
    } else {
        // Stage 3: small polish; both signals already in. Keeps the
        // loop emitting valid candidates without further movement on
        // the curve (steady-state plateau).
        format!("{body} Stay concise unless the question genuinely requires depth.")
    };
    let alt = format!("{body} Always provide a brief direct answer before any elaboration.");
    format!("--- CANDIDATE 1 ---\n{primary}\n--- CANDIDATE 2 ---\n{alt}")
}

/// Phase-7 extraction response. Splits the raw text into sentence-ish
/// facts (one `FACT:` line each), dropping obvious filler. Deterministic
/// + content-derived so the demo shows real extraction without a model.
fn extract_response(prompt: &str) -> String {
    let raw = prompt
        .split_once("Text:\n")
        .and_then(|(_, r)| r.split_once("\n\nRespond"))
        .map(|(t, _)| t)
        .unwrap_or("")
        .trim();
    let mut facts: Vec<String> = Vec::new();
    for sentence in raw.split(['.', '\n']) {
        let s = sentence.trim();
        // Skip pleasantries / very short fragments (mirrors the prompt's
        // "ignore transient chatter" instruction).
        if s.len() < 8 || is_pleasantry(s) {
            continue;
        }
        facts.push(format!("FACT: {s}"));
    }
    if facts.is_empty() {
        return "NONE".to_string();
    }
    facts.join("\n")
}

/// Phase-7 consolidation decision. Heuristic over the embedded fact +
/// candidate list:
/// - correction markers ("not", "actually", "correction") + a candidate
///   → `UPDATE 1 CONTRADICTION`;
/// - high token overlap with candidate 1 → `NOOP 1` (duplicate);
/// - otherwise `ADD`.
fn decide_response(prompt: &str) -> String {
    let fact = prompt
        .split_once("A new candidate fact has been extracted:\n")
        .and_then(|(_, r)| r.split_once('\n'))
        .map(|(f, _)| f.trim())
        .unwrap_or("");
    if prompt.contains("There are no existing memories") {
        return "DECISION: ADD".to_string();
    }
    // First candidate line looks like "1. <content>".
    let cand1 = prompt
        .split_once("Existing memories that may overlap:\n")
        .and_then(|(_, r)| r.lines().next())
        .map(|l| l.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ' '))
        .unwrap_or("")
        .trim();

    let fl = fact.to_ascii_lowercase();
    let is_correction = fl.contains(" not ")
        || fl.starts_with("actually")
        || fl.contains("correction")
        || fl.contains("instead of");
    if is_correction && !cand1.is_empty() {
        return "DECISION: UPDATE 1 CONTRADICTION".to_string();
    }
    if !cand1.is_empty() && jaccard(fact, cand1) >= 0.6 {
        return "DECISION: NOOP 1".to_string();
    }
    "DECISION: ADD".to_string()
}

fn is_pleasantry(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    matches!(
        l.as_str(),
        "hi" | "hello" | "hey" | "thanks" | "thank you" | "ok" | "okay" | "sure" | "got it"
    ) || l.starts_with("thanks")
        || l.starts_with("hello")
}

/// Token-set Jaccard overlap between two strings (lowercased, >2 chars).
fn jaccard(a: &str, b: &str) -> f32 {
    let ta = word_set(a);
    let tb = word_set(b);
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let inter = ta.iter().filter(|w| tb.contains(*w)).count() as f32;
    let union = ta.union(&tb).count() as f32;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

fn word_set(s: &str) -> std::collections::HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2)
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

/// Extract the active system prompt's body from a proposal prompt. The
/// reflective-loop prompt wraps it in `---` fences. Falls back to a
/// neutral default if the marker layout has drifted.
fn extract_system_prompt_body(prompt: &str) -> String {
    let after = prompt.split_once("Current system prompt:\n---\n");
    let body = match after {
        Some((_, rest)) => rest.split("\n---").next().unwrap_or(""),
        None => "",
    };
    let trimmed = body.trim();
    if trimmed.is_empty() {
        "You are a helpful assistant.".into()
    } else {
        trimmed.to_string()
    }
}

/// Pull the memory body out of a `note_construction` prompt. The
/// template wraps content between `Memory:\n…` and a blank line; we
/// take everything after the first `Memory:` marker and before the
/// "Respond with EXACTLY" sentinel.
fn extract_memory_content(prompt: &str) -> String {
    let after_marker = prompt.split_once("Memory:\n").map(|(_, r)| r).unwrap_or("");
    let before_sentinel = after_marker
        .split("Respond with EXACTLY")
        .next()
        .unwrap_or("");
    before_sentinel.trim().to_string()
}

/// Top-N distinct lowercased alphabetic words, in order of first
/// appearance. Skips short stop-word-ish tokens to keep the keyword
/// list informative.
fn top_words(text: &str, n: usize) -> Vec<String> {
    let stop: &[&str] = &[
        "the", "and", "for", "are", "but", "was", "with", "from", "this", "that", "into",
    ];
    let mut out: Vec<String> = Vec::new();
    for word in text.split(|c: char| !c.is_ascii_alphabetic()) {
        if word.len() < 3 {
            continue;
        }
        let lowered = word.to_ascii_lowercase();
        if stop.contains(&lowered.as_str()) {
            continue;
        }
        if !out.iter().any(|w| w == &lowered) {
            out.push(lowered);
        }
        if out.len() == n {
            break;
        }
    }
    if out.is_empty() {
        out.push("memory".into());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn note_response_varies_per_memory() {
        let c = DemoLlmClient::new();
        let p1 = "Read the following memory and extract three structured fields.\n\nMemory:\nRevenue grew 18% YoY across all segments\n\nRespond with EXACTLY three lines";
        let p2 = "Read the following memory and extract three structured fields.\n\nMemory:\nSupply chain stabilising into Q4 after H1 shortage\n\nRespond with EXACTLY three lines";
        let r1 = c.complete(p1).await.unwrap();
        let r2 = c.complete(p2).await.unwrap();
        assert!(r1.contains("revenue"));
        assert!(r2.contains("supply") || r2.contains("chain"));
        assert_ne!(r1, r2, "demo client must vary per memory");
    }

    #[tokio::test]
    async fn link_response_picks_first_candidate() {
        let c = DemoLlmClient::new();
        let p =
            "A new memory was just recorded:\nfoo\n\nHere are candidate memories:\n1. bar\n2. baz";
        assert_eq!(c.complete(p).await.unwrap(), "1");
    }

    #[tokio::test]
    async fn evolution_response_proposes_minimal_addition() {
        let c = DemoLlmClient::new();
        let p = "An existing memory and its current annotations:\nCONTENT: foo\nTAGS: t\nKEYWORDS: k\n\nA newly-recorded related memory:\nbar";
        let r = c.complete(p).await.unwrap();
        assert!(r.contains("TAGS_ADD"));
        assert!(r.contains("KEYWORDS_ADD"));
    }

    #[test]
    fn top_words_skips_stop_and_short() {
        let w = top_words("The quick brown fox and the cat", 5);
        assert!(w.contains(&"quick".to_string()));
        assert!(w.contains(&"brown".to_string()));
        assert!(!w.contains(&"the".to_string()));
    }
}
