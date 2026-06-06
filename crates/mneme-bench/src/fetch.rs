//! Real public-benchmark loader (feature `fetch`).
//!
//! Pulls a real QA dataset from the Hugging Face **datasets-server** REST API
//! and projects it into a [`MemEvalSuite`] so mneme's memory-recall harness can
//! be run against non-synthetic data. The default target is `rajpurkar/squad`:
//! each row's `context` becomes a memory (deduplicated — SQuAD repeats a
//! context across several questions), and each `(question, answers.text[0])`
//! becomes a recall pair whose gold is the answer span.
//!
//! The fetched suite is cached to disk as JSON, so subsequent runs are offline.
//! If the network is unavailable and no cache exists, [`fetch_suite`] returns a
//! clear error rather than hanging — the core build stays network-free (this
//! module only compiles under `--features fetch`).

use crate::memeval::{MemEvalSuite, MemItem, MemQuestion};
use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// The datasets-server caps `length` per request; paginate in chunks of this.
const PAGE: usize = 100;

/// Which row shape a dataset uses, so [`download`] knows how to project it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetKind {
    /// SQuAD-shaped: `row.context` (string), `row.question`,
    /// `row.answers.text[]`. Single-hop reading comprehension.
    Squad,
    /// HotpotQA-shaped: `row.context.{title[], sentences[][]}`, `row.question`,
    /// `row.answer` (string). Multi-hop — each context paragraph becomes a
    /// memory; yes/no comparison answers are skipped (not retrievable spans).
    HotpotQa,
}

/// A dataset target: which HF dataset/config/split, and how to read its fields.
#[derive(Debug, Clone)]
pub struct FetchSpec {
    pub dataset: String,
    pub config: String,
    pub split: String,
    /// How many rows to pull (paginated). The resulting memory count is ≤ this
    /// (contexts are deduplicated).
    pub rows: usize,
    /// The row shape, selecting the projection in [`download`].
    pub kind: DatasetKind,
}

impl FetchSpec {
    /// The canonical SQuAD reading-comprehension set — contexts as memories,
    /// questions+answer-spans as recall pairs.
    pub fn squad(rows: usize) -> Self {
        Self {
            dataset: "rajpurkar/squad".to_string(),
            config: "plain_text".to_string(),
            split: "validation".to_string(),
            rows,
            kind: DatasetKind::Squad,
        }
    }

    /// HotpotQA (distractor) — a **multi-hop** set: each row carries several
    /// context paragraphs (one memory each), and the answer span must be found
    /// across them. A harder retrieval test than single-hop SQuAD.
    pub fn hotpotqa(rows: usize) -> Self {
        Self {
            dataset: "hotpotqa/hotpot_qa".to_string(),
            config: "distractor".to_string(),
            split: "validation".to_string(),
            rows,
            kind: DatasetKind::HotpotQa,
        }
    }

    /// A short, filesystem-safe cache key.
    fn cache_key(&self) -> String {
        let safe: String = self
            .dataset
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        format!("{safe}-{}-{}-{}.json", self.config, self.split, self.rows)
    }
}

// --- datasets-server JSON shapes ---
//
// Row shapes differ per dataset, so we deserialize each `row` as a generic
// `serde_json::Value` and project it per [`DatasetKind`]. This keeps one
// loader working across SQuAD-shaped and HotpotQA-shaped datasets.

#[derive(Debug, Deserialize)]
struct RowsResponse {
    rows: Vec<RawRow>,
}

#[derive(Debug, Deserialize)]
struct RawRow {
    row: serde_json::Value,
}

/// Directory where fetched suites are cached. Honors `MNEME_BENCH_CACHE`, else
/// `<repo>/crates/mneme-bench/data/cache`.
pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("MNEME_BENCH_CACHE") {
        return PathBuf::from(d);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("cache")
}

/// Load a real benchmark as a [`MemEvalSuite`], using the on-disk cache when
/// present and otherwise downloading + caching it. Set `force` to bypass the
/// cache and re-download.
pub async fn fetch_suite(spec: &FetchSpec, force: bool) -> Result<MemEvalSuite> {
    let cache_path = cache_dir().join(spec.cache_key());

    if !force {
        if let Some(suite) = load_cache(&cache_path)? {
            eprintln!("# fetch: loaded cached suite from {}", cache_path.display());
            return Ok(suite);
        }
    }

    eprintln!(
        "# fetch: downloading {} rows of {} ({}/{}) from HF datasets-server…",
        spec.rows, spec.dataset, spec.config, spec.split
    );
    let suite = download(spec).await?;

    // Best-effort cache write.
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    match serde_json::to_string_pretty(&suite) {
        Ok(json) => {
            if std::fs::write(&cache_path, json).is_ok() {
                eprintln!("# fetch: cached suite to {}", cache_path.display());
            }
        }
        Err(e) => eprintln!("# fetch: warning, could not serialize cache: {e}"),
    }
    Ok(suite)
}

const LOCOMO_URL: &str =
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json";

/// Download + parse the raw `locomo10.json` into its top-level array of samples.
async fn download_locomo() -> Result<Vec<serde_json::Value>> {
    eprintln!("# fetch: downloading LOCOMO (snap-research/locomo) from GitHub…");
    let raw = reqwest::get(LOCOMO_URL)
        .await
        .map_err(|e| anyhow!("download locomo: {e}"))?
        .text()
        .await
        .map_err(|e| anyhow!("read locomo body: {e}"))?;
    let root: serde_json::Value = serde_json::from_str(&raw).context("parse locomo10.json")?;
    root.as_array()
        .cloned()
        .ok_or_else(|| anyhow!("locomo: expected a top-level array"))
}

/// Every dialogue turn of a LOCOMO sample as dated memories
/// (`[date] speaker: text`), so temporal questions are answerable from memory.
fn locomo_memories(sample: &serde_json::Value) -> Vec<MemItem> {
    let mut memories = Vec::new();
    let Some(conv) = sample.get("conversation").and_then(|c| c.as_object()) else {
        return memories;
    };
    let sample_id = sample
        .get("sample_id")
        .and_then(|s| s.as_str())
        .unwrap_or("locomo")
        .to_string();
    for (key, val) in conv {
        let n = match key.strip_prefix("session_") {
            Some(rest) if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) => rest,
            _ => continue,
        };
        let Some(turns) = val.as_array() else {
            continue;
        };
        let date = conv
            .get(&format!("session_{n}_date_time"))
            .and_then(|d| d.as_str())
            .unwrap_or("");
        for turn in turns {
            let sp = turn.get("speaker").and_then(|s| s.as_str()).unwrap_or("");
            let tx = turn.get("text").and_then(|s| s.as_str()).unwrap_or("");
            if tx.is_empty() {
                continue;
            }
            let content = if date.is_empty() {
                format!("{sp}: {tx}")
            } else {
                format!("[{date}] {sp}: {tx}")
            };
            memories.push(MemItem {
                content,
                tags: vec![sample_id.clone()],
            });
        }
    }
    memories
}

/// The *answerable* QA pairs of a LOCOMO sample (null-answer adversarials, the
/// category-5 set, are excluded).
fn locomo_answerable_questions(sample: &serde_json::Value) -> Vec<MemQuestion> {
    let mut out = Vec::new();
    let Some(qa) = sample.get("qa").and_then(|q| q.as_array()) else {
        return out;
    };
    for q in qa {
        let gold = match q.get("answer") {
            Some(serde_json::Value::String(s)) if !s.trim().is_empty() => s.clone(),
            Some(serde_json::Value::Number(num)) => num.to_string(),
            _ => continue, // null answer (adversarial) or empty → skip
        };
        let Some(question) = q.get("question").and_then(|x| x.as_str()) else {
            continue;
        };
        let category = match q.get("category").and_then(|c| c.as_i64()) {
            Some(1) => "multi-hop",
            Some(2) => "temporal",
            Some(3) => "open-domain",
            Some(4) => "single-hop",
            Some(5) => "adversarial",
            _ => "other",
        }
        .to_string();
        out.push(MemQuestion {
            question: question.to_string(),
            answer_substring: gold,
            category,
        });
    }
    out
}

/// LOCOMO (snap-research) as **one global suite** — all 10 dialogues' turns in a
/// single memory corpus, all answerable QA as questions. This is the *harder*
/// (non-canonical) setting: cross-conversation distractors. `cap = 0` = all
/// answerable QA (~1,542). For the canonical, fair setting use
/// [`fetch_locomo_conversations`].
pub async fn fetch_locomo(cap: usize, force: bool) -> Result<MemEvalSuite> {
    let tag = if cap == 0 {
        "all".to_string()
    } else {
        cap.to_string()
    };
    let cache_path = cache_dir().join(format!("locomo10-qa{tag}.json"));
    if !force {
        if let Some(suite) = load_cache(&cache_path)? {
            eprintln!(
                "# fetch: loaded cached LOCOMO suite from {}",
                cache_path.display()
            );
            return Ok(suite);
        }
    }
    let samples = download_locomo().await?;
    let want = if cap == 0 { usize::MAX } else { cap };
    let mut memories = Vec::new();
    let mut questions = Vec::new();
    for sample in &samples {
        if questions.len() >= want {
            break;
        }
        let qa = locomo_answerable_questions(sample);
        if qa.is_empty() {
            continue;
        }
        memories.extend(locomo_memories(sample));
        for q in qa {
            if questions.len() >= want {
                break;
            }
            questions.push(q);
        }
    }
    if questions.is_empty() {
        bail!("locomo: no answerable questions parsed");
    }
    let suite = MemEvalSuite {
        name: format!("locomo-answerable-qa{}", questions.len()),
        description: "LOCOMO (snap-research) answerable QA — global corpus over all 10 dialogues"
            .to_string(),
        memories,
        questions,
    };
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Ok(json) = serde_json::to_string(&suite) {
        let _ = std::fs::write(&cache_path, json);
    }
    Ok(suite)
}

/// LOCOMO **per conversation** — the *canonical* protocol: one suite per dialogue
/// whose memory is only that dialogue's turns (no cross-conversation
/// distractors). Returns one [`MemEvalSuite`] per conversation; the caller runs
/// each and aggregates. `cap = 0` = all answerable QA across the 10 dialogues.
pub async fn fetch_locomo_conversations(cap: usize, force: bool) -> Result<Vec<MemEvalSuite>> {
    let tag = if cap == 0 {
        "all".to_string()
    } else {
        cap.to_string()
    };
    let cache_path = cache_dir().join(format!("locomo10-perconv-qa{tag}.json"));
    if !force && cache_path.exists() {
        if let Ok(json) = std::fs::read_to_string(&cache_path) {
            if let Ok(v) = serde_json::from_str::<Vec<MemEvalSuite>>(&json) {
                eprintln!(
                    "# fetch: loaded cached per-conversation LOCOMO from {}",
                    cache_path.display()
                );
                return Ok(v);
            }
        }
    }
    let samples = download_locomo().await?;
    let want = if cap == 0 { usize::MAX } else { cap };
    let mut taken = 0usize;
    let mut suites = Vec::new();
    for sample in &samples {
        if taken >= want {
            break;
        }
        let mut qa = locomo_answerable_questions(sample);
        if qa.is_empty() {
            continue;
        }
        if taken + qa.len() > want {
            qa.truncate(want - taken);
        }
        taken += qa.len();
        let sample_id = sample
            .get("sample_id")
            .and_then(|s| s.as_str())
            .unwrap_or("locomo")
            .to_string();
        suites.push(MemEvalSuite {
            name: format!("locomo-{sample_id}-qa{}", qa.len()),
            description: "LOCOMO per-conversation (canonical) answerable QA".to_string(),
            memories: locomo_memories(sample),
            questions: qa,
        });
    }
    if suites.is_empty() {
        bail!("locomo: no answerable questions parsed");
    }
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Ok(json) = serde_json::to_string(&suites) {
        let _ = std::fs::write(&cache_path, json);
    }
    Ok(suites)
}

fn load_cache(path: &Path) -> Result<Option<MemEvalSuite>> {
    if !path.exists() {
        return Ok(None);
    }
    let json = std::fs::read_to_string(path).with_context(|| format!("read cache {path:?}"))?;
    let suite: MemEvalSuite =
        serde_json::from_str(&json).with_context(|| format!("parse cache {path:?}"))?;
    Ok(Some(suite))
}

async fn download(spec: &FetchSpec) -> Result<MemEvalSuite> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("mneme-bench/0.1 (+https://github.com/aniketshukla1/mneme)")
        .build()
        .map_err(|e| anyhow!("build http client: {e}"))?;

    // Deduplicate contexts → memories; collect (question, answer) recall pairs.
    let mut seen_contexts: HashSet<String> = HashSet::new();
    let mut memories: Vec<MemItem> = Vec::new();
    let mut questions: Vec<MemQuestion> = Vec::new();

    let mut offset = 0usize;
    while offset < spec.rows {
        let length = PAGE.min(spec.rows - offset);
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset={}&config={}&split={}&offset={}&length={}",
            urlencode(&spec.dataset),
            urlencode(&spec.config),
            urlencode(&spec.split),
            offset,
            length,
        );
        let resp = client.get(&url).send().await.map_err(|e| {
            anyhow!(
                "http GET (offset {offset}): {e} — no network? a cached suite is needed offline"
            )
        })?;
        if !resp.status().is_success() {
            bail!(
                "datasets-server returned HTTP {} at offset {offset}: {}",
                resp.status(),
                resp.text()
                    .await
                    .unwrap_or_default()
                    .chars()
                    .take(200)
                    .collect::<String>()
            );
        }
        let body: RowsResponse = resp
            .json()
            .await
            .map_err(|e| anyhow!("parse rows JSON (offset {offset}): {e}"))?;
        if body.rows.is_empty() {
            break; // ran past the end of the split
        }
        for env in body.rows {
            match spec.kind {
                DatasetKind::Squad => {
                    extract_squad(&env.row, &mut seen_contexts, &mut memories, &mut questions)
                }
                DatasetKind::HotpotQa => {
                    extract_hotpotqa(&env.row, &mut seen_contexts, &mut memories, &mut questions)
                }
            }
        }
        offset += length;
    }

    if memories.is_empty() {
        bail!(
            "fetched 0 memories from {} — check dataset name/config/split",
            spec.dataset
        );
    }

    Ok(MemEvalSuite {
        name: format!("{}-{}-{}", spec.dataset, spec.split, spec.rows),
        description: format!(
            "Real benchmark from HF datasets-server: {} unique contexts as memories, \
             {} question/answer recall pairs.",
            memories.len(),
            questions.len()
        ),
        memories,
        questions,
    })
}

/// Project one SQuAD-shaped row into the (deduped) memory + question sets.
fn extract_squad(
    row: &serde_json::Value,
    seen: &mut HashSet<String>,
    memories: &mut Vec<MemItem>,
    questions: &mut Vec<MemQuestion>,
) {
    let context = row.get("context").and_then(|v| v.as_str()).unwrap_or("");
    let question = row.get("question").and_then(|v| v.as_str()).unwrap_or("");
    let answer = row
        .get("answers")
        .and_then(|a| a.get("text"))
        .and_then(|t| t.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if context.is_empty() || question.is_empty() || answer.trim().is_empty() {
        return; // skip unanswerable / malformed rows
    }
    if seen.insert(context.to_string()) {
        memories.push(MemItem {
            content: context.to_string(),
            tags: vec!["squad".into(), "context".into()],
        });
    }
    questions.push(MemQuestion {
        question: question.to_string(),
        answer_substring: answer.to_string(),
        category: "reading-comprehension".to_string(),
    });
}

/// Project one HotpotQA-shaped row: each context paragraph (title + sentences)
/// becomes a memory; the (question, answer span) becomes a multi-hop recall
/// pair. yes/no comparison answers are skipped — they're not retrievable spans.
fn extract_hotpotqa(
    row: &serde_json::Value,
    seen: &mut HashSet<String>,
    memories: &mut Vec<MemItem>,
    questions: &mut Vec<MemQuestion>,
) {
    let question = row.get("question").and_then(|v| v.as_str()).unwrap_or("");
    let answer = row.get("answer").and_then(|v| v.as_str()).unwrap_or("");
    if question.is_empty() || answer.trim().is_empty() {
        return;
    }
    let a_lower = answer.trim().to_ascii_lowercase();
    if a_lower == "yes" || a_lower == "no" {
        return; // boolean comparison answer — not a retrievable span
    }

    if let Some(ctx) = row.get("context") {
        let titles = ctx.get("title").and_then(|v| v.as_array());
        let sentences = ctx.get("sentences").and_then(|v| v.as_array());
        if let (Some(titles), Some(sentences)) = (titles, sentences) {
            for (i, title) in titles.iter().enumerate() {
                let title = title.as_str().unwrap_or("");
                let para = sentences
                    .get(i)
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_default();
                if para.trim().is_empty() {
                    continue;
                }
                let content = if title.is_empty() {
                    para
                } else {
                    format!("{title}: {para}")
                };
                if seen.insert(content.clone()) {
                    memories.push(MemItem {
                        content,
                        tags: vec!["hotpotqa".into(), "context".into()],
                    });
                }
            }
        }
    }

    questions.push(MemQuestion {
        question: question.to_string(),
        answer_substring: answer.to_string(),
        category: "multi-hop".to_string(),
    });
}

/// Minimal percent-encoding for URL query values (dataset names contain `/`).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_is_filesystem_safe() {
        let key = FetchSpec::squad(500).cache_key();
        assert!(key.starts_with("rajpurkar_squad-plain_text-validation-500"));
        assert!(key.ends_with(".json"));
        assert!(
            !key.contains('/'),
            "cache key must not contain path separators"
        );
    }

    #[test]
    fn urlencode_escapes_slash_and_keeps_unreserved() {
        assert_eq!(urlencode("rajpurkar/squad"), "rajpurkar%2Fsquad");
        assert_eq!(urlencode("plain_text"), "plain_text");
        assert_eq!(urlencode("a.b-c~d"), "a.b-c~d");
    }

    #[test]
    fn squad_spec_defaults() {
        let s = FetchSpec::squad(1234);
        assert_eq!(s.dataset, "rajpurkar/squad");
        assert_eq!(s.config, "plain_text");
        assert_eq!(s.split, "validation");
        assert_eq!(s.rows, 1234);
    }

    #[test]
    fn missing_cache_returns_none_not_error() {
        let p = cache_dir().join("definitely-does-not-exist-xyz.json");
        assert!(load_cache(&p).unwrap().is_none());
    }

    #[test]
    fn hotpotqa_spec_defaults() {
        let s = FetchSpec::hotpotqa(500);
        assert_eq!(s.dataset, "hotpotqa/hotpot_qa");
        assert_eq!(s.config, "distractor");
        assert_eq!(s.split, "validation");
        assert_eq!(s.kind, DatasetKind::HotpotQa);
        // Cache key is still filesystem-safe for the slashed dataset name.
        assert!(!s.cache_key().contains('/'));
    }

    #[test]
    fn extract_squad_from_value_dedups_context() {
        let mut seen = HashSet::new();
        let mut mems = Vec::new();
        let mut qs = Vec::new();
        let row = serde_json::json!({
            "context": "Paris is the capital of France.",
            "question": "What is the capital of France?",
            "answers": { "text": ["Paris"], "answer_start": [0] }
        });
        extract_squad(&row, &mut seen, &mut mems, &mut qs);
        // Same context, different question — context must not be re-added.
        let row2 = serde_json::json!({
            "context": "Paris is the capital of France.",
            "question": "Which country is Paris the capital of?",
            "answers": { "text": ["France"], "answer_start": [27] }
        });
        extract_squad(&row2, &mut seen, &mut mems, &mut qs);
        assert_eq!(mems.len(), 1, "context deduped");
        assert_eq!(qs.len(), 2);
        assert_eq!(qs[0].answer_substring, "Paris");
    }

    #[test]
    fn extract_hotpotqa_builds_paragraph_memories_and_skips_yesno() {
        let mut seen = HashSet::new();
        let mut mems = Vec::new();
        let mut qs = Vec::new();

        // A span-answer row: two context paragraphs become two memories.
        let row = serde_json::json!({
            "question": "Which film did Tim Burton direct in 1994?",
            "answer": "Ed Wood",
            "context": {
                "title": ["Ed Wood (film)", "Tim Burton"],
                "sentences": [
                    ["Ed Wood is a 1994 American film.", "It was directed by Tim Burton."],
                    ["Tim Burton is an American filmmaker."]
                ]
            }
        });
        extract_hotpotqa(&row, &mut seen, &mut mems, &mut qs);
        assert_eq!(mems.len(), 2, "one memory per context paragraph");
        assert!(mems[0].content.starts_with("Ed Wood (film): "));
        assert_eq!(qs.len(), 1);
        assert_eq!(qs[0].category, "multi-hop");
        // The gold span appears in a built memory → recall is achievable.
        assert!(mems.iter().any(|m| m.content.contains("Ed Wood")));

        // A yes/no comparison row contributes no question.
        let before = qs.len();
        let yesno = serde_json::json!({
            "question": "Were both directors American?",
            "answer": "yes",
            "context": { "title": ["X"], "sentences": [["irrelevant"]] }
        });
        extract_hotpotqa(&yesno, &mut seen, &mut mems, &mut qs);
        assert_eq!(qs.len(), before, "yes/no answers are skipped");
    }
}
