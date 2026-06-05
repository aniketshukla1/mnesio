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

/// A dataset target: which HF dataset/config/split, and how to read its fields.
#[derive(Debug, Clone)]
pub struct FetchSpec {
    pub dataset: String,
    pub config: String,
    pub split: String,
    /// How many rows to pull (paginated). The resulting memory count is ≤ this
    /// (contexts are deduplicated).
    pub rows: usize,
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

// --- datasets-server JSON shapes (only the fields we read) ---

#[derive(Debug, Deserialize)]
struct RowsResponse {
    rows: Vec<RowEnvelope>,
}

#[derive(Debug, Deserialize)]
struct RowEnvelope {
    row: SquadRow,
}

#[derive(Debug, Deserialize)]
struct SquadRow {
    context: String,
    question: String,
    answers: SquadAnswers,
}

#[derive(Debug, Deserialize)]
struct SquadAnswers {
    text: Vec<String>,
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
            let r = env.row;
            let answer = match r.answers.text.into_iter().next() {
                Some(a) if !a.trim().is_empty() => a,
                _ => continue, // skip unanswerable / malformed rows
            };
            if seen_contexts.insert(r.context.clone()) {
                memories.push(MemItem {
                    content: r.context.clone(),
                    tags: vec!["squad".into(), "context".into()],
                });
            }
            questions.push(MemQuestion {
                question: r.question,
                answer_substring: answer,
                category: "reading-comprehension".to_string(),
            });
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
}
