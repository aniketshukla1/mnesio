//! Tier-1 extractive synthesizer.
//!
//! Tokenizes the query, walks each passage word-by-word, and emits a
//! sequence of `Plain`/`Highlight` segments where any content word
//! **prefix-matches** a non-trivial query token (case-insensitive). The
//! whole passage is preserved so the UI shows full context; only matched
//! words get the highlight class.
//!
//! Design choices:
//!
//! - **Prefix matching** (not exact) so `launch` highlights `launching`,
//!   `earn` highlights `earnings`. Approximates Snowball stemming without
//!   pulling in the stemmer crate; mirrors the user experience BM25 search
//!   already gives.
//! - **Drop short tokens** (`< 2` chars) and a small **stop-word list**
//!   from the query side so common filler ("the", "is", "a") doesn't
//!   speckle the highlights.
//! - **No truncation, no re-ranking** — the synthesizer trusts the
//!   retrieval order. `MmrCentroidSynthesizer` (next slice) is where MMR
//!   re-ordering belongs.
//! - **Deterministic, zero deps, no I/O** — same input always produces
//!   the same output, so audit trails replay cleanly.

use async_trait::async_trait;
use mnesio_core::{
    Answer, Excerpt, ExcerptSegment, MnesioError, Passage, SynthesisProvenance, Synthesizer,
};
use std::time::Instant;

/// Minimum length for a query token to be considered for highlighting.
/// Single-letter tokens are almost always noise (English `a`, `i`, or
/// stray possessive `'s` artifacts that survived parsing).
const MIN_TOKEN_LEN: usize = 2;

/// Tiny English stop-word list. Tantivy's Snowball list (used by
/// `Bm25View`) is heavier; we keep this short and use it only on the
/// query side, so the doc index isn't affected.
const QUERY_STOP_WORDS: &[&str] = &[
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "by", "with",
    "as", "from",
];

/// Deterministic extractive synthesizer.
pub struct SnippetSynthesizer;

impl SnippetSynthesizer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SnippetSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Synthesizer for SnippetSynthesizer {
    fn name(&self) -> &str {
        "snippet"
    }

    async fn synthesize(&self, query: &str, passages: &[Passage]) -> Result<Answer, MnesioError> {
        let start = Instant::now();
        let tokens = tokenize_query(query);
        let mut excerpts = Vec::with_capacity(passages.len());
        let mut citations = Vec::with_capacity(passages.len());
        for p in passages {
            excerpts.push(Excerpt {
                memory: p.memory,
                segments: highlight(&p.content, &tokens),
                retrieval_score: p.retrieval_score,
            });
            citations.push(p.memory);
        }
        Ok(Answer {
            query: query.to_string(),
            excerpts,
            citations,
            prose: None,
            provenance: SynthesisProvenance {
                synthesizer: "snippet".into(),
                model_id: None,
                elapsed_ms: start.elapsed().as_millis() as u64,
            },
        })
    }
}

/// Lowercase, alphanumeric-split, drop short tokens and stop words.
fn tokenize_query(q: &str) -> Vec<String> {
    q.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() >= MIN_TOKEN_LEN)
        .filter(|t| !QUERY_STOP_WORDS.contains(t))
        .map(|t| t.to_string())
        .collect()
}

/// Walk `text` once, partition into Plain/Highlight runs. A word is a
/// maximal run of ASCII alphanumerics; it counts as a highlight if any
/// query token is a prefix (case-insensitive) of its lowercased form.
fn highlight(text: &str, tokens: &[String]) -> Vec<ExcerptSegment> {
    if tokens.is_empty() || text.is_empty() {
        return vec![ExcerptSegment::Plain {
            text: text.to_string(),
        }];
    }
    let bytes = text.as_bytes();
    let lower = text.to_ascii_lowercase();
    let lower_bytes = lower.as_bytes();
    let mut segments: Vec<ExcerptSegment> = Vec::new();
    let mut plain_start = 0;
    let mut i = 0;

    while i < bytes.len() {
        // Skip non-word characters.
        while i < bytes.len() && !bytes[i].is_ascii_alphanumeric() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let word_start = i;
        while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
            i += 1;
        }
        let word_end = i;
        let word_lower = &lower_bytes[word_start..word_end];

        let matches = tokens.iter().any(|q| word_lower.starts_with(q.as_bytes()));

        if matches {
            if word_start > plain_start {
                segments.push(ExcerptSegment::Plain {
                    text: text[plain_start..word_start].to_string(),
                });
            }
            segments.push(ExcerptSegment::Highlight {
                text: text[word_start..word_end].to_string(),
            });
            plain_start = word_end;
        }
    }

    if plain_start < text.len() {
        segments.push(ExcerptSegment::Plain {
            text: text[plain_start..].to_string(),
        });
    }
    if segments.is_empty() {
        // Defensive: should be impossible since text wasn't empty.
        segments.push(ExcerptSegment::Plain {
            text: text.to_string(),
        });
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::types::{new_id, MemoryRef};

    fn passage(content: &str) -> Passage {
        Passage {
            memory: MemoryRef(new_id()),
            content: content.into(),
            tags: vec![],
            retrieval_score: 1.0,
        }
    }

    fn rendered(segments: &[ExcerptSegment]) -> String {
        segments
            .iter()
            .map(|s| match s {
                ExcerptSegment::Plain { text } => text.clone(),
                ExcerptSegment::Highlight { text } => format!("<{text}>"),
            })
            .collect()
    }

    #[tokio::test]
    async fn single_word_query_highlights_match() {
        let s = SnippetSynthesizer::new();
        let p = passage("Acme reported Q3 earnings beating consensus.");
        let a = s
            .synthesize("earnings", std::slice::from_ref(&p))
            .await
            .unwrap();
        assert_eq!(a.excerpts.len(), 1);
        assert_eq!(
            rendered(&a.excerpts[0].segments),
            "Acme reported Q3 <earnings> beating consensus."
        );
    }

    #[tokio::test]
    async fn multi_word_query_highlights_each_match() {
        let s = SnippetSynthesizer::new();
        let p = passage("Revenue up in EMEA driven by enterprise SaaS.");
        let a = s
            .synthesize("revenue EMEA", std::slice::from_ref(&p))
            .await
            .unwrap();
        assert_eq!(
            rendered(&a.excerpts[0].segments),
            "<Revenue> up in <EMEA> driven by enterprise SaaS."
        );
    }

    #[tokio::test]
    async fn case_insensitive() {
        let s = SnippetSynthesizer::new();
        let p = passage("EMEA had a strong quarter.");
        let a = s
            .synthesize("emea", std::slice::from_ref(&p))
            .await
            .unwrap();
        assert_eq!(
            rendered(&a.excerpts[0].segments),
            "<EMEA> had a strong quarter."
        );
    }

    #[tokio::test]
    async fn prefix_match_finds_inflected_forms() {
        let s = SnippetSynthesizer::new();
        let p = passage("Launching v3.4 with shipping items including SSO.");
        let a = s
            .synthesize("launch ship", std::slice::from_ref(&p))
            .await
            .unwrap();
        // "launch" prefix-matches "Launching", "ship" prefix-matches
        // "shipping". The "3" inside "v3.4" is a standalone word but
        // matches neither query token, so it stays plain.
        assert_eq!(
            rendered(&a.excerpts[0].segments),
            "<Launching> v3.4 with <shipping> items including SSO."
        );
    }

    #[tokio::test]
    async fn stop_words_dont_speckle_highlights() {
        let s = SnippetSynthesizer::new();
        let p = passage("The product is the new offering this quarter.");
        let a = s
            .synthesize("the product", std::slice::from_ref(&p))
            .await
            .unwrap();
        // "the" and "is" are stop words on the query side; only "product"
        // ends up driving highlights.
        assert_eq!(
            rendered(&a.excerpts[0].segments),
            "The <product> is the new offering this quarter."
        );
    }

    #[tokio::test]
    async fn empty_query_returns_all_plain() {
        let s = SnippetSynthesizer::new();
        let p = passage("Anything goes.");
        let a = s.synthesize("", std::slice::from_ref(&p)).await.unwrap();
        assert_eq!(a.excerpts.len(), 1);
        assert!(matches!(
            a.excerpts[0].segments.as_slice(),
            [ExcerptSegment::Plain { .. }]
        ));
    }

    #[tokio::test]
    async fn no_match_returns_all_plain() {
        let s = SnippetSynthesizer::new();
        let p = passage("Revenue up across all segments.");
        let a = s
            .synthesize("xyzzy", std::slice::from_ref(&p))
            .await
            .unwrap();
        assert!(matches!(
            a.excerpts[0].segments.as_slice(),
            [ExcerptSegment::Plain { .. }]
        ));
        assert_eq!(
            a.excerpts[0].segments[0].text(),
            "Revenue up across all segments."
        );
    }

    #[tokio::test]
    async fn citations_preserve_passage_order() {
        let s = SnippetSynthesizer::new();
        let p1 = passage("first");
        let p2 = passage("second");
        let p3 = passage("third");
        let r1 = p1.memory;
        let r2 = p2.memory;
        let r3 = p3.memory;
        let a = s.synthesize("anything", &[p1, p2, p3]).await.unwrap();
        assert_eq!(a.citations, vec![r1, r2, r3]);
    }

    #[tokio::test]
    async fn provenance_records_synthesizer_name() {
        let s = SnippetSynthesizer::new();
        let a = s.synthesize("anything", &[passage("text")]).await.unwrap();
        assert_eq!(a.provenance.synthesizer, "snippet");
        assert!(a.provenance.model_id.is_none());
    }

    #[tokio::test]
    async fn punctuation_between_words_is_preserved() {
        let s = SnippetSynthesizer::new();
        let p = passage("EMEA, revenue, and growth.");
        let a = s
            .synthesize("revenue growth", std::slice::from_ref(&p))
            .await
            .unwrap();
        assert_eq!(
            rendered(&a.excerpts[0].segments),
            "EMEA, <revenue>, and <growth>."
        );
    }
}
