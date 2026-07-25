//! Paragraph-boundary chunker.
//!
//! Splits on one-or-more blank lines, trims each chunk, drops empties. The
//! obvious default for structured prose (board memos, reports, blog
//! posts) — preserves the author's own natural break points without
//! requiring tokenizer-aware windowing.
//!
//! Not the right tool when paragraphs are routinely long enough to blow
//! past an embedder's context window (~512 tokens for bge-small-en-v1.5,
//! roughly 350-400 English words). A `WindowChunker` lands later to handle
//! that case.

use super::Chunker;

#[derive(Debug, Default)]
pub struct ParagraphChunker;

impl ParagraphChunker {
    pub fn new() -> Self {
        Self
    }
}

impl Chunker for ParagraphChunker {
    fn chunk(&self, text: &str) -> Vec<String> {
        // Normalize line endings then split on "\n\n+". We use a small
        // hand-rolled scanner instead of a regex dep — chunking is hot
        // enough on ingest to make that worth it.
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let mut chunks = Vec::new();
        let mut start = 0usize;
        let bytes = normalized.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\n' {
                // Count consecutive newlines.
                let mut j = i;
                while j < bytes.len() && bytes[j] == b'\n' {
                    j += 1;
                }
                if j - i >= 2 {
                    push_trimmed(&normalized[start..i], &mut chunks);
                    start = j;
                }
                i = j;
            } else {
                i += 1;
            }
        }
        if start < bytes.len() {
            push_trimmed(&normalized[start..], &mut chunks);
        }
        chunks
    }
}

fn push_trimmed(s: &str, out: &mut Vec<String>) {
    let t = s.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_paragraph_round_trips() {
        let c = ParagraphChunker::new();
        let got = c.chunk("a single paragraph");
        assert_eq!(got, vec!["a single paragraph"]);
    }

    #[test]
    fn multiple_paragraphs_split_on_blank_lines() {
        let c = ParagraphChunker::new();
        let got = c.chunk("first paragraph.\n\nsecond paragraph.\n\nthird paragraph.");
        assert_eq!(
            got,
            vec!["first paragraph.", "second paragraph.", "third paragraph."]
        );
    }

    #[test]
    fn extra_blank_lines_dont_create_empty_chunks() {
        let c = ParagraphChunker::new();
        let got = c.chunk("first\n\n\n\n\nsecond");
        assert_eq!(got, vec!["first", "second"]);
    }

    #[test]
    fn trims_surrounding_whitespace_per_chunk() {
        let c = ParagraphChunker::new();
        let got = c.chunk("  first  \n\n  second\twith tab  ");
        assert_eq!(got, vec!["first", "second\twith tab"]);
    }

    #[test]
    fn empty_input_yields_empty() {
        let c = ParagraphChunker::new();
        assert!(c.chunk("").is_empty());
        assert!(c.chunk("\n\n\n").is_empty());
        assert!(c.chunk("   \n  \n  ").is_empty());
    }

    #[test]
    fn crlf_line_endings_treated_like_lf() {
        let c = ParagraphChunker::new();
        let got = c.chunk("first paragraph.\r\n\r\nsecond paragraph.");
        assert_eq!(got, vec!["first paragraph.", "second paragraph."]);
    }

    #[test]
    fn single_newlines_inside_a_paragraph_are_preserved() {
        // One newline inside a paragraph (e.g. a wrapped line) is part of
        // the chunk, not a chunk boundary.
        let c = ParagraphChunker::new();
        let got = c.chunk("line one\nline two\n\nnext paragraph");
        assert_eq!(got, vec!["line one\nline two", "next paragraph"]);
    }

    #[test]
    fn chunks_a_six_paragraph_memo() {
        let c = ParagraphChunker::new();
        let memo = "para 1.\n\npara 2.\n\npara 3.\n\npara 4.\n\npara 5.\n\npara 6.";
        let got = c.chunk(memo);
        assert_eq!(got.len(), 6);
    }
}
