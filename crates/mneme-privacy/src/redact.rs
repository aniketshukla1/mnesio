//! PII detection + redaction, dependency-free.
//!
//! The workspace has no regex crate, so detection is hand-rolled
//! character scanning — which is actually a feature: it's auditable, has
//! no catastrophic-backtracking risk, and is fast on the write path.
//! Each detector recognises one [`PiiKind`] and replaces the span with a
//! kind-tagged placeholder (`[EMAIL]`, `[PHONE]`, …) so downstream text
//! stays human-readable while the identifier is gone.
//!
//! This is *minimisation*, not the erasure mechanism — for erasure of
//! already-stored data see [`crate::keyring`]. Redaction stops PII from
//! entering the log at all.

use serde::{Deserialize, Serialize};

/// A class of personally-identifying information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiiKind {
    Email,
    Phone,
    /// US SSN-shaped `NNN-NN-NNNN`.
    Ssn,
    /// 13–16 digit card-number-shaped run (Luhn-checked).
    CreditCard,
    /// IPv4 dotted-quad.
    IpV4,
}

impl PiiKind {
    /// The placeholder substituted for a detected span.
    pub fn placeholder(self) -> &'static str {
        match self {
            PiiKind::Email => "[EMAIL]",
            PiiKind::Phone => "[PHONE]",
            PiiKind::Ssn => "[SSN]",
            PiiKind::CreditCard => "[CARD]",
            PiiKind::IpV4 => "[IP]",
        }
    }
}

/// Which detectors to run. All on by default.
#[derive(Debug, Clone)]
pub struct RedactConfig {
    pub email: bool,
    pub phone: bool,
    pub ssn: bool,
    pub credit_card: bool,
    pub ipv4: bool,
}

impl Default for RedactConfig {
    fn default() -> Self {
        Self {
            email: true,
            phone: true,
            ssn: true,
            credit_card: true,
            ipv4: true,
        }
    }
}

/// What a redaction pass found + did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RedactionReport {
    pub redacted: String,
    /// Count per kind, sorted by kind discriminant for determinism.
    pub counts: Vec<(PiiKind, usize)>,
}

impl RedactionReport {
    /// Total spans redacted across all kinds.
    pub fn total(&self) -> usize {
        self.counts.iter().map(|(_, n)| n).sum()
    }
    pub fn changed(&self) -> bool {
        self.total() > 0
    }
}

/// The redaction seam. The host calls this on the write path before any
/// text is stored.
pub trait Redactor: Send + Sync {
    fn redact(&self, text: &str) -> RedactionReport;
}

/// Default dependency-free redactor.
#[derive(Debug, Clone, Default)]
pub struct RegexlessRedactor {
    pub config: RedactConfig,
}

impl RegexlessRedactor {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_config(config: RedactConfig) -> Self {
        Self { config }
    }
}

impl Redactor for RegexlessRedactor {
    fn redact(&self, text: &str) -> RedactionReport {
        redact_with(text, &self.config)
    }
}

/// Convenience free function using the default config.
pub fn redact(text: &str) -> RedactionReport {
    redact_with(text, &RedactConfig::default())
}

/// Core scanner. Runs detectors in a fixed priority order so overlapping
/// matches resolve deterministically: the most specific / highest-risk
/// pattern wins (card → ssn → email → ip → phone).
fn redact_with(text: &str, cfg: &RedactConfig) -> RedactionReport {
    // Token-aware pass over whitespace-separated words handles email/ip
    // cleanly; digit-run scanning handles card/ssn/phone within or across
    // punctuation. We build the output incrementally.
    let mut counts: std::collections::BTreeMap<u8, (PiiKind, usize)> =
        std::collections::BTreeMap::new();
    let mut bump = |k: PiiKind| {
        let e = counts.entry(kind_rank(k)).or_insert((k, 0));
        e.1 += 1;
    };

    // Word-level detectors (email, ipv4) replace whole tokens; everything
    // else is reassembled verbatim so spacing/punctuation is preserved.
    let mut out = String::with_capacity(text.len());
    for (i, token) in split_keep_delims(text).into_iter().enumerate() {
        let _ = i;
        if token.is_delim {
            out.push_str(token.text);
            continue;
        }
        let w = token.text;
        if cfg.email && looks_like_email(w) {
            out.push_str(PiiKind::Email.placeholder());
            bump(PiiKind::Email);
        } else if cfg.ipv4 && looks_like_ipv4(w) {
            out.push_str(PiiKind::IpV4.placeholder());
            bump(PiiKind::IpV4);
        } else {
            // Digit-pattern detectors operate on the bare token.
            match classify_digits(w, cfg) {
                Some(kind) => {
                    out.push_str(kind.placeholder());
                    bump(kind);
                }
                None => out.push_str(w),
            }
        }
    }

    let mut counts: Vec<(PiiKind, usize)> = counts.into_values().collect();
    counts.sort_by_key(|(k, _)| kind_rank(*k));
    RedactionReport {
        redacted: out,
        counts,
    }
}

/// Priority/sort rank — lower = matched first when ambiguous.
fn kind_rank(k: PiiKind) -> u8 {
    match k {
        PiiKind::CreditCard => 0,
        PiiKind::Ssn => 1,
        PiiKind::Email => 2,
        PiiKind::IpV4 => 3,
        PiiKind::Phone => 4,
    }
}

struct Tok<'a> {
    text: &'a str,
    is_delim: bool,
}

/// Split into alternating non-delimiter / delimiter runs, *keeping* the
/// delimiters, so the output can be reassembled byte-for-byte. A
/// "delimiter" is any run of chars that can't be inside the patterns we
/// detect (whitespace + a few separators), but `.`, `@`, `-` stay inside
/// tokens because emails/IPs/SSNs use them.
fn split_keep_delims(s: &str) -> Vec<Tok<'_>> {
    let is_tokenchar =
        |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '@' | '-' | '_' | '+' | '%');
    let mut out = Vec::new();
    let mut start = 0;
    let mut in_tok = None::<bool>;
    for (idx, c) in s.char_indices() {
        let t = is_tokenchar(c);
        match in_tok {
            None => {
                in_tok = Some(t);
                start = idx;
            }
            Some(prev) if prev != t => {
                out.push(Tok {
                    text: &s[start..idx],
                    is_delim: !prev,
                });
                start = idx;
                in_tok = Some(t);
            }
            _ => {}
        }
    }
    if let Some(prev) = in_tok {
        out.push(Tok {
            text: &s[start..],
            is_delim: !prev,
        });
    }
    out
}

fn looks_like_email(w: &str) -> bool {
    // exactly one '@', non-empty local part, domain with a dot and a
    // 2+ char TLD.
    let at = w.find('@');
    let Some(at) = at else { return false };
    if w[at + 1..].contains('@') {
        return false;
    }
    let (local, domain) = (&w[..at], &w[at + 1..]);
    if local.is_empty()
        || !local
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '+' | '%' | '-'))
    {
        return false;
    }
    match domain.rsplit_once('.') {
        Some((host, tld)) => {
            !host.is_empty()
                && host
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
                && tld.len() >= 2
                && tld.chars().all(|c| c.is_ascii_alphabetic())
        }
        None => false,
    }
}

fn looks_like_ipv4(w: &str) -> bool {
    let parts: Vec<&str> = w.split('.').collect();
    if parts.len() != 4 {
        return false;
    }
    parts.iter().all(|p| {
        !p.is_empty()
            && p.len() <= 3
            && p.chars().all(|c| c.is_ascii_digit())
            && p.parse::<u16>().map(|n| n <= 255).unwrap_or(false)
    })
}

/// Detect SSN / credit-card / phone from a token's digit content.
fn classify_digits(w: &str, cfg: &RedactConfig) -> Option<PiiKind> {
    let digits: String = w.chars().filter(|c| c.is_ascii_digit()).collect();
    let n = digits.len();
    // The token must be "mostly digits + separators" — avoid nuking
    // alphanumeric IDs that merely contain numbers.
    let non_sep_non_digit = w
        .chars()
        .any(|c| !c.is_ascii_digit() && !matches!(c, '-' | '.' | ' ' | '(' | ')' | '+'));
    if non_sep_non_digit {
        return None;
    }

    // SSN: exactly NNN-NN-NNNN shape.
    if cfg.ssn && n == 9 && is_ssn_shaped(w) {
        return Some(PiiKind::Ssn);
    }
    // Credit card: 13–16 digits passing Luhn.
    if cfg.credit_card && (13..=16).contains(&n) && luhn_ok(&digits) {
        return Some(PiiKind::CreditCard);
    }
    // Phone: 10–11 digits (US-ish), with separators or a leading +.
    // Token-based, so it catches separator-joined formats
    // (`415-555-0132`, `415.555.0132`, `+14155550132`) but *not*
    // space-broken ones like `(415) 555-0132` — those split into
    // sub-tokens. The `Redactor` trait seam is where a stronger
    // NER/ML detector handles the long tail.
    if cfg.phone && (10..=11).contains(&n) && w.chars().any(|c| !c.is_ascii_digit()) {
        return Some(PiiKind::Phone);
    }
    None
}

fn is_ssn_shaped(w: &str) -> bool {
    // NNN-NN-NNNN exactly.
    let b = w.as_bytes();
    if b.len() != 11 {
        return false;
    }
    let digit = |i: usize| b[i].is_ascii_digit();
    (0..3).all(digit) && b[3] == b'-' && (4..6).all(digit) && b[6] == b'-' && (7..11).all(digit)
}

/// Luhn checksum — keeps random 16-digit runs (e.g. an order id) from
/// being misflagged as card numbers.
fn luhn_ok(digits: &str) -> bool {
    let mut sum = 0u32;
    let mut alt = false;
    for c in digits.chars().rev() {
        let mut d = c.to_digit(10).unwrap_or(0);
        if alt {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        sum += d;
        alt = !alt;
    }
    sum % 10 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_email() {
        let r = redact("contact me at alice.smith+tag@example.co.uk please");
        assert_eq!(r.redacted, "contact me at [EMAIL] please");
        assert_eq!(r.counts, vec![(PiiKind::Email, 1)]);
    }

    #[test]
    fn redacts_ssn() {
        let r = redact("SSN 123-45-6789 on file");
        assert_eq!(r.redacted, "SSN [SSN] on file");
        assert_eq!(r.total(), 1);
    }

    #[test]
    fn redacts_valid_credit_card_luhn() {
        // 4242 4242 4242 4242 is a Luhn-valid test card.
        let r = redact("card 4242-4242-4242-4242 charged");
        assert_eq!(r.redacted, "card [CARD] charged");
    }

    #[test]
    fn does_not_redact_non_luhn_16_digit_run() {
        // A 16-digit order id that fails the Luhn check must survive
        // (1234567812345678 → checksum 8, not 0).
        assert!(!luhn_ok("1234567812345678"), "test fixture must fail Luhn");
        let r = redact("order 1234567812345678 shipped");
        assert!(
            !r.redacted.contains("[CARD]"),
            "non-Luhn run wrongly flagged: {:?}",
            r.redacted
        );
    }

    #[test]
    fn redacts_phone_and_ip() {
        // Separator-joined phone format (the regexless scanner's coverage);
        // space-broken `(415) 555-0132` is a documented gap → NER seam.
        let r = redact("call 415-555-0132 from 192.168.1.42");
        assert!(r.redacted.contains("[PHONE]"), "{:?}", r.redacted);
        assert!(r.redacted.contains("[IP]"), "{:?}", r.redacted);
    }

    #[test]
    fn ipv4_octet_range_enforced() {
        // 999.1.1.1 is not a valid IP → not redacted.
        let r = redact("host 999.1.1.1 down");
        assert!(!r.redacted.contains("[IP]"));
    }

    #[test]
    fn plain_text_is_untouched() {
        let r = redact("the quarterly revenue grew 18 percent");
        assert!(!r.changed());
        assert_eq!(r.redacted, "the quarterly revenue grew 18 percent");
    }

    #[test]
    fn multiple_kinds_counted_and_sorted() {
        let r = redact("a@b.com and 123-45-6789 and a@b.com");
        // 2 emails + 1 ssn; counts sorted by rank (ssn=1 before email=2).
        assert_eq!(r.counts, vec![(PiiKind::Ssn, 1), (PiiKind::Email, 2)]);
        assert_eq!(r.total(), 3);
    }

    #[test]
    fn config_can_disable_a_detector() {
        let cfg = RedactConfig {
            email: false,
            ..Default::default()
        };
        let red = RegexlessRedactor::with_config(cfg);
        let r = red.redact("reach a@b.com now");
        assert!(!r.redacted.contains("[EMAIL]"));
        assert_eq!(r.redacted, "reach a@b.com now");
    }

    #[test]
    fn reassembly_is_byte_exact_for_clean_text() {
        let input = "Hello,  world!\n\tTabs and   spaces — kept.";
        let r = redact(input);
        assert_eq!(r.redacted, input);
    }

    #[test]
    fn luhn_helper_correct() {
        assert!(luhn_ok("4242424242424242"));
        assert!(!luhn_ok("4242424242424241"));
    }
}
