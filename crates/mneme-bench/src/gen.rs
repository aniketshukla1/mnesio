//! Synthetic corpus generator for large-scale + load benchmarking.
//!
//! Builds an arbitrarily large, *deterministic* memory corpus from templated
//! multi-domain facts, with a labeled **needle set** (known query → gold
//! memory) salted among distractors so recall@k is measurable at any size. It
//! also injects **contradictions** and **evolution chains** so the corpus
//! exercises consolidation / provenance, not just flat retrieval.
//!
//! Everything is seeded (SplitMix64, dependency-free — same family as the
//! keyring's keystream), so a given `(n, seed)` always yields the identical
//! corpus: reproducible scale runs, comparable across mneme versions.
//!
//! The output reuses the harness's [`crate::memeval::MemItem`] /
//! [`crate::memeval::MemQuestion`] shapes, so a generated corpus drops straight
//! into [`crate::memeval::run_memeval`] — and the scale harness consumes the
//! richer [`GeneratedCorpus`] (which also exposes the gold needle ids).

use crate::memeval::{MemEvalSuite, MemItem, MemQuestion};

/// Deterministic SplitMix64 PRNG — tiny, dependency-free, good enough for
/// reproducible synthetic data (NOT for cryptography).
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, n)`. `n == 0` returns 0.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

/// One generated memory plus whether it's a labeled needle (and for which
/// query) — the scale harness needs the gold mapping to score recall.
#[derive(Debug, Clone)]
pub struct GenMemory {
    pub content: String,
    pub tags: Vec<String>,
    /// `Some(query_id)` if this memory is the gold answer for a needle query.
    pub needle_for: Option<usize>,
    /// `Some(parent_index)` if this memory is an evolution of an earlier one.
    pub evolves: Option<usize>,
    /// `true` if this memory deliberately contradicts an earlier one.
    pub contradicts: bool,
}

/// A generated needle: a query whose gold answer is a specific memory.
#[derive(Debug, Clone)]
pub struct GenNeedle {
    pub query: String,
    /// Case-insensitive substring proving the gold memory was retrieved.
    pub answer_substring: String,
    pub category: String,
}

/// A fully-generated synthetic corpus: the haystack + the labeled needles.
#[derive(Debug, Clone)]
pub struct GeneratedCorpus {
    pub memories: Vec<GenMemory>,
    pub needles: Vec<GenNeedle>,
    pub n: usize,
    pub seed: u64,
    /// How many memories are evolution children.
    pub evolution_count: usize,
    /// How many memories are deliberate contradictions.
    pub contradiction_count: usize,
}

impl GeneratedCorpus {
    /// Project to the harness's [`MemEvalSuite`] so a generated corpus runs
    /// through [`crate::memeval::run_memeval`] unchanged.
    pub fn to_suite(&self) -> MemEvalSuite {
        MemEvalSuite {
            name: format!("synthetic-{}-seed{}", self.n, self.seed),
            description: format!(
                "Deterministic synthetic corpus: {} memories, {} needles, \
                 {} evolution children, {} contradictions.",
                self.memories.len(),
                self.needles.len(),
                self.evolution_count,
                self.contradiction_count
            ),
            memories: self
                .memories
                .iter()
                .map(|m| MemItem {
                    content: m.content.clone(),
                    tags: m.tags.clone(),
                })
                .collect(),
            questions: self
                .needles
                .iter()
                .map(|q| MemQuestion {
                    question: q.query.clone(),
                    answer_substring: q.answer_substring.clone(),
                    category: q.category.clone(),
                })
                .collect(),
        }
    }
}

// --- templated fact vocabulary (multi-domain, paraphrase-friendly) ---

/// Distinct "subjects" — unique tokens so each needle has an unambiguous gold.
const COMPANIES: &[&str] = &[
    "Acme",
    "Globex",
    "Initech",
    "Umbrella",
    "Stark",
    "Wayne",
    "Wonka",
    "Hooli",
    "Vandelay",
    "Soylent",
    "Tyrell",
    "Cyberdyne",
    "Aperture",
    "Massive",
    "Pied-Piper",
    "Gekko",
    "Oscorp",
    "Bluth",
    "Dunder",
    "Prestige",
];
const CITIES: &[&str] = &[
    "Lagos", "Oslo", "Quito", "Perth", "Riga", "Cusco", "Hanoi", "Tunis", "Davao", "Bergen",
    "Kochi", "Mendoza", "Aarhus", "Galway", "Hobart", "Toledo",
];
const METRICS: &[&str] = &[
    "revenue",
    "headcount",
    "churn",
    "margin",
    "latency",
    "uptime",
    "NPS",
    "burn rate",
    "runway",
    "ARR",
];
const DOMAINS: &[&str] = &[
    "finance", "product", "ops", "people", "market", "infra", "legal",
];

impl GeneratedCorpus {
    /// Generate a corpus of `n` memories (plus a proportional set of evolution
    /// children + contradictions) with `seed`. Roughly 5% of base memories are
    /// promoted to needles, ~3% get an evolution child, ~2% a contradiction.
    pub fn generate(n: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut memories: Vec<GenMemory> = Vec::with_capacity(n + n / 20);
        let mut needles: Vec<GenNeedle> = Vec::new();
        // Indices of distractor (non-needle) base memories — only these are
        // eligible as evolution/contradiction parents, so a child can never
        // duplicate a needle's unique gold token (keeps recall unambiguous).
        let mut distractors: Vec<usize> = Vec::new();

        // Base haystack.
        for i in 0..n {
            // Promote ~1 in 20 to a needle with a unique, recoverable token.
            let is_needle = i % 20 == 7;
            if is_needle {
                let qid = needles.len();
                let company = COMPANIES[i % COMPANIES.len()];
                let city = CITIES[(i / 7) % CITIES.len()];
                let metric = METRICS[(i / 3) % METRICS.len()];
                let value = 100 + (rng.below(900));
                // Gold token: a sentinel-delimited, fixed-width id. Fixed width
                // means no token is a prefix of another (NDL0000007 ⊄ NDL0000070),
                // and the 'Z' sentinels keep it out of ordinary prose — so the
                // token appears in exactly one memory at any scale.
                let token = format!("ZNDL{qid:09}Z");
                let content = format!(
                    "[{}] {token} ({company}): {metric} of {value} units in {city} for the quarter.",
                    DOMAINS[i % DOMAINS.len()]
                );
                memories.push(GenMemory {
                    content,
                    tags: vec![metric.to_string(), company.to_lowercase(), "needle".into()],
                    needle_for: Some(qid),
                    evolves: None,
                    contradicts: false,
                });
                needles.push(GenNeedle {
                    query: format!("what {metric} did {token} report in {city}?"),
                    answer_substring: token,
                    category: "single-hop".to_string(),
                });
            } else {
                // A distractor: plausible but not a gold target.
                let company = rng.pick(COMPANIES);
                let city = rng.pick(CITIES);
                let metric = rng.pick(METRICS);
                let domain = rng.pick(DOMAINS);
                let value = rng.below(1000);
                distractors.push(memories.len());
                memories.push(GenMemory {
                    content: format!(
                        "[{domain}] {company} team in {city} noted {metric} trending around {value} this period."
                    ),
                    tags: vec![metric.to_string(), domain.to_string()],
                    needle_for: None,
                    evolves: None,
                    contradicts: false,
                });
            }
        }

        // Inject evolution chains: ~3% of distractor memories get a refining
        // child. (Parents are distractors so a child never copies a gold token.)
        let evo_target = if distractors.is_empty() {
            0
        } else {
            (n / 33).max(1)
        };
        let mut evolution_count = 0;
        for _ in 0..evo_target {
            let parent = distractors[rng.below(distractors.len())];
            let pc = memories[parent].content.clone();
            memories.push(GenMemory {
                content: format!("[update] Refined: {pc} (figures restated after review)"),
                tags: vec!["update".into()],
                needle_for: None,
                evolves: Some(parent),
                contradicts: false,
            });
            evolution_count += 1;
        }

        // Inject contradictions: ~2% deliberately conflict with a distractor.
        let contra_target = if distractors.is_empty() {
            0
        } else {
            (n / 50).max(1)
        };
        let mut contradiction_count = 0;
        for _ in 0..contra_target {
            let target = distractors[rng.below(distractors.len())];
            let tc = memories[target].content.clone();
            memories.push(GenMemory {
                content: format!(
                    "[correction] Contrary to an earlier note, the opposite holds: {tc}"
                ),
                tags: vec!["correction".into(), "contradiction".into()],
                needle_for: None,
                evolves: None,
                contradicts: true,
            });
            contradiction_count += 1;
        }

        Self {
            memories,
            needles,
            n,
            seed,
            evolution_count,
            contradiction_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic_for_same_seed() {
        let a = GeneratedCorpus::generate(500, 42);
        let b = GeneratedCorpus::generate(500, 42);
        assert_eq!(a.memories.len(), b.memories.len());
        for (x, y) in a.memories.iter().zip(b.memories.iter()) {
            assert_eq!(x.content, y.content, "same seed must reproduce content");
        }
        assert_eq!(a.needles.len(), b.needles.len());
    }

    #[test]
    fn different_seeds_differ() {
        let a = GeneratedCorpus::generate(500, 1);
        let b = GeneratedCorpus::generate(500, 2);
        // Overwhelmingly likely to differ somewhere in the distractor stream.
        let diffs = a
            .memories
            .iter()
            .zip(b.memories.iter())
            .filter(|(x, y)| x.content != y.content)
            .count();
        assert!(
            diffs > 0,
            "different seeds should produce different corpora"
        );
    }

    #[test]
    fn needles_have_unique_recoverable_gold_tokens() {
        let c = GeneratedCorpus::generate(2000, 7);
        assert!(!c.needles.is_empty(), "should produce needles at this size");
        // Each needle's answer_substring must appear in exactly one memory
        // (unambiguous gold), and that memory must be flagged as its needle.
        for (qid, needle) in c.needles.iter().enumerate() {
            let matches: Vec<&GenMemory> = c
                .memories
                .iter()
                .filter(|m| m.content.contains(&needle.answer_substring))
                .collect();
            assert_eq!(
                matches.len(),
                1,
                "needle {qid:?} token {:?} must be unique; matched {}",
                needle.answer_substring,
                matches.len()
            );
            assert_eq!(matches[0].needle_for, Some(qid));
        }
    }

    #[test]
    fn injects_evolution_and_contradiction_at_scale() {
        let c = GeneratedCorpus::generate(5000, 3);
        assert!(c.evolution_count > 0, "expected evolution children");
        assert!(c.contradiction_count > 0, "expected contradictions");
        // Total memories = base + evolution + contradiction.
        assert_eq!(
            c.memories.len(),
            c.n + c.evolution_count + c.contradiction_count
        );
        // The flagged children round-trip.
        let evos = c.memories.iter().filter(|m| m.evolves.is_some()).count();
        let contras = c.memories.iter().filter(|m| m.contradicts).count();
        assert_eq!(evos, c.evolution_count);
        assert_eq!(contras, c.contradiction_count);
    }

    #[test]
    fn to_suite_projects_cleanly() {
        let c = GeneratedCorpus::generate(300, 9);
        let suite = c.to_suite();
        assert_eq!(suite.memories.len(), c.memories.len());
        assert_eq!(suite.questions.len(), c.needles.len());
        assert!(suite.name.contains("synthetic"));
    }

    #[test]
    fn tiny_n_does_not_panic() {
        for n in [0usize, 1, 5, 19, 20, 21] {
            let c = GeneratedCorpus::generate(n, 1);
            assert_eq!(c.n, n);
            // Projection must always be valid.
            let _ = c.to_suite();
        }
    }
}
