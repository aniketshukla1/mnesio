//! Competitive comparison harness — mneme in context, honestly.
//!
//! Two things matter when claiming a memory layer is "best in market", and
//! they are *different metrics that must not be conflated*:
//!
//! 1. **Capability** — what the system can structurally do. This is where
//!    mneme's moat lives (append-only + replayable + bi-temporal substrate
//!    behind a non-bypassable safety gate, and the frontier features that only
//!    that substrate makes possible). The [`capability_matrix`] is a factual,
//!    defensible comparison against the published architectures of the leading
//!    systems.
//!
//! 2. **Benchmark score** — how well it answers questions. The leading systems
//!    publish **end-to-end QA accuracy** (an LLM-as-judge score, `J`) on
//!    LOCOMO / LongMemEval. mneme's offline, LLM-free number is **retrieval
//!    recall@k** — a *retrieval-quality proxy*, not the same metric. We report
//!    both side by side but label them honestly: recall@k says "the answer was
//!    in the retrieved set", QA-J says "the model produced the right answer".
//!    A true apples-to-apples QA-J run needs an LLM judge over mneme's
//!    retrieved context (mneme has the `Judge` + synthesizer path for it; it's
//!    just not part of the offline CI number).
//!
//! The cited competitor numbers below are transcribed from the referenced
//! papers; see [`cited_results`] for the exact source per row. We never ship a
//! competitor's self-reported number as if it were ours, and we never present
//! recall@k as if it beat a QA-J score.

use crate::memeval::{load_memeval_suite, run_memeval, MemEvalReport};
use anyhow::Result;

const LOCOMO_JSON: &str = include_str!("../data/locomo_mini.json");
const LONGMEMEVAL_JSON: &str = include_str!("../data/longmemeval_mini.json");

/// How fully a system supports a capability, per its published architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// Shipped / structural.
    Yes,
    /// Partial — present in a weaker or non-reconciled form.
    Partial,
    /// Not part of the published design.
    No,
}

impl Support {
    pub fn mark(&self) -> &'static str {
        match self {
            Support::Yes => "✅",
            Support::Partial => "◑",
            Support::No => "—",
        }
    }
}

/// One row of the capability matrix. Competitor cells reflect each system's
/// *published* architecture (papers/docs cited in the README), not a running
/// install; they may evolve.
#[derive(Debug, Clone)]
pub struct CapabilityRow {
    pub capability: &'static str,
    pub mneme: Support,
    pub mem0: Support,
    pub zep: Support,
    pub letta: Support,
    pub amem: Support,
}

/// The capability comparison. Conservative on competitors: `Yes`/`Partial`
/// only where their published design clearly supports it.
pub fn capability_matrix() -> Vec<CapabilityRow> {
    use Support::*;
    vec![
        CapabilityRow {
            capability: "Append-only, replayable event log as system of record",
            mneme: Yes,
            mem0: No,
            zep: Partial,
            letta: No,
            amem: No,
        },
        CapabilityRow {
            capability: "Bi-temporal versioning (never overwrite; invalidate-and-supersede)",
            mneme: Yes,
            mem0: Partial,
            zep: Yes,
            letta: No,
            amem: No,
        },
        CapabilityRow {
            capability: "Hybrid retrieval (vector + BM25 + RRF) with explainable breakdown",
            mneme: Yes,
            mem0: Partial,
            zep: Partial,
            letta: Partial,
            amem: Partial,
        },
        CapabilityRow {
            capability: "Procedural self-improvement (agent gets better at tasks over time)",
            mneme: Yes,
            mem0: No,
            zep: No,
            letta: Partial,
            amem: No,
        },
        CapabilityRow {
            capability:
                "Non-bypassable commit gate (canaries + safety probe before any learned change)",
            mneme: Yes,
            mem0: No,
            zep: No,
            letta: No,
            amem: No,
        },
        CapabilityRow {
            capability:
                "Counterfactual contribution scoring + GC by measurement (not time-decay guess)",
            mneme: Yes,
            mem0: No,
            zep: No,
            letta: No,
            amem: No,
        },
        CapabilityRow {
            capability: "Self-falsifying memory (acceptance probes auto-supersede on failure)",
            mneme: Yes,
            mem0: No,
            zep: No,
            letta: No,
            amem: No,
        },
        CapabilityRow {
            capability: "Crypto-shred erasure reconciled with an append-only audit log",
            mneme: Yes,
            mem0: No,
            zep: No,
            letta: No,
            amem: No,
        },
        CapabilityRow {
            capability: "Time-travel reconstruction + provenance chains (answer as-of past T)",
            mneme: Yes,
            mem0: No,
            zep: Partial,
            letta: No,
            amem: No,
        },
        CapabilityRow {
            capability: "Certified skill exchange (portable gated artifact, re-gated on import)",
            mneme: Yes,
            mem0: No,
            zep: No,
            letta: No,
            amem: No,
        },
        CapabilityRow {
            capability: "Self-contained / embedded (no external vector or graph DB required)",
            mneme: Yes,
            mem0: Partial,
            zep: Partial,
            letta: Yes,
            amem: Partial,
        },
    ]
}

/// A published benchmark number, with its exact source. **Not** mneme's — these
/// are competitor / baseline figures transcribed from the cited papers.
#[derive(Debug, Clone)]
pub struct CitedResult {
    pub system: &'static str,
    pub benchmark: &'static str,
    pub metric: &'static str,
    pub score_pct: f64,
    pub source: &'static str,
}

/// Published end-to-end QA scores from the leading memory-systems papers.
///
/// LOCOMO numbers are the overall **LLM-as-a-Judge (J)** score from the Mem0
/// paper's Table 2 (arXiv:2504.19413). LongMemEval numbers are the overall
/// **QA accuracy** from the Zep paper's Table 2 (arXiv:2501.13956). Both are
/// end-to-end answer-correctness metrics — distinct from mneme's retrieval
/// recall@k (see the module docs).
pub fn cited_results() -> Vec<CitedResult> {
    vec![
        // --- LOCOMO, LLM-as-a-Judge (J) overall, from Mem0 paper Table 2 ---
        CitedResult {
            system: "Full-context (upper bound)",
            benchmark: "LOCOMO",
            metric: "LLM-as-Judge (J)",
            score_pct: 72.90,
            source: "Mem0 paper, arXiv:2504.19413, Table 2",
        },
        CitedResult {
            system: "Mem0 (graph)",
            benchmark: "LOCOMO",
            metric: "LLM-as-Judge (J)",
            score_pct: 68.44,
            source: "Mem0 paper, arXiv:2504.19413, Table 2",
        },
        CitedResult {
            system: "Mem0",
            benchmark: "LOCOMO",
            metric: "LLM-as-Judge (J)",
            score_pct: 66.88,
            source: "Mem0 paper, arXiv:2504.19413, Table 2",
        },
        CitedResult {
            system: "Zep",
            benchmark: "LOCOMO",
            metric: "LLM-as-Judge (J)",
            score_pct: 65.99,
            source: "Mem0 paper, arXiv:2504.19413, Table 2",
        },
        CitedResult {
            system: "RAG (k=2, 256-tok)",
            benchmark: "LOCOMO",
            metric: "LLM-as-Judge (J)",
            score_pct: 60.97,
            source: "Mem0 paper, arXiv:2504.19413, Table 2",
        },
        CitedResult {
            system: "LangMem",
            benchmark: "LOCOMO",
            metric: "LLM-as-Judge (J)",
            score_pct: 58.10,
            source: "Mem0 paper, arXiv:2504.19413, Table 2",
        },
        CitedResult {
            system: "OpenAI memory",
            benchmark: "LOCOMO",
            metric: "LLM-as-Judge (J)",
            score_pct: 52.90,
            source: "Mem0 paper, arXiv:2504.19413, Table 2",
        },
        CitedResult {
            system: "A-Mem",
            benchmark: "LOCOMO",
            metric: "LLM-as-Judge (J)",
            score_pct: 48.38,
            source: "Mem0 paper, arXiv:2504.19413, Table 2",
        },
        // --- LongMemEval, QA accuracy, from Zep paper Table 2 ---
        CitedResult {
            system: "Zep (gpt-4o)",
            benchmark: "LongMemEval",
            metric: "QA accuracy",
            score_pct: 71.2,
            source: "Zep paper, arXiv:2501.13956, Table 2",
        },
        CitedResult {
            system: "Full-context (gpt-4o)",
            benchmark: "LongMemEval",
            metric: "QA accuracy",
            score_pct: 60.2,
            source: "Zep paper, arXiv:2501.13956, Table 2",
        },
        CitedResult {
            system: "Zep (gpt-4o-mini)",
            benchmark: "LongMemEval",
            metric: "QA accuracy",
            score_pct: 63.8,
            source: "Zep paper, arXiv:2501.13956, Table 2",
        },
        CitedResult {
            system: "Full-context (gpt-4o-mini)",
            benchmark: "LongMemEval",
            metric: "QA accuracy",
            score_pct: 55.4,
            source: "Zep paper, arXiv:2501.13956, Table 2",
        },
    ]
}

/// A full competitive report: mneme's *measured* retrieval recall on the
/// LOCOMO/LongMemEval-style suites, plus the capability matrix and the cited
/// competitor QA scores.
pub struct CompeteReport {
    pub k: usize,
    pub embedder: String,
    pub mneme_locomo: MemEvalReport,
    pub mneme_longmemeval: MemEvalReport,
    pub capabilities: Vec<CapabilityRow>,
    pub cited: Vec<CitedResult>,
}

/// Run mneme's own measured numbers and assemble the comparison.
pub async fn run_compete(k: usize, embedder: &str) -> Result<CompeteReport> {
    let locomo = load_memeval_suite(LOCOMO_JSON)?;
    let longmemeval = load_memeval_suite(LONGMEMEVAL_JSON)?;
    let mneme_locomo = run_memeval(&locomo, k, embedder).await?;
    let mneme_longmemeval = run_memeval(&longmemeval, k, embedder).await?;
    Ok(CompeteReport {
        k,
        embedder: embedder.to_string(),
        mneme_locomo,
        mneme_longmemeval,
        capabilities: capability_matrix(),
        cited: cited_results(),
    })
}

/// Render the comparison as Markdown, with the methodology caveat up front so
/// the two metrics are never read as one.
pub fn compete_markdown(r: &CompeteReport) -> String {
    let mut out = String::new();
    out.push_str("# How mneme compares\n\n");
    out.push_str(
        "> **Two different metrics, kept separate.** The capability matrix is a \
         structural comparison. The benchmark tables mix *cited* competitor \
         **end-to-end QA accuracy** with mneme's *measured* **retrieval \
         recall@k** — a retrieval-quality proxy, not the same metric. recall@k \
         answers \"was the gold answer in the retrieved set?\"; QA accuracy \
         answers \"did the model produce the right answer?\". Do not read them \
         as a single ranking.\n\n",
    );

    // --- Capability matrix ---
    out.push_str("## Capability matrix\n\n");
    out.push_str("| Capability | mneme | Mem0 | Zep | Letta | A-MEM |\n");
    out.push_str("|---|:---:|:---:|:---:|:---:|:---:|\n");
    for row in &r.capabilities {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            row.capability,
            row.mneme.mark(),
            row.mem0.mark(),
            row.zep.mark(),
            row.letta.mark(),
            row.amem.mark(),
        ));
    }
    out.push_str(
        "\n✅ shipped · ◑ partial · — not in published design. Competitor cells \
         reflect each system's published architecture and may evolve. mneme is \
         the only column with every row because the frontier features require \
         the append-only + replayable + bi-temporal substrate behind a \
         non-bypassable gate.\n\n",
    );

    // --- mneme measured ---
    out.push_str(&format!(
        "## mneme — measured retrieval recall@{} (embedder: {})\n\n",
        r.k, r.embedder
    ));
    out.push_str("| Suite | Memories | Questions | recall@k |\n");
    out.push_str("|---|---:|---:|---:|\n");
    for (label, rep) in [
        ("LOCOMO-mini", &r.mneme_locomo),
        ("LongMemEval-mini", &r.mneme_longmemeval),
    ] {
        out.push_str(&format!(
            "| {} | {} | {} | {:.1}% |\n",
            label,
            rep.memory_count,
            rep.total_questions,
            rep.recall() * 100.0
        ));
    }
    out.push_str(
        "\nMeasured live through the real `FjallEventLog → VectorView + Bm25View \
         → HybridRetriever` path. These are curated mini-suites for offline CI; \
         for a published end-to-end number, run `mneme-bench fetch` (SQuAD) or \
         an LLM-judge pass over the full LOCOMO/LongMemEval.\n\n",
    );

    // --- cited competitor QA scores ---
    out.push_str("## Cited competitor / baseline scores (end-to-end QA)\n\n");
    out.push_str("| System | Benchmark | Metric | Score | Source |\n");
    out.push_str("|---|---|---|---:|---|\n");
    for c in &r.cited {
        out.push_str(&format!(
            "| {} | {} | {} | {:.2}% | {} |\n",
            c.system, c.benchmark, c.metric, c.score_pct, c.source
        ));
    }
    out.push_str(
        "\nThese are competitor/baseline numbers transcribed from the cited \
         papers — **not** mneme's, and **not** directly comparable to the \
         recall@k above (different metric). They establish the landscape mneme \
         enters; mneme's differentiation is the capability matrix.\n",
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_matrix_is_complete_and_mneme_leads_the_moat_rows() {
        let m = capability_matrix();
        assert!(m.len() >= 10, "expected a substantial matrix");
        // Every row must have mneme support (Yes/Partial), never No.
        for row in &m {
            assert_ne!(
                row.mneme,
                Support::No,
                "mneme should support every listed capability: {}",
                row.capability
            );
        }
        // The frontier moat rows should be mneme-only (no competitor Yes).
        let moat = m
            .iter()
            .find(|r| r.capability.contains("Non-bypassable commit gate"))
            .unwrap();
        assert_eq!(moat.mneme, Support::Yes);
        assert_eq!(moat.mem0, Support::No);
        assert_eq!(moat.zep, Support::No);
    }

    #[test]
    fn cited_results_all_carry_a_source() {
        let cited = cited_results();
        assert!(!cited.is_empty());
        for c in &cited {
            assert!(
                c.source.contains("arXiv"),
                "every cited number must name its source: {} {}",
                c.system,
                c.benchmark
            );
            assert!(c.score_pct > 0.0 && c.score_pct <= 100.0);
        }
        // Sanity-check a couple of transcribed anchors.
        let zep_4o = cited
            .iter()
            .find(|c| c.system == "Zep (gpt-4o)" && c.benchmark == "LongMemEval")
            .unwrap();
        assert!((zep_4o.score_pct - 71.2).abs() < 1e-6);
        let mem0g = cited
            .iter()
            .find(|c| c.system == "Mem0 (graph)" && c.benchmark == "LOCOMO")
            .unwrap();
        assert!((mem0g.score_pct - 68.44).abs() < 1e-6);
    }

    #[tokio::test]
    async fn run_compete_produces_measured_recall_and_markdown() {
        let r = run_compete(10, "mock").await.unwrap();
        assert!(r.mneme_locomo.total_questions > 0);
        assert!(r.mneme_longmemeval.total_questions > 0);
        let md = compete_markdown(&r);
        // The methodology caveat and both metric kinds must be present.
        assert!(md.contains("Two different metrics"));
        assert!(md.contains("Capability matrix"));
        assert!(md.contains("recall@"));
        assert!(md.contains("LLM-as-Judge"));
        assert!(md.contains("arXiv:2504.19413"));
        assert!(md.contains("arXiv:2501.13956"));
    }
}
