//! The [`Consolidator`] — extract → decide ADD/UPDATE/NOOP per fact.

use crate::extractor::Extractor;
use crate::parse::{parse_decision, Decision};
use crate::prompts;
use crate::{ConsolidationAction, ConsolidationPlan, ExistingMemory};
use mneme_core::types::{new_id, MemoryRef};
use mneme_core::{LlmClient, MnemeError};
use std::sync::Arc;

/// Tunables for the consolidation pass.
#[derive(Debug, Clone)]
pub struct ConsolidateConfig {
    /// How many existing candidate memories to show the decision prompt.
    /// The host supplies the candidate pool (usually retriever top-k);
    /// this caps how many actually go into the prompt to bound tokens.
    pub max_candidates: usize,
    /// When two facts extracted from the *same* turn are
    /// case-insensitively identical, collapse them to one ADD rather
    /// than asking the LLM twice. Cheap intra-batch dedup.
    pub intra_batch_dedup: bool,
}

impl Default for ConsolidateConfig {
    fn default() -> Self {
        Self {
            max_candidates: 6,
            intra_batch_dedup: true,
        }
    }
}

/// Turns raw text into a [`ConsolidationPlan`]. Pure with respect to
/// I/O: it calls the [`Extractor`] and an [`LlmClient`] (both injected)
/// but never touches the event log or any store. The host applies the
/// plan's actions as events.
pub struct Consolidator<E: Extractor> {
    extractor: E,
    llm: Arc<dyn LlmClient>,
    config: ConsolidateConfig,
}

impl<E: Extractor> Consolidator<E> {
    pub fn new(extractor: E, llm: Arc<dyn LlmClient>) -> Self {
        Self {
            extractor,
            llm,
            config: ConsolidateConfig::default(),
        }
    }

    pub fn with_config(mut self, config: ConsolidateConfig) -> Self {
        self.config = config;
        self
    }

    /// Extract facts from `raw` and decide an action for each against the
    /// supplied `candidates` (the memories the host already holds that
    /// might overlap — typically retriever top-k, scope-filtered).
    ///
    /// Decisions are made *sequentially* and are aware of earlier
    /// decisions in the same batch: a fact that ADDs becomes a candidate
    /// for subsequent facts, so two paraphrases of the same new fact in
    /// one turn collapse to one ADD + one NOOP rather than two ADDs.
    pub async fn consolidate(
        &self,
        raw: &str,
        candidates: &[ExistingMemory],
    ) -> Result<ConsolidationPlan, MnemeError> {
        let facts = self.extractor.extract(raw).await?;
        let mut plan = ConsolidationPlan::default();

        // Working candidate set = the host-supplied existing memories,
        // capped, plus any facts we ADD during this batch. Each ADD gets
        // a provisional ULID and joins this set, so a later fact in the
        // same batch can dedup (NOOP) or supersede (UPDATE) against it
        // with a stable reference the host will honour.
        let mut existing: Vec<ExistingMemory> = candidates
            .iter()
            .take(self.config.max_candidates)
            .cloned()
            .collect();

        // Records an ADD: assigns a provisional id, emits the action,
        // and registers it as a candidate for subsequent facts.
        let do_add =
            |plan: &mut ConsolidationPlan, existing: &mut Vec<ExistingMemory>, content: String| {
                let id = MemoryRef(new_id());
                existing.push(ExistingMemory::new(id, content.clone()));
                plan.actions.push(ConsolidationAction::Add { id, content });
            };

        for fact in facts {
            // Cheap exact dedup (no LLM call) against the whole working
            // set — catches verbatim duplicates of both host-held and
            // batch-added memories.
            if self.config.intra_batch_dedup {
                if let Some(dup) = existing
                    .iter()
                    .find(|e| e.content.eq_ignore_ascii_case(fact.trim()))
                {
                    plan.actions.push(ConsolidationAction::Noop {
                        duplicate_of: dup.id,
                    });
                    continue;
                }
            }

            let candidate_refs: Vec<&str> = existing.iter().map(|e| e.content.as_str()).collect();
            let prompt = prompts::decide_action(&fact, &candidate_refs);
            let response = self.llm.complete(&prompt).await?;
            let decision = parse_decision(&response, candidate_refs.len());

            match decision {
                Decision::Add => do_add(&mut plan, &mut existing, fact),
                Decision::Noop(i) => {
                    if let Some(target) = existing.get(i) {
                        let dup = target.id;
                        plan.actions
                            .push(ConsolidationAction::Noop { duplicate_of: dup });
                    } else {
                        // Parse bounds-checks, so this is unreachable in
                        // practice; add rather than drop to be safe.
                        do_add(&mut plan, &mut existing, fact);
                    }
                }
                Decision::Update(i, reason) => {
                    if let Some(target) = existing.get(i) {
                        let target_id = target.id;
                        plan.actions.push(ConsolidationAction::Update {
                            target: target_id,
                            content: fact,
                            reason,
                        });
                    } else {
                        do_add(&mut plan, &mut existing, fact);
                    }
                }
            }
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::UpdateReason;
    use crate::ConsolidationAction;
    use async_trait::async_trait;
    use mneme_core::types::{new_id, MemoryRef};
    use mneme_llm::FakeLlmClient;

    /// A canned extractor so tests control the fact list exactly.
    struct CannedExtractor(Vec<String>);
    #[async_trait]
    impl Extractor for CannedExtractor {
        async fn extract(&self, _raw: &str) -> Result<Vec<String>, MnemeError> {
            Ok(self.0.clone())
        }
    }

    fn em(content: &str) -> ExistingMemory {
        ExistingMemory::new(MemoryRef(new_id()), content)
    }

    #[tokio::test]
    async fn add_when_no_candidates() {
        let llm = Arc::new(FakeLlmClient::new().with_default("DECISION: ADD"));
        let c = Consolidator::new(CannedExtractor(vec!["Alice likes oat milk".into()]), llm);
        let plan = c.consolidate("raw", &[]).await.unwrap();
        assert_eq!(plan.adds(), 1);
        match &plan.actions[0] {
            ConsolidationAction::Add { content, .. } => {
                assert_eq!(content, "Alice likes oat milk")
            }
            other => panic!("expected Add, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn noop_when_duplicate() {
        let dup = em("Alice likes oat milk");
        let dup_id = dup.id;
        let llm = Arc::new(FakeLlmClient::new().with_default("DECISION: NOOP 1"));
        let c = Consolidator::new(CannedExtractor(vec!["Alice likes oat milk".into()]), llm);
        let plan = c.consolidate("raw", &[dup]).await.unwrap();
        assert_eq!(plan.noops(), 1);
        assert_eq!(
            plan.actions[0],
            ConsolidationAction::Noop {
                duplicate_of: dup_id
            }
        );
    }

    #[tokio::test]
    async fn update_on_contradiction() {
        let old = em("Acme Q3 revenue grew 18%");
        let old_id = old.id;
        let llm = Arc::new(FakeLlmClient::new().with_default("DECISION: UPDATE 1 CONTRADICTION"));
        let c = Consolidator::new(
            CannedExtractor(vec!["Acme Q3 revenue grew 16%, not 18%".into()]),
            llm,
        );
        let plan = c.consolidate("raw", &[old]).await.unwrap();
        assert_eq!(plan.updates(), 1);
        match &plan.actions[0] {
            ConsolidationAction::Update {
                target,
                reason,
                content,
            } => {
                assert_eq!(*target, old_id);
                assert_eq!(*reason, UpdateReason::Contradiction);
                assert_eq!(content, "Acme Q3 revenue grew 16%, not 18%");
            }
            other => panic!("expected Update, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn intra_batch_dedup_collapses_identical_facts() {
        // Two identical facts in one turn, no candidates. First ADDs;
        // the second is caught by the cheap exact-dedup → NOOP pointing
        // at the just-added memory, with no second LLM call.
        let llm = Arc::new(FakeLlmClient::new().with_default("DECISION: ADD"));
        let c = Consolidator::new(
            CannedExtractor(vec![
                "Alice likes oat milk".into(),
                "alice likes oat milk".into(),
            ]),
            llm.clone(),
        );
        let plan = c.consolidate("raw", &[]).await.unwrap();
        assert_eq!(plan.adds(), 1);
        assert_eq!(plan.noops(), 1);
        assert_eq!(plan.actions.len(), 2);
        // The NOOP must point at the id the ADD was assigned.
        if let (ConsolidationAction::Add { id, .. }, ConsolidationAction::Noop { duplicate_of }) =
            (&plan.actions[0], &plan.actions[1])
        {
            assert_eq!(
                id, duplicate_of,
                "dedup NOOP must reference the batch ADD's id"
            );
        } else {
            panic!("expected [Add, Noop], got {:?}", plan.actions);
        }
        assert_eq!(
            llm.call_count(),
            1,
            "second identical fact must not hit the LLM"
        );
    }

    #[tokio::test]
    async fn mixed_batch_add_then_noop_against_batch_member() {
        // Fact 1 is new (ADD); fact 2 is a paraphrase the LLM marks as
        // NOOP against candidate 1 — which is the just-added fact, since
        // batch-added facts join the working candidate set.
        let new_fact = "Bob moved to Berlin";
        let para = "Bob now lives in Berlin";
        // First call: no candidates → ADD. Second call: 1 candidate
        // (the batch-added fact) → NOOP 1.
        let llm = Arc::new(
            FakeLlmClient::new()
                .with_prefix_match(
                    // decide_action prompt for the paraphrase contains the
                    // batch-added fact as candidate 1.
                    "A new candidate fact has been extracted:\nBob now lives in Berlin",
                    "DECISION: NOOP 1",
                )
                .with_default("DECISION: ADD"),
        );
        let c = Consolidator::new(CannedExtractor(vec![new_fact.into(), para.into()]), llm);
        let plan = c.consolidate("raw", &[]).await.unwrap();
        assert_eq!(plan.adds(), 1);
        assert_eq!(plan.noops(), 1);
    }

    #[tokio::test]
    async fn empty_extraction_yields_empty_plan() {
        let llm = Arc::new(FakeLlmClient::new().with_default("DECISION: ADD"));
        let c = Consolidator::new(CannedExtractor(vec![]), llm.clone());
        let plan = c.consolidate("raw", &[em("x")]).await.unwrap();
        assert!(plan.actions.is_empty());
        assert_eq!(llm.call_count(), 0);
    }

    #[tokio::test]
    async fn candidates_capped_by_config() {
        // 8 candidates but max_candidates=2 → only 2 reach the prompt,
        // so a NOOP target of 3+ can't be selected (parse bounds-checks
        // against the capped count and falls back to ADD).
        let cands: Vec<ExistingMemory> = (0..8).map(|i| em(&format!("memory {i}"))).collect();
        let llm = Arc::new(FakeLlmClient::new().with_default("DECISION: NOOP 5"));
        let c = Consolidator::new(CannedExtractor(vec!["new fact".into()]), llm).with_config(
            ConsolidateConfig {
                max_candidates: 2,
                intra_batch_dedup: true,
            },
        );
        let plan = c.consolidate("raw", &cands).await.unwrap();
        // NOOP 5 is out of range for the 2 shown candidates → ADD.
        assert_eq!(plan.adds(), 1);
    }
}
