//! Async ingestion worker — Phase 7 made live.
//!
//! Tails the event log for [`Event::ObservationRecorded`] (raw turns
//! appended on the fast write path) and runs the `mnesio-extract`
//! pipeline on each:
//!
//! 1. **Extract** atomic facts from the raw content.
//! 2. **Consolidate** each fact against retriever-fetched candidates →
//!    ADD / UPDATE(contradiction|refinement) / NOOP.
//! 3. **Admit** ADDs through the importance floor (cull noise/dups).
//! 4. **Apply** as events: `MemoryWritten` (ADD), the supersede triple
//!    `MemoryWritten`+`MemoryEvolved`+`MemoryInvalidated` (UPDATE), or
//!    nothing (NOOP / rejected).
//!
//! Off the write path (Hard Rule #5): the raw observation is appended
//! fast; all the LLM work happens here, async. Emitted memories are
//! fanned to the vector + BM25 views immediately (same pattern as the
//! demo writer) so they're searchable — and so later observations in the
//! stream can dedup against them via the retriever.

use mnesio_core::entity::{Memory, Provenance};
use mnesio_core::event::{ChangeSet, Event, LogEntry};
use mnesio_core::traits::MaterializedView;
use mnesio_core::types::{new_id, BiTemporal, MemoryRef, Scope};
use mnesio_core::{EventLog, Id, LlmClient, MnesioError, Query, Retriever};
use mnesio_extract::{
    AdmissionPolicy, ConsolidationAction, Consolidator, ExistingMemory, LlmExtractor, UpdateReason,
};
use mnesio_index::{Bm25View, VectorView};
use mnesio_privacy::{Redactor, RegexlessRedactor};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Counters surfaced at `/api/ingest/metrics` for the dashboard.
#[derive(Debug, Clone, Default)]
pub struct IngestMetrics {
    pub observations: u64,
    pub facts_extracted: u64,
    pub adds_committed: u64,
    pub adds_rejected: u64,
    pub updates: u64,
    pub contradictions: u64,
    pub refinements: u64,
    pub noops: u64,
    /// PII spans redacted from observation text before extraction (P1#8).
    pub pii_redacted: u64,
}

/// How many candidate memories to pull from the retriever per fact.
const CANDIDATE_K: usize = 6;

pub struct IngestionWorker {
    log: Arc<dyn EventLog>,
    retriever: Arc<dyn Retriever>,
    vector: Arc<VectorView>,
    bm25: Arc<Bm25View>,
    consolidator: Consolidator<LlmExtractor>,
    admission: AdmissionPolicy,
    /// PII redactor applied to raw observation text before extraction, so
    /// identifiers never reach the extracted memories or the log's derived
    /// state (P1#8 — minimisation).
    redactor: RegexlessRedactor,
    /// id → (content, scope) cache, for resolving retriever hits to
    /// candidate content and reading an UPDATE target's scope.
    cache: RwLock<HashMap<MemoryRef, (String, Scope)>>,
    metrics: RwLock<IngestMetrics>,
    poll_interval: Duration,
}

impl IngestionWorker {
    pub fn new(
        log: Arc<dyn EventLog>,
        retriever: Arc<dyn Retriever>,
        vector: Arc<VectorView>,
        bm25: Arc<Bm25View>,
        llm: Arc<dyn LlmClient>,
    ) -> Self {
        let consolidator = Consolidator::new(LlmExtractor::new(llm.clone()), llm);
        Self {
            log,
            retriever,
            vector,
            bm25,
            consolidator,
            admission: AdmissionPolicy::default(),
            redactor: RegexlessRedactor::new(),
            cache: RwLock::new(HashMap::new()),
            metrics: RwLock::new(IngestMetrics::default()),
            poll_interval: Duration::from_millis(300),
        }
    }

    pub async fn metrics(&self) -> IngestMetrics {
        self.metrics.read().await.clone()
    }

    /// Keep the content/scope cache current from any log entry.
    async fn absorb(&self, entry: &LogEntry) {
        match &entry.event {
            Event::MemoryWritten(m) => {
                self.cache
                    .write()
                    .await
                    .insert(MemoryRef(m.id), (m.content.clone(), m.scope.clone()));
            }
            Event::MemoryInvalidated { id, .. } => {
                self.cache.write().await.remove(id);
            }
            _ => {}
        }
    }

    /// Process one log entry. Public so tests can drive it directly.
    /// No-ops for everything except `ObservationRecorded`.
    pub async fn process(&self, entry: &LogEntry) -> Result<(), MnesioError> {
        self.absorb(entry).await;
        let (scope, raw_content, actor) = match &entry.event {
            Event::ObservationRecorded {
                scope,
                content,
                actor,
            } => (scope.clone(), content.clone(), actor.clone()),
            _ => return Ok(()),
        };

        // P1#8 — redact PII from the raw turn *before* extraction, so no
        // identifier reaches the extracted facts, the memories, or any
        // derived view. The raw ObservationRecorded already landed on the
        // fast write path; future work can seal that with the keyring.
        let report = self.redactor.redact(&raw_content);
        if report.changed() {
            self.metrics.write().await.pii_redacted += report.total() as u64;
            tracing::debug!(
                spans = report.total(),
                "ingestion: redacted PII from observation"
            );
        }
        let content = report.redacted;

        // Fetch candidate memories via the retriever, resolve to content.
        let candidates = self.fetch_candidates(&content, &scope).await;

        let plan = self.consolidator.consolidate(&content, &candidates).await?;
        {
            let mut m = self.metrics.write().await;
            m.observations += 1;
            m.facts_extracted += plan.actions.len() as u64;
        }

        for action in plan.actions {
            match action {
                ConsolidationAction::Add { id, content } => {
                    self.apply_add(id, content, &scope, actor.as_deref(), &candidates)
                        .await?;
                }
                ConsolidationAction::Update {
                    target,
                    content,
                    reason,
                } => {
                    self.apply_update(target, content, &scope, actor.as_deref(), reason)
                        .await?;
                }
                ConsolidationAction::Noop { .. } => {
                    self.metrics.write().await.noops += 1;
                }
            }
        }
        Ok(())
    }

    async fn fetch_candidates(&self, content: &str, scope: &Scope) -> Vec<ExistingMemory> {
        let query = Query {
            text: content.to_string(),
            scope: scope.clone(),
            k: CANDIDATE_K,
            time_filter: None,
        };
        let hits = match self.retriever.search(&query).await {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!(error = %e, "ingestion: candidate search failed");
                return Vec::new();
            }
        };
        let cache = self.cache.read().await;
        hits.into_iter()
            .filter_map(|h| {
                cache
                    .get(&h.memory)
                    .map(|(c, _)| ExistingMemory::new(h.memory, c.clone()))
            })
            .collect()
    }

    /// Admit (or reject) and write an ADD.
    async fn apply_add(
        &self,
        id: MemoryRef,
        content: String,
        scope: &Scope,
        actor: Option<&str>,
        candidates: &[ExistingMemory],
    ) -> Result<(), MnesioError> {
        let cand_refs: Vec<&str> = candidates.iter().map(|c| c.content.as_str()).collect();
        let trust = 0.8;
        let importance = mnesio_extract::heuristic_importance(&content, &cand_refs, trust, 0.6);
        if !self.admission.admit(&importance) {
            self.metrics.write().await.adds_rejected += 1;
            tracing::debug!(%content, "ingestion: ADD rejected by admission floor");
            return Ok(());
        }
        let mem = Memory {
            id: id.0,
            scope: scope.clone(),
            content,
            keywords: vec![],
            tags: vec![],
            context: String::new(),
            embedding: None,
            links: vec![],
            parent: None,
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: Provenance {
                source: actor.unwrap_or("ingestion").to_string(),
                trust,
            },
            source: None,
            position: None,
        };
        self.publish(Event::MemoryWritten(mem)).await?;
        self.metrics.write().await.adds_committed += 1;
        Ok(())
    }

    /// Write the supersede triple for an UPDATE.
    async fn apply_update(
        &self,
        target: MemoryRef,
        content: String,
        obs_scope: &Scope,
        actor: Option<&str>,
        reason: UpdateReason,
    ) -> Result<(), MnesioError> {
        // Resolve the target's scope (fall back to the observation scope).
        let scope = self
            .cache
            .read()
            .await
            .get(&target)
            .map(|(_, s)| s.clone())
            .unwrap_or_else(|| obs_scope.clone());

        let new_mem = Memory {
            id: new_id(),
            scope,
            content,
            keywords: vec![],
            tags: vec![],
            context: String::new(),
            embedding: None,
            links: vec![],
            parent: Some(target),
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: Provenance {
                source: actor.unwrap_or("ingestion").to_string(),
                trust: 0.8,
            },
            source: None,
            position: None,
        };
        let new_ref = MemoryRef(new_mem.id);
        let reason_str = match reason {
            UpdateReason::Contradiction => "superseded by ingestion (contradiction)",
            UpdateReason::Refinement => "superseded by ingestion (refinement)",
        };

        self.publish(Event::MemoryWritten(new_mem)).await?;
        self.publish(Event::MemoryEvolved {
            from: target,
            to: new_ref,
            diff: ChangeSet {
                keywords_added: vec![],
                keywords_removed: vec![],
                tags_added: vec![],
                tags_removed: vec![],
                context_rewritten: true,
            },
        })
        .await?;
        self.publish(Event::MemoryInvalidated {
            id: target,
            reason: reason_str.to_string(),
        })
        .await?;

        let mut m = self.metrics.write().await;
        m.updates += 1;
        match reason {
            UpdateReason::Contradiction => m.contradictions += 1,
            UpdateReason::Refinement => m.refinements += 1,
        }
        Ok(())
    }

    /// Append an event and fan it to the cache + retrieval views so the
    /// memory is immediately searchable (and a candidate for later
    /// observations in the same stream).
    async fn publish(&self, event: Event) -> Result<(), MnesioError> {
        let id = self.log.append(event.clone()).await?;
        let entry = LogEntry { id, event };
        self.absorb(&entry).await;
        // Best-effort view fan-out; a view error shouldn't abort ingestion.
        if let Err(e) = self.vector.apply(&entry).await {
            tracing::warn!(error = %e, "ingestion: vector apply failed");
        }
        if let Err(e) = self.bm25.apply(&entry).await {
            tracing::warn!(error = %e, "ingestion: bm25 apply failed");
        }
        Ok(())
    }
}

/// Spawn the worker; returns the handle so the server can read metrics.
pub fn spawn(
    log: Arc<dyn EventLog>,
    retriever: Arc<dyn Retriever>,
    vector: Arc<VectorView>,
    bm25: Arc<Bm25View>,
    llm: Arc<dyn LlmClient>,
) -> Arc<IngestionWorker> {
    let worker = Arc::new(IngestionWorker::new(log, retriever, vector, bm25, llm));
    let w = worker.clone();
    tokio::spawn(async move { run_loop(w).await });
    worker
}

async fn run_loop(worker: Arc<IngestionWorker>) {
    // Rebuild the content cache from history first so candidate
    // resolution + UPDATE-target scope lookups work from boot.
    let mut last_seen: Option<Id> = None;
    if let Ok(entries) = worker.log.read_from(None).await {
        for entry in &entries {
            worker.absorb(entry).await;
            last_seen = Some(entry.id);
        }
    }
    tracing::info!("ingestion worker: started");
    loop {
        match worker.log.read_from(last_seen).await {
            Ok(entries) => {
                for entry in entries {
                    last_seen = Some(entry.id);
                    if let Err(e) = worker.process(&entry).await {
                        tracing::warn!(error = %e, "ingestion worker: process error");
                    }
                }
            }
            Err(e) => tracing::error!(error = %e, "ingestion worker: log read failed"),
        }
        tokio::time::sleep(worker.poll_interval).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use mnesio_core::{Hit, Query};
    use mnesio_llm::FakeLlmClient;
    use mnesio_store::FjallEventLog;

    /// Retriever that returns whatever refs it's told to, so the test
    /// controls the candidate set deterministically.
    struct FakeRetriever {
        hits: std::sync::Mutex<Vec<MemoryRef>>,
    }
    #[async_trait]
    impl Retriever for FakeRetriever {
        async fn search(&self, _q: &Query) -> Result<Vec<Hit>, MnesioError> {
            Ok(self
                .hits
                .lock()
                .unwrap()
                .iter()
                .map(|m| Hit {
                    memory: *m,
                    score: 1.0,
                    breakdown: vec![],
                })
                .collect())
        }
    }

    fn temp_dir() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mnesio-ingest-test-{}", new_id()))
    }

    async fn observe(log: &Arc<dyn EventLog>, content: &str) -> LogEntry {
        let event = Event::ObservationRecorded {
            scope: Scope::global("t"),
            content: content.to_string(),
            actor: None,
        };
        let id = log.append(event.clone()).await.unwrap();
        LogEntry { id, event }
    }

    #[tokio::test]
    async fn add_then_update_then_noop_end_to_end() {
        let dir = temp_dir();
        let log: Arc<dyn EventLog> = FjallEventLog::open(&dir).unwrap();
        let vector = Arc::new(VectorView::new(4, "test"));
        let bm25 = Arc::new(Bm25View::new().unwrap());

        // Deterministic LLM: extraction returns one FACT echoing the raw
        // content; the decision is driven by content keywords.
        let llm = Arc::new(
            FakeLlmClient::new()
                // --- extraction prompts ---
                .with_prefix_match(
                    "Extract the durable, atomic facts worth remembering from the text below. Each fact must be a single self-contained statement understandable on its own (resolve pronouns, include the subject). Ignore pleasantries, questions, and transient chatter.\n\nText:\nAcme Q3 revenue grew 18%",
                    "FACT: Acme Q3 revenue grew 18%",
                )
                .with_prefix_match(
                    "Extract the durable, atomic facts worth remembering from the text below. Each fact must be a single self-contained statement understandable on its own (resolve pronouns, include the subject). Ignore pleasantries, questions, and transient chatter.\n\nText:\nActually Acme Q3 revenue grew 16% not 18%",
                    "FACT: Actually Acme Q3 revenue grew 16% not 18%",
                )
                .with_prefix_match(
                    "Extract the durable, atomic facts worth remembering from the text below. Each fact must be a single self-contained statement understandable on its own (resolve pronouns, include the subject). Ignore pleasantries, questions, and transient chatter.\n\nText:\nAcme Q3 revenue grew 16%",
                    "FACT: Acme Q3 revenue grew 16%",
                )
                // --- decision prompts ---
                .with_prefix_match(
                    "A new candidate fact has been extracted:\nActually Acme Q3 revenue grew 16% not 18%",
                    "DECISION: UPDATE 1 CONTRADICTION",
                )
                .with_prefix_match(
                    "A new candidate fact has been extracted:\nAcme Q3 revenue grew 16%",
                    "DECISION: NOOP 1",
                )
                .with_default("DECISION: ADD"),
        );

        let retr = Arc::new(FakeRetriever {
            hits: std::sync::Mutex::new(vec![]),
        });
        let worker =
            IngestionWorker::new(log.clone(), retr.clone(), vector.clone(), bm25.clone(), llm);

        // 1) ADD — fresh fact, no candidates.
        let e1 = observe(&log, "Acme Q3 revenue grew 18%").await;
        worker.process(&e1).await.unwrap();
        let m = worker.metrics().await;
        assert_eq!(m.adds_committed, 1, "first fact should ADD");

        // The added memory is now in the cache; point the retriever at it
        // so the next observation sees it as candidate 1.
        let added: MemoryRef = *worker.cache.read().await.keys().next().unwrap();
        *retr.hits.lock().unwrap() = vec![added];

        // 2) UPDATE (contradiction) — supersedes the original.
        let e2 = observe(&log, "Actually Acme Q3 revenue grew 16% not 18%").await;
        worker.process(&e2).await.unwrap();
        let m = worker.metrics().await;
        assert_eq!(m.updates, 1, "contradiction should UPDATE");
        assert_eq!(m.contradictions, 1);

        // The original was invalidated (removed from cache); the new
        // version remains. Point the retriever at the surviving version.
        let survivor: MemoryRef = *worker.cache.read().await.keys().next().unwrap();
        assert_ne!(survivor, added, "the superseding version is a new id");
        *retr.hits.lock().unwrap() = vec![survivor];

        // 3) NOOP — duplicate of the surviving fact.
        let e3 = observe(&log, "Acme Q3 revenue grew 16%").await;
        worker.process(&e3).await.unwrap();
        let m = worker.metrics().await;
        assert_eq!(m.noops, 1, "duplicate should NOOP");

        // History is intact: the log holds the original ADD, the
        // supersede triple, and the raw observations — append-only.
        let all = log.read_from(None).await.unwrap();
        let invalidations = all
            .iter()
            .filter(|e| matches!(e.event, Event::MemoryInvalidated { .. }))
            .count();
        assert_eq!(invalidations, 1, "exactly one supersede invalidation");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn admission_rejects_low_signal_add() {
        let dir = temp_dir();
        let log: Arc<dyn EventLog> = FjallEventLog::open(&dir).unwrap();
        let vector = Arc::new(VectorView::new(4, "test"));
        let bm25 = Arc::new(Bm25View::new().unwrap());
        // Extraction yields a trivial 1-token "fact"; decision says ADD;
        // admission floor should reject it.
        let llm = Arc::new(
            FakeLlmClient::new()
                .with_prefix_match("Extract the durable", "FACT: ok")
                .with_default("DECISION: ADD"),
        );
        let retr = Arc::new(FakeRetriever {
            hits: std::sync::Mutex::new(vec![]),
        });
        let worker = IngestionWorker::new(log.clone(), retr, vector, bm25, llm);
        let e = observe(&log, "ok thanks!").await;
        worker.process(&e).await.unwrap();
        let m = worker.metrics().await;
        assert_eq!(m.adds_committed, 0);
        assert_eq!(m.adds_rejected, 1, "trivial fact rejected by admission");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn pii_is_redacted_before_extraction() {
        // The observation carries an email; the redactor must mask it
        // before the extractor sees it, so the email never reaches the
        // extracted fact, the stored memory, or the BM25 index. We capture
        // what the extractor was asked to summarise via the LLM call log.
        let dir = temp_dir();
        let log: Arc<dyn EventLog> = FjallEventLog::open(&dir).unwrap();
        let vector = Arc::new(VectorView::new(4, "test"));
        let bm25 = Arc::new(Bm25View::new().unwrap());
        let llm = Arc::new(
            FakeLlmClient::new()
                .with_prefix_match("Extract the durable", "FACT: a contact was shared")
                .with_default("DECISION: ADD"),
        );
        let retr = Arc::new(FakeRetriever {
            hits: std::sync::Mutex::new(vec![]),
        });
        let worker = IngestionWorker::new(log.clone(), retr, vector, bm25, llm.clone());
        let e = observe(&log, "reach me at alice@example.com anytime").await;
        worker.process(&e).await.unwrap();

        // Metric incremented.
        let m = worker.metrics().await;
        assert_eq!(m.pii_redacted, 1, "one email span should be redacted");

        // The extraction prompt the LLM received must contain the
        // placeholder, never the raw email.
        let saw_email = llm
            .call_log()
            .iter()
            .any(|p| p.contains("alice@example.com"));
        let saw_placeholder = llm.call_log().iter().any(|p| p.contains("[EMAIL]"));
        assert!(!saw_email, "raw email must not reach the extractor");
        assert!(
            saw_placeholder,
            "redacted placeholder should reach the extractor"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
