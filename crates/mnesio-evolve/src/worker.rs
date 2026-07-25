//! The bounded A-MEM evolution worker.
//!
//! Tails the event log, runs the three-step pipeline on each newly-
//! written memory (note construction → link generation → bounded
//! evolution), and emits the appropriate events: `MemoryNoteEnriched`
//! / `MemoryLinksUpdated` for the lightweight enrichment steps,
//! `MemoryWritten` + `MemoryEvolved` + `MemoryInvalidated` for the
//! heavy bounded-evolution step that supersedes a neighbor with a new
//! bi-temporal version.
//!
//! All scheduling bounds documented in [`crate::EvolveConfig`] are
//! enforced inside [`EvolutionWorker::process`].

use crate::parse::{parse_evolution, parse_link_selection, parse_note, EvolutionChanges};
use crate::prompts;
use crate::EvolveConfig;
use mnesio_core::entity::{Memory, Provenance};
use mnesio_core::event::{ChangeSet, Event, LogEntry};
use mnesio_core::types::{new_id, BiTemporal, MemoryRef};
use mnesio_core::{EventLog, Id, LlmClient, MnesioError, Query, Retriever};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Per-memory mutable state the worker tracks for bounded scheduling.
#[derive(Debug, Clone, Default)]
struct EvolutionState {
    /// Timestamp (unix ms) of the most recent evolution. Used by the
    /// cooldown check.
    last_evolved_at_ms: u64,
    /// How many times this memory has been the target of a successful
    /// bounded-evolution rewrite. Capped at
    /// [`EvolveConfig::max_lifetime_evolutions`].
    evolution_count: u16,
}

/// The async evolution worker.
///
/// Holds references to the log, retriever, LLM client, and an
/// in-memory join table of all live memories (rebuilt from the log on
/// startup and kept current by tailing). The `state` map tracks
/// per-memory cooldown + lifetime counts.
pub struct EvolutionWorker {
    log: Arc<dyn EventLog>,
    retriever: Arc<dyn Retriever>,
    llm: Arc<dyn LlmClient>,
    config: EvolveConfig,
    /// Live memory cache, indexed by id. Used both for content lookup
    /// during link/evolution prompts and to construct the new bi-
    /// temporal versions on evolution. Rebuilt on every restart by
    /// replaying the log.
    memories: Arc<RwLock<HashMap<MemoryRef, Memory>>>,
    state: Arc<RwLock<HashMap<MemoryRef, EvolutionState>>>,
}

impl EvolutionWorker {
    pub fn new(
        log: Arc<dyn EventLog>,
        retriever: Arc<dyn Retriever>,
        llm: Arc<dyn LlmClient>,
        config: EvolveConfig,
    ) -> Self {
        Self {
            log,
            retriever,
            llm,
            config,
            memories: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Replay the entire log to rebuild the in-memory caches. Idempotent
    /// — events that don't affect cache state are no-ops. Used by both
    /// the boot path and tests.
    pub async fn replay(&self) -> Result<Option<Id>, MnesioError> {
        let entries = self.log.read_from(None).await?;
        let last = entries.last().map(|e| e.id);
        for entry in &entries {
            self.absorb(entry).await;
        }
        Ok(last)
    }

    /// Update internal caches from a single log entry without running
    /// the evolution pipeline. Used by `replay` and by `process` to
    /// keep the cache current as events flow.
    async fn absorb(&self, entry: &LogEntry) {
        match &entry.event {
            Event::MemoryWritten(m) => {
                self.memories
                    .write()
                    .await
                    .insert(MemoryRef(m.id), m.clone());
            }
            Event::MemoryNoteEnriched {
                id,
                keywords,
                tags,
                context,
            } => {
                if let Some(m) = self.memories.write().await.get_mut(id) {
                    m.keywords = keywords.clone();
                    m.tags = tags.clone();
                    m.context = context.clone();
                }
            }
            Event::MemoryLinksUpdated { id, links } => {
                if let Some(m) = self.memories.write().await.get_mut(id) {
                    m.links = links.clone();
                }
            }
            Event::MemoryEvolved { from, .. } => {
                // Record that `from` has been evolved another time; the
                // timestamp encoded in the event id is what the
                // cooldown check uses.
                let mut state = self.state.write().await;
                let s = state.entry(*from).or_default();
                s.evolution_count = s.evolution_count.saturating_add(1);
                s.last_evolved_at_ms = entry.id.timestamp_ms();
            }
            Event::MemoryInvalidated { id, .. } => {
                self.memories.write().await.remove(id);
            }
            _ => {}
        }
    }

    /// Run the three-step pipeline on a single `MemoryWritten` event.
    /// Public so tests can drive it without the tail loop, and so a
    /// future synchronous-mode host could trigger it inline if it
    /// wanted to.
    ///
    /// Quietly no-ops for events that aren't `MemoryWritten`, and for
    /// memories with a `parent` pointer (loop prevention — these are
    /// themselves the result of an earlier evolution).
    pub async fn process(&self, entry: &LogEntry) -> Result<(), MnesioError> {
        // Keep the cache current first — `process` is the only path the
        // tail loop calls, so the cache stays consistent.
        self.absorb(entry).await;

        let memory = match &entry.event {
            Event::MemoryWritten(m) => m.clone(),
            _ => return Ok(()),
        };
        if memory.parent.is_some() {
            tracing::trace!(
                memory = %memory.id,
                "evolve: skipping memory that is itself an evolution result"
            );
            return Ok(());
        }

        // Step 1 — note construction.
        if let Err(e) = self.run_note_construction(&memory).await {
            tracing::warn!(memory = %memory.id, error = %e, "evolve: note construction failed");
        }

        // Step 2 — link generation. Pull semantically similar neighbors
        // first; reused by step 3 to avoid a second retrieval call.
        let neighbors = match self.fetch_neighbors(&memory).await {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!(memory = %memory.id, error = %e, "evolve: neighbor fetch failed");
                Vec::new()
            }
        };
        let selected = match self.run_link_generation(&memory, &neighbors).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(memory = %memory.id, error = %e, "evolve: link generation failed");
                Vec::new()
            }
        };

        // Step 3 — bounded evolution on each selected neighbor.
        let mut cascade = 0usize;
        for nbr in selected {
            if cascade >= self.config.max_evolve_per_write {
                break;
            }
            if !self
                .eligible_for_evolution(nbr, entry.id.timestamp_ms())
                .await
            {
                continue;
            }
            match self.evolve_neighbor(nbr, &memory).await {
                Ok(true) => {
                    cascade += 1;
                }
                Ok(false) => {} // proposal was below threshold
                Err(e) => {
                    tracing::warn!(neighbor = ?nbr, error = %e, "evolve: evolution failed");
                }
            }
        }
        Ok(())
    }

    async fn run_note_construction(&self, memory: &Memory) -> Result<(), MnesioError> {
        let prompt = prompts::note_construction(&memory.content);
        let response = self.llm.complete(&prompt).await?;
        let fields = parse_note(&response);
        // No-op when the LLM didn't follow the format at all.
        if fields.keywords.is_empty() && fields.tags.is_empty() && fields.context.is_empty() {
            return Ok(());
        }
        // No-op when the proposed enrichment equals what we already have.
        if fields.keywords == memory.keywords
            && fields.tags == memory.tags
            && fields.context == memory.context
        {
            return Ok(());
        }
        let event = Event::MemoryNoteEnriched {
            id: MemoryRef(memory.id),
            keywords: fields.keywords,
            tags: fields.tags,
            context: fields.context,
        };
        let id = self.log.append(event.clone()).await?;
        self.absorb(&LogEntry { id, event }).await;
        Ok(())
    }

    async fn fetch_neighbors(&self, memory: &Memory) -> Result<Vec<MemoryRef>, MnesioError> {
        // Pull 2× the cap so the LLM's link-selection step has headroom
        // to discard noisy candidates without cutting into our actual
        // mutation budget.
        let k = self.config.max_evolve_per_write.saturating_mul(2).max(1);
        let query = Query {
            text: memory.content.clone(),
            scope: memory.scope.clone(),
            k,
            time_filter: None,
        };
        let hits = self.retriever.search(&query).await?;
        Ok(hits
            .into_iter()
            .map(|h| h.memory)
            // Self-exclusion — the freshly-indexed memory may surface
            // as its own nearest neighbor; skip it.
            .filter(|m| m.0 != memory.id)
            .collect())
    }

    async fn run_link_generation(
        &self,
        memory: &Memory,
        neighbors: &[MemoryRef],
    ) -> Result<Vec<MemoryRef>, MnesioError> {
        if neighbors.is_empty() {
            return Ok(Vec::new());
        }
        // Build the candidate list with *owned* content + tags +
        // keywords so the cache lock can be released before the LLM
        // call. Skip any neighbor whose content is no longer in the
        // cache (best-effort consistency in the face of concurrent
        // invalidation).
        struct CandidateOwned {
            memory: MemoryRef,
            content: String,
            tags: Vec<String>,
            keywords: Vec<String>,
        }
        let candidates: Vec<CandidateOwned> = {
            let mems = self.memories.read().await;
            neighbors
                .iter()
                .filter_map(|nbr| {
                    mems.get(nbr).map(|m| CandidateOwned {
                        memory: *nbr,
                        content: m.content.clone(),
                        tags: m.tags.clone(),
                        keywords: m.keywords.clone(),
                    })
                })
                .collect()
        };
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let prompt_candidates: Vec<prompts::Candidate<'_>> = candidates
            .iter()
            .map(|c| prompts::Candidate {
                memory: c.memory,
                content: c.content.as_str(),
                tags: c.tags.as_slice(),
                keywords: c.keywords.as_slice(),
            })
            .collect();
        let prompt = prompts::link_generation(&memory.content, &prompt_candidates);
        let response = self.llm.complete(&prompt).await?;
        let indices = parse_link_selection(&response);
        // Map 1-based indices back to MemoryRefs, silently dropping
        // any out-of-range numbers the LLM hallucinated.
        let selected: Vec<MemoryRef> = indices
            .into_iter()
            .filter_map(|i| candidates.get(i.saturating_sub(1)).map(|c| c.memory))
            .collect();
        if selected.is_empty() {
            return Ok(Vec::new());
        }
        let event = Event::MemoryLinksUpdated {
            id: MemoryRef(memory.id),
            links: selected.clone(),
        };
        let id = self.log.append(event.clone()).await?;
        self.absorb(&LogEntry { id, event }).await;
        Ok(selected)
    }

    /// Cooldown + lifetime-cap gate, evaluated before issuing the
    /// evolution prompt so we don't burn an LLM call on a memory we
    /// won't act on anyway.
    ///
    /// The lifetime cap is checked against `memory.evolution_count` —
    /// the per-concept count carried along the lineage chain (X → X1
    /// → X2 …). This bounds the *chain depth*, which is the actual
    /// runaway we want to prevent. Cooldown is checked per-ref via
    /// the state map; a fresh evolution version (X2) starts with no
    /// cooldown of its own, which is fine because its chain depth
    /// will already be near the cap.
    async fn eligible_for_evolution(&self, neighbor_ref: MemoryRef, now_ms: u64) -> bool {
        let chain_depth = {
            let mems = self.memories.read().await;
            match mems.get(&neighbor_ref) {
                Some(m) => m.evolution_count,
                // Memory was invalidated between selection and this
                // check — treat as ineligible rather than racing.
                None => return false,
            }
        };
        if chain_depth >= self.config.max_lifetime_evolutions {
            return false;
        }
        let state = self.state.read().await;
        if let Some(s) = state.get(&neighbor_ref) {
            let cooldown_ms = self.config.cooldown_secs.saturating_mul(1000);
            if now_ms.saturating_sub(s.last_evolved_at_ms) < cooldown_ms {
                return false;
            }
        }
        true
    }

    /// Run the evolution prompt on a single neighbor and, if the LLM
    /// proposes a change above the threshold, emit the three-event
    /// supersede-and-invalidate triple. Returns `Ok(true)` when an
    /// evolution actually committed, `Ok(false)` when the proposal was
    /// trivial (no-op).
    async fn evolve_neighbor(
        &self,
        neighbor_ref: MemoryRef,
        new_memory: &Memory,
    ) -> Result<bool, MnesioError> {
        let (neighbor, prompt) = {
            let mems = self.memories.read().await;
            let n = match mems.get(&neighbor_ref) {
                Some(n) => n.clone(),
                None => return Ok(false), // invalidated between selection and now
            };
            let cand = prompts::Candidate {
                memory: neighbor_ref,
                content: n.content.as_str(),
                tags: n.tags.as_slice(),
                keywords: n.keywords.as_slice(),
            };
            let p = prompts::evolution_proposal(&cand, &new_memory.content);
            (n, p)
        };
        let response = self.llm.complete(&prompt).await?;
        let changes = parse_evolution(&response);
        if changes.total_additions() < self.config.min_change_threshold {
            return Ok(false);
        }
        self.commit_evolution(&neighbor, changes).await?;
        Ok(true)
    }

    /// Apply an `EvolutionChanges` to a neighbor by writing a new bi-
    /// temporal version, emitting a `MemoryEvolved` lineage record, and
    /// invalidating the old version. Three events total, in canonical
    /// order, so log replay reconstructs the same final state.
    async fn commit_evolution(
        &self,
        old: &Memory,
        changes: EvolutionChanges,
    ) -> Result<(), MnesioError> {
        let mut new_tags = old.tags.clone();
        for t in &changes.tags_add {
            if !new_tags.iter().any(|existing| existing == t) {
                new_tags.push(t.clone());
            }
        }
        let mut new_keywords = old.keywords.clone();
        for k in &changes.keywords_add {
            if !new_keywords.iter().any(|existing| existing == k) {
                new_keywords.push(k.clone());
            }
        }

        let old_ref = MemoryRef(old.id);
        let new_memory = Memory {
            id: new_id(),
            scope: old.scope.clone(),
            content: old.content.clone(),
            keywords: new_keywords,
            tags: new_tags,
            context: old.context.clone(),
            embedding: old.embedding.clone(),
            links: old.links.clone(),
            parent: Some(old_ref),
            evolution_count: old.evolution_count.saturating_add(1),
            time: BiTemporal::now(),
            provenance: Provenance {
                source: "evolution-worker".into(),
                trust: old.provenance.trust,
            },
            source: old.source,
            position: old.position,
        };
        let new_ref = MemoryRef(new_memory.id);

        // Emit the three events. The order matters for replay: written
        // first so the new version exists when the evolved+invalidated
        // events refer to it.
        let written = Event::MemoryWritten(new_memory.clone());
        let id1 = self.log.append(written.clone()).await?;
        self.absorb(&LogEntry {
            id: id1,
            event: written,
        })
        .await;

        let evolved = Event::MemoryEvolved {
            from: old_ref,
            to: new_ref,
            diff: ChangeSet {
                keywords_added: changes.keywords_add.clone(),
                keywords_removed: Vec::new(),
                tags_added: changes.tags_add.clone(),
                tags_removed: Vec::new(),
                context_rewritten: false,
            },
        };
        let id2 = self.log.append(evolved.clone()).await?;
        self.absorb(&LogEntry {
            id: id2,
            event: evolved,
        })
        .await;

        let invalidated = Event::MemoryInvalidated {
            id: old_ref,
            reason: "superseded by bounded evolution".into(),
        };
        let id3 = self.log.append(invalidated.clone()).await?;
        self.absorb(&LogEntry {
            id: id3,
            event: invalidated,
        })
        .await;

        Ok(())
    }

    /// Inspect a memory's evolution state for tests / dashboards.
    pub async fn evolution_count(&self, memory: MemoryRef) -> u16 {
        self.state
            .read()
            .await
            .get(&memory)
            .map(|s| s.evolution_count)
            .unwrap_or(0)
    }

    /// Snapshot of the worker's per-memory state. Used by the dashboard
    /// (Slice C) to plot the evolution-count histogram.
    pub async fn snapshot_state(&self) -> Vec<(MemoryRef, u16, u64)> {
        self.state
            .read()
            .await
            .iter()
            .map(|(m, s)| (*m, s.evolution_count, s.last_evolved_at_ms))
            .collect()
    }
}

/// Spawn the worker as a background tokio task. Returns immediately;
/// the worker runs for the life of the runtime.
pub fn spawn(
    log: Arc<dyn EventLog>,
    retriever: Arc<dyn Retriever>,
    llm: Arc<dyn LlmClient>,
    config: EvolveConfig,
) -> Arc<EvolutionWorker> {
    let worker = Arc::new(EvolutionWorker::new(log, retriever, llm, config));
    let w = worker.clone();
    tokio::spawn(async move { run_loop(w).await });
    worker
}

async fn run_loop(worker: Arc<EvolutionWorker>) {
    let mut last_seen = match worker.replay().await {
        Ok(last) => last,
        Err(e) => {
            tracing::error!(error = %e, "evolution worker: initial replay failed");
            None
        }
    };
    tracing::info!("evolution worker: started");
    loop {
        match worker.log.read_from(last_seen).await {
            Ok(entries) => {
                for entry in entries {
                    last_seen = Some(entry.id);
                    if let Err(e) = worker.process(&entry).await {
                        tracing::warn!(error = %e, "evolution worker: process error");
                    }
                }
            }
            Err(e) => tracing::error!(error = %e, "evolution worker: log read failed"),
        }
        tokio::time::sleep(Duration::from_millis(400)).await;
    }
}
