//! End-to-end tests for [`crate::worker::EvolutionWorker`] using
//! `FakeLlmClient` + an in-memory event log + a fake retriever.
//!
//! The worker is deliberately driven via `process(entry)` rather than
//! the tail loop — tests stay deterministic and run in microseconds.

use crate::{worker::EvolutionWorker, EvolveConfig};
use async_trait::async_trait;
use mnesio_core::entity::{Memory, Provenance};
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::types::{new_id, BiTemporal, MemoryRef};
use mnesio_core::{EventLog, Hit, Id, MnesioError, Query, Retriever, Scope};
use mnesio_llm::FakeLlmClient;
use std::sync::{Arc, Mutex};

// --- test infrastructure ---

/// In-memory append-only log. Mirrors the `FjallEventLog` surface but
/// stays in the process, so worker tests don't touch disk.
#[derive(Default)]
struct MemoryLog {
    entries: Mutex<Vec<LogEntry>>,
}

impl MemoryLog {
    fn new() -> Self {
        Self::default()
    }
    fn snapshot(&self) -> Vec<LogEntry> {
        self.entries.lock().unwrap().clone()
    }
    fn count_events<F: Fn(&Event) -> bool>(&self, pred: F) -> usize {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .filter(|e| pred(&e.event))
            .count()
    }
}

#[async_trait]
impl EventLog for MemoryLog {
    async fn append(&self, event: Event) -> Result<Id, MnesioError> {
        let id = new_id();
        self.entries.lock().unwrap().push(LogEntry { id, event });
        Ok(id)
    }
    async fn read_from(&self, after: Option<Id>) -> Result<Vec<LogEntry>, MnesioError> {
        let entries = self.entries.lock().unwrap();
        let start = match after {
            None => 0,
            Some(a) => entries
                .iter()
                .position(|e| e.id == a)
                .map(|p| p + 1)
                .unwrap_or(0),
        };
        Ok(entries[start..].to_vec())
    }
}

/// Fake retriever that returns a pre-configured list of `Hit`s
/// regardless of the query. Lets tests pin which memories the worker
/// will consider as neighbors.
struct FakeRetriever {
    neighbors: Mutex<Vec<MemoryRef>>,
}

impl FakeRetriever {
    fn new(neighbors: Vec<MemoryRef>) -> Self {
        Self {
            neighbors: Mutex::new(neighbors),
        }
    }
}

#[async_trait]
impl Retriever for FakeRetriever {
    async fn search(&self, query: &Query) -> Result<Vec<Hit>, MnesioError> {
        let n = self.neighbors.lock().unwrap();
        Ok(n.iter()
            .take(query.k)
            .map(|m| Hit {
                memory: *m,
                score: 1.0,
                breakdown: vec![],
            })
            .collect())
    }
}

// --- helpers ---

fn scope() -> Scope {
    Scope::global("test")
}

fn mem(content: &str) -> Memory {
    Memory {
        id: new_id(),
        scope: scope(),
        content: content.into(),
        keywords: Vec::new(),
        tags: Vec::new(),
        context: String::new(),
        embedding: None,
        links: Vec::new(),
        parent: None,
        evolution_count: 0,
        time: BiTemporal::now(),
        provenance: Provenance {
            source: "test".into(),
            trust: 1.0,
        },
        source: None,
        position: None,
    }
}

fn mem_with(content: &str, tags: &[&str], keywords: &[&str]) -> Memory {
    let mut m = mem(content);
    m.tags = tags.iter().map(|s| (*s).into()).collect();
    m.keywords = keywords.iter().map(|s| (*s).into()).collect();
    m
}

/// Standard note-construction response used by tests that don't care
/// about the specific extraction, just that the worker emitted the
/// event.
fn canned_note_response() -> &'static str {
    "KEYWORDS: alpha, beta, gamma\nTAGS: topic-a, topic-b\nCONTEXT: A test memory."
}

// --- tests ---

#[tokio::test]
async fn note_construction_emits_enriched_event() {
    let log = Arc::new(MemoryLog::new());
    let llm = Arc::new(
        FakeLlmClient::new()
            .with_prefix_match("Read the following memory", canned_note_response())
            .with_default("NONE"),
    );
    let retriever = Arc::new(FakeRetriever::new(vec![]));
    let worker = EvolutionWorker::new(log.clone(), retriever, llm, EvolveConfig::default());

    let m = mem("Revenue grew 18% YoY");
    let id = log.append(Event::MemoryWritten(m.clone())).await.unwrap();
    worker
        .process(&LogEntry {
            id,
            event: Event::MemoryWritten(m),
        })
        .await
        .unwrap();

    let n = log.count_events(|e| matches!(e, Event::MemoryNoteEnriched { .. }));
    assert_eq!(n, 1, "exactly one MemoryNoteEnriched should be emitted");
}

#[tokio::test]
async fn note_construction_no_op_when_unchanged() {
    let log = Arc::new(MemoryLog::new());
    // The note response matches what the memory already has — worker
    // should detect "no actual change" and skip the event emission.
    let llm = Arc::new(
        FakeLlmClient::new()
            .with_prefix_match(
                "Read the following memory",
                "KEYWORDS: same\nTAGS: t\nCONTEXT: ctx",
            )
            .with_default("NONE"),
    );
    let retriever = Arc::new(FakeRetriever::new(vec![]));
    let worker = EvolutionWorker::new(log.clone(), retriever, llm, EvolveConfig::default());

    let mut m = mem("anything");
    m.keywords = vec!["same".into()];
    m.tags = vec!["t".into()];
    m.context = "ctx".into();
    let id = log.append(Event::MemoryWritten(m.clone())).await.unwrap();
    worker
        .process(&LogEntry {
            id,
            event: Event::MemoryWritten(m),
        })
        .await
        .unwrap();

    let n = log.count_events(|e| matches!(e, Event::MemoryNoteEnriched { .. }));
    assert_eq!(n, 0, "unchanged enrichment must not emit an event");
}

#[tokio::test]
async fn memory_with_parent_is_skipped() {
    // Memories that are themselves evolution results must not trigger
    // a fresh evolution pass — that's the loop-prevention rule.
    let log = Arc::new(MemoryLog::new());
    let llm = Arc::new(FakeLlmClient::new().with_default("KEYWORDS: a\nTAGS: t\nCONTEXT: c"));
    let retriever = Arc::new(FakeRetriever::new(vec![]));
    let worker = EvolutionWorker::new(log.clone(), retriever, llm, EvolveConfig::default());

    let mut m = mem("evolved memory");
    m.parent = Some(MemoryRef(new_id()));
    let id = log.append(Event::MemoryWritten(m.clone())).await.unwrap();
    worker
        .process(&LogEntry {
            id,
            event: Event::MemoryWritten(m),
        })
        .await
        .unwrap();

    // No follow-up events should appear.
    let n = log.count_events(|e| matches!(e, Event::MemoryNoteEnriched { .. }));
    assert_eq!(n, 0, "memory with parent must not be processed");
}

#[tokio::test]
async fn self_excluded_from_neighbor_list() {
    let log = Arc::new(MemoryLog::new());
    // The retriever returns the freshly-written memory as its own
    // top neighbor (this happens for real when the memory is already
    // in the index). The worker must filter it out, so no
    // MemoryLinksUpdated is emitted (no other neighbors exist).
    let m = mem("self-loop test");
    let m_ref = MemoryRef(m.id);
    let retriever = Arc::new(FakeRetriever::new(vec![m_ref]));
    let llm = Arc::new(
        FakeLlmClient::new()
            .with_prefix_match("Read the following memory", canned_note_response())
            .with_default("NONE"),
    );
    let worker = EvolutionWorker::new(log.clone(), retriever, llm, EvolveConfig::default());

    let id = log.append(Event::MemoryWritten(m.clone())).await.unwrap();
    worker
        .process(&LogEntry {
            id,
            event: Event::MemoryWritten(m),
        })
        .await
        .unwrap();

    let n = log.count_events(|e| matches!(e, Event::MemoryLinksUpdated { .. }));
    assert_eq!(n, 0, "self-as-neighbor must be filtered, leaving no links");
}

#[tokio::test]
async fn link_generation_emits_selected_neighbors() {
    let log = Arc::new(MemoryLog::new());
    // Seed two pre-existing neighbors via direct appends + absorb.
    let n1 = mem_with("neighbor one", &["t1"], &["k1"]);
    let n2 = mem_with("neighbor two", &["t2"], &["k2"]);
    let n1_ref = MemoryRef(n1.id);
    let n2_ref = MemoryRef(n2.id);
    log.append(Event::MemoryWritten(n1.clone())).await.unwrap();
    log.append(Event::MemoryWritten(n2.clone())).await.unwrap();

    let retriever = Arc::new(FakeRetriever::new(vec![n1_ref, n2_ref]));
    let llm = Arc::new(
        FakeLlmClient::new()
            .with_prefix_match("Read the following memory", canned_note_response())
            // The link prompt: select candidate 1 (n1). Worker should
            // emit a MemoryLinksUpdated with [n1_ref].
            .with_prefix_match("A new memory was just recorded", "1")
            // Evolution proposal — short-circuit so we test only the
            // link path here.
            .with_default("NONE"),
    );
    let worker = EvolutionWorker::new(log.clone(), retriever, llm, EvolveConfig::default());
    worker.replay().await.unwrap();

    let m = mem("the new memory");
    let id = log.append(Event::MemoryWritten(m.clone())).await.unwrap();
    worker
        .process(&LogEntry {
            id,
            event: Event::MemoryWritten(m),
        })
        .await
        .unwrap();

    let updates: Vec<_> = log
        .snapshot()
        .into_iter()
        .filter_map(|e| match e.event {
            Event::MemoryLinksUpdated { id: _, links } => Some(links),
            _ => None,
        })
        .collect();
    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0], vec![n1_ref]);
}

#[tokio::test]
async fn bounded_evolution_emits_supersede_triple() {
    let log = Arc::new(MemoryLog::new());
    let neighbor = mem_with("acme earnings", &["earnings"], &["acme"]);
    let neighbor_ref = MemoryRef(neighbor.id);
    log.append(Event::MemoryWritten(neighbor.clone()))
        .await
        .unwrap();

    let retriever = Arc::new(FakeRetriever::new(vec![neighbor_ref]));
    let llm = Arc::new(
        FakeLlmClient::new()
            .with_prefix_match(
                "Read the following memory",
                "KEYWORDS: revenue\nTAGS: revenue\nCONTEXT: new",
            )
            .with_prefix_match("A new memory was just recorded", "1")
            .with_prefix_match(
                "An existing memory and its current annotations",
                "TAGS_ADD: correction\nKEYWORDS_ADD: restated",
            )
            .with_default("NONE"),
    );
    let worker = EvolutionWorker::new(log.clone(), retriever, llm, EvolveConfig::default());
    worker.replay().await.unwrap();

    let m = mem("acme revenue restated to 16%");
    let id = log.append(Event::MemoryWritten(m.clone())).await.unwrap();
    worker
        .process(&LogEntry {
            id,
            event: Event::MemoryWritten(m),
        })
        .await
        .unwrap();

    let events = log.snapshot();
    let written_count = events
        .iter()
        .filter(|e| matches!(e.event, Event::MemoryWritten(_)))
        .count();
    // 1 original neighbor + 1 new memory + 1 evolved version = 3
    assert_eq!(
        written_count, 3,
        "expected old neighbor + new + evolved-new"
    );

    let evolved_count = events
        .iter()
        .filter(|e| matches!(e.event, Event::MemoryEvolved { .. }))
        .count();
    assert_eq!(evolved_count, 1);

    let invalidated_count = events
        .iter()
        .filter(|e| matches!(&e.event, Event::MemoryInvalidated { id, .. } if *id == neighbor_ref))
        .count();
    assert_eq!(invalidated_count, 1, "old neighbor must be invalidated");

    // The new version must carry parent + incremented evolution_count.
    let evolved_written: Vec<&Memory> = events
        .iter()
        .filter_map(|e| match &e.event {
            Event::MemoryWritten(m) if m.parent == Some(neighbor_ref) => Some(m),
            _ => None,
        })
        .collect();
    assert_eq!(evolved_written.len(), 1);
    assert_eq!(evolved_written[0].evolution_count, 1);
    assert!(evolved_written[0].tags.iter().any(|t| t == "correction"));
    assert!(evolved_written[0].keywords.iter().any(|k| k == "restated"));
}

#[tokio::test]
async fn lifetime_cap_blocks_evolution_at_chain_depth_limit() {
    // The cap binds the chain depth (X → X1 → X2 → …), tracked via
    // `Memory.evolution_count`. Seed a neighbor already at the cap;
    // the worker must refuse to evolve it further.
    let log = Arc::new(MemoryLog::new());
    let mut neighbor = mem_with("the neighbor", &["t"], &["k"]);
    neighbor.evolution_count = 2; // already at the configured cap
    let neighbor_ref = MemoryRef(neighbor.id);
    log.append(Event::MemoryWritten(neighbor.clone()))
        .await
        .unwrap();

    let retriever = Arc::new(FakeRetriever::new(vec![neighbor_ref]));
    let llm = Arc::new(
        FakeLlmClient::new()
            .with_prefix_match("Read the following memory", canned_note_response())
            .with_prefix_match("A new memory was just recorded", "1")
            .with_prefix_match(
                "An existing memory and its current annotations",
                "TAGS_ADD: x\nKEYWORDS_ADD: y",
            )
            .with_default("NONE"),
    );
    let cfg = EvolveConfig {
        max_lifetime_evolutions: 2,
        cooldown_secs: 0,
        ..EvolveConfig::default()
    };
    let worker = EvolutionWorker::new(log.clone(), retriever, llm, cfg);
    worker.replay().await.unwrap();

    let m = mem("trigger");
    let id = log.append(Event::MemoryWritten(m.clone())).await.unwrap();
    worker
        .process(&LogEntry {
            id,
            event: Event::MemoryWritten(m),
        })
        .await
        .unwrap();

    let evolved_count = log.count_events(|e| matches!(e, Event::MemoryEvolved { .. }));
    assert_eq!(
        evolved_count, 0,
        "neighbor whose chain depth equals the cap must not be evolved further"
    );
}

#[tokio::test]
async fn evolution_chain_increments_count() {
    // Sanity check the lifetime-cap semantics: a fresh neighbor gets
    // evolved once and the new version's `evolution_count` is 1.
    let log = Arc::new(MemoryLog::new());
    let neighbor = mem_with("neighbor", &["t"], &["k"]);
    let neighbor_ref = MemoryRef(neighbor.id);
    log.append(Event::MemoryWritten(neighbor.clone()))
        .await
        .unwrap();

    let retriever = Arc::new(FakeRetriever::new(vec![neighbor_ref]));
    let llm = Arc::new(
        FakeLlmClient::new()
            .with_prefix_match("Read the following memory", canned_note_response())
            .with_prefix_match("A new memory was just recorded", "1")
            .with_prefix_match(
                "An existing memory and its current annotations",
                "TAGS_ADD: x\nKEYWORDS_ADD: y",
            )
            .with_default("NONE"),
    );
    let worker = EvolutionWorker::new(log.clone(), retriever, llm, EvolveConfig::default());
    worker.replay().await.unwrap();

    let m = mem("trigger");
    let id = log.append(Event::MemoryWritten(m.clone())).await.unwrap();
    worker
        .process(&LogEntry {
            id,
            event: Event::MemoryWritten(m),
        })
        .await
        .unwrap();

    let new_versions: Vec<Memory> = log
        .snapshot()
        .into_iter()
        .filter_map(|e| match e.event {
            Event::MemoryWritten(m) if m.parent == Some(neighbor_ref) => Some(m),
            _ => None,
        })
        .collect();
    assert_eq!(new_versions.len(), 1);
    assert_eq!(new_versions[0].evolution_count, 1);
}

#[tokio::test]
async fn cooldown_blocks_immediate_re_evolution() {
    let log = Arc::new(MemoryLog::new());
    let neighbor = mem_with("the neighbor", &["t"], &["k"]);
    let neighbor_ref = MemoryRef(neighbor.id);
    log.append(Event::MemoryWritten(neighbor.clone()))
        .await
        .unwrap();

    let retriever = Arc::new(FakeRetriever::new(vec![neighbor_ref]));
    let llm = Arc::new(
        FakeLlmClient::new()
            .with_prefix_match("Read the following memory", canned_note_response())
            .with_prefix_match("A new memory was just recorded", "1")
            .with_prefix_match(
                "An existing memory and its current annotations",
                "TAGS_ADD: x\nKEYWORDS_ADD: y",
            )
            .with_default("NONE"),
    );
    let cfg = EvolveConfig {
        // An hour — no real evolution will land twice.
        cooldown_secs: 3600,
        max_lifetime_evolutions: 10,
        ..EvolveConfig::default()
    };
    let worker = EvolutionWorker::new(log.clone(), retriever, llm, cfg);
    worker.replay().await.unwrap();

    // Three back-to-back triggers should yield exactly one evolution
    // (the first), because the cooldown blocks the next two.
    for i in 0..3 {
        let m = mem(&format!("trigger {i}"));
        let id = log.append(Event::MemoryWritten(m.clone())).await.unwrap();
        worker
            .process(&LogEntry {
                id,
                event: Event::MemoryWritten(m),
            })
            .await
            .unwrap();
    }

    let evolved_count = log.count_events(|e| matches!(e, Event::MemoryEvolved { .. }));
    assert_eq!(evolved_count, 1, "cooldown must throttle repeat evolutions");
}

#[tokio::test]
async fn min_change_threshold_drops_trivial_proposals() {
    let log = Arc::new(MemoryLog::new());
    let neighbor = mem_with("neighbor", &["t"], &["k"]);
    let neighbor_ref = MemoryRef(neighbor.id);
    log.append(Event::MemoryWritten(neighbor.clone()))
        .await
        .unwrap();

    let retriever = Arc::new(FakeRetriever::new(vec![neighbor_ref]));
    let llm = Arc::new(
        FakeLlmClient::new()
            .with_prefix_match("Read the following memory", canned_note_response())
            .with_prefix_match("A new memory was just recorded", "1")
            // Only one addition — below the threshold of 2.
            .with_prefix_match(
                "An existing memory and its current annotations",
                "TAGS_ADD: x\nKEYWORDS_ADD:",
            )
            .with_default("NONE"),
    );
    let cfg = EvolveConfig {
        min_change_threshold: 2,
        ..EvolveConfig::default()
    };
    let worker = EvolutionWorker::new(log.clone(), retriever, llm, cfg);
    worker.replay().await.unwrap();

    let m = mem("triggering memory");
    let id = log.append(Event::MemoryWritten(m.clone())).await.unwrap();
    worker
        .process(&LogEntry {
            id,
            event: Event::MemoryWritten(m),
        })
        .await
        .unwrap();

    let evolved_count = log.count_events(|e| matches!(e, Event::MemoryEvolved { .. }));
    assert_eq!(
        evolved_count, 0,
        "below-threshold proposal must not commit an evolution"
    );
}

#[tokio::test]
async fn max_evolve_per_write_bounds_cascade() {
    let log = Arc::new(MemoryLog::new());
    // Seed five neighbors. The worker is configured to mutate at most
    // 2 per write, so the fifth+ should not be touched.
    let neighbors: Vec<Memory> = (0..5)
        .map(|i| mem_with(&format!("n{i}"), &["t"], &["k"]))
        .collect();
    let neighbor_refs: Vec<MemoryRef> = neighbors.iter().map(|m| MemoryRef(m.id)).collect();
    for n in &neighbors {
        log.append(Event::MemoryWritten(n.clone())).await.unwrap();
    }

    let retriever = Arc::new(FakeRetriever::new(neighbor_refs.clone()));
    let llm = Arc::new(
        FakeLlmClient::new()
            .with_prefix_match("Read the following memory", canned_note_response())
            // Link prompt picks all 5 candidates.
            .with_prefix_match("A new memory was just recorded", "1, 2, 3, 4, 5")
            .with_prefix_match(
                "An existing memory and its current annotations",
                "TAGS_ADD: added\nKEYWORDS_ADD: added",
            )
            .with_default("NONE"),
    );
    let cfg = EvolveConfig {
        max_evolve_per_write: 2,
        cooldown_secs: 0,
        max_lifetime_evolutions: 10,
        ..EvolveConfig::default()
    };
    let worker = EvolutionWorker::new(log.clone(), retriever, llm, cfg);
    worker.replay().await.unwrap();

    let m = mem("trigger");
    let id = log.append(Event::MemoryWritten(m.clone())).await.unwrap();
    worker
        .process(&LogEntry {
            id,
            event: Event::MemoryWritten(m),
        })
        .await
        .unwrap();

    let evolved_count = log.count_events(|e| matches!(e, Event::MemoryEvolved { .. }));
    assert_eq!(
        evolved_count, 2,
        "cascade must be capped at max_evolve_per_write"
    );
}

#[tokio::test]
async fn replay_reconstructs_evolution_state() {
    // Seed the log with: original memory + a full evolution triple.
    // After replay, the worker's evolution_count for the original
    // memory should be 1.
    let log = Arc::new(MemoryLog::new());
    let original = mem_with("original", &["t"], &["k"]);
    let original_ref = MemoryRef(original.id);
    log.append(Event::MemoryWritten(original.clone()))
        .await
        .unwrap();

    let mut evolved = original.clone();
    evolved.id = new_id();
    evolved.parent = Some(original_ref);
    evolved.evolution_count = 1;
    let evolved_ref = MemoryRef(evolved.id);
    log.append(Event::MemoryWritten(evolved)).await.unwrap();
    log.append(Event::MemoryEvolved {
        from: original_ref,
        to: evolved_ref,
        diff: mnesio_core::event::ChangeSet {
            keywords_added: Vec::new(),
            keywords_removed: Vec::new(),
            tags_added: vec!["x".into()],
            tags_removed: Vec::new(),
            context_rewritten: false,
        },
    })
    .await
    .unwrap();
    log.append(Event::MemoryInvalidated {
        id: original_ref,
        reason: "test".into(),
    })
    .await
    .unwrap();

    let llm = Arc::new(FakeLlmClient::new());
    let retriever = Arc::new(FakeRetriever::new(vec![]));
    let worker = EvolutionWorker::new(log.clone(), retriever, llm, EvolveConfig::default());
    worker.replay().await.unwrap();

    assert_eq!(worker.evolution_count(original_ref).await, 1);
}
