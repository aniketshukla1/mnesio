//! Pure-Rust client that owns the retrieval stack + a `tokio` runtime
//! handle. The pyo3 facade in `lib.rs` is a thin shell that delegates
//! every call to one of these methods.
//!
//! Why an inner-Rust client at all? Because the surface area we want
//! to test is the wiring — log appends, view application, search
//! synthesis — and we don't want to require a Python interpreter in
//! CI to verify it. Splitting like this keeps the integration test
//! matrix small.

use anyhow::{anyhow, Result};
use mnesio_core::entity::{JudgeSource, Memory, Outcome, Provenance};
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::synthesizer::Passage;
use mnesio_core::traits::MaterializedView;
use mnesio_core::types::{
    new_id, ArtifactRef, BiTemporal, EpisodeRef, Id, MemoryRef, Scope, TrajectoryRef,
};
use mnesio_core::{Embedder, EventLog, Query, Retriever};
use mnesio_index::{Bm25View, HybridRetriever, MockEmbedder, SnippetSynthesizer, VectorView};
use mnesio_store::FjallEventLog;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Internal client. Owns:
///
/// - the fjall-backed event log
/// - vector + BM25 views over it
/// - the hybrid retriever + extractive synthesizer
/// - a dedicated tokio runtime so callers from sync contexts (Python,
///   CLI tools, etc.) don't have to manage one themselves
pub struct InnerClient {
    rt: Runtime,
    log: Arc<dyn EventLog>,
    vector: Arc<VectorView>,
    bm25: Arc<Bm25View>,
    embedder: Arc<dyn Embedder>,
    retriever: Arc<HybridRetriever>,
    synthesizer: Arc<dyn mnesio_core::Synthesizer>,
}

/// Args for [`InnerClient::write_memory`]. Mirrors the MCP tool's
/// shape so the two interfaces stay congruent.
pub struct WriteMemoryArgs {
    pub content: String,
    pub tenant: String,
    pub tags: Vec<String>,
}

pub struct SearchArgs {
    pub query: String,
    pub tenant: String,
    pub k: usize,
}

/// Result of a search. Carries enough to render in any UI — synthesized
/// prose, citation ids, individual hits with content + score.
#[derive(Debug)]
pub struct SearchOutcome {
    pub answer: Option<String>,
    pub hits: Vec<SearchHitRecord>,
    pub citations: Vec<String>,
}

#[derive(Debug)]
pub struct SearchHitRecord {
    pub memory_id: String,
    pub content: String,
    pub tags: Vec<String>,
    pub score: f32,
}

pub struct RecordOutcomeArgs {
    pub episode: Option<String>,
    pub artifacts_used: Vec<String>,
    pub success: bool,
    pub scores: HashMap<String, f32>,
    pub error: Option<String>,
}

impl InnerClient {
    /// Open or create a mnesio store at `data_dir`. `embedder` is
    /// `"mock"` (deterministic, no model download) or `"fastembed"`
    /// (real semantic embeddings via bge-small-en-v1.5).
    pub fn open(data_dir: &Path, embedder_choice: &str) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let (log, vector, bm25, embedder) = rt.block_on(async {
            let log = FjallEventLog::open(data_dir)?;
            let log_trait: Arc<dyn EventLog> = log.clone();
            let embedder: Arc<dyn Embedder> = match embedder_choice {
                "mock" => Arc::new(MockEmbedder::new(32)),
                "fastembed" => Arc::new(mnesio_index::FastEmbedEmbedder::new()?),
                other => anyhow::bail!(
                    "unknown embedder choice {other:?}; expected `mock` or `fastembed`"
                ),
            };

            // Reject mixing embedders against an existing log.
            let entries = log_trait.read_from(None).await?;
            for entry in &entries {
                if let Event::MemoryEmbedded { model_id, .. } = &entry.event {
                    if model_id != embedder.model_id() {
                        anyhow::bail!(
                            "log contains embeddings from {model_id:?} but configured embedder is {:?}",
                            embedder.model_id()
                        );
                    }
                }
            }

            let vector = Arc::new(VectorView::new(
                embedder.dim(),
                embedder.model_id().to_string(),
            ));
            let bm25 = Arc::new(Bm25View::new()?);
            for entry in &entries {
                vector.apply(entry).await?;
                bm25.apply(entry).await?;
            }
            Ok::<_, anyhow::Error>((log_trait, vector, bm25, embedder))
        })?;
        let retriever = Arc::new(HybridRetriever::new(
            vector.clone(),
            bm25.clone(),
            embedder.clone(),
        ));
        let synthesizer: Arc<dyn mnesio_core::Synthesizer> = Arc::new(SnippetSynthesizer::new());
        Ok(Self {
            rt,
            log,
            vector,
            bm25,
            embedder,
            retriever,
            synthesizer,
        })
    }

    pub fn write_memory(&self, args: WriteMemoryArgs) -> Result<String> {
        if args.content.trim().is_empty() {
            return Err(anyhow!("content must be non-empty"));
        }
        self.rt.block_on(async {
            let mem = Memory {
                id: new_id(),
                scope: Scope::global(&args.tenant),
                content: args.content.clone(),
                keywords: vec![],
                tags: args.tags.clone(),
                context: String::new(),
                embedding: None,
                links: vec![],
                parent: None,
                evolution_count: 0,
                time: BiTemporal::now(),
                provenance: Provenance {
                    source: "python".into(),
                    trust: 0.5,
                },
                source: None,
                position: None,
            };
            let memory_id = mem.id;
            let written = Event::MemoryWritten(mem.clone());
            let id1 = self.log.append(written.clone()).await?;
            self.vector
                .apply(&LogEntry {
                    id: id1,
                    event: written.clone(),
                })
                .await?;
            self.bm25
                .apply(&LogEntry {
                    id: id1,
                    event: written,
                })
                .await?;

            let embeddings = self
                .embedder
                .embed(std::slice::from_ref(&args.content))
                .await?;
            let Some(embedding) = embeddings.into_iter().next() else {
                return Err(anyhow!("embedder returned no vectors"));
            };
            let embedded = Event::MemoryEmbedded {
                id: MemoryRef(memory_id),
                embedding,
                model_id: self.embedder.model_id().to_string(),
            };
            let id2 = self.log.append(embedded.clone()).await?;
            self.vector
                .apply(&LogEntry {
                    id: id2,
                    event: embedded,
                })
                .await?;
            Ok::<_, anyhow::Error>(memory_id.to_string())
        })
    }

    pub fn search(&self, args: SearchArgs) -> Result<SearchOutcome> {
        if args.query.trim().is_empty() {
            return Err(anyhow!("query must be non-empty"));
        }
        let k = args.k.clamp(1, 50);
        self.rt.block_on(async {
            // Resolve content + tags by walking the log. Demo-scale
            // pattern — production would cache.
            let entries = self.log.read_from(None).await?;
            let mut contents: HashMap<Id, (String, Vec<String>)> = HashMap::new();
            for entry in &entries {
                if let Event::MemoryWritten(m) = &entry.event {
                    contents.insert(m.id, (m.content.clone(), m.tags.clone()));
                }
            }
            let scope = Scope::global(&args.tenant);
            let hits = self
                .retriever
                .search(&Query {
                    text: args.query.clone(),
                    scope,
                    k,
                    time_filter: None,
                })
                .await?;
            let hit_records: Vec<SearchHitRecord> = hits
                .iter()
                .map(|h| {
                    let (content, tags) = contents
                        .get(&h.memory.0)
                        .cloned()
                        .unwrap_or_else(|| ("<unknown memory>".into(), vec![]));
                    SearchHitRecord {
                        memory_id: h.memory.0.to_string(),
                        content,
                        tags,
                        score: h.score,
                    }
                })
                .collect();
            let passages: Vec<Passage> = hits
                .iter()
                .map(|h| {
                    let (content, tags) = contents
                        .get(&h.memory.0)
                        .cloned()
                        .unwrap_or_else(|| ("<unknown memory>".into(), vec![]));
                    Passage {
                        memory: h.memory,
                        content,
                        tags,
                        retrieval_score: h.score,
                    }
                })
                .collect();
            let answer = self.synthesizer.synthesize(&args.query, &passages).await?;
            Ok::<_, anyhow::Error>(SearchOutcome {
                answer: answer.prose,
                hits: hit_records,
                citations: answer.citations.iter().map(|c| c.0.to_string()).collect(),
            })
        })
    }

    pub fn record_outcome(&self, args: RecordOutcomeArgs) -> Result<String> {
        if args.artifacts_used.is_empty() {
            return Err(anyhow!("artifacts_used must be non-empty"));
        }
        let mut artifact_refs = Vec::with_capacity(args.artifacts_used.len());
        for s in &args.artifacts_used {
            let id = s
                .parse::<Id>()
                .map_err(|e| anyhow!("artifact_id {s:?} not a valid ULID: {e}"))?;
            artifact_refs.push(ArtifactRef(id));
        }
        let episode_id = match args.episode {
            None => new_id(),
            Some(s) => s
                .parse::<Id>()
                .map_err(|e| anyhow!("episode {s:?} not a valid ULID: {e}"))?,
        };
        self.rt.block_on(async {
            let outcome = Outcome {
                id: new_id(),
                episode: EpisodeRef(episode_id),
                artifacts_used: artifact_refs,
                success: Some(args.success),
                scores: args.scores,
                error: args.error,
                judge: JudgeSource::Environment,
                trajectory: TrajectoryRef(new_id()),
            };
            let id = outcome.id;
            self.log.append(Event::OutcomeRecorded(outcome)).await?;
            Ok::<_, anyhow::Error>(id.to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fresh() -> (TempDir, InnerClient) {
        let dir = TempDir::new().unwrap();
        let client = InnerClient::open(dir.path(), "mock").unwrap();
        (dir, client)
    }

    #[test]
    fn write_memory_appends_two_events() {
        let (_dir, c) = fresh();
        let id = c
            .write_memory(WriteMemoryArgs {
                content: "hello world".into(),
                tenant: "t".into(),
                tags: vec!["greeting".into()],
            })
            .unwrap();
        assert!(!id.is_empty());
        // Log should now have a MemoryWritten + MemoryEmbedded pair.
        let entries = c.rt.block_on(c.log.read_from(None)).unwrap();
        let written = entries
            .iter()
            .filter(|e| matches!(e.event, Event::MemoryWritten(_)))
            .count();
        let embedded = entries
            .iter()
            .filter(|e| matches!(e.event, Event::MemoryEmbedded { .. }))
            .count();
        assert_eq!(written, 1);
        assert_eq!(embedded, 1);
    }

    #[test]
    fn write_then_search_round_trips_the_memory() {
        let (_dir, c) = fresh();
        c.write_memory(WriteMemoryArgs {
            content: "the capital of france is paris".into(),
            tenant: "t".into(),
            tags: vec!["geography".into()],
        })
        .unwrap();
        let result = c
            .search(SearchArgs {
                query: "capital france".into(),
                tenant: "t".into(),
                k: 5,
            })
            .unwrap();
        assert!(!result.hits.is_empty(), "search must surface the memory");
        assert!(result.hits[0].content.to_lowercase().contains("paris"));
        assert!(!result.citations.is_empty());
    }

    #[test]
    fn empty_content_rejected() {
        let (_dir, c) = fresh();
        let err = c
            .write_memory(WriteMemoryArgs {
                content: "   ".into(),
                tenant: "t".into(),
                tags: vec![],
            })
            .unwrap_err();
        assert!(err.to_string().contains("non-empty"));
    }

    #[test]
    fn empty_query_rejected() {
        let (_dir, c) = fresh();
        let err = c
            .search(SearchArgs {
                query: "  ".into(),
                tenant: "t".into(),
                k: 5,
            })
            .unwrap_err();
        assert!(err.to_string().contains("non-empty"));
    }

    #[test]
    fn record_outcome_appends_an_event() {
        let (_dir, c) = fresh();
        let aref = new_id().to_string();
        let id = c
            .record_outcome(RecordOutcomeArgs {
                episode: None,
                artifacts_used: vec![aref],
                success: true,
                scores: HashMap::from([("accuracy".into(), 0.95)]),
                error: None,
            })
            .unwrap();
        assert!(!id.is_empty());
        let entries = c.rt.block_on(c.log.read_from(None)).unwrap();
        let count = entries
            .iter()
            .filter(|e| matches!(e.event, Event::OutcomeRecorded(_)))
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn record_outcome_rejects_malformed_artifact_id() {
        let (_dir, c) = fresh();
        let err = c
            .record_outcome(RecordOutcomeArgs {
                episode: None,
                artifacts_used: vec!["not-a-ulid".into()],
                success: true,
                scores: HashMap::new(),
                error: None,
            })
            .unwrap_err();
        assert!(err.to_string().contains("not a valid ULID"));
    }

    #[test]
    fn record_outcome_rejects_empty_artifacts_used() {
        let (_dir, c) = fresh();
        let err = c
            .record_outcome(RecordOutcomeArgs {
                episode: None,
                artifacts_used: vec![],
                success: true,
                scores: HashMap::new(),
                error: None,
            })
            .unwrap_err();
        assert!(err.to_string().contains("non-empty"));
    }
}
