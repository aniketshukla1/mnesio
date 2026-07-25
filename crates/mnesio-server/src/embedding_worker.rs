//! Async embedding worker.
//!
//! Tails the event log, finds `MemoryWritten` events whose `embedding` was
//! `None`, runs them through the configured [`Embedder`], and appends a
//! `MemoryEmbedded` event back to the log. This is the piece that lets the
//! write path stay fast (Rule #5 — writes never wait on an LLM-class call)
//! while still feeding the vector view real semantic vectors when a heavier
//! embedder (FastEmbed, FastEmbed-Q, future Qwen3) is in use.
//!
//! Phase-0 implementation notes:
//! - Single worker task per process. Multiple writers append; the worker
//!   serialises embedding through the embedder's `&mut`-style interior.
//! - Polling tail (200ms interval). A push-based event tail (mpsc channel
//!   tied to `EventLog::append`) is a future refinement once we have a
//!   `LoggedWriter` abstraction in `mnesio-store`.
//! - Worker also keeps the `VectorView` directly updated as a fan-out so
//!   the view sees the embedding immediately instead of waiting on its own
//!   tail loop. Same "writer fans events to views" pattern the demo writer
//!   already uses.

use mnesio_core::entity::Memory;
use mnesio_core::event::{Event, LogEntry};
use mnesio_core::traits::MaterializedView;
use mnesio_core::types::MemoryRef;
use mnesio_core::{Embedder, EventLog, Id};
use mnesio_index::VectorView;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

/// Spawns the embedding worker onto the current Tokio runtime. The handle
/// is intentionally dropped — the worker lives for the life of the server.
pub fn spawn(log: Arc<dyn EventLog>, embedder: Arc<dyn Embedder>, vector: Arc<VectorView>) {
    tokio::spawn(async move {
        let worker = Worker {
            log,
            embedder,
            vector,
            poll_interval: Duration::from_millis(200),
        };
        worker.run().await;
    });
}

struct Worker {
    log: Arc<dyn EventLog>,
    embedder: Arc<dyn Embedder>,
    vector: Arc<VectorView>,
    poll_interval: Duration,
}

impl Worker {
    async fn run(self) {
        let mut last_seen: Option<Id> = None;
        // Track which memories already have a `MemoryEmbedded` event in the
        // log so we don't double-embed when the worker restarts and replays
        // its own history.
        let mut embedded: HashSet<MemoryRef> = HashSet::new();

        tracing::info!(
            model_id = self.embedder.model_id(),
            dim = self.embedder.dim(),
            "embedding worker started"
        );

        loop {
            let entries = match self.log.read_from(last_seen).await {
                Ok(e) => e,
                Err(e) => {
                    tracing::error!(error = %e, "embedding worker: log read failed");
                    tokio::time::sleep(self.poll_interval).await;
                    continue;
                }
            };

            for entry in entries {
                last_seen = Some(entry.id);
                match &entry.event {
                    Event::MemoryEmbedded { id, .. } => {
                        embedded.insert(*id);
                    }
                    Event::MemoryWritten(mem) => {
                        let mref = MemoryRef(mem.id);
                        if mem.embedding.is_some() || embedded.contains(&mref) {
                            continue;
                        }
                        match self.embed_and_publish(mem).await {
                            Ok(()) => {
                                embedded.insert(mref);
                            }
                            Err(e) => {
                                tracing::error!(
                                    memory = %mem.id,
                                    error = %e,
                                    "embedding worker: embed failed; will retry on next poll"
                                );
                            }
                        }
                    }
                    Event::MemoryInvalidated { id, .. } => {
                        // Drop from the set so a re-issued memory under the
                        // same id (unlikely in practice — ids are ULIDs) would
                        // get re-embedded.
                        embedded.remove(id);
                    }
                    _ => {}
                }
            }

            tokio::time::sleep(self.poll_interval).await;
        }
    }

    async fn embed_and_publish(&self, mem: &Memory) -> anyhow::Result<()> {
        let vectors = self
            .embedder
            .embed(std::slice::from_ref(&mem.content))
            .await
            .map_err(|e| anyhow::anyhow!("embed: {e}"))?;
        let embedding = vectors
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("embedder returned no vectors"))?;
        let event = Event::MemoryEmbedded {
            id: MemoryRef(mem.id),
            embedding,
            model_id: self.embedder.model_id().to_string(),
        };
        let id = self.log.append(event.clone()).await?;
        let entry = LogEntry { id, event };
        // Apply directly to the view so search doesn't have to wait for the
        // next poll cycle to discover its own embedding.
        self.vector
            .apply(&entry)
            .await
            .map_err(|e| anyhow::anyhow!("vector apply: {e}"))?;
        tracing::debug!(
            memory = %mem.id,
            dim = self.embedder.dim(),
            "embedding worker: embedded memory"
        );
        Ok(())
    }
}
