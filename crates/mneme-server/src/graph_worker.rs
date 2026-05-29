//! Async graph-view worker.
//!
//! Tails the event log and applies every entry to the persistent
//! [`FjallGraphView`]. Unlike the demo writer — which fans its *own*
//! events out to the vector + BM25 views inline — the graph view is
//! fed exclusively from the log tail. That matters because the
//! interesting graph mutations (`MemoryEvolved`, `MemoryLinksUpdated`,
//! `MemoryNoteEnriched`) are produced by the **evolution worker**, not
//! the demo writer; a log-tailing worker is the only thing that sees
//! all of them without coupling the two workers together.
//!
//! Because the graph store is persistent (fjall), the worker resumes
//! from `FjallGraphView::checkpoint()` rather than replaying the whole
//! log every boot — only genuinely new entries get applied. This is
//! the "materialized view rebuildable by replaying events" contract
//! (Hard Rule #4) in incremental form.

use mneme_core::traits::MaterializedView;
use mneme_core::{EventLog, Id};
use mneme_graph::FjallGraphView;
use std::sync::Arc;
use std::time::Duration;

/// Spawns the graph worker onto the current Tokio runtime. The handle
/// is intentionally dropped — the worker lives for the life of the
/// server.
pub fn spawn(log: Arc<dyn EventLog>, graph: Arc<FjallGraphView>) {
    tokio::spawn(async move {
        let worker = Worker {
            log,
            graph,
            poll_interval: Duration::from_millis(250),
        };
        worker.run().await;
    });
}

struct Worker {
    log: Arc<dyn EventLog>,
    graph: Arc<FjallGraphView>,
    poll_interval: Duration,
}

impl Worker {
    async fn run(self) {
        // Resume from where the persistent store left off. On a fresh
        // (demo) data dir this is `None` → full replay; on an existing
        // store it's the last-applied id → incremental tail only.
        let mut last_seen: Option<Id> = match self.graph.checkpoint().await {
            Ok(cp) => cp,
            Err(e) => {
                tracing::error!(error = %e, "graph worker: checkpoint read failed; starting from head of log");
                None
            }
        };
        tracing::info!(
            resume_from = ?last_seen.map(|id| id.to_string()),
            "graph worker started"
        );

        loop {
            match self.log.read_from(last_seen).await {
                Ok(entries) => {
                    for entry in entries {
                        if let Err(e) = self.graph.apply(&entry).await {
                            tracing::error!(
                                entry = %entry.id,
                                error = %e,
                                "graph worker: apply failed; will retry from this point"
                            );
                            // Don't advance `last_seen` past the failed
                            // entry — next poll re-attempts it.
                            break;
                        }
                        last_seen = Some(entry.id);
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "graph worker: log read failed");
                }
            }
            tokio::time::sleep(self.poll_interval).await;
        }
    }
}
