//! Async tail that keeps the in-memory [`AgentAclView`] current.
//!
//! Attribution is driven by `MemoryWritten` events, which arrive
//! post-boot from several producers (the ingestion worker, the evolution
//! worker, the demo writer). None of them fan out to the ACL view
//! directly, so — like the graph view — it's fed from the log tail.
//! Starts from the boot-replay head so it never double-counts what the
//! startup replay already absorbed.

use mneme_core::traits::MaterializedView;
use mneme_core::{EventLog, Id};
use mneme_index::AgentAclView;
use std::sync::Arc;
use std::time::Duration;

pub fn spawn(log: Arc<dyn EventLog>, acl: Arc<AgentAclView>, start_after: Option<Id>) {
    tokio::spawn(async move {
        let mut last_seen = start_after;
        tracing::info!(resume_from = ?last_seen.map(|i| i.to_string()), "acl worker started");
        loop {
            match log.read_from(last_seen).await {
                Ok(entries) => {
                    for entry in entries {
                        last_seen = Some(entry.id);
                        if let Err(e) = acl.apply(&entry).await {
                            tracing::warn!(error = %e, "acl worker: apply failed");
                        }
                    }
                }
                Err(e) => tracing::error!(error = %e, "acl worker: log read failed"),
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    });
}
