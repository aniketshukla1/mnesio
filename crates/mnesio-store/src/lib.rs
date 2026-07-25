//! # mnesio-store
//!
//! The append-only event log (system of record) and materialized-view
//! plumbing. Backed by **fjall** — chosen over `sled` (in long-term limbo)
//! per the architecture report §7.
//!
//! Phase 0 deliverable: write/read the log, replay it to rebuild views.

use mnesio_core::{Event, EventLog, Id, LogEntry, MnesioError};
use std::path::Path;
use std::sync::Arc;

/// A fjall-backed event log. Keys are ULIDs (lexicographically time-ordered),
/// values are bincode-encoded [`LogEntry`]s.
pub struct FjallEventLog {
    keyspace: fjall::Keyspace,
    events: fjall::PartitionHandle,
}

impl FjallEventLog {
    pub fn open(path: impl AsRef<Path>) -> Result<Arc<Self>, MnesioError> {
        let keyspace = fjall::Config::new(path)
            .open()
            .map_err(|e| MnesioError::Storage(e.to_string()))?;
        let events = keyspace
            .open_partition("events", fjall::PartitionCreateOptions::default())
            .map_err(|e| MnesioError::Storage(e.to_string()))?;
        Ok(Arc::new(Self { keyspace, events }))
    }
}

#[async_trait::async_trait]
impl EventLog for FjallEventLog {
    async fn append(&self, event: Event) -> Result<Id, MnesioError> {
        let id = mnesio_core::new_id();
        let entry = LogEntry { id, event };
        let bytes = bincode::serialize(&entry).map_err(|e| MnesioError::Storage(e.to_string()))?;
        // ULID -> 16 big-endian bytes keeps key order == time order.
        self.events
            .insert(id.to_bytes(), bytes)
            .map_err(|e| MnesioError::Storage(e.to_string()))?;
        self.keyspace
            .persist(fjall::PersistMode::Buffer)
            .map_err(|e| MnesioError::Storage(e.to_string()))?;
        Ok(id)
    }

    async fn read_from(&self, after: Option<Id>) -> Result<Vec<LogEntry>, MnesioError> {
        let mut out = Vec::new();
        let bounds = match after {
            Some(id) => {
                let mut start = id.to_bytes();
                // exclusive lower bound: smallest key strictly greater
                for byte in start.iter_mut().rev() {
                    if *byte == 0xFF {
                        *byte = 0;
                    } else {
                        *byte += 1;
                        break;
                    }
                }
                (std::ops::Bound::Included(start), std::ops::Bound::Unbounded)
            }
            None => (std::ops::Bound::Unbounded, std::ops::Bound::Unbounded),
        };
        for kv in self.events.range(bounds) {
            let (_, v) = kv.map_err(|e| MnesioError::Storage(e.to_string()))?;
            let entry: LogEntry =
                bincode::deserialize(&v).map_err(|e| MnesioError::Storage(e.to_string()))?;
            out.push(entry);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::{Memory, Provenance, Scope};

    fn sample_memory() -> Memory {
        Memory {
            id: mnesio_core::new_id(),
            scope: Scope::global("test"),
            content: "the market closed green".into(),
            keywords: vec![],
            tags: vec![],
            context: String::new(),
            embedding: None,
            links: vec![],
            parent: None,
            evolution_count: 0,
            time: mnesio_core::BiTemporal::now(),
            provenance: Provenance::default(),
            source: None,
            position: None,
        }
    }

    #[tokio::test]
    async fn append_then_replay() {
        let dir = std::env::temp_dir().join(format!("mnesio-test-{}", mnesio_core::new_id()));
        let log = FjallEventLog::open(&dir).unwrap();

        let a = log
            .append(Event::MemoryWritten(sample_memory()))
            .await
            .unwrap();
        let _b = log
            .append(Event::MemoryWritten(sample_memory()))
            .await
            .unwrap();

        let all = log.read_from(None).await.unwrap();
        assert_eq!(all.len(), 2);

        let after_a = log.read_from(Some(a)).await.unwrap();
        assert_eq!(after_a.len(), 1, "read_from(a) must exclude a itself");

        std::fs::remove_dir_all(&dir).ok();
    }
}
