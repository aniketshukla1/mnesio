//! [`TenantPartitionedVectorView`] — one [`VectorView`] per tenant.
//!
//! ## Why partition?
//!
//! The single global [`VectorView`] post-filters by `Scope` after
//! pulling K best neighbours from the HNSW graph. That's correct, but
//! traversal still walks every tenant's vectors — so a search in
//! tenant A pays for tenant B's index density even when it cannot
//! possibly return any of those B vectors.
//!
//! `TenantPartitionedVectorView` routes every event + every query by
//! `scope.tenant`. The HNSW graph for tenant A only ever contains
//! tenant-A vectors; traversal is bounded by tenant size, not total
//! index size. This is the scaling fix for multi-tenant deployments.
//!
//! ## Tradeoffs
//!
//! - **Memory**: each tenant pays a per-HNSW fixed overhead (one
//!   `RwLock`, one slot table, one tombstone set). For small tenants
//!   this can be wasteful — N hundred tiny indexes occupies more than
//!   one big one.
//! - **No cross-tenant search**: queries with no `tenant` set will
//!   refuse rather than scan every partition. Cross-tenant aggregates
//!   are an explicit privacy concern (Hard Rule #3) — by construction
//!   this view cannot violate it.
//! - **User / session sub-scopes**: still post-filtered within the
//!   tenant's view, same as the single-tenant case. Partitioning only
//!   happens at the tenant boundary because that's the security
//!   boundary (Rule #3).
//!
//! ## When to use which
//!
//! - **`VectorView`**: single-tenant deployments, demos, dev. Less
//!   memory, simpler to reason about, slightly faster for the common
//!   single-tenant case.
//! - **`TenantPartitionedVectorView`**: multi-tenant deployments
//!   where individual tenants might have millions of vectors but
//!   queries are always tenant-scoped.

use crate::VectorView;
use async_trait::async_trait;
use mneme_core::event::{Event, LogEntry};
use mneme_core::traits::MaterializedView;
use mneme_core::types::Id;
use mneme_core::{Hit, MnemeError, Scope};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Multi-tenant vector view. Owns one [`VectorView`] per tenant.
pub struct TenantPartitionedVectorView {
    partitions: RwLock<HashMap<String, Arc<VectorView>>>,
    dim: usize,
    model_id: String,
    last_checkpoint: RwLock<Option<Id>>,
}

impl TenantPartitionedVectorView {
    pub fn new(dim: usize, model_id: impl Into<String>) -> Self {
        Self {
            partitions: RwLock::new(HashMap::new()),
            dim,
            model_id: model_id.into(),
            last_checkpoint: RwLock::new(None),
        }
    }

    /// Number of distinct tenants the view has indices for.
    pub async fn tenant_count(&self) -> usize {
        self.partitions.read().await.len()
    }

    /// Snapshot of `tenant → (slot_count, live_count, tombstone_ratio)`.
    /// The dashboard polls this to render a per-tenant fill-factor
    /// strip.
    pub async fn tenant_stats(&self) -> Vec<TenantStats> {
        let g = self.partitions.read().await;
        let mut out: Vec<TenantStats> = g
            .iter()
            .map(|(tenant, view)| TenantStats {
                tenant: tenant.clone(),
                slot_count: view.slot_count(),
                live_count: view.live_count(),
                tombstone_ratio: view.tombstone_ratio(),
            })
            .collect();
        out.sort_by(|a, b| a.tenant.cmp(&b.tenant));
        out
    }

    /// Borrow the partition for `tenant` if it exists. Returns `None`
    /// when the tenant has no memories yet.
    pub async fn partition(&self, tenant: &str) -> Option<Arc<VectorView>> {
        self.partitions.read().await.get(tenant).cloned()
    }

    /// k-nearest within the query's tenant. Cross-tenant searches
    /// (scopes that don't pin a tenant) are refused — see the module
    /// docs.
    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        scope: &Scope,
    ) -> Result<Vec<Hit>, MnemeError> {
        let partition = self.partitions.read().await.get(&scope.tenant).cloned();
        match partition {
            Some(view) => view.search(query, k, scope),
            None => Ok(Vec::new()),
        }
    }

    /// Get-or-create the partition for `tenant`. Used internally by
    /// `apply`.
    async fn partition_for_write(&self, tenant: &str) -> Arc<VectorView> {
        // Fast path: already exists.
        if let Some(view) = self.partitions.read().await.get(tenant).cloned() {
            return view;
        }
        // Slow path: create. Re-check inside the write lock to avoid
        // duplicate construction if two writers race here.
        let mut g = self.partitions.write().await;
        if let Some(view) = g.get(tenant).cloned() {
            return view;
        }
        let view = Arc::new(VectorView::new(self.dim, self.model_id.clone()));
        g.insert(tenant.to_string(), view.clone());
        view
    }
}

/// Per-tenant fill-factor snapshot, returned by [`TenantPartitionedVectorView::tenant_stats`].
#[derive(Debug, Clone)]
pub struct TenantStats {
    pub tenant: String,
    pub slot_count: usize,
    pub live_count: usize,
    pub tombstone_ratio: f32,
}

#[async_trait]
impl MaterializedView for TenantPartitionedVectorView {
    fn name(&self) -> &str {
        "tenant-partitioned-vector-view"
    }

    async fn apply(&self, entry: &LogEntry) -> Result<(), MnemeError> {
        // Route by tenant. For events that don't carry a memory (e.g.
        // procedural commits, source-level events) we no-op — the same
        // semantics as the single-tenant `VectorView`.
        match &entry.event {
            Event::MemoryWritten(mem) => {
                let view = self.partition_for_write(&mem.scope.tenant).await;
                view.apply(entry).await?;
            }
            Event::MemoryEmbedded { id, .. } => {
                // The embedding doesn't tell us the tenant directly,
                // so we forward to every partition. Each one will
                // either match its pending entry (and insert) or
                // ignore (no pending). Cheap at typical tenant counts.
                let views: Vec<Arc<VectorView>> =
                    self.partitions.read().await.values().cloned().collect();
                for v in &views {
                    v.apply(entry).await?;
                }
                let _ = id;
            }
            Event::MemoryEvolved { .. } | Event::MemoryInvalidated { .. } => {
                // Same broadcast trick — only the partition that holds
                // the affected memory will tombstone a slot; the rest
                // see an unknown MemoryRef and no-op.
                let views: Vec<Arc<VectorView>> =
                    self.partitions.read().await.values().cloned().collect();
                for v in &views {
                    v.apply(entry).await?;
                }
            }
            _ => {}
        }
        *self.last_checkpoint.write().await = Some(entry.id);
        Ok(())
    }

    async fn checkpoint(&self) -> Result<Option<Id>, MnemeError> {
        Ok(*self.last_checkpoint.read().await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mneme_core::entity::{Memory, Provenance};
    use mneme_core::types::{new_id, BiTemporal, MemoryRef};

    fn mem(tenant: &str, content: &str, embedding: Vec<f32>) -> Memory {
        Memory {
            id: new_id(),
            scope: Scope::global(tenant),
            content: content.into(),
            keywords: vec![],
            tags: vec![],
            context: String::new(),
            embedding: Some(embedding),
            links: vec![],
            parent: None,
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: Provenance::default(),
            source: None,
            position: None,
        }
    }

    fn entry(event: Event) -> LogEntry {
        LogEntry {
            id: new_id(),
            event,
        }
    }

    #[tokio::test]
    async fn each_tenant_gets_its_own_partition() {
        let view = TenantPartitionedVectorView::new(4, "test-v1");
        view.apply(&entry(Event::MemoryWritten(mem(
            "a",
            "foo",
            vec![1.0, 0.0, 0.0, 0.0],
        ))))
        .await
        .unwrap();
        view.apply(&entry(Event::MemoryWritten(mem(
            "b",
            "bar",
            vec![0.0, 1.0, 0.0, 0.0],
        ))))
        .await
        .unwrap();
        assert_eq!(view.tenant_count().await, 2);
        let stats = view.tenant_stats().await;
        assert_eq!(stats[0].tenant, "a");
        assert_eq!(stats[1].tenant, "b");
        assert_eq!(stats[0].slot_count, 1);
        assert_eq!(stats[1].slot_count, 1);
    }

    #[tokio::test]
    async fn cross_tenant_search_is_isolated() {
        let view = TenantPartitionedVectorView::new(4, "test-v1");
        // Both memories have identical vectors so a naive global
        // search would surface both for any query.
        let m_a = mem("a", "tenant-a-memory", vec![1.0, 0.0, 0.0, 0.0]);
        let m_b = mem("b", "tenant-b-memory", vec![1.0, 0.0, 0.0, 0.0]);
        let m_a_id = MemoryRef(m_a.id);
        let m_b_id = MemoryRef(m_b.id);
        view.apply(&entry(Event::MemoryWritten(m_a))).await.unwrap();
        view.apply(&entry(Event::MemoryWritten(m_b))).await.unwrap();

        let hits_a = view
            .search(&[1.0, 0.0, 0.0, 0.0], 5, &Scope::global("a"))
            .await
            .unwrap();
        assert_eq!(hits_a.len(), 1);
        assert_eq!(hits_a[0].memory, m_a_id);

        let hits_b = view
            .search(&[1.0, 0.0, 0.0, 0.0], 5, &Scope::global("b"))
            .await
            .unwrap();
        assert_eq!(hits_b.len(), 1);
        assert_eq!(hits_b[0].memory, m_b_id);
    }

    #[tokio::test]
    async fn unknown_tenant_search_returns_empty_not_error() {
        let view = TenantPartitionedVectorView::new(4, "test-v1");
        view.apply(&entry(Event::MemoryWritten(mem(
            "a",
            "foo",
            vec![1.0, 0.0, 0.0, 0.0],
        ))))
        .await
        .unwrap();
        let hits = view
            .search(&[1.0, 0.0, 0.0, 0.0], 5, &Scope::global("nobody"))
            .await
            .unwrap();
        assert!(hits.is_empty());
    }
}
