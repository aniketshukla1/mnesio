//! [`AgentAclView`] — multi-agent attribution + inter-agent read ACLs.
//!
//! `Scope` (tenant / user / session) is the *primary* security boundary
//! (Hard Rule #3). In a multi-agent deployment several agents share one
//! scope but shouldn't automatically read each other's private memories.
//! This view adds a finer, opt-in boundary *inside* a scope:
//!
//! - Every memory is **attributed** to an owning agent — its
//!   [`mneme_core::entity::Provenance::source`] (the ingestion worker
//!   sets this to the observation's `actor`). No new field on `Memory`.
//! - An agent may always read **its own** memories and **system/shared**
//!   memories (those whose owner is a known system source like
//!   `"ingestion"` / `"demo"` / `"evolution-worker"`).
//! - Reading **another agent's** memory requires an explicit
//!   `AgentAccessGranted` (owner → grantee).
//!
//! Enforcement is opt-in: a read that specifies *no* requesting actor
//! sees everything within its scope (the pre-ACL behaviour). A read that
//! specifies an actor is filtered through [`AgentAclView::can_read_memory`].
//! Rebuildable from the log (Hard Rule #4).

use async_trait::async_trait;
use mneme_core::event::{Event, LogEntry};
use mneme_core::traits::MaterializedView;
use mneme_core::types::{Id, MemoryRef};
use mneme_core::MnemeError;
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

/// Owner-source strings that are *not* real agents — they're system
/// machinery, and their memories are readable by every agent in the
/// scope.
fn is_system_owner(owner: &str) -> bool {
    matches!(
        owner,
        "ingestion"
            | "evolution-worker"
            | "evolution"
            | "demo"
            | "demo-article"
            | "memeval"
            | "unknown"
            | ""
    )
}

#[derive(Default)]
struct Inner {
    /// `tenant → owner → {grantees}`.
    grants: HashMap<String, HashMap<String, HashSet<String>>>,
    /// `memory → (tenant, owner)` so a hit can be ACL-checked by ref.
    owner_by_ref: HashMap<MemoryRef, (String, String)>,
    /// `tenant → owner → count` — attribution for dashboards.
    attribution: HashMap<String, HashMap<String, u64>>,
}

/// Multi-agent ACL + attribution view.
#[derive(Default)]
pub struct AgentAclView {
    inner: RwLock<Inner>,
    last_checkpoint: RwLock<Option<Id>>,
}

/// One agent's attribution tally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentAttribution {
    pub agent: String,
    pub memories: u64,
    /// Agents this agent has granted read access to.
    pub grants_to: Vec<String>,
    pub is_system: bool,
}

impl AgentAclView {
    pub fn new() -> Self {
        Self::default()
    }

    /// Policy: may `requester` read a memory owned by `owner` in `tenant`?
    pub fn can_read(&self, tenant: &str, requester: &str, owner: &str) -> bool {
        if requester == owner || is_system_owner(owner) {
            return true;
        }
        let g = self.inner.read().unwrap();
        g.grants
            .get(tenant)
            .and_then(|by_owner| by_owner.get(owner))
            .map(|grantees| grantees.contains(requester))
            .unwrap_or(false)
    }

    /// Enforce on a concrete memory ref. Unknown refs (not yet attributed
    /// or created before the ACL view existed) are *permitted* — ACL is
    /// an additive filter, never a way to silently lose pre-existing
    /// reads.
    pub fn can_read_memory(&self, requester: &str, memory: MemoryRef) -> bool {
        let g = self.inner.read().unwrap();
        match g.owner_by_ref.get(&memory) {
            Some((tenant, owner)) => {
                if requester == owner || is_system_owner(owner) {
                    return true;
                }
                g.grants
                    .get(tenant)
                    .and_then(|by_owner| by_owner.get(owner))
                    .map(|grantees| grantees.contains(requester))
                    .unwrap_or(false)
            }
            None => true,
        }
    }

    /// Per-agent attribution + grant edges for a tenant, sorted by agent.
    pub fn attribution(&self, tenant: &str) -> Vec<AgentAttribution> {
        let g = self.inner.read().unwrap();
        let counts = g.attribution.get(tenant);
        let grants = g.grants.get(tenant);
        let mut agents: HashSet<String> = HashSet::new();
        if let Some(c) = counts {
            agents.extend(c.keys().cloned());
        }
        if let Some(gr) = grants {
            agents.extend(gr.keys().cloned());
        }
        let mut out: Vec<AgentAttribution> = agents
            .into_iter()
            .map(|agent| {
                let memories = counts.and_then(|c| c.get(&agent)).copied().unwrap_or(0);
                let mut grants_to: Vec<String> = grants
                    .and_then(|gr| gr.get(&agent))
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default();
                grants_to.sort();
                AgentAttribution {
                    is_system: is_system_owner(&agent),
                    agent,
                    memories,
                    grants_to,
                }
            })
            .collect();
        out.sort_by(|a, b| a.agent.cmp(&b.agent));
        out
    }
}

#[async_trait]
impl MaterializedView for AgentAclView {
    fn name(&self) -> &str {
        "agent-acl-view"
    }

    async fn apply(&self, entry: &LogEntry) -> Result<(), MnemeError> {
        match &entry.event {
            Event::MemoryWritten(m) => {
                let mut g = self.inner.write().unwrap();
                let owner = m.provenance.source.clone();
                g.owner_by_ref
                    .insert(MemoryRef(m.id), (m.scope.tenant.clone(), owner.clone()));
                *g.attribution
                    .entry(m.scope.tenant.clone())
                    .or_default()
                    .entry(owner)
                    .or_insert(0) += 1;
            }
            Event::MemoryInvalidated { id, .. } => {
                // Decrement attribution + drop the ref mapping.
                let mut g = self.inner.write().unwrap();
                if let Some((tenant, owner)) = g.owner_by_ref.remove(id) {
                    if let Some(c) = g
                        .attribution
                        .get_mut(&tenant)
                        .and_then(|t| t.get_mut(&owner))
                    {
                        *c = c.saturating_sub(1);
                    }
                }
            }
            Event::AgentAccessGranted {
                tenant,
                owner,
                grantee,
            } => {
                let mut g = self.inner.write().unwrap();
                g.grants
                    .entry(tenant.clone())
                    .or_default()
                    .entry(owner.clone())
                    .or_default()
                    .insert(grantee.clone());
            }
            Event::AgentAccessRevoked {
                tenant,
                owner,
                grantee,
            } => {
                let mut g = self.inner.write().unwrap();
                if let Some(grantees) = g
                    .grants
                    .get_mut(tenant)
                    .and_then(|by_owner| by_owner.get_mut(owner))
                {
                    grantees.remove(grantee);
                }
            }
            _ => {}
        }
        *self.last_checkpoint.write().unwrap() = Some(entry.id);
        Ok(())
    }

    async fn checkpoint(&self) -> Result<Option<Id>, MnemeError> {
        Ok(*self.last_checkpoint.read().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mneme_core::entity::{Memory, Provenance};
    use mneme_core::types::{new_id, BiTemporal, Scope};

    fn mem_owned_by(tenant: &str, owner: &str) -> Memory {
        Memory {
            id: new_id(),
            scope: Scope::global(tenant),
            content: "x".into(),
            keywords: vec![],
            tags: vec![],
            context: String::new(),
            embedding: None,
            links: vec![],
            parent: None,
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: Provenance {
                source: owner.into(),
                trust: 1.0,
            },
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

    async fn write(v: &AgentAclView, m: Memory) -> MemoryRef {
        let r = MemoryRef(m.id);
        v.apply(&entry(Event::MemoryWritten(m))).await.unwrap();
        r
    }

    #[tokio::test]
    async fn agent_reads_own_and_system_but_not_others() {
        let v = AgentAclView::new();
        let scout_mem = write(&v, mem_owned_by("t", "scout")).await;
        let analyst_mem = write(&v, mem_owned_by("t", "analyst")).await;
        let system_mem = write(&v, mem_owned_by("t", "ingestion")).await;

        // scout reads own + system, but not analyst's (no grant).
        assert!(v.can_read_memory("scout", scout_mem));
        assert!(v.can_read_memory("scout", system_mem));
        assert!(!v.can_read_memory("scout", analyst_mem));
    }

    #[tokio::test]
    async fn grant_enables_cross_agent_read_revoke_disables() {
        let v = AgentAclView::new();
        let analyst_mem = write(&v, mem_owned_by("t", "analyst")).await;
        assert!(!v.can_read_memory("scout", analyst_mem));

        v.apply(&entry(Event::AgentAccessGranted {
            tenant: "t".into(),
            owner: "analyst".into(),
            grantee: "scout".into(),
        }))
        .await
        .unwrap();
        assert!(
            v.can_read_memory("scout", analyst_mem),
            "grant enables read"
        );

        v.apply(&entry(Event::AgentAccessRevoked {
            tenant: "t".into(),
            owner: "analyst".into(),
            grantee: "scout".into(),
        }))
        .await
        .unwrap();
        assert!(
            !v.can_read_memory("scout", analyst_mem),
            "revoke disables read"
        );
    }

    #[tokio::test]
    async fn grant_is_directional_and_tenant_scoped() {
        let v = AgentAclView::new();
        // analyst→scout grant doesn't let analyst read scout's memory.
        let scout_mem = write(&v, mem_owned_by("t", "scout")).await;
        v.apply(&entry(Event::AgentAccessGranted {
            tenant: "t".into(),
            owner: "analyst".into(),
            grantee: "scout".into(),
        }))
        .await
        .unwrap();
        assert!(
            !v.can_read_memory("analyst", scout_mem),
            "grant is one-directional"
        );

        // A grant in tenant "t" doesn't apply in tenant "z".
        assert!(!v.can_read("z", "scout", "analyst"));
        assert!(v.can_read("t", "scout", "analyst"));
    }

    #[tokio::test]
    async fn unknown_ref_is_permitted() {
        let v = AgentAclView::new();
        assert!(
            v.can_read_memory("anyone", MemoryRef(new_id())),
            "ACL is additive — unknown refs are not hidden"
        );
    }

    #[tokio::test]
    async fn attribution_counts_and_grant_edges() {
        let v = AgentAclView::new();
        write(&v, mem_owned_by("t", "scout")).await;
        write(&v, mem_owned_by("t", "scout")).await;
        write(&v, mem_owned_by("t", "analyst")).await;
        v.apply(&entry(Event::AgentAccessGranted {
            tenant: "t".into(),
            owner: "scout".into(),
            grantee: "analyst".into(),
        }))
        .await
        .unwrap();

        let attr = v.attribution("t");
        let scout = attr.iter().find(|a| a.agent == "scout").unwrap();
        assert_eq!(scout.memories, 2);
        assert_eq!(scout.grants_to, vec!["analyst".to_string()]);
        assert!(!scout.is_system);
        let analyst = attr.iter().find(|a| a.agent == "analyst").unwrap();
        assert_eq!(analyst.memories, 1);
    }

    #[tokio::test]
    async fn invalidation_decrements_attribution() {
        let v = AgentAclView::new();
        let m = write(&v, mem_owned_by("t", "scout")).await;
        assert_eq!(v.attribution("t")[0].memories, 1);
        v.apply(&entry(Event::MemoryInvalidated {
            id: m,
            reason: "x".into(),
        }))
        .await
        .unwrap();
        let scout = v.attribution("t").into_iter().find(|a| a.agent == "scout");
        assert_eq!(scout.map(|a| a.memories), Some(0));
    }
}
