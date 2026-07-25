//! The timeline fold + time-travel + provenance chains + erasure redaction.
//!
//! [`Timeline`] folds a `LogEntry` stream into per-memory facts keyed by the
//! ULID transaction clock. Everything else — snapshot-as-of, provenance — is a
//! pure read over that fold, with content passed through a [`RedactionPolicy`]
//! so a crypto-shredded subject is blank at every `T`.

use mnesio_core::event::{Event, LogEntry};
use mnesio_core::types::MemoryRef;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

/// Decides which subjects have been crypto-shredded (forgotten). Redaction is
/// applied uniformly across *all* timepoints, so a forgotten subject is absent
/// from live reads and historical replays alike. In production the set is
/// derived from the keyring's forgotten-subjects view (Phase 8); here it's a
/// plain set so the timeline stays dependency-light and testable.
#[derive(Debug, Clone, Default)]
pub struct RedactionPolicy {
    forgotten: HashSet<String>,
}

impl RedactionPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a subject forgotten (crypto-shredded).
    pub fn forget(mut self, subject: impl Into<String>) -> Self {
        self.forgotten.insert(subject.into());
        self
    }

    pub fn is_forgotten(&self, subject: &str) -> bool {
        self.forgotten.contains(subject)
    }

    /// The redaction marker shown in place of erased content.
    pub const REDACTED: &'static str = "[redacted: subject forgotten]";

    /// Project content for a memory whose subject may be forgotten.
    fn project(&self, subject: &str, content: &str) -> String {
        if self.is_forgotten(subject) {
            Self::REDACTED.to_string()
        } else {
            content.to_string()
        }
    }
}

/// Internal per-memory fact accumulated during the fold.
#[derive(Debug, Clone)]
struct MemoryFact {
    memory: MemoryRef,
    content: String,
    subject: String,
    written_ms: u64,
    invalidated_ms: Option<u64>,
    invalidation_reason: Option<String>,
    parent: Option<MemoryRef>,
    /// The memory that superseded this one (the `to` of a `MemoryEvolved`
    /// whose `from` is this memory).
    superseded_by: Option<MemoryRef>,
}

/// A reconstructed memory as seen at some transaction time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MemoryView {
    pub memory_id: String,
    /// Content, redacted if the subject was forgotten.
    pub content: String,
    pub subject: String,
    pub written_ms: u64,
}

/// One step in a provenance chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ProvenanceLinkKind {
    /// The originating write that created this memory.
    Written,
    /// An evolution edge `from → to` (this memory was refined into another).
    Evolved,
    /// The memory was invalidated (retired/superseded).
    Invalidated,
}

/// One link in a [`ProvenanceChain`], ordered by transaction time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProvenanceLink {
    pub kind: ProvenanceLinkKind,
    pub memory_id: String,
    /// For `Evolved`, the memory this evolved *into*; otherwise `None`.
    pub to: Option<String>,
    /// Content snapshot at this link (redacted if forgotten).
    pub content: String,
    /// Free-text reason for `Invalidated` links (supersession cause).
    pub reason: Option<String>,
    pub tx_ms: u64,
}

/// The full provenance of a belief: every source event + supersession across
/// its lineage, oldest first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProvenanceChain {
    pub root: String,
    pub links: Vec<ProvenanceLink>,
}

/// A replayable projection of the log: the system of record, read-only.
pub struct Timeline {
    facts: HashMap<MemoryRef, MemoryFact>,
    /// Insertion order of memories (write tx order), for stable output.
    order: Vec<MemoryRef>,
    /// Number of log entries folded — the append-only invariant witness.
    entry_count: usize,
}

/// Default subject resolver: everything is subject `"public"` (nothing
/// forgettable). Callers that want per-memory erasure pass their own resolver.
pub fn subject_passthrough(_memory: MemoryRef, _content: &str) -> String {
    "public".to_string()
}

impl Timeline {
    /// Fold a log entry stream into a timeline. `subject_of` assigns each
    /// written memory a crypto-shred subject (e.g. derived from content); use
    /// [`subject_passthrough`] when erasure isn't in play.
    pub fn from_entries<F>(entries: &[LogEntry], subject_of: F) -> Self
    where
        F: Fn(MemoryRef, &str) -> String,
    {
        let mut facts: HashMap<MemoryRef, MemoryFact> = HashMap::new();
        let mut order = Vec::new();
        for entry in entries {
            let tx_ms = entry.id.timestamp_ms();
            match &entry.event {
                Event::MemoryWritten(m) => {
                    let r = MemoryRef(m.id);
                    let subject = subject_of(r, &m.content);
                    if !facts.contains_key(&r) {
                        order.push(r);
                    }
                    facts.insert(
                        r,
                        MemoryFact {
                            memory: r,
                            content: m.content.clone(),
                            subject,
                            written_ms: tx_ms,
                            invalidated_ms: None,
                            invalidation_reason: None,
                            parent: m.parent,
                            superseded_by: None,
                        },
                    );
                }
                Event::MemoryEvolved { from, to, .. } => {
                    if let Some(f) = facts.get_mut(from) {
                        f.superseded_by = Some(*to);
                    }
                }
                Event::MemoryInvalidated { id, reason } => {
                    if let Some(f) = facts.get_mut(id) {
                        // Keep the *first* invalidation time (append-only: a
                        // memory is retired once; later duplicates are no-ops).
                        if f.invalidated_ms.is_none() {
                            f.invalidated_ms = Some(tx_ms);
                            f.invalidation_reason = Some(reason.clone());
                        }
                    }
                }
                _ => {}
            }
        }
        Self {
            facts,
            order,
            entry_count: entries.len(),
        }
    }

    /// Number of log entries this timeline was folded from. Provenance never
    /// changes this — erasure redacts the projection, not the log (Hard Rule
    /// #2). Callers assert `entry_count` is stable across a `forget`.
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// The live memory set **as of transaction time `at_ms`**: every memory
    /// written at or before `at_ms` that had not been invalidated at or before
    /// `at_ms`. Content is redacted per `policy`. This is the agent's belief
    /// state at `T`, reconstructed exactly.
    pub fn snapshot_as_of(&self, at_ms: u64, policy: &RedactionPolicy) -> Vec<MemoryView> {
        self.order
            .iter()
            .filter_map(|r| self.facts.get(r))
            .filter(|f| f.written_ms <= at_ms)
            // Live at T iff not invalidated, or invalidated strictly after T.
            // (`Option::is_none_or` would read cleaner but is 1.82+; MSRV 1.79.)
            .filter(|f| f.invalidated_ms.map(|inv| inv > at_ms).unwrap_or(true))
            .map(|f| MemoryView {
                memory_id: f.memory.0.to_string(),
                content: policy.project(&f.subject, &f.content),
                subject: f.subject.clone(),
                written_ms: f.written_ms,
            })
            .collect()
    }

    /// The current live set (as of "now" = max tx time seen). Convenience over
    /// [`Timeline::snapshot_as_of`] with the latest timepoint.
    pub fn live_now(&self, policy: &RedactionPolicy) -> Vec<MemoryView> {
        self.snapshot_as_of(u64::MAX, policy)
    }

    /// The provenance chain for `memory`: walk its lineage (up via `parent`,
    /// down via `superseded_by`) and emit every source event + supersession,
    /// oldest first. Content is redacted per `policy`.
    pub fn provenance(
        &self,
        memory: MemoryRef,
        policy: &RedactionPolicy,
    ) -> Option<ProvenanceChain> {
        if !self.facts.contains_key(&memory) {
            return None;
        }
        // Collect the connected lineage: ancestors (follow parent) + the
        // descendant supersession chain (follow superseded_by).
        let mut lineage: HashSet<MemoryRef> = HashSet::new();
        // ancestors
        let mut cur = Some(memory);
        while let Some(m) = cur {
            if !lineage.insert(m) {
                break; // cycle guard
            }
            cur = self.facts.get(&m).and_then(|f| f.parent);
        }
        // descendants
        let mut cur = self.facts.get(&memory).and_then(|f| f.superseded_by);
        while let Some(m) = cur {
            if !lineage.insert(m) {
                break;
            }
            cur = self.facts.get(&m).and_then(|f| f.superseded_by);
        }

        let mut links: Vec<ProvenanceLink> = Vec::new();
        for m in &lineage {
            let Some(f) = self.facts.get(m) else { continue };
            let content = policy.project(&f.subject, &f.content);
            // the originating write
            links.push(ProvenanceLink {
                kind: ProvenanceLinkKind::Written,
                memory_id: f.memory.0.to_string(),
                to: None,
                content: content.clone(),
                reason: None,
                tx_ms: f.written_ms,
            });
            // an evolution edge, if this memory was superseded
            if let Some(to) = f.superseded_by {
                links.push(ProvenanceLink {
                    kind: ProvenanceLinkKind::Evolved,
                    memory_id: f.memory.0.to_string(),
                    to: Some(to.0.to_string()),
                    content: content.clone(),
                    reason: None,
                    // an evolution's tx-time is the child's write time
                    tx_ms: self
                        .facts
                        .get(&to)
                        .map(|c| c.written_ms)
                        .unwrap_or(f.written_ms),
                });
            }
            // an invalidation, if retired
            if let Some(inv) = f.invalidated_ms {
                links.push(ProvenanceLink {
                    kind: ProvenanceLinkKind::Invalidated,
                    memory_id: f.memory.0.to_string(),
                    to: None,
                    content,
                    reason: f.invalidation_reason.clone(),
                    tx_ms: inv,
                });
            }
        }
        links.sort_by_key(|l| (l.tx_ms, kind_rank(l.kind)));
        Some(ProvenanceChain {
            root: memory.0.to_string(),
            links,
        })
    }
}

/// Stable secondary sort so links at the same tx_ms order Written < Evolved <
/// Invalidated.
fn kind_rank(k: ProvenanceLinkKind) -> u8 {
    match k {
        ProvenanceLinkKind::Written => 0,
        ProvenanceLinkKind::Evolved => 1,
        ProvenanceLinkKind::Invalidated => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::entity::{Memory, Provenance};
    use mnesio_core::types::{new_id, BiTemporal, Id, Scope};
    use std::thread::sleep;
    use std::time::Duration;

    fn mem(content: &str, parent: Option<MemoryRef>) -> Memory {
        Memory {
            id: new_id(),
            scope: Scope::global("t"),
            content: content.into(),
            keywords: vec![],
            tags: vec![],
            context: String::new(),
            embedding: None,
            links: vec![],
            parent,
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

    // Subject resolver keyed by a marker word, for erasure tests.
    fn subject_by_marker(_m: MemoryRef, content: &str) -> String {
        let lc = content.to_ascii_lowercase();
        if lc.contains("alice") {
            "alice".into()
        } else if lc.contains("bob") {
            "bob".into()
        } else {
            "public".into()
        }
    }

    #[test]
    fn snapshot_as_of_reconstructs_belief_at_t() {
        // t0: write A. (pause) t1: write B. (pause) t2: invalidate A.
        let a = mem("fact A", None);
        let a_ref = MemoryRef(a.id);
        let e_a = entry(Event::MemoryWritten(a));
        let t0 = e_a.id.timestamp_ms();
        sleep(Duration::from_millis(3));
        let b = mem("fact B", None);
        let e_b = entry(Event::MemoryWritten(b));
        let t1 = e_b.id.timestamp_ms();
        sleep(Duration::from_millis(3));
        let e_inv = entry(Event::MemoryInvalidated {
            id: a_ref,
            reason: "superseded".into(),
        });
        let t2 = e_inv.id.timestamp_ms();

        let tl = Timeline::from_entries(&[e_a, e_b, e_inv], subject_passthrough);
        let policy = RedactionPolicy::new();

        // As of t0: only A is live.
        let s0 = tl.snapshot_as_of(t0, &policy);
        assert_eq!(s0.len(), 1);
        assert_eq!(s0[0].content, "fact A");

        // As of t1: A and B both live (A not yet invalidated).
        let s1 = tl.snapshot_as_of(t1, &policy);
        assert_eq!(s1.len(), 2);

        // As of t2 (and now): A is gone, only B remains.
        let s2 = tl.snapshot_as_of(t2, &policy);
        assert_eq!(s2.len(), 1);
        assert_eq!(s2[0].content, "fact B");
        assert_eq!(tl.live_now(&policy).len(), 1);
    }

    #[test]
    fn erasure_redacts_across_all_timepoints_without_touching_the_log() {
        // Write an alice-subject memory + a public one.
        let a = mem("alice account 12345", None);
        let p = mem("public earnings note", None);
        let e_a = entry(Event::MemoryWritten(a));
        let t_a = e_a.id.timestamp_ms();
        sleep(Duration::from_millis(2));
        let e_p = entry(Event::MemoryWritten(p));
        let entries = vec![e_a, e_p];

        let tl = Timeline::from_entries(&entries, subject_by_marker);
        let before = tl.entry_count();

        // Forget alice.
        let policy = RedactionPolicy::new().forget("alice");

        // Live now: alice redacted, public intact.
        let now = tl.live_now(&policy);
        let alice = now.iter().find(|v| v.subject == "alice").unwrap();
        assert_eq!(alice.content, RedactionPolicy::REDACTED);
        let pubv = now.iter().find(|v| v.subject == "public").unwrap();
        assert_eq!(pubv.content, "public earnings note");

        // Historical replay (as of when alice was first written): STILL
        // redacted — erasure spans all T, not just the present.
        let past = tl.snapshot_as_of(t_a, &policy);
        let alice_past = past.iter().find(|v| v.subject == "alice").unwrap();
        assert_eq!(alice_past.content, RedactionPolicy::REDACTED);

        // The log is untouched: same entry count (append-only, Hard Rule #2).
        assert_eq!(tl.entry_count(), before);
    }

    #[test]
    fn forgetting_unknown_or_multiple_subjects_is_safe() {
        let a = mem("alice account 12345", None);
        let b = mem("bob ledger entry", None);
        let p = mem("public earnings note", None);
        let entries = vec![
            entry(Event::MemoryWritten(a)),
            entry(Event::MemoryWritten(b)),
            entry(Event::MemoryWritten(p)),
        ];
        let tl = Timeline::from_entries(&entries, subject_by_marker);
        let before = tl.entry_count();

        // Forget a subject that isn't in the timeline at all, plus two that
        // are — chained. Must not panic and must not touch the log.
        let policy = RedactionPolicy::new()
            .forget("ghost-never-here")
            .forget("alice")
            .forget("bob");

        let now = tl.live_now(&policy);
        assert_eq!(
            now.iter().find(|v| v.subject == "alice").unwrap().content,
            RedactionPolicy::REDACTED
        );
        assert_eq!(
            now.iter().find(|v| v.subject == "bob").unwrap().content,
            RedactionPolicy::REDACTED
        );
        // The unrelated public memory is untouched; the unknown subject is a
        // no-op (nothing to redact).
        assert_eq!(
            now.iter().find(|v| v.subject == "public").unwrap().content,
            "public earnings note"
        );
        assert_eq!(tl.entry_count(), before, "append-only: log unchanged");

        // Redaction is a pure read — re-snapshotting yields the same result.
        let again = tl.live_now(&policy);
        assert_eq!(now.len(), again.len());
    }

    #[test]
    fn provenance_chain_traces_write_evolve_invalidate() {
        // A → evolves into B; A is then invalidated.
        let a = mem("Acme revenue up 18%", None);
        let a_ref = MemoryRef(a.id);
        let e_a = entry(Event::MemoryWritten(a));
        sleep(Duration::from_millis(2));
        let b = mem("Acme revenue up 16% (corrected)", Some(a_ref));
        let b_ref = MemoryRef(b.id);
        let e_b = entry(Event::MemoryWritten(b));
        let e_evo = entry(Event::MemoryEvolved {
            from: a_ref,
            to: b_ref,
            diff: mnesio_core::event::ChangeSet {
                keywords_added: vec![],
                keywords_removed: vec![],
                tags_added: vec![],
                tags_removed: vec![],
                context_rewritten: true,
            },
        });
        let e_inv = entry(Event::MemoryInvalidated {
            id: a_ref,
            reason: "corrected".into(),
        });

        let tl = Timeline::from_entries(&[e_a, e_b, e_evo, e_inv], subject_passthrough);
        let chain = tl.provenance(a_ref, &RedactionPolicy::new()).unwrap();

        // The chain covers both A and B and includes a Written, an Evolved
        // (A→B), and an Invalidated (A).
        assert!(chain
            .links
            .iter()
            .any(|l| l.kind == ProvenanceLinkKind::Written && l.memory_id == a_ref.0.to_string()));
        assert!(chain
            .links
            .iter()
            .any(|l| l.kind == ProvenanceLinkKind::Written && l.memory_id == b_ref.0.to_string()));
        let evo = chain
            .links
            .iter()
            .find(|l| l.kind == ProvenanceLinkKind::Evolved)
            .unwrap();
        assert_eq!(evo.to.as_deref(), Some(b_ref.0.to_string().as_str()));
        let inv = chain
            .links
            .iter()
            .find(|l| l.kind == ProvenanceLinkKind::Invalidated)
            .unwrap();
        assert_eq!(inv.reason.as_deref(), Some("corrected"));

        // Querying B gives the same lineage (connected component).
        let from_b = tl.provenance(b_ref, &RedactionPolicy::new()).unwrap();
        assert!(from_b
            .links
            .iter()
            .any(|l| l.memory_id == a_ref.0.to_string()));
    }

    #[test]
    fn provenance_links_are_tx_ordered() {
        let a = mem("first", None);
        let a_ref = MemoryRef(a.id);
        let e_a = entry(Event::MemoryWritten(a));
        sleep(Duration::from_millis(2));
        let e_inv = entry(Event::MemoryInvalidated {
            id: a_ref,
            reason: "x".into(),
        });
        let tl = Timeline::from_entries(&[e_a, e_inv], subject_passthrough);
        let chain = tl.provenance(a_ref, &RedactionPolicy::new()).unwrap();
        // Written before Invalidated.
        let written_pos = chain
            .links
            .iter()
            .position(|l| l.kind == ProvenanceLinkKind::Written)
            .unwrap();
        let inv_pos = chain
            .links
            .iter()
            .position(|l| l.kind == ProvenanceLinkKind::Invalidated)
            .unwrap();
        assert!(written_pos < inv_pos);
    }

    #[test]
    fn forgotten_subject_is_redacted_in_provenance_too() {
        let a = mem("alice secret note", None);
        let a_ref = MemoryRef(a.id);
        let tl = Timeline::from_entries(&[entry(Event::MemoryWritten(a))], subject_by_marker);
        let policy = RedactionPolicy::new().forget("alice");
        let chain = tl.provenance(a_ref, &policy).unwrap();
        assert!(chain
            .links
            .iter()
            .all(|l| l.content == RedactionPolicy::REDACTED));
    }

    #[test]
    fn unknown_memory_has_no_provenance() {
        let tl = Timeline::from_entries(&[], subject_passthrough);
        assert!(tl
            .provenance(MemoryRef(new_id()), &RedactionPolicy::new())
            .is_none());
    }

    #[test]
    fn empty_subject_resolver_is_never_forgotten() {
        // Sanity: passthrough subject "public" isn't redacted by an unrelated
        // forget.
        let a = mem("note", None);
        let _id: Id = a.id;
        let tl = Timeline::from_entries(&[entry(Event::MemoryWritten(a))], subject_passthrough);
        let policy = RedactionPolicy::new().forget("alice");
        assert_eq!(tl.live_now(&policy)[0].content, "note");
    }
}
