//! Node + edge records and the key-encoding used by the fjall view.
//!
//! Key layouts are intentionally hand-rolled. Encoding via `bincode`
//! would also work but would hide the prefix-scan contract behind a
//! length prefix and break the ordering guarantees we rely on for
//! traversal — see [`encode_edge_out_key`] for the contract.

use mneme_core::types::{MemoryRef, Scope, SourceRef};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

/// Typed edge labels. Keep the discriminants stable — they're part of
/// the on-disk key so changing a value migrates the whole graph.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Relation {
    /// An explicit cross-reference from one memory to another (the
    /// A-MEM evolution worker is the main source; user-authored links
    /// land here too).
    Linked = 1,
    /// `src` is an evolved version of `dst`. Traversing this edge
    /// walks lineage *backwards* (newer → older).
    EvolvedFrom = 2,
    /// Reverse of `EvolvedFrom` — stored so lineage walks forward
    /// (older → newer) are also a single prefix scan.
    EvolvedTo = 3,
    /// `src` is a chunk of source document `dst`.
    ContainedIn = 4,
}

impl Relation {
    pub fn as_byte(self) -> u8 {
        self as u8
    }

    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            1 => Some(Relation::Linked),
            2 => Some(Relation::EvolvedFrom),
            3 => Some(Relation::EvolvedTo),
            4 => Some(Relation::ContainedIn),
            _ => None,
        }
    }
}

/// One node in the graph. Mirrors the slice of `Memory` we care about
/// for routing/filtering. We deliberately do **not** carry the memory
/// content here — the event log is the system of record (Hard Rule
/// #4); resolve content via `EventLog::read_from` if you need it.
///
/// Most nodes are memories, but a node can also represent a `Source`
/// document (created from `SourceIngested`) so that chunk→source
/// `ContainedIn` edges resolve to a real endpoint. Source nodes set
/// [`NodeRecord::is_source`] and carry the document title in
/// [`NodeRecord::label`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRecord {
    pub id: MemoryRef,
    pub scope: Scope,
    pub tags: Vec<String>,
    pub keywords: Vec<String>,
    pub source: Option<SourceRef>,
    pub position: Option<u32>,
    pub evolution_count: u16,
    /// Bi-temporal validity. `valid_from`/`valid_to` come from
    /// `Memory.time`. `tx_to` becomes `Some(_)` on
    /// `MemoryInvalidated`; everything else updates `valid_to` only.
    pub valid_from: OffsetDateTime,
    pub valid_to: Option<OffsetDateTime>,
    pub tx_from: OffsetDateTime,
    pub tx_to: Option<OffsetDateTime>,
    /// Human-readable label. `Some(_)` for source nodes (the document
    /// title); `None` for memory nodes (whose label is derived from
    /// tags/keywords downstream). Appended last with `#[serde(default)]`
    /// so the field is back-compatible with pre-source graph records.
    #[serde(default)]
    pub label: Option<String>,
    /// `true` when this node represents a `Source` document rather than
    /// a `Memory`. Back-compat default `false`.
    #[serde(default)]
    pub is_source: bool,
}

impl NodeRecord {
    /// True iff this node is part of the live graph at instant `at`.
    /// "Live" here means both: the memory was valid at `at` *and* the
    /// system hadn't tombstoned it before `at` (transaction time).
    pub fn is_live_at(&self, at: OffsetDateTime) -> bool {
        let valid = self.valid_from <= at && self.valid_to.map_or(true, |e| at < e);
        let known = self.tx_from <= at && self.tx_to.map_or(true, |e| at < e);
        valid && known
    }
}

/// One edge in the graph. The edge's *identity* lives in the key
/// (`src`, `relation`, `dst`, `tx_from`); the value holds the
/// transaction-time end and any properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeRecord {
    pub src: MemoryRef,
    pub relation: Relation,
    pub dst: MemoryRef,
    pub tx_from: OffsetDateTime,
    /// `Some(_)` once the edge has been superseded (e.g. by a
    /// `MemoryLinksUpdated` event replacing this Linked edge, or by
    /// `MemoryInvalidated` on either endpoint).
    pub tx_to: Option<OffsetDateTime>,
    /// Optional weight set by the evolution worker. `None` means
    /// "no preference".
    pub weight: Option<f32>,
}

impl EdgeRecord {
    pub fn is_live_at(&self, at: OffsetDateTime) -> bool {
        self.tx_from <= at && self.tx_to.map_or(true, |e| at < e)
    }
}

/// Edge-key contract: `[src_id (16B) | relation (1B) | dst_id (16B) | tx_from_micros_be (8B)]`.
///
/// The prefix `[src_id]` lets us scan all edges out of a node in one
/// fjall range; `[src_id | relation]` narrows to one relation type.
/// The trailing `tx_from` ensures multiple incarnations of the same
/// (src, relation, dst) triple (e.g. a link that was removed and then
/// re-added) get distinct keys rather than overwriting each other —
/// otherwise we'd silently lose the previous tx interval and break
/// bi-temporal replay.
pub fn encode_edge_out_key(
    src: MemoryRef,
    relation: Relation,
    dst: MemoryRef,
    tx_from: OffsetDateTime,
) -> [u8; 41] {
    let mut k = [0u8; 41];
    k[0..16].copy_from_slice(&src.0.to_bytes());
    k[16] = relation.as_byte();
    k[17..33].copy_from_slice(&dst.0.to_bytes());
    let micros = tx_from.unix_timestamp_nanos() / 1_000;
    k[33..41].copy_from_slice(&(micros as i64).to_be_bytes());
    k
}

/// Reverse-direction edge-key: `[dst_id | relation | src_id | tx_from]`.
/// Same shape so the same prefix-scan tricks work for in-neighbours.
pub fn encode_edge_in_key(
    src: MemoryRef,
    relation: Relation,
    dst: MemoryRef,
    tx_from: OffsetDateTime,
) -> [u8; 41] {
    let mut k = [0u8; 41];
    k[0..16].copy_from_slice(&dst.0.to_bytes());
    k[16] = relation.as_byte();
    k[17..33].copy_from_slice(&src.0.to_bytes());
    let micros = tx_from.unix_timestamp_nanos() / 1_000;
    k[33..41].copy_from_slice(&(micros as i64).to_be_bytes());
    k
}

/// Node key — just the 16-byte ULID.
pub fn encode_node_key(id: MemoryRef) -> [u8; 16] {
    id.0.to_bytes()
}

/// Prefix for "all out-edges from `src`" — first 16 bytes.
pub fn out_prefix(src: MemoryRef) -> [u8; 16] {
    src.0.to_bytes()
}

/// Prefix for "all out-edges from `src` with `relation`" — 17 bytes.
pub fn out_prefix_rel(src: MemoryRef, relation: Relation) -> [u8; 17] {
    let mut p = [0u8; 17];
    p[0..16].copy_from_slice(&src.0.to_bytes());
    p[16] = relation.as_byte();
    p
}

/// Prefix for "all in-edges to `dst`".
pub fn in_prefix(dst: MemoryRef) -> [u8; 16] {
    dst.0.to_bytes()
}

/// Prefix for "all in-edges to `dst` with `relation`".
pub fn in_prefix_rel(dst: MemoryRef, relation: Relation) -> [u8; 17] {
    let mut p = [0u8; 17];
    p[0..16].copy_from_slice(&dst.0.to_bytes());
    p[16] = relation.as_byte();
    p
}

/// Half-open upper bound for a prefix scan. Returns `None` when the
/// prefix is all `0xff` — fjall callers should treat that as "scan
/// to the end of the partition" via `Bound::Unbounded`.
pub fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    for byte in end.iter_mut().rev() {
        if *byte == 0xff {
            *byte = 0;
        } else {
            *byte += 1;
            return Some(end);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use mneme_core::new_id;

    fn mref() -> MemoryRef {
        MemoryRef(new_id())
    }

    #[test]
    fn relation_byte_roundtrip() {
        for r in [
            Relation::Linked,
            Relation::EvolvedFrom,
            Relation::EvolvedTo,
            Relation::ContainedIn,
        ] {
            assert_eq!(Relation::from_byte(r.as_byte()), Some(r));
        }
        assert_eq!(Relation::from_byte(0), None);
        assert_eq!(Relation::from_byte(99), None);
    }

    #[test]
    fn edge_key_layout_is_prefix_scannable() {
        let src = mref();
        let dst_a = mref();
        let dst_b = mref();
        let t = OffsetDateTime::now_utc();

        let k_a = encode_edge_out_key(src, Relation::Linked, dst_a, t);
        let k_b = encode_edge_out_key(src, Relation::Linked, dst_b, t);
        let k_c = encode_edge_out_key(src, Relation::EvolvedFrom, dst_a, t);

        let prefix_src = out_prefix(src);
        assert!(k_a.starts_with(&prefix_src));
        assert!(k_b.starts_with(&prefix_src));
        assert!(k_c.starts_with(&prefix_src));

        let prefix_src_linked = out_prefix_rel(src, Relation::Linked);
        assert!(k_a.starts_with(&prefix_src_linked));
        assert!(k_b.starts_with(&prefix_src_linked));
        // The wrong-relation prefix must not match.
        assert!(!k_c.starts_with(&prefix_src_linked));
    }

    #[test]
    fn edge_key_distinguishes_tx_from() {
        // Same (src, relation, dst) at different tx-times must produce
        // different keys — otherwise re-adding a removed link would
        // overwrite its prior tx interval and corrupt bi-temporal
        // replay.
        let src = mref();
        let dst = mref();
        let t1 = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let t2 = OffsetDateTime::from_unix_timestamp(1_700_000_001).unwrap();
        let k1 = encode_edge_out_key(src, Relation::Linked, dst, t1);
        let k2 = encode_edge_out_key(src, Relation::Linked, dst, t2);
        assert_ne!(k1, k2);
    }

    #[test]
    fn prefix_upper_bound_advances_last_non_ff_byte() {
        let p = vec![0x01, 0x02, 0x03];
        assert_eq!(prefix_upper_bound(&p), Some(vec![0x01, 0x02, 0x04]));

        let p = vec![0x01, 0xff, 0xff];
        assert_eq!(prefix_upper_bound(&p), Some(vec![0x02, 0x00, 0x00]));

        let p = vec![0xff, 0xff];
        assert_eq!(prefix_upper_bound(&p), None);
    }

    #[test]
    fn node_record_liveness_respects_both_clocks() {
        let id = mref();
        let scope = Scope::global("t");
        let t = |s: i64| OffsetDateTime::from_unix_timestamp(s).unwrap();
        let n = NodeRecord {
            id,
            scope,
            tags: vec![],
            keywords: vec![],
            source: None,
            position: None,
            evolution_count: 0,
            valid_from: t(100),
            valid_to: Some(t(200)),
            tx_from: t(100),
            tx_to: None,
            label: None,
            is_source: false,
        };
        assert!(!n.is_live_at(t(50)), "before valid_from");
        assert!(n.is_live_at(t(150)), "inside both intervals");
        assert!(!n.is_live_at(t(250)), "after valid_to");

        let mut n2 = n.clone();
        n2.tx_to = Some(t(120));
        assert!(!n2.is_live_at(t(150)), "tombstoned in tx-time before now");
        assert!(n2.is_live_at(t(110)), "still live in tx-time at this point");
    }
}
