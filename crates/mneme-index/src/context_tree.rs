//! Hierarchical Context-Tree traversal (Phase 16 — retrieval parity).
//!
//! Flat hybrid retrieval treats the corpus as one undifferentiated pool. The
//! 2026 leaders (ByteRover, OpenViking) instead organise memories into a
//! `domain → topic → subtopic` tree and *route* a query to the most relevant
//! branch before scoring — so an on-topic memory in a small sub-pool isn't
//! out-competed by a lexically-similar but off-topic memory from elsewhere.
//!
//! [`ContextTree`] builds that hierarchy from each memory's `tags` (tag *i* is
//! the label at depth *i*), and [`ContextTree::relevant_subtree`] walks the
//! tree greedily — at each level descending into the child whose label best
//! overlaps the query — and returns the [`MemoryRef`]s under the chosen branch.
//! Retrieval can then be *scoped* to that candidate set.
//!
//! **Safety / no-regression discipline.** Routing only ever *narrows*, never
//! reorders, and it degrades to the full corpus whenever the query doesn't
//! clearly match a branch (no matching child at the root → return everything).
//! So a mis-tagged or ambiguous query can't silently drop the answer — the
//! worst case is "no narrowing", i.e. identical to flat retrieval. This mirrors
//! the additive-bonus discipline in [`crate::rerank`]: a new signal may help,
//! but is constructed so it cannot make recall worse.

use mneme_core::types::MemoryRef;
use std::collections::{BTreeMap, BTreeSet};

/// A node in the context tree. Memories are stored at the deepest node their
/// tag path reaches; untagged memories live at the root.
#[derive(Debug, Default)]
struct TreeNode {
    /// Direct memories at this node (tag path ended here).
    memories: Vec<MemoryRef>,
    /// Child branches, keyed by their (lowercased) label for stable ordering.
    children: BTreeMap<String, TreeNode>,
}

impl TreeNode {
    /// Collect every memory in this subtree (this node + all descendants).
    fn collect(&self, out: &mut Vec<MemoryRef>) {
        out.extend_from_slice(&self.memories);
        for child in self.children.values() {
            child.collect(out);
        }
    }
}

/// A `domain → topic → subtopic` tree over the memory corpus, built from tags.
#[derive(Debug, Default)]
pub struct ContextTree {
    root: TreeNode,
    len: usize,
}

impl ContextTree {
    /// Build a tree from `(memory, tags)` pairs. Tag *i* is the label at
    /// depth *i*; a memory is filed at the deepest node its tags reach.
    pub fn build<I, T>(items: I) -> Self
    where
        I: IntoIterator<Item = (MemoryRef, T)>,
        T: AsRef<[String]>,
    {
        let mut tree = ContextTree::default();
        for (memory, tags) in items {
            tree.insert(memory, tags.as_ref());
        }
        tree
    }

    /// File one memory along its tag path.
    pub fn insert(&mut self, memory: MemoryRef, tags: &[String]) {
        let mut node = &mut self.root;
        for tag in tags {
            let key = tag.trim().to_ascii_lowercase();
            if key.is_empty() {
                continue;
            }
            node = node.children.entry(key).or_default();
        }
        node.memories.push(memory);
        self.len += 1;
    }

    /// Total memories filed in the tree.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The labels of the branch [`relevant_subtree`](Self::relevant_subtree)
    /// would scope to for `query` — the "why" behind the narrowing. Empty when
    /// the query matches no branch (i.e. no narrowing happens).
    pub fn route(&self, query: &str) -> Vec<String> {
        self.locate(&tokenize(query)).unwrap_or_default()
    }

    /// Route `query` to the best-matching branch and return the memories under
    /// it. Falls back to the *entire* corpus when no branch clearly matches, so
    /// narrowing can only help recall, never hurt it.
    pub fn relevant_subtree(&self, query: &str) -> Vec<MemoryRef> {
        let q = tokenize(query);
        let mut out = Vec::new();
        match self.locate(&q) {
            Some(path) => {
                self.node_at(&path).collect(&mut out);
                // Always include the root's own (untagged) memories so a routed
                // query never loses access to un-filed facts.
                out.extend_from_slice(&self.root.memories);
            }
            None => self.root.collect(&mut out),
        }
        out
    }

    /// Collect every memory (unscoped). Useful for callers that want to fall
    /// back explicitly.
    pub fn all(&self) -> Vec<MemoryRef> {
        let mut out = Vec::new();
        self.root.collect(&mut out);
        out
    }

    /// Find the label path to the node whose root→node labels cover the most
    /// distinct query tokens. Ties (equal coverage) prefer the *shallower*
    /// node — broader scoping is safer, since over-narrowing is the only way
    /// routing could drop an answer. Returns `None` when nothing matches.
    fn locate(&self, q: &[String]) -> Option<Vec<String>> {
        // best: (coverage, depth, path)
        let mut best: Option<(usize, usize, Vec<String>)> = None;
        let mut path: Vec<String> = Vec::new();
        visit(&self.root, &mut path, &BTreeSet::new(), q, &mut best);
        best.map(|(_, _, p)| p)
    }

    /// Resolve a label path back to its node (path always originates from
    /// [`locate`](Self::locate), so every key exists).
    fn node_at(&self, path: &[String]) -> &TreeNode {
        let mut node = &self.root;
        for label in path {
            match node.children.get(label) {
                Some(child) => node = child,
                None => break,
            }
        }
        node
    }
}

/// DFS scoring every node by the distinct query tokens its path labels cover.
fn visit(
    node: &TreeNode,
    path: &mut Vec<String>,
    matched: &BTreeSet<String>,
    q: &[String],
    best: &mut Option<(usize, usize, Vec<String>)>,
) {
    let coverage = matched.len();
    if coverage > 0 {
        let better = match best {
            None => true,
            Some((bc, bd, _)) => coverage > *bc || (coverage == *bc && path.len() < *bd),
        };
        if better {
            *best = Some((coverage, path.len(), path.clone()));
        }
    }
    for (label, child) in &node.children {
        let mut m = matched.clone();
        for tok in label.split(|c: char| !c.is_alphanumeric()) {
            let tok = tok.to_ascii_lowercase();
            if tok.len() > 1 && q.iter().any(|w| w == &tok) {
                m.insert(tok);
            }
        }
        path.push(label.clone());
        visit(child, path, &m, q, best);
        path.pop();
    }
}

/// Lowercase alphanumeric tokens (length > 1). Deliberately simple and
/// dependency-free — labels are short tags, not prose.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 1)
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mneme_core::types::new_id;

    fn tags(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn routes_query_to_matching_branch() {
        let finance_a = MemoryRef(new_id());
        let finance_b = MemoryRef(new_id());
        let hr = MemoryRef(new_id());
        let tree = ContextTree::build([
            (finance_a, tags(&["finance", "revenue"])),
            (finance_b, tags(&["finance", "costs"])),
            (hr, tags(&["people", "hiring"])),
        ]);
        assert_eq!(tree.len(), 3);

        // A domain-level finance query scopes to the whole finance branch and
        // returns both its memories (not the HR one). Naming only the domain
        // keeps both sub-topics in play — broader is safer.
        let scoped = tree.relevant_subtree("what happened in finance this year");
        assert!(scoped.contains(&finance_a));
        assert!(scoped.contains(&finance_b));
        assert!(
            !scoped.contains(&hr),
            "off-topic HR memory must be excluded"
        );
        assert_eq!(tree.route("finance overview"), vec!["finance"]);
        // Naming the sub-topic narrows further to just that leaf.
        let revenue_only = tree.relevant_subtree("finance revenue");
        assert!(revenue_only.contains(&finance_a));
        assert!(!revenue_only.contains(&finance_b));
    }

    #[test]
    fn descends_multiple_levels() {
        let deep = MemoryRef(new_id());
        let shallow = MemoryRef(new_id());
        let tree = ContextTree::build([
            (deep, tags(&["eng", "payments", "latency"])),
            (shallow, tags(&["eng", "search"])),
        ]);
        // Query keys on the deep path.
        let path = tree.route("payments latency in eng");
        assert_eq!(path, vec!["eng", "payments", "latency"]);
        let scoped = tree.relevant_subtree("payments latency");
        assert!(scoped.contains(&deep));
        assert!(!scoped.contains(&shallow));
    }

    #[test]
    fn unmatched_query_falls_back_to_full_corpus() {
        let a = MemoryRef(new_id());
        let b = MemoryRef(new_id());
        let tree = ContextTree::build([(a, tags(&["finance"])), (b, tags(&["people"]))]);
        // No branch matches → return everything (no narrowing → no regression).
        let scoped = tree.relevant_subtree("completely unrelated astronomy question");
        assert_eq!(scoped.len(), 2);
        assert!(tree.route("astronomy").is_empty());
    }

    #[test]
    fn untagged_memories_stay_reachable_after_routing() {
        let tagged = MemoryRef(new_id());
        let untagged = MemoryRef(new_id());
        let tree = ContextTree::build([
            (tagged, tags(&["finance"])),
            (untagged, tags(&[])), // no tags → filed at root
        ]);
        // Routing into finance still surfaces the untagged root memory.
        let scoped = tree.relevant_subtree("finance");
        assert!(scoped.contains(&tagged));
        assert!(
            scoped.contains(&untagged),
            "untagged root memories must remain reachable after routing"
        );
    }

    #[test]
    fn empty_tree_is_empty() {
        let tree = ContextTree::default();
        assert!(tree.is_empty());
        assert!(tree.relevant_subtree("anything").is_empty());
    }
}
