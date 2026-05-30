//! [`ProfileView`] — first-class profile / persona memory.
//!
//! Most "facts" an agent stores are episodic (something happened). A
//! *profile* is different: it's the small set of **stable, current**
//! attributes about a subject — preferences, traits, identity — that you
//! want every response conditioned on. Mem0/MemMachine/Memori all ship a
//! dedicated profile store for exactly this; mneme's is a
//! [`MaterializedView`] over `ProfileSet` events.
//!
//! ## Semantics
//!
//! - **Keyed by `(scope, attribute)`.** Setting the same attribute again
//!   supersedes the previous value.
//! - **Never overwrite (Hard Rule #2).** The prior value is retained in
//!   [`AttributeState::history`]; only the *current* pointer moves. Full
//!   history is queryable for audit ("what did we think their diet was
//!   last month?").
//! - **Scope is a boundary (Hard Rule #3).** Reads are exact-scope: a
//!   profile lookup for user A never returns user B's attributes. (Unlike
//!   memories, profile attributes are personal and not meant to be shared
//!   up a scope hierarchy without an explicit step.)
//! - **Rebuildable from the log (Hard Rule #4).** Drop the view, replay
//!   the `ProfileSet` events, same state.

use async_trait::async_trait;
use mneme_core::event::{Event, LogEntry};
use mneme_core::traits::MaterializedView;
use mneme_core::types::{Id, Scope};
use mneme_core::MnemeError;
use std::collections::HashMap;
use std::sync::RwLock;

/// One value an attribute held, with when/who set it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileValue {
    pub value: String,
    /// Set time in unix ms (from the event id — ULIDs are time-ordered).
    pub set_at_ms: u64,
    pub actor: Option<String>,
}

/// The current value of an attribute plus its superseded history
/// (oldest → newest, current excluded).
#[derive(Debug, Clone, Default)]
pub struct AttributeState {
    pub current: Option<ProfileValue>,
    pub history: Vec<ProfileValue>,
}

/// In-memory profile store. Cheap to rebuild from the log; small (a
/// handful of attributes per subject), so a plain map behind a `RwLock`
/// is plenty.
#[derive(Default)]
pub struct ProfileView {
    /// `scope_key → attribute → state`.
    subjects: RwLock<HashMap<String, HashMap<String, AttributeState>>>,
    last_checkpoint: RwLock<Option<Id>>,
}

/// Canonical string key for a scope. Profile reads are *exact-scope*, so
/// the key includes tenant + user + session verbatim.
fn scope_key(s: &Scope) -> String {
    format!(
        "{}\u{1f}{}\u{1f}{}",
        s.tenant,
        s.user.as_deref().unwrap_or(""),
        s.session.as_deref().unwrap_or("")
    )
}

impl ProfileView {
    pub fn new() -> Self {
        Self::default()
    }

    /// Current value of one attribute for a subject, if set + non-empty.
    pub fn get(&self, scope: &Scope, attribute: &str) -> Option<String> {
        let g = self.subjects.read().unwrap();
        g.get(&scope_key(scope))
            .and_then(|attrs| attrs.get(attribute))
            .and_then(|st| st.current.as_ref())
            .map(|v| v.value.clone())
    }

    /// All current attributes for a subject as `(attribute, value)`,
    /// sorted by attribute for deterministic output. Cleared attributes
    /// (empty current) are omitted.
    pub fn all(&self, scope: &Scope) -> Vec<(String, String)> {
        let g = self.subjects.read().unwrap();
        let mut out: Vec<(String, String)> = g
            .get(&scope_key(scope))
            .map(|attrs| {
                attrs
                    .iter()
                    .filter_map(|(k, st)| st.current.as_ref().map(|v| (k.clone(), v.value.clone())))
                    .collect()
            })
            .unwrap_or_default();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Full value history of one attribute (oldest → newest, including
    /// the current value as the last element).
    pub fn history(&self, scope: &Scope, attribute: &str) -> Vec<ProfileValue> {
        let g = self.subjects.read().unwrap();
        match g
            .get(&scope_key(scope))
            .and_then(|attrs| attrs.get(attribute))
        {
            Some(st) => {
                let mut h = st.history.clone();
                if let Some(c) = &st.current {
                    h.push(c.clone());
                }
                h
            }
            None => Vec::new(),
        }
    }

    /// Number of subjects (distinct scopes) with at least one attribute.
    pub fn subject_count(&self) -> usize {
        self.subjects.read().unwrap().len()
    }

    fn apply_set(
        &self,
        entry_id: Id,
        scope: &Scope,
        attribute: &str,
        value: &str,
        actor: &Option<String>,
    ) {
        let mut g = self.subjects.write().unwrap();
        let attrs = g.entry(scope_key(scope)).or_default();
        let st = attrs.entry(attribute.to_string()).or_default();
        // Move the current value (if any) into history — never overwrite.
        if let Some(prev) = st.current.take() {
            st.history.push(prev);
        }
        if value.is_empty() {
            // Empty value clears the attribute (history retained).
            st.current = None;
        } else {
            st.current = Some(ProfileValue {
                value: value.to_string(),
                set_at_ms: entry_id.timestamp_ms(),
                actor: actor.clone(),
            });
        }
    }
}

#[async_trait]
impl MaterializedView for ProfileView {
    fn name(&self) -> &str {
        "profile-view"
    }

    async fn apply(&self, entry: &LogEntry) -> Result<(), MnemeError> {
        if let Event::ProfileSet {
            scope,
            attribute,
            value,
            actor,
        } = &entry.event
        {
            self.apply_set(entry.id, scope, attribute, value, actor);
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
    use mneme_core::types::new_id;

    fn set(scope: &Scope, attr: &str, val: &str) -> LogEntry {
        LogEntry {
            id: new_id(),
            event: Event::ProfileSet {
                scope: scope.clone(),
                attribute: attr.into(),
                value: val.into(),
                actor: Some("tester".into()),
            },
        }
    }

    #[tokio::test]
    async fn set_and_get_current_value() {
        let v = ProfileView::new();
        let s = Scope {
            tenant: "t".into(),
            user: Some("alice".into()),
            session: None,
        };
        v.apply(&set(&s, "diet", "vegetarian")).await.unwrap();
        assert_eq!(v.get(&s, "diet").as_deref(), Some("vegetarian"));
        assert_eq!(v.get(&s, "locale"), None);
    }

    #[tokio::test]
    async fn resetting_supersedes_and_keeps_history() {
        let v = ProfileView::new();
        let s = Scope::global("t");
        v.apply(&set(&s, "diet", "vegetarian")).await.unwrap();
        v.apply(&set(&s, "diet", "vegan")).await.unwrap();
        assert_eq!(v.get(&s, "diet").as_deref(), Some("vegan"));
        let h = v.history(&s, "diet");
        let vals: Vec<&str> = h.iter().map(|p| p.value.as_str()).collect();
        assert_eq!(
            vals,
            vec!["vegetarian", "vegan"],
            "history oldest→newest incl current"
        );
    }

    #[tokio::test]
    async fn empty_value_clears_attribute_but_keeps_history() {
        let v = ProfileView::new();
        let s = Scope::global("t");
        v.apply(&set(&s, "diet", "vegan")).await.unwrap();
        v.apply(&set(&s, "diet", "")).await.unwrap();
        assert_eq!(v.get(&s, "diet"), None, "cleared");
        assert_eq!(
            v.history(&s, "diet").len(),
            1,
            "the vegan value is retained"
        );
    }

    #[tokio::test]
    async fn scope_isolation_is_exact() {
        let v = ProfileView::new();
        let alice = Scope {
            tenant: "t".into(),
            user: Some("alice".into()),
            session: None,
        };
        let bob = Scope {
            tenant: "t".into(),
            user: Some("bob".into()),
            session: None,
        };
        v.apply(&set(&alice, "diet", "vegan")).await.unwrap();
        assert_eq!(v.get(&alice, "diet").as_deref(), Some("vegan"));
        assert_eq!(
            v.get(&bob, "diet"),
            None,
            "bob must not see alice's profile"
        );
        assert_eq!(v.subject_count(), 1);
    }

    #[tokio::test]
    async fn all_returns_sorted_current_attributes() {
        let v = ProfileView::new();
        let s = Scope::global("t");
        v.apply(&set(&s, "timezone", "PT")).await.unwrap();
        v.apply(&set(&s, "diet", "vegan")).await.unwrap();
        v.apply(&set(&s, "locale", "en-GB")).await.unwrap();
        let all = v.all(&s);
        assert_eq!(
            all,
            vec![
                ("diet".to_string(), "vegan".to_string()),
                ("locale".to_string(), "en-GB".to_string()),
                ("timezone".to_string(), "PT".to_string()),
            ]
        );
    }

    #[tokio::test]
    async fn replay_rebuilds_identical_state() {
        let s = Scope::global("t");
        let events = vec![
            set(&s, "diet", "vegetarian"),
            set(&s, "diet", "vegan"),
            set(&s, "locale", "en-GB"),
        ];
        let a = ProfileView::new();
        let b = ProfileView::new();
        for e in &events {
            a.apply(e).await.unwrap();
        }
        for e in &events {
            b.apply(e).await.unwrap();
        }
        assert_eq!(a.all(&s), b.all(&s));
        assert_eq!(a.get(&s, "diet"), b.get(&s, "diet"));
    }

    #[tokio::test]
    async fn checkpoint_advances() {
        let v = ProfileView::new();
        assert!(v.checkpoint().await.unwrap().is_none());
        let e = set(&Scope::global("t"), "diet", "vegan");
        let id = e.id;
        v.apply(&e).await.unwrap();
        assert_eq!(v.checkpoint().await.unwrap(), Some(id));
    }
}
