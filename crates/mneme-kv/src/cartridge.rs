//! The cartridge view + the tensor-backend seam.
//!
//! A [`Cartridge`] is a versioned, audited, derived artifact: it remembers the
//! [`CartridgeKey`] it was compiled under, the ids of the memories that went
//! into it, and the opaque blob a [`KvBackend`] produced. [`compile`] is the
//! pure pipeline: open key-sealed memories, hand the surviving plaintext to the
//! backend, and wrap the result.

use async_trait::async_trait;
use mneme_core::types::MemoryRef;
use mneme_privacy::{Keyring, SealedBox};
use serde::{Deserialize, Serialize};

/// Identity of the model/runtime a cartridge is valid for, plus the log prefix
/// it was built from. A cartridge built under one key is meaningless under
/// another — a model swap, a requant, or a rope-config change all change the
/// tensor layout, and a newer `log_head` means newer knowledge. Equality of
/// keys is what lets the store detect "this cartridge is stale, recompile".
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CartridgeKey {
    pub model_id: String,
    pub quant: String,
    pub rope_config: String,
    /// The id of the last log entry whose state this cartridge reflects
    /// (stringified ULID). `None` = built from an empty/unknown prefix.
    pub log_head: Option<String>,
}

impl CartridgeKey {
    pub fn new(
        model_id: impl Into<String>,
        quant: impl Into<String>,
        rope_config: impl Into<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            quant: quant.into(),
            rope_config: rope_config.into(),
            log_head: None,
        }
    }

    pub fn at_head(mut self, log_head: Option<String>) -> Self {
        self.log_head = log_head;
        self
    }

    /// True if `self` and `other` target the same model/runtime, ignoring the
    /// log prefix — i.e. a cartridge under `other` could be *recompiled* to
    /// serve `self` (same tensor layout, possibly newer knowledge).
    pub fn same_runtime(&self, other: &CartridgeKey) -> bool {
        self.model_id == other.model_id
            && self.quant == other.quant
            && self.rope_config == other.rope_config
    }
}

/// A memory as it enters compilation: its id + the [`SealedBox`] holding its
/// (key-encrypted) content. The cartridge is compiled from *sealed* memories so
/// that destroying a subject's key removes them from any recompile — the
/// crypto-shred reconciliation.
#[derive(Debug, Clone)]
pub struct SealedMemory {
    pub id: MemoryRef,
    pub sealed: SealedBox,
}

/// What a backend returns for a query against a cartridge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvAnswer {
    /// The answer text the cartridge produced (empty = no answer).
    pub text: String,
    /// Simulated/real wall-clock cost of answering *from the cartridge*, in
    /// microseconds. The Phase-12 latency claim compares this against
    /// text-context retrieval.
    pub latency_us: u64,
    /// Whether the cartridge actually had the knowledge to answer.
    pub answered: bool,
}

/// The tensor-backend seam (Hard Rule #7). A real impl compiles plaintext into
/// a model's KV cache and answers from it; [`FakeKvBackend`] models that
/// deterministically for tests.
#[async_trait]
pub trait KvBackend: Send + Sync {
    /// Stable id of the underlying model/runtime — must match
    /// [`CartridgeKey::model_id`] the caller compiles under.
    fn model_id(&self) -> &str;

    /// Compile the given plaintext memories into an opaque KV blob. The blob is
    /// the only thing persisted as the cartridge's tensor; it must be fully
    /// determined by `contents` (so a recompile from a shrunk corpus produces a
    /// strictly smaller/erased blob).
    async fn compile_blob(&self, contents: &[String]) -> Vec<u8>;

    /// Answer a query *from a compiled blob*. Must not consult anything outside
    /// the blob — that's what makes the erasure guarantee meaningful.
    async fn answer(&self, blob: &[u8], query: &str) -> KvAnswer;
}

/// Why compilation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// The backend's model id doesn't match the key's model id.
    ModelMismatch { key: String, backend: String },
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::ModelMismatch { key, backend } => write!(
                f,
                "cartridge key targets model {key:?} but backend is {backend:?}"
            ),
        }
    }
}

impl std::error::Error for CompileError {}

/// Lifecycle state of a cartridge in the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CartridgeStatus {
    /// Compiled but not yet gated — inert.
    Compiled,
    /// Passed the gate and serving queries.
    Active,
    /// Failed the gate, or superseded by a newer version — inert.
    Rejected,
}

/// A compiled KV cartridge: a derived, versioned, audited view of the log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cartridge {
    pub key: CartridgeKey,
    pub version: u32,
    pub status: CartridgeStatus,
    /// Audit: exactly which memories were compiled in (post key-open). A
    /// memory whose key was destroyed before compile is absent here.
    pub member_ids: Vec<MemoryRef>,
    /// The opaque tensor blob the backend produced.
    pub blob: Vec<u8>,
}

impl Cartridge {
    /// How many memories actually made it into the blob.
    pub fn member_count(&self) -> usize {
        self.member_ids.len()
    }

    /// True if `id` contributed to this cartridge.
    pub fn contains(&self, id: MemoryRef) -> bool {
        self.member_ids.contains(&id)
    }
}

/// Compile a cartridge from key-sealed memories.
///
/// Each [`SealedMemory`] is opened through `keyring`; any whose key has been
/// destroyed (forgotten) returns `None` from `open` and is silently dropped —
/// **this is the crypto-shred reconciliation**: erased subjects can't enter a
/// recompile. The surviving plaintext is handed to the backend, and the result
/// is wrapped as a `Compiled` (not yet gated) cartridge under `key`.
pub async fn compile<C: mneme_privacy::Cipher>(
    backend: &dyn KvBackend,
    keyring: &Keyring<C>,
    key: CartridgeKey,
    version: u32,
    members: &[SealedMemory],
) -> Result<Cartridge, CompileError> {
    if backend.model_id() != key.model_id {
        return Err(CompileError::ModelMismatch {
            key: key.model_id.clone(),
            backend: backend.model_id().to_string(),
        });
    }

    let mut member_ids = Vec::new();
    let mut contents = Vec::new();
    for m in members {
        // A forgotten subject's box no longer opens — it vanishes from the
        // recompiled cartridge entirely.
        if let Some(text) = keyring.open(&m.sealed) {
            member_ids.push(m.id);
            contents.push(text);
        }
    }

    let blob = backend.compile_blob(&contents).await;
    Ok(Cartridge {
        key,
        version,
        status: CartridgeStatus::Compiled,
        member_ids,
        blob,
    })
}

/// Deterministic, dependency-free [`KvBackend`] for tests + offline demos.
///
/// Models the KV blob as the JSON-encoded list of member texts. `answer` does a
/// case-insensitive substring scan of the blob and reports a low, fixed latency
/// — far below the simulated text-retrieval cost the demo compares against, so
/// the "lower latency" claim is observable. Crucially, `answer` reads *only*
/// the blob, so once a text is gone from the blob (erased subject), the
/// cartridge genuinely can't answer from it.
pub struct FakeKvBackend {
    model_id: String,
    /// Simulated per-answer latency from the cartridge.
    answer_latency_us: u64,
}

impl FakeKvBackend {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            answer_latency_us: 50,
        }
    }

    pub fn with_latency_us(mut self, us: u64) -> Self {
        self.answer_latency_us = us;
        self
    }
}

#[async_trait]
impl KvBackend for FakeKvBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn compile_blob(&self, contents: &[String]) -> Vec<u8> {
        // The blob is fully determined by its contents — a recompile from
        // fewer memories yields a strictly smaller blob (erasure is real).
        serde_json::to_vec(contents).unwrap_or_default()
    }

    async fn answer(&self, blob: &[u8], query: &str) -> KvAnswer {
        let contents: Vec<String> = serde_json::from_slice(blob).unwrap_or_default();
        let q = query.to_ascii_lowercase();
        // Match the query's salient terms against blob contents. A hit returns
        // the first matching member text.
        let terms: Vec<&str> = q.split_whitespace().filter(|t| t.len() > 3).collect();
        let hit = contents.iter().find(|c| {
            let lc = c.to_ascii_lowercase();
            terms.iter().any(|t| lc.contains(t))
        });
        match hit {
            Some(text) => KvAnswer {
                text: text.clone(),
                latency_us: self.answer_latency_us,
                answered: true,
            },
            None => KvAnswer {
                text: String::new(),
                latency_us: self.answer_latency_us,
                answered: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mneme_core::types::new_id;

    fn mref() -> MemoryRef {
        MemoryRef(new_id())
    }

    fn sealed(keyring: &Keyring, subject: &str, id: MemoryRef, text: &str) -> SealedMemory {
        SealedMemory {
            id,
            sealed: keyring.seal(subject, text.as_bytes()).unwrap(),
        }
    }

    fn key() -> CartridgeKey {
        CartridgeKey::new("fake-llm-v1", "q8", "rope-default")
    }

    #[tokio::test]
    async fn compile_includes_all_unforgotten_members() {
        let kr = Keyring::new();
        let backend = FakeKvBackend::new("fake-llm-v1");
        let (m1, m2) = (mref(), mref());
        let members = vec![
            sealed(&kr, "alice", m1, "alice prefers window seats"),
            sealed(&kr, "bob", m2, "bob is allergic to peanuts"),
        ];
        let c = compile(&backend, &kr, key(), 1, &members).await.unwrap();
        assert_eq!(c.member_count(), 2);
        assert!(c.contains(m1) && c.contains(m2));
        assert_eq!(c.status, CartridgeStatus::Compiled);
    }

    #[tokio::test]
    async fn cartridge_answers_from_its_blob() {
        let kr = Keyring::new();
        let backend = FakeKvBackend::new("fake-llm-v1");
        let m1 = mref();
        let members = vec![sealed(&kr, "alice", m1, "alice prefers window seats")];
        let c = compile(&backend, &kr, key(), 1, &members).await.unwrap();
        let a = backend
            .answer(&c.blob, "what seats does alice prefer")
            .await;
        assert!(a.answered);
        assert!(a.text.contains("window seats"));
        assert!(a.latency_us > 0);
    }

    #[tokio::test]
    async fn model_mismatch_is_rejected() {
        let kr = Keyring::new();
        let backend = FakeKvBackend::new("some-other-model");
        let err = compile(&backend, &kr, key(), 1, &[]).await.unwrap_err();
        assert!(matches!(err, CompileError::ModelMismatch { .. }));
    }

    #[tokio::test]
    async fn forgotten_subject_drops_out_of_recompile() {
        // The crypto-shred reconciliation, at the compile level.
        let kr = Keyring::new();
        let backend = FakeKvBackend::new("fake-llm-v1");
        let (m1, m2) = (mref(), mref());
        let members = vec![
            sealed(&kr, "alice", m1, "alice prefers window seats"),
            sealed(&kr, "bob", m2, "bob is allergic to peanuts"),
        ];

        // v1 sees both.
        let v1 = compile(&backend, &kr, key(), 1, &members).await.unwrap();
        assert_eq!(v1.member_count(), 2);
        assert!(backend.answer(&v1.blob, "alice window").await.answered);

        // Forget alice, recompile from the SAME sealed inputs.
        kr.forget("alice");
        let v2 = compile(&backend, &kr, key(), 2, &members).await.unwrap();
        assert_eq!(v2.member_count(), 1, "alice's sealed box no longer opens");
        assert!(v2.contains(m2) && !v2.contains(m1));

        // The recompiled cartridge cannot answer about alice from its blob.
        let a = backend.answer(&v2.blob, "alice window seats").await;
        assert!(
            !a.answered,
            "erased subject is unrecoverable from the cartridge"
        );
        // …but bob still works.
        assert!(backend.answer(&v2.blob, "bob peanuts").await.answered);
    }

    #[test]
    fn same_runtime_ignores_log_head() {
        let a = key().at_head(Some("01AAA".into()));
        let b = key().at_head(Some("01BBB".into()));
        assert!(a.same_runtime(&b), "same model/quant/rope → recompilable");
        let other = CartridgeKey::new("fake-llm-v1", "q4", "rope-default");
        assert!(
            !a.same_runtime(&other),
            "different quant → different layout"
        );
    }
}
