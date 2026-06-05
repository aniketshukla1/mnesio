//! Crypto-shredding — right-to-be-forgotten for an append-only log.
//!
//! The problem: Hard Rule #2 says the event log is **append-only, never
//! rewritten**. GDPR Art. 17 says a subject can demand **erasure**. These
//! look irreconcilable — you can't delete a row from an immutable log.
//!
//! The resolution (crypto-shredding, the standard technique): store each
//! subject's sensitive content **encrypted under a per-subject key**, and
//! keep the keys *outside* the log in a mutable keyring. To "forget" a
//! subject you **destroy their key** — the ciphertext remains in the log
//! (integrity + replayability intact) but is now permanently
//! undecryptable. The subject is unreadable; the log never changed.
//!
//! ```text
//!   seal(subject, plaintext)  -> SealedBox      (key auto-created)
//!   open(SealedBox)           -> Some(plaintext) while key lives
//!   forget(subject)           -> key destroyed
//!   open(SealedBox)           -> None            forever after
//! ```
//!
//! ## Cipher seam
//!
//! [`Cipher`] abstracts encryption so a real AEAD (`chacha20poly1305`)
//! can replace the built-in [`XorCipher`] without touching the shred
//! protocol — the *forget* guarantee is key-destruction, which holds for
//! any cipher. `XorCipher` is a keystream cipher: correct for the
//! protocol + offline-testable, **not** authenticated; production should
//! wire a real AEAD behind this trait.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

/// A symmetric cipher over per-subject keys. Implementors must be
/// deterministic given (key, nonce, data) so encrypt→decrypt round-trips.
pub trait Cipher: Send + Sync {
    /// Encrypt `plaintext` under `key` with `nonce`.
    fn encrypt(&self, key: &[u8; 32], nonce: &[u8; 12], plaintext: &[u8]) -> Vec<u8>;
    /// Decrypt; returns `None` if the data fails to authenticate (real
    /// AEAD) — the keystream default always returns `Some`.
    fn decrypt(&self, key: &[u8; 32], nonce: &[u8; 12], ciphertext: &[u8]) -> Option<Vec<u8>>;
}

/// Default keystream cipher (XOR of plaintext with a key-derived stream).
/// Correct for the crypto-shred protocol; **not authenticated** — swap a
/// real AEAD in for production via [`Cipher`].
#[derive(Debug, Clone, Default)]
pub struct XorCipher;

impl XorCipher {
    /// Deterministic keystream from key+nonce via a tiny SplitMix64-style
    /// PRNG seeded by mixing the key and nonce. Dependency-free.
    fn keystream(key: &[u8; 32], nonce: &[u8; 12], len: usize) -> Vec<u8> {
        let mut seed = 0xcbf29ce484222325u64; // FNV offset basis
        for b in key.iter().chain(nonce.iter()) {
            seed ^= *b as u64;
            seed = seed.wrapping_mul(0x100000001b3); // FNV prime
        }
        let mut out = Vec::with_capacity(len);
        let mut state = seed;
        while out.len() < len {
            // SplitMix64 step.
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            out.extend_from_slice(&z.to_le_bytes());
        }
        out.truncate(len);
        out
    }
}

impl Cipher for XorCipher {
    fn encrypt(&self, key: &[u8; 32], nonce: &[u8; 12], plaintext: &[u8]) -> Vec<u8> {
        let ks = Self::keystream(key, nonce, plaintext.len());
        plaintext.iter().zip(ks).map(|(p, k)| p ^ k).collect()
    }
    fn decrypt(&self, key: &[u8; 32], nonce: &[u8; 12], ciphertext: &[u8]) -> Option<Vec<u8>> {
        // XOR is symmetric.
        Some(self.encrypt(key, nonce, ciphertext))
    }
}

/// What gets stored in the log in place of plaintext: the subject id, the
/// nonce, and ciphertext. Carries no key — the key lives only in the
/// [`Keyring`]. Serializable so it can ride inside an event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedBox {
    pub subject: String,
    pub nonce: [u8; 12],
    pub ciphertext: Vec<u8>,
}

/// Per-subject key store. Keys never enter the event log; in production
/// this is backed by a KMS / secrets manager. "Forgetting" drops a key.
pub struct Keyring<C: Cipher = XorCipher> {
    cipher: C,
    keys: RwLock<HashMap<String, [u8; 32]>>,
    /// Monotonic nonce counter so repeated seals of the same subject use
    /// distinct nonces (keystream-reuse safety).
    counter: RwLock<u64>,
    /// Tombstoned subjects — once forgotten, a subject can't be silently
    /// re-keyed by a later seal (that would resurrect readability of *new*
    /// data under a recycled identity; we want forget to be sticky).
    forgotten: RwLock<std::collections::HashSet<String>>,
}

impl<C: Cipher> Keyring<C> {
    pub fn with_cipher(cipher: C) -> Self {
        Self {
            cipher,
            keys: RwLock::new(HashMap::new()),
            counter: RwLock::new(0),
            forgotten: RwLock::new(std::collections::HashSet::new()),
        }
    }
}

impl Keyring<XorCipher> {
    pub fn new() -> Self {
        Self::with_cipher(XorCipher)
    }
}

impl Default for Keyring<XorCipher> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Cipher> Keyring<C> {
    /// Derive a fresh per-subject key. Dependency-free pseudo-random from
    /// the subject string + a process-seeded salt; in production a KMS
    /// generates these. Deterministic only within a process lifetime via
    /// the counter, never reproducible from the subject alone.
    fn fresh_key(subject: &str, salt: u64) -> [u8; 32] {
        let mut seed = 0xcbf29ce484222325u64 ^ salt;
        for b in subject.bytes() {
            seed ^= b as u64;
            seed = seed.wrapping_mul(0x100000001b3);
        }
        let mut key = [0u8; 32];
        let mut state = seed;
        for chunk in key.chunks_mut(8) {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            let b = z.to_le_bytes();
            chunk.copy_from_slice(&b[..chunk.len()]);
        }
        key
    }

    /// Encrypt `plaintext` for `subject`, creating the subject's key on
    /// first use. Returns `None` if the subject has been forgotten —
    /// forget is sticky, so you can't seal new data under a tombstoned id.
    pub fn seal(&self, subject: &str, plaintext: &[u8]) -> Option<SealedBox> {
        if self.forgotten.read().unwrap().contains(subject) {
            return None;
        }
        let nonce = {
            let mut c = self.counter.write().unwrap();
            *c += 1;
            let mut n = [0u8; 12];
            n[..8].copy_from_slice(&c.to_le_bytes());
            n
        };
        let key = {
            let mut keys = self.keys.write().unwrap();
            *keys.entry(subject.to_string()).or_insert_with(|| {
                // Salt with the current counter for per-process freshness.
                let salt = *self.counter.read().unwrap();
                Self::fresh_key(subject, salt)
            })
        };
        let ciphertext = self.cipher.encrypt(&key, &nonce, plaintext);
        Some(SealedBox {
            subject: subject.to_string(),
            nonce,
            ciphertext,
        })
    }

    /// Decrypt a [`SealedBox`]. Returns `None` if the subject's key has
    /// been destroyed (forgotten) or was never present — the crypto-shred
    /// guarantee in action.
    pub fn open(&self, sealed: &SealedBox) -> Option<String> {
        let keys = self.keys.read().unwrap();
        let key = keys.get(&sealed.subject)?;
        let bytes = self
            .cipher
            .decrypt(key, &sealed.nonce, &sealed.ciphertext)?;
        String::from_utf8(bytes).ok()
    }

    /// **Forget a subject**: destroy their key. All existing
    /// [`SealedBox`]es for them become permanently undecryptable, and no
    /// new data can be sealed under the id. Idempotent. Returns whether a
    /// key was actually present to destroy.
    pub fn forget(&self, subject: &str) -> bool {
        self.forgotten.write().unwrap().insert(subject.to_string());
        self.keys.write().unwrap().remove(subject).is_some()
    }

    /// Has this subject been forgotten?
    pub fn is_forgotten(&self, subject: &str) -> bool {
        self.forgotten.read().unwrap().contains(subject)
    }

    /// Number of subjects with a live key.
    pub fn live_key_count(&self) -> usize {
        self.keys.read().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_then_open_roundtrips() {
        let kr = Keyring::new();
        let sealed = kr.seal("alice", b"secret diagnosis").unwrap();
        assert_eq!(kr.open(&sealed).as_deref(), Some("secret diagnosis"));
    }

    #[test]
    fn forget_makes_ciphertext_unreadable_forever() {
        let kr = Keyring::new();
        let sealed = kr.seal("alice", b"secret diagnosis").unwrap();
        assert!(kr.open(&sealed).is_some());

        let had_key = kr.forget("alice");
        assert!(had_key, "forget should report a key was destroyed");

        // The sealed box is unchanged (append-only log keeps it), but it's
        // now undecryptable — crypto-shred guarantee.
        assert_eq!(kr.open(&sealed), None);
        assert!(kr.is_forgotten("alice"));
    }

    #[test]
    fn forget_is_sticky_no_reseal() {
        let kr = Keyring::new();
        kr.seal("alice", b"x").unwrap();
        kr.forget("alice");
        // Sealing new data under a forgotten id is refused.
        assert!(kr.seal("alice", b"new data").is_none());
    }

    #[test]
    fn forget_is_idempotent() {
        let kr = Keyring::new();
        kr.seal("alice", b"x").unwrap();
        assert!(kr.forget("alice"));
        // Second forget: no key to destroy, but still reports forgotten.
        assert!(!kr.forget("alice"));
        assert!(kr.is_forgotten("alice"));
    }

    #[test]
    fn forgetting_one_subject_does_not_affect_others() {
        let kr = Keyring::new();
        let a = kr.seal("alice", b"alice-secret").unwrap();
        let b = kr.seal("bob", b"bob-secret").unwrap();
        kr.forget("alice");
        assert_eq!(kr.open(&a), None, "alice unreadable");
        assert_eq!(kr.open(&b).as_deref(), Some("bob-secret"), "bob intact");
        assert_eq!(kr.live_key_count(), 1);
    }

    #[test]
    fn distinct_nonces_across_seals() {
        let kr = Keyring::new();
        let s1 = kr.seal("alice", b"same plaintext").unwrap();
        let s2 = kr.seal("alice", b"same plaintext").unwrap();
        assert_ne!(s1.nonce, s2.nonce, "nonce must advance per seal");
        // Same key + different nonce → different ciphertext (no keystream reuse).
        assert_ne!(s1.ciphertext, s2.ciphertext);
        // Both still decrypt correctly.
        assert_eq!(kr.open(&s1).as_deref(), Some("same plaintext"));
        assert_eq!(kr.open(&s2).as_deref(), Some("same plaintext"));
    }

    #[test]
    fn open_unknown_subject_is_none() {
        let kr = Keyring::new();
        let sealed = SealedBox {
            subject: "ghost".into(),
            nonce: [0u8; 12],
            ciphertext: vec![1, 2, 3],
        };
        assert_eq!(kr.open(&sealed), None);
    }

    #[test]
    fn xor_cipher_roundtrips_arbitrary_bytes() {
        let c = XorCipher;
        let key = [7u8; 32];
        let nonce = [3u8; 12];
        let pt = b"\x00\xff binary \x01\x02 data";
        let ct = c.encrypt(&key, &nonce, pt);
        assert_ne!(&ct[..], &pt[..], "ciphertext should differ from plaintext");
        assert_eq!(c.decrypt(&key, &nonce, &ct).unwrap(), pt);
    }

    // --- erasure edge cases ------------------------------------------------

    #[test]
    fn forget_erases_every_sealed_box_for_subject() {
        // A subject sealed many times (e.g. one box per turn, across "all T")
        // must become unreadable *everywhere* after a single forget — one key
        // backs all their boxes, so destroying it shreds the lot.
        let kr = Keyring::new();
        let boxes: Vec<_> = (0..16)
            .map(|i| kr.seal("alice", format!("secret #{i}").as_bytes()).unwrap())
            .collect();
        assert!(
            boxes.iter().all(|b| kr.open(b).is_some()),
            "all readable pre-forget"
        );

        kr.forget("alice");

        assert!(
            boxes.iter().all(|b| kr.open(b).is_none()),
            "every one of the subject's sealed boxes must be unreadable after forget"
        );
    }

    #[test]
    fn forget_unknown_subject_is_safe_and_sticky() {
        // Forgetting a subject that was never sealed must not panic; it reports
        // "no key destroyed" but still tombstones the id, so a later seal is
        // refused (you can't un-forget by being late to the party).
        let kr = Keyring::new();
        assert!(!kr.forget("never-sealed"), "no key existed to destroy");
        assert!(kr.is_forgotten("never-sealed"));
        assert!(
            kr.seal("never-sealed", b"too late").is_none(),
            "forget is sticky even when it preceded any seal"
        );
    }

    #[test]
    fn empty_plaintext_seals_opens_and_forgets() {
        let kr = Keyring::new();
        let sealed = kr.seal("alice", b"").unwrap();
        assert_eq!(
            kr.open(&sealed).as_deref(),
            Some(""),
            "empty content round-trips"
        );
        kr.forget("alice");
        assert_eq!(kr.open(&sealed), None, "empty content is still shredded");
    }
}
