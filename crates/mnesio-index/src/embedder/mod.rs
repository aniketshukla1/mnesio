//! Embedder implementations.
//!
//! [`MockEmbedder`] is the always-compiled, dependency-free reference
//! implementation. The whole workspace's tests run against it so they stay
//! fast, deterministic, and offline.
//!
//! [`fastembed::FastEmbedEmbedder`] is the production default — local ONNX
//! inference via the `fastembed` crate, behind the `fastembed` feature flag
//! so a `--no-default-features` build can skip the ort dependency tree.
//! `ApiEmbedder` (OpenAI-shaped HTTP) is a later slice. See the long-form
//! project description §4.5.

#[cfg(feature = "fastembed")]
pub mod fastembed;

#[cfg(feature = "fastembed")]
pub use self::fastembed::FastEmbedEmbedder;

use async_trait::async_trait;
use mnesio_core::{Embedder, MnesioError};

/// Deterministic stub embedder for tests and CI.
///
/// Produces a stable, L2-normalized embedding for each input string by
/// hashing the bytes with FNV-1a and expanding the seed through an
/// xorshift64 stream into `dim` floats. Identical inputs always produce
/// identical outputs; distinct inputs almost certainly produce distinct
/// outputs.
///
/// Not suitable for semantic retrieval — there is no meaningful proximity
/// between embeddings of related concepts. Use it for round-trip tests of
/// the index and write pipeline; reach for `FastEmbedEmbedder` or
/// `ApiEmbedder` when actual semantic quality matters.
pub struct MockEmbedder {
    dim: usize,
    model_id: String,
}

impl MockEmbedder {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            // Two MockEmbedders with different `dim`s produce incompatible
            // vector spaces, so the id must encode the dim. The startup
            // mismatch check (later slice) compares this string verbatim.
            model_id: format!("mock-v1-d{dim}"),
        }
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, MnesioError> {
        Ok(texts
            .iter()
            .map(|t| seeded_unit_vector(t, self.dim))
            .collect())
    }
}

/// FNV-1a 64-bit hash → xorshift64 stream → L2-normalized `f32` vector.
fn seeded_unit_vector(text: &str, dim: usize) -> Vec<f32> {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // xorshift64 has a degenerate cycle at state == 0; guard against it.
    let mut state = h | 1;
    let mut v = Vec::with_capacity(dim);
    for _ in 0..dim {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bits = (state >> 32) as u32;
        // Map u32 → [-1.0, 1.0). Spread isn't perfectly uniform but it
        // doesn't need to be: this exists only so distinct strings produce
        // distinct unit vectors.
        v.push((bits as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deterministic_for_same_input() {
        let e = MockEmbedder::new(8);
        let a = e.embed(&["hello".into()]).await.unwrap();
        let b = e.embed(&["hello".into()]).await.unwrap();
        assert_eq!(a, b);
        assert_eq!(a[0].len(), 8);
    }

    #[tokio::test]
    async fn distinguishes_distinct_inputs() {
        let e = MockEmbedder::new(8);
        let a = e.embed(&["hello".into()]).await.unwrap();
        let b = e.embed(&["world".into()]).await.unwrap();
        assert_ne!(a[0], b[0]);
    }

    #[tokio::test]
    async fn produces_unit_vectors() {
        let e = MockEmbedder::new(16);
        let v = &e.embed(&["anything".into()]).await.unwrap()[0];
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
    }

    #[test]
    fn model_id_is_dim_aware() {
        let a = MockEmbedder::new(8);
        let b = MockEmbedder::new(16);
        assert_ne!(
            a.model_id(),
            b.model_id(),
            "different dims must produce different model_ids"
        );
    }

    #[tokio::test]
    async fn dim_reported_matches_output() {
        let e = MockEmbedder::new(32);
        assert_eq!(e.dim(), 32);
        let v = e.embed(&["x".into()]).await.unwrap();
        assert_eq!(v[0].len(), 32);
    }
}
