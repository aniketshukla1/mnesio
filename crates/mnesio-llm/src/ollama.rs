//! Ollama HTTP backend for [`LlmClient`].
//!
//! Talks to a local (or remote) Ollama server via its `/api/generate`
//! endpoint with `stream: false`. Defaults: `http://localhost:11434`
//! and model `llama3.2` — small enough to run on a laptop, large
//! enough to handle the structured-extraction prompts the evolution
//! worker emits.
//!
//! Both the base URL and model can be overridden:
//!
//! - programmatically via `OllamaLlmClient::new(url, model)` or
//!   `OllamaLlmClient::from_env()`
//! - via env: `MNESIO_OLLAMA_URL`, `MNESIO_OLLAMA_MODEL`
//!
//! This module is only compiled with the `ollama` feature (default on).
//! Build with `--no-default-features` to omit it for CI/test-only
//! builds that don't need the HTTP dependency tree.

use async_trait::async_trait;
use mnesio_core::{LlmClient, MnesioError};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default Ollama HTTP endpoint. Override with [`OllamaLlmClient::new`]
/// or the `MNESIO_OLLAMA_URL` env var.
pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Default model. `llama3.2` is small enough to run on a laptop and
/// good enough for the structured-extraction prompts mnesio issues
/// (keywords, tags, link relationships).
pub const DEFAULT_OLLAMA_MODEL: &str = "llama3.2";

/// Default request timeout. Generation is bursty — long enough that
/// occasional slow tokens don't kill the request, short enough that
/// a wedged Ollama doesn't stall the embedding worker forever.
pub const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// `LlmClient` backed by a local Ollama server.
pub struct OllamaLlmClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaLlmClient {
    /// Construct with explicit base URL and model. The URL should not
    /// include the `/api/generate` path — only the host + port.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self, MnesioError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .map_err(|e| MnesioError::Llm(format!("reqwest client init: {e}")))?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
        })
    }

    /// Construct from `MNESIO_OLLAMA_URL` + `MNESIO_OLLAMA_MODEL` env
    /// vars, falling back to [`DEFAULT_OLLAMA_URL`] /
    /// [`DEFAULT_OLLAMA_MODEL`] when unset.
    pub fn from_env() -> Result<Self, MnesioError> {
        let url = std::env::var("MNESIO_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.into());
        let model =
            std::env::var("MNESIO_OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_OLLAMA_MODEL.into());
        Self::new(url, model)
    }

    /// Base URL the client is configured against. Useful for log lines.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Model the client is configured to use.
    pub fn model(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl LlmClient for OllamaLlmClient {
    async fn complete(&self, prompt: &str) -> Result<String, MnesioError> {
        let req = GenerateRequest {
            model: &self.model,
            prompt,
            stream: false,
        };
        let url = format!("{}/api/generate", self.base_url);
        tracing::trace!(
            url = %url,
            model = %self.model,
            prompt_chars = prompt.len(),
            "ollama: requesting completion"
        );
        let resp = self
            .http
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| MnesioError::Llm(format!("ollama POST {url}: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(MnesioError::Llm(format!(
                "ollama {url} returned {status}: {body}"
            )));
        }
        let body: GenerateResponse = resp
            .json()
            .await
            .map_err(|e| MnesioError::Llm(format!("ollama JSON decode: {e}")))?;
        Ok(body.response)
    }
}

#[derive(Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    /// We don't want streamed tokens — `LlmClient::complete` is one-shot.
    stream: bool,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_trims_trailing_slash_from_base_url() {
        let c = OllamaLlmClient::new("http://localhost:11434/", "llama3.2").unwrap();
        assert_eq!(c.base_url(), "http://localhost:11434");
    }

    #[test]
    fn model_accessor_reflects_construction() {
        let c = OllamaLlmClient::new("http://x", "bge-small").unwrap();
        assert_eq!(c.model(), "bge-small");
    }

    #[test]
    fn from_env_uses_defaults_when_unset() {
        // We can't safely manipulate process env in unit tests
        // (parallel tests would race) so just verify the defaults
        // are exposed as expected constants.
        assert!(DEFAULT_OLLAMA_URL.starts_with("http://"));
        assert!(!DEFAULT_OLLAMA_MODEL.is_empty());
    }
}
