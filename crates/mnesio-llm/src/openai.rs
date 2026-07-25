//! OpenAI-compatible chat backend for [`LlmClient`].
//!
//! Talks to any `/v1/chat/completions` endpoint that follows the OpenAI
//! wire format. That single shape reaches three things mnesio cares about:
//!
//! - **OpenRouter** (`https://openrouter.ai/api/v1`) — the default. One key
//!   reaches both Claude (`anthropic/claude-3.5-sonnet`, …) and GPT
//!   (`openai/gpt-4o-mini`, …) models, which is exactly what the published
//!   LoCoMo / LongMemEval run needs (a frontier-class answerer + judge).
//! - **OpenAI** directly (`https://api.openai.com/v1`).
//! - Any self-hosted OpenAI-compatible gateway.
//!
//! The API key is **read from the environment only** — never a constructor
//! argument that could get logged or committed. Resolution order:
//! `OPENROUTER_API_KEY` → `OPENAI_API_KEY` → `MNESIO_OPENAI_API_KEY`.
//!
//! Overrides (all env, with sensible defaults):
//!
//! - `MNESIO_OPENAI_BASE_URL` — default [`DEFAULT_OPENAI_BASE_URL`] (OpenRouter).
//! - `MNESIO_OPENAI_MODEL`    — default [`DEFAULT_OPENAI_MODEL`].
//!
//! This module is only compiled with the `openai` feature. It shares the
//! `reqwest` + `serde_json` dependency tree with the `ollama` backend.

use async_trait::async_trait;
use mnesio_core::{LlmClient, MnesioError};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default endpoint — OpenRouter's OpenAI-compatible base. A single
/// OpenRouter key then reaches Claude *and* GPT models by id.
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Default model. A cheap, reliable answerer/judge; override with
/// `MNESIO_OPENAI_MODEL` (e.g. `anthropic/claude-3.5-sonnet` for a
/// frontier-class judge, or `openai/gpt-4o` for the paper-standard one).
pub const DEFAULT_OPENAI_MODEL: &str = "openai/gpt-4o-mini";

/// Request timeout. Frontier models can take a while on long-context
/// answers; generous enough to not clip them, bounded so a wedged
/// endpoint can't stall a whole eval run forever.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// `LlmClient` backed by an OpenAI-compatible chat endpoint.
pub struct OpenAiCompatClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
}

impl OpenAiCompatClient {
    /// Construct with explicit base URL, model, and key. The base URL
    /// should be the API root (…`/v1`), without the `/chat/completions`
    /// suffix. Prefer [`OpenAiCompatClient::from_env`] so the key is only
    /// ever read from the environment.
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self, MnesioError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .map_err(|e| MnesioError::Llm(format!("reqwest client init: {e}")))?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            api_key: api_key.into(),
        })
    }

    /// Construct from the environment. The key is resolved from
    /// `OPENROUTER_API_KEY` → `OPENAI_API_KEY` → `MNESIO_OPENAI_API_KEY`;
    /// base URL and model from `MNESIO_OPENAI_BASE_URL` /
    /// `MNESIO_OPENAI_MODEL` (with defaults). Returns an actionable error
    /// if no key is set — the one thing that can't have a default.
    pub fn from_env() -> Result<Self, MnesioError> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .or_else(|_| std::env::var("MNESIO_OPENAI_API_KEY"))
            .map_err(|_| {
                MnesioError::Llm(
                    "no API key found: set OPENROUTER_API_KEY (or OPENAI_API_KEY / \
                     MNESIO_OPENAI_API_KEY) in the environment. The key is never \
                     read from a file or flag."
                        .into(),
                )
            })?;
        let base_url = std::env::var("MNESIO_OPENAI_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_OPENAI_BASE_URL.into());
        let model =
            std::env::var("MNESIO_OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.into());
        Self::new(base_url, model, api_key)
    }

    /// Base URL the client is configured against (no key). For log lines.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Model the client is configured to use.
    pub fn model(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl LlmClient for OpenAiCompatClient {
    async fn complete(&self, prompt: &str) -> Result<String, MnesioError> {
        let req = ChatRequest {
            model: &self.model,
            messages: vec![ChatMessage {
                role: "user",
                content: prompt,
            }],
            // Deterministic decoding so the same eval reproduces the same
            // answer/judgement across runs.
            temperature: 0.0,
            stream: false,
        };
        let url = format!("{}/chat/completions", self.base_url);
        tracing::trace!(
            url = %url,
            model = %self.model,
            prompt_chars = prompt.len(),
            "openai-compat: requesting completion"
        );
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            // OpenRouter uses these for attribution/leaderboards; harmless
            // elsewhere. No PII — just identifies the caller as mnesio.
            .header("HTTP-Referer", "https://github.com/mnesio/mnesio")
            .header("X-Title", "mnesio-bench")
            .json(&req)
            .send()
            .await
            .map_err(|e| MnesioError::Llm(format!("openai-compat POST {url}: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(MnesioError::Llm(format!(
                "openai-compat {url} returned {status}: {body}"
            )));
        }
        let body: ChatResponse = resp
            .json()
            .await
            .map_err(|e| MnesioError::Llm(format!("openai-compat JSON decode: {e}")))?;
        body.choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| MnesioError::Llm("openai-compat: response had no choices".into()))
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    /// One-shot — `LlmClient::complete` doesn't stream.
    stream: bool,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: RespMessage,
}

#[derive(Deserialize)]
struct RespMessage {
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_trims_trailing_slash_from_base_url() {
        let c = OpenAiCompatClient::new(
            "https://openrouter.ai/api/v1/",
            "openai/gpt-4o-mini",
            "sk-x",
        )
        .unwrap();
        assert_eq!(c.base_url(), "https://openrouter.ai/api/v1");
    }

    #[test]
    fn model_accessor_reflects_construction() {
        let c =
            OpenAiCompatClient::new("https://x/v1", "anthropic/claude-3.5-sonnet", "sk-x").unwrap();
        assert_eq!(c.model(), "anthropic/claude-3.5-sonnet");
    }

    #[test]
    fn defaults_are_openrouter_shaped() {
        assert!(DEFAULT_OPENAI_BASE_URL.starts_with("https://"));
        assert!(DEFAULT_OPENAI_BASE_URL.ends_with("/v1"));
        assert!(!DEFAULT_OPENAI_MODEL.is_empty());
    }

    #[test]
    fn request_serializes_to_openai_shape() {
        let req = ChatRequest {
            model: "m",
            messages: vec![ChatMessage {
                role: "user",
                content: "hi",
            }],
            temperature: 0.0,
            stream: false,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "m");
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "hi");
        assert_eq!(json["stream"], false);
    }

    #[test]
    fn response_parses_first_choice_content() {
        let raw = r#"{"choices":[{"message":{"role":"assistant","content":"Paris"}}]}"#;
        let parsed: ChatResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.choices[0].message.content, "Paris");
    }
}
