//! `mnesio_record_outcome` — append an `OutcomeRecorded` event.
//!
//! The procedural compiler consumes these. Agents that want their
//! system prompts iteratively improved by mnesio should call this
//! after every relevant interaction with the user — success/fail
//! per task, optional fine-grained scores.

use crate::context::AppContext;
use crate::protocol::{CallToolResult, ToolDescriptor};
use mnesio_core::entity::{JudgeSource, Outcome};
use mnesio_core::event::Event;
use mnesio_core::types::{new_id, ArtifactRef, EpisodeRef, TrajectoryRef};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;

pub fn descriptor() -> ToolDescriptor {
    ToolDescriptor {
        name: "mnesio_record_outcome",
        description: "Record the outcome of an agent task — success/fail plus optional \
                      numeric scores. The procedural compiler consumes these to learn what \
                      prompt patterns lead to good outcomes. Call this AFTER completing a \
                      task that used a mnesio-managed system prompt.",
        input_schema: json!({
            "type": "object",
            "properties": {
                "episode": {
                    "type": "string",
                    "description": "ULID-style episode identifier — opaque to mnesio. If omitted, a \
                                    fresh id is allocated. Pass the same id across multiple turns \
                                    that share an episode to group them."
                },
                "artifacts_used": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "ULID-style artifact ids whose system prompts contributed to this \
                                    outcome. Required so the compiler can do credit assignment."
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the task completed successfully."
                },
                "scores": {
                    "type": "object",
                    "additionalProperties": { "type": "number" },
                    "description": "Optional numeric scores (e.g. {\"accuracy\": 0.92, \
                                    \"latency_ms\": 1850}). The compiler uses these for \
                                    multi-objective optimization."
                },
                "error": {
                    "type": "string",
                    "description": "Optional error message when success=false."
                }
            },
            "required": ["artifacts_used", "success"]
        }),
    }
}

#[derive(Debug, Deserialize)]
struct Args {
    #[serde(default)]
    episode: Option<String>,
    artifacts_used: Vec<String>,
    success: bool,
    #[serde(default)]
    scores: HashMap<String, f32>,
    #[serde(default)]
    error: Option<String>,
}

pub async fn handle(ctx: &AppContext, arguments: Value) -> anyhow::Result<CallToolResult> {
    let args: Args = match serde_json::from_value(arguments) {
        Ok(a) => a,
        Err(e) => {
            return Ok(CallToolResult::error_text(format!(
                "invalid arguments: {e}"
            )))
        }
    };
    if args.artifacts_used.is_empty() {
        return Ok(CallToolResult::error_text(
            "artifacts_used must be non-empty — the compiler needs to know which artifact(s) drove this outcome",
        ));
    }

    // Parse artifact ids. Malformed ULIDs surface as a tool error so
    // the LLM can correct.
    let mut artifact_refs = Vec::with_capacity(args.artifacts_used.len());
    for s in &args.artifacts_used {
        match s.parse::<mnesio_core::Id>() {
            Ok(id) => artifact_refs.push(ArtifactRef(id)),
            Err(e) => {
                return Ok(CallToolResult::error_text(format!(
                    "artifacts_used[..]={s:?} is not a valid ULID: {e}"
                )));
            }
        }
    }

    let episode_id = match args.episode.as_deref() {
        None => new_id(),
        Some(s) => match s.parse::<mnesio_core::Id>() {
            Ok(id) => id,
            Err(e) => {
                return Ok(CallToolResult::error_text(format!(
                    "episode={s:?} is not a valid ULID: {e}"
                )));
            }
        },
    };

    let outcome = Outcome {
        id: new_id(),
        episode: EpisodeRef(episode_id),
        artifacts_used: artifact_refs,
        success: Some(args.success),
        scores: args.scores,
        error: args.error,
        judge: JudgeSource::Environment,
        trajectory: TrajectoryRef(new_id()),
    };
    let outcome_id = outcome.id;
    ctx.log.append(Event::OutcomeRecorded(outcome)).await?;
    Ok(CallToolResult::text(format!(
        "recorded outcome {outcome_id} (success={}, episode={episode_id})",
        args.success
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::AppContext;
    use serde_json::json;
    use tempfile::TempDir;

    async fn fresh_ctx() -> (TempDir, AppContext) {
        let dir = TempDir::new().unwrap();
        let ctx = AppContext::open(dir.path(), "mock").await.unwrap();
        (dir, ctx)
    }

    #[tokio::test]
    async fn happy_path_appends_outcome_event() {
        let (_dir, ctx) = fresh_ctx().await;
        let fake_artifact = new_id().to_string();
        let r = handle(
            &ctx,
            json!({
                "artifacts_used": [fake_artifact],
                "success": true,
                "scores": {"accuracy": 0.95}
            }),
        )
        .await
        .unwrap();
        assert!(!r.is_error);
        let entries = ctx.log.read_from(None).await.unwrap();
        let outcomes: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.event, Event::OutcomeRecorded(_)))
            .collect();
        assert_eq!(outcomes.len(), 1);
    }

    #[tokio::test]
    async fn empty_artifacts_used_is_tool_error() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle(&ctx, json!({"artifacts_used": [], "success": true}))
            .await
            .unwrap();
        assert!(r.is_error);
    }

    #[tokio::test]
    async fn malformed_artifact_id_is_tool_error() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle(
            &ctx,
            json!({"artifacts_used": ["not-a-ulid"], "success": true}),
        )
        .await
        .unwrap();
        assert!(r.is_error);
    }

    #[test]
    fn descriptor_schema_is_stable() {
        let d = descriptor();
        assert_eq!(d.name, "mnesio_record_outcome");
        let schema_str = serde_json::to_string(&d.input_schema).unwrap();
        assert!(schema_str.contains(r#""required":["artifacts_used","success"]"#));
    }
}
