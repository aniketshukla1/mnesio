//! `mneme_write_memory` — append a new memory.
//!
//! Synchronous embedding: the handler computes the embedding inline
//! and appends both `MemoryWritten` AND `MemoryEmbedded` events
//! before returning. Trades write latency for predictability — by
//! the time the tool call resolves, the new memory is searchable.

use crate::context::AppContext;
use crate::protocol::{CallToolResult, ToolDescriptor};
use mneme_core::entity::{Memory, Provenance};
use mneme_core::event::Event;
use mneme_core::traits::MaterializedView;
use mneme_core::types::{new_id, BiTemporal, MemoryRef, Scope};
use serde::Deserialize;
use serde_json::{json, Value};

pub fn descriptor() -> ToolDescriptor {
    ToolDescriptor {
        name: "mneme_write_memory",
        description: "Append a new memory to mneme. The content is embedded synchronously, \
                      so the memory is searchable as soon as this tool call returns. \
                      Returns the new memory's id.",
        input_schema: json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to remember. Free-form text."
                },
                "tenant": {
                    "type": "string",
                    "description": "Tenant the memory belongs to. Scope is a security boundary — \
                                    memories never leak across tenants. Defaults to `default`.",
                    "default": "default"
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional tags to attach to the memory for filtering."
                }
            },
            "required": ["content"]
        }),
    }
}

#[derive(Debug, Deserialize)]
struct Args {
    content: String,
    #[serde(default = "default_tenant")]
    tenant: String,
    #[serde(default)]
    tags: Vec<String>,
}

fn default_tenant() -> String {
    "default".into()
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
    if args.content.trim().is_empty() {
        return Ok(CallToolResult::error_text("content must be non-empty"));
    }

    let mem = Memory {
        id: new_id(),
        scope: Scope::global(&args.tenant),
        content: args.content.clone(),
        keywords: vec![],
        tags: args.tags.clone(),
        context: String::new(),
        embedding: None,
        links: vec![],
        parent: None,
        evolution_count: 0,
        time: BiTemporal::now(),
        provenance: Provenance {
            source: "mcp".into(),
            trust: 0.5,
        },
        source: None,
        position: None,
    };
    let memory_id = mem.id;

    // Append MemoryWritten first so the log records the write before
    // we attempt the embed — even if embedding fails, the memory
    // exists.
    let written = Event::MemoryWritten(mem.clone());
    let id1 = ctx.log.append(written.clone()).await?;
    ctx.vector
        .apply(&mneme_core::event::LogEntry {
            id: id1,
            event: written,
        })
        .await?;
    ctx.bm25
        .apply(&mneme_core::event::LogEntry {
            id: id1,
            event: Event::MemoryWritten(mem.clone()),
        })
        .await?;

    // Embed + append MemoryEmbedded so the vector view is current.
    let embeddings = ctx
        .embedder
        .embed(std::slice::from_ref(&args.content))
        .await?;
    let Some(embedding) = embeddings.into_iter().next() else {
        return Ok(CallToolResult::error_text("embedder returned no vectors"));
    };
    let embedded = Event::MemoryEmbedded {
        id: MemoryRef(memory_id),
        embedding: embedding.clone(),
        model_id: ctx.embedder.model_id().to_string(),
    };
    let id2 = ctx.log.append(embedded.clone()).await?;
    ctx.vector
        .apply(&mneme_core::event::LogEntry {
            id: id2,
            event: embedded,
        })
        .await?;

    Ok(CallToolResult::text(format!(
        "wrote memory {memory_id} (tenant={}, tags={})",
        args.tenant,
        args.tags.join(",")
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
    async fn write_appends_two_events_per_memory() {
        let (_dir, ctx) = fresh_ctx().await;
        let result = handle(&ctx, json!({"content": "hello world", "tenant": "t"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        let entries = ctx.log.read_from(None).await.unwrap();
        let written = entries
            .iter()
            .filter(|e| matches!(e.event, Event::MemoryWritten(_)))
            .count();
        let embedded = entries
            .iter()
            .filter(|e| matches!(e.event, Event::MemoryEmbedded { .. }))
            .count();
        assert_eq!(written, 1);
        assert_eq!(embedded, 1);
    }

    #[tokio::test]
    async fn rejects_empty_content() {
        let (_dir, ctx) = fresh_ctx().await;
        let result = handle(&ctx, json!({"content": "   "})).await.unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn rejects_malformed_arguments() {
        let (_dir, ctx) = fresh_ctx().await;
        // Missing required field
        let result = handle(&ctx, json!({})).await.unwrap();
        assert!(result.is_error);
    }

    #[test]
    fn descriptor_schema_is_stable() {
        let d = descriptor();
        assert_eq!(d.name, "mneme_write_memory");
        let schema_str = serde_json::to_string(&d.input_schema).unwrap();
        // The presence of these keys is the contract MCP clients
        // rely on; if a refactor breaks them the snapshot fails.
        assert!(schema_str.contains(r#""required":["content"]"#));
        assert!(schema_str.contains(r#""tenant""#));
        assert!(schema_str.contains(r#""tags""#));
    }
}
