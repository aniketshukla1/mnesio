//! `mneme_search` — hybrid retrieval + synthesized answer.
//!
//! Runs the same `HybridRetriever` + `SnippetSynthesizer` the HTTP
//! server uses, but renders the result as a single text block
//! consumable by an LLM (rather than JSON consumed by a UI). The
//! output bundles the synthesized answer plus a "citations" list so
//! the LLM can reason about provenance.

use crate::context::AppContext;
use crate::protocol::{CallToolResult, ToolDescriptor};
use mneme_core::synthesizer::Passage;
use mneme_core::{Query, Retriever, Scope};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;

pub fn descriptor() -> ToolDescriptor {
    ToolDescriptor {
        name: "mneme_search",
        description: "Search mneme's memory store. Returns a synthesized answer composed of \
                      direct quotes from the matching memories, plus a list of memory ids \
                      cited. Uses hybrid retrieval (vector + BM25 with RRF fusion).",
        input_schema: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query to match against stored memories."
                },
                "tenant": {
                    "type": "string",
                    "description": "Tenant to search within. Cross-tenant search is forbidden.",
                    "default": "default"
                },
                "k": {
                    "type": "integer",
                    "description": "Maximum number of memories to retrieve. Defaults to 5.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                }
            },
            "required": ["query"]
        }),
    }
}

#[derive(Debug, Deserialize)]
struct Args {
    query: String,
    #[serde(default = "default_tenant")]
    tenant: String,
    #[serde(default = "default_k")]
    k: usize,
}

fn default_tenant() -> String {
    "default".into()
}

fn default_k() -> usize {
    5
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
    if args.query.trim().is_empty() {
        return Ok(CallToolResult::error_text("query must be non-empty"));
    }
    let k = args.k.clamp(1, 20);

    // Resolve memory id → content so the synthesizer has passages to
    // quote from. Replay the log; demo-scale only — production would
    // cache.
    let entries = ctx.log.read_from(None).await?;
    let mut contents: HashMap<mneme_core::Id, (String, Vec<String>)> = HashMap::new();
    for entry in &entries {
        if let mneme_core::event::Event::MemoryWritten(m) = &entry.event {
            contents.insert(m.id, (m.content.clone(), m.tags.clone()));
        }
    }

    let scope = Scope::global(&args.tenant);
    let hits = ctx
        .retriever
        .search(&Query {
            text: args.query.clone(),
            scope,
            k,
            time_filter: None,
        })
        .await?;

    if hits.is_empty() {
        return Ok(CallToolResult::text(format!(
            "No memories matched query {:?} under tenant={:?}.",
            args.query, args.tenant
        )));
    }

    let passages: Vec<Passage> = hits
        .iter()
        .map(|h| {
            let (content, tags) = contents
                .get(&h.memory.0)
                .cloned()
                .unwrap_or_else(|| ("<unknown memory>".into(), vec![]));
            Passage {
                memory: h.memory,
                content,
                tags,
                retrieval_score: h.score,
            }
        })
        .collect();

    let answer = ctx.synthesizer.synthesize(&args.query, &passages).await?;

    // Render the answer as plain text the LLM can consume directly.
    // Three sections:
    //   1. synthesized prose (if the synthesizer produced any)
    //   2. raw excerpts from each cited memory (always — the LLM
    //      benefits from seeing the actual content, not just ids)
    //   3. citation list
    let mut sections: Vec<String> = Vec::new();
    if let Some(prose) = &answer.prose {
        sections.push(prose.clone());
    }
    if !answer.excerpts.is_empty() {
        let mut excerpts = String::from("Relevant memories:");
        for ex in &answer.excerpts {
            let body = passages
                .iter()
                .find(|p| p.memory == ex.memory)
                .map(|p| p.content.as_str())
                .unwrap_or("<unknown>");
            // Cap each excerpt so a single overlong memory doesn't
            // dominate the context window.
            let snippet: String = body.chars().take(280).collect();
            excerpts.push_str(&format!("\n  • {} — {snippet}", ex.memory.0));
        }
        sections.push(excerpts);
    }
    let citations: Vec<String> = answer.citations.iter().map(|c| c.0.to_string()).collect();
    if !citations.is_empty() {
        sections.push(format!("— citations: [{}]", citations.join(", ")));
    }
    Ok(CallToolResult::text(sections.join("\n\n")))
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
    async fn empty_query_is_tool_error() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle(&ctx, json!({"query": "   "})).await.unwrap();
        assert!(r.is_error);
    }

    #[tokio::test]
    async fn search_with_no_memories_returns_no_match_message() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle(&ctx, json!({"query": "anything"})).await.unwrap();
        assert!(!r.is_error);
        let text = match &r.content[0] {
            crate::protocol::ContentBlock::Text { text } => text.clone(),
        };
        assert!(text.contains("No memories matched"));
    }

    #[tokio::test]
    async fn search_finds_a_just_written_memory() {
        let (_dir, ctx) = fresh_ctx().await;
        // Write a memory through the write_memory tool so the views
        // get populated through the same path search will use.
        super::super::write_memory::handle(
            &ctx,
            json!({"content": "the capital of france is paris", "tenant": "t"}),
        )
        .await
        .unwrap();
        let r = handle(&ctx, json!({"query": "capital france", "tenant": "t"}))
            .await
            .unwrap();
        assert!(!r.is_error);
        let text = match &r.content[0] {
            crate::protocol::ContentBlock::Text { text } => text.clone(),
        };
        assert!(
            text.to_lowercase().contains("paris"),
            "expected the search result to surface the memory; got: {text}"
        );
        assert!(text.contains("citations"));
    }

    #[test]
    fn descriptor_schema_is_stable() {
        let d = descriptor();
        assert_eq!(d.name, "mneme_search");
        let schema_str = serde_json::to_string(&d.input_schema).unwrap();
        assert!(schema_str.contains(r#""required":["query"]"#));
        assert!(schema_str.contains(r#""maximum":20"#));
    }
}
