//! JSON-RPC method dispatcher.
//!
//! Three methods are handled explicitly:
//! - `initialize` — handshake; returns server info + capabilities.
//! - `tools/list` — return every tool descriptor.
//! - `tools/call` — route to [`crate::tools::dispatch`].
//!
//! `notifications/*` (no `id`) are silently accepted — we just don't
//! respond. Anything else returns method-not-found.

use crate::context::AppContext;
use crate::protocol::{
    error_codes, CallToolParams, InitializeResult, ListToolsResult, Request, Response,
    ResponseError, ServerCapabilities, ServerInfo, PROTOCOL_VERSION,
};
use crate::tools;
use serde_json::json;

/// Handle a single decoded JSON-RPC request. Returns `Some(Response)`
/// for requests that need a reply; `None` for notifications (no id).
pub async fn handle_request(ctx: &AppContext, req: Request) -> Option<Response> {
    if req.is_notification() {
        tracing::debug!(method = %req.method, "received notification — no response sent");
        return None;
    }
    // Safe: is_notification() is the negation of `id.is_some()`.
    let id = req.id.clone().expect("notification handled above");

    let resp = match req.method.as_str() {
        "initialize" => Response::success(
            id,
            serde_json::to_value(InitializeResult {
                protocol_version: PROTOCOL_VERSION,
                capabilities: ServerCapabilities::default(),
                server_info: ServerInfo {
                    name: "mneme-mcp",
                    version: env!("CARGO_PKG_VERSION"),
                },
            })
            .expect("InitializeResult always serializes"),
        ),
        "tools/list" => Response::success(
            id,
            serde_json::to_value(ListToolsResult {
                tools: tools::all_tools(),
            })
            .expect("ListToolsResult always serializes"),
        ),
        "tools/call" => {
            let params: CallToolParams =
                match req.params.clone().map(serde_json::from_value).transpose() {
                    Ok(Some(p)) => p,
                    Ok(None) => {
                        return Some(Response::failure(
                            id,
                            ResponseError::new(error_codes::INVALID_PARAMS, "missing params"),
                        ));
                    }
                    Err(e) => {
                        return Some(Response::failure(
                            id,
                            ResponseError::new(
                                error_codes::INVALID_PARAMS,
                                format!("invalid tools/call params: {e}"),
                            ),
                        ));
                    }
                };
            match tools::dispatch(ctx, &params.name, params.arguments).await {
                Ok(result) => Response::success(
                    id,
                    serde_json::to_value(result).expect("CallToolResult always serializes"),
                ),
                Err(e) => Response::failure(
                    id,
                    ResponseError::new(
                        error_codes::INVALID_PARAMS,
                        format!("tool dispatch failed: {e}"),
                    ),
                ),
            }
        }
        // No-op handshake completion the client sometimes sends as a
        // notification AND sometimes as a request with id. If it has
        // an id we acknowledge.
        "initialized" | "notifications/initialized" => Response::success(id, json!({})),
        other => Response::failure(
            id,
            ResponseError::new(
                error_codes::METHOD_NOT_FOUND,
                format!("unknown method {other:?}"),
            ),
        ),
    };
    Some(resp)
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

    fn req(id: i64, method: &str, params: serde_json::Value) -> Request {
        serde_json::from_value(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn initialize_returns_server_info_and_protocol_version() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle_request(&ctx, req(1, "initialize", json!({})))
            .await
            .unwrap();
        let result = r.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "mneme-mcp");
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
    }

    #[tokio::test]
    async fn tools_list_returns_three_tools() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle_request(&ctx, req(2, "tools/list", json!({})))
            .await
            .unwrap();
        let tools = r.result.unwrap();
        let names: Vec<_> = tools["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"mneme_write_memory".to_string()));
        assert!(names.contains(&"mneme_search".to_string()));
        assert!(names.contains(&"mneme_record_outcome".to_string()));
    }

    #[tokio::test]
    async fn tools_call_routes_to_the_named_handler() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle_request(
            &ctx,
            req(
                3,
                "tools/call",
                json!({
                    "name": "mneme_write_memory",
                    "arguments": {"content": "hello", "tenant": "t"}
                }),
            ),
        )
        .await
        .unwrap();
        let result = r.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("wrote memory"));
    }

    #[tokio::test]
    async fn unknown_method_returns_method_not_found() {
        let (_dir, ctx) = fresh_ctx().await;
        let r = handle_request(&ctx, req(4, "no/such/method", json!({})))
            .await
            .unwrap();
        let err = r.error.unwrap();
        assert_eq!(err.code, error_codes::METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn notification_returns_none() {
        let (_dir, ctx) = fresh_ctx().await;
        let mut notif = req(99, "initialize", json!({}));
        notif.id = None; // strip the id → notification
        let r = handle_request(&ctx, notif).await;
        assert!(r.is_none(), "notifications must produce no response");
    }

    #[tokio::test]
    async fn malformed_tools_call_params_returns_invalid_params() {
        let (_dir, ctx) = fresh_ctx().await;
        // `name` field missing — params don't deserialize to CallToolParams.
        let mut bad = req(5, "tools/call", json!({"args": "wrong-shape"}));
        bad.params = Some(json!({"args": "wrong-shape"}));
        let r = handle_request(&ctx, bad).await.unwrap();
        let err = r.error.unwrap();
        assert_eq!(err.code, error_codes::INVALID_PARAMS);
    }
}
