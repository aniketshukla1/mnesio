//! JSON-RPC 2.0 wire format + the Model Context Protocol additions
//! mneme-mcp needs.
//!
//! Transport: newline-delimited JSON on stdin / stdout. One JSON
//! object per line, no LSP-style `Content-Length:` headers. This is
//! what Claude Desktop's stdio launcher uses; it's also the simplest
//! thing that satisfies the MCP spec.
//!
//! Scope: we implement only the subset of MCP needed for the tools
//! we expose — `initialize`, `tools/list`, `tools/call`. Resources
//! and prompts are deferred; they'd live in this module if added.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Version of the MCP protocol we speak. Send back in the
/// `initialize` response so clients can negotiate compatibility.
/// We follow the 2024-11-05 spec which is what Claude Desktop ships.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

// ---------- JSON-RPC 2.0 wire format ----------

#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    /// MUST be exactly "2.0". We deserialize but don't enforce
    /// strictly — Claude Desktop sends "2.0" and that's what matters.
    #[allow(dead_code)]
    pub jsonrpc: String,
    /// Request id. `None` indicates a notification (no response
    /// expected). Both numeric and string ids are valid in
    /// JSON-RPC 2.0; we treat them opaquely so we don't lose them
    /// when echoing back.
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

impl Request {
    /// True iff this is a notification (no id → no response
    /// expected). Notifications must NOT receive a response.
    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
}

impl Response {
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn failure(id: Value, error: ResponseError) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(error),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl ResponseError {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }
}

/// Standard JSON-RPC 2.0 error codes. We use a subset; full list at
/// <https://www.jsonrpc.org/specification#error_object>.
pub mod error_codes {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
}

// ---------- MCP-specific payloads ----------

/// Server identity returned by `initialize`. Clients display this in
/// their UI (Claude Desktop shows it under MCP server settings).
#[derive(Debug, Clone, Serialize)]
pub struct ServerInfo {
    pub name: &'static str,
    pub version: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: &'static str,
    pub capabilities: ServerCapabilities,
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ServerCapabilities {
    /// Empty object signals "I support tools but make no special
    /// claims about list-change notifications etc". Claude Desktop
    /// accepts `{}` here.
    pub tools: ToolsCapability,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ToolsCapability {
    /// `true` iff we send `notifications/tools/list_changed` when
    /// our tool set changes. We don't (the set is static for a
    /// given binary), so `false`.
    #[serde(rename = "listChanged")]
    pub list_changed: bool,
}

/// Descriptor returned from `tools/list`. Clients display these in
/// their tool-picker UI; LLMs see them in the system context so they
/// know what's callable.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDescriptor {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON Schema for the tool's arguments. Schema validation runs
    /// client-side too, but we re-check on the server (defensive).
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ListToolsResult {
    pub tools: Vec<ToolDescriptor>,
}

/// What the client sends on `tools/call`. `arguments` is whatever
/// JSON value the LLM produced for the tool's parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct CallToolParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

/// What the server returns from `tools/call`. The content vector is
/// the tool's textual output, displayed to the user and fed back to
/// the LLM for follow-up reasoning. Most tools return one text
/// element; multi-element returns are useful for tools that surface
/// structured + human-readable results in the same call.
#[derive(Debug, Clone, Serialize)]
pub struct CallToolResult {
    pub content: Vec<ContentBlock>,
    /// `true` if this tool call failed in a way the LLM should learn
    /// from (vs. surfacing the failure as transport-level error).
    #[serde(rename = "isError", skip_serializing_if = "is_false")]
    pub is_error: bool,
}

impl CallToolResult {
    /// Convenience: a successful single-text result.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(content)],
            is_error: false,
        }
    }

    /// Convenience: a tool-level error returned to the LLM. The
    /// transport response is still HTTP-200-equivalent; the LLM is
    /// expected to see `isError: true` and adjust.
    pub fn error_text(content: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(content)],
            is_error: true,
        }
    }
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentBlock {
    Text { text: String },
}

impl ContentBlock {
    pub fn text(t: impl Into<String>) -> Self {
        ContentBlock::Text { text: t.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_round_trips_via_serde() {
        let s = r#"{"jsonrpc":"2.0","id":42,"method":"tools/list"}"#;
        let r: Request = serde_json::from_str(s).unwrap();
        assert_eq!(r.method, "tools/list");
        assert_eq!(r.id, Some(json!(42)));
        assert!(!r.is_notification());
    }

    #[test]
    fn notification_is_request_without_id() {
        let s = r#"{"jsonrpc":"2.0","method":"initialized"}"#;
        let r: Request = serde_json::from_str(s).unwrap();
        assert!(r.is_notification());
    }

    #[test]
    fn response_success_serializes_with_result_no_error() {
        let r = Response::success(json!(1), json!({"ok": true}));
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains(r#""result":{"ok":true}"#));
        assert!(!s.contains("error"));
    }

    #[test]
    fn response_failure_serializes_with_error_no_result() {
        let r = Response::failure(
            json!(2),
            ResponseError::new(error_codes::METHOD_NOT_FOUND, "no such method"),
        );
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains(r#""code":-32601"#));
        assert!(!s.contains(r#""result""#));
    }

    #[test]
    fn call_tool_result_text_helper_round_trips() {
        let r = CallToolResult::text("hello");
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains(r#""type":"text""#));
        assert!(s.contains(r#""text":"hello""#));
        // is_error: false should be elided by serde skip.
        assert!(!s.contains("isError"));
    }

    #[test]
    fn call_tool_result_error_surfaces_is_error_true() {
        let r = CallToolResult::error_text("oops");
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains(r#""isError":true"#));
    }
}
