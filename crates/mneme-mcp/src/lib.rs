//! # mneme-mcp
//!
//! MCP (Model Context Protocol) server exposing mneme's core APIs as
//! tools that any MCP-compatible client — Claude Desktop, Cline,
//! Cursor, OpenAI's agent platform — can call directly.
//!
//! Three tools ship today:
//!
//! - `mneme_write_memory` — append a memory (synchronous embed)
//! - `mneme_search` — hybrid retrieval with synthesized answer
//! - `mneme_record_outcome` — append an `OutcomeRecorded` event the
//!   procedural compiler consumes
//!
//! Transport is newline-delimited JSON-RPC 2.0 over stdio. Logging
//! goes to stderr only so it doesn't pollute the protocol channel.

pub mod context;
pub mod handler;
pub mod protocol;
pub mod tools;
