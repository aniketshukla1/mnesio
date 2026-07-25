//! # mnesio-mcp
//!
//! MCP (Model Context Protocol) server exposing mnesio's core APIs as
//! tools that any MCP-compatible client — Claude Desktop, Cline,
//! Cursor, OpenAI's agent platform — can call directly.
//!
//! Three tools ship today:
//!
//! - `mnesio_write_memory` — append a memory (synchronous embed)
//! - `mnesio_search` — hybrid retrieval with synthesized answer
//! - `mnesio_record_outcome` — append an `OutcomeRecorded` event the
//!   procedural compiler consumes
//!
//! Transport is newline-delimited JSON-RPC 2.0 over stdio. Logging
//! goes to stderr only so it doesn't pollute the protocol channel.

pub mod context;
pub mod handler;
pub mod protocol;
pub mod tools;
