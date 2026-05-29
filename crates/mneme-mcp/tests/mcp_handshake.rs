//! End-to-end integration test: spawn the `mneme-mcp` binary as a
//! child process, drive it with JSON-RPC over stdio, and assert on
//! the responses.
//!
//! This is the only test that covers the full stdio framing path —
//! line buffering, newline-delimited JSON, stderr-only logging.
//! Everything else lives in unit tests against the in-process
//! `handle_request`.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;

/// Path to the binary cargo built for these integration tests.
/// Cargo injects this env var per-bin in the package under test.
const BIN: &str = env!("CARGO_BIN_EXE_mneme-mcp");

/// RAII wrapper that kills + reaps the child on drop. Without it,
/// a panicking test would leak a zombie mneme-mcp process holding
/// the temp fjall lockfile.
struct Server {
    child: Child,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Spawn `mneme-mcp` against a fresh temp data dir + the mock
/// embedder (no model download, instant boot).
fn spawn() -> (
    Server,
    ChildStdin,
    BufReader<ChildStdout>,
    tempfile::TempDir,
) {
    let data_dir = tempfile::tempdir().expect("create temp data dir");
    let child = Command::new(BIN)
        .env("MNEME_DATA", data_dir.path())
        .env("MNEME_EMBEDDER", "mock")
        // RUST_LOG=off keeps stderr quiet so test output stays readable.
        .env("RUST_LOG", "off")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mneme-mcp");
    let mut child = child;
    let stdin = child.stdin.take().expect("stdin piped");
    let stdout = BufReader::new(child.stdout.take().expect("stdout piped"));
    (Server { child }, stdin, stdout, data_dir)
}

/// Send one JSON-RPC line. The server expects newline-delimited
/// frames — every request ends with `\n`.
fn send(stdin: &mut ChildStdin, req: &serde_json::Value) {
    let s = serde_json::to_string(req).expect("serialize request");
    stdin
        .write_all(s.as_bytes())
        .expect("write to mneme-mcp stdin");
    stdin.write_all(b"\n").expect("write newline");
    stdin.flush().expect("flush mneme-mcp stdin");
}

/// Read one response line. Times out after 10s — generous because
/// the first call has to spin up the fjall keyspace + tantivy
/// reader, which is non-trivial on cold caches.
fn recv(stdout: &mut BufReader<ChildStdout>) -> serde_json::Value {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut stdout_buf = String::new();
    std::thread::scope(|s| {
        s.spawn(|| {
            let r = stdout.read_line(&mut stdout_buf);
            let _ = tx.send(r);
        });
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(n)) => assert!(n > 0, "mneme-mcp closed stdout"),
            Ok(Err(e)) => panic!("read from mneme-mcp stdout failed: {e}"),
            Err(_) => panic!("timed out waiting for mneme-mcp response"),
        }
    });
    serde_json::from_str(stdout_buf.trim()).expect("parse JSON-RPC response")
}

#[test]
fn initialize_lists_tools_and_runs_write_then_search() {
    let (_server, mut stdin, mut stdout, _data_dir) = spawn();

    // ---- 1. initialize ----
    send(
        &mut stdin,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "integration-test", "version": "0.1.0"}
            }
        }),
    );
    let resp = recv(&mut stdout);
    assert_eq!(resp["id"], 1);
    assert_eq!(
        resp["result"]["serverInfo"]["name"], "mneme-mcp",
        "initialize should return mneme-mcp identity"
    );

    // ---- 2. tools/list ----
    send(
        &mut stdin,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }),
    );
    let resp = recv(&mut stdout);
    let tools = resp["result"]["tools"].as_array().expect("tools array");
    let names: Vec<String> = tools
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect();
    assert!(names.contains(&"mneme_write_memory".into()));
    assert!(names.contains(&"mneme_search".into()));
    assert!(names.contains(&"mneme_record_outcome".into()));

    // ---- 3. tools/call mneme_write_memory ----
    send(
        &mut stdin,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "mneme_write_memory",
                "arguments": {
                    "content": "Acme Q3 revenue grew 18% YoY across all segments.",
                    "tenant": "test",
                    "tags": ["earnings", "acme"]
                }
            }
        }),
    );
    let resp = recv(&mut stdout);
    let text = resp["result"]["content"][0]["text"]
        .as_str()
        .expect("text content");
    assert!(
        text.starts_with("wrote memory "),
        "write_memory should confirm with the new id; got: {text}"
    );

    // ---- 4. tools/call mneme_search ----
    send(
        &mut stdin,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "mneme_search",
                "arguments": {
                    "query": "Acme revenue earnings",
                    "tenant": "test"
                }
            }
        }),
    );
    let resp = recv(&mut stdout);
    let text = resp["result"]["content"][0]["text"]
        .as_str()
        .expect("search text content");
    assert!(
        text.contains("Acme") || text.contains("revenue"),
        "search should surface the just-written memory; got: {text}"
    );

    // ---- 5. tools/call unknown tool → returns is_error=true ----
    send(
        &mut stdin,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "nope",
                "arguments": {}
            }
        }),
    );
    let resp = recv(&mut stdout);
    // Unknown tool surfaces at the JSON-RPC error layer (dispatch returns
    // Err, handler maps it to INVALID_PARAMS).
    assert!(
        resp["error"].is_object(),
        "unknown tool should produce a JSON-RPC error response"
    );
}

#[test]
fn parse_error_for_garbage_input_uses_null_id() {
    let (_server, mut stdin, mut stdout, _data_dir) = spawn();
    // Send something that doesn't parse as JSON at all.
    stdin
        .write_all(b"not json at all\n")
        .expect("write garbage");
    stdin.flush().expect("flush");
    let resp = recv(&mut stdout);
    assert_eq!(
        resp["error"]["code"].as_i64().unwrap(),
        -32700,
        "PARSE_ERROR code per JSON-RPC 2.0"
    );
    assert!(resp["id"].is_null(), "parse-error responses have null id");
}
