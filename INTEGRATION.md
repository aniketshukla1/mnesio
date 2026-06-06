# Integrating mneme with agents (OpenClaw, Hermes, Claude Desktop, …)

mneme is **agent-framework agnostic**. It does not embed in any one agent; it
exposes a memory layer over stable seams that agents already speak:

| Surface | Crate | Transport | Use it from |
|---|---|---|---|
| **MCP server** | `mneme-mcp` | stdio JSON-RPC | any MCP client (OpenClaw, Hermes, Claude Desktop, Cursor, …) |
| HTTP API + dashboard | `mneme-server` | HTTP/REST | any language; live metrics at `/dashboard` |
| Python | `mneme-py` (pyo3) | in-process | LangChain / custom Python agents |

For **OpenClaw** and **Hermes** the path is **MCP** — both are MCP clients:

- **OpenClaw** — its skill system is largely MCP-server wrappers; you add mneme
  as an MCP server (or a thin ClawHub-style skill that points at it).
- **Hermes** (Nous Research) — has a native MCP client (stdio + HTTP transports,
  selective tool loading). Register mneme under its `mcp_servers` config.

> The interesting bit with Hermes: it has its *own* "create a skill after a
> complex task" loop. mneme's procedural compiler does the same thing **but
> behind a non-bypassable safety gate** (canaries + safety probe + non-negative
> objective delta). So mneme becomes Hermes' *verifiable, versioned, erasable*
> memory + skill store — the guarantees its built-in skills don't have.

---

## 1. The three tools

`mneme-mcp` exposes exactly three tools (names are stable):

| Tool | Required args | Optional args | What it does |
|---|---|---|---|
| `mneme_write_memory` | `content` (string) | `tenant` (string) | append a memory to the log (async embed/evolve off the write path) |
| `mneme_search` | `query` (string) | `tenant`, `k` (int) | hybrid retrieval (vector + BM25 + RRF); returns memories + citations |
| `mneme_record_outcome` | `artifacts_used` (string[]), `success` (bool) | `episode`, `scores` (obj), `error` | feed an agent task outcome to the **gated** procedural compiler for credit assignment |

`tenant` is mneme's scope boundary (Hard Rule #3) — give each user/agent its own
tenant for isolation.

---

## 2. Build the server

```bash
# stdio MCP server binary → target/release/mneme-mcp
cargo build -p mneme-mcp --release
```

It reads `MNEME_DATA` for the event-log directory (defaults to a temp dir).
Point all clients at the same `MNEME_DATA` to share memory.

Verify it with a raw stdio session (exactly what an agent does):

```bash
MNEME_DATA=/tmp/mneme printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"mneme_write_memory","arguments":{"content":"The capital of France is Paris","tenant":"demo"}}}' \
  '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"mneme_search","arguments":{"query":"capital of France","tenant":"demo","k":3}}}' \
  | ./target/release/mneme-mcp
```

Expected (verified): `initialize` → `serverInfo.name = "mneme-mcp"`,
`protocolVersion = 2024-11-05`; `tools/list` → the three tools; the search
returns the memory you just wrote, with a citation. The same flow is asserted in
CI by the `agent_session_over_stdio` integration test (no real agent needed).

---

## 3. Hermes

Hermes loads MCP servers from its `mcp_servers` config and auto-reloads on
change. Register mneme as a **stdio** server:

```json
{
  "mcp_servers": {
    "mneme": {
      "command": "/abs/path/to/target/release/mneme-mcp",
      "args": [],
      "env": { "MNEME_DATA": "/var/lib/mneme" }
    }
  }
}
```

A ready-to-edit copy lives at [`examples/integrations/hermes.mcp.json`](examples/integrations/hermes.mcp.json).

Then in a Hermes session the agent will see `mneme_write_memory`,
`mneme_search`, `mneme_record_outcome` and can use them like any other tool. Use
Hermes' selective tool-loading to expose only what a given agent needs.

> **HTTP transport:** Hermes also supports MCP over HTTP. `mneme-mcp` is **stdio
> only** today; MCP-over-HTTP is a planned addition (the substrate is the same
> handler). Until then, use stdio for MCP, or call `mneme-server`'s REST API
> (`/api/search`, etc.) from Hermes' generic HTTP tools.

---

## 4. OpenClaw

OpenClaw installs skills, ~most of which wrap MCP servers. Register mneme as an
MCP server in its config (key is typically `mcpServers`):

```json
{
  "mcpServers": {
    "mneme": {
      "command": "/abs/path/to/target/release/mneme-mcp",
      "args": [],
      "env": { "MNEME_DATA": "/var/lib/mneme" }
    }
  }
}
```

A ready-to-edit copy + a minimal ClawHub-style skill manifest live at
[`examples/integrations/openclaw.mcp.json`](examples/integrations/openclaw.mcp.json)
and [`examples/integrations/openclaw.skill.json`](examples/integrations/openclaw.skill.json).

---

## 5. The loop that makes the agent better (the wedge)

Storage-shaped memory stops at write/search. mneme adds the **self-improvement
loop** — and that's what `mneme_record_outcome` is for:

```
agent runs a task using a mneme system prompt / skill (an "artifact")
        │
        ├── mneme_search(query)              → relevant memories injected into context
        │
   task completes
        │
        └── mneme_record_outcome(artifacts_used=[…], success, scores)
                    │
                    ▼
        procedural compiler (offline): reflect → propose K → shadow-eval → gate
                    │
        EvalReport::is_committable()?  (canaries + safety probe + Δobjective ≥ 0)
              │ yes                                  │ no
              ▼                                      ▼
     new PolicyArtifact version              rejected — old version stays active
     hot-swapped in (atomic)                 (Hard Rule #1: nothing unsafe commits)
```

Every committed artifact is versioned and reversible; a forgotten subject is
crypto-shredded; every belief traces to its source events. That is the
difference between "the agent remembers" and "the agent gets **verifiably**
better and can take it back."

---

## 6. What to test

1. **Protocol loop (deterministic, CI):** `agent_session_over_stdio` — spawns
   `mneme-mcp` and drives initialize → tools/list → writes → search (asserts
   recall) → record_outcome. No real agent needed.
1b. **Real LLM agent loop (live):** `python3 examples/agent_loop_eval.py` — a
   real Ollama model **decides its own tool calls** against the real `mneme-mcp`
   server and answers questions about private facts it can't otherwise know.
   Measured: **0% → 83%** with mneme (llama3.2). This is the closest proxy to a
   real OpenClaw/Hermes session that runs without their full stack, and it
   exercises the exact MCP transport they use.
2. **Real-agent smoke (your environment):** drop the config above into OpenClaw
   / Hermes, run a task, watch the tool calls and the dashboard
   (`mneme-server` → `/dashboard`).
3. **Value at scale (the moat):** `mneme-bench` measures recall@k, latency
   p50/p95/p99, and throughput at 1k–100k memories, plus the procedural learning
   curve. See [`BENCHMARKS.md`](BENCHMARKS.md) for the real numbers.
