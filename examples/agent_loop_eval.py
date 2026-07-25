#!/usr/bin/env python3
"""Real LLM-driven MCP agent loop — does mnesio make a real agent better?

This is an *eval harness*, not part of the build. It wires a real LLM agent to
mnesio's real MCP server and measures the thing that matters: can the agent answer
questions whose answers live *only* in memory?

  - The agent is **local Ollama** (default model `llama3.2`). The model itself
    decides whether to call a tool — we pass mnesio's `mnesio_search` as a real
    Ollama function-calling tool and execute whatever the model asks for.
  - The tool calls hit the **real `mnesio-mcp` binary** over stdio JSON-RPC —
    exactly the transport OpenClaw / Hermes / Claude Desktop use.
  - The questions are about **private facts** pre-seeded into mnesio that the base
    model cannot know (synthetic codenames, regions, dates). So:
        without memory → the model must guess → ~0 correct
        with memory    → the model searches mnesio → recovers the fact → correct
    The gap is the memory layer's value, measured end-to-end with a real model
    making real tool decisions.

Prereqs: a running Ollama (`ollama serve`) with the model pulled
(`ollama pull llama3.2`), and a built server (`cargo build -p mnesio-mcp --release`).

Run:  python3 examples/agent_loop_eval.py
Env:  MNESIO_MCP_BIN, MNESIO_OLLAMA_URL (default http://localhost:11434),
      MNESIO_OLLAMA_MODEL (default llama3.2)
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

OLLAMA_URL = os.environ.get("MNESIO_OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("MNESIO_OLLAMA_MODEL", "llama3.2")
TENANT = "agent-eval"

# Private facts the base model cannot know — seeded into mnesio. (fact, question, gold)
CASES = [
    ("The internal codename for Project Atlas is Nimbus-7.",
     "What is the internal codename for Project Atlas?", "Nimbus-7"),
    ("Aniket's preferred deployment region is ap-south-1.",
     "Which deployment region does Aniket prefer?", "ap-south-1"),
    ("The Q3 board meeting is scheduled for November 14th.",
     "When is the Q3 board meeting scheduled?", "November 14"),
    ("The staging database password rotates every 30 days.",
     "How often does the staging database password rotate?", "30 day"),
    ("Customer Acme's SLA target is 99.95% uptime.",
     "What is customer Acme's SLA uptime target?", "99.95"),
    ("The mnesio release train ships on the third Tuesday of each month.",
     "On which day does the mnesio release train ship?", "third Tuesday"),
]

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "mnesio_search",
        "description": "Search the agent's long-term memory for relevant facts. "
                       "Call this whenever the user asks about something you may "
                       "have been told earlier but don't know from general knowledge.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what to look up"}
            },
            "required": ["query"],
        },
    },
}


def find_bin():
    cand = os.environ.get("MNESIO_MCP_BIN")
    if cand and os.path.exists(cand):
        return cand
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for p in ("target/release/mnesio-mcp", "target/debug/mnesio-mcp"):
        full = os.path.join(root, p)
        if os.path.exists(full):
            return full
    sys.exit("mnesio-mcp binary not found — run `cargo build -p mnesio-mcp --release`")


class Mcp:
    """Minimal MCP stdio client driving the real mnesio-mcp binary."""

    def __init__(self, binary, data_dir):
        self.p = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env={**os.environ, "MNESIO_DATA": data_dir, "MNESIO_EMBEDDER": "mock",
                 "RUST_LOG": "off"},
            text=True, bufsize=1,
        )
        self._id = 0
        self._rpc("initialize", {})

    def _rpc(self, method, params):
        self._id += 1
        self.p.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        return json.loads(line)

    def _call(self, name, arguments):
        r = self._rpc("tools/call", {"name": name, "arguments": arguments})
        return r.get("result", {}).get("content", [{}])[0].get("text", "")

    def write(self, content):
        return self._call("mnesio_write_memory", {"content": content, "tenant": TENANT})

    def search(self, query):
        return self._call("mnesio_search", {"query": query, "tenant": TENANT, "k": 3})

    def close(self):
        try:
            self.p.stdin.close()
            self.p.terminate()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def ollama_chat(messages, tools=None):
    body = {"model": MODEL, "messages": messages, "stream": False}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["message"]


def answer_with_memory(mcp, question):
    """Agent loop: the model decides when to call mnesio_search; we execute it."""
    messages = [
        {"role": "system",
         "content": "You are an assistant with a long-term memory tool "
                    "(mnesio_search). The user will ask about facts you were told "
                    "earlier. Use mnesio_search to look them up, then answer "
                    "concisely with the specific fact. If a tool result contains "
                    "the answer, state it directly."},
        {"role": "user", "content": question},
    ]
    for _ in range(4):  # bounded tool-use turns
        msg = ollama_chat(messages, tools=[SEARCH_TOOL])
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content", "")
        messages.append(msg)
        for c in calls:
            fn = c.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"query": args}
            result = mcp.search(args.get("query", "")) if fn.get("name") == "mnesio_search" else "unknown tool"
            messages.append({"role": "tool", "content": result})
    # ran out of tool turns — ask for a final answer
    messages.append({"role": "user", "content": "Answer now with the fact."})
    return ollama_chat(messages).get("content", "")


def answer_without_memory(question):
    msg = ollama_chat([
        {"role": "system", "content": "Answer the question concisely. If you do "
                                      "not know, say you don't know."},
        {"role": "user", "content": question},
    ])
    return msg.get("content", "")


def correct(ans, gold):
    return gold.lower() in (ans or "").lower()


def main():
    binary = find_bin()
    # sanity: Ollama reachable?
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
    except Exception as e:
        sys.exit(f"Ollama not reachable at {OLLAMA_URL}: {e}  (run `ollama serve`)")

    data_dir = tempfile.mkdtemp(prefix="mnesio-agenteval-")
    mcp = Mcp(binary, data_dir)
    print(f"# agent: Ollama {MODEL}  |  memory: real mnesio-mcp ({os.path.basename(binary)}) over stdio")
    print(f"# seeding {len(CASES)} private facts into mnesio…")
    for fact, _, _ in CASES:
        mcp.write(fact)

    with_ok = without_ok = 0
    t0 = time.time()
    for i, (_, q, gold) in enumerate(CASES, 1):
        wo = answer_without_memory(q)
        wm = answer_with_memory(mcp, q)
        wo_c, wm_c = correct(wo, gold), correct(wm, gold)
        without_ok += wo_c
        with_ok += wm_c
        print(f"\n[{i}] Q: {q}")
        print(f"    gold: {gold!r}")
        print(f"    no-memory : {'✓' if wo_c else '✗'}  {wo.strip()[:90]!r}")
        print(f"    +memory   : {'✓' if wm_c else '✗'}  {wm.strip()[:90]!r}")

    mcp.close()
    n = len(CASES)
    dt = time.time() - t0
    print("\n" + "=" * 60)
    print(f"RESULT over {n} private-fact questions ({dt:.0f}s, model={MODEL}):")
    print(f"  without memory : {without_ok}/{n} = {100*without_ok/n:.0f}%")
    print(f"  with mnesio     : {with_ok}/{n} = {100*with_ok/n:.0f}%")
    print(f"  memory lift    : +{100*(with_ok-without_ok)/n:.0f} percentage points")
    # Non-zero exit if memory didn't help — makes this CI-able.
    sys.exit(0 if with_ok > without_ok else 1)


if __name__ == "__main__":
    main()
