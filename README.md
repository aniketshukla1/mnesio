<div align="center">
  <h1>🧠 mneme</h1>
  <p><strong>A self-improving long-term memory layer for AI agents, built in Rust.</strong></p>

  <p>
    <a href="https://github.com/aniketshukla1/mneme/actions"><img alt="Build Status" src="https://img.shields.io/badge/build-passing-brightgreen"></a>
    <a href="https://crates.io/crates/mneme"><img alt="Version" src="https://img.shields.io/badge/version-v0.1.0-blue"></a>
    <a href="https://github.com/aniketshukla1/mneme/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
    <a href="#"><img alt="Tests" src="https://img.shields.io/badge/tests-380%20passing-brightgreen"></a>
  </p>
</div>

---

Most agent memory systems (Mem0, Zep, Letta, Cognee) are **storage-shaped**: they remember facts. mneme's differentiator is **procedural self-improvement** — the agent gets *better at doing things* over time, with a regression guard the literature omits.

Two continuous loops operate over a single append-only event log:

1. **Procedural-memory compiler** (the wedge) — turns batches of agent `Outcome`s into improved, versioned `PolicyArtifact`s (system prompts, heuristics, retrieval rules) via a GEPA-style reflective loop: reflect → propose K candidates → shadow-evaluate → Pareto-select → **gated commit**.
2. **Memory evolution** (supporting) — when a memory is written, a bounded async worker retroactively re-tags and re-links related memories (A-MEM style), keeping the knowledge graph the compiler learns from adaptive.

### Why the wedge matters

> **Hard Rule #1: Nothing procedural commits without passing `EvalReport::is_committable()`** — canaries 100%, safety probe passing, objective Δ ≥ 0. This is the regression guard LangMem omits. Mechanically enforced — setting every configurable gate threshold to its weakest value *still* cannot bypass the baseline. Held by a dedicated integration test on every commit.

---

## ⚡ Quick start

```bash
git clone https://github.com/aniketshukla1/mneme.git
cd mneme

# Run the workspace tests (380 passing)
cargo test --workspace

# Boot the demo: live retrieval + memory evolution + procedural compiler
MNEME_DEMO=1 MNEME_PROCEDURAL=on cargo run -p mneme-server
```

Then open:
- **http://127.0.0.1:7777/** — live chat-style retrieval (hybrid vector + BM25 + extractive synthesis)
- **http://127.0.0.1:7777/dashboard** — real-time benchmarks: latency, BM25 tier distribution, memory evolution chains, **procedural learning curve**, an **ingestion-intelligence** panel (raw turns → ADD / UPDATE(contradiction) / NOOP, served by `/api/ingest/metrics`), and a **bi-temporal knowledge-graph** panel (`/api/graph`), and a **profile/persona** panel (`/api/profile`)

In the demo, watch the **PROCEDURAL** section's learning curve climb from ~33% to 100% while the safety probe line stays glued at 100% — that's the Phase 2 "done when" criterion satisfied live.

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `MNEME_DEMO` | `0` | `1` → use a temp data dir + synthetic writer (no persistence) |
| `MNEME_EMBEDDER` | `fastembed` | `mock` for a 32-dim deterministic embedder (no model download) |
| `MNEME_EVOLVE` | `on` | `off` to disable the memory-evolution worker |
| `MNEME_PROCEDURAL` | `off` | `on` to enable the procedural compiler (LLM-heavy) |
| `MNEME_EVOLVE_LLM` | `demo` | `ollama` for a real local model via `MNEME_OLLAMA_URL` / `MNEME_OLLAMA_MODEL` |
| `MNEME_DATA` | `./mneme-data` | Path to the fjall keyspace |
| `MNEME_PORT` | `7777` | HTTP listen port |

---

## 🏗️ Workspace architecture

Twelve crates, each with a focused responsibility. External dependencies sit behind traits (`LlmClient`, `Embedder`, `EventLog`, `MaterializedView`, `Retriever`, `Synthesizer`, `PolicyExecutor`, `Judge`) so providers are swappable.

| Crate | Role | Status |
|---|---|---|
| `mneme-core` | Types + traits + event log shape. No I/O. | ✅ |
| `mneme-store` | `fjall`-backed append-only event log | ✅ Phase 0 |
| `mneme-index` | `hnsw_rs` vector + `tantivy` BM25 + RRF hybrid + extractive synthesis | ✅ Phase 0 |
| `mneme-graph` | Bi-temporal property graph store on `fjall` — nodes, edges, BFS, `as_of` queries | ✅ Phase 4 |
| `mneme-extract` | Ingestion intelligence — fact extraction, ADD/UPDATE/NOOP consolidation, importance admission + decay | ✅ Phase 7 |
| `mneme-privacy` | PII redaction (minimisation) + crypto-shred keyring (right-to-be-forgotten on an append-only log) | ✅ Phase 8 |
| `mneme-llm` | `LlmClient` implementations: `FakeLlmClient`, `OllamaLlmClient` (feature-gated) | ✅ |
| `mneme-evolve` | Bounded A-MEM-style memory evolution worker | ✅ Phase 1 |
| `mneme-procedural` | GEPA-style procedural compiler + gate + eval suite + learning curve | ✅ Phase 2 |
| `mneme-causal` | Counterfactual contribution scoring + GC by measurement (leave-one-out ablation over the replayable log) | ✅ Phase 10 |
| `mneme-probe` | Self-falsifying memory — acceptance probes + belief calibration; a refuted claim invalidates-and-supersedes itself (history kept) | ✅ Phase 11 |
| `mneme-kv` | Gated KV cartridges — KV cache as a versioned, gated, erasable view of the log (tensor backend simulated; reconciliations real) | ◑ Phase 12 |
| `mneme-exchange` | Certified skill exchange — export a gated artifact as a signed certificate; the importer re-runs its own gate before activation | ✅ Phase 13 |
| `mneme-dream` | Negative memory + dreaming — gated suppression rules from bad outcomes; bounded offline prune-by-contribution + re-anchor drifted notes | ✅ Phase 14 |
| `mneme-provenance` | Regulator-grade provenance — time-travel reconstruction + provenance chains + verifiable erasure over the append-only log | ✅ Phase 15 |
| `mneme-bench` | Eval-as-product harness — procedural learning curve (GSM8K/HumanEval) + memory recall@k (LOCOMO/LongMemEval) | ✅ Phase 2/6 |
| `mneme-server` | Host process: HTTP API, dashboard, demo wiring | ✅ |
| `mneme-mcp` | MCP server: exposes mneme as tools to Claude Desktop / Cline / any MCP client | ✅ Phase 5 |
| `mneme-py` | Python bindings via pyo3 — pip-installable | ✅ Phase 5 |
| `sdk/node` | TypeScript/Node SDK over the HTTP surface — zero runtime deps | ✅ Phase 9 |

---

## 📈 Eval-as-product (`mneme-bench`)

The bench harness isn't a one-off demo — it's a CLI you can wire into your dev loop or your CI. Two subcommands, four output formats, exit codes that block PRs on regression.

### Run mode — iterative improvement curve

```bash
# Iterate the procedural compiler against gsm8k-tiny and emit a
# self-contained HTML report you can attach to a PR.
cargo run -p mneme-bench -- run \
  --suite gsm8k \
  --max-versions 6 \
  --output html \
  --out curve.html
```

The HTML is self-contained — inline SVG line chart of `benchmark_score` + `safety_probe_pass_rate` over versions, KPI strip, seed-vs-final prompt diff. No JS, no external assets, no Chart.js dep.

### Compare mode — A vs B prompt evaluation

```bash
cargo run -p mneme-bench -- compare \
  --suite gsm8k \
  --baseline "Answer the question." \
  --candidate "Answer the question. Show your work step by step." \
  --output markdown
```

Output (paste-into-PR-friendly):

```
| | benchmark | safety |
|---|---|---|
| baseline  |   0.0% | 100.0% |
| candidate |  70.0% | 100.0% |
| **Δ**     | **+70.0pp** | **+0.0pp** |
```

### CI mode — block regressions

```bash
cargo run -p mneme-bench -- run \
  --suite gsm8k \
  --max-versions 6 \
  --regression-threshold 0.05 \
  --output json --out bench.json
```

Exit code semantics:
- **`0`** — benchmark held or improved within threshold; safety probe at 100% throughout.
- **`1`** — benchmark fell more than `--regression-threshold` below v1.
- **`1` (no threshold needed)** — any safety probe regression. Alignment drift is the hard stop; you don't get to set a threshold for it.

Drop it in a GitHub Actions step:

```yaml
- name: mneme-bench gates
  run: |
    cargo run --release -p mneme-bench -- run \
      --suite gsm8k \
      --regression-threshold 0.05 \
      --output json --out bench.json
- uses: actions/upload-artifact@v4
  with: { name: bench-results, path: bench.json }
```

A PR that regresses the bench fails the gate. The artifact is downloadable from the run page for inspection.

### Suites

Two suites ship in-binary today — hand-curated, license-clean:

| Suite | Tasks | Safety probes | Categories |
|---|---|---|---|
| `gsm8k` | 10 grade-school math word problems | 3 | math, rate, geometry, arithmetic, percent, fractions |
| `humaneval` | 5 Python code-completion prompts | 3 | predicate, builtins, string, branching |

External suites land via a future `--suite path/to/suite.json` flag (JSON schema in `crates/mneme-bench/data/`).

### Memory recall — LOCOMO / LongMemEval

A third subcommand, `memeval`, benchmarks the **memory layer itself** (not the
procedural compiler): it ingests a haystack of memories through the *real*
`FjallEventLog → VectorView + Bm25View → HybridRetriever` path, then asks
questions and reports **recall@k** — does any top-`k` memory contain the gold
answer span? — overall and per category (single-hop / multi-hop / temporal /
knowledge-update / open-domain).

```bash
# Offline smoke (mock embedder, BM25-dominated):
cargo run -p mneme-bench -- memeval --suite locomo --k 10
cargo run -p mneme-bench -- memeval --suite longmemeval --k 10 --output json

# Real semantic number (downloads bge-small on first run):
cargo run -p mneme-bench -- memeval --suite locomo --embedder fastembed

# CI floor — exit 1 if recall@k drops below the bar:
cargo run -p mneme-bench -- memeval --suite locomo --min-recall 0.8
```

Two hand-curated, license-clean mini suites ship in-binary (`locomo_mini`,
`longmemeval_mini`). They're **smoke-scale** (≈12 memories) — under the `mock`
embedder recall is BM25-driven and HNSW tie-breaks make the borderline
question non-deterministic, so set CI floors with margin and quote published
numbers from `--embedder fastembed` against the full datasets.

---

## 🐍 Using mneme from Python

```bash
pip install maturin
maturin develop --release --manifest-path crates/mneme-py/Cargo.toml
```

`maturin develop` builds the Rust extension and drops a `mneme` package into your active Python environment. Then:

```python
import mneme

client = mneme.Client(data_dir="./mneme-data", embedder="fastembed")

# Write a memory.
memory_id = client.write_memory(
    content="My partner's coffee order is oat-milk flat white",
    tenant="default",
    tags=["coffee", "preference"],
)

# Hybrid retrieval with synthesized answer.
result = client.search(query="what coffee do I like?", k=5)
print(result.answer)                 # synthesized prose (or None)
for hit in result.hits:              # ranked individual hits
    print(hit.memory_id, hit.score, hit.content)
print(result.citations)              # memory ids the synthesizer cited

# Record outcomes for the procedural compiler to learn from.
client.record_outcome(
    artifacts_used=["01ABC..."],     # ULID-string artifact ids
    success=True,
    scores={"accuracy": 0.95, "latency_ms": 1850.0},
)
```

### Plugging into LangChain

mneme works as a drop-in retriever inside any LangChain pipeline by wrapping `client.search` in a `BaseRetriever`:

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
import mneme

class MnemeRetriever(BaseRetriever):
    client: mneme.Client
    tenant: str = "default"
    k: int = 5

    def _get_relevant_documents(self, query, *, run_manager):
        result = self.client.search(query=query, tenant=self.tenant, k=self.k)
        return [
            Document(page_content=h.content, metadata={"memory_id": h.memory_id, "score": h.score})
            for h in result.hits
        ]

retriever = MnemeRetriever(client=mneme.Client("./mneme-data"))
```

### Async support

The current API is **synchronous** — each call blocks until complete. Agent-call latency is dominated by the LLM itself, so this is rarely the bottleneck. A future release will add a native `AsyncClient` using `pyo3-asyncio`.

---

## 🟦 Using mneme from Node / TypeScript

`sdk/node` ships a tiny client over the HTTP surface — zero runtime
dependencies (it uses Node 18+ built-in `fetch`). Mirrors the DTOs from
`mneme-server` 1:1 with full TypeScript types.

```ts
import { MnemeClient } from "@mneme/sdk";

const mneme = new MnemeClient({ baseUrl: "http://127.0.0.1:7777" });

// One round-trip: post-gate PolicyArtifacts + hybrid retrieval.
const { skills, hits } = await mneme.retrieveWithSkills(
  "what did our last call decide about pricing?",
  5,
  { actor: "analyst" }, // optional — enforces inter-agent ACL
);

const system =
  skills.map(s => s.injection).join("\n\n") +
  "\n\nContext:\n" +
  hits.map(h => `- ${h.content}`).join("\n");
```

Every returned `skills[i]` has cleared the mechanical safety gate
(canaries 100%, safety probe passing, objective Δ ≥ 0) — drop the
`injection` straight into your prompt.

The same client wraps cleanly into LangChain `BaseRetriever`,
LlamaIndex `BaseRetriever`, and CrewAI `Tool`. See `sdk/node/README.md`
for adapter sketches.

```bash
cd sdk/node
npm install
npm run build && npm test     # 8 tests, no server required
```

---

## 🔌 Using mneme from Claude Desktop (MCP)

The `mneme-mcp` binary speaks the [Model Context Protocol](https://modelcontextprotocol.io). Add it to your Claude Desktop config and three tools become available in any conversation:

- **`mneme_write_memory(content, tenant?, tags?)`** — append a new memory. Embeds synchronously so it's searchable immediately.
- **`mneme_search(query, tenant?, k?)`** — hybrid retrieval (vector + BM25) returning a synthesized answer plus excerpts and citations.
- **`mneme_record_outcome(episode?, artifacts_used, success, scores?, error?)`** — record the outcome of an agent task. The procedural compiler consumes these to learn what prompt patterns lead to good outcomes.

### Install

```bash
cargo install --path crates/mneme-mcp
```

### Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "mneme": {
      "command": "mneme-mcp",
      "env": {
        "MNEME_DATA": "/Users/you/mneme-data",
        "MNEME_EMBEDDER": "fastembed"
      }
    }
  }
}
```

Restart Claude Desktop. The 🔌 icon in the input bar will show the three `mneme_*` tools available.

### Try it

```
> Remember: my partner's coffee order is oat-milk flat white, two shots.

[Claude calls mneme_write_memory]

> What does my partner drink?

[Claude calls mneme_search → finds + cites the memory]
```

### Configuration

| Env var | Default | Meaning |
|---|---|---|
| `MNEME_DATA` | `./mneme-data` | Path to the fjall keyspace. **Use an absolute path in your Claude config** — relative paths resolve to wherever Claude launched. |
| `MNEME_EMBEDDER` | `mock` | `mock` (32-dim deterministic, no model download) or `fastembed` (real bge-small-en-v1.5). `mock` is fine for trying it out; `fastembed` for real use. |
| `RUST_LOG` | `warn` | Standard tracing-subscriber filter. Logs go to stderr only (stdout is the protocol channel). |

### Transport

Newline-delimited JSON-RPC 2.0 over stdio. Three methods: `initialize`, `tools/list`, `tools/call`. Hand-rolled because the protocol is small enough that depending on an SDK adds more risk than it removes — `crates/mneme-mcp/src/protocol.rs` is ~300 lines including doc comments and tests.

---

## 🔒 Hard rules (non-negotiable invariants)

These are enforced in code, not by convention. Each has a dedicated test that fails if the invariant breaks:

1. **Nothing procedural commits without passing `EvalReport::is_committable()`** — canaries + safety probe + non-negative objective delta. The configurable `EvalGates` layer can only add rejection reasons on top of this baseline; it can never relax it. Test: `loosening_configurable_gates_cannot_bypass_strict_baseline`.
2. **Never overwrite history** — memory evolution invalidates the old version and writes a new bi-temporal version with a `parent` pointer. Same for any fact update. The event log is append-only.
3. **Scope is a security boundary** — procedural learning + memory evolution never cross a `Scope` without explicit aggregation. Every cross-entity read goes through `Scope::contains`.
4. **The event log is the single system of record** — every index (vector, BM25, graph, procedural) is a materialized view, fully reconstructible by replaying events. Tested end-to-end.
5. **The write path stays fast** — embedding, evolution, and procedural compilation are async behind bounded queues. The write path target is < 5 ms; LLM calls never block it.
6. **Cascades are bounded** — `EvolveConfig` caps cascade fan-out, per-memory cooldown, lifetime evolution count, and minimum structural delta. A-MEM has no convergence guarantee; these bounds replace it.

---

## 📊 Phase 2 "done when" — verified

> Phase 2 is "done when" the system demonstrates a *positive learning curve on an ALFWorld-style suite with no safety-probe regression*.

Live demo output (`MNEME_PROCEDURAL=on`):

```
v1:  benchmark=33.33% safety=100%
v2:  benchmark=66.67% safety=100%
v3:  benchmark=100.00% safety=100%
v4+: benchmark=100.00% safety=100%  (plateau — both improvement signals integrated)
```

The dashboard renders this as a dual-line chart with a `safety 100%` pill that flips red on any regression.

---

## 📊 Scale & real-data benchmarks

All numbers below are **measured**, not projected — produced by `mneme-bench`
on a 2021 M1-class laptop (8 cores, 16 GB), release build. Reproduce with the
commands shown.

### Real public benchmarks — SQuAD + HotpotQA (recall@10)

`mneme-bench fetch` downloads a real dataset from the Hugging Face
datasets-server and runs it through the *actual* ingest → hybrid-retrieve path.
**SQuAD** (single-hop reading comprehension): each context → a memory
(deduplicated), each question/answer-span → a recall pair. **HotpotQA**
(multi-hop): each of a row's context paragraphs → a memory, the answer span
must be found across them (yes/no comparison answers are skipped — not
retrievable spans).

```bash
cargo run -p mneme-bench --features fetch --release -- \
  fetch --dataset squad   --rows 2000 --k 10 --embedder fastembed
cargo run -p mneme-bench --features fetch --release -- \
  fetch --dataset hotpotqa --rows 1000 --k 10 --embedder fastembed
```

| Dataset | Embedder | Memories | Questions | recall@10 | ms/query |
|---|---|---:|---:|---:|---:|
| SQuAD v1.1 (single-hop) | `fastembed` (384-d) | 315 | 2,000 | **98.1%** | 9.24 |
| SQuAD v1.1 (single-hop) | `mock` (32-d, BM25) | 315 | 2,000 | 93.9% | 1.81 |
| HotpotQA (multi-hop) | `fastembed` (384-d) | 9,227 | 941 | **88.7%** | 17.97 |
| HotpotQA (multi-hop) | `mock` (32-d, BM25) | 9,227 | 941 | 83.4% | 8.61 |

Real semantic embeddings lift recall over keyword-only on the same real
questions — **+4.2 pts on single-hop SQuAD, +5.3 pts on the harder multi-hop
HotpotQA** — the hybrid path earning its keep on non-synthetic data. The
HotpotQA run is also a real-corpus scale check: 9k+ memories, 941 multi-hop
questions, sub-18 ms/query.

### Scale & load — synthetic corpus up to 105k memories

`mneme-bench scale` ingests a deterministic synthetic corpus (labeled needles
salted among distractors, plus evolution chains + contradictions) through the
real storage→views→retriever path, and **separates the two write phases** so
the numbers reflect mneme's architecture: the *append* path is the user-facing
write (Hard Rule #5, <5ms), while *index build* (HNSW + BM25) is what the
server does asynchronously off the write path. The index phase uses the bulk
replay-rebuild path (stage all docs, one BM25 commit), so its throughput is
HNSW-bound rather than dominated by per-document segment flushes.

```bash
cargo run -p mneme-bench --release -- scale --sizes 1000,10000,50000,100000 --embedder mock
```

| Memories | Append/s | Append p50 | Index/s | Index p50 | Query p50 | Query p99 | recall@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,050 | 218,082 | 0.0022 ms | 7,648 | 0.13 ms | 0.88 ms | 1.76 ms | 100% |
| 10,503 | 385,546 | 0.0017 ms | 2,725 | 0.35 ms | 1.36 ms | 4.20 ms | 100% |
| 52,515 | 246,886 | 0.0018 ms | 1,625 | 0.60 ms | 2.21 ms | 2.96 ms | 100% |
| 105,030 | 299,564 | 0.0017 ms | 1,326 | 0.75 ms | 3.60 ms | 4.90 ms | 100% |

Read of the curve: **append latency is flat (~0.0017 ms p50) across a 100×
size increase** — the write path genuinely doesn't degrade with corpus size.
Index build is HNSW-bound and degrades gracefully (per-insert p50 0.13 ms →
0.75 ms as the graph deepens). Query latency grows *sub-linearly* (HNSW): p50
0.88 ms → 3.60 ms from 1k to 105k. Recall stays **100%** on the exact-gold
needle set through 105k memories, confirming retrieval correctness holds at
scale. (The synthetic generator is deterministic — same `--seed` reproduces
the identical corpus.)

**With a real semantic embedder** (`--embedder fastembed`, 384-d) at 5,251
memories: append still **182,519/s, p50 0.00 ms** (embedding is computed in a
separate pre-phase, off the write path — Hard Rule #5), index 1,345/s, query
p50 9.84 ms (per-query embedding dominates), recall@10 99.2%. The write path
stays fast whether the embedder is mock or a real model.

These recall floors are enforced in CI — the `bench-gate` job fails the build
if LOCOMO/LongMemEval mini-suite recall or synthetic-scale recall drops below
its floor (eval-as-product, the moat made into a regression gate).

### Edge-case / adversarial stress

Throughput stress is only half of "ready". `mneme-bench edge` drives the *real*
ingest→retrieve→replay path with hostile inputs and asserts the seven hard-rule
invariants hold — exiting non-zero (and gating CI) on any violation:

```bash
cargo run -p mneme-bench -- edge
```

| Scenario | Invariant checked |
|---|---|
| degenerate queries | empty / whitespace / stopword-only / `k=0` / `k≫N` never panic or error |
| pathological syntax | 12 operator/`AND OR NOT`/unicode/emoji queries are sanitized, not 500'd |
| unicode & emoji content | CJK / accented / emoji memories ingest and stay retrievable |
| huge & empty content | a ~1 MB memory and an empty one both ingest; gold still retrieved |
| **scope isolation extreme** | 1 tenant-A needle among 4,000 tenant-B → found, **zero cross-tenant leakage** (Hard Rule #3) |
| supersede keeps history | a corrected fact leaves retrieval but its original write stays in the log (Hard Rule #2) |
| tombstone-heavy index | 195/200 invalidated → only the 5 live returned; counts consistent |
| dim mismatch | a wrong-dimension vector is rejected with an error, not a panic |
| **replay rebuild** | fresh views replayed from the log reproduce identical BM25 + recall (Hard Rule #4) |
| concurrent writes | 256 concurrent appends all land with unique, monotonic ids (Hard Rule #2/#4) |

This suite **found and fixed a real bug**: an all-stopword query (`"the of a"`)
or one with bare boolean operators (`"a AND OR NOT b"`) used to surface a hard
tantivy parse error — i.e. a 500 on adversarial search input. The BM25 query
path now treats unparseable free-text as "no results for this tier" (graceful
empty), while still honoring valid explicit-operator queries like
`revenue OR growth`.

---

## ⚖️ How mneme compares

```bash
cargo run -p mneme-bench -- compete --k 10 --embedder fastembed
```

> **Two different metrics, kept separate.** The capability matrix below is a
> *structural* comparison. The benchmark numbers further down mix **cited
> competitor end-to-end QA accuracy** with mneme's **measured retrieval
> recall@k** — a retrieval-quality proxy, *not the same metric*. recall@k asks
> "was the gold answer in the retrieved set?"; QA accuracy asks "did the model
> produce the right answer?". We never present one as if it beat the other.

### Capability matrix

| Capability | mneme | Mem0 | Zep | Letta | A-MEM |
|---|:---:|:---:|:---:|:---:|:---:|
| Append-only, replayable event log as system of record | ✅ | — | ◑ | — | — |
| Bi-temporal versioning (never overwrite; invalidate-and-supersede) | ✅ | ◑ | ✅ | — | — |
| Hybrid retrieval (vector + BM25 + RRF) with explainable breakdown | ✅ | ◑ | ◑ | ◑ | ◑ |
| Procedural self-improvement (gets better at tasks over time) | ✅ | — | — | ◑ | — |
| Non-bypassable commit gate (canaries + safety probe) | ✅ | — | — | — | — |
| Counterfactual contribution scoring + GC by measurement | ✅ | — | — | — | — |
| Self-falsifying memory (probes auto-supersede on failure) | ✅ | — | — | — | — |
| Crypto-shred erasure reconciled with an append-only log | ✅ | — | — | — | — |
| Time-travel reconstruction + provenance chains | ✅ | — | ◑ | — | — |
| Certified skill exchange (re-gated on import) | ✅ | — | — | — | — |
| Self-contained / embedded (no external vector or graph DB) | ✅ | ◑ | ◑ | ✅ | ◑ |

✅ shipped · ◑ partial · — not in published design. Competitor cells reflect
each system's published architecture and may evolve. **mneme is the only
column with every row** — the frontier features require the append-only +
replayable + bi-temporal substrate behind a non-bypassable gate, which a
storage-shaped system can't add without rebuilding its foundation.

### Benchmark landscape (cited end-to-end QA — *different metric from recall@k*)

| System | Benchmark | Metric | Score | Source |
|---|---|---|---:|---|
| Full-context (upper bound) | LOCOMO | LLM-as-Judge (J) | 72.90% | Mem0 paper, [arXiv:2504.19413](https://arxiv.org/abs/2504.19413), Table 2 |
| Mem0 (graph) | LOCOMO | LLM-as-Judge (J) | 68.44% | Mem0 paper, [arXiv:2504.19413](https://arxiv.org/abs/2504.19413), Table 2 |
| Mem0 | LOCOMO | LLM-as-Judge (J) | 66.88% | Mem0 paper, [arXiv:2504.19413](https://arxiv.org/abs/2504.19413), Table 2 |
| Zep | LOCOMO | LLM-as-Judge (J) | 65.99% | Mem0 paper, [arXiv:2504.19413](https://arxiv.org/abs/2504.19413), Table 2 |
| LangMem | LOCOMO | LLM-as-Judge (J) | 58.10% | Mem0 paper, [arXiv:2504.19413](https://arxiv.org/abs/2504.19413), Table 2 |
| A-Mem | LOCOMO | LLM-as-Judge (J) | 48.38% | Mem0 paper, [arXiv:2504.19413](https://arxiv.org/abs/2504.19413), Table 2 |
| Zep (gpt-4o) | LongMemEval | QA accuracy | 71.20% | Zep paper, [arXiv:2501.13956](https://arxiv.org/abs/2501.13956), Table 2 |
| Full-context (gpt-4o) | LongMemEval | QA accuracy | 60.20% | Zep paper, [arXiv:2501.13956](https://arxiv.org/abs/2501.13956), Table 2 |

These are competitor/baseline numbers from the cited papers — **not mneme's**.
mneme's *measured* numbers are retrieval recall@k: **98.1%** on real SQuAD
(fastembed, §Scale & real-data above) and 100% on the curated LOCOMO/
LongMemEval mini-suites. An apples-to-apples QA-accuracy run would put an LLM
judge over mneme's retrieved context (mneme ships the `Judge` + synthesizer
path for it; it's just not part of the offline CI number). mneme's
differentiation is the capability matrix, not a single leaderboard cell.

---

## 🗺️ Roadmap

- **Phase 0** ✅ Foundation — event log, hybrid retrieval, dashboard
- **Phase 1** ✅ Memory evolution — bounded A-MEM-style worker
- **Phase 2** ✅ Procedural compiler — the wedge, with mechanically-enforced commit gate, ALFWorld-style bench harness
- **Phase 3** ✅ Filtered HNSW — adaptive over-fetch on selective scopes, per-tenant partitioning (`TenantPartitionedVectorView`), soft-delete observability (`tombstone_ratio`, `live_count`)
- **Phase 4** ✅ Bi-temporal property graph store on fjall — typed `Relation` edges (`Linked` / `EvolvedFrom` / `EvolvedTo` / `ContainedIn`), `as_of` time-travel, scope-filtered BFS + shortest-path, replay-rebuildable
- **Phase 5** ✅ Distribution — MCP server + Python (`pyo3`) bindings, both reachable from any agent framework
- **Phase 6** ✅ Eval harness as a first-class product (the real moat) — `mneme-bench` run/compare CLI, self-contained HTML reports, CI regression gates with exit-code semantics

### Competitive layer (parity-plus-wedge)

- **Phase 7** ✅ Ingestion intelligence — extract atomic facts → consolidate ADD / UPDATE(contradiction|refinement) / NOOP, importance admission + decay (`mneme-extract`)
- **Phase 8** ✅ Retrieval + personalization + privacy — graph/recency fusion + reranker, profile memory, multi-agent ACLs, PII redaction + crypto-shred forget (`mneme-privacy`)
- **Phase 9** ✅ Skill reuse + distribution — committed-artifact injection at query time, Node/TS SDK (`sdk/node`)

### Frontier layer (the bets no one else can ship)

- **Phase 10** ✅ Causal memory — counterfactual contribution scoring + GC by measurement (`mneme-causal`)
- **Phase 11** ✅ Self-falsifying memory — acceptance probes + belief calibration; a refuted claim auto-supersedes (`mneme-probe`)
- **Phase 12** ◑ Gated KV cartridges — KV cache as a versioned, gated, erasable view of the log; substrate done, real tensor backend behind the `KvBackend` trait remaining (`mneme-kv`)
- **Phase 13** ✅ Certified skill exchange — signed certificate; importer re-runs its own gate before activation (`mneme-exchange`)
- **Phase 14** ✅ Negative memory + dreaming — gated suppression rules + bounded offline prune-by-contribution & re-anchor (`mneme-dream`)
- **Phase 15** ✅ Regulator-grade provenance — time-travel reconstruction + provenance chains + verifiable erasure (`mneme-provenance`)

The frontier layer (10–15) is what a storage-shaped competitor (Mem0, Zep, Letta, Cognee, A-MEM) can't follow without rebuilding its foundation — each bet exploits the append-only + replayable + bi-temporal log behind the non-bypassable safety gate. See `COMPETITIVE.md` → "P3 — frontier bets".

---

## 🧪 Test counts

```
mneme-core        :   3 tests
mneme-llm         :  11 tests
mneme-index       :  83 tests
mneme-evolve      :  27 tests
mneme-procedural  : 107 tests
mneme-causal      :  15 tests
mneme-probe       :  14 tests
mneme-kv          :  10 tests
mneme-exchange    :  11 tests
mneme-dream       :  10 tests
mneme-provenance  :   7 tests
mneme-bench       :  25 tests (+7 under --features fetch: SQuAD + HotpotQA loaders)
mneme-mcp         :  26 tests (unit + integration)
mneme-py          :   7 tests (Rust-side inner-client coverage)
mneme-server      :  27 tests
mneme-store       :   1 test
mneme-graph       :  27 tests
mneme-extract     :  33 tests
mneme-privacy     :  19 tests
sdk/node (TS)     :   8 tests (offline, stub fetch)
──────────────────────────────
TOTAL             : 463 Rust tests (460 on --no-default-features) + 8 SDK tests · all passing
                    (+7 more with --features fetch on mneme-bench)
```

---

## 🤝 Contributing

Contributions welcome. A few specific patterns the project enforces:

- **The gate is sacred.** Any change to `mneme-procedural::gate` requires a corresponding test demonstrating that the property still holds. Loosening default thresholds requires a code review comment explaining the trade-off.
- **External dependencies behind traits.** New backends (LLMs, embedders, judges, executors) go behind the existing trait surface; concrete implementations live in their own crate.
- **`cargo fmt` + `cargo clippy -- -D warnings` must pass** on both `--no-default-features` and the default config before any commit.
- **Tests live next to code** in `#[cfg(test)] mod tests`. Storage tests use a temp dir keyed by a fresh ULID and clean up after themselves.
- **Conventional commits** — `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.

---

## 📚 Background

Two design documents back this project:
1. A comparative survey of agent memory systems (Mem0, Zep, Letta, A-MEM, etc.) and where each falls short.
2. The Rust-native self-improving memory architecture + phased build plan.

Section numbers in code comments (e.g. "report §3") refer to document 2.

References embedded in the code:
- A-MEM: Lyu et al., *Agentic Memory for LLM Agents*, arXiv:2502.12110 (memory evolution model)
- GEPA: Du et al., *General Evolutionary Prompt Adaptation*, arXiv:2507.19457 (reflective-loop pattern)
- ACORN: Wu et al., *ACORN: Performant Hybrid Search* (filtered HNSW — informs the Phase 3 adaptive over-fetch + partitioning approach)

---

## ⚠️ Stability

This is `0.1.0` — the first usable release. The system is end-to-end working with 380 passing tests across all six build phases, but the public API surface will still move as the graph store and procedural compiler gain real-world mileage. Pin a specific version in your `Cargo.toml`; expect breaking changes between `0.x.y` bumps.

---

## 📜 License

Apache License 2.0. See [LICENSE](LICENSE).
