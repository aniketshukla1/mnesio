# Changelog

All notable changes to mneme are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — with
the understanding that pre-`1.0.0` releases may break API compatibility
between `0.x.y` bumps.

---

## [Unreleased]

Work landed since the `0.1.0` notes were written. `0.1.0` has never been
published to a public remote or to crates.io, so when the launch tag is cut this
section should be folded into `0.1.0` rather than released separately.

### Added
- **OpenAI-compatible LLM backend** (`mneme-llm`, `openai` feature) targeting any
  `/v1/chat/completions` gateway and defaulting to OpenRouter, so one key reaches
  both Claude and GPT models. The API key is read from the environment only.
  Wired into `mneme-bench` as `--llm openrouter` / `--executor openrouter`.
- **Phase 16 retrieval components** in `mneme-index`, both **opt-in**:
  `rerank::LexicalReranker` (a content-aware final stage over a new
  `ContentProvider` seam, scoring query coverage, temporal date match, phrase
  overlap, and an update cue, with coverage weighted by candidate-set IDF) and
  `context_tree::ContextTree` (tag-derived domain/topic hierarchy that routes a
  query to the best-matching branch and only ever narrows).
- `mneme-bench memeval --rerank` and `--compare-rerank`, the latter running a
  *paired* A/B over a single ingested index.
- Community documentation: `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`.
- GitHub Pages deploy workflow for the docs site, and a
  "How mneme differs" comparison page.

### Changed
- Filled in crates.io metadata across all 19 crates; `mneme-py` is marked
  `publish = false` since it ships to PyPI via maturin.
- Standardised three inconsistent repository URLs (including the literal
  `YOURNAME` placeholder in the workspace manifest) onto one.
- Corrected the README test-count badge to the real total (509).

### Fixed
- **Benchmark methodology.** The A/B comparison built a separate index per arm,
  which let HNSW's build randomness dominate per-category deltas on small suites
  — it reported a 6/12 regression rate that was measurement noise. Arms are now
  paired against one index, and the `--compare-rerank` CI guard fails on
  per-category regression rather than only the overall figure.
- Docs site links: 30 root-absolute markdown links plus the landing page's
  hero/card links would have 404'd under a project-path deploy.
- Disabled four placeholder funding links that pointed at accounts which do not
  exist for this project.

### Known limitations
- The Phase 16 reranker does **not** meet its "done when" and is therefore not in
  the default retrieval path. Measured on LongMemEval-mini with `fastembed`:
  at recall@1, 50/50 runs improved overall and `knowledge-update` went 0% → 100%,
  but `preference` regressed in 3/50 runs; at recall@3 over 20 runs, 3 regressed
  −10pp overall.
- **No LoCoMo number is published yet.** The harness and the frontier-model path
  are in place; the run itself is still outstanding.

---

## [0.1.0] — 2026-05-25 — First usable release

> **The wedge is real.** mneme's procedural-memory compiler is the first
> open-source implementation of a GEPA-style reflective loop with a
> **mechanically non-bypassable commit gate** — the regression guard the
> literature ([LangMem](https://github.com/langchain-ai/langmem),
> [Letta](https://github.com/letta-ai/letta), Mem0) omits.
>
> A bad-faith operator who sets every configurable gate threshold to its
> weakest value still cannot commit a regression — held by an integration
> test that runs on every PR.

### What works in v0.1.0

#### Phase 0 — Foundation
- **Event log** (fjall-backed, append-only) is the single system of record.
  Every other store is a materialized view, fully rebuildable by replaying
  events.
- **Hybrid retrieval** combining `hnsw_rs` vector search + `tantivy` BM25
  with RRF fusion, weighted by embedder maturity (mock embedder gets
  down-weighted so it can't drown out BM25).
- **4-tier BM25 fallback** — strict-AND → fuzzy-AND → strict-OR ∪ fuzzy-OR
  merged → empty. Surfaces the tier in the response so the dashboard can
  show when fallbacks fire.
- **Source-boost** re-ranking — chunks of a known document get a 5% lift
  over equivalently-scoring standalone memories. Breaks BM25's
  length-normalization penalty without disturbing clearly-stronger hits.
- **`SnippetSynthesizer`** for extractive, deterministic, citation-bearing
  answer cards — every word in the answer comes from a real memory.
- **`FastEmbedEmbedder`** (`bge-small-en-v1.5`, 384-dim, ONNX) for real
  semantic embedding + `MockEmbedder` for tests / offline CI.
- **Live dashboard** at `/dashboard` — Chart.js latency/tier/recall charts
  + recent-queries table + rolling p50/p95 stats.
- **`ParagraphChunker`** + `Source` entity for ingesting long documents
  as multiple linked `Memory` chunks.
- Async embedding worker keeps the write path < 5 ms even with a heavy
  embedder.

#### Phase 1 — Memory evolution
- Bounded **A-MEM-style worker** (arXiv:2502.12110) implementing the
  three-step pipeline: note construction → link generation → bounded
  evolution.
- **Never-overwrite invariant**: evolution invalidates the old version and
  writes a new bi-temporal version with a `parent` pointer + lineage.
- **Bounded cascades**: `EvolveConfig` caps `max_evolve_per_write`,
  per-memory `cooldown_secs`, `max_lifetime_evolutions`, and
  `min_change_threshold`. A-MEM has no convergence guarantee; these are
  what replace it.
- Dashboard **PHASE 1** panel showing live chain timelines + chain-depth
  bars (how close each memory is to the lifetime cap).

#### Phase 2 — Procedural compiler (the wedge)
- **GEPA-style reflective loop** (arXiv:2507.19457): reflect → propose
  K candidates → shadow-evaluate → gate → atomic commit.
- **`EvalGates::evaluate`** is non-bypassable: setting every configurable
  threshold to its weakest value still rejects via the baseline
  `EvalReport::is_committable()`. Tested by
  `loosening_configurable_gates_cannot_bypass_strict_baseline`.
- **Judge panel** with diversity gate — defends against in-context reward
  hacking. Single-judge probes are always rejected.
- **`ShadowEvaluator`** uses **unanimous** judgment for canaries + safety
  probes (explicit guardrails) and **majority** for replay (long-tail
  noise tolerance).
- **`EvalSuite` + `LearningCurveCollector`** record the absolute benchmark
  score + safety-probe pass rate per committed version. Phase-2 "done
  when" criterion: positive curve, no safety regression.
- **Live demo verifies the criterion**: benchmark climbs `33% → 67% →
  100%` while safety probe pass rate stays glued at **100%** across every
  version. Dashboard renders this as a dual-line chart with a `safety
  100%` pill that flips red on any regression.
- **`ProceduralStore`** with atomic active-version hot-swap. The store is
  derived state; replay rebuilds it exactly from the event log.
- Top-rejection-reasons tally in the dashboard so operators can see
  *why* a proposal was rejected, not just *that* it was.

#### Infrastructure
- **`mneme-llm` crate**: `FakeLlmClient` (deterministic, dep-free) +
  `OllamaLlmClient` (real local backend, feature-gated behind `ollama`,
  default-on).
- **`DemoLlmClient`** in `mneme-server`: content-derived responses for
  every prompt the workers issue (note-construction, link-generation,
  evolution-proposal, reflection, proposal). Demo runs offline.
- **Vendored Chart.js** — no CDN dependency. Self-contained demo.
- **`MNEME_EVOLVE` / `MNEME_PROCEDURAL`** env flags for opt-in worker
  enablement. `MNEME_EVOLVE_LLM=ollama` swaps to a real model.
- **Apache-2.0 LICENSE**, **GitHub Actions CI** (fmt + clippy + test,
  both feature configurations, with cancellation + caching).

### Hard rules (mechanically enforced)

These are invariants the codebase will not let you violate. Each has a
dedicated test that fails if the property breaks:

1. **Nothing procedural commits without passing
   `EvalReport::is_committable()`** — canaries + safety probe + Δ ≥ 0.
2. **Never overwrite history** — evolution invalidates + writes a new
   bi-temporal version. The log is append-only.
3. **Scope is a security boundary** — procedural learning + evolution
   never cross a `Scope` without explicit aggregation.
4. **The event log is the single system of record** — every index is a
   materialized view, fully reconstructible by replaying events.
5. **The write path stays fast** — embedding, evolution, and procedural
   compilation are async behind bounded queues. LLM calls never block a
   write.
6. **Cascades are bounded** — `EvolveConfig` caps fan-out, cooldown,
   lifetime count, minimum delta.

### Tests

```
mneme-core        :   3 tests
mneme-llm         :  27 tests
mneme-index       :  57 tests
mneme-evolve      :  11 tests
mneme-procedural  :  94 unit + 5 integration tests
mneme-server      :  20 tests
mneme-store       :   1 test
──────────────────────────────
TOTAL             : 218 tests · all passing on both feature configs
```

### Try it

```bash
git clone https://github.com/aniketshukla1/mneme.git
cd mneme
MNEME_DEMO=1 MNEME_PROCEDURAL=on cargo run -p mneme-server
```

Open <http://127.0.0.1:7777/dashboard> and watch the **PROCEDURAL**
section's learning curve climb in real time, with the safety probe line
held at 100%.

### Known limitations (intentional — Phases 3–6 ahead)

- **Public API is unstable.** Expect breaking changes between `0.x.y`
  bumps. Pin a specific version in `Cargo.toml`.
- **HNSW is unfiltered.** Scope filtering today happens post-search,
  which works at demo scale but won't scale to large multi-tenant
  deployments. ACORN-style filtered HNSW lands in Phase 3.
- **No property graph yet.** Relationships are tracked as `MemoryRef`
  links on the `Memory` struct, not as a true graph. Phase 4.
- **No Python bindings.** `pyo3` + MCP server are Phase 5.
- **Reflector only handles `SystemPrompt` artifacts.** `Heuristic`,
  `Skill`, `RetrievalRule`, `Reflection` kinds are defined but error
  visibly when proposed. Coming in subsequent slices.
- **Demo judges are content-derived, not LLM-driven.** The
  `LlmJudge` shipping in `mneme-procedural` works end-to-end; the demo
  uses fake judges so it runs offline. Pointing it at Ollama via
  `MNEME_EVOLVE_LLM=ollama` exercises the real path.
- **Single-writer to the event log.** Multi-host deployments are not
  yet supported — that's the Phase 4 graph store's territory.

### Background

Two design documents back this project (kept outside the repo):
1. Comparative survey of agent memory systems (Mem0, Zep, Letta, A-MEM, ...).
2. Rust-native self-improving memory architecture + phased build plan.

Embedded references:
- A-MEM — Lyu et al., *Agentic Memory for LLM Agents*, arXiv:2502.12110
  (memory evolution model).
- GEPA — Du et al., *General Evolutionary Prompt Adaptation*,
  arXiv:2507.19457 (reflective-loop pattern).
- ACORN — Wu et al., *ACORN: Performant Hybrid Search* (filtered HNSW,
  Phase 3 target).

### Roadmap

- **Phase 3** — Custom filtered HNSW (ACORN-style, soft-delete).
- **Phase 4** — Bi-temporal property graph store on fjall.
- **Phase 5** — `pyo3` bindings + MCP (Model Context Protocol) server.
- **Phase 6** — Eval harness as a first-class product (the real moat).

---

[0.1.0]: https://github.com/aniketshukla1/mneme/releases/tag/v0.1.0
