# mneme competitor tracker

*Living doc — last refreshed **2026-06-11**. Strategy + landscape narrative live in
[`COMPETITIVE-2026.md`](COMPETITIVE-2026.md); this file is the **per-competitor
scorecard** we keep current. Re-check quarterly (or when a competitor ships).*

> **All third-party numbers are self-reported unless marked "reproduced."** Vendors
> run their own harness/answerer/judge, so scores conflict (Mem0 self-claims 92.5%
> LoCoMo; ByteRover's board lists Mem0 at 66.9%; atlan/vectorize cite Mem0 at 49.0%
> LongMemEval). Reproduce before quoting. Where an independent "Agent Memory
> Benchmark" reproduced a number, it's flagged.

## How to read this

We score every competitor on **mneme's seven moat dimensions**:

| Code | Dimension | What it means |
|---|---|---|
| **GATE** | Non-bypassable safety gate | self-improvement only commits if `is_committable()` (canaries + safety probe + Δ≥0) |
| **BITEMP** | Append-only + bi-temporal | immutable log; invalidate-and-supersede, not overwrite/delete |
| **KV** | Gated KV cartridge | KV cache as a versioned, gated, erasable view of the log |
| **PROV** | Provenance + time-travel | trace any belief to its source events; reconstruct state as-of T |
| **SHRED** | Crypto-shred erasure | forget a subject on an append-only log (drop per-subject key) |
| **PROC** | Procedural self-improvement | gets better at *tasks*, not just stores facts |
| **REPRO** | Reproducible eval | numbers anyone can re-run with stated methodology |

Legend: ✅ has it · ◑ partial/adjacent · ❌ absent. **Threat tiers:** 🔴 direct
(same buyer + overlapping moat) · 🟠 strong (better on some axis) · 🟡 watch.

## Scorecard matrix

| System | GATE | BITEMP | KV | PROV | SHRED | PROC | REPRO | Headline (self-reported) | Threat |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|:--:|
| **mneme** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅(gated) | ✅ | LoCoMo **not yet published** (98.1% SQuAD recall) | — |
| Mem0 | ❌ | ❌ | ❌ | ◑ | ❌ | ❌ | ❌ | 49.0% LongMemEval (indep.) / 92.5% LoCoMo (self) | 🔴 |
| Zep / Graphiti | ❌ | ✅ | ❌ | ◑ | ❌ | ❌ | ◑ | 63.8% LongMemEval | 🔴 |
| Hindsight | ❌ | ◑ | ❌ | ◑ | ❌ | ◑(reflect) | ✅ | **94.6% LongMemEval (reproduced)** | 🟠 |
| MemOS | ❌ | ❌ | ❌ | ◑ | ❌ | ✅(ungated) | ◑ | 75.8% LoCoMo, 35% token savings | 🔴 |
| Letta (MemGPT) | ❌ | ❌ | ❌ | ❌ | ❌ | ◑(self-edit) | ❌ | — ($10M seed) | 🟠 |
| LangMem | ❌ | ❌ | ❌ | ❌ | ❌ | ◑(edits own prompt) | ❌ | — | 🟡 |
| Cognee | ❌ | ❌ | ❌ | ◑ | ❌ | ❌ | ❌ | — (€7.5M seed) | 🟡 |
| Supermemory | ❌ | ◑ | ❌ | ❌ | ◑(forget) | ❌ | ❌ | 81.6% LongMemEval | 🟠 |
| ByteRover 2.0 | ❌ | ◑ | ❌ | ❌ | ❌ | ❌ | ◑ | 92.2% LoCoMo | 🟠 |
| Honcho | ❌ | ❌ | ❌ | ◑ | ❌ | ◑(dream) | ◑ | 89.9% LoCoMo @ ~5% ctx | 🟡 |
| MemMachine | ❌ | ◑ | ❌ | ◑ | ❌ | ❌ | ◑ | 91.7% LoCoMo | 🟡 |
| Memori | ❌ | ◑(history) | ❌ | ◑(schema) | ❌ | ❌ | ❌ | — | 🟡 |
| Memobase | ❌ | ❌ | ❌ | ◑(profile) | ❌ | ❌ | ❌ | — | 🟡 |
| Redis Agent Memory Server | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | sub-ms retrieval | 🟡 |
| MS Kernel Memory | ❌ | ❌ | ❌ | ◑(Azure lineage) | ❌ | ❌ | ❌ | — | 🟡 |
| OpenViking (ByteDance) | ❌ | ◑ | ❌ | ◑ | ❌ | ◑(skills) | ❌ | — (filesystem context DB) | 🟠 |
| RetainDB | ❌ | ◑ | ❌ | ❌ | ❌ | ❌ | ◑ | "SOTA LongMemEval" (self) | 🟠 |
| MemU | ❌ | ❌ | ❌ | ◑ | ◑(decay) | ◑ | ◑ | 92.09% LoCoMo (self) | 🟡 |
| MIRIX | ❌ | ❌ | ❌ | ◑ | ❌ | ◑(proc type) | ◑ | 85.4% LoCoMo; multimodal | 🟡 |
| Second Me (Mindverse) | ❌ | ❌ | ❌ | ◑ | ◑(local-only) | ❌ | ❌ | — (AI identity model) | 🟡 |

**mneme is the only row that's filled across all seven.** GATE and SHRED are
unmatched by *every* competitor; KV is unmatched as a *governed* view. The honest
hole is REPRO+headline: mneme has the only reproducible-by-design harness but **no
published LoCoMo number yet** — fix first (see COMPETITIVE-2026 Tier-1).

---

## Tier-1 deep profiles (direct / strong competitors)

### Mem0 — 🔴 the category default
- **What:** the most-adopted standalone memory layer (~48K stars, ~14M downloads). YC S24, **$24M Series A** (Oct 2025).
- **Architecture:** dual-store vector + knowledge graph; extraction pipeline → atomic facts scoped to user/session/agent; many vector backends (Qdrant/Chroma/Milvus/pgvector/Redis). Graph tier paywalled ($249/mo).
- **Numbers:** independent cites 49.0% LongMemEval; Mem0's own blog claims 91.6–92.5% LoCoMo / 93–94% LongMemEval at ~7k tokens/query. Big self-vs-independent gap.
- **vs mneme — they win:** distribution, ecosystem, a *published* (if disputed) number, funding.
- **vs mneme — we win:** no gate, treats user change as *replacement not evolution* (their own admitted gap), no append-only/bi-temporal, no provenance/crypto-shred, no gated procedural, no KV. License Apache-2.0 but advanced features paywalled; mneme is fully open.

### Zep / Graphiti — 🔴 the temporal-graph leader (closest on BITEMP)
- **Architecture:** temporal knowledge graph; episodes → entities/edges with **validity windows + interval-tree indexing** → can answer "who led in January vs now." 63.8% LongMemEval. Graphiti OSS (~24K stars); **Zep Cloud is managed-only, Community Edition deprecated.**
- **vs mneme — they win:** mature temporal-graph retrieval, strong on facts-that-change, real adoption.
- **vs mneme — we win:** no gate, no append-only event log (graph is the store, not a replayable view), no KV, no crypto-shred, no procedural; **open-core is closing** (CE deprecated) → mneme's fully-OSS + self-host story is a wedge. atlan explicitly notes Zep has "no constitutional governance layer."

### Hindsight — 🟠 the accuracy + reproducibility leader
- **Architecture:** four parallel retrieval strategies (semantic + BM25 + entity graph + temporal) with **cross-encoder rerank** + a `reflect` synthesis op; Postgres+pgvector; MIT.
- **Numbers:** **94.6% LongMemEval — top *officially reproduced* result.** This is the one to beat on the board.
- **vs mneme — they win:** best reproduced accuracy; richer multi-strategy retrieval than mneme's RRF; MIT + self-host.
- **vs mneme — we win:** no gate, no append-only/bi-temporal substrate, no KV, no crypto-shred, no gated procedural. Their "reflect" ≈ mneme dreaming but ungated. **Borrow their 4-strategy+rerank retrieval; beat them on governance + verifiable improvement.**

### MemOS — 🔴 the self-evolving "memory OS" (closest on PROC)
- **Architecture:** L1 traces → L2 policies → L3 world-model → **crystallized skills + cross-task skill reuse**; Neo4j + Qdrant; Apache-2.0; OpenClaw plugin (72% lower tokens); arXiv 2507.03724. 75.8% LoCoMo, +40% LongMemEval vs baseline.
- **vs mneme — they win:** mainstreamed the self-improvement/skill-reuse *narrative*; distribution (plugin, MemTensor backing, paper); explicit token-savings number.
- **vs mneme — we win:** **no safety/eval gate before committing memory changes** (their docs show none), not append-only (delete API, no immutable history/versioning), not KV-cache, no provenance/time-travel/crypto-shred. **This is the key reframe: MemOS = self-improving; mneme = *safely* self-improving + governed.**

### Letta (MemGPT) — 🟠 the stateful-agent runtime
- **Architecture:** OS-tiered core/recall/archival memory; agents self-edit memory blocks via tools. Apache-2.0, $10M seed, managed cloud.
- **vs mneme — they win:** agent-as-its-own-memory runtime, strong mindshare/lineage.
- **vs mneme — we win:** self-editing is ungated (no canary/safety gate), no bi-temporal log, no KV/provenance/crypto-shred. mneme is a *memory layer* you bolt on; Letta is a *runtime* — partly complementary.

### Supermemory — 🟠 the MCP-native dev favorite
- **Architecture:** memory graph + RAG on Cloudflare Workers + Postgres/pgvector; static vs dynamic facts; auto contradiction-resolution + explicit **forgetting**; MCP-native (Claude Code/OpenCode). 81.6% LongMemEval. **Closed-source**, $3M seed.
- **vs mneme — they win:** fastest dev setup, MCP-native polish, RAG bundled.
- **vs mneme — we win:** closed-source (mneme fully OSS), no gate/append-only/KV/provenance, "forget" is a delete not crypto-shred-on-immutable-log, "compliance posture unestablished" (per atlan).

### ByteRover 2.0 — 🟠 the LoCoMo board-topper (coding agents)
- **Architecture:** hierarchical **Context Tree** (domain→topic→subtopic) + hierarchical traversal; curate/retrieve/justify. **92.2% LoCoMo** (its own board: Mem0 66.9%, Zep 75.1%, Hindsight 89.6%). Note: a Gemini-3-Flash run hit 90.9% → architecture, not model, drives it.
- **vs mneme — they win:** top LoCoMo, best multi-hop/temporal retrieval, coding-agent fit.
- **vs mneme — we win:** none of GATE/BITEMP/KV/PROV/SHRED/PROC (per their own post). **Adopt the Context-Tree idea into mneme retrieval.**

### Honcho — 🟡 reasoning-first user modeling
- **Architecture:** ingestion model extracts preferences/beliefs/contradictions; a background **"dream"** deduction loop; "peers" (humans/agents) as first-class for multi-agent. 89.9% LoCoMo / 90.4% LongMem-S at ~5% context. Apache/AGPL, FastAPI/Postgres/Redis.
- **vs mneme — they win:** deep user modeling, token efficiency, a real "dream" loop shipped.
- **vs mneme — we win:** no gate/append-only/KV/provenance/crypto-shred; dream is ungated. mneme's dreaming is gated + prunes-by-contribution.

### MemMachine — 🟡 "ground-truth-preserving"
- **Architecture:** stores episodes **verbatim**, heavy lifting at retrieval; interoperable storage primitives; OSS; arXiv 2604 (Apr 2026). **91.7% LoCoMo** (gpt-4.1-mini).
- **vs mneme — they win:** strong accuracy, verbatim fidelity, recent paper.
- **vs mneme — we win:** verbatim-everything doesn't scale-govern (no consolidation/decay gate, no KV amortization, no provenance/crypto-shred, no procedural).

### Memori — 🟡 "memory as data with schema + history"
- **Architecture:** structured, queryable, schema'd memory from agent traces; positions on trustworthiness/history. Closest to mneme's *positioning*.
- **vs mneme — they win:** clean schema-first story, agent-trace ingestion.
- **vs mneme — we win:** "history" ≠ append-only+bi-temporal+invalidate-supersede; no gate/KV/crypto-shred/gated-procedural; no reproducible eval shown.

### Cognee / LangMem / Memobase / Redis AMS / MS Kernel Memory — 🟡 watch
- **Cognee:** poly-store (graph+vector+relational), 6-line onboarding, €7.5M seed, OSS, no managed cloud. Win: ingestion breadth. We win: no governance substrate.
- **LangMem:** LangGraph-coupled KV+vector; notably **"procedural memory — agents update their own system prompts"** (ungated) — the direct conceptual overlap with mneme's compiler, minus the gate. MIT, Python-only.
- **Memobase:** evolving structured *user profile* (not raw facts). Win: profile UX. We win: not a governance/verifiability play.
- **Redis Agent Memory Server:** infra (sub-ms vector), composes *under* Mem0/LangMem — not a framework. Potential **substrate dependency**, not a rival.
- **MS Kernel Memory:** RAG + Azure IAM access control (≠ a safety gate). Enterprise lock-in; no temporal/graph/procedural.

### Tier-2 deep profiles (named-request batch, 2026-06-11)

### OpenViking — 🟠 ByteDance's filesystem context DB
- **What:** open-source "context database" for agents (OpenClaw-targeted) from the Volcengine/Viking team at **ByteDance**.
- **Architecture:** treats agent context as a **filesystem** — unifies memory + resources + **skills** in a hierarchical tree (not flat RAG); modular storage/retrieval/parsing/session; a built-in self-iteration loop compresses conversation + extracts long-term memory ("self-evolving"); **no external DB**. Cross-session user modeling + dialectic reasoning.
- **Numbers:** none published found.
- **vs mneme — they win:** ByteDance backing + distribution; **hierarchical context delivery + skills** is the single closest system to *both* mneme's wedge *and* the Context-Tree retrieval idea we're roadmapping; zero-infra.
- **vs mneme — we win:** no gate; filesystem "self-evolving" ≠ append-only + invalidate-and-supersede on an immutable log; no KV cartridge, no crypto-shred, no reproducible eval; skill reuse is **ungated**.

### RetainDB — 🟠 production memory infra (mirrors our retrieval plan)
- **What:** commercial persistent-memory infrastructure for agents (retaindb.com); markets "**SOTA on LongMemEval**."
- **Architecture:** hybrid retrieval = **vector + BM25 + reranking**, "no LLM-extraction overhead," cross-session/device, noise-filtering + signal reinforcement; positions on "decide what to remember, when old facts are invalidated."
- **Numbers:** self-claimed SOTA LongMemEval (unverified).
- **vs mneme — they win:** production polish; its retrieval stack is **exactly the multi-signal + reranker** parity work below; low overhead (no LLM extraction on write).
- **vs mneme — we win:** closed/commercial (mneme fully OSS); no gate/append-only/KV/provenance/crypto-shred shown; "invalidation" without an immutable-log substrate.

### MemU — 🟡 companion-focused, big community
- **What:** open-source memory framework for **AI companions** (NevaMind-AI); broad integrations (n8n, LangGraph, AutoGPT, Dify, LlamaIndex), FastAPI server + Go SDK.
- **Architecture:** multi-modal ingest → structured memory; RAG + LLM retrieval (semantic/hybrid/contextual); **usage-based prioritization + forgetting** (Ebbinghaus-style decay).
- **Numbers:** **92.09% LoCoMo** (self-reported).
- **vs mneme — they win:** companion UX, ecosystem reach, strong self-reported LoCoMo, decay/forgetting shipped.
- **vs mneme — we win:** no gate/append-only/KV/provenance/crypto-shred; "forgetting" is decay/deprioritize, not crypto-shred on an immutable log; no reproducible methodology. Different buyer (companions).

### MIRIX — 🟡 richest taxonomy + multimodal/screen
- **What:** research multi-agent memory system (arXiv 2507.07957, MIRIX AI); a packaged app monitors the screen and stores locally.
- **Architecture:** **six memory types** — Core / Episodic / Semantic / **Procedural** / Resource / Knowledge Vault — each with its own Manager, plus a Meta Memory Manager for routing; multimodal; local-first for privacy.
- **Numbers:** **85.4% LoCoMo**; ScreenshotVQA +35% vs RAG at 99.9% less storage.
- **vs mneme — they win:** the most complete memory **taxonomy** (incl. a Procedural type), multimodal/screenshot capture, local privacy.
- **vs mneme — we win:** MIRIX's "Procedural" is a *storage category*, not a **gated self-improvement loop**; no gate/append-only/KV/crypto-shred/provenance. (Good system to cite when explaining mneme's procedural is *gated*, not just a bucket.)

### Second Me — 🟡 personal AI identity (adjacent category)
- **What:** open-source **AI identity model** (Mindverse) — "train your AI self," 100% local/private; arXiv "AI-native Memory 2.0."
- **Architecture:** Hierarchical Memory Modeling (HMM) + Me-Alignment; Chat + Bridge modes; fully local deployment.
- **vs mneme — they win:** privacy-by-design (local-only), personal-identity framing.
- **vs mneme — we win:** it's a *personal AI identity* product, not agent-memory infra — no gate/append-only/KV/provenance/crypto-shred/procedural, not a benchmark player. Mostly a different market than mneme.

## Long-tail tracker (categories from COMPETITIVE-2026 §1)

Characterized from the TeleAI index, not individually deep-fetched — promote to a
Tier-1 profile if one gains traction.

| Bucket | Systems | Why we watch | Overlap |
|---|---|---|---|
| Git/append-only/versioned | Memov, taOSmd, Puppyone, Omnigraph, Mimir, archon-memory-core | nibble at BITEMP | append-only/versioning, no gate/KV/shred |
| Decay/forgetting | PowerMem (Ebbinghaus), Vestige (FSRS-6), Suyi (dual-temporal), widemem-ai | nibble at consolidation | heuristic decay vs mneme's causal/contribution GC |
| Self-evolving/procedural (research) | MemSkill, ProcMEM, MemRL, Mem-α, EvolveR, MUSE, AgentEvolver, SE-GA, EverOS, "Agent Knowledge Cycle" | nibble at PROC | none gate the self-improvement |
| KV / model-level memory | HERMES (KV-as-memory, video), Memory Decoder (pretrained plug-in), RecMem, PRIME | nibble at KV | not a governed log view |
| Profile / enterprise context | MemPalace (valid_from/to), memco, Hyper, Glia, MemClaw/Caura, CommonGround Kernel | enterprise governance pull | MemPalace closest on BITEMP; OpenViking profiled above |
| Multimodal/video memory | m3-agent, HippoMM, WorldMM, MemVerse, Visual Agentic Memory | adjacent market | out of mneme's current scope; MIRIX profiled above |

## The throughline — how mneme beats the field

1. **GATE + SHRED are unmatched by everyone.** Lead with "*safely* self-improving + can take it back," not "self-improving" (MemOS owns that now).
2. **atlan's verdict — "no framework provides enterprise governance / lineage / policy compliance" — is mneme's lane.** Own governed/verifiable/regulator-grade memory; the ~90 others fight over recall.
3. **Get on the board** (publish reproducible LoCoMo) so the moat is taken seriously, then **make the moat measurable** (verifiable-improvement-with-gate, auto-falsification recovery, provenance/erasure compliance).
4. **Absorb two ideas:** Hindsight's 4-strategy+cross-encoder retrieval and ByteRover's Context-Tree → mneme's RRF.

*Maintenance: when re-checking, update the matrix row + headline number + threat tier, and bump the "last refreshed" date. Add a Tier-1 profile when a long-tail system raises a real funding round, ships a reproduced benchmark, or lands distribution.*
