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

## Long-tail tracker (categories from COMPETITIVE-2026 §1)

Characterized from the TeleAI index, not individually deep-fetched — promote to a
Tier-1 profile if one gains traction.

| Bucket | Systems | Why we watch | Overlap |
|---|---|---|---|
| Git/append-only/versioned | Memov, taOSmd, Puppyone, Omnigraph, Mimir, archon-memory-core | nibble at BITEMP | append-only/versioning, no gate/KV/shred |
| Decay/forgetting | PowerMem (Ebbinghaus), Vestige (FSRS-6), Suyi (dual-temporal), widemem-ai | nibble at consolidation | heuristic decay vs mneme's causal/contribution GC |
| Self-evolving/procedural (research) | MemSkill, ProcMEM, MemRL, Mem-α, EvolveR, MUSE, AgentEvolver, SE-GA, EverOS, "Agent Knowledge Cycle" | nibble at PROC | none gate the self-improvement |
| KV / model-level memory | HERMES (KV-as-memory, video), Memory Decoder (pretrained plug-in), RecMem, PRIME | nibble at KV | not a governed log view |
| Profile / enterprise context | OpenViking, MemPalace (valid_from/to), memco, Hyper, Glia, MemClaw/Caura, CommonGround Kernel | enterprise governance pull | MemPalace closest on BITEMP |
| Multimodal/video memory | MIRIX, m3-agent, HippoMM, WorldMM, MemVerse, Visual Agentic Memory | adjacent market | out of mneme's current scope |

## The throughline — how mneme beats the field

1. **GATE + SHRED are unmatched by everyone.** Lead with "*safely* self-improving + can take it back," not "self-improving" (MemOS owns that now).
2. **atlan's verdict — "no framework provides enterprise governance / lineage / policy compliance" — is mneme's lane.** Own governed/verifiable/regulator-grade memory; the ~90 others fight over recall.
3. **Get on the board** (publish reproducible LoCoMo) so the moat is taken seriously, then **make the moat measurable** (verifiable-improvement-with-gate, auto-falsification recovery, provenance/erasure compliance).
4. **Absorb two ideas:** Hindsight's 4-strategy+cross-encoder retrieval and ByteRover's Context-Tree → mneme's RRF.

*Maintenance: when re-checking, update the matrix row + headline number + threat tier, and bump the "last refreshed" date. Add a Tier-1 profile when a long-tail system raises a real funding round, ships a reproduced benchmark, or lands distribution.*
