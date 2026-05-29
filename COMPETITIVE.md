# Competitive analysis — how mneme becomes the best memory layer

_Last updated: 2026-05. Sources are linked at the bottom._

## TL;DR

mneme's **wedge is unique**: nobody else combines (1) self-improving
*procedural* memory, (2) a *mechanically non-bypassable* safety/regression
commit gate, and (3) a *fully replayable append-only event log*. MemOS is
closest on self-evolution but has no safety gate; Zep has the bi-temporal
graph but no procedural learning; Hindsight reflects but doesn't gate.

The wedge wins the *vision*. What loses the *demo* is missing table-stakes:
ingestion-time fact extraction, contradiction resolution, standard benchmark
numbers, and the personalization / multi-agent / privacy basics. This doc is
the plan to close those while leaning into the moat.

## Where mneme already stands

| Capability | Field leader | mneme |
|---|---|---|
| Bi-temporal knowledge graph | Zep/Graphiti | ✅ Phase 4 — parity |
| Append-only auditable system-of-record | ~nobody | ✅ ahead |
| Hybrid multi-signal retrieval | Mem0 | ✅ vector+BM25+RRF; graph/rerank signal pending |
| Procedural / self-improving memory | MemOS, Hindsight | ✅ differentiated |
| **Safety gate on self-improvement** | **nobody** | ✅ **unique moat** |
| Eval-as-product | nobody | ✅ unique |
| Explainable retrieval / observability | 2026 must-have | ✅ ahead (`Hit.breakdown` + dashboard) |
| Rust / single-binary / local-first | — | ✅ differentiated |

## Gaps → roadmap (Phases 7–9)

### P0 — table stakes
1. **Ingestion extraction + consolidation** (`mneme-extract`): raw turn →
   atomic facts → ADD / UPDATE / NOOP with dedup. _(done — pure engine)_
2. **Contradiction → bi-temporal invalidation**: conflict detected at
   consolidation triggers supersede-and-invalidate. _(done — `UpdateReason::Contradiction`)_
3. **LOCOMO + LongMemEval harnesses** in `mneme-bench` + published numbers.
4. **Forgetting / decay / importance admission** (FadeMem / A-MAC style):
   score what's worth admitting; decay + consolidation re-anchoring.

### P1 — competitive parity
5. **Retrieval fusion**: fold graph-proximity + recency/importance into RRF;
   add a `Reranker` stage.
6. **Profile / persona memory**: first-class user-profile store.
7. **Multi-agent**: actor attribution + inter-agent ACLs within a tenant.
8. **PII redaction + crypto-shred forget**: redact before storage; reconcile
   the append-only log with GDPR erasure by dropping per-subject keys.

### P2 — amplify the moat
9. **Skill reuse at inference**: retrieve committed `PolicyArtifact::Skill`
   and inject — cross-task skill reuse *with* the safety gate competitors lack.
10. **Distribution/DX**: Node/TS SDK + LangChain/LlamaIndex/CrewAI adapters.

## Positioning

> **The only memory layer that gets your agent _better at its job_ over time —
> and proves, on every update, that it didn't get worse or less safe.**

## Sources
- Mem0 (arXiv:2504.19413); Mem0 — State of AI Agent Memory 2026; OpenMemory MCP
- Zep / Graphiti (arXiv:2501.13956); Zep vs Mem0 (Atlan)
- Letta — Sleep-time Compute, Memory Blocks
- Cognee — ECL pipeline; MemOS (github MemTensor/MemOS); MemMachine; Memori; ByteRover; Supermemory
- "Agent Memory Systems in 2026: What Actually Matters" (bymar.co)
- FadeMem (arXiv:2601.18642); When to Forget (arXiv:2604.12007)
