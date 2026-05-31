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
numbers, and the personalization / multi-agent / privacy basics. P0–P2 below
close those while leaning into the moat (all shipped). **P3 — frontier bets**
is the layer past parity: six things only mneme's substrate can ship safely.

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

**Status: P0–P2 all shipped** (Phases 7–9 complete). The frontier layer (P3,
Phases 10+) is specced below and in `CLAUDE.md`.

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

## P3 — frontier bets (nobody else can ship these)

P0–P2 reached parity-plus-wedge. P3 is the *unique* layer: bets that are only
possible — or only *safe* — because mneme's system of record is an append-only,
replayable, bi-temporal log behind a non-bypassable safety gate. The honest
column is "why a competitor can't follow without rebuilding their foundation".
Sequenced execution lives in `CLAUDE.md` → "Frontier roadmap (Phase 10+)".

Thesis: **mneme is not a place to put facts — it's a memory that gets
*verifiably* better, can *prove* what it knew and when, and can *take it back*.**

| Bet (phase) | What it is | Why a storage-shaped competitor can't follow |
|---|---|---|
| **Causal memory / counterfactual GC** (10) | Replay outcomes with a memory masked → measure its causal contribution; GC by *provable* zero-contribution, not age. | Mem0/Zep/Letta mutate a DB in place — there's no past to replay, so they can only decay by heuristic. |
| **Self-falsifying memory ("memory with CI")** (11) | Memories carry re-checkable probes; a failed probe auto-supersedes the belief (history kept) + re-triggers evolution; retrieval returns belief + confidence + why. | Needs an eval substrate wired into writes *and* invalidate-and-supersede versioning. Overwrite-in-place systems have no falsification chain to show. |
| **Gated KV cartridges** (12, moonshot) | KV cache as a materialized view of the log — GEPA-compiled, gate-activated, crypto-shred-reconciled by recompile. KV-cache memory *minus* the reasons it's never productized. | A tensor blob is unauditable/un-erasable unless the *log*, not the blob, is the truth (Hard Rule #4). Prompt-caching vendors get latency only — never gated, versioned, *forgettable* KV. |
| **Certified skill exchange** (13) | Export a gated `PolicyArtifact` as a certificate (artifact + canaries + `EvalReport`); the importer re-runs *its own* gate before activation. Marketplace w/ network effects. | Nobody else has a gated unit of competence to certify; without `is_committable()` an imported "skill" is unverified text. |
| **Negative memory + dreaming** (14) | Learn gated *suppression* rules from bad outcomes (what *not* to retrieve) + a bounded offline "dream" pass that consolidates, prunes by Phase-10 contribution, re-anchors evolved notes. | Both are compiler extensions behind the gate; a system with no outcome loop can only ever store the positive. |
| **Regulator-grade provenance (black-box recorder)** (15) | Time-travel reconstruction ("what did the agent know at T") + provenance chains + verifiable erasure coexisting with an immutable log. Placeable under EU-AI-Act-style audit. | Mutable storage can't reconstruct a past state it overwrote, and can't prove erasure against a log it doesn't keep. |

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
