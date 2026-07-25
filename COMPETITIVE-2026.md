# Competitive landscape — agent memory, 2026

*Snapshot: June 2026. Companion to [`COMPETITIVE.md`](COMPETITIVE.md), which captured
the mid-2025 field (Mem0 / Zep / Letta / Cognee / A-MEM). That doc is not wrong, but
the field has since exploded from ~5 systems to dozens, so this file is the current
read. The maintained **per-competitor scorecard + deep profiles** live in
[`COMPETITORS.md`](COMPETITORS.md).*

> **Read every number here as self-reported and provisional.** Each vendor runs its
> own harness, answerer model, and judge, so scores conflict badly across sources
> (Mem0's own blog claims **92.5%** LoCoMo; ByteRover's leaderboard lists Mem0 at
> **66.9%**). Treat the leaderboard as marketing until reproduced. That noise is
> itself a strategic opening for mnesio (see Strategy §4).

## 0. Why this matters now

Agent memory became its own infrastructure category in 2026: its own benchmarks
(LoCoMo, LongMemEval, **BEAM** — the new ICLR-2026 1M/10M-token test), its own
research literature, and real funding (Mem0: YC S24, **$24M Series A**, Oct 2025).
One market estimate puts the segment at **~$6.3B (2026) → ~$28B (2030)**. The
category has also *fragmented* into ~8 sub-categories — which means "win agent
memory" is no longer a single race. Picking the right lane matters more than topping
any one leaderboard.

## 1. The field, by category

**1. Incumbent memory layers:** Mem0, Zep/Graphiti, Letta (MemGPT), Cognee,
LangMem, Supermemory, OpenMemory, TeleMem.

**2. Accuracy-leaderboard chasers (benchmark-led, new):**
- **ByteRover 2.0** — "Context Tree" hierarchical retrieval (domain→topic→subtopic);
  ~**92.2% LoCoMo** (tops its own board); coding-agent focus.
- **MemMachine** (arXiv 2604, Apr 2026) — "ground-truth-preserving," stores episodes
  **verbatim**, work done at retrieval; ~**0.917 LoCoMo** (gpt-4.1-mini); OSS.
- **Hindsight** — retain / recall / **reflect**; structured entity extraction; ~**89.6%**.
- **Honcho** (Plastic Labs) — reasoning-first user modeling + a background **"dream"**
  deduction loop; ~**89.9% LoCoMo** at ~5% context.

**3. Profile / user-model memory:** Memobase (evolving structured user profile),
OpenViking (cross-session modeling + dialectic reasoning), Second Me / Second Brain.

**4. "Memory OS" / self-evolving / procedural (overlap mnesio's wedge):**
- **MemOS** (MemTensor) — the most direct overlap: L1 traces → L2 policies → L3
  world-model → **crystallized skills + cross-task skill reuse**; OpenClaw plugin;
  ~35% token savings; ~75.8% LoCoMo.
- EverOS, "Agent Knowledge Cycle" (sessions→skills); research: MemSkill, ProcMEM,
  MemRL, Mem-α, EvolveR, MUSE, AgentEvolver, SE-GA.

**5. Schema'd / governed / "trustworthy" memory (overlap mnesio's positioning):**
- **Memori** (Memori Labs / GibsonAI) — "memory as **data with schema, constraints,
  and history**," from agent traces.
- **MemPalace** — explicit **temporal validity windows** (`valid_from`/`valid_to`).
- MemClaw/Caura, CommonGround Kernel (Postgres shared work-record), Hyper / Glia
  (enterprise "company brain" + permissions), **memco** (shared memory layer).

**6. Git-based / append-only / versioned (overlap mnesio's substrate):** Memov,
taOSmd (append-only transcript + typed temporal KG), Puppyone (auto-versioning),
Omnigraph (git-style graph), Mimir (Rust binary, SQLite+FTS5), archon-memory-core.

**7. Decay / forgetting-centric:** PowerMem (Ebbinghaus), Vestige (FSRS-6),
Suyi/溯忆 (dual-temporal + decay), widemem-ai, archon (active forgetting).

**8. KV-cache / model-level memory (overlap mnesio's cartridges):** HERMES (KV cache
as hierarchical memory, streaming video), Memory Decoder (pretrained plug-and-play
memory for LLMs), RecMem; also **PRIME** (predictive-retrieval engine).

*(The TeleAI "Awesome-Agent-Memory" index lists ~80 OSS + ~15 commercial systems;
the above is the load-bearing subset. Many entries are research papers or
OpenClaw/Claude-Code memory plugins, not production layers.)*

## 2. Benchmark snapshot (self-reported — see caveat)

| System | LoCoMo | Notes |
|---|---:|---|
| ByteRover 2.0 | ~92.2% | hierarchical Context Tree |
| MemMachine | ~91.7% | verbatim episodes, gpt-4.1-mini |
| Mem0 (token-efficient algo) | ~91.6–92.5% | own numbers; ~7k tokens/query |
| Honcho | ~89.9% | ~5% median context |
| Hindsight | ~89.6% | strong open-domain |
| Supermemory | ~85.4% | MCP-native |
| Zep / Graphiti | ~75–79.8% | temporal KG; 63.8% LongMemEval |
| MemOS | ~75.8% | + 35% token savings |
| **mnesio** | **not yet published** | retrieval recall 98.1% SQuAD; LOCOMO QA parked on a frontier answerer |

**mnesio is currently invisible on the metric buyers read — and behind on it.** The
SQuAD-100 QA number (66%, local 3B) is a weaker, non-comparable setup. Closing this
is Tier-1 below.

## 3. Per-pillar threat table — mnesio's moat vs the 2026 field

| mnesio pillar | Closest rivals | Matched? |
|---|---|---|
| **Non-bypassable safety gate** on self-improvement (`is_committable()`) | MemOS, MemSkill, EvolveR | ❌ All self-evolve *freely*; none gate. **mnesio-unique.** |
| **Append-only + bi-temporal** (invalidate-and-supersede) | Memov, taOSmd (git/append-only); MemPalace, Suyi (temporal validity) | ◑ Partial — git-trace *or* valid-from/to, not all three together. |
| **Gated KV cartridges** (KV cache as a gated, erasable, versioned view) | HERMES, Memory Decoder | ◑ KV-as-memory exists (video/streaming/pretrained), not as a governed log view. |
| **Provenance + crypto-shred erasure** on an immutable log | Memori (schema+history), Hyper (permissions) | ❌ "delete API" is common; cryptographic erasure reconciled with append-only is **mnesio-unique.** |
| **Procedural self-improvement** | **MemOS** (mainstreamed it), ProcMEM, Agent Knowledge Cycle | ⚠️ The *story* is now table stakes; the *gated/verifiable* version is not. |
| **Eval-as-product / reproducible numbers** | (everyone publishes marketing numbers) | ✅ Open lane — nobody is selling reproducibility. |

**Net:** every individual pillar is being chipped at, but **no competitor has the
combination**, and two pillars (the gate; crypto-shred-on-append-only) remain
genuinely unmatched. mnesio's moat is the *governed, verifiable substrate*, not raw
recall accuracy.

## 4. The threats, named

1. **Leaderboard invisibility + a real accuracy gap.** Everyone has a LoCoMo number;
   mnesio doesn't. This is the #1 commercial problem.
2. **MemOS commoditized the self-improvement narrative** (and has distribution: an
   OpenClaw plugin, a paper, MemTensor backing). mnesio can no longer lead with
   "self-improving memory" — only with "*safely* self-improving."
3. **Retrieval moved on.** Hierarchical Context-Tree traversal (ByteRover) and
   reasoning-first ingestion (Honcho) beat flat hybrid+graph on the hard LoCoMo
   categories (multi-hop, temporal).
4. **Token-efficiency is now a headline metric** (Honcho ~5% context, MemOS 35%
   savings, Mem0 ~7k tokens/query); mnesio doesn't publish a tokens/query number.

## 5. Strategy — how mnesio stays ahead

**Pick the lane.** Do **not** chase all 8 categories. Own **"governed, verifiable,
regulator-grade memory"** — the lane almost no one is contesting while ~80 systems
fight over recall accuracy and coding-agent UX.

**Tier 1 — get on the board (non-negotiable):**
1. Publish a real, reproducible **LoCoMo + LongMemEval** number (the parked
   OpenRouter run). Goal: *credible* (~88–92%), not #1.
2. Ship **`mnesio-bench` as a neutral, reproducible, methodology-printing harness** —
   "run every vendor's number yourself." In a field of contradictory self-reported
   scores, **reproducibility is the brand.**

**Tier 2 — make the moat measurable (new axes rivals can't pass):**
3. **Verifiable-improvement benchmark:** learning curve *with the gate rejecting a
   canary-breaking update* — nobody else can demo "got better AND provably safe."
4. **Staleness / auto-falsification recovery:** mnesio's Phase-11 probes auto-supersede
   a fact whose probe fails — directly answers Mem0's own admitted gap ("memories
   become confidently wrong").
5. **Provenance + erasure compliance demo:** "prove what it knew at T, then make a
   subject unreadable in live + historical replay" — for the regulated buyers the
   leaderboard ignores.

**Tier 3 — absorb the best new ideas (keep retrieval competitive):**
6. **Hierarchical / Context-Tree retrieval** folded into RRF (targets multi-hop +
   temporal).
7. Extend **dreaming** to generate *gated inferences* (Honcho-style), behind the gate.
8. **Own cost:** zero-LLM write path + KV cartridges → publish a **tokens/query**
   number; lead on "cheapest at scale, with the safety substrate."
9. **BEAM (1M/10M)** scale eval — mnesio's 105k-scale harness is most of the way there.

**Repositioning, one line:** everyone now says "self-improving memory" — only mnesio
can say **"self-improving *behind a gate*, that can *prove* what it knew and *take it
back*, with numbers you can reproduce."** The gate / provenance / erasure are exactly
the production gaps Mem0's own 2026 essay lists as unsolved.

## Sources

- [Awesome-Agent-Memory (TeleAI)](https://github.com/TeleAI-UAGI/Awesome-Agent-Memory) — the ~80+15 system index
- [mem0: State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026) — production gaps
- [mem0: AI Memory Benchmarks 2026](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)
- [ByteRover 2.0 benchmark](https://www.byterover.dev/blog/benchmark-ai-agent-memory)
- [MemOS (GitHub)](https://github.com/MemTensor/MemOS) · paper arXiv:2507.03724
- [MemMachine](https://memmachine.ai/) · arXiv:2604.04853
- [Honcho review](https://andrew.ooo/posts/honcho-plastic-labs-agent-memory-review/)
- [Memori Labs](https://memorilabs.ai/)
- [Top 10 AI Memory Products 2026](https://medium.com/@bumurzaqov2/top-10-ai-memory-products-2026-09d7900b5ab1)
- [Agent memory = new context layer (New2026)](https://new2026.medium.com/agent-memory-is-becoming-the-new-context-layer-hindsight-byterover-honcho-retaindb-openviking-849248d621e5)
- [particula: Mem0 vs Zep vs Letta vs Cognee](https://particula.tech/blog/agent-memory-frameworks-tested-mem0-zep-letta-cognee-2026)
- BEAM: *Beyond a Million Tokens* (ICLR 2026)

*Numbers self-reported; reproduce before quoting. Update cadence: re-snapshot each quarter.*
