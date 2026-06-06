# mneme benchmarks — the "best memory layer" case, with real numbers

This file consolidates **measured** numbers from mneme's own harness (`mneme-bench`)
plus the live GPU/LLM runs, alongside **cited** competitor numbers from their
papers. Everything here is reproducible from the commands shown; nothing is
hand-waved.

> **Read the metric labels.** Two different things get measured in this space and
> they are *not interchangeable*:
> - **recall@k** — does a top-`k` retrieval contain the gold memory? A
>   retrieval-quality proxy. mneme measures this through its *real*
>   ingest→retrieve pipeline.
> - **QA accuracy / LLM-as-judge** — did the agent *answer* correctly given the
>   retrieved context? An end-to-end metric. mneme measures this live via Ollama;
>   competitor figures below are cited from their papers.
>
> Where numbers come from different setups (model, judge, split) they are **not
> directly comparable** — they're shown to map the landscape, not to claim a
> head-to-head win on someone else's metric. mneme's actual differentiator is the
> **capability matrix** (§2) and the **self-improvement loop** (§7), not a recall
> bake-off.

---

## 1. TL;DR

| Axis | mneme (measured) | Notes |
|---|---|---|
| Write path | **append p50 ~0.0017 ms, flat from 1k→105k** | Hard Rule #5; embedding/evolution are async, off the path |
| Query latency @105k | **p50 3.60 ms, p99 4.90 ms** | HNSW, sub-linear growth |
| Retrieval recall@10 @105k | **100%** (exact-gold needles, mock embedder) | deterministic synthetic corpus |
| Retrieval recall@10, real data | **98.1% SQuAD**, 99.2% @5k synthetic (fastembed) | real semantic embedder |
| End-to-end QA (LLM-judged) | **100% LOCOMO-mini, 100% LongMemEval-mini** | llama3.2 3B (local Ollama); curated mini-suites |
| **Real LLM agent loop** | **0% → 83%** with mneme (private facts) | llama3.2 decides its own tool calls over real MCP; see §5.1 |
| GPU KV cartridge (0.5B) | **107× warm prefill vs CPU** (same f32) | Apple M1 Pro, Metal |
| GPU KV cartridge (1.5B) | **~1577× prefill** (Metal bf16 3.34 ms vs CPU f32 5.27 s) | stacks GPU + bf16; see §6 |
| KV cartridge quantization | **q8 = 4.0× smaller**, same answer | 1.18 MB → 0.30 MB |

---

## 2. The capability matrix (the real differentiator)

Storage-shaped memory (Mem0, Zep, Letta, Cognee, A-MEM) remembers facts. mneme
adds the things they don't have. See the full matrix + per-row evidence in
[README.md → "How mneme compares"](README.md#️-how-mneme-compares). The rows that
matter:

- **Procedural self-improvement behind a non-bypassable safety gate** — outcomes
  compile into versioned policy artifacts; nothing commits without passing
  canaries + a safety probe + a non-negative objective delta (Hard Rule #1).
- **Bi-temporal, append-only, replayable log** — every index is a materialized
  view rebuildable by replay; you can reconstruct "what the agent knew at time T".
- **Crypto-shred erasure** over an append-only log (right-to-be-forgotten).
- **Counterfactual contribution scoring**, **self-falsifying probes**,
  **certified skill exchange**, **gated KV cartridges** — the frontier layer.

---

## 3. Substrate at scale (mock embedder, deterministic)

```bash
cargo run -p mneme-bench --release -- scale --sizes 1000,10000,50000,100000 --embedder mock
```

| Memories | Append/s | Append p50 | Index/s | Index p50 | Query p50 | Query p99 | recall@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,050 | 218,082 | 0.0022 ms | 7,648 | 0.13 ms | 0.88 ms | 1.76 ms | 100% |
| 10,503 | 385,546 | 0.0017 ms | 2,725 | 0.35 ms | 1.36 ms | 4.20 ms | 100% |
| 52,515 | 246,886 | 0.0018 ms | 1,625 | 0.60 ms | 2.21 ms | 2.96 ms | 100% |
| 105,030 | 299,564 | 0.0017 ms | 1,326 | 0.75 ms | 3.60 ms | 4.90 ms | 100% |

**Append latency is flat (~0.0017 ms p50) across a 100× corpus increase** — the
write path genuinely doesn't degrade with size. Query latency grows
*sub-linearly* (HNSW): 0.88 → 3.60 ms p50 from 1k→105k. Recall holds **100%** on
the exact-gold needle set through 105k. (Re-confirmed this session, full
1k→105k run: recall@10 **100%** at every size, and at 105,030 memories append
**327,392/s**, query **p50 3.45 ms / p99 4.11 ms** — matching the table within
run-to-run noise.)

---

## 4. Retrieval recall on **real** public data

```bash
cargo run -p mneme-bench --release --features fetch -- fetch --dataset squad --embedder fastembed
```

| Dataset | Embedder | recall@10 |
|---|---|---:|
| SQuAD (real) | fastembed (384-d) | **98.1%** |
| Synthetic @5,251 | fastembed (384-d) | 99.2% |
| LOCOMO-mini / LongMemEval-mini | mock / fastembed | 100% (small curated set) |

With a real semantic embedder the write path still stays fast (append p50
~0.00 ms — embedding runs in a pre-phase off the write path, Hard Rule #5).

---

## 5. End-to-end QA accuracy (live, LLM-judged)

```bash
# retrieve context → an LLM answers → an LLM judges (needs a local Ollama)
cargo run -p mneme-bench --release --features ollama -- qaeval --suite locomo
```

| Suite | Embedder | Judge / Answerer | QA accuracy | ms/q |
|---|---|---|---:|---:|
| LOCOMO-mini | fastembed | llama3.2 3B (Ollama, local) | **100% (10/10)** | 1,765 |
| LongMemEval-mini | fastembed | llama3.2 3B (Ollama, local) | **100% (10/10)** | 1,377 |

These are *curated mini-suites*, so 100% is a small-n result — run the full
splits for a headline number. They prove the **harness measures answer
correctness end-to-end**, not just retrieval.

### Cited competitor QA numbers (different metric/setup — landscape only)

| System | Suite | Metric | Score | Source |
|---|---|---|---:|---|
| Full-context (upper bound) | LOCOMO | LLM-as-Judge | 72.90% | Mem0 [arXiv:2504.19413](https://arxiv.org/abs/2504.19413), Tbl 2 |
| Mem0 (graph) | LOCOMO | LLM-as-Judge | 68.44% | same |
| Zep | LOCOMO | LLM-as-Judge | 65.99% | same |
| A-Mem | LOCOMO | LLM-as-Judge | 48.38% | same |
| Zep (gpt-4o) | LongMemEval | QA accuracy | 71.20% | Zep [arXiv:2501.13956](https://arxiv.org/abs/2501.13956), Tbl 2 |

### 5.1 Real LLM agent loop — does memory make a *real agent* better?

The most honest end-to-end test: a real LLM agent that **decides its own tool
calls**, talking to the **real `mneme-mcp` server over stdio** (the same
transport OpenClaw/Hermes use), answering questions about **private facts the
base model cannot know** (synthetic codenames, regions, dates seeded into mneme).

```bash
# needs Ollama running + `cargo build -p mneme-mcp --release`
python3 examples/agent_loop_eval.py
```

Measured live (llama3.2 via Ollama, 6 private-fact questions, ~54 s):

| Condition | Correct | Accuracy |
|---|---:|---:|
| **without memory** | 0 / 6 | **0%** |
| **with mneme** (agent calls `mneme_search`) | 5 / 6 | **83%** |
| **memory lift** | | **+83 pp** |

Without memory the model correctly answers "I don't know" to every private fact;
with mneme it searches, recovers the fact, and answers. The one miss was a
small-model artifact (llama3.2-3B emitted a malformed tool-call as plain text on
one question, so no search ran) — not a memory failure; every time the model
actually called the tool, mneme returned the fact and the answer was correct.
This is the wedge made concrete: **a real agent goes from 0% to 83% purely
because of the memory layer**, with the model — not a script — driving the tools.

---

## 6. Gated KV cartridges on GPU (Phase 12 — the moonshot)

The cartridge **is** the model's key/value cache — versioned, gated, and erasable
(a materialized view of the log). Measured live on an **Apple M1 Pro (Metal)**:

| Measurement | Result |
|---|---|
| 0.5B prefill, **GPU vs CPU** (same f32, warm) | **107×** (CPU 768.8 ms → Metal 7.2 ms) |
| 1.5B prefill, **Metal bf16 vs CPU f32** | **~1577×** (3.34 ms vs 5.27 s) — stacks GPU + bf16† |
| q8 quantization of the cartridge blob | **4.0× smaller** (1.18 MB → 0.30 MB), same answer |
| Deep model in half precision | **bf16** answers correctly (f16 overflows on 28 layers) |
| Self-consistency | cartridge generation == full-prompt generation (token-identical) |
| Erasure-by-recompile | a shredded fact can no longer be generated |

† The 1.5B figure stacks two effects (GPU-vs-CPU *and* bf16-vs-f32) because
candle's CPU backend has no bf16 matmul kernel, so f32 is the honest CPU
baseline. The clean same-precision number is the 0.5B 107×.

```bash
# reproduce (downloads the model on first run; needs Apple Metal)
cargo test -p mneme-kv --release --features candle-kv,metal -- --ignored --nocapture
```

---

## 7. The wedge — procedural self-improvement (gated)

The point storage-shaped memory can't reach: the agent gets **verifiably better**.
Outcomes (`mneme_record_outcome`) feed a GEPA-style compiler that proposes K
candidate policy artifacts, shadow-evaluates them, and **commits only if
`EvalReport::is_committable()`** (canaries + safety probe + Δobjective ≥ 0). The
learning curve over generations + the safety floor are live at
`/dashboard` (Phase 2 panel) and asserted by `mneme-bench`'s suite. A
canary-breaking proposal is *rejected* — improvement never trades away safety.

See [INTEGRATION.md](INTEGRATION.md) for the agent-side loop and
[COMPETITIVE.md](COMPETITIVE.md) for why a storage-shaped competitor can't copy
it without rebuilding its foundation.

---

## 8. Reproduce everything

```bash
cargo run -p mneme-bench --release -- scale --sizes 1000,10000,50000,100000 --embedder mock
cargo run -p mneme-bench --release --features fetch  -- fetch  --dataset squad --embedder fastembed
cargo run -p mneme-bench --release --features ollama -- qaeval --suite locomo
cargo run -p mneme-bench --release -- compare --suite locomo
cargo test  -p mneme-kv   --release --features candle-kv,metal -- --ignored --nocapture
```

## 9. Honesty / caveats

- Mock-embedder recall uses exact-gold needles — it measures the *pipeline*, not
  semantic quality; the real-embedder SQuAD number is the honest recall figure.
- QA mini-suites are small (n=10); treat 100% as "the harness works end-to-end",
  not a headline rank.
- GPU numbers are an M1 Pro; the first run pays one-time Metal shader compilation.
- Cited competitor numbers are from their papers under their setups — different
  model/judge/split, shown for landscape, not head-to-head.
