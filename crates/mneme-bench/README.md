# mneme-bench

Eval-as-product harness for mneme. Two jobs in one crate:

1. **Procedural-compiler evaluation** (`run` / `compare`) — does the agent get
   *better at a task* over generations? Emits a learning curve with a safety
   floor; powers the CI regression gate.
2. **Memory-layer evaluation** (`memeval` / `scale` / `fetch`) — does the
   *memory* recall the right thing, fast, at scale? Runs the real
   `FjallEventLog → VectorView + Bm25View → HybridRetriever` path end to end.

All recall metrics are computed through the *actual* ingest→retrieve pipeline —
no mocked retriever, no shortcut. The only pluggable piece is the embedder
(`mock` for offline/CI, `fastembed` for real semantic quality).

---

## Quick start

```bash
# learning curve for the procedural compiler (default subcommand)
cargo run -p mneme-bench -- run --suite gsm8k --max-versions 6

# memory recall@k over a built-in suite (offline, mock embedder)
cargo run -p mneme-bench -- memeval --suite locomo --k 10 --embedder mock

# large-scale load test: throughput + latency percentiles + recall
cargo run -p mneme-bench --release -- scale --sizes 1000,10000,50000 --embedder mock

# REAL public benchmark (SQuAD) — needs the `fetch` feature + network on first run
cargo run -p mneme-bench --features fetch --release -- \
  fetch --dataset squad --rows 2000 --k 10 --embedder fastembed
```

`--help` prints the full option matrix for every subcommand.

---

## Subcommands

### `run` — procedural learning curve
Iterates the compiler against a suite (`gsm8k` | `humaneval`), reflecting →
proposing → shadow-evaluating → gating each generation. Emits the per-version
objective score and the safety-probe pass rate. Output: `csv | json | html |
markdown`. Use `--regression-threshold N` in CI to exit non-zero if the curve
falls more than `N` below baseline or the safety probe regresses.

### `compare` — A vs B
Scores two prompt bodies (`--baseline` / `--candidate`) against a fixed suite
and reports the delta. The same gate logic that guards a real commit.

### `memeval` — recall@k on a curated suite
Ingests a haystack of memories, asks each question, and checks whether any of
the top-`k` retrieved memories contains the gold answer span. Suites: `locomo`,
`longmemeval` (LOCOMO / LongMemEval style, shipped in `data/`). `--min-recall N`
gates CI.

### `scale` — load test on a synthetic corpus
Generates a deterministic synthetic corpus (see below) and drives it through the
real path, **separating the two write phases** so the numbers reflect mneme's
architecture:

- **Append** — the user-facing write (Hard Rule #5: <5ms target). Timed alone.
- **Index build** — HNSW + BM25 apply, which the server does *asynchronously
  off the write path*. Timed separately.

Reports throughput and p50/p95/p99 latency for each phase, plus query latency
and recall@k. CSV via `--out`.

### `compete` — competitive comparison
Runs mneme's *measured* retrieval recall@k on the LOCOMO/LongMemEval mini-suites
and assembles a Markdown report with (1) a factual capability matrix vs
Mem0/Zep/Letta/A-MEM and (2) cited competitor end-to-end QA scores from the
Mem0 (arXiv:2504.19413) and Zep (arXiv:2501.13956) papers. The report leads
with a methodology note: recall@k (retrieval-quality proxy) and QA accuracy are
*different metrics* and are never presented as one ranking.

### `edge` — adversarial / edge-case stress
Drives the real ingest→retrieve→replay path with hostile inputs (degenerate /
syntax-laden / unicode queries, huge & empty content, scope-isolation extremes,
supersede-keeps-history, tombstone-heavy indexes, dim mismatch, replay
determinism, concurrent writes) and asserts the seven hard-rule invariants.
Exits non-zero on any violation (CI gate). This suite found and fixed a real
BM25 bug where adversarial queries 500'd instead of returning empty.

### `fetch` — real public benchmark *(feature `fetch`)*
Downloads a real dataset from the Hugging Face datasets-server, projects it into
a recall suite, caches it to disk, and runs `memeval` over it. Two datasets:
- `--dataset squad` (single-hop): each context → a memory (deduplicated), each
  question/answer-span → a recall pair.
- `--dataset hotpotqa` (multi-hop): each of a row's context paragraphs → a
  memory; the answer span must be found across them. yes/no comparison answers
  are skipped (not retrievable spans).

Subsequent runs are offline from the cache. `--force` re-downloads;
`--fetch-only` caches without evaluating.

The `fetch` feature is the *only* thing that pulls a network dependency
(`reqwest`); the default build stays network-free.

---

## The synthetic generator (`gen.rs`)

Deterministic, dependency-free (a `SplitMix64` PRNG), reproducible from a
`--seed`. Produces:

- **Needles** — memories carrying a unique, sentinel-delimited gold token
  (`ZNDL{qid:09}Z`). The fixed width guarantees no token is a prefix of another
  and each appears in exactly one memory, so recall@k is *unambiguous at any
  scale*.
- **Distractors** — the bulk of the corpus.
- **Evolution chains** (~n/33) and **contradictions** (~n/50), with parents
  drawn only from distractors so they never pollute a gold needle.

This is what lets the scale harness report an honest recall number on a corpus
of any size — a needle either was retrieved or it wasn't, with no fuzzy match.

---

## Measured results

All numbers below are **measured**, not projected — `mneme-bench` on a 2021
M1-class laptop (8 cores, 16 GB), release build.

### Real benchmarks — SQuAD + HotpotQA, recall@10

| Dataset | Embedder | Memories | Questions | recall@10 | ms/query |
|---|---|---:|---:|---:|---:|
| SQuAD v1.1 (single-hop) | `fastembed` | 315 | 2,000 | **98.1%** | 9.24 |
| SQuAD v1.1 (single-hop) | `mock` | 315 | 2,000 | 93.9% | 1.81 |
| HotpotQA (multi-hop) | `fastembed` | 9,227 | 941 | **88.7%** | 17.97 |
| HotpotQA (multi-hop) | `mock` | 9,227 | 941 | 83.4% | 8.61 |

Real semantic embeddings lift recall +4.2 pts on single-hop SQuAD and +5.3 pts
on the harder multi-hop HotpotQA over keyword-only.

### Scale sweep — synthetic corpus, mock embedder

| Memories | Append/s | Append p50 | Index/s | Index p50 | Query p50 | Query p99 | recall@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,050 | 218,082 | 0.0022 ms | 7,648 | 0.13 ms | 0.88 ms | 1.76 ms | 100% |
| 10,503 | 385,546 | 0.0017 ms | 2,725 | 0.35 ms | 1.36 ms | 4.20 ms | 100% |
| 52,515 | 246,886 | 0.0018 ms | 1,625 | 0.60 ms | 2.21 ms | 2.96 ms | 100% |
| 105,030 | 299,564 | 0.0017 ms | 1,326 | 0.75 ms | 3.60 ms | 4.90 ms | 100% |

**Append latency is flat (~0.0017 ms p50) across a 100× size increase** — the
write path doesn't degrade with corpus size. Index build is HNSW-bound (the
BM25 commit is batched once per build, not per doc) and degrades gracefully.
Query latency grows *sub-linearly* (HNSW). Recall holds at **100%** on the
exact-gold needle set through 105k memories.

---

## Tests

```bash
cargo test -p mneme-bench                  # 24 tests (offline, no network)
cargo test -p mneme-bench --features fetch # 31 tests (+7 real-data loaders)
```

The fetch-feature tests cover the loaders' pure parts (cache-key safety,
URL-encoding, spec defaults, missing-cache handling, and SQuAD/HotpotQA row
projection) without hitting the network.

---

## Caching & environment

- Downloaded suites cache to `crates/mneme-bench/data/cache/` (gitignored).
  Override with `MNEME_BENCH_CACHE=/path`.
- `fastembed` downloads its model to `.fastembed_cache/` on first run (~100 MB,
  gitignored).
