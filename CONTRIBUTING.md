# Contributing to mneme

Thanks for considering a contribution. mneme is a self-improving long-term
memory layer for AI agents, and it has a couple of conventions that are stricter
than the average Rust project. They exist for a reason — please read this before
opening a PR.

## The seven hard rules

These are architectural invariants, not style preferences. A PR that violates one
will be asked to change regardless of how good the rest is.

1. **Nothing procedural commits without passing `EvalReport::is_committable()`.**
   Canaries, a safety probe, and a non-negative objective delta. This is the
   regression guard the literature omits. No exceptions, no "temporary" bypass —
   an integration test proves that setting every configurable gate threshold to
   its weakest value still cannot get under the baseline.
2. **Never overwrite history.** Updating a fact *invalidates and creates a new
   bi-temporal version* (a `parent` pointer plus a new `BiTemporal`). The event
   log is append-only and immutable.
3. **Scope is a security boundary.** Procedural learning and evolution must never
   cross a `Scope` without an explicit aggregation/anonymization step. Use
   `Scope::contains` for every cross-entity read.
4. **The event log is the single system of record.** Every index — vector, BM25,
   graph, procedural, KV — is a materialized view that must be rebuildable by
   replaying events. A view must never hold state that isn't derivable from the log.
5. **The write path stays fast (<5 ms target).** Embedding, evolution, and graph
   extraction are async behind bounded queues. Never block a write on an LLM call.
6. **Bound the cascades.** Evolution respects `EvolveConfig` caps. A-MEM has no
   convergence guarantee; our bounds are what replace it.
7. **Every external dependency sits behind a trait** (`LlmClient`, `Embedder`,
   `EventLog`, `MaterializedView`, `Retriever`, `KvBackend`, `Signer`, `Cipher`).
   Providers stay swappable.

## Honest numbers

This is the one cultural rule worth stating explicitly: **every number in this
repo is measured, reproducible, and reported with its caveats.** If a result is
mixed, it gets published as mixed. If a feature doesn't meet its "done when"
criterion, it does not get marked done, and it does not get switched on by
default.

Concretely, when you contribute a benchmark result:

- Give the exact command that reproduces it, including the embedder and `k`.
- If the metric is noisy, **run it repeatedly and report the distribution**, not
  your best run. HNSW index construction is randomized; a single run of a small
  suite can move ±10pp for reasons that have nothing to do with your change.
- Compare arms **paired** — ingest the corpus once and evaluate both
  configurations against the same index. Unpaired comparisons let build
  randomness masquerade as a real effect.
- Never quote a number you haven't run yourself.

## Development workflow

```bash
# run everything the way CI does
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Both `fmt` and a **warning-free** `clippy` are required before a commit. CI
additionally runs the benchmark gates (recall floors on LOCOMO / LongMemEval, a
synthetic scale floor, the adversarial `edge` suite, and KV cartridge parity +
erasure), and compile-checks each feature-gated KV backend.

The fastest way to see the system running:

```bash
make demo          # or: docker compose up --build
```

Both start the server with the mock embedder and demo data — zero external
downloads, dashboard on <http://localhost:7777>.

## Code conventions

- **Every public item gets a doc comment.** Where it implements a specific part
  of the design, reference the section (e.g. "report §3").
- Mark unfinished work with `// TODO(phase-N):` so the build order stays visible.
- Tests live next to the code in `#[cfg(test)] mod tests`. Storage tests use a
  temp directory keyed by a fresh ULID and clean up after themselves.
- Write code that reads like the code around it — match the surrounding comment
  density, naming, and idiom.
- **Conventional commits**: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`,
  `ci:`, `chore:`. Scope them where useful (`feat(index):`).

## Build order

The workspace is built in phases, in sequence, each with a hard "done when"
criterion. Please don't skip ahead — if a change can't demonstrate its phase's
criterion, the right move is to stop and instrument rather than pile on more
features. The current phase map lives in the [roadmap](website/src/content/docs/reference/roadmap.mdx).

## Pull requests

- Keep a PR to one coherent change. Two unrelated fixes are two PRs.
- Say what you measured, not just what you wrote. If it changes retrieval or the
  procedural loop, include before/after numbers and the command you ran.
- New behaviour needs a test. Bug fixes need a test that fails without the fix.
- If you touched a hard rule's enforcement, say so explicitly in the PR body.

## Reporting bugs

Open an issue with the smallest reproduction you can manage — ideally a failing
test. Include the mneme version or commit, your OS, and the relevant feature
flags, since several subsystems (KV backends, real crypto, embedders) are behind
feature gates and behave differently.

## Security

Please don't file security problems as public issues — see
[SECURITY.md](SECURITY.md).

## Licence

By contributing, you agree that your contributions are licensed under the
Apache-2.0 licence that covers this project.
