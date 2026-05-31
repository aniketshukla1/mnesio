# @mneme/sdk — Node / TypeScript SDK

Thin client for the [mneme](https://github.com/aniketshukla1/mneme) HTTP
surface. Zero runtime dependencies; targets Node ≥ 18 (uses the built-in
global `fetch`).

```bash
npm install @mneme/sdk
```

## Use it

```ts
import { MnemeClient } from "@mneme/sdk";

const mneme = new MnemeClient({ baseUrl: "http://127.0.0.1:7777" });

// One-shot: gated skills + hybrid memory retrieval in parallel.
const { skills, hits, answer } = await mneme.retrieveWithSkills(
  "what does my partner drink?",
  5,
);

// Build the prompt: prepend each injectable artifact, then memory context.
const system =
  skills.map((s) => s.injection).join("\n\n") +
  "\n\nContext:\n" +
  hits.map((h) => `- ${h.content}`).join("\n");
```

`skills` are post-gate `PolicyArtifact`s — every entry has passed the
mechanical safety gate (canaries 100%, safety probe passing, objective
Δ ≥ 0) and carries `version` + `canary_count` for client-side audit.

## Multi-agent ACL

Pass an `actor` to enforce the inter-agent read ACL (see Phase-8 in the
root README):

```ts
const results = await mneme.search("Globex pricing", 5, { actor: "analyst" });
```

Memories owned by other agents are filtered out unless the owner granted
read access.

## Framework adapters

The client is plain enough to wrap directly. A LangChain `BaseRetriever`:

```ts
import { BaseRetriever } from "@langchain/core/retrievers";
import { Document } from "@langchain/core/documents";
import { MnemeClient } from "@mneme/sdk";

export class MnemeRetriever extends BaseRetriever {
  lc_namespace = ["mneme"];
  constructor(private client: MnemeClient, private k = 5) { super(); }

  async _getRelevantDocuments(query: string) {
    const { hits } = await this.client.retrieveWithSkills(query, this.k);
    return hits.map(
      (h) =>
        new Document({
          pageContent: h.content,
          metadata: { memoryId: h.memory_id, score: h.score, tags: h.tags },
        }),
    );
  }
}
```

A LlamaIndex retriever follows the same shape (`BaseRetriever._retrieve`
returns `NodeWithScore[]`), and a CrewAI tool wraps `retrieveWithSkills`
in a `Tool` whose `func` returns a string.

## API surface

All methods return typed DTOs that mirror `mneme-server`'s JSON exactly
(see `src/index.ts`). The endpoints are:

| Method | Endpoint | Purpose |
|---|---|---|
| `search(q, k, { actor? })` | `GET /api/search` | Hybrid retrieval, ACL-filtered |
| `skills()` | `GET /api/skills` | Active gated `PolicyArtifact`s ready to inject |
| `retrieveWithSkills(q, k, { actor? })` | both, in parallel | Prompt-ready bundle |
| `profile()` | `GET /api/profile` | Stable per-subject attributes + bi-temporal history |
| `agents()` | `GET /api/agents` | Multi-agent attribution + grant matrix |
| `ingestMetrics()` | `GET /api/ingest/metrics` | Extract / consolidate / PII-redaction counters |

## Build + test

```bash
cd sdk/node
npm install
npm run build
npm test
```

Tests run against a stub `fetch` — no running mneme-server required.

## License

Apache-2.0.
