// Tests for @mneme/sdk. Uses Node's built-in test runner — zero
// external test deps. Drives the client against a stub `fetch` so the
// suite runs offline (no running mneme-server required).

import { test } from "node:test";
import assert from "node:assert/strict";
import { MnemeClient, MnemeError } from "./index.js";

/**
 * Build a stub fetch that responds with a fixed JSON body for the path
 * substring it's keyed on, and records every URL it saw so tests can
 * assert routing + query-string composition.
 */
function stubFetch(routes: Record<string, unknown>): {
  fetch: typeof fetch;
  calls: string[];
} {
  const calls: string[] = [];
  const fn: typeof fetch = async (input) => {
    const url = typeof input === "string" ? input : input.toString();
    calls.push(url);
    for (const [match, body] of Object.entries(routes)) {
      if (url.includes(match)) {
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
    }
    return new Response(`no stub for ${url}`, { status: 404 });
  };
  return { fetch: fn, calls };
}

test("constructor rejects empty baseUrl", () => {
  assert.throws(() => new MnemeClient({ baseUrl: "" }));
});

test("constructor strips trailing slash from baseUrl", async () => {
  const stub = stubFetch({ "/api/search": searchFixture() });
  const c = new MnemeClient({ baseUrl: "http://h:7/", fetch: stub.fetch });
  await c.search("revenue");
  assert.ok(stub.calls[0].startsWith("http://h:7/api/search"));
});

test("search composes q + k + optional actor", async () => {
  const stub = stubFetch({ "/api/search": searchFixture() });
  const c = new MnemeClient({ baseUrl: "http://h:7", fetch: stub.fetch });
  await c.search("hello world", 8, { actor: "scout" });
  const url = stub.calls[0];
  assert.match(url, /q=hello\+world/);
  assert.match(url, /k=8/);
  assert.match(url, /actor=scout/);
});

test("skills() returns parsed body", async () => {
  const stub = stubFetch({
    "/api/skills": {
      tenant: "demo",
      count: 1,
      skills: [
        {
          artifact_id: "01ABC",
          version: 5,
          kind: "SystemPrompt",
          injection: "Be concise.",
          lang: null,
          canary_count: 2,
        },
      ],
    },
  });
  const c = new MnemeClient({ baseUrl: "http://h:7", fetch: stub.fetch });
  const r = await c.skills();
  assert.equal(r.count, 1);
  assert.equal(r.skills[0].kind, "SystemPrompt");
  assert.equal(r.skills[0].version, 5);
});

test("retrieveWithSkills fans out + merges", async () => {
  const stub = stubFetch({
    "/api/skills": {
      tenant: "demo",
      count: 1,
      skills: [
        { artifact_id: "x", version: 1, kind: "Heuristic", injection: "If X then Y", lang: null, canary_count: 0 },
      ],
    },
    "/api/search": searchFixture(),
  });
  const c = new MnemeClient({ baseUrl: "http://h:7", fetch: stub.fetch });
  const bundle = await c.retrieveWithSkills("revenue", 3);
  assert.equal(bundle.skills.length, 1);
  assert.equal(bundle.hits.length, 1);
  assert.ok(bundle.answer);
  // Two requests were issued (one for each endpoint).
  assert.equal(stub.calls.length, 2);
});

test("HTTP 500 surfaces as MnemeError with status + body", async () => {
  const fn: typeof fetch = async () =>
    new Response("boom", { status: 500 });
  const c = new MnemeClient({ baseUrl: "http://h:7", fetch: fn });
  await assert.rejects(
    c.search("x"),
    (err: unknown) => {
      assert.ok(err instanceof MnemeError);
      assert.equal((err as MnemeError).status, 500);
      assert.equal((err as MnemeError).body, "boom");
      return true;
    },
  );
});

test("non-JSON 200 surfaces as MnemeError with status=200", async () => {
  const fn: typeof fetch = async () =>
    new Response("oh hai", { status: 200 });
  const c = new MnemeClient({ baseUrl: "http://h:7", fetch: fn });
  await assert.rejects(c.search("x"), (err: unknown) => {
    assert.ok(err instanceof MnemeError);
    assert.equal((err as MnemeError).status, 200);
    return true;
  });
});

test("profile() + agents() + ingestMetrics() route correctly", async () => {
  const stub = stubFetch({
    "/api/profile": { tenant: "demo", user: "alice", attributes: [] },
    "/api/agents": { agents: [] },
    "/api/ingest/metrics": {
      enabled: true,
      backend: "demo",
      observations: 3,
      facts_extracted: 3,
      adds_committed: 1,
      adds_rejected: 0,
      updates: 1,
      contradictions: 1,
      refinements: 0,
      noops: 1,
      pii_redacted: 0,
    },
  });
  const c = new MnemeClient({ baseUrl: "http://h:7", fetch: stub.fetch });
  const p = await c.profile();
  assert.equal(p.user, "alice");
  const a = await c.agents();
  assert.equal(a.agents.length, 0);
  const m = await c.ingestMetrics();
  assert.equal(m.observations, 3);
});

// ---- fixtures ----

function searchFixture() {
  return {
    query: "revenue",
    k: 3,
    embedder_id: "mock-v1-d32",
    embedder_dim: 32,
    vector_hits: [],
    bm25_hits: [],
    hybrid_hits: [
      {
        memory_id: "01ABC",
        content: "Widget Inc Q3 revenue up 22%",
        tags: ["earnings"],
        score: 0.016,
        breakdown: [["bm25", 1.59], ["rrf", 0.016]],
        source_id: null,
        source_title: null,
        position: null,
        is_invalidated: false,
      },
    ],
    answer: { query: "revenue", excerpts: [], citations: [], prose: null },
  };
}
