// @mnesio/sdk — TypeScript client for mnesio's HTTP surface.
//
// Zero runtime dependencies: relies on the global `fetch` shipped by
// Node ≥ 18. The DTOs mirror what `mnesio-server` returns under
// crates/mnesio-server/src/viz.rs; keep them in sync when fields evolve.
//
// The wedge in one call:
//   const client = new MnesioClient({ baseUrl: "http://localhost:7777" });
//   const skills = await client.skills();          // gated PolicyArtifacts
//   const results = await client.search("…", 5);   // hybrid retrieval
//   // Prepend skills[0].injection to your prompt, then send memories
//   // as context — same model gets a *gated-improved* skill on every call.

/** Server connection options. */
export interface MnesioClientOptions {
  /** e.g. "http://127.0.0.1:7777". No trailing slash required. */
  baseUrl: string;
  /** Optional bearer token; reserved for future auth wiring. */
  token?: string;
  /** Per-request timeout in ms. Default 10_000. */
  timeoutMs?: number;
  /** Optional fetch override (for tests). Defaults to the global `fetch`. */
  fetch?: typeof fetch;
}

export interface SearchHit {
  memory_id: string;
  content: string;
  tags: string[];
  score: number;
  /** `[(signal_name, raw_score), …]` — e.g. `[["vector",0.47],["bm25",1.5],["rrf",0.016]]`. */
  breakdown: [string, number][];
  source_id: string | null;
  source_title: string | null;
  position: number | null;
  is_invalidated: boolean;
}

export interface AnswerExcerpt {
  memory_id: string;
  /** Highlighted segments for client-side rendering. */
  segments: { kind: "plain" | "highlight"; text: string }[];
  retrieval_score: number;
  source_id: string | null;
  source_title: string | null;
  position: number | null;
}

export interface AnswerCard {
  query: string;
  excerpts: AnswerExcerpt[];
  citations: string[];
  prose: string | null;
}

export interface SearchResponse {
  query: string;
  k: number;
  embedder_id: string;
  embedder_dim: number;
  vector_hits: SearchHit[];
  bm25_hits: SearchHit[];
  hybrid_hits: SearchHit[];
  answer: AnswerCard;
}

/** Phase-9 (P2#9) — what the agent should *inject* into its prompt. */
export interface InjectableSkill {
  artifact_id: string;
  version: number;
  /** "SystemPrompt" | "Heuristic" | "Skill" | "RetrievalRule" | "Reflection" */
  kind: string;
  /** Plain-text content to fold into the prompt. */
  injection: string;
  /** Set only for `kind == "Skill"`. */
  lang: string | null;
  canary_count: number;
}

export interface SkillsResponse {
  tenant: string;
  count: number;
  skills: InjectableSkill[];
}

export interface ProfileHistoryEntry {
  value: string;
  set_at_ms: number;
  actor: string | null;
}

export interface ProfileAttribute {
  attribute: string;
  value: string;
  history: ProfileHistoryEntry[];
}

export interface ProfileResponse {
  tenant: string;
  user: string | null;
  attributes: ProfileAttribute[];
}

export interface AgentRow {
  agent: string;
  memories: number;
  grants_to: string[];
  is_system: boolean;
}

export interface AgentsResponse {
  agents: AgentRow[];
}

export interface IngestMetrics {
  enabled: boolean;
  backend: string;
  observations: number;
  facts_extracted: number;
  adds_committed: number;
  adds_rejected: number;
  updates: number;
  contradictions: number;
  refinements: number;
  noops: number;
  pii_redacted: number;
}

/** Thin error class so callers can `instanceof`. */
export class MnesioError extends Error {
  constructor(
    message: string,
    /** HTTP status, or 0 if the request never made it to the wire. */
    public readonly status: number,
    /** Raw response body if available. */
    public readonly body?: string,
  ) {
    super(message);
    this.name = "MnesioError";
  }
}

/** Client for the mnesio HTTP surface. Stateless; safe to share. */
export class MnesioClient {
  private readonly baseUrl: string;
  private readonly token?: string;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(opts: MnesioClientOptions) {
    if (!opts.baseUrl) {
      throw new Error("MnesioClient: baseUrl is required");
    }
    this.baseUrl = opts.baseUrl.replace(/\/$/, "");
    this.token = opts.token;
    this.timeoutMs = opts.timeoutMs ?? 10_000;
    this.fetchImpl = opts.fetch ?? globalThis.fetch;
    if (!this.fetchImpl) {
      throw new Error(
        "MnesioClient: no global fetch found. Use Node 18+ or pass `fetch` in options.",
      );
    }
  }

  // ---- public methods ----

  /**
   * Hybrid retrieval. Pass an `actor` to enforce the multi-agent ACL
   * (only own + system + granted memories will surface).
   */
  async search(
    query: string,
    k = 5,
    opts?: { actor?: string },
  ): Promise<SearchResponse> {
    const params = new URLSearchParams({ q: query, k: String(k) });
    if (opts?.actor) params.set("actor", opts.actor);
    return this.get<SearchResponse>(`/api/search?${params.toString()}`);
  }

  /**
   * The active, gated `PolicyArtifact`s the agent should inject into its
   * prompt. Combine with `search()` for the wedge in one round-trip
   * pattern.
   */
  async skills(): Promise<SkillsResponse> {
    return this.get<SkillsResponse>("/api/skills");
  }

  /** Stable per-subject attributes + bi-temporal history. */
  async profile(): Promise<ProfileResponse> {
    return this.get<ProfileResponse>("/api/profile");
  }

  /** Multi-agent attribution + the inter-agent grant matrix. */
  async agents(): Promise<AgentsResponse> {
    return this.get<AgentsResponse>("/api/agents");
  }

  /** Ingestion telemetry (extract → consolidate counters, PII redaction). */
  async ingestMetrics(): Promise<IngestMetrics> {
    return this.get<IngestMetrics>("/api/ingest/metrics");
  }

  /**
   * Convenience: fetch `skills()` + `search()` together and build a
   * prompt-ready bundle. Mirror this in any framework adapter (LangChain
   * `Retriever`, LlamaIndex `BaseRetriever`, CrewAI `Tool`).
   */
  async retrieveWithSkills(
    query: string,
    k = 5,
    opts?: { actor?: string },
  ): Promise<{ skills: InjectableSkill[]; hits: SearchHit[]; answer: AnswerCard }> {
    const [skillsRes, search] = await Promise.all([
      this.skills(),
      this.search(query, k, opts),
    ]);
    return {
      skills: skillsRes.skills,
      hits: search.hybrid_hits,
      answer: search.answer,
    };
  }

  // ---- internals ----

  private async get<T>(path: string): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const headers: Record<string, string> = { Accept: "application/json" };
      if (this.token) headers.Authorization = `Bearer ${this.token}`;
      const res = await this.fetchImpl(url, { signal: controller.signal, headers });
      const text = await res.text();
      if (!res.ok) {
        throw new MnesioError(
          `mnesio ${path} returned HTTP ${res.status}`,
          res.status,
          text,
        );
      }
      try {
        return JSON.parse(text) as T;
      } catch (parseErr) {
        throw new MnesioError(
          `mnesio ${path} returned non-JSON: ${(parseErr as Error).message}`,
          res.status,
          text,
        );
      }
    } catch (err) {
      if (err instanceof MnesioError) throw err;
      const msg = (err as Error).message ?? String(err);
      throw new MnesioError(`mnesio ${path} request failed: ${msg}`, 0);
    } finally {
      clearTimeout(timer);
    }
  }
}
