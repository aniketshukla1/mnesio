//! Synthetic data writer that tells a small visual story.
//!
//! Drops ~20 memories spanning finance, product, ops, competition, people,
//! strategy, and customer signal — enough content to exercise both BM25
//! (literal keywords) and vector search (semantic neighbours like
//! *income → revenue*, *rival → competitor*, *Europe → EMEA*). Two
//! evolution chains and a handful of explicit `links` add the graph
//! structure the live view is designed to visualise.
//!
//! The writer never embeds inline anymore — it just appends the
//! `MemoryWritten` events. The async embedding worker picks them up and
//! emits `MemoryEmbedded` events, which the vector view then consumes.
//! That keeps the write path < 5ms (Rule #5) even when the embedder is
//! something heavy like FastEmbed.

use mneme_core::entity::{Memory, Provenance, Source};
use mneme_core::event::{ChangeSet, Event, LogEntry};
use mneme_core::traits::MaterializedView;
use mneme_core::types::{new_id, BiTemporal, MemoryRef, SourceRef};
use mneme_core::{EventLog, Scope};
use mneme_index::{Bm25View, Chunker, ParagraphChunker, ProfileView, VectorView};
use std::sync::Arc;
use std::time::Duration;

/// Pace between memory writes. Slow enough that a viewer can watch each
/// arrival; fast enough that the whole demo finishes in ~20 seconds.
const TICK: Duration = Duration::from_millis(700);

/// Spawns the demo writer onto the current Tokio runtime.
pub fn spawn(
    log: Arc<dyn EventLog>,
    vector: Arc<VectorView>,
    bm25: Arc<Bm25View>,
    profile: Arc<ProfileView>,
) {
    tokio::spawn(async move {
        if let Err(e) = run(log, vector, bm25, profile).await {
            tracing::error!(error = %e, "demo writer failed");
        }
    });
}

/// One memory in the seeded corpus. `links_to` is a list of zero-based
/// indices into earlier corpus entries — at write time we resolve them to
/// real `MemoryRef`s and put them on the memory's `links` field.
struct Seed {
    content: &'static str,
    tags: &'static [&'static str],
    links_to: &'static [usize],
}

const fn s(
    content: &'static str,
    tags: &'static [&'static str],
    links_to: &'static [usize],
) -> Seed {
    Seed {
        content,
        tags,
        links_to,
    }
}

/// The corpus the demo writer streams in. Diverse domains + paraphrased
/// vocabulary (income/revenue, rival/competitor, Europe/EMEA) so vector
/// search has work to do beyond literal keyword overlap.
const CORPUS: &[Seed] = &[
    // --- finance ---
    s(
        "Acme Corp reported Q3 earnings beating consensus by 12%. Revenue grew 18% YoY driven by enterprise SaaS expansion, while operating margins compressed 80 bps on higher cloud spend.",
        &["earnings", "q3", "acme", "revenue"],
        &[],
    ),
    s(
        "Widget Inc Q3 results: revenue up 22% YoY led by EMEA recovery and strong mid-market traction. Management raised FY guidance for the second consecutive quarter.",
        &["earnings", "widget-inc", "emea", "guidance"],
        &[],
    ),
    s(
        "Operating cash flow improved sequentially to $142M in Q3, well ahead of the $115M consensus, on better collections and tighter working capital discipline.",
        &["cash-flow", "q3", "operations"],
        &[0],
    ),
    s(
        "Gross margin held steady at 76.2% despite component cost pressure. Pricing discipline in renewals offset roughly half of the unit-economics drag.",
        &["margin", "pricing", "q3"],
        &[],
    ),
    // --- product / engineering ---
    s(
        "Launched the v3.4 platform release on schedule. Key shipping items: SSO via SAML 2.0, role-based access control rework, and an audit log export API. Early enterprise customer feedback positive.",
        &["product", "release", "v3.4", "enterprise"],
        &[],
    ),
    s(
        "The new mid-market product line goes GA in late Q4. Pilot customers report 30% faster onboarding compared to the legacy stack; pricing simplified to three tiers.",
        &["product", "mid-market", "q4", "onboarding"],
        &[4],
    ),
    s(
        "Engineering org grew 18% YoY to 412 engineers. Hiring concentrated in platform infra and ML systems. Voluntary attrition remains below 9% — a multi-year low.",
        &["people", "engineering", "hiring", "attrition"],
        &[],
    ),
    // --- operations / supply chain ---
    s(
        "Supply chain stabilizing into Q4 after the H1 component shortage. Lead times are back within 6 to 8 weeks. The Vietnam backup supplier came online ahead of schedule.",
        &["operations", "supply-chain", "q4"],
        &[],
    ),
    s(
        "Manufacturing yield improved 240 bps QoQ following the automated optical inspection rollout at the Penang facility. Defect escape rate at a four-quarter low.",
        &["operations", "manufacturing", "yield"],
        &[7],
    ),
    // --- market / competition ---
    s(
        "Primary competitor lost 3 points of market share in the SMB segment per the latest IDC tracker. Win rate improved markedly in head-to-head deals.",
        &["competition", "market-share", "smb"],
        &[],
    ),
    s(
        "A new open-source entrant is gaining developer-led adoption from the bottom up. Minimal impact on the enterprise pipeline so far — worth watching into next year.",
        &["competition", "open-source", "developers"],
        &[9],
    ),
    s(
        "Customer NPS climbed to 58 from 51 last quarter. Detractors clustered around onboarding complexity, which the upcoming v3.5 simplification directly addresses.",
        &["customer", "nps", "onboarding"],
        &[5],
    ),
    // --- people & leadership ---
    s(
        "CEO confirmed prior FY guidance on the analyst call and reiterated the commitment to 25%+ revenue growth through FY26. Tone notably more bullish than last quarter.",
        &["guidance", "ceo", "growth", "fy26"],
        &[1],
    ),
    s(
        "Hired a new Chief Revenue Officer from a Series D peer. Start date end of Q4; mandate is to rebuild enterprise go-to-market and shorten the average sales cycle.",
        &["people", "hiring", "cro", "gtm"],
        &[],
    ),
    s(
        "The Bangalore engineering team expanded to 78 from 54 in the last six months. Local leadership bench is growing; first VP-level promotion announced internally last week.",
        &["people", "bangalore", "engineering", "hiring"],
        &[6],
    ),
    // --- strategy / risk ---
    s(
        "Margin compression in EMEA threatens FY guidance if the FX environment stays adverse. The hedging program has been extended an additional two quarters as insurance.",
        &["margin", "emea", "risk", "fx", "guidance"],
        &[1, 3],
    ),
    s(
        "Cloud infrastructure costs are growing faster than topline. A working group is chartered to evaluate reserved-capacity commitments versus the current spot-heavy posture.",
        &["cost", "cloud", "infrastructure", "strategy"],
        &[0],
    ),
    s(
        "Regulatory inquiry in Germany regarding data residency. External counsel engaged; provisioned $2M for potential resolution. Disclosure tracked through audit committee.",
        &["regulatory", "germany", "compliance", "risk"],
        &[],
    ),
    // --- customer signal ---
    s(
        "Top 20 customer health scores all green for the third consecutive quarter. Renewal pipeline for FY26 is already 65% covered — well ahead of the historical pace.",
        &["customer", "health", "renewal", "fy26"],
        &[],
    ),
    s(
        "Two strategic logos announced as public references: GlobalBank EU and Acme Industries. The co-marketing motion launches in Q1 with case studies and a joint webinar.",
        &["customer", "references", "marketing"],
        &[],
    ),
];

async fn run(
    log: Arc<dyn EventLog>,
    vector: Arc<VectorView>,
    bm25: Arc<Bm25View>,
    profile: Arc<ProfileView>,
) -> anyhow::Result<()> {
    let scope = Scope {
        tenant: "demo".into(),
        user: Some("alice".into()),
        session: None,
    };

    // Stream the corpus, recording each memory's ref so later seeds can
    // link to it.
    let mut refs: Vec<MemoryRef> = Vec::with_capacity(CORPUS.len());
    for (idx, seed) in CORPUS.iter().enumerate() {
        let links: Vec<MemoryRef> = seed
            .links_to
            .iter()
            .filter_map(|i| refs.get(*i).copied())
            .collect();
        let m = synth_memory(seed.content, seed.tags, &scope, None, 0, links);
        refs.push(MemoryRef(m.id));
        publish(
            Event::MemoryWritten(m),
            log.as_ref(),
            vector.as_ref(),
            bm25.as_ref(),
        )
        .await?;
        tracing::debug!(idx, "demo: wrote seed memory");
        tokio::time::sleep(TICK).await;
    }

    // --- evolution chain 1: revenue restatement ---
    // The original "Acme Q3 earnings" memory (idx 0) gets a correction.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let acme_parent = refs[0];
    let corrected = synth_memory(
        "Acme Q3 revenue growth was 16% YoY, not 18%. Pre-release reconciliation error caught by audit; all other Q3 metrics including margin and cash flow are confirmed unchanged.",
        &["earnings", "q3", "acme", "correction"],
        &scope,
        Some(acme_parent),
        1,
        vec![],
    );
    let corrected_ref = MemoryRef(corrected.id);
    publish(
        Event::MemoryWritten(corrected),
        log.as_ref(),
        vector.as_ref(),
        bm25.as_ref(),
    )
    .await?;
    tokio::time::sleep(Duration::from_millis(600)).await;
    publish(
        Event::MemoryEvolved {
            from: acme_parent,
            to: corrected_ref,
            diff: ChangeSet {
                keywords_added: vec![],
                keywords_removed: vec![],
                tags_added: vec!["correction".into()],
                tags_removed: vec![],
                context_rewritten: false,
            },
        },
        log.as_ref(),
        vector.as_ref(),
        bm25.as_ref(),
    )
    .await?;
    tokio::time::sleep(Duration::from_millis(800)).await;
    publish(
        Event::MemoryInvalidated {
            id: acme_parent,
            reason: "superseded by audited Q3 revenue figure".into(),
        },
        log.as_ref(),
        vector.as_ref(),
        bm25.as_ref(),
    )
    .await?;

    // --- evolution chain 2: guidance refinement ---
    // The "CEO confirmed FY guidance" memory (idx 12) gets refined.
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    let ceo_parent = refs[12];
    let refined = synth_memory(
        "FY guidance refined: management is holding the formal range at 24-25% growth. The 25%+ language used on the analyst call was internal aspiration, not a formal guidance update.",
        &["guidance", "ceo", "growth", "fy26", "clarification"],
        &scope,
        Some(ceo_parent),
        1,
        vec![],
    );
    let refined_ref = MemoryRef(refined.id);
    publish(
        Event::MemoryWritten(refined),
        log.as_ref(),
        vector.as_ref(),
        bm25.as_ref(),
    )
    .await?;
    tokio::time::sleep(Duration::from_millis(600)).await;
    publish(
        Event::MemoryEvolved {
            from: ceo_parent,
            to: refined_ref,
            diff: ChangeSet {
                keywords_added: vec![],
                keywords_removed: vec![],
                tags_added: vec!["clarification".into()],
                tags_removed: vec![],
                context_rewritten: false,
            },
        },
        log.as_ref(),
        vector.as_ref(),
        bm25.as_ref(),
    )
    .await?;
    tokio::time::sleep(Duration::from_millis(800)).await;
    publish(
        Event::MemoryInvalidated {
            id: ceo_parent,
            reason: "superseded by formal FY guidance clarification".into(),
        },
        log.as_ref(),
        vector.as_ref(),
        bm25.as_ref(),
    )
    .await?;

    // --- chunked article ingest ---
    // Demonstrates a longer document broken into many `Memory` chunks that
    // all share a single `Source`. Each chunk gets its own embedding +
    // BM25 entry; the UI groups them under the source title.
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    ingest_article(
        ARTICLE_TITLE,
        ARTICLE_BODY,
        &scope,
        log.as_ref(),
        vector.as_ref(),
        bm25.as_ref(),
    )
    .await?;

    // --- Phase 7: raw observations for the ingestion worker ---
    // These are *un-consolidated* turns. The ingestion worker extracts a
    // fact from each and decides ADD / NOOP(duplicate) / UPDATE
    // (contradiction), which the dashboard's Phase-7 panel surfaces live.
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    stream_observations(log.as_ref(), &scope).await?;

    // --- Phase 8 (P1#6): profile / persona memory ---
    // Set a few stable attributes, including a supersede (vegetarian →
    // vegan) so the dashboard shows the bi-temporal history.
    tokio::time::sleep(Duration::from_millis(900)).await;
    stream_profile(log.as_ref(), profile.as_ref(), &scope).await?;

    tracing::info!(seeds = CORPUS.len(), evolutions = 2, "demo writer finished");
    Ok(())
}

/// Append + apply a handful of `ProfileSet` events for the demo subject.
async fn stream_profile(
    log: &dyn EventLog,
    profile: &ProfileView,
    scope: &Scope,
) -> anyhow::Result<()> {
    let sets: &[(&str, &str)] = &[
        ("locale", "en-GB"),
        ("timezone", "Pacific Time"),
        ("diet", "vegetarian"),
        ("answer_style", "concise"),
        // Supersede: the subject went vegan — old value kept as history.
        ("diet", "vegan"),
    ];
    for (attribute, value) in sets {
        let event = Event::ProfileSet {
            scope: scope.clone(),
            attribute: (*attribute).into(),
            value: (*value).into(),
            actor: Some("alice".into()),
        };
        let id = log.append(event.clone()).await?;
        profile.apply(&LogEntry { id, event }).await?;
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    tracing::info!("demo: set 5 profile attributes (incl. one supersede)");
    Ok(())
}

/// Append three raw observations that exercise the ingestion pipeline:
/// a fresh fact (ADD), a verbatim repeat (NOOP/duplicate), and a
/// correction (UPDATE/contradiction).
async fn stream_observations(log: &dyn EventLog, scope: &Scope) -> anyhow::Result<()> {
    let turns = [
        "Priya was promoted to Staff Engineer in March.",
        "Priya was promoted to Staff Engineer in March.",
        "Actually Priya was promoted to Principal Engineer in March, not Staff.",
    ];
    for turn in turns {
        log.append(Event::ObservationRecorded {
            scope: scope.clone(),
            content: turn.to_string(),
            actor: Some("alice".into()),
        })
        .await?;
        // Space them out so the ingestion worker processes each (and
        // indexes the resulting memory) before the next arrives.
        tokio::time::sleep(Duration::from_millis(900)).await;
    }
    tracing::info!("demo: streamed 3 raw observations for ingestion");
    Ok(())
}

fn synth_memory(
    content: &str,
    tags: &[&str],
    scope: &Scope,
    parent: Option<MemoryRef>,
    evolution_count: u16,
    links: Vec<MemoryRef>,
) -> Memory {
    Memory {
        id: new_id(),
        scope: scope.clone(),
        content: content.into(),
        keywords: vec![],
        tags: tags.iter().map(|t| (*t).into()).collect(),
        context: String::new(),
        // Embedding is `None`: the async worker fills it in. This is what
        // keeps the write path fast even when the embedder is heavy.
        embedding: None,
        links,
        parent,
        evolution_count,
        time: BiTemporal::now(),
        provenance: Provenance {
            source: "demo".into(),
            trust: 1.0,
        },
        source: None,
        position: None,
    }
}

/// Append to the log, then apply to both views. The vector view needs to
/// see `MemoryWritten` even when the embedding is `None` so it can record
/// the memory's scope in its `pending` map — the embedding worker's later
/// `MemoryEmbedded` event drains that map and completes the insert.
async fn publish(
    event: Event,
    log: &dyn EventLog,
    vector: &VectorView,
    bm25: &Bm25View,
) -> anyhow::Result<()> {
    let id = log.append(event.clone()).await?;
    let entry = LogEntry { id, event };
    vector.apply(&entry).await?;
    bm25.apply(&entry).await?;
    Ok(())
}

/// Title of the chunked demo article — chosen so multi-word semantic
/// queries can pick out specific paragraphs.
const ARTICLE_TITLE: &str = "Q3 Board Prep Memo (internal draft)";

/// Six-paragraph article that exercises chunking + source-aware retrieval.
/// Each paragraph deliberately covers a distinct topic so different queries
/// hit different chunks, making the "X of 6 chunks matched" framing visible.
const ARTICLE_BODY: &str = "\
Q3 was a beat-and-raise quarter for the company. Revenue grew 18% YoY to $284M, ahead of our $268M internal target and the $271M consensus. Operating cash flow of $142M was substantially better than the $115M consensus, on stronger collections and tighter working capital management. Gross margin held at 76.2% despite component cost pressure.

The v3.4 platform release shipped on schedule with three priority features: SSO via SAML 2.0, role-based access control, and an audit log export. Enterprise customer feedback has been notably positive on the security improvements. The mid-market product line remains on track for late-Q4 GA, with pilot customers reporting 30% faster onboarding versus the legacy stack.

Component supply has stabilized; lead times are back to the normal 6 to 8 week range after the H1 shortage. The Vietnam backup supplier came online ahead of schedule and is now handling 22% of board-level component volume. Manufacturing yield improved 240 bps QoQ following the automated optical inspection rollout in Penang, taking the defect escape rate to a four-quarter low.

We hired a new Chief Revenue Officer from a Series D peer; she starts end of Q4. Her mandate is rebuilding enterprise go-to-market and shortening the average sales cycle, currently 142 days. Win rate in head-to-head SMB deals improved markedly per the latest IDC tracker — our primary competitor lost 3 points of segment share over the period.

Engineering org grew 18% YoY to 412 engineers; the Bangalore team specifically expanded to 78 from 54 in six months. Voluntary attrition remains below 9%, the lowest in three years. The first VP-level promotion from the Bangalore office was announced internally last week, which is a positive signal for the leadership bench.

Three risks for board attention. First, margin compression in EMEA remains a guidance risk if FX stays adverse — our hedging program has been extended an additional two quarters as insurance. Second, cloud infrastructure costs are growing faster than topline; a working group has been chartered to evaluate reserved-capacity commitments versus the current spot-heavy posture. Third, a regulatory inquiry in Germany regarding data residency is being managed through external counsel; we've provisioned $2M for potential resolution.";

async fn ingest_article(
    title: &str,
    body: &str,
    scope: &Scope,
    log: &dyn EventLog,
    vector: &VectorView,
    bm25: &Bm25View,
) -> anyhow::Result<()> {
    let chunker = ParagraphChunker::new();
    let chunks = chunker.chunk(body);
    let chunk_count = u32::try_from(chunks.len()).unwrap_or(u32::MAX);

    // Allocate the source first so chunks can carry its ref.
    let source = Source {
        id: new_id(),
        scope: scope.clone(),
        title: title.into(),
        uri: None,
        chunk_count,
        time: BiTemporal::now(),
        provenance: Provenance {
            source: "demo-article".into(),
            trust: 1.0,
        },
    };
    let source_ref = SourceRef(source.id);
    publish(Event::SourceIngested(source), log, vector, bm25).await?;

    // Quick beat so the SourceIngested event lands separately in the
    // live-view's event log before the chunks start arriving.
    tokio::time::sleep(Duration::from_millis(400)).await;

    for (idx, chunk_text) in chunks.into_iter().enumerate() {
        let m = Memory {
            id: new_id(),
            scope: scope.clone(),
            content: chunk_text,
            keywords: vec![],
            tags: vec!["article".into(), "board-memo".into()],
            context: String::new(),
            embedding: None,
            links: vec![],
            parent: None,
            evolution_count: 0,
            time: BiTemporal::now(),
            provenance: Provenance {
                source: "demo-article".into(),
                trust: 1.0,
            },
            source: Some(source_ref),
            position: Some(idx as u32),
        };
        publish(Event::MemoryWritten(m), log, vector, bm25).await?;
        tokio::time::sleep(Duration::from_millis(600)).await;
    }
    tracing::info!(
        source = %source_ref.0,
        chunks = chunk_count,
        "demo: article ingested"
    );
    Ok(())
}
