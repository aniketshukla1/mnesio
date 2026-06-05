//! End-to-end Slice B test: drive the full pipeline (Reflector →
//! ShadowEvaluator → Gate) and assert on the [`CompileResult`].
//!
//! These tests live as an integration test (rather than `#[cfg(test)]
//! mod`) because the pipeline is the *public* contract — they exercise
//! the same surface a real host would use, with no internal-detail
//! peeking.

use mneme_core::entity::{ArtifactKind, Canary, JudgeSource, Outcome, PolicyArtifact};
use mneme_core::types::{new_id, ArtifactRef, BiTemporal, EpisodeRef, Scope, TrajectoryRef};
use mneme_llm::FakeLlmClient;
use mneme_procedural::{
    EvalGates, FakeJudge, FakePolicyExecutor, JudgeVerdict, ProceduralCompiler, RejectReason,
    ShadowInputs,
};
use std::collections::HashMap;
use std::sync::Arc;

// ---------- fixtures ----------

fn artifact(version: u32, body: &str, canaries: Vec<Canary>) -> PolicyArtifact {
    PolicyArtifact {
        id: new_id(),
        version,
        scope: Scope::global("t"),
        kind: ArtifactKind::SystemPrompt { body: body.into() },
        canaries,
        time: BiTemporal::now(),
    }
}

fn canary(input: &str, expect: &str) -> Canary {
    Canary {
        input: input.into(),
        expect: expect.into(),
    }
}

fn outcome(success: bool) -> Outcome {
    let mut scores = HashMap::new();
    scores.insert("accuracy".into(), if success { 1.0 } else { 0.0 });
    Outcome {
        id: new_id(),
        episode: EpisodeRef(new_id()),
        artifacts_used: vec![ArtifactRef(new_id())],
        success: Some(success),
        scores,
        error: if success { None } else { Some("wrong".into()) },
        judge: JudgeSource::Environment,
        trajectory: TrajectoryRef(new_id()),
    }
}

/// FakeLlmClient that returns canned reflection + 2 candidate proposals.
fn happy_llm() -> Arc<FakeLlmClient> {
    Arc::new(
        FakeLlmClient::new()
            .with_prefix_match(
                "You are reviewing a recent batch",
                "FINDING: the prompt is too terse for compound questions",
            )
            .with_prefix_match(
                "You are revising a system prompt",
                "--- CANDIDATE 1 ---\n\
                 You are a careful, helpful assistant. Answer in 1-3 sentences.\n\
                 --- CANDIDATE 2 ---\n\
                 You are a careful, helpful assistant. For compound questions, address each part.",
            ),
    )
}

// ---------- happy path ----------

#[tokio::test]
async fn pipeline_produces_committable_proposal_when_everything_clears() {
    let active = artifact(
        1,
        "Answer briefly.",
        vec![canary("sky?", "blue"), canary("grass?", "green")],
    );

    // The executor returns the expected canary answers for the bumped
    // candidate version (2). Default for everything else: "answer".
    let executor = Arc::new(
        FakePolicyExecutor::new()
            .with_default("answer")
            .with_response(2, "sky?", "blue")
            .with_response(2, "grass?", "green"),
    );

    // Two distinct judges, both default-pass.
    let judges = vec![
        Arc::new(FakeJudge::new("judge-a")) as Arc<dyn mneme_procedural::Judge>,
        Arc::new(FakeJudge::new("judge-b")),
    ];

    let compiler = ProceduralCompiler::new(happy_llm(), executor, judges, 2);

    let inputs = ShadowInputs {
        baseline: active.clone(),
        replay: vec![],
        safety_probes: vec![],
    };

    let result = compiler
        .compile_with_inputs(&active, &[outcome(false), outcome(false)], &inputs, "test")
        .await
        .unwrap()
        .expect("pipeline must produce a CompileResult");

    assert_eq!(result.outcomes.len(), 2, "K=2 candidates evaluated");
    assert!(
        result.has_winner(),
        "at least one candidate must be committable"
    );
    let winner = result.winner().unwrap();
    assert!(winner.report.is_committable());
    assert_eq!(winner.report.judges_consulted, 2);
    assert_eq!(winner.report.canaries_passed, 2);
    assert_eq!(winner.report.canaries_total, 2);
}

// ---------- gate rejections (whole-pipeline level) ----------

#[tokio::test]
async fn pipeline_rejects_when_judge_panel_too_small_for_diversity_gate() {
    // Default gates require min_judges = 2. Provide only one.
    let active = artifact(1, "body", vec![]);
    let compiler = ProceduralCompiler::new(
        happy_llm(),
        Arc::new(FakePolicyExecutor::new().with_default("ok")),
        vec![Arc::new(FakeJudge::new("solo"))],
        2,
    );

    let inputs = ShadowInputs {
        baseline: active.clone(),
        replay: vec![],
        safety_probes: vec![],
    };

    let result = compiler
        .compile_with_inputs(&active, &[outcome(false)], &inputs, "test")
        .await
        .unwrap()
        .expect("pipeline still produces a result — gate rejects ≠ pipeline fails");

    assert!(!result.has_winner());
    for o in &result.outcomes {
        assert!(o.verdict.rejected());
        assert!(o
            .verdict
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::InsufficientJudges { .. })));
    }
}

#[tokio::test]
async fn pipeline_rejects_when_canary_fails_under_candidate() {
    // Executor returns the WRONG answer for the canary under v=2,
    // even though judges would pass on anything. The Judge sees the
    // expected vs actual mismatch and rejects.
    let active = artifact(1, "body", vec![canary("sky?", "blue")]);

    let executor = Arc::new(
        FakePolicyExecutor::new()
            .with_default("ok")
            .with_response(2, "sky?", "purple"), // not blue!
    );

    // Two judges; the canary-strict path requires unanimous PASS. We
    // configure judge-b to fail when actual="purple".
    let judges: Vec<Arc<dyn mneme_procedural::Judge>> = vec![
        Arc::new(FakeJudge::new("judge-a")),
        Arc::new(FakeJudge::new("judge-b").with_verdict(
            "sky?",
            "purple",
            JudgeVerdict::fail("not blue"),
        )),
    ];

    let compiler = ProceduralCompiler::new(happy_llm(), executor, judges, 2);
    let inputs = ShadowInputs {
        baseline: active.clone(),
        replay: vec![],
        safety_probes: vec![],
    };

    let result = compiler
        .compile_with_inputs(&active, &[outcome(false)], &inputs, "test")
        .await
        .unwrap()
        .unwrap();

    assert!(!result.has_winner());
    for o in &result.outcomes {
        // Both BaselineFailed (canary tripped baseline) and
        // CanariesFailing (structured) should surface.
        assert!(o.verdict.reasons.contains(&RejectReason::BaselineFailed));
        assert!(o
            .verdict
            .reasons
            .iter()
            .any(|r| matches!(r, RejectReason::CanariesFailing { .. })));
    }
}

// ---------- soft-failure: nothing to propose ----------

#[tokio::test]
async fn pipeline_returns_none_when_reflector_finds_nothing() {
    let active = artifact(1, "body", vec![]);
    let llm =
        Arc::new(FakeLlmClient::new().with_prefix_match("You are reviewing", "FINDING: none"));
    let compiler = ProceduralCompiler::new(
        llm,
        Arc::new(FakePolicyExecutor::new()),
        vec![Arc::new(FakeJudge::new("a")), Arc::new(FakeJudge::new("b"))],
        2,
    );
    let inputs = ShadowInputs {
        baseline: active.clone(),
        replay: vec![],
        safety_probes: vec![],
    };
    let result = compiler
        .compile_with_inputs(&active, &[outcome(true)], &inputs, "test")
        .await
        .unwrap();
    assert!(result.is_none(), "no findings → no compile output");
}

// ---------- doctrine: BaselineFailed cannot be relaxed away ----------

#[tokio::test]
async fn fully_relaxed_gates_still_reject_baseline_failure_through_pipeline() {
    // Active artifact has a canary the candidate executor will FAIL.
    // Operator misconfigures gates to be maximally permissive. Pipeline
    // must STILL reject — BaselineFailed is the non-bypassable backstop.
    let active = artifact(1, "body", vec![canary("sky?", "blue")]);
    let executor = Arc::new(
        FakePolicyExecutor::new()
            .with_default("ok")
            .with_response(2, "sky?", "purple"),
    );
    let judges: Vec<Arc<dyn mneme_procedural::Judge>> = vec![Arc::new(
        FakeJudge::new("a").with_verdict("sky?", "purple", JudgeVerdict::fail("wrong")),
    )];
    let compiler =
        ProceduralCompiler::new(happy_llm(), executor, judges, 2).with_gates(EvalGates {
            min_canary_pass_rate: 0.0,
            require_safety_probe: false,
            min_objective_delta: f32::NEG_INFINITY,
            min_replay_success_rate: 0.0,
            min_judges: 0,
            require_canaries: false,
        });
    let inputs = ShadowInputs {
        baseline: active.clone(),
        replay: vec![],
        safety_probes: vec![],
    };
    let result = compiler
        .compile_with_inputs(&active, &[outcome(false)], &inputs, "test")
        .await
        .unwrap()
        .unwrap();
    assert!(
        !result.has_winner(),
        "baseline must reject even fully-relaxed"
    );
    for o in &result.outcomes {
        assert!(o.verdict.reasons.contains(&RejectReason::BaselineFailed));
    }
}
