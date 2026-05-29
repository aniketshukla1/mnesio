//! `mneme-bench` — eval harness CLI.
//!
//! Two subcommands:
//!
//! - `run` — iterate the procedural compiler against a suite, emit a
//!   learning curve.
//! - `compare` — evaluate two artifacts (baseline body vs candidate
//!   body) against the same suite, emit an A/B diff.
//!
//! Output formats (`--output`):
//!
//! - `csv` — raw rows; pipes cleanly into `>` (default for `run`).
//! - `json` — machine-readable summary (use for CI).
//! - `html` — self-contained HTML report with inline SVG chart.
//! - `markdown` — table summary (paste-into-PR-friendly).
//!
//! Regression CI (`--regression-threshold`):
//!
//! - `run` mode exits 1 if the final benchmark falls more than the
//!   threshold below the v1 baseline.
//! - `compare` mode exits 1 if the candidate scores more than the
//!   threshold below the baseline.
//! - Always exits 1 on any safety probe regression — no threshold,
//!   alignment drift is the hard stop.

use anyhow::{bail, Result};
use mneme_bench::report::{render_comparison, render_learning_curve};
use mneme_bench::{
    compare_artifacts, load_suite, run_bench, BenchRun, BenchSuite, ComparisonReport, DemoBenchLlm,
    DemoSuiteExecutor, PolicyExecutor,
};
use mneme_core::entity::{ArtifactKind, PolicyArtifact};
use mneme_core::types::{new_id, BiTemporal, Scope};
use mneme_core::LlmClient;
use std::sync::Arc;

const GSM8K_JSON: &str = include_str!("../data/gsm8k_tiny.json");
const HUMANEVAL_JSON: &str = include_str!("../data/humaneval_tiny.json");

const SEED_PROMPT: &str = "You are a helpful assistant. Answer the question.";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let args = parse_args()?;
    match args.command {
        Command::Run(opts) => cmd_run(opts).await,
        Command::Compare(opts) => cmd_compare(opts).await,
    }
}

// ---------------- run ----------------

async fn cmd_run(opts: RunOpts) -> Result<()> {
    let suite = load_suite_by_name(&opts.suite)?;
    let (llm, executor) = build_executor(&opts.executor, &suite)?;
    eprintln!(
        "# mneme-bench run · suite={} · max_versions={} · executor={}",
        suite.name, opts.max_versions, opts.executor
    );
    let result = run_bench(&suite, SEED_PROMPT, opts.max_versions, llm, executor).await?;

    // Emit the requested output. stderr always carries the summary
    // so users get a quick read regardless of format.
    let out_text = match opts.output {
        OutputFormat::Csv => format_run_csv(&result),
        OutputFormat::Json => format_run_json(&result)?,
        OutputFormat::Html => format_run_html(&result),
        OutputFormat::Markdown => format_run_markdown(&result),
    };
    write_output(&opts.out_path, &out_text)?;
    write_run_summary_to_stderr(&result);

    // Regression gate.
    if let Some(threshold) = opts.regression_threshold {
        check_run_regression(&result, threshold)?;
    }
    check_safety_regression_curve(&result)?;
    Ok(())
}

fn format_run_csv(result: &BenchRun) -> String {
    let mut out = String::from(
        "version,benchmark_score,safety_pass_rate,objective_delta,judges_consulted,timestamp_ms\n",
    );
    for p in &result.curve {
        out.push_str(&format!(
            "{},{:.4},{:.4},{:.4},{},{}\n",
            p.version,
            p.benchmark_score,
            p.safety_probe_pass_rate,
            p.objective_delta,
            p.judges_consulted,
            p.timestamp_ms
        ));
    }
    out
}

fn format_run_json(result: &BenchRun) -> Result<String> {
    use serde_json::json;
    let curve: Vec<serde_json::Value> = result
        .curve
        .iter()
        .map(|p| {
            json!({
                "version": p.version,
                "benchmark_score": p.benchmark_score,
                "safety_pass_rate": p.safety_probe_pass_rate,
                "objective_delta": p.objective_delta,
                "judges_consulted": p.judges_consulted,
                "timestamp_ms": p.timestamp_ms,
            })
        })
        .collect();
    let first = result
        .curve
        .first()
        .map(|p| p.benchmark_score)
        .unwrap_or(0.0);
    let last = result
        .curve
        .last()
        .map(|p| p.benchmark_score)
        .unwrap_or(0.0);
    let safety_min = result
        .curve
        .iter()
        .map(|p| p.safety_probe_pass_rate)
        .fold(1.0_f32, |a, b| a.min(b));
    let payload = json!({
        "suite_name": result.suite_name,
        "committed": result.committed,
        "rejected": result.rejected,
        "benchmark_v1": first,
        "benchmark_final": last,
        "benchmark_delta": last - first,
        "safety_min": safety_min,
        "safety_regressed": safety_min < 1.0 - 1e-6,
        "curve": curve,
    });
    Ok(serde_json::to_string_pretty(&payload)?)
}

fn format_run_html(result: &BenchRun) -> String {
    let final_body = match &result.final_active_artifact.kind {
        ArtifactKind::SystemPrompt { body } => body.clone(),
        _ => "(non-SystemPrompt artifact)".into(),
    };
    render_learning_curve(
        &result.suite_name,
        &result.seed_body,
        &final_body,
        &result.curve,
        result.committed,
        result.rejected,
    )
}

fn format_run_markdown(result: &BenchRun) -> String {
    let mut out = format!("# mneme-bench · {} · learning curve\n\n", result.suite_name);
    out.push_str("| version | benchmark | safety | Δ |\n");
    out.push_str("|---|---|---|---|\n");
    for p in &result.curve {
        out.push_str(&format!(
            "| v{} | {:.1}% | {:.1}% | {:+.3} |\n",
            p.version,
            p.benchmark_score * 100.0,
            p.safety_probe_pass_rate * 100.0,
            p.objective_delta
        ));
    }
    out.push_str(&format!(
        "\n_{} commits · {} rejections_\n",
        result.committed, result.rejected
    ));
    out
}

fn write_run_summary_to_stderr(result: &BenchRun) {
    let first = result
        .curve
        .first()
        .map(|p| p.benchmark_score)
        .unwrap_or(0.0);
    let last = result
        .curve
        .last()
        .map(|p| p.benchmark_score)
        .unwrap_or(0.0);
    let last_v = result.curve.last().map(|p| p.version).unwrap_or(1);
    let safety_min = result
        .curve
        .iter()
        .map(|p| p.safety_probe_pass_rate)
        .fold(1.0_f32, |a, b| a.min(b));
    eprintln!();
    eprintln!("# summary:");
    eprintln!("#   versions evaluated:  {}", result.curve.len());
    eprintln!("#   committed proposals: {}", result.committed);
    eprintln!("#   rejected proposals:  {}", result.rejected);
    eprintln!(
        "#   benchmark v1 -> v{last_v}: {:.1}% -> {:.1}% (Δ {:+.1}pp)",
        first * 100.0,
        last * 100.0,
        (last - first) * 100.0
    );
    if safety_min >= 1.0 - 1e-6 {
        eprintln!("#   safety probe pass rate: 100% throughout");
    } else {
        eprintln!(
            "#   safety probe pass rate: REGRESSED — min {:.1}%",
            safety_min * 100.0
        );
    }
}

fn check_run_regression(result: &BenchRun, threshold: f32) -> Result<()> {
    let first = result
        .curve
        .first()
        .map(|p| p.benchmark_score)
        .unwrap_or(0.0);
    let last = result
        .curve
        .last()
        .map(|p| p.benchmark_score)
        .unwrap_or(0.0);
    let delta = last - first;
    if delta < -threshold {
        eprintln!(
            "# REGRESSION: benchmark fell {:.1}pp from v1 (threshold: {:.1}pp). Exit 1.",
            delta * 100.0,
            threshold * 100.0
        );
        std::process::exit(1);
    }
    Ok(())
}

fn check_safety_regression_curve(result: &BenchRun) -> Result<()> {
    let safety_min = result
        .curve
        .iter()
        .map(|p| p.safety_probe_pass_rate)
        .fold(1.0_f32, |a, b| a.min(b));
    if safety_min < 1.0 - 1e-6 {
        eprintln!(
            "# SAFETY REGRESSION: alignment-drift hard stop. min={:.1}%. Exit 1.",
            safety_min * 100.0
        );
        std::process::exit(1);
    }
    Ok(())
}

// ---------------- compare ----------------

async fn cmd_compare(opts: CompareOpts) -> Result<()> {
    let suite = load_suite_by_name(&opts.suite)?;
    let (_llm, executor) = build_executor(&opts.executor, &suite)?;
    eprintln!(
        "# mneme-bench compare · suite={} · executor={}",
        suite.name, opts.executor
    );

    let a = make_artifact(&opts.baseline);
    let b = make_artifact(&opts.candidate);
    let report = compare_artifacts(
        &a,
        &b,
        &suite,
        executor,
        opts.label_a.clone(),
        opts.label_b.clone(),
    )
    .await?;

    let out_text = match opts.output {
        OutputFormat::Csv => format_compare_csv(&report),
        OutputFormat::Json => format_compare_json(&report)?,
        OutputFormat::Html => render_comparison(&report),
        OutputFormat::Markdown => format_compare_markdown(&report),
    };
    write_output(&opts.out_path, &out_text)?;
    write_compare_summary_to_stderr(&report);

    // Regression gate: candidate scoring below baseline by more than threshold.
    if let Some(threshold) = opts.regression_threshold {
        if -report.benchmark_delta > threshold {
            eprintln!(
                "# REGRESSION: candidate fell {:.1}pp below baseline (threshold: {:.1}pp). Exit 1.",
                -report.benchmark_delta * 100.0,
                threshold * 100.0
            );
            std::process::exit(1);
        }
    }
    if report.safety_regressed() {
        eprintln!(
            "# SAFETY REGRESSION: candidate safety dropped {:.1}pp. Exit 1.",
            -report.safety_delta * 100.0
        );
        std::process::exit(1);
    }
    Ok(())
}

fn make_artifact(body: &str) -> PolicyArtifact {
    PolicyArtifact {
        id: new_id(),
        version: 1,
        scope: Scope::global("bench"),
        kind: ArtifactKind::SystemPrompt { body: body.into() },
        canaries: vec![],
        time: BiTemporal::now(),
    }
}

fn format_compare_csv(report: &ComparisonReport) -> String {
    let mut out = String::from("category,a_passed,b_passed,total,delta\n");
    for c in &report.per_category {
        out.push_str(&format!(
            "{},{},{},{},{}\n",
            c.category,
            c.a_passed,
            c.b_passed,
            c.total,
            c.b_passed as i32 - c.a_passed as i32
        ));
    }
    out
}

fn format_compare_json(report: &ComparisonReport) -> Result<String> {
    use serde_json::json;
    let categories: Vec<_> = report
        .per_category
        .iter()
        .map(|c| {
            json!({
                "category": c.category,
                "a_passed": c.a_passed,
                "b_passed": c.b_passed,
                "total": c.total,
            })
        })
        .collect();
    let payload = json!({
        "suite_name": report.suite_name,
        "a_label": report.artifact_a_label,
        "b_label": report.artifact_b_label,
        "a_benchmark": report.report_a.benchmark_score,
        "b_benchmark": report.report_b.benchmark_score,
        "benchmark_delta": report.benchmark_delta,
        "a_safety": report.report_a.safety_probe_pass_rate,
        "b_safety": report.report_b.safety_probe_pass_rate,
        "safety_delta": report.safety_delta,
        "safety_regressed": report.safety_regressed(),
        "per_category": categories,
    });
    Ok(serde_json::to_string_pretty(&payload)?)
}

fn format_compare_markdown(report: &ComparisonReport) -> String {
    let mut out = format!(
        "# mneme-bench · {} · {} vs {}\n\n",
        report.suite_name, report.artifact_a_label, report.artifact_b_label
    );
    out.push_str(&format!(
        "| | benchmark | safety |\n|---|---|---|\n| {} | {:.1}% | {:.1}% |\n| {} | {:.1}% | {:.1}% |\n| **Δ** | **{:+.1}pp** | **{:+.1}pp** |\n\n",
        report.artifact_a_label,
        report.report_a.benchmark_score * 100.0,
        report.report_a.safety_probe_pass_rate * 100.0,
        report.artifact_b_label,
        report.report_b.benchmark_score * 100.0,
        report.report_b.safety_probe_pass_rate * 100.0,
        report.benchmark_delta * 100.0,
        report.safety_delta * 100.0,
    ));
    out.push_str("| category | A | B | total |\n|---|---|---|---|\n");
    for c in &report.per_category {
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            c.category, c.a_passed, c.b_passed, c.total
        ));
    }
    out
}

fn write_compare_summary_to_stderr(report: &ComparisonReport) {
    eprintln!();
    eprintln!("# summary:");
    eprintln!(
        "#   {}: {:.1}% benchmark, {:.1}% safety",
        report.artifact_a_label,
        report.report_a.benchmark_score * 100.0,
        report.report_a.safety_probe_pass_rate * 100.0,
    );
    eprintln!(
        "#   {}: {:.1}% benchmark, {:.1}% safety",
        report.artifact_b_label,
        report.report_b.benchmark_score * 100.0,
        report.report_b.safety_probe_pass_rate * 100.0,
    );
    eprintln!(
        "#   Δ benchmark: {:+.1}pp, Δ safety: {:+.1}pp",
        report.benchmark_delta * 100.0,
        report.safety_delta * 100.0
    );
}

// ---------------- helpers ----------------

fn load_suite_by_name(name: &str) -> Result<BenchSuite> {
    let json = match name {
        "gsm8k" => GSM8K_JSON,
        "humaneval" => HUMANEVAL_JSON,
        other => bail!("unknown --suite {other:?}; expected `gsm8k` or `humaneval`"),
    };
    load_suite(json)
}

#[allow(unused_variables)]
fn build_executor(
    choice: &str,
    suite: &BenchSuite,
) -> Result<(Arc<dyn LlmClient>, Arc<dyn PolicyExecutor>)> {
    match choice {
        "demo" => Ok((
            Arc::new(DemoBenchLlm),
            Arc::new(DemoSuiteExecutor::from_suite(suite)) as Arc<dyn PolicyExecutor>,
        )),
        #[cfg(feature = "ollama")]
        "ollama" => {
            use mneme_llm::OllamaLlmClient;
            use mneme_procedural::LlmExecutor;
            let ollama = Arc::new(OllamaLlmClient::from_env()?);
            let llm: Arc<dyn LlmClient> = ollama.clone();
            let exec: Arc<dyn PolicyExecutor> = Arc::new(LlmExecutor::new(ollama));
            Ok((llm, exec))
        }
        #[cfg(not(feature = "ollama"))]
        "ollama" => bail!(
            "--executor ollama requires the `ollama` feature; rebuild with \
             `cargo run -p mneme-bench --features ollama ...`"
        ),
        other => bail!("unknown --executor {other:?}; expected `demo` or `ollama`"),
    }
}

fn write_output(out_path: &Option<std::path::PathBuf>, content: &str) -> Result<()> {
    match out_path {
        Some(path) => {
            std::fs::write(path, content)?;
            eprintln!("# wrote {} bytes to {}", content.len(), path.display());
        }
        None => print!("{content}"),
    }
    Ok(())
}

// ---------------- arg parsing ----------------

enum Command {
    Run(RunOpts),
    Compare(CompareOpts),
}

struct RunOpts {
    suite: String,
    max_versions: u32,
    executor: String,
    output: OutputFormat,
    out_path: Option<std::path::PathBuf>,
    regression_threshold: Option<f32>,
}

struct CompareOpts {
    suite: String,
    executor: String,
    baseline: String,
    candidate: String,
    label_a: String,
    label_b: String,
    output: OutputFormat,
    out_path: Option<std::path::PathBuf>,
    regression_threshold: Option<f32>,
}

enum OutputFormat {
    Csv,
    Json,
    Html,
    Markdown,
}

impl OutputFormat {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "csv" => OutputFormat::Csv,
            "json" => OutputFormat::Json,
            "html" => OutputFormat::Html,
            "md" | "markdown" => OutputFormat::Markdown,
            other => bail!("unknown --output {other:?}; expected csv|json|html|markdown"),
        })
    }
}

fn parse_args() -> Result<RootArgs> {
    let mut iter = std::env::args().skip(1).peekable();
    // First positional = subcommand (default: run).
    let subcommand = match iter.peek().map(|s| s.as_str()) {
        Some("run") => {
            iter.next();
            "run"
        }
        Some("compare") => {
            iter.next();
            "compare"
        }
        Some("--help") | Some("-h") => {
            print_help();
            std::process::exit(0);
        }
        Some(s) if s.starts_with('-') => "run", // flags-only → default to run
        Some(_) => "run",
        None => "run",
    };

    match subcommand {
        "run" => Ok(RootArgs {
            command: Command::Run(parse_run(iter)?),
        }),
        "compare" => Ok(RootArgs {
            command: Command::Compare(parse_compare(iter)?),
        }),
        _ => unreachable!(),
    }
}

struct RootArgs {
    command: Command,
}

fn parse_run(mut iter: std::iter::Peekable<impl Iterator<Item = String>>) -> Result<RunOpts> {
    let mut opts = RunOpts {
        suite: "gsm8k".into(),
        max_versions: 6,
        executor: "demo".into(),
        output: OutputFormat::Csv,
        out_path: None,
        regression_threshold: None,
    };
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--suite" => opts.suite = next_value(&mut iter, "--suite")?,
            "--max-versions" => {
                opts.max_versions = next_value(&mut iter, "--max-versions")?.parse()?
            }
            "--executor" => opts.executor = next_value(&mut iter, "--executor")?,
            "--output" => opts.output = OutputFormat::parse(&next_value(&mut iter, "--output")?)?,
            "--out" => opts.out_path = Some(next_value(&mut iter, "--out")?.into()),
            "--regression-threshold" => {
                opts.regression_threshold =
                    Some(next_value(&mut iter, "--regression-threshold")?.parse()?);
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}; pass --help for usage"),
        }
    }
    Ok(opts)
}

fn parse_compare(
    mut iter: std::iter::Peekable<impl Iterator<Item = String>>,
) -> Result<CompareOpts> {
    let mut opts = CompareOpts {
        suite: "gsm8k".into(),
        executor: "demo".into(),
        baseline: SEED_PROMPT.into(),
        candidate: String::new(),
        label_a: "baseline".into(),
        label_b: "candidate".into(),
        output: OutputFormat::Markdown,
        out_path: None,
        regression_threshold: None,
    };
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--suite" => opts.suite = next_value(&mut iter, "--suite")?,
            "--executor" => opts.executor = next_value(&mut iter, "--executor")?,
            "--baseline" => opts.baseline = next_value(&mut iter, "--baseline")?,
            "--candidate" => opts.candidate = next_value(&mut iter, "--candidate")?,
            "--label-a" => opts.label_a = next_value(&mut iter, "--label-a")?,
            "--label-b" => opts.label_b = next_value(&mut iter, "--label-b")?,
            "--output" => opts.output = OutputFormat::parse(&next_value(&mut iter, "--output")?)?,
            "--out" => opts.out_path = Some(next_value(&mut iter, "--out")?.into()),
            "--regression-threshold" => {
                opts.regression_threshold =
                    Some(next_value(&mut iter, "--regression-threshold")?.parse()?);
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument {other:?}; pass --help for usage"),
        }
    }
    if opts.candidate.is_empty() {
        bail!("compare requires --candidate <prompt body>");
    }
    Ok(opts)
}

fn next_value(
    iter: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String> {
    iter.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

#[allow(unused_must_use)]
fn print_help() {
    eprintln!(
        "mneme-bench — eval harness for the procedural compiler\n\
         \n\
         USAGE:\n  cargo run -p mneme-bench -- [SUBCOMMAND] [OPTIONS]\n\
         \n\
         SUBCOMMANDS:\n\
         \x20\x20run        iterate the compiler against a suite, emit a learning curve\n\
         \x20\x20             (default — invoked when no subcommand is given)\n\
         \x20\x20compare    A vs B evaluation of two artifact bodies against a fixed suite\n\
         \n\
         SHARED OPTIONS:\n\
         \x20\x20--suite          gsm8k | humaneval                (default: gsm8k)\n\
         \x20\x20--executor       demo | ollama                    (default: demo)\n\
         \x20\x20--output         csv | json | html | markdown     (default: csv for run, md for compare)\n\
         \x20\x20--out PATH       file to write the output to       (default: stdout)\n\
         \x20\x20--regression-threshold N\n\
         \x20\x20                  exit 1 if benchmark falls more than N below baseline\n\
         \n\
         RUN OPTIONS:\n\
         \x20\x20--max-versions   N                                (default: 6)\n\
         \n\
         COMPARE OPTIONS:\n\
         \x20\x20--baseline TEXT   prompt body to score as A         (default: a generic seed)\n\
         \x20\x20--candidate TEXT  prompt body to score as B         (required)\n\
         \x20\x20--label-a TEXT    display label for A               (default: baseline)\n\
         \x20\x20--label-b TEXT    display label for B               (default: candidate)\n\
         \n\
         EXIT CODES:\n\
         \x20\x200   ok\n\
         \x20\x201   benchmark regressed past threshold, or safety probe regressed\n\
         \x20\x201   bad arguments / IO errors\n"
    );
}
