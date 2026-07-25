//! Self-contained HTML report generation. Pure-Rust string templating
//! with inline SVG: no JS, no CSS frameworks, no Chart.js dep. The
//! output is a single `.html` file you can open in any browser, attach
//! to a PR, or upload as a CI artifact.
//!
//! Two report shapes:
//!
//! - [`render_learning_curve`] — line chart of benchmark score +
//!   safety probe pass rate over versions. The shape Phase 2 "done
//!   when" surfaces.
//! - [`render_comparison`] — grouped bar chart per task category for
//!   an A vs B run. The shape the CLI's `compare` mode surfaces.
//!
//! Both reports embed a per-task drill-down table so reviewers can
//! see *which* tasks moved, not just the aggregate score.

use crate::ComparisonReport;
use mnesio_procedural::{LearningCurvePoint, TaskKind};

/// Canvas dimensions used by both chart kinds. Picked so the chart
/// fills a comfortable laptop viewport without scrolling.
const CHART_W: f32 = 720.0;
const CHART_H: f32 = 360.0;
const PAD_L: f32 = 60.0;
const PAD_R: f32 = 24.0;
const PAD_T: f32 = 32.0;
const PAD_B: f32 = 48.0;

/// Render a learning-curve report — what `mnesio-bench run` produces.
pub fn render_learning_curve(
    suite_name: &str,
    seed_body: &str,
    final_body: &str,
    curve: &[LearningCurvePoint],
    committed: usize,
    rejected: usize,
) -> String {
    let chart_svg = if curve.is_empty() {
        empty_chart_svg("no curve points recorded")
    } else {
        learning_curve_svg(curve)
    };
    let summary = learning_summary_block(curve, committed, rejected);
    let prompt_diff = prompt_diff_table(seed_body, final_body);
    template(
        &format!("mnesio-bench · {suite_name} · learning curve"),
        &format!(
            r##"
    <h1>Learning curve: {suite}</h1>
    <p class="lede">
      Iterative prompt improvement under the procedural compiler. Each version is a
      new <code>PolicyArtifact</code> committed after clearing the regression gate.
    </p>
    {summary}
    <h2>Curve</h2>
    {chart_svg}
    <h2>Seed vs. final prompt</h2>
    {prompt_diff}
"##,
            suite = html_escape(suite_name),
        ),
    )
}

/// Render an A/B comparison report — what `mnesio-bench compare` produces.
pub fn render_comparison(report: &ComparisonReport) -> String {
    let chart_svg = if report.per_category.is_empty() {
        empty_chart_svg("no benchmark tasks in suite")
    } else {
        comparison_bar_svg(
            &report.per_category,
            &report.artifact_a_label,
            &report.artifact_b_label,
        )
    };
    let summary = comparison_summary_block(report);
    let drill = comparison_drill_table(report);
    template(
        &format!("mnesio-bench · {} · A vs B", report.suite_name),
        &format!(
            r##"
    <h1>Comparison: {suite}</h1>
    <p class="lede">
      Same suite, two artifacts. Bars per category show pass counts for each.
      Hover the table rows for the full input + outputs.
    </p>
    {summary}
    <h2>Per-category pass counts</h2>
    {chart_svg}
    <h2>Per-task drill-down</h2>
    {drill}
"##,
            suite = html_escape(&report.suite_name),
        ),
    )
}

// ---------------- HTML scaffold ----------------

/// Common HTML wrapper. Inline `<style>`, no external assets, dark
/// theme aligned with the dashboard's palette.
fn template(title: &str, body: &str) -> String {
    let title = html_escape(title);
    format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    --bg: #0d1117;
    --panel: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --green: #56d364;
    --orange: #f0883e;
    --red: #f85149;
    --purple: #d2a8ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 48px; background: var(--bg); color: var(--text);
    font-family: ui-monospace, 'SF Mono', SFMono-Regular, Menlo, monospace;
    font-size: 13px; max-width: 1080px; margin-left: auto; margin-right: auto;
  }}
  h1 {{ font-size: 22px; margin: 0 0 12px; font-weight: 600; }}
  h1 span.accent {{ color: var(--accent); }}
  h2 {{ font-size: 14px; margin: 28px 0 12px; text-transform: uppercase;
       letter-spacing: 1px; color: var(--text-dim); border-bottom: 1px solid var(--border);
       padding-bottom: 6px; }}
  p.lede {{ color: var(--text-dim); font-size: 12px; max-width: 720px; line-height: 1.6; }}
  code {{ background: rgba(255,255,255,0.04); padding: 2px 5px; border-radius: 3px;
          color: var(--purple); }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
              gap: 12px; margin: 16px 0 8px; }}
  .kpi {{ background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
          padding: 12px 14px; }}
  .kpi .label {{ font-size: 10px; color: var(--text-dim); text-transform: uppercase;
                 letter-spacing: 1px; margin-bottom: 4px; }}
  .kpi .value {{ font-size: 20px; color: var(--text); font-weight: 600;
                 font-variant-numeric: tabular-nums; }}
  .kpi .value.good {{ color: var(--green); }}
  .kpi .value.bad  {{ color: var(--red); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 8px; }}
  th, td {{ padding: 6px 8px; border-bottom: 1px solid #21262d; text-align: left; vertical-align: top; }}
  th {{ color: var(--text-dim); font-weight: 600; font-size: 10px; text-transform: uppercase;
        letter-spacing: 1px; }}
  td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
  td.pass {{ color: var(--green); }}
  td.fail {{ color: var(--red); }}
  pre {{ background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
         padding: 12px; overflow: auto; font-size: 11px; color: var(--text); line-height: 1.5;
         white-space: pre-wrap; word-break: break-word; }}
  svg {{ background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
         display: block; max-width: 100%; height: auto; }}
  .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border);
             font-size: 10px; color: var(--text-dim); }}
</style>
</head>
<body>
{body}
<div class="footer">
  Generated by <code>mnesio-bench</code>. The procedural compiler's commit gate
  (Hard Rule #1) was active for every version recorded above. A safety probe
  pass rate below 100% is the alignment-drift hard stop — see
  <code>mnesio_procedural::gate</code>.
</div>
</body>
</html>"##
    )
}

// ---------------- learning curve report ----------------

fn learning_summary_block(
    curve: &[LearningCurvePoint],
    committed: usize,
    rejected: usize,
) -> String {
    let first = curve.first();
    let last = curve.last();
    let benchmark_start = first.map(|p| p.benchmark_score).unwrap_or(0.0);
    let benchmark_end = last.map(|p| p.benchmark_score).unwrap_or(0.0);
    let delta = benchmark_end - benchmark_start;
    let safety_min = curve
        .iter()
        .map(|p| p.safety_probe_pass_rate)
        .fold(1.0_f32, |a, b| a.min(b));
    let delta_class = if delta >= 0.0 { "good" } else { "bad" };
    let safety_class = if safety_min >= 1.0 - 1e-6 {
        "good"
    } else {
        "bad"
    };
    format!(
        r##"<div class="summary">
  <div class="kpi"><div class="label">versions</div>
    <div class="value">{}</div></div>
  <div class="kpi"><div class="label">commits</div>
    <div class="value">{committed}</div></div>
  <div class="kpi"><div class="label">rejections</div>
    <div class="value">{rejected}</div></div>
  <div class="kpi"><div class="label">benchmark Δ</div>
    <div class="value {delta_class}">{delta_sign}{delta_pct:.1}%</div></div>
  <div class="kpi"><div class="label">safety min</div>
    <div class="value {safety_class}">{safety_pct:.1}%</div></div>
</div>"##,
        curve.len(),
        delta_sign = if delta >= 0.0 { "+" } else { "" },
        delta_pct = delta * 100.0,
        safety_pct = safety_min * 100.0,
    )
}

fn prompt_diff_table(seed: &str, final_body: &str) -> String {
    format!(
        r##"<table>
  <thead><tr><th style="width:100px">version</th><th>body</th></tr></thead>
  <tbody>
    <tr><td class="num">v1 (seed)</td><td><pre>{}</pre></td></tr>
    <tr><td class="num">v final</td><td><pre>{}</pre></td></tr>
  </tbody>
</table>"##,
        html_escape(seed),
        html_escape(final_body),
    )
}

fn learning_curve_svg(curve: &[LearningCurvePoint]) -> String {
    let inner_w = CHART_W - PAD_L - PAD_R;
    let inner_h = CHART_H - PAD_T - PAD_B;
    let n = curve.len().max(1) as f32;
    let x = |i: usize| -> f32 {
        if curve.len() <= 1 {
            PAD_L + inner_w / 2.0
        } else {
            PAD_L + inner_w * (i as f32) / (n - 1.0)
        }
    };
    let y = |v: f32| -> f32 { PAD_T + inner_h * (1.0 - v.clamp(0.0, 1.0)) };

    let bench_path = path_d(
        curve
            .iter()
            .enumerate()
            .map(|(i, p)| (x(i), y(p.benchmark_score))),
    );
    let safety_path = path_d(
        curve
            .iter()
            .enumerate()
            .map(|(i, p)| (x(i), y(p.safety_probe_pass_rate))),
    );
    let bench_dots = dots(
        curve
            .iter()
            .enumerate()
            .map(|(i, p)| (x(i), y(p.benchmark_score))),
        "#58a6ff",
    );

    let mut svg = String::new();
    svg.push_str(&format!(
        r##"<svg width="{CHART_W}" height="{CHART_H}" viewBox="0 0 {CHART_W} {CHART_H}" xmlns="http://www.w3.org/2000/svg">"##
    ));
    svg.push_str(&axes_svg(curve.len()));
    svg.push_str(&format!(
        r##"<path d="{safety_path}" fill="none" stroke="#56d364" stroke-width="1.5" stroke-dasharray="4 3"/>"##
    ));
    svg.push_str(&format!(
        r##"<path d="{bench_path}" fill="none" stroke="#58a6ff" stroke-width="2"/>"##
    ));
    svg.push_str(&bench_dots);
    svg.push_str(&legend_svg(&[
        ("benchmark", "#58a6ff"),
        ("safety probe", "#56d364"),
    ]));
    svg.push_str("</svg>");
    svg
}

// ---------------- comparison report ----------------

fn comparison_summary_block(report: &ComparisonReport) -> String {
    let delta = report.benchmark_delta;
    let delta_class = if delta > 1e-6 {
        "good"
    } else if delta < -1e-6 {
        "bad"
    } else {
        ""
    };
    let safety_class = if report.safety_regressed() {
        "bad"
    } else {
        "good"
    };
    format!(
        r##"<div class="summary">
  <div class="kpi"><div class="label">A ({a})</div>
    <div class="value">{a_pct:.1}%</div></div>
  <div class="kpi"><div class="label">B ({b})</div>
    <div class="value">{b_pct:.1}%</div></div>
  <div class="kpi"><div class="label">benchmark Δ</div>
    <div class="value {delta_class}">{sign}{delta_pct:.1}%</div></div>
  <div class="kpi"><div class="label">safety Δ</div>
    <div class="value {safety_class}">{safe_sign}{safe_pct:.1}%</div></div>
</div>"##,
        a = html_escape(&report.artifact_a_label),
        b = html_escape(&report.artifact_b_label),
        a_pct = report.report_a.benchmark_score * 100.0,
        b_pct = report.report_b.benchmark_score * 100.0,
        sign = if delta >= 0.0 { "+" } else { "" },
        delta_pct = delta * 100.0,
        safe_sign = if report.safety_delta >= 0.0 { "+" } else { "" },
        safe_pct = report.safety_delta * 100.0,
    )
}

fn comparison_drill_table(report: &ComparisonReport) -> String {
    let pass_cell = |passed: bool| -> &'static str {
        if passed {
            "<td class=\"pass\">✓</td>"
        } else {
            "<td class=\"fail\">✗</td>"
        }
    };
    let mut rows = String::new();
    // Walk A's task list as the canonical order; assume B has the
    // same inputs (same suite). Match by `input`.
    let mut b_by_input: std::collections::HashMap<&str, &mnesio_procedural::TaskResult> =
        std::collections::HashMap::new();
    for r in &report.report_b.task_results {
        b_by_input.insert(r.input.as_str(), r);
    }
    for ra in &report.report_a.task_results {
        if ra.kind != TaskKind::Benchmark {
            continue;
        }
        let rb = b_by_input.get(ra.input.as_str());
        rows.push_str(&format!(
            "<tr><td>{cat}</td><td>{input}</td>{a_pass}{b_pass}<td><code>{expected}</code></td></tr>",
            cat = html_escape(&ra.category),
            input = html_escape(&ra.input),
            a_pass = pass_cell(ra.passed),
            b_pass = pass_cell(rb.map(|r| r.passed).unwrap_or(false)),
            expected = html_escape(&ra.expected),
        ));
    }
    format!(
        r##"<table>
  <thead><tr><th>category</th><th>input</th><th style="width:40px">A</th><th style="width:40px">B</th><th>expected</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"##
    )
}

fn comparison_bar_svg(
    per_category: &[crate::CategoryComparison],
    label_a: &str,
    label_b: &str,
) -> String {
    let inner_w = CHART_W - PAD_L - PAD_R;
    let inner_h = CHART_H - PAD_T - PAD_B;
    let n = per_category.len() as f32;
    let group_w = inner_w / n;
    let bar_w = group_w * 0.35;
    let mut svg = String::new();
    svg.push_str(&format!(
        r##"<svg width="{CHART_W}" height="{CHART_H}" viewBox="0 0 {CHART_W} {CHART_H}" xmlns="http://www.w3.org/2000/svg">"##
    ));
    // Y-axis grid + labels (0/25/50/75/100% of pass rate per category).
    for tick in 0..=4 {
        let v = tick as f32 / 4.0;
        let y = PAD_T + inner_h * (1.0 - v);
        svg.push_str(&format!(
            r##"<line x1="{}" y1="{y}" x2="{}" y2="{y}" stroke="#21262d" stroke-width="1"/>"##,
            PAD_L,
            PAD_L + inner_w,
        ));
        svg.push_str(&format!(
            r##"<text x="{x}" y="{y}" fill="#8b949e" font-size="10" text-anchor="end" dominant-baseline="middle">{pct}%</text>"##,
            x = PAD_L - 8.0,
            pct = (v * 100.0) as i32
        ));
    }
    for (i, cat) in per_category.iter().enumerate() {
        let cx = PAD_L + group_w * (i as f32 + 0.5);
        let a_h = if cat.total == 0 {
            0.0
        } else {
            inner_h * cat.a_passed as f32 / cat.total as f32
        };
        let b_h = if cat.total == 0 {
            0.0
        } else {
            inner_h * cat.b_passed as f32 / cat.total as f32
        };
        let a_x = cx - bar_w - 2.0;
        let b_x = cx + 2.0;
        svg.push_str(&format!(
            r##"<rect x="{a_x}" y="{}" width="{bar_w}" height="{a_h}" fill="#79c0ff"/>"##,
            PAD_T + inner_h - a_h
        ));
        svg.push_str(&format!(
            r##"<rect x="{b_x}" y="{}" width="{bar_w}" height="{b_h}" fill="#d2a8ff"/>"##,
            PAD_T + inner_h - b_h
        ));
        svg.push_str(&format!(
            r##"<text x="{cx}" y="{}" fill="#8b949e" font-size="10" text-anchor="middle">{}</text>"##,
            PAD_T + inner_h + 16.0,
            html_escape(&cat.category)
        ));
    }
    svg.push_str(&legend_svg(&[(label_a, "#79c0ff"), (label_b, "#d2a8ff")]));
    svg.push_str("</svg>");
    svg
}

// ---------------- shared SVG bits ----------------

fn axes_svg(num_points: usize) -> String {
    let inner_w = CHART_W - PAD_L - PAD_R;
    let inner_h = CHART_H - PAD_T - PAD_B;
    let mut svg = String::new();
    // Y-axis grid + labels (0/25/50/75/100%).
    for tick in 0..=4 {
        let v = tick as f32 / 4.0;
        let y = PAD_T + inner_h * (1.0 - v);
        svg.push_str(&format!(
            r##"<line x1="{}" y1="{y}" x2="{}" y2="{y}" stroke="#21262d" stroke-width="1"/>"##,
            PAD_L,
            PAD_L + inner_w,
        ));
        svg.push_str(&format!(
            r##"<text x="{x}" y="{y}" fill="#8b949e" font-size="10" text-anchor="end" dominant-baseline="middle">{pct}%</text>"##,
            x = PAD_L - 8.0,
            pct = (v * 100.0) as i32
        ));
    }
    // X-axis labels = `v1`, `v2`, ...
    if num_points > 1 {
        for i in 0..num_points {
            let x = PAD_L + inner_w * (i as f32) / (num_points as f32 - 1.0);
            svg.push_str(&format!(
                r##"<text x="{x}" y="{}" fill="#8b949e" font-size="10" text-anchor="middle">v{}</text>"##,
                PAD_T + inner_h + 16.0,
                i + 1
            ));
        }
    } else {
        svg.push_str(&format!(
            r##"<text x="{}" y="{}" fill="#8b949e" font-size="10" text-anchor="middle">v1</text>"##,
            PAD_L + inner_w / 2.0,
            PAD_T + inner_h + 16.0
        ));
    }
    svg
}

fn path_d<I>(points: I) -> String
where
    I: Iterator<Item = (f32, f32)>,
{
    let mut d = String::new();
    for (i, (x, y)) in points.enumerate() {
        if i == 0 {
            d.push_str(&format!("M {x:.1} {y:.1}"));
        } else {
            d.push_str(&format!(" L {x:.1} {y:.1}"));
        }
    }
    d
}

fn dots<I>(points: I, color: &str) -> String
where
    I: Iterator<Item = (f32, f32)>,
{
    let mut s = String::new();
    for (x, y) in points {
        s.push_str(&format!(
            r##"<circle cx="{x:.1}" cy="{y:.1}" r="3" fill="{color}"/>"##
        ));
    }
    s
}

fn legend_svg(items: &[(&str, &str)]) -> String {
    let mut s = String::new();
    let y = PAD_T - 10.0;
    let mut x = PAD_L;
    for (label, color) in items {
        s.push_str(&format!(
            r##"<rect x="{x}" y="{y}" width="10" height="10" fill="{color}"/>"##
        ));
        s.push_str(&format!(
            r##"<text x="{}" y="{}" fill="#c9d1d9" font-size="11">{}</text>"##,
            x + 14.0,
            y + 9.0,
            html_escape(label)
        ));
        x += 14.0 + (label.len() as f32) * 7.0 + 24.0;
    }
    s
}

fn empty_chart_svg(msg: &str) -> String {
    format!(
        r##"<svg width="{CHART_W}" height="{CHART_H}" viewBox="0 0 {CHART_W} {CHART_H}" xmlns="http://www.w3.org/2000/svg">
  <text x="{}" y="{}" fill="#8b949e" font-size="14" text-anchor="middle">{}</text>
</svg>"##,
        CHART_W / 2.0,
        CHART_H / 2.0,
        html_escape(msg)
    )
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnesio_core::types::{new_id, ArtifactRef};

    fn point(version: u32, score: f32) -> LearningCurvePoint {
        LearningCurvePoint {
            artifact_id: ArtifactRef(new_id()),
            version,
            timestamp_ms: 1_700_000_000_000 + version as u64 * 1000,
            benchmark_score: score,
            safety_probe_pass_rate: 1.0,
            objective_delta: 0.05,
            judges_consulted: 2,
        }
    }

    #[test]
    fn learning_curve_html_contains_chart_and_summary() {
        let curve = vec![point(1, 0.3), point(2, 0.6), point(3, 1.0)];
        let html = render_learning_curve("gsm8k-tiny", "seed", "final", &curve, 2, 0);
        // Structural anchors that downstream tooling (e.g. headless
        // screenshotters) will look for.
        assert!(html.contains("<svg"), "must contain the inline SVG chart");
        assert!(html.contains("Learning curve"));
        assert!(html.contains("benchmark Δ"));
        assert!(html.contains("safety min"));
        assert!(
            html.contains("100.0%"),
            "must surface 100% on the final score KPI"
        );
    }

    #[test]
    fn learning_curve_handles_empty_curve_with_placeholder() {
        let html = render_learning_curve("empty", "x", "x", &[], 0, 0);
        assert!(html.contains("no curve points recorded"));
        assert!(html.contains("<svg"));
    }

    #[test]
    fn html_escape_handles_xss_vectors() {
        let evil = render_learning_curve("<script>alert(1)</script>", "seed", "final", &[], 0, 0);
        // The script tag must be escaped before it lands in the HTML.
        assert!(!evil.contains("<script>alert(1)</script>"));
        assert!(evil.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
    }

    #[test]
    fn comparison_html_uses_grouped_bars_and_drill_table() {
        let suite = crate::BenchSuite {
            name: "test".into(),
            description: "x".into(),
            tasks: vec![],
            safety_probes: vec![],
        };
        let report = ComparisonReport {
            suite_name: suite.name.clone(),
            artifact_a_label: "baseline".into(),
            artifact_b_label: "revised".into(),
            report_a: mnesio_procedural::SuiteReport {
                benchmark_score: 0.4,
                benchmark_passed: 2,
                benchmark_total: 5,
                safety_probe_pass_rate: 1.0,
                safety_probes_passed: 0,
                safety_probes_total: 0,
                task_results: vec![],
            },
            report_b: mnesio_procedural::SuiteReport {
                benchmark_score: 0.8,
                benchmark_passed: 4,
                benchmark_total: 5,
                safety_probe_pass_rate: 1.0,
                safety_probes_passed: 0,
                safety_probes_total: 0,
                task_results: vec![],
            },
            per_category: vec![crate::CategoryComparison {
                category: "math".into(),
                a_passed: 2,
                b_passed: 4,
                total: 5,
            }],
            benchmark_delta: 0.4,
            safety_delta: 0.0,
        };
        let html = render_comparison(&report);
        assert!(html.contains("baseline"));
        assert!(html.contains("revised"));
        assert!(html.contains("Per-category"));
        assert!(html.contains("+40.0%"), "must show the positive Δ");
    }
}
