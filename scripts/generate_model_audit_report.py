#!/usr/bin/env python3
"""
Generate a model audit report for Olympus.

The goal is not to prove the model is profitable. It is to summarize whether
the current signal survives realistic, leakage-aware evaluation and to make the
project's engineering rigor easy to inspect.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_PATH = ROOT / "data" / "features.csv"
MODEL_PATH = ROOT / "models" / "direction_model.pkl"
REPORT_DIR = ROOT / "reports"
PORTFOLIO_DIR = ROOT / "portfolio"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_metrics_log(path: Path) -> pd.DataFrame:
    """Read metrics_log.csv even if older runs wrote a shorter header."""
    if not path.exists():
        return pd.DataFrame()
    rows: list[list[str]] = []
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return pd.DataFrame()

    body = rows[1:]
    max_len = max(len(r) for r in body)
    full_cols = [
        "timestamp_utc",
        "model_kind",
        "train_cutoff",
        "n_features",
        "train_rows",
        "test_rows",
        "task",
        "target",
        "calibrated_global",
        "recency_weights",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "majority_baseline",
        "roc_auc",
        "brier",
        "log_loss",
        "rmse",
        "pearson_r",
    ]
    if max_len >= len(full_cols):
        normalized = [r + [""] * (len(full_cols) - len(r)) for r in body]
        return pd.DataFrame(normalized, columns=full_cols)
    return read_csv(path)


def fmt_pct(x: Any) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "n/a"


def fmt_num(x: Any, digits: int = 3) -> str:
    try:
        if pd.isna(x):
            return "n/a"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "n/a"


def safe_float(x: Any) -> float | None:
    try:
        val = float(x)
    except Exception:
        return None
    if pd.isna(val):
        return None
    return val


def status_for(value: float | None, pass_at: float, monitor_at: float, higher_is_better: bool = True) -> str:
    if value is None or pd.isna(value):
        return "missing"
    if higher_is_better:
        if value >= pass_at:
            return "pass"
        if value >= monitor_at:
            return "monitor"
        return "fail"
    if value <= pass_at:
        return "pass"
    if value <= monitor_at:
        return "monitor"
    return "fail"


def status_rank(status: str) -> int:
    return {"pass": 0, "monitor": 1, "missing": 2, "fail": 3}.get(status, 3)


def status_label(status: str) -> str:
    return {
        "pass": "PASS",
        "monitor": "MONITOR",
        "missing": "MISSING",
        "fail": "FAIL",
    }.get(status, "FAIL")


def latest_metrics(metrics: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    if metrics.empty:
        return {}
    if target_column and "target" in metrics.columns:
        matched = metrics[metrics["target"].astype(str) == str(target_column)]
        if not matched.empty:
            return matched.iloc[-1].to_dict()
    return metrics.iloc[-1].to_dict()


def best_target_row(bench: pd.DataFrame) -> dict[str, Any]:
    if bench.empty or "roc_auc" not in bench.columns:
        return {}
    b = bench.copy()
    b["roc_auc"] = pd.to_numeric(b["roc_auc"], errors="coerce")
    b = b.dropna(subset=["roc_auc"])
    if b.empty:
        return {}
    return b.sort_values("roc_auc", ascending=False).iloc[0].to_dict()


def strategy_row(df: pd.DataFrame, name: str) -> dict[str, Any]:
    if df.empty:
        return {}
    key_col = "strategy" if "strategy" in df.columns else "label" if "label" in df.columns else None
    if key_col is None:
        return {}
    row = df[df[key_col].astype(str) == name]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def wf_summary(wf: pd.DataFrame) -> dict[str, Any]:
    if wf.empty or "roc_auc" not in wf.columns:
        return {}
    w = wf.copy()
    w["roc_auc"] = pd.to_numeric(w["roc_auc"], errors="coerce")
    if "portfolio_mode" in w.columns:
        model_rows = w[~w["portfolio_mode"].astype(str).str.startswith("baseline", na=False)]
    else:
        model_rows = w
    if model_rows.empty:
        model_rows = w
    auc_by_fold = model_rows.dropna(subset=["roc_auc"]).groupby("fold")["roc_auc"].mean()
    if auc_by_fold.empty:
        return {}
    return {
        "folds": int(auc_by_fold.shape[0]),
        "mean_auc": float(auc_by_fold.mean()),
        "min_auc": float(auc_by_fold.min()),
        "max_auc": float(auc_by_fold.max()),
    }


def feature_health(features: pd.DataFrame) -> dict[str, Any]:
    if features.empty:
        return {}
    out: dict[str, Any] = {
        "rows": int(len(features)),
        "tickers": int(features["ticker"].nunique()) if "ticker" in features.columns else 0,
        "dates": int(features["Date"].nunique()) if "Date" in features.columns else 0,
        "feature_set_version": str(features["feature_set_version"].iloc[0])
        if "feature_set_version" in features.columns and len(features)
        else "unknown",
    }
    if "Date" in features.columns:
        d = pd.to_datetime(features["Date"], errors="coerce")
        out["date_min"] = str(d.min().date()) if d.notna().any() else "unknown"
        out["date_max"] = str(d.max().date()) if d.notna().any() else "unknown"

    news_cols = [
        c
        for c in features.columns
        if any(token in c for token in ("news_", "sentiment_", "weighted_sentiment"))
    ]
    out["news_feature_count"] = len(news_cols)
    if news_cols:
        zero_fracs = [
            float((pd.to_numeric(features[c], errors="coerce").fillna(0.0) == 0.0).mean())
            for c in news_cols
        ]
        out["max_news_zero_frac"] = float(max(zero_fracs))
        out["mean_news_zero_frac"] = float(np.mean(zero_fracs))
    return out


def build_audit(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any], str]:
    features = pd.read_csv(args.features, parse_dates=["Date"]) if Path(args.features).exists() else pd.DataFrame()
    metrics = read_metrics_log(args.reports_dir / "metrics_log.csv")
    bench = read_csv(args.reports_dir / "model_target_benchmark.csv")
    backtest = read_csv(args.reports_dir / "backtest_comparison.csv")
    ranking = read_csv(args.reports_dir / "ranking_alpha_summary.csv")
    sweep = read_csv(args.reports_dir / "rank_feature_sweep.csv")
    wf = read_csv(args.reports_dir / "walk_forward.csv")
    wf_next3d = read_csv(args.reports_dir / "walk_forward_next3d.csv")

    bundle: dict[str, Any] = {}
    if Path(args.model).exists():
        try:
            bundle = joblib.load(args.model)
        except Exception:
            bundle = {}

    fh = feature_health(features)
    latest = latest_metrics(metrics, str(bundle.get("target_column") or "") if bundle else None)
    best_target = best_target_row(bench)
    model_top = strategy_row(backtest, "model_top_0.20_long")
    spy = strategy_row(backtest, "baseline_spy_long_only")
    rank_spy = strategy_row(ranking, "top_minus_spy")
    rank_spread = strategy_row(ranking, "top_minus_bottom")
    best_raw = sweep.iloc[0].to_dict() if not sweep.empty else {}
    wf_default = wf_summary(wf)
    wf_3d = wf_summary(wf_next3d)

    checks = [
        {
            "area": "Data coverage",
            "check": "Universe has enough rows and assets for cross-sectional tests",
            "value": f"{fh.get('rows', 0):,} rows / {fh.get('tickers', 0)} tickers / {fh.get('dates', 0)} dates",
            "status": "pass" if fh.get("rows", 0) >= 20_000 and fh.get("tickers", 0) >= 25 else "monitor",
            "evidence": f"{fh.get('date_min', 'unknown')} to {fh.get('date_max', 'unknown')} | features {fh.get('feature_set_version', 'unknown')}",
        },
        {
            "area": "Label integrity",
            "check": "Model uses realistic next-open target metadata",
            "value": str(bundle.get("target_column") or latest.get("target") or "unknown"),
            "status": "pass"
            if str(bundle.get("target_column") or latest.get("target") or "").startswith("target_intraday")
            or "next_open_to_close" in str(bundle.get("target_column") or latest.get("target") or "")
            else "monitor",
            "evidence": "Bundle stores target_column and feature_set_version.",
        },
        {
            "area": "Feature health",
            "check": "Sparse news features are detected instead of trusted blindly",
            "value": f"mean news zero frac {fmt_pct(fh.get('mean_news_zero_frac'))}",
            "status": "monitor" if fh.get("mean_news_zero_frac", 0.0) > 0.95 else "pass",
            "evidence": "Sparse-feature pruning is enabled in train_model.py with --max-zero-frac.",
        },
        {
            "area": "Holdout model quality",
            "check": "Latest model beats random ranking quality",
            "value": f"ROC AUC {fmt_num(latest.get('roc_auc'), 4)}",
            "status": status_for(safe_float(latest.get("roc_auc")), pass_at=0.55, monitor_at=0.52),
            "evidence": "Latest chronological holdout run from reports/metrics_log.csv.",
        },
        {
            "area": "Target benchmark",
            "check": "Best tested target shows meaningful separation",
            "value": f"{best_target.get('target_key', 'n/a')} AUC {fmt_num(best_target.get('roc_auc'), 4)}",
            "status": status_for(safe_float(best_target.get("roc_auc")), pass_at=0.55, monitor_at=0.52),
            "evidence": "All targets are trained on the same chronological split.",
        },
        {
            "area": "Backtest baseline",
            "check": "Top-ranked model basket beats SPY on holdout",
            "value": f"model {fmt_pct(model_top.get('total_return'))} vs SPY {fmt_pct(spy.get('total_return'))}",
            "status": "pass"
            if safe_float(model_top.get("total_return")) is not None
            and safe_float(spy.get("total_return")) is not None
            and safe_float(model_top.get("total_return")) > safe_float(spy.get("total_return"))
            else "fail",
            "evidence": "Uses next-open-to-close execution and holdout-only scoring.",
        },
        {
            "area": "Ranking alpha",
            "check": "Top-ranked names beat SPY and bottom-ranked names",
            "value": (
                f"top-SPY Sharpe {fmt_num(rank_spy.get('sharpe'))}; "
                f"top-bottom Sharpe {fmt_num(rank_spread.get('sharpe'))}"
            ),
            "status": "pass"
            if (safe_float(rank_spy.get("sharpe")) or -999.0) > 0
            and (safe_float(rank_spread.get("sharpe")) or -999.0) > 0
            else "fail",
            "evidence": "Cross-sectional ranking alpha report excludes SPY from the traded universe.",
        },
        {
            "area": "Raw feature sanity",
            "check": "Standalone feature sweep finds a robust simple signal",
            "value": (
                f"{best_raw.get('feature', 'n/a')}:{best_raw.get('direction', 'n/a')} "
                f"spread Sharpe {fmt_num(best_raw.get('top_minus_bottom_sharpe'))}"
            ),
            "status": "pass" if (safe_float(best_raw.get("top_minus_bottom_sharpe")) or -999.0) > 0.5 else "fail",
            "evidence": "Every numeric model feature is tested both high and low.",
        },
        {
            "area": "Walk-forward",
            "check": "Retrained folds remain above random",
            "value": f"default mean AUC {fmt_num(wf_default.get('mean_auc'), 4)}",
            "status": status_for(safe_float(wf_default.get("mean_auc")), pass_at=0.55, monitor_at=0.52),
            "evidence": "Each fold trains on past data and scores the next unseen block.",
        },
        {
            "area": "Longer-horizon research",
            "check": "3-day model improves enough to justify further research",
            "value": f"3d WF mean AUC {fmt_num(wf_3d.get('mean_auc'), 4)}",
            "status": status_for(safe_float(wf_3d.get("mean_auc")), pass_at=0.55, monitor_at=0.52),
            "evidence": "3d returns are overlapping diagnostics, not final portfolio stats.",
        },
    ]

    checks_df = pd.DataFrame(checks)
    worst = max((status_rank(s) for s in checks_df["status"]), default=3)
    if worst >= status_rank("fail"):
        deploy_verdict = "REJECT_FOR_TRADING"
    elif worst >= status_rank("monitor"):
        deploy_verdict = "MONITOR_ONLY"
    else:
        deploy_verdict = "RESEARCH_CANDIDATE"

    project_verdict = "PORTFOLIO_READY_MODEL_AUDIT_PLATFORM"
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "deploy_verdict": deploy_verdict,
        "project_verdict": project_verdict,
        "model_path": str(Path(args.model).resolve()),
        "features_path": str(Path(args.features).resolve()),
        "target_column": bundle.get("target_column") or latest.get("target"),
        "model_kind": bundle.get("model_kind") or latest.get("model_kind"),
        "feature_count": len(bundle.get("feature_names") or []),
        "feature_set_version": bundle.get("feature_set_version") or fh.get("feature_set_version"),
    }
    return checks_df, meta, render_markdown(checks_df, meta, fh, latest, best_target)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No data available._"
    d = df[cols].astype(str).copy()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(str(row[c]).replace("\n", " ").replace("|", "\\|") for c in cols) + " |"
        for _, row in d.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def render_markdown(
    checks_df: pd.DataFrame,
    meta: dict[str, Any],
    fh: dict[str, Any],
    latest: dict[str, Any],
    best_target: dict[str, Any],
) -> str:
    status_counts = checks_df["status"].value_counts().to_dict()
    return f"""# Olympus Model Audit Report

Generated: `{meta['generated_utc']}`

## Executive Verdict

**Deployment verdict:** `{meta['deploy_verdict']}`

**Portfolio verdict:** `{meta['project_verdict']}`

Olympus is currently best presented as an **honest ML research and model-audit platform**, not as a production trading system. The current signal does not show stable selection alpha, but the project now demonstrates the stronger engineering story: leakage-aware labels, realistic execution assumptions, baseline comparisons, cross-sectional ranking evaluation, raw-feature sanity checks, and walk-forward testing.

## System Snapshot

- Model: `{meta.get('model_path')}`
- Model kind: `{meta.get('model_kind')}`
- Target column: `{meta.get('target_column')}`
- Feature set: `{meta.get('feature_set_version')}`
- Model features: `{meta.get('feature_count')}`
- Dataset: `{fh.get('rows', 0):,}` rows, `{fh.get('tickers', 0)}` tickers, `{fh.get('dates', 0)}` trading dates
- Date range: `{fh.get('date_min', 'unknown')}` to `{fh.get('date_max', 'unknown')}`
- Latest holdout AUC: `{fmt_num(latest.get('roc_auc'), 4)}`
- Best target benchmark: `{best_target.get('target_key', 'n/a')}` at AUC `{fmt_num(best_target.get('roc_auc'), 4)}`

## Audit Checklist

{markdown_table(checks_df, ["area", "check", "value", "status", "evidence"])}

## Status Counts

- Pass: `{status_counts.get('pass', 0)}`
- Monitor: `{status_counts.get('monitor', 0)}`
- Missing: `{status_counts.get('missing', 0)}`
- Fail: `{status_counts.get('fail', 0)}`

## Interpretation

The project is compelling because it shows the full lifecycle of an ML system under pressure:

1. Build an end-to-end prediction pipeline.
2. Discover that attractive initial metrics can be misleading.
3. Add safeguards for target leakage, execution realism, benchmark fairness, and walk-forward validation.
4. Reject the current signal when it fails those tests.
5. Produce reproducible reports that explain exactly why the model is or is not deployable.

That is the differentiated portfolio story: **the tool tells the truth even when the model is not ready.**

## Recommended Next Research

- Add sector-relative and beta-adjusted features.
- Add VIX, rates, credit-spread, and market-regime context.
- Build a non-overlapping multi-day portfolio simulator.
- Replace sparse news input with broader timestamped coverage or remove news features from the active feature list.
- Use the audit checklist as the promotion gate before any paper-trading or live-trading claim.
"""


def render_html(md: str, checks_df: pd.DataFrame, meta: dict[str, Any]) -> str:
    rows = []
    for _, r in checks_df.iterrows():
        cls = str(r["status"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(r['area']))}</td>"
            f"<td>{html.escape(str(r['check']))}</td>"
            f"<td><code>{html.escape(str(r['value']))}</code></td>"
            f"<td><span class='status {cls}'>{status_label(cls)}</span></td>"
            f"<td>{html.escape(str(r['evidence']))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Olympus Model Audit Report</title>
  <style>
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f172a; color: #e2e8f0; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 48px 24px 72px; }}
    .hero {{ border: 1px solid rgba(148, 163, 184, .2); background: linear-gradient(135deg, rgba(14,165,233,.14), rgba(15,23,42,.9)); padding: 28px; border-radius: 12px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: -0.02em; }}
    h2 {{ margin-top: 34px; color: #f8fafc; }}
    p, li {{ color: #94a3b8; line-height: 1.62; }}
    code {{ color: #bae6fd; background: rgba(14, 165, 233, .12); padding: 2px 6px; border-radius: 6px; }}
    .verdict {{ display: inline-flex; margin-top: 14px; padding: 8px 12px; border-radius: 999px; font-weight: 700; letter-spacing: .04em; font-size: 12px; background: rgba(251, 191, 36, .13); color: #fbbf24; border: 1px solid rgba(251, 191, 36, .32); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }}
    th {{ color: #94a3b8; text-align: left; text-transform: uppercase; letter-spacing: .06em; font-size: 11px; border-bottom: 1px solid rgba(148, 163, 184, .22); padding: 10px; }}
    td {{ border-bottom: 1px solid rgba(148, 163, 184, .12); padding: 12px 10px; vertical-align: top; }}
    .status {{ display: inline-flex; padding: 4px 8px; border-radius: 999px; font-size: 11px; font-weight: 800; }}
    .pass {{ color: #86efac; background: rgba(34, 197, 94, .13); }}
    .monitor {{ color: #fbbf24; background: rgba(251, 191, 36, .13); }}
    .missing {{ color: #cbd5e1; background: rgba(148, 163, 184, .14); }}
    .fail {{ color: #fda4af; background: rgba(244, 63, 94, .14); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 18px; }}
    .card {{ border: 1px solid rgba(148, 163, 184, .15); border-radius: 10px; padding: 14px; background: rgba(15, 23, 42, .65); }}
    .label {{ color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; font-weight: 800; }}
    .value {{ margin-top: 6px; color: #f8fafc; font-size: 18px; font-weight: 750; }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>Olympus Model Audit Report</h1>
    <p>Leakage-aware evaluation report for a market-signal ML research platform.</p>
    <div class="verdict">{html.escape(str(meta["deploy_verdict"]))}</div>
  </section>
  <div class="grid">
    <div class="card"><div class="label">Project Verdict</div><div class="value">{html.escape(str(meta["project_verdict"]))}</div></div>
    <div class="card"><div class="label">Target</div><div class="value">{html.escape(str(meta.get("target_column")))}</div></div>
    <div class="card"><div class="label">Model Kind</div><div class="value">{html.escape(str(meta.get("model_kind")))}</div></div>
    <div class="card"><div class="label">Feature Set</div><div class="value">{html.escape(str(meta.get("feature_set_version")))}</div></div>
  </div>
  <h2>Audit Checklist</h2>
  <table>
    <thead><tr><th>Area</th><th>Check</th><th>Value</th><th>Status</th><th>Evidence</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Interpretation</h2>
  <p>Olympus is portfolio-ready as an ML evaluation platform. The current signal is not deployment-ready, and the report says so plainly. That honesty is the point: the system is designed to catch inflated results before they become claims.</p>
  <p>Generated at <code>{html.escape(str(meta["generated_utc"]))}</code>.</p>
</main>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Olympus model audit report")
    p.add_argument("--features", type=Path, default=DATA_PATH)
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    p.add_argument("--reports-dir", type=Path, default=REPORT_DIR)
    p.add_argument("--out-md", type=Path, default=PORTFOLIO_DIR / "olympus_model_audit_report.md")
    p.add_argument("--out-html", type=Path, default=PORTFOLIO_DIR / "olympus_model_audit_report.html")
    p.add_argument("--out-summary", type=Path, default=REPORT_DIR / "model_audit_summary.csv")
    p.add_argument("--out-json", type=Path, default=REPORT_DIR / "model_audit_summary.json")
    args = p.parse_args()

    args.reports_dir = args.reports_dir.resolve()
    checks_df, meta, md = build_audit(args)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    args.out_md.write_text(md, encoding="utf-8")
    args.out_html.write_text(render_html(md, checks_df, meta), encoding="utf-8")
    checks_df.to_csv(args.out_summary, index=False)
    args.out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Deployment verdict: {meta['deploy_verdict']}")
    print(f"Portfolio verdict: {meta['project_verdict']}")
    print(f"Saved {args.out_md}")
    print(f"Saved {args.out_html}")
    print(f"Saved {args.out_summary}")


if __name__ == "__main__":
    main()
