#!/usr/bin/env python3
"""
Sweep raw features as cross-sectional rank signals.

For each candidate feature, evaluate both directions:
  - high: higher feature value ranks better
  - low: lower feature value ranks better

Outputs summary, monthly stability, and daily IC/portfolio files. This helps
answer whether any simple signal beats the current ML ranker under realistic
execution before spending time on model tuning.

Usage:
  python scripts/sweep_rank_features.py
  python scripts/sweep_rank_features.py --top-n 2
  python scripts/sweep_rank_features.py --date-from 2025-04-28 --out-dir reports/recent_sweep
  python scripts/sweep_rank_features.py --features-list return_5d momentum_20d rsi_14
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from evaluate_backtest import EXECUTION_PRICE_CHOICES, attach_execution_returns
from evaluate_ranking_alpha import (
    build_daily_ranking_returns,
    monthly_stability,
    score_buckets,
    summarize_daily,
)

DATA_PATH = ROOT / "data" / "features.csv"
REPORT_DIR = ROOT / "reports"


def _load_train_module():
    path = ROOT / "scripts" / "train_model.py"
    spec = importlib.util.spec_from_file_location("_srf_train_model", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def candidate_features(
    df: pd.DataFrame,
    requested: list[str] | None = None,
    max_zero_frac: float | None = 0.995,
) -> list[str]:
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(f"Missing requested features: {missing}")
        return requested

    tm = _load_train_module()
    cols = [c for c in tm.FEATURE_COLUMNS if c in df.columns]
    numeric_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 20:
            numeric_cols.append(c)
    if numeric_cols and hasattr(tm, "sparsity_prune"):
        numeric_cols = tm.sparsity_prune(numeric_cols, df[numeric_cols], max_zero_frac=max_zero_frac)
    return numeric_cols


def _pick_summary(summary: pd.DataFrame, strategy: str) -> dict:
    row = summary[summary["strategy"] == strategy]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def evaluate_feature_signal(
    df: pd.DataFrame,
    feature: str,
    direction: str,
    return_col: str,
    cost_bps: float,
    top_n: int | None,
    top_pct: float,
    include_spy: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if direction not in {"high", "low"}:
        raise ValueError("direction must be high or low")
    d = df.copy()
    score_col = "_rank_score"
    d[score_col] = d[feature].astype(float)
    if direction == "low":
        d[score_col] = -d[score_col]

    daily = build_daily_ranking_returns(
        d,
        score_col=score_col,
        return_col=return_col,
        cost_bps=cost_bps,
        top_n=top_n,
        top_pct=top_pct,
        include_spy=include_spy,
    )
    summary = summarize_daily(daily)
    buckets = score_buckets(d, score_col, return_col, include_spy=include_spy)
    return daily, summary, buckets


def summarize_feature(
    feature: str,
    direction: str,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    buckets: pd.DataFrame,
) -> dict:
    top = _pick_summary(summary, "top_long")
    spread = _pick_summary(summary, "top_minus_bottom")
    spy = _pick_summary(summary, "top_minus_spy")
    ic = _pick_summary(summary, "ic_spearman")
    ew = _pick_summary(summary, "ew_ranked")

    bucket_spread = float("nan")
    if len(buckets) and {"score_bucket", "mean_return"}.issubset(buckets.columns):
        b = buckets.set_index(buckets["score_bucket"].astype(str))
        if "q5_high" in b.index and "q1_low" in b.index:
            bucket_spread = float(b.loc["q5_high", "mean_return"] - b.loc["q1_low", "mean_return"])

    return {
        "feature": feature,
        "direction": direction,
        "days": int(top.get("days", len(daily))) if top else int(len(daily)),
        "top_total_return": float(top.get("total_return", np.nan)),
        "top_sharpe": float(top.get("sharpe", np.nan)),
        "top_max_drawdown": float(top.get("max_drawdown", np.nan)),
        "top_win_rate": float(top.get("daily_win_rate", np.nan)),
        "top_minus_bottom_total_return": float(spread.get("total_return", np.nan)),
        "top_minus_bottom_sharpe": float(spread.get("sharpe", np.nan)),
        "top_minus_spy_total_return": float(spy.get("total_return", np.nan)),
        "top_minus_spy_sharpe": float(spy.get("sharpe", np.nan)),
        "ew_ranked_total_return": float(ew.get("total_return", np.nan)),
        "ic_mean": float(ic.get("mean_daily_return", np.nan)),
        "ic_sharpe": float(ic.get("sharpe", np.nan)),
        "ic_positive_rate": float(ic.get("daily_win_rate", np.nan)),
        "bucket_q5_minus_q1_mean_return": bucket_spread,
        "top_beats_spy_rate": float(daily["top_beats_spy"].mean()) if "top_beats_spy" in daily else np.nan,
        "top_beats_bottom_rate": (
            float(daily["top_beats_bottom"].mean()) if "top_beats_bottom" in daily else np.nan
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep raw features as ranking signals")
    p.add_argument("--features", type=str, default=str(DATA_PATH))
    p.add_argument(
        "--execution-price",
        choices=EXECUTION_PRICE_CHOICES,
        default="next_open_to_close",
    )
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--top-n", type=int, default=2)
    p.add_argument("--top-pct", type=float, default=0.33)
    p.add_argument(
        "--include-spy",
        action="store_true",
        help="Include SPY in ranked/traded universe. Default excludes SPY and uses it as benchmark.",
    )
    p.add_argument(
        "--features-list",
        nargs="*",
        default=None,
        help="Optional explicit feature names to sweep.",
    )
    p.add_argument("--date-from", type=str, default=None, help="Only evaluate rows on/after this date")
    p.add_argument("--date-to", type=str, default=None, help="Only evaluate rows on/before this date")
    p.add_argument(
        "--max-zero-frac",
        type=float,
        default=0.995,
        help="Drop candidate features whose zero fraction is above this (1.0 disables)",
    )
    p.add_argument("--out-dir", type=str, default=str(REPORT_DIR))
    args = p.parse_args()

    df = pd.read_csv(args.features, parse_dates=["Date"]).sort_values(["Date", "ticker"])
    df, return_col = attach_execution_returns(df, args.execution_price)
    if args.date_from:
        df = df[df["Date"] >= pd.to_datetime(args.date_from)].copy()
    if args.date_to:
        df = df[df["Date"] <= pd.to_datetime(args.date_to)].copy()
    feats = candidate_features(df, args.features_list, max_zero_frac=args.max_zero_frac)

    rows: list[dict] = []
    daily_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    bucket_parts: list[pd.DataFrame] = []

    for feature in feats:
        base = df.dropna(subset=[feature, return_col, "Date", "ticker"]).copy()
        if len(base) == 0:
            continue
        if base[feature].nunique(dropna=True) < 2:
            continue
        for direction in ["high", "low"]:
            daily, summary, buckets = evaluate_feature_signal(
                base,
                feature=feature,
                direction=direction,
                return_col=return_col,
                cost_bps=args.cost_bps,
                top_n=args.top_n,
                top_pct=args.top_pct,
                include_spy=args.include_spy,
            )
            if len(daily) == 0:
                continue
            rows.append(summarize_feature(feature, direction, daily, summary, buckets))

            daily_tagged = daily.copy()
            daily_tagged.insert(0, "feature", feature)
            daily_tagged.insert(1, "direction", direction)
            daily_parts.append(daily_tagged)

            monthly = monthly_stability(daily)
            monthly.insert(0, "feature", feature)
            monthly.insert(1, "direction", direction)
            monthly_parts.append(monthly)

            buckets = buckets.copy()
            buckets.insert(0, "feature", feature)
            buckets.insert(1, "direction", direction)
            bucket_parts.append(buckets)

    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values(
            ["top_minus_bottom_sharpe", "ic_mean", "top_minus_spy_sharpe"],
            ascending=[False, False, False],
        )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "rank_feature_sweep.csv"
    daily_path = out_dir / "rank_feature_sweep_daily.csv"
    monthly_path = out_dir / "rank_feature_sweep_by_month.csv"
    bucket_path = out_dir / "rank_feature_sweep_buckets.csv"

    summary.to_csv(summary_path, index=False)
    pd.concat(daily_parts, ignore_index=True).to_csv(daily_path, index=False)
    pd.concat(monthly_parts, ignore_index=True).to_csv(monthly_path, index=False)
    pd.concat(bucket_parts, ignore_index=True).to_csv(bucket_path, index=False)

    print(f"Features swept: {len(feats)} ({len(summary)} directional signals)")
    print(f"Execution: {args.execution_price} | return_col={return_col} | cost_bps={args.cost_bps}")
    if args.execution_price in {"next_open_to_close_3d", "next_open_to_close_5d"}:
        print("Note: 3d/5d ranking returns overlap; use them as signal diagnostics, not final portfolio stats.")
    print(f"Ranked universe: {'includes SPY' if args.include_spy else 'excludes SPY'}")
    print("\nTop 12 by top-minus-bottom Sharpe:")
    cols = [
        "feature",
        "direction",
        "top_minus_bottom_sharpe",
        "top_minus_bottom_total_return",
        "top_minus_spy_sharpe",
        "ic_mean",
        "ic_positive_rate",
        "top_beats_bottom_rate",
    ]
    if len(summary):
        print(summary[cols].head(12).to_string(index=False))
    print(f"\nSaved {summary_path}, {daily_path}, {monthly_path}, {bucket_path}")


if __name__ == "__main__":
    main()
