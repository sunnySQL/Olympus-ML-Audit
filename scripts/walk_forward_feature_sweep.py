#!/usr/bin/env python3
"""
Walk-forward raw feature selection and ranking evaluation.

Each fold:
  1. Sweep candidate raw features on past/training dates only.
  2. Select the top K feature+direction signals by a training metric.
  3. Evaluate those selected signals on the next unseen test block.
  4. Evaluate a simple ensemble that averages selected per-date rank scores.

This guards against choosing raw signals with hindsight.

Usage:
  python scripts/walk_forward_feature_sweep.py
  python scripts/walk_forward_feature_sweep.py --folds 3 --top-k 5
  python scripts/walk_forward_feature_sweep.py --metric ic_mean --min-train-metric 0
"""
from __future__ import annotations

import argparse
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
from sweep_rank_features import (
    candidate_features,
    evaluate_feature_signal,
    summarize_feature,
)
from utils.expanding_splits import expanding_splits

DATA_PATH = ROOT / "data" / "features.csv"
REPORT_DIR = ROOT / "reports"

DEFAULT_METRIC = "top_minus_bottom_sharpe"


def evaluate_candidates(
    df: pd.DataFrame,
    features: list[str],
    return_col: str,
    cost_bps: float,
    top_n: int | None,
    top_pct: float,
    include_spy: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in features:
        base = df.dropna(subset=[feature, return_col, "Date", "ticker"]).copy()
        if len(base) == 0 or base[feature].nunique(dropna=True) < 2:
            continue
        for direction in ["high", "low"]:
            daily, summary, buckets = evaluate_feature_signal(
                base,
                feature=feature,
                direction=direction,
                return_col=return_col,
                cost_bps=cost_bps,
                top_n=top_n,
                top_pct=top_pct,
                include_spy=include_spy,
            )
            if len(daily) == 0:
                continue
            rows.append(summarize_feature(feature, direction, daily, summary, buckets))
    return pd.DataFrame(rows)


def select_signals(
    train_summary: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
    top_k: int = 5,
    min_train_metric: float | None = None,
) -> pd.DataFrame:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if train_summary.empty:
        return train_summary.copy()
    if metric not in train_summary.columns:
        raise ValueError(f"missing selection metric: {metric}")

    s = train_summary.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric]).copy()
    if min_train_metric is not None:
        s = s[s[metric] >= min_train_metric].copy()
    if s.empty:
        return s
    s = s.sort_values([metric, "ic_mean", "top_minus_spy_sharpe"], ascending=[False, False, False])
    out = s.head(top_k).copy()
    out.insert(0, "selection_rank", np.arange(1, len(out) + 1))
    out["selection_metric"] = metric
    out["selection_metric_value"] = out[metric].astype(float)
    return out


def add_ensemble_score(df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rank_cols: list[str] = []
    for idx, row in selected.reset_index(drop=True).iterrows():
        feature = str(row["feature"])
        direction = str(row["direction"])
        if feature not in out.columns:
            continue
        score = out[feature].astype(float)
        if direction == "low":
            score = -score
        elif direction != "high":
            raise ValueError(f"unknown direction: {direction}")
        col = f"_rank_{idx}"
        out[col] = score.groupby(out["Date"]).rank(pct=True, method="first")
        rank_cols.append(col)
    if rank_cols:
        out["_ensemble_score"] = out[rank_cols].mean(axis=1, skipna=True)
    else:
        out["_ensemble_score"] = np.nan
    return out


def evaluate_selected_on_test(
    test_df: pd.DataFrame,
    selected: pd.DataFrame,
    return_col: str,
    cost_bps: float,
    top_n: int | None,
    top_pct: float,
    include_spy: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    daily_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []

    for _, sel in selected.iterrows():
        feature = str(sel["feature"])
        direction = str(sel["direction"])
        base = test_df.dropna(subset=[feature, return_col, "Date", "ticker"]).copy()
        if len(base) == 0 or base[feature].nunique(dropna=True) < 2:
            continue
        daily, summary, buckets = evaluate_feature_signal(
            base,
            feature=feature,
            direction=direction,
            return_col=return_col,
            cost_bps=cost_bps,
            top_n=top_n,
            top_pct=top_pct,
            include_spy=include_spy,
        )
        row = summarize_feature(feature, direction, daily, summary, buckets)
        row["signal_type"] = "selected_feature"
        row["selection_rank"] = int(sel["selection_rank"])
        row["train_metric_value"] = float(sel["selection_metric_value"])
        summary_rows.append(row)

        d = daily.copy()
        d.insert(0, "signal_type", "selected_feature")
        d.insert(1, "feature", feature)
        d.insert(2, "direction", direction)
        d.insert(3, "selection_rank", int(sel["selection_rank"]))
        daily_parts.append(d)

        m = monthly_stability(daily)
        m.insert(0, "signal_type", "selected_feature")
        m.insert(1, "feature", feature)
        m.insert(2, "direction", direction)
        m.insert(3, "selection_rank", int(sel["selection_rank"]))
        monthly_parts.append(m)

    if len(selected):
        ensemble_df = add_ensemble_score(test_df, selected)
        daily = build_daily_ranking_returns(
            ensemble_df,
            score_col="_ensemble_score",
            return_col=return_col,
            cost_bps=cost_bps,
            top_n=top_n,
            top_pct=top_pct,
            include_spy=include_spy,
        )
        if len(daily):
            summary = summarize_daily(daily)
            buckets = score_buckets(
                ensemble_df,
                score_col="_ensemble_score",
                return_col=return_col,
                include_spy=include_spy,
            )
            row = summarize_feature("ensemble_selected", "high", daily, summary, buckets)
            row["signal_type"] = "ensemble_selected"
            row["selection_rank"] = 0
            row["train_metric_value"] = float(selected["selection_metric_value"].mean())
            summary_rows.append(row)

            d = daily.copy()
            d.insert(0, "signal_type", "ensemble_selected")
            d.insert(1, "feature", "ensemble_selected")
            d.insert(2, "direction", "high")
            d.insert(3, "selection_rank", 0)
            daily_parts.append(d)

            m = monthly_stability(daily)
            m.insert(0, "signal_type", "ensemble_selected")
            m.insert(1, "feature", "ensemble_selected")
            m.insert(2, "direction", "high")
            m.insert(3, "selection_rank", 0)
            monthly_parts.append(m)

    return (
        pd.DataFrame(summary_rows),
        pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame(),
        pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame(),
    )


def _tag_fold(df: pd.DataFrame, fold_meta: dict) -> pd.DataFrame:
    out = df.copy()
    for key, val in reversed(list(fold_meta.items())):
        out.insert(0, key, val)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward raw feature selection/evaluation")
    p.add_argument("--features", type=str, default=str(DATA_PATH))
    p.add_argument(
        "--execution-price",
        choices=EXECUTION_PRICE_CHOICES,
        default="next_open_to_close",
    )
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--min-train-days", type=int, default=400)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--top-n", type=int, default=2)
    p.add_argument("--top-pct", type=float, default=0.33)
    p.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        help="Training metric used to select signals.",
    )
    p.add_argument(
        "--min-train-metric",
        type=float,
        default=None,
        help="Optional lower bound for selected training metric.",
    )
    p.add_argument(
        "--include-spy",
        action="store_true",
        help="Include SPY in ranked/traded universe. Default excludes SPY and uses it as benchmark.",
    )
    p.add_argument("--features-list", nargs="*", default=None)
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
    feats = candidate_features(df, args.features_list, max_zero_frac=args.max_zero_frac)
    dates_u = np.sort(df["Date"].dropna().unique())
    splits = expanding_splits(dates_u, args.min_train_days, args.folds)

    selected_parts: list[pd.DataFrame] = []
    result_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []

    print(
        f"Walk-forward feature sweep: folds={len(splits)}, features={len(feats)}, "
        f"metric={args.metric}, top_k={args.top_k}"
    )
    print(f"Execution: {args.execution_price} | return_col={return_col} | cost_bps={args.cost_bps}\n")
    if args.execution_price in {"next_open_to_close_3d", "next_open_to_close_5d"}:
        print("Note: 3d/5d ranking returns overlap; use them as signal diagnostics, not final portfolio stats.\n")

    for fold, (train_dates, test_dates) in enumerate(splits, start=1):
        train_df = df[df["Date"].isin(train_dates)].copy()
        test_df = df[df["Date"].isin(test_dates)].copy()
        fold_meta = {
            "fold": fold,
            "train_start": str(pd.Timestamp(train_dates[0]).date()),
            "train_end": str(pd.Timestamp(train_dates[-1]).date()),
            "test_start": str(pd.Timestamp(test_dates[0]).date()),
            "test_end": str(pd.Timestamp(test_dates[-1]).date()),
            "n_train_days": int(len(train_dates)),
            "n_test_days": int(len(test_dates)),
        }

        print(
            f"Fold {fold}: train {fold_meta['train_start']}..{fold_meta['train_end']} "
            f"test {fold_meta['test_start']}..{fold_meta['test_end']}"
        )

        train_summary = evaluate_candidates(
            train_df,
            feats,
            return_col=return_col,
            cost_bps=args.cost_bps,
            top_n=args.top_n,
            top_pct=args.top_pct,
            include_spy=args.include_spy,
        )
        selected = select_signals(
            train_summary,
            metric=args.metric,
            top_k=args.top_k,
            min_train_metric=args.min_train_metric,
        )
        if selected.empty:
            print("  No signals selected.")
            continue
        selected_parts.append(_tag_fold(selected, fold_meta))
        print(
            "  Selected: "
            + ", ".join(f"{r.feature}:{r.direction} ({getattr(r, args.metric):.3f})" for r in selected.itertuples())
        )

        test_summary, daily, monthly = evaluate_selected_on_test(
            test_df,
            selected,
            return_col=return_col,
            cost_bps=args.cost_bps,
            top_n=args.top_n,
            top_pct=args.top_pct,
            include_spy=args.include_spy,
        )
        if len(test_summary):
            result_parts.append(_tag_fold(test_summary, fold_meta))
            ens = test_summary[test_summary["signal_type"] == "ensemble_selected"]
            if len(ens):
                r = ens.iloc[0]
                print(
                    f"  Ensemble test: spread_sharpe={r['top_minus_bottom_sharpe']:.3f} "
                    f"spread_return={r['top_minus_bottom_total_return']:.3f} "
                    f"ic={r['ic_mean']:.4f}"
                )
        if len(daily):
            daily_parts.append(_tag_fold(daily, fold_meta))
        if len(monthly):
            monthly_parts.append(_tag_fold(monthly, fold_meta))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    selected_all = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    daily_all = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    monthly_all = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()

    result_path = out_dir / "wf_feature_sweep.csv"
    selected_path = out_dir / "wf_feature_sweep_selected.csv"
    daily_path = out_dir / "wf_feature_sweep_daily.csv"
    monthly_path = out_dir / "wf_feature_sweep_by_month.csv"
    results.to_csv(result_path, index=False)
    selected_all.to_csv(selected_path, index=False)
    daily_all.to_csv(daily_path, index=False)
    monthly_all.to_csv(monthly_path, index=False)

    print("\nSaved:")
    print(f"  {result_path}")
    print(f"  {selected_path}")
    print(f"  {daily_path}")
    print(f"  {monthly_path}")

    if len(results):
        print("\nTest summary by signal type:")
        show_cols = [
            "fold",
            "signal_type",
            "feature",
            "direction",
            "top_minus_bottom_sharpe",
            "top_minus_bottom_total_return",
            "top_minus_spy_sharpe",
            "ic_mean",
        ]
        print(results[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
