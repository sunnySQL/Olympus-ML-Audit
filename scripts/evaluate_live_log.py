#!/usr/bin/env python3
"""
Join prediction_log.csv to features.csv on (ticker, as_of_date) and measure how well
logged probabilities matched the model's realized target.

Run after you have appended scores over time and features.csv includes those dates.

Usage:
  python scripts/evaluate_live_log.py
  python scripts/evaluate_live_log.py --log reports/prediction_log.csv --features data/features.csv
  python scripts/evaluate_live_log.py --target-column target_intraday_next_direction
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate logged live predictions vs realized labels")
    ap.add_argument("--log", type=str, default=str(ROOT / "reports" / "prediction_log.csv"))
    ap.add_argument("--features", type=str, default=str(ROOT / "data" / "features.csv"))
    ap.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Override realized label column. Defaults to log target_column, then target_intraday_next_direction.",
    )
    args = ap.parse_args()

    log_path = Path(args.log)
    feat_path = Path(args.features)
    if not log_path.is_file():
        print(f"No prediction log at {log_path}. Run live_score.py --append-log first.", file=sys.stderr)
        sys.exit(1)
    if not feat_path.is_file():
        print(f"No features at {feat_path}", file=sys.stderr)
        sys.exit(1)

    log_df = pd.read_csv(log_path)
    if "as_of_date" not in log_df.columns:
        print("Log must contain as_of_date", file=sys.stderr)
        sys.exit(1)
    if "logged_at_utc" in log_df.columns:
        log_df["logged_at_utc"] = pd.to_datetime(log_df["logged_at_utc"], utc=True, errors="coerce")

    log_df["as_of_date"] = pd.to_datetime(log_df["as_of_date"]).dt.normalize()

    # Latest entry per (ticker, as_of_date) if re-scored same day
    if "logged_at_utc" in log_df.columns:
        log_df = log_df.sort_values("logged_at_utc").drop_duplicates(subset=["ticker", "as_of_date"], keep="last")
    else:
        log_df = log_df.drop_duplicates(subset=["ticker", "as_of_date"], keep="last")

    feat = pd.read_csv(feat_path, parse_dates=["Date"])
    target_col = args.target_column
    if target_col is None and "target_column" in log_df.columns:
        vals = [v for v in log_df["target_column"].dropna().astype(str).unique() if v]
        if vals:
            target_col = vals[-1]
    if target_col is None:
        target_col = "target_intraday_next_direction" if "target_intraday_next_direction" in feat.columns else "target_direction"

    if target_col not in feat.columns:
        print(f"features.csv must contain {target_col}", file=sys.stderr)
        sys.exit(1)

    feat = feat.copy()
    feat["as_of_date"] = pd.to_datetime(feat["Date"]).dt.normalize()

    m = log_df.merge(
        feat[["ticker", "as_of_date", target_col]],
        on=["ticker", "as_of_date"],
        how="inner",
    ).dropna(subset=[target_col])
    if len(m) == 0:
        print("No overlapping labeled (ticker, as_of_date) between log and features.", file=sys.stderr)
        sys.exit(1)

    y = m[target_col].astype(int).values
    p = m["pred_prob"].astype(float).clip(1e-6, 1 - 1e-6).values
    y_hat = (p >= 0.5).astype(int)

    print(f"Rows evaluated: {len(m)}  (unique dates: {m['as_of_date'].nunique()})")
    print(f"Target column: {target_col}")
    print(f"Accuracy (p>=0.5): {accuracy_score(y, y_hat):.4f}")
    try:
        print(f"ROC AUC: {roc_auc_score(y, p):.4f}")
    except Exception as e:
        print(f"ROC AUC: n/a ({e})")
    try:
        print(f"Brier: {brier_score_loss(y, p):.4f}")
    except Exception as e:
        print(f"Brier: n/a ({e})")

    if "signal" in m.columns and "long_th" in m.columns:
        long_th = float(m["long_th"].median())
        sub = m[m["signal"] == "long"]
        if len(sub) > 0:
            hit_long = sub[target_col].mean()
            print(f"Mean({target_col} | signal=long): {hit_long:.4f} (median long_th in log: {long_th:.2f})")


if __name__ == "__main__":
    main()
