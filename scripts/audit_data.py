#!/usr/bin/env python3
"""Quick health check for features.csv: shapes, targets, news usage, missing data."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Audit features.csv")
    p.add_argument("--features", type=str, default=str(ROOT / "data" / "features.csv"))
    args = p.parse_args()

    path = Path(args.features)
    if not path.exists():
        print(f"Missing {path}")
        return

    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"File: {path}")
    print(f"Rows: {len(df):,}  Unique dates: {df['Date'].nunique()}  Tickers: {df['ticker'].nunique()}")

    if "feature_set_version" in df.columns:
        print(f"feature_set_version: {df['feature_set_version'].iloc[0]}")

    for col in [
        "target_intraday_next_direction",
        "target_direction_next_open_to_close_3d",
        "target_direction_next_open_to_close_5d",
        "target_direction",
        "target_excess_up",
        "target_direction_5d",
        "target_return_next_open_to_close",
        "target_return_next_open_to_close_3d",
        "target_return_next_open_to_close_5d",
        "target_return_1d",
        "target_return_5d",
    ]:
        if col not in df.columns:
            print(f"  {col}: (missing — run build_features.py)")
            continue
        s = df[col]
        if col.startswith("target_direction") or col in {"target_intraday_next_direction", "target_excess_up"}:
            vc = s.value_counts(normalize=True)
            missing = float(s.isna().mean())
            print(f"  {col}: balance 0={vc.get(0, 0):.2%} 1={vc.get(1, 0):.2%}  missing={missing:.2%}")
        else:
            print(f"  {col}: mean={s.mean():.5f} std={s.std():.5f}")

    news_cols = [
        c
        for c in df.columns
        if any(x in c for x in ("news_", "sentiment_", "weighted_sentiment"))
    ]
    if news_cols:
        print("\nNews / sentiment columns (pct all-zero rows):")
        for c in sorted(news_cols)[:20]:
            z = float((df[c].fillna(0) == 0).mean())
            nz = 1.0 - z
            print(f"  {c}: {z:.1%} zero  ({nz:.2%} non-zero)")
        if len(news_cols) > 20:
            print(f"  ... +{len(news_cols) - 20} more")

    na = df.isna().sum()
    top = na[na > 0].sort_values(ascending=False).head(8)
    if len(top):
        print("\nTop NA counts:")
        print(top.to_string())


if __name__ == "__main__":
    main()
