#!/usr/bin/env python3
"""
Grid-search long_threshold (long-only portfolio) on the holdout window.
Writes reports/threshold_sweep.csv and prints best rows by Sharpe and total return.

Usage:
  python scripts/threshold_sweep.py
  python scripts/threshold_sweep.py --long-min 0.52 --long-max 0.62 --steps 11
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils.predict_bundle import add_pred_prob

DATA_PATH = ROOT / "data" / "features.csv"
MODEL_PATH = ROOT / "models" / "direction_model.pkl"


def _load_eval():
    p = ROOT / "scripts" / "evaluate_backtest.py"
    spec = importlib.util.spec_from_file_location("_ts_eval", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ev = _load_eval()
    p = argparse.ArgumentParser(description="Sweep long_threshold for long-only backtest")
    p.add_argument("--model", type=str, default=str(MODEL_PATH))
    p.add_argument("--features", type=str, default=str(DATA_PATH))
    p.add_argument("--long-min", type=float, default=0.50)
    p.add_argument("--long-max", type=float, default=0.65)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--short-threshold", type=float, default=0.45, help="Unused for long-only; kept for CSV clarity")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument(
        "--execution-price",
        choices=ev.EXECUTION_PRICE_CHOICES,
        default="next_open_to_close",
        help="Realized return used by portfolio metrics.",
    )
    p.add_argument("--out", type=str, default=str(ROOT / "reports" / "threshold_sweep.csv"))
    p.add_argument("--full-sample", action="store_true")
    args = p.parse_args()

    if not Path(args.features).exists() or not Path(args.model).exists():
        print("Need features.csv and direction_model.pkl")
        sys.exit(1)

    df = pd.read_csv(args.features, parse_dates=["Date"])
    bundle = joblib.load(args.model)
    feats = bundle["feature_names"]
    y_col = str(bundle.get("target_column") or "target_direction")
    if y_col not in df.columns:
        raise SystemExit(f"Missing target column {y_col}; run scripts/build_features.py first.")
    df = df.dropna(subset=feats + [y_col])
    df = df.sort_values(["Date", "ticker"])

    cutoff = bundle.get("train_cutoff_date")
    if cutoff and not args.full_sample:
        ct = pd.to_datetime(cutoff)
        df = df[df["Date"] >= ct].copy()
        print(f"Holdout from {ct.date()}  rows={len(df)}")

    df = add_pred_prob(df, bundle)
    df, return_col = ev.attach_execution_returns(df, args.execution_price)
    df = df.dropna(subset=[return_col])
    y = df[y_col].values
    prob = df["pred_prob"].values
    try:
        auc = float(roc_auc_score(y, prob))
    except Exception:
        auc = float("nan")

    long_grid = np.linspace(args.long_min, args.long_max, max(2, args.steps))
    rows = []
    for long_th in long_grid:
        sig = ev.attach_signals(df, long_th, args.short_threshold, long_only=True)
        bt = ev.apply_positions(
            sig,
            cost_bps=args.cost_bps,
            long_threshold=long_th,
            short_threshold=args.short_threshold,
            long_only=True,
            return_col=return_col,
        )
        daily = bt.groupby("Date", as_index=False)["position_return"].mean()["position_return"]
        pm = ev.compute_metrics(daily)
        long_m = bt["pred_signal"] == 1
        hit_long = float((bt.loc[long_m, return_col] > 0).mean()) if long_m.any() else float("nan")
        frac_long = float(long_m.mean())
        rows.append(
            {
                "long_threshold": float(long_th),
                "frac_positions_long": frac_long,
                "hit_rate_when_long": hit_long,
                "roc_auc_holdout": auc,
                "target_column": y_col,
                "execution_price": args.execution_price,
                **pm,
            }
        )

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Holdout ROC AUC (all rows): {auc:.4f}\n")
    if args.execution_price in {"next_open_to_close_3d", "next_open_to_close_5d"}:
        print("Note: 3d/5d returns overlap; use threshold results as signal diagnostics, not final portfolio stats.\n")
    print("Top 5 by Sharpe:")
    print(out.sort_values("sharpe", ascending=False).head(5).to_string(index=False))
    print("\nTop 5 by total_return:")
    print(out.sort_values("total_return", ascending=False).head(5).to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
