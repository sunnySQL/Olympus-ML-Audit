#!/usr/bin/env python3
"""
Score the latest complete row per ticker (same logic as the Olympus demo), print CSV,
and optionally append to reports/prediction_log.csv for live tracking vs outcomes.

Usage (from project root):
  python scripts/live_score.py
  python scripts/live_score.py --model models/direction_model.pkl --append-log
  python scripts/live_score.py --long-th 0.58 --short-th 0.42
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from utils.live_score import score_latest_per_ticker, signal_from_thresholds

DEFAULT_FEATURES = ROOT / "data" / "features.csv"
DEFAULT_MODEL = ROOT / "models" / "direction_model.pkl"
DEFAULT_LOG = ROOT / "reports" / "prediction_log.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="Score latest features per ticker (live batch)")
    p.add_argument("--features", type=str, default=str(DEFAULT_FEATURES))
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    p.add_argument("--long-th", type=float, default=0.55)
    p.add_argument("--short-th", type=float, default=0.45)
    p.add_argument(
        "--append-log",
        action="store_true",
        help=f"Append rows to {DEFAULT_LOG} (UTC timestamp per run)",
    )
    p.add_argument("--log-path", type=str, default=str(DEFAULT_LOG), help="Override prediction log path")
    args = p.parse_args()

    feat_path = Path(args.features).resolve()
    model_path = Path(args.model).resolve()
    if not feat_path.is_file():
        print(f"Missing features: {feat_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.is_file():
        print(f"Missing model: {model_path}", file=sys.stderr)
        sys.exit(1)

    bundle = joblib.load(model_path)
    df = pd.read_csv(feat_path, parse_dates=["Date"])
    feature_names = bundle.get("feature_names") or []
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        print(f"features.csv missing columns: {missing[:8]}", file=sys.stderr)
        sys.exit(1)

    scored = score_latest_per_ticker(df, bundle)
    logged_at = datetime.now(timezone.utc).isoformat()

    rows = []
    for _, r in scored.iterrows():
        sig = signal_from_thresholds(float(r["pred_prob"]), args.long_th, args.short_th)
        row = {
            "logged_at_utc": logged_at,
            "as_of_date": r["Date"].strftime("%Y-%m-%d") if pd.notna(r["Date"]) else "",
            "ticker": str(r["ticker"]),
            "pred_prob": float(r["pred_prob"]),
            "signal": sig,
            "long_th": args.long_th,
            "short_th": args.short_th,
            "model_path": str(model_path),
            "feature_set_version": str(bundle.get("feature_set_version", "")),
            "train_cutoff_date": str(bundle.get("train_cutoff_date", "")),
            "task": str(bundle.get("task", "classification")),
            "target_column": str(bundle.get("target_column", "")),
        }
        if "pred_return" in r.index and pd.notna(r.get("pred_return")):
            row["pred_return"] = float(r["pred_return"])
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("ticker")
    print(out.to_csv(index=False), end="")

    if args.append_log:
        log_path = Path(args.log_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Empty pre-created file must still get a header row
        need_header = (not log_path.exists()) or log_path.stat().st_size == 0
        out.to_csv(log_path, mode="a", header=need_header, index=False)
        print(f"\n# Appended {len(out)} rows to {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
