#!/usr/bin/env python3
"""
Build a v6-equivalent feature table (drop v7-only columns) from current features.csv,
then train global models for v6-slice vs full v7 with identical settings and print metrics.

Usage (from project root):
  python scripts/compare_v6_v7.py
  python scripts/compare_v6_v7.py --per-ticker
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.metrics_parse import parse_train_stdout  # noqa: E402

DATA = ROOT / "data" / "features.csv"
V6_SLICE = ROOT / "data" / "features_v6_compare.csv"
V6_MODEL = ROOT / "models" / "direction_model_v6_ab.pkl"
V7_MODEL = ROOT / "models" / "direction_model_v7_ab.pkl"

# v7-only columns (see build_features.py FEATURE_SET_VERSION v7)
V7_EXTRA_COLS = ["vol_ratio_5_10", "macd_line", "hl_range_mean_5d", "return_1d_lag1"]


def build_v6_slice():
    if not DATA.exists():
        raise SystemExit(f"Missing {DATA}; run build_features.py first.")
    df = pd.read_csv(DATA, parse_dates=["Date"])
    for c in V7_EXTRA_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    if "feature_set_version" in df.columns:
        df["feature_set_version"] = "v6"
    df.to_csv(V6_SLICE, index=False)
    print(f"Wrote {V6_SLICE} ({len(df)} rows) — dropped v7 extras: {V7_EXTRA_COLS}")


def run_train(features_csv: Path, model_path: Path, extra: list[str]) -> str:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_model.py"),
        "--features-csv",
        str(features_csv),
        "--model-path",
        str(model_path),
        *extra,
    ]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        print(out)
        raise SystemExit(f"train_model failed with code {r.returncode}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare v6 vs v7 feature sets (same split).")
    ap.add_argument(
        "--per-ticker",
        action="store_true",
        help="Train per-ticker stack (default: global only for a clean A/B)",
    )
    args = ap.parse_args()

    build_v6_slice()

    train_extra: list[str] = [] if args.per_ticker else ["--no-per-ticker"]

    print("\n=== Training v6-slice (without v7 columns) ===\n")
    out_v6 = run_train(V6_SLICE, V6_MODEL, train_extra)

    print("\n=== Training v7 (full) ===\n")
    out_v7 = run_train(DATA, V7_MODEL, train_extra)

    print(out_v6)
    print(out_v7)

    a = parse_train_stdout(out_v6)
    b = parse_train_stdout(out_v7)

    print("\n" + "=" * 60)
    print("SUMMARY (holdout test set, same chronological split & hyperparameters)")
    print("=" * 60)
    print(f"{'Metric':<14} {'v6-slice':>14} {'v7':>14} {'delta (v7-v6)':>18}")
    print("-" * 60)
    for k, label in [
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC AUC"),
        ("log_loss", "Log loss"),
        ("brier", "Brier"),
        ("n_features", "Feat count"),
    ]:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            print(f"{label:<14} {str(va):>14} {str(vb):>14}")
            continue
        if k == "n_features":
            print(f"{label:<14} {int(va):>14} {int(vb):>14} {int(vb - va):>18}")
        else:
            d = vb - va
            print(f"{label:<14} {va:>14.6f} {vb:>14.6f} {d:>+18.6f}")
    print("=" * 60)
    print(f"v6-slice model: {V6_MODEL}")
    print(f"v7 model: {V7_MODEL}")


if __name__ == "__main__":
    main()
