#!/usr/bin/env python3
"""
Build a v5-equivalent feature table (drop return_10d, vol_10d_over_20) from the current
features.csv, then train global models for v5 vs v6 with identical settings and print metrics.

Usage (from project root):
  python scripts/compare_v5_v6.py
  python scripts/compare_v5_v6.py --per-ticker
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
V5_PATH = ROOT / "data" / "features_v5_compare.csv"
V5_MODEL = ROOT / "models" / "direction_model_v5_compare.pkl"
V6_MODEL = ROOT / "models" / "direction_model_v6_compare.pkl"


def build_v5_slice():
    if not DATA.exists():
        raise SystemExit(f"Missing {DATA}; run build_features.py first.")
    df = pd.read_csv(DATA, parse_dates=["Date"])
    drop = ["return_10d", "vol_10d_over_20"]
    for c in drop:
        if c in df.columns:
            df = df.drop(columns=[c])
    if "feature_set_version" in df.columns:
        df["feature_set_version"] = "v5"
    df.to_csv(V5_PATH, index=False)
    print(f"Wrote {V5_PATH} ({len(df)} rows)")


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
    ap = argparse.ArgumentParser(description="Compare v5 vs v6 feature sets (same data, same split).")
    ap.add_argument(
        "--per-ticker",
        action="store_true",
        help="Train per-ticker stack (default: global only for a clean A/B)",
    )
    args = ap.parse_args()

    build_v5_slice()

    train_extra: list[str] = [] if args.per_ticker else ["--no-per-ticker"]

    print("\n=== Training v5 (without return_10d, vol_10d_over_20) ===\n")
    out_v5 = run_train(V5_PATH, V5_MODEL, train_extra)

    print("\n=== Training v6 (full) ===\n")
    out_v6 = run_train(DATA, V6_MODEL, train_extra)

    print(out_v5)
    print(out_v6)

    a = parse_train_stdout(out_v5)
    b = parse_train_stdout(out_v6)

    print("\n" + "=" * 60)
    print("SUMMARY (holdout test set, same chronological split & hyperparameters)")
    print("=" * 60)
    print(f"{'Metric':<14} {'v5':>14} {'v6':>14} {'delta (v6-v5)':>18}")
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
    print(f"v5 model: {V5_MODEL}")
    print(f"v6 model: {V6_MODEL}")
    print("\nInterpretation: higher ROC AUC / accuracy and lower log-loss / Brier on holdout is better.")
    print("Small differences are often noise; look for consistent gains across seeds or walk-forward.")


if __name__ == "__main__":
    main()
