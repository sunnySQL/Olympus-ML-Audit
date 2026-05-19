#!/usr/bin/env python3
"""
Grid-search a probability threshold on a single time-based holdout split.

Use this only as an exploratory sanity check. Choosing the threshold on the
same split you report inflates metrics; prefer walk-forward evaluation or a
fresh holdout when fixing a production rule.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_model import FEATURE_COLUMNS, TARGET_CHOICES, time_based_split
from utils.predict_bundle import add_pred_prob


def main() -> None:
    p = argparse.ArgumentParser(description="Grid-search classifier threshold on holdout (exploratory)")
    p.add_argument("--features-csv", type=str, default=str(ROOT / "data" / "features.csv"))
    p.add_argument("--model-path", type=str, default=str(ROOT / "models" / "direction_model.pkl"))
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--lo", type=float, default=0.45)
    p.add_argument("--hi", type=float, default=0.65)
    p.add_argument("--steps", type=int, default=41)
    p.add_argument(
        "--target",
        type=str,
        default=None,
        choices=list(TARGET_CHOICES.keys()),
        help="Override target label; defaults to the model bundle target_column",
    )
    args = p.parse_args()

    import joblib

    bundle = joblib.load(args.model_path)
    if args.target is None:
        y_col = str(bundle.get("target_column") or TARGET_CHOICES["next_intraday"])
    else:
        y_col = TARGET_CHOICES[args.target]
    df = pd.read_csv(args.features_csv, parse_dates=["Date"])
    fnames = bundle.get("feature_names") or [c for c in FEATURE_COLUMNS if c in df.columns]
    need = fnames + ["ticker", y_col]
    df = df.dropna(subset=[c for c in need if c in df.columns])
    train_df, test_df, cutoff = time_based_split(df, test_frac=args.test_frac)
    scored = add_pred_prob(test_df, bundle)
    y_true = test_df[y_col].values.astype(int)
    y_prob = scored["pred_prob"].values.astype(float)

    print("Model:", args.model_path)
    print("Target:", y_col, "| test rows:", len(test_df), "| split >=", cutoff.date())
    print("t      acc    bal_acc  f1     prec   rec")
    for t in np.linspace(args.lo, args.hi, max(2, args.steps)):
        pred = (y_prob >= t).astype(int)
        print(
            f"{t:.3f}  {accuracy_score(y_true, pred):.4f}  "
            f"{balanced_accuracy_score(y_true, pred):.4f}  "
            f"{f1_score(y_true, pred, zero_division=0):.4f}  "
            f"{precision_score(y_true, pred, zero_division=0):.4f}  "
            f"{recall_score(y_true, pred, zero_division=0):.4f}"
        )


if __name__ == "__main__":
    main()
