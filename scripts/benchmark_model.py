#!/usr/bin/env python3
"""
Holdout benchmark across classification targets (same split as train_model.py).
Writes reports/model_target_benchmark.csv and prints a summary table.

Usage (from project root):
  python scripts/benchmark_model.py
  python scripts/benchmark_model.py --per-ticker --light
  python scripts/benchmark_model.py --features data/features.csv --out reports/my_benchmark.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from utils.predict_bundle import add_pred_prob

import importlib.util


def _load_train_module():
    path = ROOT / "scripts" / "train_model.py"
    spec = importlib.util.spec_from_file_location("_bm_train_model", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    tm = _load_train_module()
    p = argparse.ArgumentParser(description="Benchmark classification targets on chronological holdout")
    p.add_argument("--features", type=str, default=str(ROOT / "data" / "features.csv"))
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--per-ticker", action="store_true", help="Train per-ticker stack (slower, prod-like)")
    p.add_argument("--light", action="store_true", help="Smaller XGBoost (train_model --light)")
    p.add_argument("--robust", action="store_true", help="train_model.py --robust preset")
    p.add_argument(
        "--max-zero-frac",
        type=float,
        default=0.995,
        help="Same sparse-feature prune as train_model.py (1.0 disables)",
    )
    p.add_argument("--out", type=str, default=str(ROOT / "reports" / "model_target_benchmark.csv"))
    p.add_argument("--save-best", type=str, default="", help="Optional joblib path to save bundle for best row by roc_auc")
    args = p.parse_args()

    df = pd.read_csv(args.features, parse_dates=["Date"])
    available = [c for c in tm.FEATURE_COLUMNS if c in df.columns]

    xgb_extra: dict = {}
    if args.robust:
        xgb_extra.update(
            {
                "learning_rate": 0.02,
                "min_child_weight": 10.0,
                "reg_lambda": 4.0,
                "reg_alpha": 0.2,
                "gamma": 0.25,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.75,
                "n_estimators": 400 if args.light else 700,
                "max_depth": 3,
            }
        )

    rows = []
    bundles_by_target: dict[str, dict] = {}

    for target_key in tm.TARGET_CHOICES:
        y_col = tm.TARGET_CHOICES[target_key]
        if y_col not in df.columns:
            print(f"Skipping {target_key}: missing {y_col} (run build_features.py)")
            continue
        need = available + ["ticker", y_col]
        d = df.dropna(subset=[c for c in need if c in df.columns])
        train_df, test_df, cutoff = tm.time_based_split(d, test_frac=args.test_frac)
        df_f0, _, _ = tm.train_triple_split(train_df)
        cols = tm.variance_prune(available, df_f0[available])
        cols = tm.sparsity_prune(cols, df_f0[cols], max_zero_frac=args.max_zero_frac)
        if not cols:
            continue

        cal_on = True
        rec_on = True
        task = "classification"
        es_rounds = 45
        spw_override = None
        max_spw = None

        if args.per_ticker:
            global_m, mbt, g_cal = tm.train_per_ticker(
                train_df,
                cols,
                min_rows=380,
                calibrate=cal_on,
                cal_min_rows_global=450,
                cal_min_rows_ticker=180,
                use_recency=rec_on,
                recency_hl=252.0,
                y_col=y_col,
                task=task,
                light=args.light,
                xgb_extra=xgb_extra if xgb_extra else None,
                early_stopping_rounds=es_rounds,
                scale_pos_weight_override=spw_override,
                max_scale_pos_weight=max_spw,
            )
            bundle = {
                "model": global_m,
                "model_kind": "per_ticker",
                "models_by_ticker": mbt,
                "feature_names": cols,
                "feature_set_version": str(d["feature_set_version"].iloc[0])
                if "feature_set_version" in d.columns
                else "v7",
                "train_cutoff_date": str(cutoff.date()),
                "task": task,
                "target_column": y_col,
                "return_prob_scale": 30.0,
                "calibrated_global": g_cal,
                "max_zero_frac": args.max_zero_frac,
            }
        else:
            global_m, g_cal = tm.train_one_stack(
                train_df,
                cols,
                cal_on,
                450,
                rec_on,
                252.0,
                y_col,
                task,
                args.light,
                xgb_extra if xgb_extra else None,
                es_rounds,
                spw_override,
                max_spw,
            )
            bundle = {
                "model": global_m,
                "model_kind": "global",
                "models_by_ticker": None,
                "feature_names": cols,
                "feature_set_version": str(d["feature_set_version"].iloc[0])
                if "feature_set_version" in d.columns
                else "v7",
                "train_cutoff_date": str(cutoff.date()),
                "task": task,
                "target_column": y_col,
                "return_prob_scale": 30.0,
                "calibrated_global": g_cal,
                "max_zero_frac": args.max_zero_frac,
            }

        scored = add_pred_prob(test_df, bundle)
        y_true = test_df[y_col].values
        prob = scored["pred_prob"].values
        pred = (prob >= 0.5).astype(int)
        maj = float(max((y_true == 0).mean(), (y_true == 1).mean()))
        try:
            auc = float(roc_auc_score(y_true, prob))
        except Exception:
            auc = float("nan")

        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target_key": target_key,
            "y_column": y_col,
            "model_kind": bundle["model_kind"],
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": len(cols),
            "train_cutoff": str(cutoff.date()),
            "accuracy_0p5": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "roc_auc": auc,
            "majority_baseline_acc": maj,
            "above_baseline": float(accuracy_score(y_true, pred)) - maj,
        }
        rows.append(row)
        bundles_by_target[target_key] = bundle

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path}\n")
    if len(out):
        display = out[
            [
                "target_key",
                "model_kind",
                "accuracy_0p5",
                "balanced_accuracy",
                "roc_auc",
                "majority_baseline_acc",
                "above_baseline",
            ]
        ]
        print(display.to_string(index=False))

    if args.save_best and len(out) and out["roc_auc"].notna().any():
        best_key = out.sort_values("roc_auc", ascending=False).iloc[0]["target_key"]
        best_key = str(best_key)
        joblib.dump(bundles_by_target[best_key], Path(args.save_best).resolve())
        print(f"\nSaved best-by-AUC bundle ({best_key}) → {args.save_best}")


if __name__ == "__main__":
    main()
