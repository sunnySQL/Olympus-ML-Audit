#!/usr/bin/env python3
"""
Optuna-based Bayesian hyperparameter search with walk-forward (expanding-window) cross-validation.

Objective: mean ROC AUC across walk-forward folds — avoids overfitting to a single holdout.

Usage:
  python scripts/tune_model.py
  python scripts/tune_model.py --target excess --n-trials 150 --folds 4
  python scripts/tune_model.py --target direction --out reports/tuned_params.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from utils.expanding_splits import expanding_splits
from utils.predict_bundle import add_pred_prob


def _load_train_module():
    path = ROOT / "scripts" / "train_model.py"
    spec = importlib.util.spec_from_file_location("_tune_tm", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    available: list[str],
    y_col: str,
    n_folds: int,
    min_train_days: int,
    max_zero_frac: float,
    tm,
) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=50),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 0.9),
        "min_child_weight": trial.suggest_float("min_child_weight", 3, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
    }

    dates_u = np.sort(df["Date"].unique())
    splits = expanding_splits(dates_u, min_train_days, n_folds)

    aucs = []
    for train_dates, test_dates in splits:
        train_df = df[df["Date"].isin(train_dates)]
        test_df = df[df["Date"].isin(test_dates)]
        if len(test_df) == 0 or test_df[y_col].nunique() < 2:
            continue

        df_f0, _, _ = tm.train_triple_split(train_df)
        cols = tm.variance_prune(available, df_f0[available])
        cols = tm.sparsity_prune(cols, df_f0[cols], max_zero_frac=max_zero_frac)
        if not cols:
            continue

        global_m, g_cal = tm.train_one_stack(
            train_df,
            cols,
            calibrate=True,
            cal_min_rows=450,
            use_recency=True,
            recency_hl=252.0,
            y_col=y_col,
            task="classification",
            light=False,
            xgb_extra=params,
            early_stopping_rounds=45,
            scale_pos_weight_override=None,
            max_scale_pos_weight=None,
        )
        bundle = {
            "model": global_m,
            "model_kind": "global",
            "models_by_ticker": None,
            "feature_names": cols,
            "feature_set_version": "tune",
            "task": "classification",
            "return_prob_scale": 30.0,
        }
        scored = add_pred_prob(test_df, bundle)
        prob = scored["pred_prob"].values
        y = test_df[y_col].values
        try:
            auc = float(roc_auc_score(y, prob))
        except Exception:
            auc = 0.5
        aucs.append(auc)

        trial.report(np.mean(aucs), len(aucs) - 1)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(aucs)) if aucs else 0.5


def main() -> None:
    tm = _load_train_module()
    p = argparse.ArgumentParser(description="Optuna walk-forward hyperparameter search")
    p.add_argument("--features", type=str, default=str(ROOT / "data" / "features.csv"))
    p.add_argument("--target", choices=list(tm.TARGET_CHOICES.keys()), default="excess")
    p.add_argument("--n-trials", type=int, default=120)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--min-train-days", type=int, default=400)
    p.add_argument(
        "--max-zero-frac",
        type=float,
        default=0.995,
        help="Same sparse-feature prune as train_model.py (1.0 disables)",
    )
    p.add_argument("--out", type=str, default=str(ROOT / "reports" / "tuned_params.json"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.features, parse_dates=["Date"])
    available = [c for c in tm.FEATURE_COLUMNS if c in df.columns]
    y_col = tm.TARGET_CHOICES[args.target]
    df = df.dropna(subset=available + [y_col, "ticker", "target_return_1d"])
    df = df.sort_values(["Date", "ticker"])

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"olympus_{args.target}",
    )

    study.optimize(
        lambda trial: objective(
            trial, df, available, y_col, args.folds, args.min_train_days, args.max_zero_frac, tm
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}: mean walk-forward AUC = {best.value:.4f}")
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "target": args.target,
        "y_column": y_col,
        "mean_wf_auc": best.value,
        "n_trials": args.n_trials,
        "n_folds": args.folds,
        "params": best.params,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved best params to {out_path}")


if __name__ == "__main__":
    main()
