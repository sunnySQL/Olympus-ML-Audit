#!/usr/bin/env python3
"""
Rigorous out-of-sample evaluation: bootstrap confidence intervals, regime analysis,
and paper-trading readiness checks.

Usage:
  python scripts/evaluate_robustness.py
  python scripts/evaluate_robustness.py --target excess --folds 4
  python scripts/evaluate_robustness.py --n-bootstrap 2000
"""
from __future__ import annotations

import argparse
import importlib.util
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
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from utils.expanding_splits import expanding_splits
from utils.predict_bundle import add_pred_prob


def _load_train_module():
    path = ROOT / "scripts" / "train_model.py"
    spec = importlib.util.spec_from_file_location("_rob_tm", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Returns (point_estimate, lower, upper) for a metric."""
    rng = np.random.default_rng(seed)
    point = float(metric_fn(y_true, y_prob))
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            s = float(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            continue
        scores.append(s)
    if not scores:
        return point, point, point
    alpha = (1 - ci) / 2
    lo = float(np.percentile(scores, 100 * alpha))
    hi = float(np.percentile(scores, 100 * (1 - alpha)))
    return point, lo, hi


def regime_analysis(
    scored_df: pd.DataFrame,
    y_col: str,
) -> pd.DataFrame:
    """Split scored results by market regime (high/low SPY volatility, up/down SPY)."""
    rows = []
    df = scored_df.copy()

    if "spy_vol_10d" in df.columns:
        med_vol = df["spy_vol_10d"].median()
        for label, mask in [
            ("high_vol", df["spy_vol_10d"] >= med_vol),
            ("low_vol", df["spy_vol_10d"] < med_vol),
        ]:
            sub = df[mask]
            if len(sub) < 20 or sub[y_col].nunique() < 2:
                continue
            y = sub[y_col].values
            p = sub["pred_prob"].values
            try:
                auc = float(roc_auc_score(y, p))
            except Exception:
                auc = float("nan")
            rows.append({
                "regime": label,
                "n_rows": len(sub),
                "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
                "roc_auc": auc,
            })

    if "spy_return_1d" in df.columns:
        for label, mask in [
            ("spy_up", df["spy_return_1d"] >= 0),
            ("spy_down", df["spy_return_1d"] < 0),
        ]:
            sub = df[mask]
            if len(sub) < 20 or sub[y_col].nunique() < 2:
                continue
            y = sub[y_col].values
            p = sub["pred_prob"].values
            try:
                auc = float(roc_auc_score(y, p))
            except Exception:
                auc = float("nan")
            rows.append({
                "regime": label,
                "n_rows": len(sub),
                "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
                "roc_auc": auc,
            })

    df["_month"] = pd.to_datetime(df["Date"]).dt.to_period("M")
    for period, sub in df.groupby("_month"):
        if len(sub) < 10 or sub[y_col].nunique() < 2:
            continue
        y = sub[y_col].values
        p = sub["pred_prob"].values
        try:
            auc = float(roc_auc_score(y, p))
        except Exception:
            auc = float("nan")
        rows.append({
            "regime": f"month_{period}",
            "n_rows": len(sub),
            "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
            "roc_auc": auc,
        })

    return pd.DataFrame(rows)


def main() -> None:
    tm = _load_train_module()
    p = argparse.ArgumentParser(description="Rigorous OOS evaluation with CIs and regime analysis")
    p.add_argument("--features", type=str, default=str(ROOT / "data" / "features.csv"))
    p.add_argument("--target", choices=list(tm.TARGET_CHOICES.keys()), default="next_intraday")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--min-train-days", type=int, default=400)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument(
        "--max-zero-frac",
        type=float,
        default=0.995,
        help="Same sparse-feature prune as train_model.py (1.0 disables)",
    )
    p.add_argument("--out", type=str, default=str(ROOT / "reports" / "robustness_report.csv"))
    args = p.parse_args()

    df = pd.read_csv(args.features, parse_dates=["Date"])
    available = [c for c in tm.FEATURE_COLUMNS if c in df.columns]
    y_col = tm.TARGET_CHOICES[args.target]
    if y_col not in df.columns:
        raise SystemExit(f"Missing target column {y_col}; run scripts/build_features.py first.")
    df = df.dropna(subset=available + [y_col, "ticker"])
    df = df.sort_values(["Date", "ticker"])

    dates_u = np.sort(df["Date"].unique())
    splits = expanding_splits(dates_u, args.min_train_days, args.folds)

    all_scored = []
    fold_rows = []

    print(f"Robustness evaluation: {len(splits)} folds, target={args.target}\n")

    for fold_idx, (train_dates, test_dates) in enumerate(splits, start=1):
        train_df = df[df["Date"].isin(train_dates)]
        test_df = df[df["Date"].isin(test_dates)]
        if len(test_df) == 0 or test_df[y_col].nunique() < 2:
            continue

        df_f0, _, _ = tm.train_triple_split(train_df)
        cols = tm.variance_prune(available, df_f0[available])
        cols = tm.sparsity_prune(cols, df_f0[cols], max_zero_frac=args.max_zero_frac)
        if not cols:
            continue

        global_m, g_cal = tm.train_one_stack(
            train_df, cols, True, 450, True, 252.0,
            y_col, "classification", False, None, 45,
        )
        bundle = {
            "model": global_m,
            "model_kind": "global",
            "models_by_ticker": None,
            "feature_names": cols,
            "feature_set_version": "robustness",
            "task": "classification",
            "target_column": y_col,
            "return_prob_scale": 30.0,
        }

        scored = add_pred_prob(test_df, bundle)
        scored["pred_prob"] = scored["pred_prob"].astype(float)
        all_scored.append(scored)

        y = test_df[y_col].values
        prob = scored["pred_prob"].values

        auc_pt, auc_lo, auc_hi = bootstrap_ci(y, prob, roc_auc_score, args.n_bootstrap)
        acc = float(accuracy_score(y, (prob >= 0.5).astype(int)))

        def brier_fn(yt, yp):
            return brier_score_loss(yt, yp)
        brier_pt, brier_lo, brier_hi = bootstrap_ci(y, prob, brier_fn, args.n_bootstrap)

        fold_rows.append({
            "fold": fold_idx,
            "test_start": str(pd.Timestamp(test_dates[0]).date()),
            "test_end": str(pd.Timestamp(test_dates[-1]).date()),
            "n_test": len(test_df),
            "accuracy": acc,
            "roc_auc": auc_pt,
            "roc_auc_ci_lo": auc_lo,
            "roc_auc_ci_hi": auc_hi,
            "brier": brier_pt,
            "brier_ci_lo": brier_lo,
            "brier_ci_hi": brier_hi,
        })

        print(
            f"Fold {fold_idx}  {fold_rows[-1]['test_start']} .. {fold_rows[-1]['test_end']}  "
            f"n={len(test_df)}  AUC={auc_pt:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]  "
            f"Acc={acc:.4f}  Brier={brier_pt:.4f}"
        )

    # Overall (pooled)
    if all_scored:
        pooled = pd.concat(all_scored, ignore_index=True)
        y_all = pooled[y_col].values
        p_all = pooled["pred_prob"].values

        auc_pt, auc_lo, auc_hi = bootstrap_ci(y_all, p_all, roc_auc_score, args.n_bootstrap)
        acc_all = float(accuracy_score(y_all, (p_all >= 0.5).astype(int)))

        print(f"\nPooled OOS:  n={len(pooled)}  AUC={auc_pt:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]  Acc={acc_all:.4f}")

        # Regime analysis
        print("\n--- Regime Analysis ---")
        regime_df = regime_analysis(pooled, y_col)
        if len(regime_df):
            for _, r in regime_df.iterrows():
                if not str(r["regime"]).startswith("month_"):
                    print(f"  {r['regime']}: n={r['n_rows']}  acc={r['accuracy']:.4f}  auc={r['roc_auc']:.4f}")

            regime_path = Path(args.out).parent / "regime_analysis.csv"
            regime_df.to_csv(regime_path, index=False)
            print(f"\nSaved regime details to {regime_path}")

    # Save fold results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out_path, index=False)
    print(f"Saved robustness report to {out_path}")

    # Paper trading readiness check
    print("\n--- Paper Trading Readiness ---")
    if all_scored:
        mean_auc = np.mean([r["roc_auc"] for r in fold_rows])
        auc_stable = all(r["roc_auc_ci_lo"] > 0.52 for r in fold_rows)
        enough_folds = len(fold_rows) >= 3
        print(f"  Mean OOS AUC: {mean_auc:.4f}")
        print(f"  All fold CIs above 0.52: {'YES' if auc_stable else 'NO'}")
        print(f"  At least 3 folds: {'YES' if enough_folds else 'NO'}")
        if auc_stable and enough_folds and mean_auc > 0.55:
            print("  RECOMMENDATION: Model shows stable OOS edge. Consider paper trading.")
        else:
            print("  RECOMMENDATION: Continue improving before paper trading.")


if __name__ == "__main__":
    main()
