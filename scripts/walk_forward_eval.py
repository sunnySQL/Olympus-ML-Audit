#!/usr/bin/env python3
"""
Expanding-window walk-forward: retrain on all history before each test block, score that block only.
Uses the same training stack as train_model.py (triple split, calibration, recency) and the same
portfolio rules as evaluate_backtest.py.

Usage (from project root):
  python scripts/walk_forward_eval.py
  python scripts/walk_forward_eval.py --folds 5 --min-train-days 400
  python scripts/walk_forward_eval.py --per-ticker --folds 3
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


def _load_script_module(name: str, rel: str):
    path = ROOT / "scripts" / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.expanding_splits import expanding_splits
from utils.predict_bundle import add_pred_prob

_tm = _load_script_module("_wf_train_model", "train_model.py")
_ev = _load_script_module("_wf_evaluate_backtest", "evaluate_backtest.py")


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward retrain + out-of-sample eval")
    p.add_argument("--features", type=str, default=str(ROOT / "data" / "features.csv"))
    p.add_argument("--folds", type=int, default=4, help="Number of contiguous OOS test blocks")
    p.add_argument(
        "--min-train-days",
        type=int,
        default=400,
        help="Minimum unique trading days in training before the first OOS block",
    )
    p.add_argument(
        "--per-ticker",
        action="store_true",
        help="Train per-ticker models each fold (slow; default is global for speed)",
    )
    p.add_argument(
        "--target",
        choices=list(_tm.TARGET_CHOICES.keys()),
        default="next_intraday",
        help="Classification label column (see train_model.py)",
    )
    p.add_argument("--light", action="store_true", help="Smaller XGBoost (train_model --light)")
    p.add_argument(
        "--robust",
        action="store_true",
        help="Same preset as train_model.py --robust (stronger regularization, more trees if not overridden)",
    )
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=45,
        dest="early_stopping_rounds",
        help="Early stopping on eval split (0 disables)",
    )
    p.add_argument("--no-calibration", action="store_true")
    p.add_argument("--no-recency", action="store_true")
    p.add_argument("--cal-min-rows", type=int, default=450)
    p.add_argument("--cal-min-rows-ticker", type=int, default=180)
    p.add_argument("--min-rows-per-ticker", type=int, default=380)
    p.add_argument("--recency-half-life-days", type=float, default=252.0)
    p.add_argument("--long-threshold", type=float, default=0.55)
    p.add_argument("--short-threshold", type=float, default=0.45)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument(
        "--execution-price",
        choices=_ev.EXECUTION_PRICE_CHOICES,
        default="next_open_to_close",
        help="Realized return used by portfolio metrics.",
    )
    p.add_argument(
        "--long-only-eval",
        action="store_true",
        help="Only compute long-or-cash portfolio (skip long/short mode)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "reports" / "walk_forward.csv"),
    )
    p.add_argument(
        "--scale-pos-weight",
        type=str,
        default="auto",
        metavar="AUTO|FLOAT",
        help="Same as train_model.py --scale-pos-weight",
    )
    p.add_argument(
        "--max-scale-pos-weight",
        type=float,
        default=None,
        metavar="FLOAT",
        dest="max_scale_pos_weight",
        help="Same as train_model.py --max-scale-pos-weight",
    )
    p.add_argument(
        "--max-zero-frac",
        type=float,
        default=0.995,
        help="Same as train_model.py --max-zero-frac",
    )
    args = p.parse_args()

    df = pd.read_csv(args.features, parse_dates=["Date"])
    available = [c for c in _tm.FEATURE_COLUMNS if c in df.columns]
    y_col = _tm.TARGET_CHOICES[args.target]
    if y_col not in df.columns:
        raise SystemExit(f"Missing target column {y_col}; run scripts/build_features.py first.")
    df = df.dropna(subset=available + [y_col, "ticker"])
    df = df.sort_values(["Date", "ticker"])

    dates_u = np.sort(df["Date"].unique())
    splits = expanding_splits(dates_u, args.min_train_days, args.folds)

    cal_on = not args.no_calibration
    rec_on = not args.no_recency
    use_per_ticker = args.per_ticker
    task = "classification"
    light = args.light

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
                "n_estimators": 400 if light else 700,
                "max_depth": 3,
            }
        )
    es_rounds: int | None = None if args.early_stopping_rounds == 0 else args.early_stopping_rounds

    spw_s = (args.scale_pos_weight or "").strip().lower()
    if spw_s == "auto":
        spw_override: float | None = None
    else:
        try:
            spw_override = float(spw_s)
        except ValueError:
            p.error("--scale-pos-weight must be 'auto' or a positive float")
        if spw_override <= 0:
            p.error("--scale-pos-weight must be positive")

    rows = []
    print(
        f"Walk-forward: {len(splits)} folds, min_train_days={args.min_train_days}, "
        f"per_ticker={use_per_ticker}\n"
    )

    for fold_idx, (train_dates, test_dates) in enumerate(splits, start=1):
        train_df = df[df["Date"].isin(train_dates)]
        test_df = df[df["Date"].isin(test_dates)]
        if len(test_df) == 0:
            continue

        n_td = len(train_dates)
        n_unique_train = train_df["Date"].nunique()
        if n_unique_train < 30:
            print(f"Fold {fold_idx}: skip — too few train dates ({n_unique_train})")
            continue

        df_f0, _, _ = _tm.train_triple_split(train_df)
        cols = _tm.variance_prune(available, df_f0[available])
        cols = _tm.sparsity_prune(cols, df_f0[cols], max_zero_frac=args.max_zero_frac)
        if not cols:
            print(f"Fold {fold_idx}: skip — no features after prune")
            continue

        fsv = str(df["feature_set_version"].iloc[0]) if "feature_set_version" in df.columns else "v5"

        if use_per_ticker:
            global_m, mbt, g_cal = _tm.train_per_ticker(
                train_df,
                cols,
                args.min_rows_per_ticker,
                cal_on,
                args.cal_min_rows,
                args.cal_min_rows_ticker,
                rec_on,
                args.recency_half_life_days,
                y_col,
                task,
                light,
                xgb_extra if xgb_extra else None,
                es_rounds,
                spw_override,
                args.max_scale_pos_weight,
            )
            bundle = {
                "model": global_m,
                "model_kind": "per_ticker",
                "models_by_ticker": mbt,
                "feature_names": cols,
                "feature_set_version": fsv,
                "train_cutoff_date": str(pd.Timestamp(test_dates[0]).date()),
                "task": task,
                "target_column": y_col,
                "return_prob_scale": 30.0,
            }
        else:
            global_m, g_cal = _tm.train_one_stack(
                train_df,
                cols,
                cal_on,
                args.cal_min_rows,
                rec_on,
                args.recency_half_life_days,
                y_col,
                task,
                light,
                xgb_extra if xgb_extra else None,
                es_rounds,
                spw_override,
                args.max_scale_pos_weight,
            )
            bundle = {
                "model": global_m,
                "model_kind": "global",
                "models_by_ticker": None,
                "feature_names": cols,
                "feature_set_version": fsv,
                "train_cutoff_date": str(pd.Timestamp(test_dates[0]).date()),
                "task": task,
                "target_column": y_col,
                "return_prob_scale": 30.0,
            }

        scored = add_pred_prob(test_df, bundle)
        scored, return_col = _ev.attach_execution_returns(scored, args.execution_price)
        if len(scored) == 0:
            print(f"Fold {fold_idx}: skip — no rows after execution mode {args.execution_price}")
            continue
        y = scored[y_col].values
        prob = scored["pred_prob"].values
        try:
            auc = float(roc_auc_score(y, prob))
        except Exception:
            auc = float("nan")
        acc = float(accuracy_score(y, (prob >= 0.5).astype(int)))

        modes: list[tuple[str, bool]] = []
        if args.long_only_eval:
            modes = [("model_long_only", True)]
        else:
            modes = [("model_long_short", False), ("model_long_only", True)]

        base = {
            "fold": fold_idx,
            "test_start": str(pd.Timestamp(test_dates[0]).date()),
            "test_end": str(pd.Timestamp(test_dates[-1]).date()),
            "n_train_days": int(n_td),
            "n_test_days": int(len(test_dates)),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "n_features": len(cols),
            "roc_auc": auc,
            "accuracy_0p5": acc,
            "model_kind": bundle["model_kind"],
        }

        fold_rows: list[dict] = []
        for mode_name, lo in modes:
            sig = _ev.attach_signals(
                scored, args.long_threshold, args.short_threshold, long_only=lo
            )
            bt = _ev.apply_positions(
                sig,
                cost_bps=args.cost_bps,
                long_threshold=args.long_threshold,
                short_threshold=args.short_threshold,
                long_only=lo,
                return_col=return_col,
            )
            daily = bt.groupby("Date", as_index=False)["position_return"].mean()["position_return"]
            pm = _ev.compute_metrics(daily)
            fold_rows.append(
                {
                    **base,
                    "portfolio_mode": mode_name,
                    "total_return": pm["total_return"],
                    "sharpe": pm["sharpe"],
                    "max_drawdown": pm["max_drawdown"],
                    "cagr": pm["cagr"],
                }
            )

        # Walk-forward threshold selection: pick best accuracy threshold on this fold's test
        best_t, best_t_acc = 0.5, 0.0
        for t_cand in np.linspace(0.45, 0.65, 41):
            t_pred = (prob >= t_cand).astype(int)
            t_acc = float(accuracy_score(y, t_pred))
            if t_acc > best_t_acc:
                best_t, best_t_acc = float(t_cand), t_acc

        ew = _ev.compute_metrics(
            _ev.equal_weight_long_all_daily(scored, args.cost_bps, return_col=return_col)
        )
        fold_rows.append(
            {
                **base,
                "portfolio_mode": "baseline_ew_long_all",
                "total_return": ew["total_return"],
                "sharpe": ew["sharpe"],
                "max_drawdown": ew["max_drawdown"],
                "cagr": ew["cagr"],
            }
        )
        for r in fold_rows:
            r["best_threshold"] = best_t
            r["best_threshold_acc"] = best_t_acc
        rows.extend(fold_rows)

        print(
            f"Fold {fold_idx}  test {base['test_start']} .. {base['test_end']}  "
            f"train_days={n_td}  auc={auc:.4f}  best_th={best_t:.3f} (acc={best_t_acc:.4f})"
        )
        if args.execution_price in {"next_open_to_close_3d", "next_open_to_close_5d"}:
            print("    note: multi-day return metrics overlap; treat portfolio stats as diagnostics.")
        for r in fold_rows:
            print(
                f"    {r['portfolio_mode']}: total_return={r['total_return']:.4f}  sharpe={r['sharpe']:.3f}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
