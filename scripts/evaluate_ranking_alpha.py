#!/usr/bin/env python3
"""
Evaluate whether model scores rank assets usefully cross-sectionally.

This is stricter than a binary up/down report: each date, rank the tradable
universe by model score, evaluate top-name portfolios, top-minus-bottom spread,
top-minus-SPY spread, and information coefficient (Spearman score vs return).

Usage:
  python scripts/evaluate_ranking_alpha.py
  python scripts/evaluate_ranking_alpha.py --top-n 2
  python scripts/evaluate_ranking_alpha.py --execution-price close_to_close
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evaluate_backtest import EXECUTION_PRICE_CHOICES, attach_execution_returns, compute_metrics
from utils.predict_bundle import add_pred_prob

DATA_PATH = ROOT / "data" / "features.csv"
MODEL_PATH = ROOT / "models" / "direction_model.pkl"
REPORT_DIR = ROOT / "reports"


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    try:
        val = spearmanr(x.astype(float), y.astype(float), nan_policy="omit").correlation
    except Exception:
        return float("nan")
    return float(val) if pd.notna(val) else float("nan")


def score_column(scored: pd.DataFrame) -> str:
    """Use regression prediction if available, otherwise classification probability."""
    if "pred_return" in scored.columns and scored["pred_return"].notna().any():
        return "pred_return"
    return "pred_prob"


def ranked_rows(
    scored: pd.DataFrame,
    score_col: str,
    return_col: str,
    include_spy: bool = False,
) -> pd.DataFrame:
    """Rows eligible for ranking, with within-date rank percentile."""
    d = scored.dropna(subset=[score_col, return_col, "ticker", "Date"]).copy()
    if not include_spy:
        d = d[d["ticker"].astype(str).str.upper() != "SPY"].copy()
    d["_score_rank_pct"] = d.groupby("Date")[score_col].rank(pct=True, method="first")
    return d


def build_daily_ranking_returns(
    scored: pd.DataFrame,
    score_col: str,
    return_col: str,
    cost_bps: float,
    top_n: int | None = None,
    top_pct: float = 0.33,
    include_spy: bool = False,
) -> pd.DataFrame:
    """
    Build daily returns from cross-sectional rankings.

    Costs are applied as one daily entry/exit cost for each long leg. Spread
    portfolios charge both long and short legs.
    """
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be positive")
    if not 0 < top_pct <= 1:
        raise ValueError("top_pct must be in (0, 1]")

    cost = cost_bps / 10000.0
    ranked = ranked_rows(scored, score_col, return_col, include_spy=include_spy)
    ranked_by_date = {
        dt: g.sort_values(score_col, ascending=False)
        for dt, g in ranked.groupby("Date", sort=True)
    }
    rows: list[dict] = []

    for dt, full_g in scored.dropna(subset=[return_col]).groupby("Date", sort=True):
        g = ranked_by_date.get(dt)
        if g is None:
            continue
        if len(g) == 0:
            continue

        n = len(g)
        k_top = int(top_n) if top_n is not None else int(np.ceil(n * top_pct))
        k_top = max(1, min(k_top, n))
        k_spread = min(k_top, n // 2)

        top = g.head(k_top)
        bottom = g.tail(k_spread) if k_spread > 0 else g.iloc[0:0]

        top_raw = float(top[return_col].mean())
        ew_raw = float(g[return_col].mean())
        bottom_raw = float(bottom[return_col].mean()) if len(bottom) else float("nan")

        spy_rows = full_g[full_g["ticker"].astype(str).str.upper() == "SPY"]
        spy_raw = float(spy_rows[return_col].iloc[0]) if len(spy_rows) else float("nan")

        ic = _safe_spearman(g[score_col], g[return_col])
        top_minus_bottom = (
            top_raw - bottom_raw - (2 * cost) if pd.notna(bottom_raw) else float("nan")
        )
        top_minus_spy = (
            top_raw - spy_raw - (2 * cost) if pd.notna(spy_raw) else float("nan")
        )

        rows.append(
            {
                "Date": dt,
                "n_ranked": n,
                "k_top": k_top,
                "k_spread": k_spread,
                "ic_spearman": ic,
                "top_long": top_raw - cost,
                "bottom_short": (-bottom_raw - cost) if pd.notna(bottom_raw) else float("nan"),
                "top_minus_bottom": top_minus_bottom,
                "top_minus_spy": top_minus_spy,
                "ew_ranked": ew_raw - cost,
                "spy": spy_raw - cost if pd.notna(spy_raw) else float("nan"),
                "top_raw": top_raw,
                "bottom_raw": bottom_raw,
                "spy_raw": spy_raw,
                "top_mean_score": float(top[score_col].mean()),
                "bottom_mean_score": float(bottom[score_col].mean()) if len(bottom) else float("nan"),
                "top_hit_positive": float((top[return_col] > 0).mean()),
                "top_beats_ew": float(top_raw > ew_raw),
                "top_beats_spy": float(top_raw > spy_raw) if pd.notna(spy_raw) else float("nan"),
                "top_beats_bottom": float(top_raw > bottom_raw) if pd.notna(bottom_raw) else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def summarize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "top_long",
        "bottom_short",
        "top_minus_bottom",
        "top_minus_spy",
        "ew_ranked",
        "spy",
    ]
    rows: list[dict] = []
    for col in metric_cols:
        if col not in daily.columns:
            continue
        s = daily[col].dropna()
        m = compute_metrics(s)
        rows.append(
            {
                "strategy": col,
                **m,
                "mean_daily_return": float(s.mean()) if len(s) else float("nan"),
                "daily_win_rate": float((s > 0).mean()) if len(s) else float("nan"),
            }
        )

    ic = daily["ic_spearman"].dropna() if "ic_spearman" in daily.columns else pd.Series(dtype=float)
    if len(ic):
        ic_std = float(ic.std())
        rows.append(
            {
                "strategy": "ic_spearman",
                "days": int(len(ic)),
                "cagr": float("nan"),
                "sharpe": float(ic.mean() / ic_std * np.sqrt(252)) if ic_std > 0 else 0.0,
                "max_drawdown": float("nan"),
                "total_return": float("nan"),
                "mean_daily_return": float(ic.mean()),
                "daily_win_rate": float((ic > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def monthly_stability(daily: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "top_long",
        "bottom_short",
        "top_minus_bottom",
        "top_minus_spy",
        "ew_ranked",
        "spy",
    ]
    out = daily.copy()
    out["month"] = pd.to_datetime(out["Date"]).dt.to_period("M").astype(str)
    rows: list[dict] = []
    for month, g in out.groupby("month", sort=True):
        row: dict = {"month": month, "trading_days": int(len(g))}
        for col in metric_cols:
            s = g[col].dropna()
            row[col] = float((1 + s).prod() - 1) if len(s) else float("nan")
        row["ic_spearman_mean"] = float(g["ic_spearman"].mean()) if "ic_spearman" in g else float("nan")
        row["top_beats_spy_rate"] = float(g["top_beats_spy"].mean()) if "top_beats_spy" in g else float("nan")
        row["top_beats_bottom_rate"] = (
            float(g["top_beats_bottom"].mean()) if "top_beats_bottom" in g else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def score_buckets(
    scored: pd.DataFrame,
    score_col: str,
    return_col: str,
    include_spy: bool = False,
) -> pd.DataFrame:
    ranked = ranked_rows(scored, score_col, return_col, include_spy=include_spy)
    if len(ranked) == 0:
        return pd.DataFrame()
    ranked["score_bucket"] = pd.cut(
        ranked["_score_rank_pct"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["q1_low", "q2", "q3", "q4", "q5_high"],
        include_lowest=True,
    )
    g = ranked.groupby("score_bucket", observed=False)
    return g.agg(
        rows=(return_col, "size"),
        mean_return=(return_col, "mean"),
        hit_rate=(return_col, lambda s: float((s > 0).mean())),
        mean_score=(score_col, "mean"),
        mean_rank_pct=("_score_rank_pct", "mean"),
    ).reset_index()


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate cross-sectional ranking alpha")
    p.add_argument("--features", type=str, default=str(DATA_PATH))
    p.add_argument("--model", type=str, default=str(MODEL_PATH))
    p.add_argument(
        "--execution-price",
        choices=EXECUTION_PRICE_CHOICES,
        default="next_open_to_close",
    )
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--top-n", type=int, default=None, help="Fixed number of names to long each day")
    p.add_argument("--top-pct", type=float, default=0.33, help="Fraction of ranked universe to long")
    p.add_argument(
        "--include-spy",
        action="store_true",
        help="Include SPY in ranked/traded universe. Default excludes SPY and uses it as benchmark.",
    )
    p.add_argument("--full-sample", action="store_true", help="Include training period")
    p.add_argument("--reports-dir", type=str, default=str(REPORT_DIR))
    args = p.parse_args()

    features_path = Path(args.features).resolve()
    model_path = Path(args.model).resolve()
    if not features_path.is_file():
        raise SystemExit(f"Missing features: {features_path}")
    if not model_path.is_file():
        raise SystemExit(f"Missing model: {model_path}")

    df = pd.read_csv(features_path, parse_dates=["Date"])
    bundle = joblib.load(model_path)
    feats = list(bundle.get("feature_names") or [])
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise SystemExit(f"features.csv missing model columns: {missing[:8]}")
    df = df.dropna(subset=feats).sort_values(["Date", "ticker"])

    cutoff = bundle.get("train_cutoff_date")
    if cutoff and not args.full_sample:
        cutoff_ts = pd.to_datetime(cutoff)
        df = df[df["Date"] >= cutoff_ts].copy()
        print(f"Holdout ranking eval from {cutoff_ts.date()} (use --full-sample to include training period)")

    if len(df) == 0:
        raise SystemExit("No rows to score after cutoff/features filter")

    scored = add_pred_prob(df, bundle)
    scored, return_col = attach_execution_returns(scored, args.execution_price)
    scored = scored.dropna(subset=[return_col]).copy()
    score_col = score_column(scored)

    daily = build_daily_ranking_returns(
        scored,
        score_col=score_col,
        return_col=return_col,
        cost_bps=args.cost_bps,
        top_n=args.top_n,
        top_pct=args.top_pct,
        include_spy=args.include_spy,
    )
    if len(daily) == 0:
        raise SystemExit("No daily ranking rows produced")

    summary = summarize_daily(daily)
    monthly = monthly_stability(daily)
    buckets = score_buckets(scored, score_col, return_col, include_spy=args.include_spy)

    report_dir = Path(args.reports_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    daily_path = report_dir / "ranking_alpha_daily.csv"
    summary_path = report_dir / "ranking_alpha_summary.csv"
    monthly_path = report_dir / "ranking_alpha_by_month.csv"
    bucket_path = report_dir / "ranking_alpha_buckets.csv"
    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    buckets.to_csv(bucket_path, index=False)

    print(f"Model: {model_path}")
    print(f"Score column: {score_col}")
    print(f"Execution: {args.execution_price} | return_col={return_col} | cost_bps={args.cost_bps}")
    if args.execution_price in {"next_open_to_close_3d", "next_open_to_close_5d"}:
        print("Note: 3d/5d ranking returns overlap; use them as signal diagnostics, not final portfolio stats.")
    print(f"Ranked universe: {'includes SPY' if args.include_spy else 'excludes SPY'}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved {summary_path}, {daily_path}, {monthly_path}, {bucket_path}")


if __name__ == "__main__":
    main()
