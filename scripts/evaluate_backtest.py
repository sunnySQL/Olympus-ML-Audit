import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import joblib
import numpy as np
import pandas as pd

from utils.predict_bundle import add_pred_prob

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "direction_model.pkl")
EXECUTION_PRICE_CHOICES = [
    "next_open_to_close",
    "next_open_to_close_3d",
    "next_open_to_close_5d",
    "close_to_close",
]


def compute_metrics(returns: pd.Series):
    returns = returns.dropna()
    if len(returns) == 0:
        return {
            "days": 0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
        }

    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    days = len(returns)
    cagr = (1 + total_return) ** (252.0 / days) - 1 if days > 0 else 0.0
    ann_vol = returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
    drawdown = cumulative / cumulative.cummax() - 1
    max_dd = drawdown.min()
    return {
        "days": int(days),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "total_return": float(total_return),
    }


def equal_weight_long_all_daily(
    df: pd.DataFrame,
    cost_bps: float,
    return_col: str = "target_return_1d",
) -> pd.Series:
    """Always long each name; same EW daily portfolio as the model path."""
    cost = cost_bps / 10000.0
    d = df.copy()
    d["_r"] = d[return_col] - cost
    return d.groupby("Date", as_index=False)["_r"].mean()["_r"]


def spy_long_only_series(
    df: pd.DataFrame,
    cost_bps: float,
    return_col: str = "target_return_1d",
) -> pd.Series:
    """SPY buy-and-hold on next-day returns (one row per date)."""
    cost = cost_bps / 10000.0
    spy = df[df["ticker"].astype(str).str.upper() == "SPY"].sort_values("Date")
    return spy[return_col] - cost


def random_sign_equal_weight_daily(
    df: pd.DataFrame,
    cost_bps: float,
    seed: int = 42,
    return_col: str = "target_return_1d",
) -> pd.Series:
    """Coin-flip long/short each row, then EW across tickers per day (sanity noise ceiling)."""
    rng = np.random.default_rng(seed)
    cost = cost_bps / 10000.0
    d = df.copy()
    sign = rng.choice(np.array([-1.0, 1.0]), size=len(d))
    d["_r"] = sign * d[return_col] - cost
    return d.groupby("Date", as_index=False)["_r"].mean()["_r"]


def ew_long_on_model_long_rows(
    bt: pd.DataFrame,
    cost_bps: float,
    return_col: str = "target_return_1d",
) -> pd.Series:
    """
    Equal-weight long only rows where the model is long (pred_signal == 1).
    Same per-name cost as the model when long; 0% return days when model has no longs.
    Compares dumb long on the model's long picks vs the model's full rule.
    """
    cost = cost_bps / 10000.0
    longs = bt.loc[bt["pred_signal"] == 1].copy()
    longs["_r"] = longs[return_col] - cost
    daily_mean = longs.groupby("Date", sort=True)["_r"].mean()
    all_dates = np.sort(bt["Date"].unique())
    return daily_mean.reindex(all_dates, fill_value=0.0)


def apply_positions(
    df: pd.DataFrame,
    cost_bps: float,
    long_threshold: float,
    short_threshold: float,
    long_only: bool = False,
    return_col: str = "target_return_1d",
) -> pd.DataFrame:
    """
    Long/short mode: long if p > long_th; short if p < short_th; else flat.
    Long-only mode: long if p > long_th; else flat (no shorting).
    """
    cost = cost_bps / 10000.0

    def row_return(r):
        p = r["pred_prob"]
        y = r[return_col]
        if p > long_threshold:
            return y - cost
        if long_only:
            return 0.0
        if p < short_threshold:
            return -y - cost
        return 0.0

    out = df.copy()
    out["position_return"] = out.apply(row_return, axis=1)
    return out


def build_top_fraction_long_backtest(
    df: pd.DataFrame,
    cost_bps: float,
    top_fraction: float,
    return_col: str = "target_return_1d",
) -> pd.DataFrame:
    """
    Each calendar date, equal-weight long the top `top_fraction` of names by pred_prob
    (at least one name per day). Others cash. Same cost bps per long leg as threshold rules.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    cost = cost_bps / 10000.0
    d = df.copy()
    rk = d.groupby("Date")["pred_prob"].rank(method="first", ascending=False)
    n = d.groupby("Date")["pred_prob"].transform("count")
    k = np.maximum(1, np.ceil(n * top_fraction).astype(int))
    d["pred_signal"] = np.where(rk <= k, 1, -1)
    d["position_return"] = np.where(d["pred_signal"] == 1, d[return_col] - cost, 0.0)
    return d


def attach_signals(df: pd.DataFrame, long_th: float, short_th: float, long_only: bool) -> pd.DataFrame:
    d = df.copy()
    if long_only:
        d["pred_signal"] = np.where(d["pred_prob"] > long_th, 1, -1)
    else:
        d["pred_signal"] = np.where(
            d["pred_prob"] > long_th,
            1,
            np.where(d["pred_prob"] < short_th, 0, -1),
        )
    return d


def attach_execution_returns(df: pd.DataFrame, execution_price: str) -> tuple[pd.DataFrame, str]:
    """
    Add/choose the realized return column for portfolio evaluation.

    close_to_close reproduces the original research backtest. next_open_to_close is the
    realistic default for signals generated after today's close and entered next session.
    The 3d/5d modes are overlapping signal-horizon returns, useful for research but not
    a non-overlapping cash-account portfolio simulation.
    """
    if execution_price == "close_to_close":
        return df.copy(), "target_return_1d"

    horizons = {
        "next_open_to_close": (1, "target_return_next_open_to_close"),
        "next_open_to_close_3d": (3, "target_return_next_open_to_close_3d"),
        "next_open_to_close_5d": (5, "target_return_next_open_to_close_5d"),
    }
    if execution_price not in horizons:
        raise ValueError(f"unknown execution_price: {execution_price}")

    horizon, col_name = horizons[execution_price]
    out = df.sort_values(["ticker", "Date"]).copy()
    if col_name in out.columns:
        out["_exec_return"] = out[col_name]
    else:
        required = {"Open", "Close", "ticker", "Date"}
        missing = sorted(required - set(out.columns))
        if missing:
            raise ValueError(
                f"{execution_price} execution requires either {col_name} or raw Open/Close columns; "
                f"missing {missing}"
            )
        next_open = out.groupby("ticker")["Open"].shift(-1)
        future_close = out.groupby("ticker")["Close"].shift(-horizon)
        out["_exec_return"] = future_close / next_open.replace(0, np.nan) - 1
    out = out.dropna(subset=["_exec_return"]).copy()
    return out.sort_values(["Date", "ticker"]), "_exec_return"


def main():
    parser = argparse.ArgumentParser(description="Backtest v2: portfolio + per-ticker metrics")
    parser.add_argument(
        "--long-threshold",
        type=float,
        default=0.55,
        help="Go long only if pred_prob exceeds this (default 0.55)",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=0.45,
        help="Go short only if pred_prob is below this (default 0.45); flat between short and long",
    )
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Round-trip cost in basis points")
    parser.add_argument(
        "--execution-price",
        choices=EXECUTION_PRICE_CHOICES,
        default="next_open_to_close",
        help=(
            "Realized return used by the backtest. next_open_to_close is realistic for "
            "after-close one-day signals; 3d/5d modes are overlapping research horizons; "
            "close_to_close reproduces older reports."
        ),
    )
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument(
        "--full-sample",
        action="store_true",
        help="Score all dates (includes training period; metrics look optimistic)",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Only run the long-or-cash strategy (no shorts). Omit to run both modes vs baselines.",
    )
    parser.add_argument(
        "--by-month",
        action="store_true",
        help="Write reports/backtest_by_month.csv (per-calendar-month compound returns on holdout)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=0,
        help="Split holdout into N contiguous date chunks (same trained model); prints per-fold metrics. 0=off",
    )
    parser.add_argument(
        "--long-top-pct",
        type=float,
        default=None,
        metavar="FRACTION",
        help="Also run cross-sectional strategy: each day long top FRACTION of tickers by pred_prob (e.g. 0.33).",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=DATA_PATH,
        help="Path to features.csv (default: data/features.csv)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "reports"),
        help="Directory for backtest CSV outputs",
    )
    args = parser.parse_args()

    features_path = os.path.abspath(args.features)

    if not os.path.exists(features_path):
        print(f"features.csv not found at {features_path}. Run build_features.py first.")
        return

    if not os.path.exists(args.model):
        print("Model not found. Run train_model.py first.")
        return

    df = pd.read_csv(features_path, parse_dates=["Date"])
    bundle = joblib.load(args.model)
    features = bundle["feature_names"]

    df = df.dropna(subset=features)
    df = df.sort_values(["Date", "ticker"])

    cutoff = bundle.get("train_cutoff_date")
    if cutoff and not args.full_sample:
        cutoff_ts = pd.to_datetime(cutoff)
        df = df[df["Date"] >= cutoff_ts].copy()
        print(f"Holdout backtest from {cutoff_ts.date()} (use --full-sample to include training period)")

    if len(df) == 0:
        print("No rows after date filter; check model bundle dates vs features.csv")
        return

    df = add_pred_prob(df, bundle)
    df, return_col = attach_execution_returns(df, args.execution_price)
    df = df.dropna(subset=[return_col])
    if len(df) == 0:
        print(f"No rows after applying execution mode: {args.execution_price}")
        return

    def run_mode(long_only: bool):
        d = attach_signals(df, args.long_threshold, args.short_threshold, long_only=long_only)
        bt = apply_positions(
            d,
            cost_bps=args.cost_bps,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            long_only=long_only,
            return_col=return_col,
        )
        return bt

    modes = []
    if args.long_only:
        modes = [("model_long_only", True)]
    else:
        modes = [("model_long_short", False), ("model_long_only", True)]

    per_rows = []
    port_rows = []
    bt_by_mode: dict[str, pd.DataFrame] = {}

    for mode_name, lo in modes:
        bt = run_mode(long_only=lo)
        bt_by_mode[mode_name] = bt
        for t, g in bt.groupby("ticker"):
            m = compute_metrics(g["position_return"])
            m["ticker"] = t
            m["mode"] = mode_name
            active = g["pred_signal"] != -1
            if active.any():
                up = g.loc[active, return_col] > 0
                sig = g.loc[active, "pred_signal"]
                if lo:
                    hit = (sig == 1) & up
                else:
                    hit = ((sig == 1) & up) | ((sig == 0) & ~up)
                m["hit_rate"] = float(hit.mean())
            else:
                m["hit_rate"] = float("nan")
            per_rows.append(m)

        daily = bt.groupby("Date", as_index=False)["position_return"].mean()
        strat_ret = daily["position_return"]
        port = compute_metrics(strat_ret)
        port["long_threshold"] = args.long_threshold
        port["short_threshold"] = args.short_threshold
        port["cost_bps"] = args.cost_bps
        port["execution_price"] = args.execution_price
        port["label"] = mode_name
        port["long_only"] = lo
        active_bt = bt["pred_signal"] != -1
        if active_bt.any():
            up = bt.loc[active_bt, return_col] > 0
            sig = bt.loc[active_bt, "pred_signal"]
            if lo:
                hit = (sig == 1) & up
            else:
                hit = ((sig == 1) & up) | ((sig == 0) & ~up)
            port["accuracy"] = float(hit.mean())
        else:
            port["accuracy"] = float("nan")
        port_rows.append(port)

    if args.long_top_pct is not None:
        if not 0 < args.long_top_pct <= 1:
            print("--long-top-pct must be in (0, 1]; ignoring")
        else:
            bt_top = build_top_fraction_long_backtest(
                df,
                args.cost_bps,
                args.long_top_pct,
                return_col=return_col,
            )
            mode_name = f"model_top_{args.long_top_pct:.2f}_long"
            bt_by_mode[mode_name] = bt_top
            for t, g in bt_top.groupby("ticker"):
                m = compute_metrics(g["position_return"])
                m["ticker"] = t
                m["mode"] = mode_name
                active = g["pred_signal"] != -1
                if active.any():
                    up = g.loc[active, return_col] > 0
                    sig = g.loc[active, "pred_signal"]
                    hit = (sig == 1) & up
                    m["hit_rate"] = float(hit.mean())
                else:
                    m["hit_rate"] = float("nan")
                per_rows.append(m)
            daily = bt_top.groupby("Date", as_index=False)["position_return"].mean()
            strat_ret = daily["position_return"]
            port = compute_metrics(strat_ret)
            port["long_threshold"] = float("nan")
            port["short_threshold"] = float("nan")
            port["cost_bps"] = args.cost_bps
            port["execution_price"] = args.execution_price
            port["label"] = mode_name
            port["long_only"] = True
            active_bt = bt_top["pred_signal"] != -1
            if active_bt.any():
                up = bt_top.loc[active_bt, return_col] > 0
                sig = bt_top.loc[active_bt, "pred_signal"]
                hit = (sig == 1) & up
                port["accuracy"] = float(hit.mean())
            else:
                port["accuracy"] = float("nan")
            port_rows.append(port)

    per_ticker = pd.DataFrame(per_rows).sort_values(["mode", "ticker"])
    port_df = pd.DataFrame(port_rows)

    # --- Baselines (same dates / universe; helps judge if the model adds anything) ---
    base_ew_long = compute_metrics(equal_weight_long_all_daily(df, args.cost_bps, return_col=return_col))
    base_ew_long["label"] = "baseline_ew_long_all"
    base_ew_long["cost_bps"] = args.cost_bps
    base_ew_long["execution_price"] = args.execution_price

    spy_ser = spy_long_only_series(df, args.cost_bps, return_col=return_col)
    base_spy = compute_metrics(spy_ser) if len(spy_ser) else compute_metrics(pd.Series(dtype=float))
    base_spy["label"] = "baseline_spy_long_only"
    base_spy["cost_bps"] = args.cost_bps
    base_spy["execution_price"] = args.execution_price

    n_days = int(port_df.iloc[0]["days"]) if len(port_df) else 0
    base_cash = compute_metrics(pd.Series(np.zeros(max(n_days, 1))))
    base_cash["label"] = "baseline_cash"
    base_cash["cost_bps"] = 0.0
    base_cash["execution_price"] = args.execution_price

    base_rand = compute_metrics(
        random_sign_equal_weight_daily(df, args.cost_bps, seed=42, return_col=return_col)
    )
    base_rand["label"] = "baseline_random_long_short_ew"
    base_rand["cost_bps"] = args.cost_bps
    base_rand["execution_price"] = args.execution_price

    fair_rows = []
    if "model_long_short" in bt_by_mode:
        s_fair_ls = ew_long_on_model_long_rows(
            bt_by_mode["model_long_short"],
            args.cost_bps,
            return_col=return_col,
        )
        m_fair_ls = compute_metrics(s_fair_ls)
        m_fair_ls["label"] = "baseline_ew_long_on_model_longs_long_short"
        m_fair_ls["cost_bps"] = args.cost_bps
        m_fair_ls["execution_price"] = args.execution_price
        m_fair_ls["strategy"] = m_fair_ls["label"]
        fair_rows.append(m_fair_ls)
    if "model_long_only" in bt_by_mode:
        s_fair_lo = ew_long_on_model_long_rows(
            bt_by_mode["model_long_only"],
            args.cost_bps,
            return_col=return_col,
        )
        m_fair_lo = compute_metrics(s_fair_lo)
        m_fair_lo["label"] = "baseline_ew_long_on_model_longs_long_only"
        m_fair_lo["cost_bps"] = args.cost_bps
        m_fair_lo["execution_price"] = args.execution_price
        m_fair_lo["strategy"] = m_fair_lo["label"]
        fair_rows.append(m_fair_lo)

    report_path = os.path.abspath(args.reports_dir)
    os.makedirs(report_path, exist_ok=True)
    port_path = os.path.join(report_path, "backtest_portfolio.csv")
    tick_path = os.path.join(report_path, "backtest_per_ticker.csv")
    compare_path = os.path.join(report_path, "backtest_comparison.csv")

    compare_rows = []
    for prow in port_rows:
        compare_rows.append({**prow, "strategy": prow["label"]})
    compare_rows.extend(
        [
            {**base_ew_long, "strategy": "baseline_ew_long_all"},
            {**base_spy, "strategy": "baseline_spy_long_only"},
            {**base_cash, "strategy": "baseline_cash"},
            {**base_rand, "strategy": "baseline_random_ew"},
        ]
    )
    for fr in fair_rows:
        compare_rows.append(fr)
    pd.DataFrame(compare_rows).to_csv(compare_path, index=False)

    port_df.to_csv(port_path, index=False)
    per_ticker.to_csv(tick_path, index=False)

    for prow in port_rows:
        lab = str(prow.get("label", ""))
        if lab.startswith("model_top_"):
            title = f"Model — cross-sectional top-fraction long ({lab})"
        elif prow.get("long_only"):
            title = "Model — long-only (long if P>long_threshold, else cash)"
        else:
            title = "Model — long / short / flat (cost only when positioned)"
        print(f"\n{title}:")
        for k, v in prow.items():
            print(f"  {k}: {v}")

    print(f"\nExecution mode: {args.execution_price}")
    if args.execution_price in {"next_open_to_close_3d", "next_open_to_close_5d"}:
        print(
            "Warning: this horizon uses overlapping multi-day signal returns. Treat CAGR/Sharpe as "
            "research diagnostics, not a finalized portfolio simulation."
        )
    print("\nBaselines — same holdout window (see backtest_comparison.csv):")
    print(
        f"  EW long all names (cost {args.cost_bps} bps/day): "
        f"total_return={base_ew_long['total_return']:.4f}  sharpe={base_ew_long['sharpe']:.3f}"
    )
    print(
        f"  SPY long only (cost {args.cost_bps} bps/day): "
        f"total_return={base_spy['total_return']:.4f}  sharpe={base_spy['sharpe']:.3f}"
    )
    print(f"  Cash: total_return={base_cash['total_return']:.4f}  sharpe={base_cash['sharpe']:.3f}")
    print(
        f"  Random long/short EW (seed=42, cost {args.cost_bps} bps/day): "
        f"total_return={base_rand['total_return']:.4f}  sharpe={base_rand['sharpe']:.3f}"
    )
    for fr in fair_rows:
        print(
            f"  Fair-ish — EW long only where model is long ({fr['label']}): "
            f"total_return={fr['total_return']:.4f}  sharpe={fr['sharpe']:.3f}"
        )
    print(
        "\nNote: classic EW long-all charges cost every day on every name. "
        "The model charges cost only when that name is long/short. "
        "The “Fair-ish” row longs the same names/days as the model’s long leg (same bps drag per long)."
    )
    print(f"Saved {port_path}, {tick_path}, {compare_path}")

    if args.by_month and port_rows:
        month_path = os.path.join(report_path, "backtest_by_month.csv")
        month_parts = []
        for prow in port_rows:
            mode_name = str(prow["label"])
            btm = bt_by_mode.get(mode_name)
            if btm is None:
                continue
            dly = btm.groupby("Date", as_index=False)["position_return"].mean()
            dly["month"] = dly["Date"].dt.to_period("M").astype(str)
            for month, g in dly.groupby("month"):
                r = g["position_return"]
                compound = float((1 + r).prod() - 1) if len(r) else 0.0
                month_parts.append(
                    {
                        "mode": mode_name,
                        "month": month,
                        "trading_days": len(r),
                        "month_return": compound,
                        "sharpe_month": float(r.mean() / r.std() * np.sqrt(252)) if r.std() and r.std() > 0 else 0.0,
                    }
                )
        if month_parts:
            pd.DataFrame(month_parts).to_csv(month_path, index=False)
            print(f"Saved {month_path}")

    if args.folds >= 2 and port_rows:
        dates_u = np.sort(df["Date"].unique())
        if len(dates_u) >= args.folds:
            chunks = np.array_split(dates_u, args.folds)
            print(f"\nHoldout split into {args.folds} contiguous date folds (same model, no retrain):")
            for fi, block in enumerate(chunks):
                if len(block) == 0:
                    continue
                sub = df[df["Date"].isin(block)]
                if len(sub) == 0:
                    continue
                d0, d1 = block[0], block[-1]
                print(f"  Fold {fi + 1}  {pd.Timestamp(d0).date()} .. {pd.Timestamp(d1).date()}  ({len(block)} days)")
                for prow in port_rows:
                    mode_name = str(prow["label"])
                    btm = bt_by_mode.get(mode_name)
                    if btm is None:
                        continue
                    bsub = btm[btm["Date"].isin(block)]
                    dly = bsub.groupby("Date", as_index=False)["position_return"].mean()["position_return"]
                    mm = compute_metrics(dly)
                    print(
                        f"    {mode_name}: total_return={mm['total_return']:.4f}  sharpe={mm['sharpe']:.3f}"
                    )
                s_ew = equal_weight_long_all_daily(sub, args.cost_bps, return_col=return_col)
                m_ew = compute_metrics(s_ew)
                print(
                    f"    baseline_ew_long_all: total_return={m_ew['total_return']:.4f}  sharpe={m_ew['sharpe']:.3f}"
                )


if __name__ == "__main__":
    main()
