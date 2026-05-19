import os
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURE_SET_VERSION = "v10"

TARGET_COLUMNS = {
    "target_direction",
    "target_return_1d",
    "target_return_next_open_to_close",
    "target_return_next_open_to_close_3d",
    "target_return_next_open_to_close_5d",
    "target_intraday_next_direction",
    "target_direction_next_open_to_close_3d",
    "target_direction_next_open_to_close_5d",
    "target_excess_up",
    "target_return_5d",
    "target_direction_5d",
}

ET = ZoneInfo("America/New_York")
NEWS_DECAY_HOURS = 24.0


def load_price_file(path: str, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], dayfirst=False)
    df = df[df["Date"].notna()].copy()

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).sort_values("Date")

    if "ticker" not in df.columns:
        df["ticker"] = symbol
    else:
        df["ticker"] = df["ticker"].astype(str).replace({"nan": symbol})
        df.loc[df["ticker"].str.strip() == "", "ticker"] = symbol

    prev_close = df["Close"].shift(1)
    df["gap_pct"] = (df["Open"] - prev_close) / prev_close
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["momentum_3d"] = df["Close"].pct_change(3)
    df["momentum_10d"] = df["Close"].pct_change(10)
    df["momentum_20d"] = df["Close"].pct_change(20)

    df["vol_5d"] = df["return_1d"].rolling(5).std()
    df["vol_10d"] = df["return_1d"].rolling(10).std()
    df["vol_20d"] = df["return_1d"].rolling(20).std()

    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["dist_ma_20"] = (df["Close"] - df["ma_20"]) / df["ma_20"]
    df["dist_ma_50"] = (df["Close"] - df["ma_50"]) / df["ma_50"]

    c_mu = df["Close"].rolling(20).mean()
    c_sd = df["Close"].rolling(20).std()
    df["zscore_close_20d"] = (df["Close"] - c_mu) / c_sd.replace(0, np.nan)

    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["vol_ma20"]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    cc = df["Close"].replace(0, np.nan)
    oo = df["Open"].replace(0, np.nan)
    df["hl_range_pct"] = (df["High"] - df["Low"]) / cc
    df["intraday_return"] = (df["Close"] - df["Open"]) / oo
    v20 = df["vol_20d"].replace(0, np.nan)
    v10 = df["vol_10d"].replace(0, np.nan)
    df["vol_5_over_20"] = df["vol_5d"] / v20
    df["vol_10d_over_20"] = df["vol_10d"] / v20
    df["vol_ratio_5_10"] = df["vol_5d"] / v10

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    cc_safe = df["Close"].replace(0, np.nan)
    df["macd_line"] = (ema12 - ema26) / cc_safe

    df["hl_range_mean_5d"] = df["hl_range_pct"].rolling(5).mean()
    df["return_1d_lag1"] = df["return_1d"].shift(1)
    df["return_1d_lag2"] = df["return_1d"].shift(2)
    df["return_1d_lag3"] = df["return_1d"].shift(3)
    df["return_5d_lag1"] = df["return_5d"].shift(1)

    # ATR (14-day average true range, normalized by close)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean() / cc

    # Bollinger %B
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["bollinger_pctb"] = (df["Close"] - bb_lower) / bb_range

    # Overnight return (today's open vs previous close)
    df["overnight_return"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Close-to-close vs intraday range ratio
    cc_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["close_vs_range"] = (df["Close"] - df["Open"]).abs() / cc_range

    # Vol-of-vol (rolling std of 5d vol)
    df["vol_of_vol_20d"] = df["vol_5d"].rolling(20).std()

    next_close = df["Close"].shift(-1)
    next_open = df["Open"].shift(-1)
    df["target_direction"] = np.where(
        next_close.notna(),
        (next_close > df["Close"]).astype(float),
        np.nan,
    )
    df["target_return_1d"] = next_close / df["Close"] - 1
    df["target_return_next_open_to_close"] = next_close / next_open.replace(0, np.nan) - 1
    df["target_intraday_next_direction"] = np.where(
        next_close.notna() & next_open.notna(),
        (next_close > next_open).astype(float),
        np.nan,
    )
    for horizon in (3, 5):
        future_close = df["Close"].shift(-horizon)
        ret_col = f"target_return_next_open_to_close_{horizon}d"
        dir_col = f"target_direction_next_open_to_close_{horizon}d"
        df[ret_col] = future_close / next_open.replace(0, np.nan) - 1
        df[dir_col] = np.where(
            future_close.notna() & next_open.notna(),
            (future_close > next_open).astype(float),
            np.nan,
        )

    drop_cols = ["vol_ma20"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    feature_required = [c for c in df.columns if c not in TARGET_COLUMNS]
    df = df.dropna(subset=feature_required)
    return df


def add_relative_strength_vs_spy(all_df: pd.DataFrame) -> pd.DataFrame:
    spy = all_df[all_df["ticker"].str.upper() == "SPY"][["Date", "Close"]].drop_duplicates("Date")
    spy = spy.rename(columns={"Close": "spy_close"})
    out = all_df.merge(spy, on="Date", how="left")
    out["rel_ratio"] = out["Close"] / out["spy_close"]
    out["rel_mom_20"] = out.groupby("ticker", group_keys=False)["rel_ratio"].apply(lambda s: s / s.shift(20) - 1)
    out.loc[out["ticker"].str.upper() == "SPY", "rel_mom_20"] = 0.0
    out = out.drop(columns=["spy_close"], errors="ignore")
    out = out.dropna(subset=["rel_ratio"])
    return out


def add_cross_asset_features(all_df: pd.DataFrame) -> pd.DataFrame:
    spy = all_df[all_df["ticker"].str.upper() == "SPY"][["Date", "return_1d", "vol_10d"]].drop_duplicates("Date")
    spy = spy.rename(columns={"return_1d": "spy_return_1d", "vol_10d": "spy_vol_10d"})
    out = all_df.merge(spy, on="Date", how="left")
    out["excess_return_1d"] = out["return_1d"] - out["spy_return_1d"]

    gld = out[out["ticker"].str.upper() == "GLD"][["Date", "Close"]].rename(columns={"Close": "gld_close"})
    slv = out[out["ticker"].str.upper() == "SLV"][["Date", "Close"]].rename(columns={"Close": "slv_close"})
    gs = gld.merge(slv, on="Date", how="inner")
    gs["gld_slv_ratio"] = gs["gld_close"] / gs["slv_close"].replace(0, np.nan)
    gs = gs[["Date", "gld_slv_ratio"]]
    out = out.merge(gs, on="Date", how="left")
    out["gld_slv_ratio"] = out["gld_slv_ratio"].ffill().fillna(0.0)

    return out


def add_cross_sectional_ranks(all_df: pd.DataFrame) -> pd.DataFrame:
    """Within each trading date, percentile rank vs other names (relative strength)."""
    out = all_df.copy()
    pairs = [
        ("return_5d", "cs_rank_return_5d"),
        ("momentum_20d", "cs_rank_momentum_20d"),
        ("volume_ratio", "cs_rank_volume_ratio"),
        ("rsi_14", "cs_rank_rsi"),
        ("vol_5d", "cs_rank_vol"),
        ("sentiment_mean_24h", "cs_rank_sentiment"),
    ]
    g = out.groupby("Date", sort=False)
    for src, dst in pairs:
        if src not in out.columns:
            continue
        out[dst] = g[src].rank(pct=True, method="average")
        out[dst] = out[dst].fillna(0.5)
    return out


def add_calendar_features(all_df: pd.DataFrame) -> pd.DataFrame:
    out = all_df.copy()
    d = pd.to_datetime(out["Date"])
    out["day_of_week"] = d.dt.dayofweek.astype(np.float64)
    out["month_of_year"] = d.dt.month.astype(np.float64)
    out["quarter"] = d.dt.quarter.astype(np.float64)
    out["is_month_start"] = (d.dt.day <= 3).astype(np.float64)
    out["is_month_end"] = (d.dt.day >= 27).astype(np.float64)
    return out


def prepare_news_timeseries(news_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(news_path):
        return None
    n = pd.read_csv(news_path)
    n["pub"] = pd.NaT
    if "published_at" in n.columns:
        n["pub"] = pd.to_datetime(n["published_at"], utc=True, errors="coerce")
    if "date" in n.columns:
        from_date = pd.to_datetime(n["date"], errors="coerce", utc=True)
        m = n["pub"].isna() & from_date.notna()
        n.loc[m, "pub"] = from_date[m] + pd.Timedelta(hours=12)
    n = n.dropna(subset=["pub", "ticker"])
    if len(n) == 0:
        return None
    n["ticker"] = n["ticker"].astype(str).str.upper()
    n["pub_et"] = n["pub"].dt.tz_convert(ET)
    return n


def market_asof_ts(trading_date) -> pd.Timestamp:
    d = pd.Timestamp(trading_date).normalize()
    if d.tzinfo is None:
        return d.tz_localize(ET).replace(hour=16, minute=0, second=0, microsecond=0)
    return d.tz_convert(ET).replace(hour=16, minute=0, second=0, microsecond=0)


def weighted_sentiment(sub: pd.DataFrame, asof_et: pd.Timestamp, decay_h: float) -> float:
    if len(sub) == 0:
        return 0.0
    hours = (asof_et - sub["pub_et"]).dt.total_seconds() / 3600.0
    hours = np.maximum(hours.values, 1e-6)
    w = np.exp(-hours / max(decay_h, 1e-6))
    s = sub["sentiment"].astype(float).values
    sw = (s * w).sum()
    ww = w.sum()
    return float(sw / ww) if ww > 0 else 0.0


def window_stats(sub: pd.DataFrame, asof_et: pd.Timestamp, delta: pd.Timedelta, decay_h: float):
    if len(sub) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    mask = (sub["pub_et"] <= asof_et) & (sub["pub_et"] > asof_et - delta)
    wsub = sub.loc[mask]
    cnt = int(len(wsub))
    if cnt == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    sm = float(wsub["sentiment"].mean())
    smax = float(wsub["sentiment"].max())
    smin = float(wsub["sentiment"].min())
    wmean = weighted_sentiment(wsub, asof_et, decay_h)
    return cnt, sm, smax, smin, wmean


def compute_news_window_features(all_df: pd.DataFrame, news_et: pd.DataFrame) -> pd.DataFrame:
    keys = all_df[["Date", "ticker"]].drop_duplicates().sort_values(["Date", "ticker"])
    rows = []
    w6h = pd.Timedelta(hours=6)
    w24h = pd.Timedelta(hours=24)
    w3d = pd.Timedelta(days=3)

    for _, kr in keys.iterrows():
        d = kr["Date"]
        t = str(kr["ticker"]).upper()
        asof_et = market_asof_ts(d)
        sub = news_et[news_et["ticker"] == t]

        c6, m6, mx6, mn6, w6 = window_stats(sub, asof_et, w6h, NEWS_DECAY_HOURS)
        c24, m24, mx24, mn24, w24 = window_stats(sub, asof_et, w24h, NEWS_DECAY_HOURS)
        c3d, m3d, mx3d, mn3d, w3d_val = window_stats(sub, asof_et, w3d, NEWS_DECAY_HOURS)

        rows.append(
            {
                "Date": d,
                "ticker": t,
                "news_count_6h": c6,
                "news_count_24h": c24,
                "news_count_3d": c3d,
                "sentiment_mean_6h": m6,
                "sentiment_mean_24h": m24,
                "sentiment_mean_3d": m3d,
                "sentiment_max_6h": mx6,
                "sentiment_min_6h": mn6,
                "sentiment_max_24h": mx24,
                "sentiment_min_24h": mn24,
                "weighted_sentiment_6h": w6,
                "weighted_sentiment_24h": w24,
                "weighted_sentiment_3d": w3d_val,
            }
        )

    feat = pd.DataFrame(rows)
    return all_df.merge(feat, on=["Date", "ticker"], how="left")


def fill_zero_news(all_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "news_count_6h",
        "news_count_24h",
        "news_count_3d",
        "sentiment_mean_6h",
        "sentiment_mean_24h",
        "sentiment_mean_3d",
        "sentiment_max_6h",
        "sentiment_min_6h",
        "sentiment_max_24h",
        "sentiment_min_24h",
        "weighted_sentiment_6h",
        "weighted_sentiment_24h",
        "weighted_sentiment_3d",
    ]
    for c in cols:
        if c in all_df.columns:
            all_df[c] = all_df[c].fillna(0.0)
        else:
            all_df[c] = 0.0

    count_col = "news_count_3d" if "news_count_3d" in all_df.columns else None
    if count_col:
        all_df["has_news"] = (all_df[count_col] > 0).astype(np.float64)
    else:
        all_df["has_news"] = 0.0
    return all_df


def add_interaction_features(all_df: pd.DataFrame) -> pd.DataFrame:
    out = all_df.copy()
    if "weighted_sentiment_24h" in out.columns and "vol_5d" in out.columns:
        out["sentiment_x_vol"] = out["weighted_sentiment_24h"] * out["vol_5d"]
    if "excess_return_1d" in out.columns and "volume_ratio" in out.columns:
        out["excess_x_volume"] = out["excess_return_1d"] * out["volume_ratio"]
    if "return_1d" in out.columns and "vol_5d" in out.columns:
        v = out["vol_5d"].replace(0, np.nan)
        out["return_over_vol"] = out["return_1d"] / v
    return out


def add_alternative_targets(all_df: pd.DataFrame) -> pd.DataFrame:
    """Extra labels: beat SPY next day (excess), 5d direction/return (smoother horizon)."""
    out = all_df.copy()
    spy = out[out["ticker"].astype(str).str.upper() == "SPY"][["Date", "target_return_1d"]].drop_duplicates("Date")
    spy = spy.rename(columns={"target_return_1d": "spy_next_ret"})
    out = out.merge(spy, on="Date", how="left")
    has_excess_label = (
        out["target_return_1d"].notna()
        & out["spy_next_ret"].notna()
        & (out["ticker"].astype(str).str.upper() != "SPY")
    )
    out["target_excess_up"] = np.where(
        has_excess_label,
        (out["target_return_1d"] > out["spy_next_ret"]).astype(float),
        np.nan,
    )
    close_5d = out.groupby("ticker")["Close"].shift(-5)
    out["target_return_5d"] = close_5d / out["Close"] - 1
    out["target_direction_5d"] = np.where(
        close_5d.notna(),
        (close_5d > out["Close"]).astype(float),
        np.nan,
    )
    out = out.drop(columns=["spy_next_ret"], errors="ignore")
    return out


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_daily.csv")]

    all_features = []
    for file in sorted(files):
        symbol = file.replace("_daily.csv", "").upper()
        path = os.path.join(INPUT_DIR, file)
        all_features.append(load_price_file(path, symbol))

    all_df = pd.concat(all_features, ignore_index=True)
    all_df = add_relative_strength_vs_spy(all_df)
    all_df = add_cross_asset_features(all_df)
    all_df = add_cross_sectional_ranks(all_df)
    all_df = add_calendar_features(all_df)

    news_path = os.path.join(INPUT_DIR, "news.csv")
    news_et = prepare_news_timeseries(news_path)
    if news_et is not None and len(news_et) > 0:
        all_df = compute_news_window_features(all_df, news_et)
    all_df = fill_zero_news(all_df)
    all_df = add_interaction_features(all_df)
    all_df = add_alternative_targets(all_df)

    all_df["feature_set_version"] = FEATURE_SET_VERSION

    out_path = os.path.join(OUTPUT_DIR, "features.csv")
    all_df.to_csv(out_path, index=False)
    print(f"Features ({FEATURE_SET_VERSION}) saved to {out_path}")
