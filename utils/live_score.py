"""Latest-row-per-ticker scoring (shared by demo / CLI)."""
from __future__ import annotations

import pandas as pd

from utils.predict_bundle import add_pred_prob


def latest_row_per_ticker(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    df = df.dropna(subset=list(feature_names) + ["ticker", "Date"]).copy()
    if df.empty:
        raise ValueError("No complete rows after dropping NaNs in model features.")
    idx = df.groupby("ticker")["Date"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def score_latest_per_ticker(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Adds pred_prob (and pred_return for regression bundles)."""
    feats = list(bundle.get("feature_names") or [])
    latest = latest_row_per_ticker(df, feats)
    scored = add_pred_prob(latest, bundle)
    out = latest.copy()
    out["pred_prob"] = scored["pred_prob"].values
    if bundle.get("task") == "regression" and "pred_return" in scored.columns:
        out["pred_return"] = scored["pred_return"].values
    return out


def signal_from_thresholds(p: float, long_th: float, short_th: float) -> str:
    if p > long_th:
        return "long"
    if p < short_th:
        return "short"
    return "flat"
