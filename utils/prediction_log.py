"""
Persistent prediction logger — appends daily signals to a CSV so the app
can show yesterday's scorecard and a signal-history heatmap.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PREDICTION_LOG_COLS = [
    "scored_date", "ticker", "p_up", "signal", "long_th", "short_th",
]


def append_signals(
    rows: list[dict],
    long_th: float,
    short_th: float,
    log_path: Path,
) -> None:
    """Append today's signals, deduplicating by (scored_date, ticker)."""
    if not rows:
        return

    new = pd.DataFrame([
        {
            "scored_date": r["date"],
            "ticker": r["ticker"],
            "p_up": round(r["p_up"], 6),
            "signal": r.get("signal", ""),
            "long_th": long_th,
            "short_th": short_th,
        }
        for r in rows
    ])

    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
        existing = pd.read_csv(log_path, dtype=str)
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["scored_date", "ticker"], keep="last")
    else:
        combined = new

    combined = combined.sort_values(["scored_date", "ticker"])
    tmp = log_path.with_suffix(".tmp")
    combined.to_csv(tmp, index=False)
    tmp.rename(log_path)


def load_prediction_log(log_path: Path) -> pd.DataFrame | None:
    if not log_path.exists():
        return None
    try:
        df = pd.read_csv(log_path)
        df["p_up"] = pd.to_numeric(df["p_up"], errors="coerce")
        return df
    except Exception:
        return None
