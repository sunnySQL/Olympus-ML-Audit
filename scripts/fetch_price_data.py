"""
Download daily OHLCV for project tickers via yfinance and write data/<SYMBOL>_daily.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.universe import DEFAULT_UNIVERSE_PATH, load_universe

DATA_DIR = ROOT / "data"
DEFAULT_PERIOD = "5y"


def main():
    ap = argparse.ArgumentParser(description="Download daily OHLCV via yfinance")
    ap.add_argument("--universe", type=str, default=str(DEFAULT_UNIVERSE_PATH), help="CSV/text universe file")
    ap.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers; overrides --universe")
    ap.add_argument("--period", type=str, default=DEFAULT_PERIOD, help="yfinance period, e.g. 5y, 10y, max")
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    args = ap.parse_args()

    assets = load_universe(args.universe, args.tickers)
    data_dir = Path(args.data_dir).resolve()
    if not assets:
        raise SystemExit("Universe is empty")

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(assets)} tickers: {', '.join(assets)}")
    for sym in assets:
        tk = yf.Ticker(sym)
        hist = tk.history(period=args.period, auto_adjust=True)
        if hist.empty:
            print(f"No data for {sym}, skipping")
            continue
        out = hist.reset_index()
        out = out.rename(
            columns={
                "Date": "Date",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }
        )
        out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
        out["ticker"] = sym
        cols = ["Date", "Close", "High", "Low", "Open", "Volume", "ticker"]
        path = data_dir / f"{sym}_daily.csv"
        out[cols].to_csv(path, index=False)
        print(f"Wrote {len(out)} rows to {path}")


if __name__ == "__main__":
    main()
