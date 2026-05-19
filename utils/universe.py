"""Shared project universe loading."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNIVERSE_PATH = ROOT / "config" / "universe.csv"
DEFAULT_FALLBACK_UNIVERSE = ["AAPL", "MSFT", "NVDA", "SPY", "GLD", "SLV"]


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def parse_tickers(tickers: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if tickers is None:
        return []
    if isinstance(tickers, str):
        raw = tickers.replace("\n", ",").split(",")
    else:
        raw = list(tickers)
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        sym = normalize_ticker(t)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def load_universe(path: str | Path | None = None, tickers: str | list[str] | None = None) -> list[str]:
    """Load tickers from explicit tickers, a CSV/text file, or the default universe file."""
    explicit = parse_tickers(tickers)
    if explicit:
        return explicit

    p = Path(path).expanduser().resolve() if path else DEFAULT_UNIVERSE_PATH
    if not p.exists():
        return DEFAULT_FALLBACK_UNIVERSE[:]

    if p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "ticker" in reader.fieldnames:
                return parse_tickers([row.get("ticker", "") for row in reader])

    return parse_tickers(p.read_text(encoding="utf-8").splitlines())
