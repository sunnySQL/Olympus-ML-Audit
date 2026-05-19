"""Project universe loader."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.universe import load_universe, normalize_ticker, parse_tickers


class TestUniverse(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_ticker(" spy "), "SPY")

    def test_parse_dedupes(self):
        self.assertEqual(parse_tickers("spy, aapl,SPY"), ["SPY", "AAPL"])

    def test_load_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "u.csv"
            p.write_text("ticker,role\nspy,benchmark\nmsft,equity\n", encoding="utf-8")
            self.assertEqual(load_universe(p), ["SPY", "MSFT"])

    def test_explicit_tickers_win(self):
        self.assertEqual(load_universe(tickers="nvda,amd"), ["NVDA", "AMD"])


if __name__ == "__main__":
    unittest.main()
