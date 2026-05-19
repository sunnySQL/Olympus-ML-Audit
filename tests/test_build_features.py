"""Tests for build_features pure helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_features as bf  # noqa: E402


class TestWeightedSentiment(unittest.TestCase):
    def test_empty_zero(self):
        sub = pd.DataFrame(columns=["pub_et", "sentiment"])
        asof = pd.Timestamp("2025-01-15 16:00:00", tz=bf.ET)
        self.assertEqual(bf.weighted_sentiment(sub, asof, 24.0), 0.0)

    def test_recent_weights_higher(self):
        asof = pd.Timestamp("2025-01-15 16:00:00", tz=bf.ET)
        sub = pd.DataFrame(
            {
                "pub_et": [
                    pd.Timestamp("2025-01-15 15:00:00", tz=bf.ET),
                    pd.Timestamp("2025-01-10 16:00:00", tz=bf.ET),
                ],
                "sentiment": [1.0, -1.0],
            }
        )
        w = bf.weighted_sentiment(sub, asof, decay_h=24.0)
        self.assertGreater(w, 0.0)


class TestMarketAsofTs(unittest.TestCase):
    def test_closes_1600_et(self):
        ts = bf.market_asof_ts(pd.Timestamp("2025-06-10"))
        self.assertEqual(ts.hour, 16)
        self.assertEqual(ts.minute, 0)


class TestWindowStats(unittest.TestCase):
    def test_empty_returns_zeros(self):
        empty = pd.DataFrame(columns=["pub_et", "sentiment"])
        asof = pd.Timestamp("2025-01-15 16:00:00", tz=bf.ET)
        r = bf.window_stats(empty, asof, pd.Timedelta(hours=6), 24.0)
        self.assertEqual(r, (0, 0.0, 0.0, 0.0, 0.0))


class TestTargets(unittest.TestCase):
    def test_excess_target_excludes_spy(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                "ticker": ["AAPL", "SPY"],
                "Close": [100.0, 100.0],
                "target_return_1d": [0.02, 0.01],
            }
        )
        out = bf.add_alternative_targets(df)
        aapl = out.loc[out["ticker"] == "AAPL", "target_excess_up"].iloc[0]
        spy = out.loc[out["ticker"] == "SPY", "target_excess_up"].iloc[0]
        self.assertEqual(aapl, 1.0)
        self.assertTrue(pd.isna(spy))

    def test_unlabeled_latest_row_survives_feature_build(self):
        dates = pd.bdate_range("2025-01-01", periods=80)
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": np.linspace(100.0, 120.0, len(dates)),
                "High": np.linspace(101.0, 121.0, len(dates)),
                "Low": np.linspace(99.0, 119.0, len(dates)),
                "Close": np.linspace(100.5, 120.5, len(dates)),
                "Volume": np.linspace(1_000_000, 1_100_000, len(dates)),
            }
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "AAPL_daily.csv"
            df.to_csv(path, index=False)
            out = bf.load_price_file(str(path), "AAPL")
        self.assertEqual(out["Date"].max(), dates[-1])
        self.assertTrue(pd.isna(out.loc[out["Date"] == dates[-1], "target_return_1d"]).all())

    def test_next_open_to_future_close_targets(self):
        dates = pd.bdate_range("2025-01-01", periods=80)
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": np.linspace(100.0, 120.0, len(dates)),
                "High": np.linspace(101.0, 121.0, len(dates)),
                "Low": np.linspace(99.0, 119.0, len(dates)),
                "Close": np.linspace(100.5, 120.5, len(dates)),
                "Volume": np.linspace(1_000_000, 1_100_000, len(dates)),
            }
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "AAPL_daily.csv"
            df.to_csv(path, index=False)
            out = bf.load_price_file(str(path), "AAPL")
        d = dates[-10]
        idx = int(df.index[df["Date"] == d][0])
        expected = df["Close"].iloc[idx + 5] / df["Open"].iloc[idx + 1] - 1
        got = out.loc[out["Date"] == d, "target_return_next_open_to_close_5d"].iloc[0]
        self.assertAlmostEqual(got, expected)
        self.assertTrue(pd.isna(out.loc[out["Date"] == dates[-1], "target_return_next_open_to_close_5d"]).all())


if __name__ == "__main__":
    unittest.main()
