"""Cross-sectional rank features."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_features as bf  # noqa: E402


class TestCrossSectionalRanks(unittest.TestCase):
    def test_rank_percentile_two_names(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
                "ticker": ["A", "B"],
                "return_5d": [0.10, 0.05],
                "momentum_20d": [0.0, 0.0],
                "volume_ratio": [1.0, 1.0],
            }
        )
        out = bf.add_cross_sectional_ranks(df)
        self.assertIn("cs_rank_return_5d", out.columns)
        g = out[out["Date"] == out["Date"].iloc[0]]
        lo, hi = g["cs_rank_return_5d"].min(), g["cs_rank_return_5d"].max()
        self.assertLess(lo, hi)
        self.assertTrue((g["cs_rank_return_5d"] >= 0).all() and (g["cs_rank_return_5d"] <= 1).all())


if __name__ == "__main__":
    unittest.main()
