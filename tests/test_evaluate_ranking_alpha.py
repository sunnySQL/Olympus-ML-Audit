"""Pure ranking-alpha evaluation helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_ranking_alpha as era  # noqa: E402


class TestDailyRankingReturns(unittest.TestCase):
    def test_top_long_and_spreads(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-01-02"] * 4),
                "ticker": ["A", "B", "C", "SPY"],
                "pred_prob": [0.9, 0.2, 0.7, 0.1],
                "_exec_return": [0.04, -0.03, 0.01, 0.005],
            }
        )
        out = era.build_daily_ranking_returns(
            df,
            score_col="pred_prob",
            return_col="_exec_return",
            cost_bps=0.0,
            top_n=1,
            include_spy=False,
        )
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["n_ranked"], 3)
        self.assertEqual(row["k_top"], 1)
        self.assertAlmostEqual(row["top_long"], 0.04)
        self.assertAlmostEqual(row["top_minus_bottom"], 0.07)
        self.assertAlmostEqual(row["top_minus_spy"], 0.035)
        self.assertEqual(row["top_beats_bottom"], 1.0)

    def test_costs_apply_to_spread_legs(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-01-02"] * 3),
                "ticker": ["A", "B", "SPY"],
                "pred_prob": [0.9, 0.1, 0.5],
                "_exec_return": [0.02, 0.00, 0.01],
            }
        )
        out = era.build_daily_ranking_returns(
            df,
            score_col="pred_prob",
            return_col="_exec_return",
            cost_bps=5.0,
            top_n=1,
            include_spy=False,
        )
        row = out.iloc[0]
        self.assertAlmostEqual(row["top_long"], 0.0195)
        self.assertAlmostEqual(row["top_minus_spy"], 0.009)


class TestBuckets(unittest.TestCase):
    def test_score_buckets_emit_rows(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-01-02"] * 5),
                "ticker": ["A", "B", "C", "D", "E"],
                "pred_prob": [0.1, 0.2, 0.3, 0.4, 0.5],
                "_exec_return": [-0.02, -0.01, 0.0, 0.01, 0.02],
            }
        )
        buckets = era.score_buckets(df, "pred_prob", "_exec_return", include_spy=False)
        self.assertEqual(len(buckets), 5)
        high = buckets[buckets["score_bucket"].astype(str) == "q5_high"].iloc[0]
        self.assertAlmostEqual(high["mean_return"], 0.02)


if __name__ == "__main__":
    unittest.main()
