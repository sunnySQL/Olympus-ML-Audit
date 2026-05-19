"""Raw feature ranking sweep helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import sweep_rank_features as srf  # noqa: E402


class TestCandidateFeatures(unittest.TestCase):
    def test_requested_missing_raises(self):
        df = pd.DataFrame({"return_5d": [0.1]})
        with self.assertRaises(ValueError):
            srf.candidate_features(df, ["return_5d", "missing_feature"])

    def test_requested_preserves_order(self):
        df = pd.DataFrame({"b": [1.0], "a": [2.0]})
        self.assertEqual(srf.candidate_features(df, ["a", "b"]), ["a", "b"])

    def test_default_drops_sparse_candidates(self):
        df = pd.DataFrame(
            {
                "return_5d": [0.0, 1.0] * 500,
                "news_count_3d": [0.0] * 999 + [1.0],
            }
        )
        feats = srf.candidate_features(df, max_zero_frac=0.995)
        self.assertIn("return_5d", feats)
        self.assertNotIn("news_count_3d", feats)


class TestEvaluateFeatureSignal(unittest.TestCase):
    def test_low_direction_inverts_ranking(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-01-02"] * 4),
                "ticker": ["A", "B", "C", "SPY"],
                "rsi_14": [80.0, 20.0, 50.0, 40.0],
                "_exec_return": [-0.02, 0.04, 0.01, 0.0],
            }
        )
        daily_high, _, _ = srf.evaluate_feature_signal(
            df,
            feature="rsi_14",
            direction="high",
            return_col="_exec_return",
            cost_bps=0.0,
            top_n=1,
            top_pct=0.33,
            include_spy=False,
        )
        daily_low, _, _ = srf.evaluate_feature_signal(
            df,
            feature="rsi_14",
            direction="low",
            return_col="_exec_return",
            cost_bps=0.0,
            top_n=1,
            top_pct=0.33,
            include_spy=False,
        )
        self.assertAlmostEqual(daily_high["top_long"].iloc[0], -0.02)
        self.assertAlmostEqual(daily_low["top_long"].iloc[0], 0.04)


if __name__ == "__main__":
    unittest.main()
