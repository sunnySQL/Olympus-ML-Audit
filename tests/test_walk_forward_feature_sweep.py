"""Walk-forward feature sweep helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import walk_forward_feature_sweep as wffs  # noqa: E402


class TestSelectSignals(unittest.TestCase):
    def test_selects_top_k_by_metric(self):
        df = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "direction": ["high", "low", "high"],
                "top_minus_bottom_sharpe": [0.1, 0.4, -0.2],
                "ic_mean": [0.0, 0.0, 0.0],
                "top_minus_spy_sharpe": [0.0, 0.0, 0.0],
            }
        )
        out = wffs.select_signals(df, top_k=2)
        self.assertEqual(list(out["feature"]), ["b", "a"])
        self.assertEqual(list(out["selection_rank"]), [1, 2])

    def test_min_train_metric_filters(self):
        df = pd.DataFrame(
            {
                "feature": ["a", "b"],
                "direction": ["high", "low"],
                "top_minus_bottom_sharpe": [-0.1, -0.2],
                "ic_mean": [0.0, 0.0],
                "top_minus_spy_sharpe": [0.0, 0.0],
            }
        )
        out = wffs.select_signals(df, top_k=2, min_train_metric=0.0)
        self.assertTrue(out.empty)


class TestEnsembleScore(unittest.TestCase):
    def test_low_direction_contributes_inverse_rank(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                "ticker": ["A", "B"],
                "a": [1.0, 2.0],
                "b": [10.0, 5.0],
            }
        )
        selected = pd.DataFrame(
            {
                "feature": ["a", "b"],
                "direction": ["high", "low"],
                "selection_rank": [1, 2],
                "selection_metric_value": [1.0, 1.0],
            }
        )
        out = wffs.add_ensemble_score(df, selected)
        a_score = out.loc[out["ticker"] == "A", "_ensemble_score"].iloc[0]
        b_score = out.loc[out["ticker"] == "B", "_ensemble_score"].iloc[0]
        self.assertGreater(b_score, a_score)


if __name__ == "__main__":
    unittest.main()
