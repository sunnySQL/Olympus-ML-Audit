"""Tests for train_model split and variance_prune (no model fit)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import train_model as tm  # noqa: E402


class TestVariancePrune(unittest.TestCase):
    def test_drops_constant_column(self):
        df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [0.0, 1.0, 2.0]})
        cols = ["a", "b"]
        keep = tm.variance_prune(cols, df)
        self.assertNotIn("a", keep)
        self.assertIn("b", keep)

    def test_keeps_all_when_varying(self):
        df = pd.DataFrame({"a": [0.0, 1.0], "b": [2.0, 3.0]})
        keep = tm.variance_prune(["a", "b"], df)
        self.assertEqual(len(keep), 2)

    def test_fallback_when_all_constant(self):
        """train_model keeps cols[:] if every column has zero variance."""
        df = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]})
        keep = tm.variance_prune(["a", "b"], df)
        self.assertEqual(keep, ["a", "b"])


class TestSparsityPrune(unittest.TestCase):
    def test_drops_mostly_zero_column(self):
        df = pd.DataFrame({"dead": [0.0] * 999 + [1.0], "alive": [0.0, 1.0] * 500})
        keep = tm.sparsity_prune(["dead", "alive"], df, max_zero_frac=0.995)
        self.assertNotIn("dead", keep)
        self.assertIn("alive", keep)

    def test_disable_when_threshold_one(self):
        df = pd.DataFrame({"dead": [0.0] * 999 + [1.0]})
        keep = tm.sparsity_prune(["dead"], df, max_zero_frac=1.0)
        self.assertEqual(keep, ["dead"])


class TestEffectiveScalePosWeight(unittest.TestCase):
    def test_override(self):
        y = pd.Series([0, 0, 0, 1])
        self.assertEqual(tm.effective_scale_pos_weight(y, 2.5), 2.5)

    def test_auto_max1(self):
        y = pd.Series([0, 0, 1, 1])
        self.assertEqual(tm.effective_scale_pos_weight(y, None), 1.0)

    def test_auto_favors_minority_positive(self):
        y = pd.Series([0, 0, 0, 0, 1])
        self.assertEqual(tm.effective_scale_pos_weight(y, None), 4.0)

    def test_cap(self):
        y = pd.Series([0, 0, 0, 0, 1])
        self.assertEqual(tm.effective_scale_pos_weight(y, None, max_auto=2.0), 2.0)


class TestTimeSplit(unittest.TestCase):
    def test_train_before_test_after(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2020-01-02", "2020-01-03", "2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "ticker": ["A", "A", "A", "A", "A"],
                "x": [1, 2, 3, 4, 5],
            }
        )
        train, test, cutoff = tm.time_based_split(df, test_frac=0.4)
        self.assertTrue(train["Date"].max() < cutoff)
        self.assertTrue(test["Date"].min() >= cutoff)


if __name__ == "__main__":
    unittest.main()
