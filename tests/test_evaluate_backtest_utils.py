"""Pure functions from scripts/evaluate_backtest.py"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_backtest as eb  # noqa: E402


class TestComputeMetrics(unittest.TestCase):
    def test_empty_returns_zeros(self):
        m = eb.compute_metrics(pd.Series([], dtype=float))
        self.assertEqual(m["days"], 0)
        self.assertEqual(m["sharpe"], 0.0)

    def test_constant_positive(self):
        r = pd.Series([0.001] * 10)
        m = eb.compute_metrics(r)
        self.assertEqual(m["days"], 10)
        self.assertGreater(m["total_return"], 0)


class TestApplyPositions(unittest.TestCase):
    def test_long_short_flat(self):
        df = pd.DataFrame(
            {
                "pred_prob": [0.60, 0.40, 0.50],
                "target_return_1d": [0.02, -0.01, 0.03],
            }
        )
        out = eb.apply_positions(df, cost_bps=0.0, long_threshold=0.55, short_threshold=0.45, long_only=False)
        self.assertAlmostEqual(out["position_return"].iloc[0], 0.02)
        self.assertAlmostEqual(out["position_return"].iloc[1], 0.01)
        self.assertAlmostEqual(out["position_return"].iloc[2], 0.0)

    def test_uses_selected_return_column(self):
        df = pd.DataFrame(
            {
                "pred_prob": [0.60],
                "target_return_1d": [0.02],
                "_exec_return": [-0.01],
            }
        )
        out = eb.apply_positions(
            df,
            cost_bps=0.0,
            long_threshold=0.55,
            short_threshold=0.45,
            long_only=True,
            return_col="_exec_return",
        )
        self.assertAlmostEqual(out["position_return"].iloc[0], -0.01)


class TestExecutionReturns(unittest.TestCase):
    def test_next_open_to_close_derives_return(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                "ticker": ["AAPL", "AAPL"],
                "Open": [100.0, 110.0],
                "Close": [105.0, 121.0],
            }
        )
        out, col = eb.attach_execution_returns(df, "next_open_to_close")
        self.assertEqual(col, "_exec_return")
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[col].iloc[0], 0.10)

    def test_next_open_to_close_5d_derives_return(self):
        df = pd.DataFrame(
            {
                "Date": pd.bdate_range("2020-01-02", periods=6),
                "ticker": ["AAPL"] * 6,
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "Close": [100.0, 102.0, 103.0, 104.0, 105.0, 111.1],
            }
        )
        out, col = eb.attach_execution_returns(df, "next_open_to_close_5d")
        self.assertEqual(col, "_exec_return")
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[col].iloc[0], 0.10)


class TestTopFraction(unittest.TestCase):
    def test_invalid_fraction_raises(self):
        df = pd.DataFrame({"Date": [1], "pred_prob": [0.5], "target_return_1d": [0.0]})
        with self.assertRaises(ValueError):
            eb.build_top_fraction_long_backtest(df, 0.0, top_fraction=0.0)


class TestAttachSignals(unittest.TestCase):
    def test_long_only(self):
        df = pd.DataFrame({"pred_prob": [0.6, 0.4]})
        out = eb.attach_signals(df, long_th=0.55, short_th=0.45, long_only=True)
        self.assertTrue((out["pred_signal"].values == [1, -1]).all())


if __name__ == "__main__":
    unittest.main()
