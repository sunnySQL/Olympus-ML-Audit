"""utils.live_score"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.live_score import (  # noqa: E402
    latest_row_per_ticker,
    score_latest_per_ticker,
    signal_from_thresholds,
)
from unittest.mock import MagicMock


class TestSignal(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(signal_from_thresholds(0.6, 0.55, 0.45), "long")
        self.assertEqual(signal_from_thresholds(0.4, 0.55, 0.45), "short")
        self.assertEqual(signal_from_thresholds(0.5, 0.55, 0.45), "flat")


class TestLatestRow(unittest.TestCase):
    def test_picks_max_date_per_ticker(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-02"]),
                "ticker": ["A", "A", "B"],
                "f1": [1.0, 2.0, 3.0],
            }
        )
        out = latest_row_per_ticker(df, ["f1"])
        self.assertEqual(len(out), 2)
        a = out[out["ticker"] == "A"]
        self.assertEqual(a["Date"].iloc[0].strftime("%Y-%m-%d"), "2020-01-03")


class TestScoreLatest(unittest.TestCase):
    def test_global_classify(self):
        m = MagicMock()
        m.predict_proba.side_effect = lambda X: np.array([[0.4, 0.6]] * len(X))
        bundle = {
            "feature_names": ["f1"],
            "model_kind": "global",
            "model": m,
            "task": "classification",
        }
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "ticker": ["X", "X"],
                "f1": [0.0, 1.0],
            }
        )
        out = score_latest_per_ticker(df, bundle)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out["pred_prob"].iloc[0]), 0.6)


if __name__ == "__main__":
    unittest.main()
