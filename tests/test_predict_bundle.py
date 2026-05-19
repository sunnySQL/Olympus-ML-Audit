"""Tests for utils.predict_bundle (mocked models)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.predict_bundle import add_pred_prob  # noqa: E402


class TestAddPredProb(unittest.TestCase):
    def test_classification_global(self):
        m = MagicMock()
        m.predict_proba.return_value = np.array([[0.8, 0.2], [0.4, 0.6]])
        df = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "f1": [0.0, 1.0], "f2": [1.0, 0.0]})
        bundle = {
            "task": "classification",
            "model_kind": "global",
            "model": m,
            "feature_names": ["f1", "f2"],
        }
        out = add_pred_prob(df, bundle)
        np.testing.assert_allclose(out["pred_prob"].values, [0.2, 0.6])

    def test_regression_global(self):
        m = MagicMock()
        m.predict.return_value = np.array([0.01, -0.02])
        df = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "f1": [0.0, 1.0]})
        bundle = {
            "task": "regression",
            "model_kind": "global",
            "model": m,
            "feature_names": ["f1"],
            "return_prob_scale": 30.0,
        }
        out = add_pred_prob(df, bundle)
        self.assertIn("pred_return", out.columns)
        self.assertIn("pred_prob", out.columns)
        self.assertAlmostEqual(float(out["pred_return"].iloc[0]), 0.01)


if __name__ == "__main__":
    unittest.main()
