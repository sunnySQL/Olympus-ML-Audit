"""predict_bundle per-ticker paths."""
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


class TestPerTicker(unittest.TestCase):
    def test_classification_uses_ticker_model_when_present(self):
        m_global = MagicMock()
        m_global.predict_proba.return_value = np.array([[0.5, 0.5]])
        m_aapl = MagicMock()
        m_aapl.predict_proba.return_value = np.array([[0.1, 0.9]])
        df = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "f1": [0.0, 1.0]})
        bundle = {
            "task": "classification",
            "model_kind": "per_ticker",
            "model": m_global,
            "models_by_ticker": {"AAPL": m_aapl, "MSFT": None},
            "feature_names": ["f1"],
        }
        out = add_pred_prob(df, bundle)
        self.assertAlmostEqual(float(out.loc[0, "pred_prob"]), 0.9)
        self.assertAlmostEqual(float(out.loc[1, "pred_prob"]), 0.5)

    def test_regression_per_ticker(self):
        m_g = MagicMock()
        m_g.predict.return_value = np.array([0.0])
        m_a = MagicMock()
        m_a.predict.return_value = np.array([0.05])
        df = pd.DataFrame({"ticker": ["AAPL"], "f1": [1.0]})
        bundle = {
            "task": "regression",
            "model_kind": "per_ticker",
            "model": m_g,
            "models_by_ticker": {"AAPL": m_a},
            "feature_names": ["f1"],
            "return_prob_scale": 30.0,
        }
        out = add_pred_prob(df, bundle)
        self.assertAlmostEqual(float(out["pred_return"].iloc[0]), 0.05)


if __name__ == "__main__":
    unittest.main()
