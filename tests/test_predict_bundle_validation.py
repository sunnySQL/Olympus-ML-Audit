"""predict_bundle raises on missing feature columns."""
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


class TestMissingColumns(unittest.TestCase):
    def test_classification_raises(self):
        m = MagicMock()
        m.predict_proba.return_value = np.array([[0.5, 0.5]])
        df = pd.DataFrame({"ticker": ["A"], "f1": [1.0]})
        bundle = {
            "task": "classification",
            "model_kind": "global",
            "model": m,
            "feature_names": ["f1", "missing_feat"],
        }
        with self.assertRaises(ValueError) as ctx:
            add_pred_prob(df, bundle)
        self.assertIn("missing_feat", str(ctx.exception))

    def test_regression_raises(self):
        m = MagicMock()
        m.predict.return_value = np.array([0.0])
        df = pd.DataFrame({"f1": [1.0]})
        bundle = {
            "task": "regression",
            "model_kind": "global",
            "model": m,
            "feature_names": ["f1", "f2"],
        }
        with self.assertRaises(ValueError):
            add_pred_prob(df, bundle)


if __name__ == "__main__":
    unittest.main()
