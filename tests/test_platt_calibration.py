"""Platt calibrator on mocked base classifier."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.platt_calibration import PlattCalibratedBinaryClassifier  # noqa: E402


class TestPlatt(unittest.TestCase):
    def test_fit_and_predict_proba_shape(self):
        base = MagicMock()
        base.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        pc = PlattCalibratedBinaryClassifier(base)
        X = np.zeros((2, 3))
        y = np.array([0, 1])
        pc.fit_calibrator(X, y)
        out = pc.predict_proba(X)
        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(np.allclose(out.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
