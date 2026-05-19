"""sklearn_compat patches LogisticRegression from newer sklearn pickles."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.platt_calibration import PlattCalibratedBinaryClassifier  # noqa: E402
from utils.sklearn_compat import patch_sklearn_estimators  # noqa: E402


class TestSklearnCompat(unittest.TestCase):
    def test_adds_multi_class_when_missing(self):
        lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=10)
        lr.fit([[0.0], [1.0], [2.0]], [0, 0, 1])
        if hasattr(lr, "multi_class"):
            del lr.multi_class

        wrapper = PlattCalibratedBinaryClassifier(base=object())
        wrapper.calibrator_ = lr
        patch_sklearn_estimators(wrapper)

        self.assertEqual(lr.multi_class, "ovr")
        probs = lr.predict_proba([[0.5]])
        self.assertEqual(probs.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
