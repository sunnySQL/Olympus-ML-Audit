"""utils.expanding_splits — walk-forward date blocks."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.expanding_splits import expanding_splits  # noqa: E402


class TestExpandingSplits(unittest.TestCase):
    def test_three_folds_shapes(self):
        dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-20"))
        splits = expanding_splits(dates, min_train_days=5, n_folds=3)
        self.assertEqual(len(splits), 3)
        for train_d, test_d in splits:
            self.assertGreater(len(train_d), 0)
            self.assertGreater(len(test_d), 0)

    def test_too_few_dates_raises(self):
        dates = np.array([np.datetime64("2020-01-01")])
        with self.assertRaises(ValueError):
            expanding_splits(dates, min_train_days=10, n_folds=2)


if __name__ == "__main__":
    unittest.main()
