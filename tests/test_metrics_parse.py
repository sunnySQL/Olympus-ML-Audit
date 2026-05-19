"""utils.metrics_parse"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.metrics_parse import parse_train_stdout  # noqa: E402


class TestParseTrainStdout(unittest.TestCase):
    def test_full_block(self):
        text = """
Features after variance prune: 32 (dropped 13 )
Accuracy: 0.5504840940525588
ROC AUC: 0.5330353691534595
Log loss: 0.6881066533468446
Brier: 0.24748369896840647
"""
        m = parse_train_stdout(text)
        self.assertEqual(m["n_features"], 32)
        self.assertAlmostEqual(m["accuracy"], 0.5504840940525588)
        self.assertAlmostEqual(m["roc_auc"], 0.5330353691534595)

    def test_empty(self):
        self.assertEqual(parse_train_stdout(""), {})


if __name__ == "__main__":
    unittest.main()
