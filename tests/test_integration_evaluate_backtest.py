"""End-to-end: evaluate_backtest with temp features + mocked bundle (no project data/)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.integration_dummy import DummyBinaryClassifier  # noqa: E402


def _minimal_features_csv(path: Path) -> None:
    rows = []
    for i, d in enumerate(["2020-06-01", "2020-06-02"]):
        for t in ["AAPL", "MSFT", "SPY"]:
            rows.append(
                {
                    "Date": d,
                    "ticker": t,
                    "f1": 0.25,
                    "Open": 100.0 + i,
                    "Close": 101.0 + i,
                    "target_return_1d": 0.001,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


class TestEvaluateBacktestIntegration(unittest.TestCase):
    def test_writes_reports(self):
        bundle = {
            "feature_names": ["f1"],
            "model_kind": "global",
            "model": DummyBinaryClassifier(0.62),
            "train_cutoff_date": "2020-01-01",
            "task": "classification",
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            feat = tmp / "features.csv"
            mod = tmp / "model.pkl"
            rep = tmp / "reports"
            _minimal_features_csv(feat)
            joblib.dump(bundle, mod)

            env = {**os.environ, "OMP_NUM_THREADS": "1"}
            r = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_backtest.py"),
                    "--features",
                    str(feat),
                    "--model",
                    str(mod),
                    "--reports-dir",
                    str(rep),
                    "--long-only",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue((rep / "backtest_comparison.csv").is_file())
            cmp_df = pd.read_csv(rep / "backtest_comparison.csv")
            self.assertGreater(len(cmp_df), 0)


if __name__ == "__main__":
    unittest.main()
