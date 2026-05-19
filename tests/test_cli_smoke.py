"""Smoke: scripts respond to --help without importing heavy paths unnecessarily."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestCliHelp(unittest.TestCase):
    def test_train_model_help(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_model.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "OMP_NUM_THREADS": "1"},
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--features-csv", r.stdout)
        self.assertIn("--learning-rate", r.stdout)
        self.assertIn("--variance-min-std", r.stdout)
        self.assertIn("--max-zero-frac", r.stdout)

    def test_fetch_news_help(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "fetch_news.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--fmp-stock-backfill-days", r.stdout)
        self.assertIn("--universe", r.stdout)

    def test_fetch_price_data_help(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "fetch_price_data.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--tickers", r.stdout)

    def test_evaluate_backtest_help(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate_backtest.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "OMP_NUM_THREADS": "1"},
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--features", r.stdout)
        self.assertIn("--reports-dir", r.stdout)
        self.assertIn("--execution-price", r.stdout)

    def test_live_score_help(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "live_score.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "OMP_NUM_THREADS": "1"},
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--append-log", r.stdout)

    def test_benchmark_model_help(self):
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "benchmark_model.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "OMP_NUM_THREADS": "1"},
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--per-ticker", r.stdout)
        self.assertIn("--save-best", r.stdout)


if __name__ == "__main__":
    unittest.main()
