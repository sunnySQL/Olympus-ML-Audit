"""fetch_news save_articles merge behavior (isolated OUT_CSV)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_news as fn  # noqa: E402


def _row(url, title, day="2024-01-01"):
    return {
        "date": day,
        "published_at": f"{day}T12:00:00+00:00",
        "ticker": "AAPL",
        "source": "t",
        "title": title,
        "url": url,
        "sentiment": 0.1,
    }


class TestSaveArticlesMerge(unittest.TestCase):
    def setUp(self):
        self._orig_out = fn.OUT_CSV
        self.tmp = tempfile.TemporaryDirectory()
        fn.OUT_CSV = os.path.join(self.tmp.name, "news.csv")

    def tearDown(self):
        fn.OUT_CSV = self._orig_out
        self.tmp.cleanup()

    def test_merge_keeps_prior_rows(self):
        fn.save_articles([_row("http://a.com/1", "one")], merge=False)
        fn.save_articles([_row("http://b.com/2", "two")], merge=True)
        df = pd.read_csv(fn.OUT_CSV)
        self.assertEqual(len(df), 2)

    def test_dedupe_on_second_save(self):
        fn.save_articles([_row("http://a.com/x", "same")], merge=False)
        fn.save_articles([_row("http://a.com/x", "same")], merge=True)
        df = pd.read_csv(fn.OUT_CSV)
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
