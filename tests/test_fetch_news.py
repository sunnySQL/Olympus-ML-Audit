"""Tests for fetch_news deduplication and helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_news as fn  # noqa: E402


class TestDedupeArticles(unittest.TestCase):
    def test_dedupe_by_url(self):
        rows = [
            {"url": "https://a.com/1", "title": "T", "date": "2025-01-01", "published_at": "", "ticker": "AAPL"},
            {"url": "https://a.com/1", "title": "T", "date": "2025-01-01", "published_at": "", "ticker": "AAPL"},
            {"url": "https://b.com/2", "title": "T", "date": "2025-01-01", "published_at": "", "ticker": "AAPL"},
        ]
        out = fn.dedupe_articles(rows)
        self.assertEqual(len(out), 2)

    def test_dedupe_keeps_different_tickers_same_url(self):
        rows = [
            {"url": "https://a.com/x", "title": "T", "date": "2025-01-01", "published_at": "", "ticker": "AAPL"},
            {"url": "https://a.com/x", "title": "T", "date": "2025-01-01", "published_at": "", "ticker": "MSFT"},
        ]
        out = fn.dedupe_articles(rows)
        self.assertEqual(len(out), 2)


class TestParseFmpStockItem(unittest.TestCase):
    def test_basic(self):
        row = fn._parse_fmp_stock_item(
            {
                "title": "Apple beats estimates",
                "publishedDate": "2025-06-15T14:00:00Z",
                "url": "https://example.com/n",
                "site": "TestSource",
            },
            "AAPL",
        )
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["ticker"], "AAPL")
        self.assertIn("2025-06-15", row["published_at"])

    def test_empty_returns_none(self):
        self.assertIsNone(fn._parse_fmp_stock_item({}, "SPY"))


class TestChunkDateRanges(unittest.TestCase):
    def test_yields_increasing_windows(self):
        from datetime import datetime, timezone

        end = datetime(2025, 4, 1, 12, 0, tzinfo=timezone.utc)
        chunks = list(fn._chunk_date_ranges(end, backfill_days=10, chunk_days=5))
        self.assertGreaterEqual(len(chunks), 2)
        for a, b in chunks:
            self.assertLessEqual(a, b)


if __name__ == "__main__":
    unittest.main()
