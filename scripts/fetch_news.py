import argparse
import os
import re
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.universe import DEFAULT_UNIVERSE_PATH, load_universe

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_CSV = os.path.join(DATA_DIR, "news.csv")
ASSETS = load_universe(DEFAULT_UNIVERSE_PATH)

FM_API_KEY = os.getenv("FMP_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_API_KEY = os.getenv("ALPHA_API_KEY", "demo")  # default demo key works

analyzer = SentimentIntensityAnalyzer()


def sentiment_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    return analyzer.polarity_scores(text)["compound"]


def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def dedupe_articles(articles):
    seen = set()
    out = []
    for row in articles:
        url = (row.get("url") or "").strip().lower()
        title = (row.get("title") or "").strip()[:240]
        date_key = row.get("date") or ""
        pub = row.get("published_at") or ""
        ticker = (row.get("ticker") or "").strip().upper()
        key = (url, title, date_key, pub, ticker) if url else (title, date_key, pub, ticker)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def load_existing_articles():
    """Return prior rows so new fetches can merge (historical news for training)."""
    if not os.path.isfile(OUT_CSV):
        return []
    try:
        df = pd.read_csv(OUT_CSV)
    except Exception:
        return []
    if df.empty:
        return []
    return df.to_dict("records")


def save_articles(articles, merge=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    if merge:
        articles = load_existing_articles() + list(articles)
    articles = dedupe_articles(articles)
    fieldnames = ["date", "published_at", "ticker", "source", "title", "url", "sentiment"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in articles:
            row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(row)
    return len(articles)


def from_fmp(days=7):
    if not FM_API_KEY:
        return []

    collected = []
    url = "https://financialmodelingprep.com/stable/fmp-articles"

    try:
        for page in range(5):
            params = {
                "page": page,
                "limit": 100,
                "apikey": FM_API_KEY,
            }
            r = requests.get(url, params=params, timeout=20)
            if r.status_code != 200:
                print(f"DEBUG: FMP page {page}: status={r.status_code}")
                break

            data = r.json()
            articles = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []

            if not articles:
                print(f"DEBUG: FMP page {page}: no articles")
                break

            if page == 0 and articles:
                print(f"DEBUG: FMP first article keys: {list(articles[0].keys())}")

            print(f"DEBUG: FMP page {page}: got {len(articles)} articles")

            for item in articles:
                title = item.get("title", "")
                text = (item.get("text", "") or item.get("summary", "") or title).lower()
                raw_pub = (
                    item.get("publishedDate")
                    or item.get("date")
                    or item.get("published_at")
                    or item.get("publishDate")
                    or ""
                )
                date_str = ""
                published_at = ""
                if raw_pub:
                    raw_s = str(raw_pub).strip()
                    date_str = raw_s[:10]
                    try:
                        if "T" in raw_s or len(raw_s) > 10:
                            published_at = pd.to_datetime(raw_s, utc=True).isoformat()
                        else:
                            published_at = iso_utc(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).replace(hour=12))
                    except Exception:
                        published_at = iso_utc(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).replace(hour=12))
                url_item = item.get("url", "") or item.get("link", "")
                source = item.get("site", item.get("source", "FMP"))
                score = sentiment_score(title + " " + text)

                tickers_mentioned = set()
                for t in ASSETS:
                    if t.lower() in text or t.lower() in title.lower():
                        tickers_mentioned.add(t)

                if not tickers_mentioned:
                    if "gold" in text or "silver" in text:
                        tickers_mentioned.add("GLD")
                    elif "market" in text or "s&p" in text or "nasdaq" in text or "dow" in text:
                        tickers_mentioned.add("SPY")
                    else:
                        tickers_mentioned.add("SPY")

                for t in tickers_mentioned:
                    collected.append({
                        "date": date_str,
                        "published_at": published_at,
                        "ticker": t,
                        "source": source,
                        "title": title,
                        "url": url_item,
                        "sentiment": score,
                    })

        print(f"DEBUG: FMP collected {len(collected)} ticker-article pairs")

    except Exception as e:
        print(f"DEBUG: FMP fetch error: {e}")

    return collected


def _parse_fmp_stock_item(item: Any, ticker_fallback: str) -> Optional[dict]:
    title = item.get("title", "") or ""
    text = (item.get("text", "") or item.get("summary", "") or item.get("content", "") or title).lower()
    raw_pub = (
        item.get("publishedDate")
        or item.get("date")
        or item.get("published_at")
        or item.get("publishDate")
        or ""
    )
    date_str = ""
    published_at = ""
    if raw_pub:
        raw_s = str(raw_pub).strip()
        date_str = raw_s[:10]
        try:
            if "T" in raw_s or len(raw_s) > 10:
                published_at = pd.to_datetime(raw_s, utc=True).isoformat()
            else:
                published_at = iso_utc(
                    datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).replace(hour=12)
                )
        except Exception:
            try:
                published_at = iso_utc(
                    datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).replace(hour=12)
                )
            except Exception:
                published_at = ""
    sym = (item.get("symbol") or item.get("ticker") or ticker_fallback or "").strip().upper()
    if not sym:
        sym = ticker_fallback
    url_item = item.get("url", "") or item.get("link", "")
    source = item.get("site", item.get("source", "FMP-stock"))
    score = sentiment_score(title + " " + text)
    if not title and not text:
        return None
    return {
        "date": date_str,
        "published_at": published_at,
        "ticker": sym,
        "source": str(source),
        "title": title,
        "url": url_item,
        "sentiment": score,
    }


def from_fmp_stock_news(date_from: str, date_to: str, max_pages_per_symbol: int = 40):
    """
    FMP Search Stock News: per-symbol, optional from/to for historical coverage.
    Endpoint may require a paid plan; on 402/403 the caller should skip.
    """
    if not FM_API_KEY:
        return []

    url = "https://financialmodelingprep.com/stable/news/stock"
    collected = []

    for sym in ASSETS:
        for page in range(max_pages_per_symbol):
            params = {
                "symbols": sym,
                "from": date_from,
                "to": date_to,
                "page": page,
                "limit": 100,
                "apikey": FM_API_KEY,
            }
            try:
                r = requests.get(url, params=params, timeout=25)
            except Exception as e:
                print(f"DEBUG: FMP stock news {sym} page {page}: request error {e}")
                break
            if r.status_code in (402, 403, 429):
                print(f"DEBUG: FMP stock news {sym}: status={r.status_code} (plan limit or rate limit)")
                break
            if r.status_code != 200:
                print(f"DEBUG: FMP stock news {sym} page {page}: status={r.status_code}")
                break
            try:
                data = r.json()
            except Exception:
                break
            articles = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []
            if not articles:
                break
            for item in articles:
                row = _parse_fmp_stock_item(item, sym)
                if row:
                    collected.append(row)
            if len(articles) < 100:
                break

    print(f"DEBUG: FMP stock news ({date_from} .. {date_to}) collected {len(collected)} rows")
    return collected


def from_alpha_vantage():
    collected = []
    for t in ASSETS:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": t,
            "apikey": ALPHA_API_KEY,
            "limit": 20,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            print(f"DEBUG: AlphaVantage {t}: status={r.status_code}")
            if not r.ok:
                print(f"DEBUG: AV response: {r.text[:300]}")
                continue
            data = r.json()
            print(f"DEBUG: AV {t} response keys: {list(data.keys())}")
            feed = data.get("feed", [])
            print(f"DEBUG: AV {t} feed length: {len(feed)}")
            for item in feed:
                text = item.get("summary", "") or item.get("title", "")
                score = sentiment_score(text)
                tp = item.get("time_published", "")
                date_str = tp[:8] if len(tp) >= 8 else ""
                if len(date_str) == 8:
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                published_at = ""
                if len(tp) >= 15:
                    try:
                        published_at = pd.to_datetime(tp, format="%Y%m%dT%H%M%S", utc=True).isoformat()
                    except Exception:
                        published_at = iso_utc(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).replace(hour=12))
                else:
                    published_at = iso_utc(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).replace(hour=12))
                collected.append({
                    "date": date_str,
                    "published_at": published_at,
                    "ticker": t,
                    "source": "AlphaVantage",
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "sentiment": score,
                })
        except Exception as e:
            print(f"Failed fetching {t} from AlphaVantage: {e}")
    return collected


def from_yfinance():
    collected = []
    for t in ASSETS:
        try:
            tk = yf.Ticker(t)
            for item in tk.news:
                published = item.get("providerPublishTime")
                if not published:
                    continue
                dt = datetime.utcfromtimestamp(published).replace(tzinfo=timezone.utc)
                date = dt.strftime("%Y-%m-%d")
                text = " ".join(filter(None, [item.get("title", ""), item.get("summary", "")]))
                score = sentiment_score(text)
                collected.append({
                    "date": date,
                    "published_at": iso_utc(dt),
                    "ticker": t,
                    "source": item.get("provider", "yfinance"),
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "sentiment": score,
                })
        except Exception as e:
            print(f"Failed fetching yfinance news for {t}: {e}")
    return collected


def from_rss():
    import feedparser

    rss_urls = [
        "https://www.investing.com/rss/news.rss",
        "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
    ]

    collected = []
    for url in rss_urls:
        d = feedparser.parse(url)
        for e in d.entries[:100]:
            text = e.get("title", "") + " " + e.get("summary", "")
            score = sentiment_score(text)
            date = e.get("published", "")[:10]
            pub_struct = e.get("published_parsed") or e.get("updated_parsed")
            published_at = ""
            if pub_struct:
                try:
                    dt = datetime(*pub_struct[:6], tzinfo=timezone.utc)
                    published_at = iso_utc(dt)
                except Exception:
                    published_at = ""
            if not published_at and date:
                try:
                    published_at = iso_utc(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).replace(hour=12))
                except Exception:
                    pass
            for t in ASSETS:
                if re.search(rf"\b{t}\b", text, re.IGNORECASE):
                    collected.append({
                        "date": date,
                        "published_at": published_at,
                        "ticker": t,
                        "source": url,
                        "title": e.get("title", ""),
                        "url": e.get("link", ""),
                        "sentiment": score,
                    })
    return collected


def from_dummy_news():
    collected = []
    import random

    news_templates = {
        "AAPL": [
            "Apple reports strong iPhone 15 sales ahead of earnings",
            "Apple Vision Pro sales exceed analyst expectations",
            "AAPL downgrades guidance due to China market weakness",
            "Apple announces $110B stock buyback program",
        ],
        "MSFT": [
            "Microsoft raises cloud revenue guidance amid AI demand surge",
            "Microsoft-OpenAI partnership shows strong momentum in enterprise",
            "MSFT faces regulatory scrutiny on cloud market dominance",
            "Microsoft acquires AI startup for $10B",
        ],
        "NVDA": [
            "Nvidia beats Q4 earnings estimates, raises FY2025 guidance",
            "Nvidia AI chip demand remains robust",
            "NVDA faces supply chain headwinds",
            "Nvidia announces Blackwell GPU architecture",
        ],
        "SPY": [
            "S&P 500 reaches new all-time high on strong earnings",
            "Market correction seen as healthy for equities",
            "Fed signals potential rate cuts in 2025",
            "Inflation data pressures equities lower",
        ],
        "GLD": [
            "Gold rises on geopolitical tensions",
            "Gold demand strong amid currency uncertainty",
            "Fed rate cut expectations support gold prices",
            "Gold retreats as dollar strengthens",
        ],
        "SLV": [
            "Silver industrial demand supports prices",
            "Silver volatility expected amid tech slowdown",
            "Silver supply constraints support market",
            "Silver price pressured by risk-off sentiment",
        ],
    }

    today = datetime.now(timezone.utc).date()
    for i in range(10):
        date = (today - timedelta(days=i)).isoformat()
        for t in ASSETS:
            if random.random() > 0.4:
                title = random.choice(news_templates.get(t, [f"News about {t}"]))
                sentiment = random.uniform(-0.5, 0.5)
                h = random.randint(9, 16)
                m = random.randint(0, 59)
                pub_dt = datetime.strptime(date, "%Y-%m-%d").replace(hour=h, minute=m, tzinfo=timezone.utc)
                collected.append({
                    "date": date,
                    "published_at": iso_utc(pub_dt),
                    "ticker": t,
                    "source": "synthetic",
                    "title": title,
                    "url": "",
                    "sentiment": sentiment,
                })

    return collected


def _chunk_date_ranges(end_date: datetime, backfill_days: int, chunk_days: int = 45):
    """Yield (from_iso, to_iso) strings for FMP stock news (smaller windows = smaller payloads)."""
    end = end_date.date()
    start = end - timedelta(days=max(0, backfill_days - 1))
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur.isoformat(), nxt.isoformat()
        cur = nxt + timedelta(days=1)


def main():
    ap = argparse.ArgumentParser(description="Fetch news into data/news.csv (merge preserves history)")
    ap.add_argument("--universe", type=str, default=str(DEFAULT_UNIVERSE_PATH), help="CSV/text universe file")
    ap.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers; overrides --universe")
    ap.add_argument(
        "--no-merge",
        action="store_true",
        help="Overwrite news.csv instead of merging with existing rows",
    )
    ap.add_argument(
        "--fmp-stock-backfill-days",
        type=int,
        default=0,
        metavar="N",
        help="Also call FMP Search Stock News per ticker in N-day windows (needs API access; 0=skip)",
    )
    ap.add_argument(
        "--no-fmp-articles",
        action="store_true",
        help="Skip FMP fmp-articles broad feed",
    )
    args = ap.parse_args()
    global ASSETS
    ASSETS = load_universe(args.universe, args.tickers)
    if not ASSETS:
        raise SystemExit("Universe is empty")
    print(f"News universe ({len(ASSETS)}): {', '.join(ASSETS)}")
    merge = not args.no_merge

    articles = []

    if not args.no_fmp_articles:
        articles += from_fmp(days=7)

    if args.fmp_stock_backfill_days > 0 and FM_API_KEY:
        end_dt = datetime.now(timezone.utc)
        for d0, d1 in _chunk_date_ranges(end_dt, args.fmp_stock_backfill_days, chunk_days=45):
            articles += from_fmp_stock_news(d0, d1)

    if not articles:
        print("FMP failed or no rows; trying AlphaVantage fallback")
        articles += from_alpha_vantage()

    if not articles:
        print("AlphaVantage failed or no rows; trying yfinance fallback")
        articles += from_yfinance()

    if not articles:
        print("yfinance failed or no rows; trying RSS fallback")
        articles += from_rss()

    if not articles:
        print("RSS failed or no rows; using dummy news fallback")
        articles += from_dummy_news()

    if not articles:
        print("No news articles collected, aborting")
        return

    for row in articles:
        if not row.get("published_at") and row.get("date"):
            try:
                d = str(row["date"])[:10]
                row["published_at"] = iso_utc(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).replace(hour=12))
            except Exception:
                row["published_at"] = ""

    n = save_articles(articles, merge=merge)
    print(f"Saved {n} deduplicated rows to {OUT_CSV} (merge={'on' if merge else 'off'})")


if __name__ == "__main__":
    main()
