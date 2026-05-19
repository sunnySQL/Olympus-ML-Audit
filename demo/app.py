"""
Olympus — demo dashboard (Streamlit).
Run from project root: streamlit run demo/app.py
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
# Avoid XGBoost/OpenMP shared-memory failures on some macOS setups (empty signals if predict dies).
os.environ.setdefault("KMP_USE_SHM", "0")

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.live_score import score_latest_per_ticker
from utils.prediction_log import append_signals, load_prediction_log

DATA_DIR   = ROOT / "data"
MODEL_PATH = ROOT / "models" / "direction_model.pkl"
FEATURES_CSV = DATA_DIR / "features.csv"
REPORTS_DIR  = ROOT / "reports"
PREDICTION_LOG = DATA_DIR / "prediction_log.csv"
PAPER_TRADES   = DATA_DIR / "paper_trades.csv"

ACCENT_LONG  = "#4ade80"
ACCENT_SHORT = "#fb7185"
ACCENT_FLAT  = "#94a3b8"
ACCENT_CYAN  = "#22d3ee"

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }

/* Hide Streamlit's default running status entirely and show custom text */
[data-testid="stStatusWidget"] {
  visibility: hidden !important;
  position: relative !important;
}
[data-testid="stStatusWidget"]::after {
  visibility: visible;
  position: absolute;
  right: 0; top: 50%;
  transform: translateY(-50%);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  font-weight: 500;
  color: #22d3ee;
  animation: olympus-pulse 1.4s ease-in-out infinite;
  content: "loading...";
  white-space: nowrap;
}
@keyframes olympus-pulse {
  0%   { opacity: 0.4; }
  50%  { opacity: 1.0; }
  100% { opacity: 0.4; }
}

/* hero */
.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0c4a6e 100%);
  border-radius: 18px; padding: 1.8rem 2.2rem 1.6rem; margin-bottom: 1.2rem;
  border: 1px solid rgba(148,163,184,0.14); box-shadow: 0 24px 56px rgba(15,23,42,0.4);
}
.hero h1 { color:#f8fafc; margin:0.3rem 0 0.4rem; font-weight:700; font-size:2rem; letter-spacing:-0.03em; }
.hero p  { color:#94a3b8; margin:0; font-size:0.95rem; line-height:1.55; }
.hero-badge {
  font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:500;
  color:#67e8f9; background:rgba(34,211,238,0.12); border:1px solid rgba(34,211,238,0.3);
  padding:0.18rem 0.55rem; border-radius:999px; margin-right:0.4rem; display:inline-block;
}
.hero-badge-warn {
  font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:500;
  color:#fbbf24; background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.28);
  padding:0.18rem 0.55rem; border-radius:999px; margin-right:0.4rem; display:inline-block;
}

/* stat cards */
.stat-row { display:grid; grid-template-columns:repeat(5,1fr); gap:0.85rem; margin-bottom:1.2rem; }
.stat-card {
  background:rgba(15,23,42,0.55); border:1px solid rgba(148,163,184,0.12);
  border-radius:14px; padding:1.1rem 1.2rem;
  display:flex; flex-direction:column; justify-content:space-between;
  height:110px; box-sizing:border-box;
}
.stat-label { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; margin:0; line-height:1; height:1rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.stat-value { font-family:'JetBrains Mono',monospace; font-size:1.35rem; font-weight:500; color:#f1f5f9; line-height:1; margin:0; height:1.35rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.stat-sub   { font-size:0.68rem; color:#475569; margin:0; line-height:1; height:1rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

/* signal table */
.signal-table { width:100%; border-collapse:collapse; font-size:0.88rem; }
.signal-table th { text-align:left; font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:0.07em; color:#64748b; padding:0.5rem 0.75rem; border-bottom:1px solid rgba(148,163,184,0.12); }
.signal-table td { padding:0.5rem 0.75rem; border-bottom:1px solid rgba(148,163,184,0.07); vertical-align:middle; color:#e2e8f0; }
.signal-table tr:last-child td { border-bottom:none; }
.signal-table tr:hover td { background:rgba(148,163,184,0.05); }
.ticker-chip { font-family:'JetBrains Mono',monospace; font-weight:500; font-size:0.85rem; color:#f1f5f9; }
.badge { display:inline-flex; align-items:center; gap:0.3rem; font-size:0.78rem; font-weight:600; padding:0.22rem 0.65rem; border-radius:999px; }
.badge-long  { background:rgba(74,222,128,0.15);  color:#4ade80; border:1px solid rgba(74,222,128,0.3); }
.badge-short { background:rgba(251,113,133,0.15); color:#fb7185; border:1px solid rgba(251,113,133,0.3); }
.badge-flat  { background:rgba(148,163,184,0.10); color:#94a3b8; border:1px solid rgba(148,163,184,0.22); }
.conviction-pos { font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#4ade80; }
.conviction-neg { font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#fb7185; }
.conviction-nil { font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#475569; }
.prob-bar-wrap { display:flex; align-items:center; gap:0.55rem; }
.prob-bar-bg   { flex:1; height:5px; background:rgba(148,163,184,0.15); border-radius:999px; min-width:50px; }
.prob-bar-fill { height:100%; border-radius:999px; }
.prob-val { font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:#cbd5e1; white-space:nowrap; min-width:3.2rem; text-align:right; }

/* section heading */
.section-heading { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#475569; margin:1.2rem 0 0.6rem; padding-bottom:0.35rem; border-bottom:1px solid rgba(148,163,184,0.1); }

/* sidebar */
section[data-testid="stSidebar"] { background:#0c1220 !important; }
.sb-title { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#475569; margin:1.1rem 0 0.4rem; }
.dist-bar { height:4px; border-radius:999px; background:linear-gradient(to right, #fb7185 0%, #94a3b8 40%, #4ade80 100%); margin:0.4rem 0 0.25rem; position:relative; }
.dist-info { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#64748b; display:flex; justify-content:space-between; }

/* tab override — tighter */
button[data-baseweb="tab"] { font-size:0.85rem !important; }

/* glossary cards */
.glossary-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:0.65rem; margin:0.6rem 0 1rem; }
.glossary-card {
  background:rgba(15,23,42,0.45); border:1px solid rgba(148,163,184,0.10);
  border-radius:12px; padding:0.8rem 1rem;
}
.glossary-term { font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:600; color:#67e8f9; margin:0 0 0.25rem; }
.glossary-def  { font-size:0.75rem; color:#94a3b8; margin:0; line-height:1.45; }

/* preset pill row */
.preset-row { display:flex; gap:0.5rem; margin:0.35rem 0; }
.preset-chip {
  font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#64748b;
  background:rgba(148,163,184,0.08); border:1px solid rgba(148,163,184,0.15);
  border-radius:999px; padding:0.15rem 0.55rem;
}
.preset-active { color:#67e8f9; border-color:rgba(34,211,238,0.35); background:rgba(34,211,238,0.10); }

/* freshness badges */
.hero-badge-fresh {
  font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:500;
  color:#4ade80; background:rgba(74,222,128,0.12); border:1px solid rgba(74,222,128,0.3);
  padding:0.18rem 0.55rem; border-radius:999px; margin-right:0.4rem; display:inline-block;
}
.hero-badge-stale {
  font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:500;
  color:#f97316; background:rgba(249,115,22,0.12); border:1px solid rgba(249,115,22,0.3);
  padding:0.18rem 0.55rem; border-radius:999px; margin-right:0.4rem; display:inline-block;
}

/* feature detail cards */
.feat-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr)); gap:0.55rem; margin:0.5rem 0; }
.feat-card {
  background:rgba(15,23,42,0.5); border:1px solid rgba(148,163,184,0.10);
  border-radius:10px; padding:0.65rem 0.8rem;
}
.feat-name { font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#64748b; margin:0 0 0.15rem; }
.feat-val  { font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:500; color:#f1f5f9; margin:0; }

/* sentiment bar */
.sent-bar-wrap { display:flex; align-items:center; gap:0.4rem; }
.sent-bar-bg { flex:1; height:4px; background:rgba(148,163,184,0.15); border-radius:999px; min-width:40px; }
.sent-bar-fill { height:100%; border-radius:999px; }
.sent-val { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#cbd5e1; min-width:3rem; text-align:right; }

/* scorecard */
.sc-correct { color:#4ade80; font-weight:600; }
.sc-wrong   { color:#fb7185; font-weight:600; }

/* paper-trade P&L */
.pnl-pos { color:#4ade80; font-family:'JetBrains Mono',monospace; }
.pnl-neg { color:#fb7185; font-family:'JetBrains Mono',monospace; }

/* audit */
.audit-verdict {
  background:rgba(15,23,42,0.6); border:1px solid rgba(148,163,184,0.14);
  border-radius:12px; padding:1rem 1.2rem; margin:0.5rem 0 1rem;
}
.audit-verdict-title { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; margin:0 0 0.25rem; }
.audit-verdict-value { font-family:'JetBrains Mono',monospace; font-size:1.05rem; font-weight:600; color:#fbbf24; margin:0; }
.audit-pass { color:#4ade80; background:rgba(74,222,128,0.13); border:1px solid rgba(74,222,128,0.28); }
.audit-monitor { color:#fbbf24; background:rgba(251,191,36,0.12); border:1px solid rgba(251,191,36,0.28); }
.audit-missing { color:#cbd5e1; background:rgba(148,163,184,0.12); border:1px solid rgba(148,163,184,0.22); }
.audit-fail { color:#fb7185; background:rgba(251,113,133,0.13); border:1px solid rgba(251,113,133,0.28); }
</style>
"""


# ─────────────────────── data helpers ───────────────────────

@st.cache_resource(show_spinner=False)
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False, ttl=300)
def latest_feature_rows() -> pd.DataFrame | None:
    if not FEATURES_CSV.exists():
        return None
    return pd.read_csv(FEATURES_CSV, parse_dates=["Date"])


def load_price_history(ticker: str, max_rows: int = 504) -> pd.DataFrame | None:
    path = DATA_DIR / f"{ticker}_daily.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path, parse_dates=["Date"])
    return d.sort_values("Date").tail(max_rows)


def load_backtest_comparison() -> pd.DataFrame | None:
    p = REPORTS_DIR / "backtest_comparison.csv"
    return pd.read_csv(p) if p.exists() else None


def load_metrics_log() -> pd.DataFrame | None:
    p = REPORTS_DIR / "metrics_log.csv"
    try:
        if not p.exists():
            return None
        with open(p, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) > 1:
            body = rows[1:]
            full_cols = [
                "timestamp_utc", "model_kind", "train_cutoff", "n_features",
                "train_rows", "test_rows", "task", "target", "calibrated_global",
                "recency_weights", "accuracy", "balanced_accuracy", "f1",
                "precision", "recall", "majority_baseline", "roc_auc", "brier",
                "log_loss", "rmse", "pearson_r",
            ]
            if max(len(r) for r in body) >= len(full_cols):
                normalized = [r + [""] * (len(full_cols) - len(r)) for r in body]
                out = pd.DataFrame(normalized, columns=full_cols)
                for c in [
                    "n_features", "train_rows", "test_rows", "accuracy",
                    "balanced_accuracy", "f1", "precision", "recall",
                    "majority_baseline", "roc_auc", "brier", "log_loss",
                    "rmse", "pearson_r",
                ]:
                    out[c] = pd.to_numeric(out[c], errors="coerce")
                return out
        return pd.read_csv(p)
    except Exception:
        return None


def load_walk_forward() -> pd.DataFrame | None:
    p = REPORTS_DIR / "walk_forward.csv"
    return pd.read_csv(p) if p.exists() else None


def load_robustness_report() -> pd.DataFrame | None:
    p = REPORTS_DIR / "robustness_report.csv"
    return pd.read_csv(p) if p.exists() else None


def load_regime_analysis() -> pd.DataFrame | None:
    p = REPORTS_DIR / "regime_analysis.csv"
    return pd.read_csv(p) if p.exists() else None


def load_threshold_sweep() -> pd.DataFrame | None:
    p = REPORTS_DIR / "threshold_sweep.csv"
    return pd.read_csv(p) if p.exists() else None


def load_model_audit_summary() -> pd.DataFrame | None:
    p = REPORTS_DIR / "model_audit_summary.csv"
    return pd.read_csv(p) if p.exists() else None


def load_model_audit_meta() -> dict | None:
    p = REPORTS_DIR / "model_audit_summary.json"
    if not p.exists():
        return None
    try:
        import json

        return json.loads(p.read_text())
    except Exception:
        return None


def latest_metrics_for_bundle(ml: pd.DataFrame | None, bundle: dict | None) -> pd.Series | None:
    if ml is None or len(ml) == 0:
        return None
    target = str((bundle or {}).get("target_column") or "")
    if target and "target" in ml.columns:
        matched = ml[ml["target"].astype(str) == target]
        if len(matched):
            return matched.iloc[-1]
    return ml.iloc[-1]


def load_paper_trades() -> pd.DataFrame | None:
    if not PAPER_TRADES.exists():
        return None
    try:
        df = pd.read_csv(PAPER_TRADES)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def save_paper_trade(trade: dict) -> None:
    """Append a single trade row to the paper_trades CSV."""
    PAPER_TRADES.parent.mkdir(parents=True, exist_ok=True)
    file_exists = PAPER_TRADES.exists()
    cols = ["date", "ticker", "action", "shares", "price", "notes"]
    with open(PAPER_TRADES, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: trade.get(k, "") for k in cols})


def data_freshness(as_of_str: str) -> tuple[str, str, int]:
    """Return (badge_class, label, days_old)."""
    try:
        as_of = datetime.strptime(as_of_str, "%Y-%m-%d").date()
        delta = (date.today() - as_of).days
    except Exception:
        return "hero-badge-stale", "UNKNOWN AGE", -1
    if delta <= 1:
        return "hero-badge-fresh", f"FRESH ({as_of_str})", delta
    return "hero-badge-stale", f"STALE ({delta}d old)", delta


def signal_key(p: float, long_th: float, short_th: float) -> str:
    if p > long_th:
        return "long"
    if p < short_th:
        return "short"
    return "flat"


# ─────────────────────── html helpers ───────────────────────

def badge_html(key: str) -> str:
    labels = {"long": "▲ Long", "short": "▼ Short", "flat": "— Flat"}
    return f'<span class="badge badge-{key}">{labels[key]}</span>'


def prob_bar_html(p: float, key: str) -> str:
    colors = {"long": ACCENT_LONG, "short": ACCENT_SHORT, "flat": ACCENT_FLAT}
    color = colors.get(key, ACCENT_FLAT)
    pct = max(0.0, min(1.0, p)) * 100
    return (
        f'<div class="prob-bar-wrap">'
        f'<div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
        f'<span class="prob-val">{p:.1%}</span>'
        f'</div>'
    )


def conviction_html(p: float, key: str, long_th: float, short_th: float) -> str:
    if key == "long":
        delta = p - long_th
        return f'<span class="conviction-pos">+{delta:.1%}</span>'
    if key == "short":
        delta = short_th - p
        return f'<span class="conviction-neg">+{delta:.1%}</span>'
    return '<span class="conviction-nil">—</span>'


def signal_table_html(rows: list[dict], long_th: float, short_th: float) -> str:
    rows_sorted = sorted(rows, key=lambda r: r["p_up"], reverse=True)
    has_ret = any("pred_ret" in r for r in rows_sorted)
    ret_hdr = "<th>Pred ret</th>" if has_ret else ""
    html = (
        '<div style="overflow-x:auto">'
        '<table class="signal-table"><thead><tr>'
        '<th>Ticker</th><th>As of</th><th>Signal</th>'
        '<th>P(up)</th><th>Edge</th>'
        f'{ret_hdr}</tr></thead><tbody>'
    )
    for r in rows_sorted:
        key = signal_key(r["p_up"], long_th, short_th)
        ret_td = ""
        if has_ret and "pred_ret" in r:
            ret_td = f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:0.8rem">{r["pred_ret"]:+.4f}</td>'
        html += (
            f'<tr>'
            f'<td><span class="ticker-chip">{r["ticker"]}</span></td>'
            f'<td style="font-size:0.78rem;color:#64748b">{r["date"]}</td>'
            f'<td>{badge_html(key)}</td>'
            f'<td>{prob_bar_html(r["p_up"], key)}</td>'
            f'<td>{conviction_html(r["p_up"], key, long_th, short_th)}</td>'
            f'{ret_td}</tr>'
        )
    html += "</tbody></table></div>"
    return html


def audit_table_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    html = (
        '<div style="overflow-x:auto">'
        '<table class="signal-table"><thead><tr>'
        '<th>Area</th><th>Check</th><th>Value</th><th>Status</th><th>Evidence</th>'
        '</tr></thead><tbody>'
    )
    for _, r in df.iterrows():
        status = str(r.get("status", "missing")).lower()
        label = status.upper()
        html += (
            "<tr>"
            f"<td>{r.get('area', '')}</td>"
            f"<td>{r.get('check', '')}</td>"
            f"<td style=\"font-family:'JetBrains Mono',monospace;font-size:0.78rem\">{r.get('value', '')}</td>"
            f"<td><span class=\"badge audit-{status}\">{label}</span></td>"
            f"<td style=\"font-size:0.76rem;color:#94a3b8\">{r.get('evidence', '')}</td>"
            "</tr>"
        )
    html += "</tbody></table></div>"
    return html


# ─────────────────────── charts ───────────────────────

def candlestick_chart(
    hist: pd.DataFrame,
    ticker: str,
    signal: str = "flat",
    initial_days: int | None = 183,
) -> go.Figure:
    has_ohlc = all(c in hist.columns for c in ["Open", "High", "Low", "Close"])
    has_vol  = "Volume" in hist.columns

    row_count  = 2 if has_vol else 1
    row_heights = [0.74, 0.26] if has_vol else [1.0]
    specs = [[{"type": "candlestick"}]] + ([[{"type": "bar"}]] if has_vol else [])
    fig = make_subplots(
        rows=row_count, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=row_heights, specs=specs,
    )

    # ── candlestick / line ──
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=hist["Date"],
            open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
            increasing=dict(line=dict(color=ACCENT_LONG, width=1.2), fillcolor="rgba(74,222,128,0.6)"),
            decreasing=dict(line=dict(color=ACCENT_SHORT, width=1.2), fillcolor="rgba(251,113,133,0.6)"),
            name="OHLC", showlegend=False,
            hovertext=[
                f"O:{o:.2f}  H:{h:.2f}  L:{l:.2f}  C:{c:.2f}"
                for o, h, l, c in zip(hist["Open"], hist["High"], hist["Low"], hist["Close"])
            ],
            hoverinfo="text+x",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=hist["Date"], y=hist["Close"], mode="lines",
            line=dict(color=ACCENT_CYAN, width=2),
            fill="tozeroy", fillcolor="rgba(34,211,238,0.07)",
            hovertemplate="%{x|%b %d %Y}  Close: <b>%{y:.2f}</b><extra></extra>",
        ), row=1, col=1)

    # ── SMA overlays ──
    for window, color, dash in [(20, "#f59e0b", "solid"), (50, "#818cf8", "dot")]:
        sma = hist["Close"].rolling(window).mean()
        fig.add_trace(go.Scatter(
            x=hist["Date"], y=sma, mode="lines", name=f"SMA{window}",
            line=dict(color=color, width=1.2, dash=dash),
            hovertemplate=f"SMA{window}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

    # ── signal marker at latest candle ──
    last_row = hist.iloc[-1]
    marker_colors = {"long": ACCENT_LONG, "short": ACCENT_SHORT, "flat": ACCENT_FLAT}
    marker_symbols = {"long": "triangle-up", "short": "triangle-down", "flat": "circle"}
    marker_y = float(last_row["High"]) * 1.015 if has_ohlc else float(last_row["Close"]) * 1.01
    fig.add_trace(go.Scatter(
        x=[last_row["Date"]], y=[marker_y],
        mode="markers",
        marker=dict(
            symbol=marker_symbols[signal],
            size=12,
            color=marker_colors[signal],
            line=dict(width=1.5, color="#0f172a"),
        ),
        name=f"Signal: {signal}",
        hovertemplate=f"Model signal: <b>{signal.upper()}</b><extra></extra>",
    ), row=1, col=1)

    # ── volume ──
    if has_vol:
        vol_colors = [
            "rgba(74,222,128,0.5)" if (has_ohlc and r["Close"] >= r["Open"]) else "rgba(251,113,133,0.4)"
            for _, r in hist.iterrows()
        ]
        fig.add_trace(go.Bar(
            x=hist["Date"], y=hist["Volume"], marker_color=vol_colors,
            name="Volume", showlegend=False,
            hovertemplate="%{x|%b %d %Y}  Vol: <b>%{y:,.0f}</b><extra></extra>",
        ), row=2, col=1)
        fig.update_yaxes(row=2, showgrid=False, tickformat=".2s",
                         tickfont=dict(size=9, color="#475569"), fixedrange=True, side="right")

    # ── initial x window ──
    x_range = None
    if initial_days and len(hist):
        end_dt   = hist["Date"].max()
        start_dt = end_dt - pd.Timedelta(days=initial_days)
        x_range  = [str(start_dt.date()), str((end_dt + pd.Timedelta(days=2)).date())]

    x_axis = dict(
        showgrid=False, rangeslider=dict(visible=False),
        tickfont=dict(size=10, color="#64748b"),
        linecolor="rgba(148,163,184,0.15)", fixedrange=False,
    )
    if x_range:
        x_axis["range"] = x_range

    sig_color = marker_colors[signal]
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        margin=dict(l=12, r=60, t=52, b=8),
        height=500,
        dragmode="zoom",
        hovermode="x",
        uirevision=ticker,
        title=dict(
            text=(
                f"<b>{ticker}</b>"
                f"  <span style='font-size:11px;color:{sig_color}'>◆ {signal.upper()}</span>"
                f"  <span style='font-size:10px;color:#475569'>drag to zoom · double-click to reset</span>"
            ),
            font=dict(size=14, color="#e2e8f0"), x=0, y=0.97,
        ),
        legend=dict(
            orientation="h", x=0, y=1.06, xanchor="left",
            font=dict(size=10, color="#94a3b8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=x_axis,
        yaxis=dict(
            showgrid=True, gridcolor="rgba(148,163,184,0.1)",
            tickfont=dict(size=10, color="#64748b"),
            tickformat=",.2f", side="right",
            fixedrange=False, autorange=True,
        ),
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=11),
    )
    if has_vol:
        x_axis2 = {**x_axis}
        x_axis2.pop("range", None)
        if x_range:
            x_axis2["range"] = x_range
        fig.update_layout(xaxis2=x_axis2)

    return fig


def distribution_donut(counts: dict[str, int]) -> go.Figure:
    order = ["long", "short", "flat"]
    labels = [k for k in order if k in counts]
    values = [counts[k] for k in labels]
    colors = {"long": ACCENT_LONG, "short": ACCENT_SHORT, "flat": ACCENT_FLAT}
    total  = sum(values)
    fig = go.Figure(go.Pie(
        labels=[l.capitalize() for l in labels], values=values,
        hole=0.62, marker=dict(colors=[colors[l] for l in labels]),
        textinfo="percent", textfont=dict(size=11, family="JetBrains Mono"),
        hovertemplate="%{label}: %{value} tickers<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=16, b=0), height=220,
        showlegend=True,
        legend=dict(orientation="v", x=1.0, y=0.5, xanchor="left",
                    font=dict(size=11, color="#94a3b8")),
        annotations=[dict(
            text=f'<b>{total}</b><br><span style="font-size:10px">tickers</span>',
            x=0.5, y=0.5, font=dict(size=14, color="#e2e8f0"), showarrow=False,
        )],
    )
    return fig


def backtest_bar(bdf: pd.DataFrame) -> go.Figure:
    label_map = {
        "model_long_short": "Model L/S", "model_long_only": "Model Long",
        "baseline_ew_long_all": "EW Long All", "baseline_spy_long_only": "SPY",
        "baseline_random": "Random",
    }
    order = ["Model L/S", "Model Long", "EW Long All", "SPY"]
    color_map = {"Model L/S": ACCENT_CYAN, "Model Long": ACCENT_LONG,
                 "EW Long All": "#818cf8", "SPY": "#f59e0b"}

    bdf = bdf.copy()
    bdf["display"] = bdf["strategy"].map(label_map).fillna(bdf["strategy"])
    bdf = bdf[bdf["display"].isin(order)].drop_duplicates("display")

    rows = []
    for lbl in order:
        sub = bdf[bdf["display"] == lbl]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append({"label": lbl, "ret": float(r.get("total_return", 0)) * 100,
                     "sharpe": float(r.get("sharpe", 0)), "color": color_map.get(lbl, "#64748b")})
    if not rows:
        return go.Figure()

    lbls    = [r["label"]  for r in rows]
    returns = [r["ret"]    for r in rows]
    sharpes = [r["sharpe"] for r in rows]
    clrs    = [r["color"]  for r in rows]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Total Return (%)", "Sharpe Ratio"],
                        horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=lbls, y=returns, marker_color=clrs, showlegend=False,
                         text=[f"{v:.1f}%" for v in returns], textposition="outside",
                         textfont=dict(size=10, family="JetBrains Mono")), row=1, col=1)
    fig.add_trace(go.Bar(x=lbls, y=sharpes, marker_color=clrs, showlegend=False,
                         text=[f"{v:.2f}" for v in sharpes], textposition="outside",
                         textfont=dict(size=10, family="JetBrains Mono")), row=1, col=2)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.3)", height=280,
        margin=dict(l=8, r=8, t=40, b=8),
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=10),
    )
    for ax in ["xaxis", "xaxis2"]:
        fig.update_layout(**{ax: dict(showgrid=False, tickfont=dict(size=10))})
    for ax in ["yaxis", "yaxis2"]:
        fig.update_layout(**{ax: dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)",
                                      zeroline=True, zerolinecolor="rgba(148,163,184,0.25)")})
    for ann in fig.layout.annotations:
        ann.font.size = 11; ann.font.color = "#64748b"
    return fig


def importance_chart(imp: dict) -> go.Figure:
    s = pd.Series(imp).sort_values(ascending=True).tail(15)
    colors = [ACCENT_CYAN if v == s.max() else "rgba(34,211,238,0.5)" for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h", marker_color=colors,
        text=[f"{v:.4f}" for v in s.values], textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono"),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.3)", height=420,
        margin=dict(l=8, r=16, t=20, b=8),
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(showgrid=False),
    )
    return fig


def metrics_history_chart(ml: pd.DataFrame) -> go.Figure:
    ml = ml.copy()
    if "timestamp_utc" in ml.columns:
        ml["ts"] = pd.to_datetime(ml["timestamp_utc"], utc=True, errors="coerce")
        ml = ml.dropna(subset=["ts"]).sort_values("ts")
    else:
        ml["ts"] = range(len(ml))

    fig = go.Figure()
    series = [
        ("roc_auc",           "ROC AUC",           ACCENT_CYAN,  "solid"),
        ("accuracy",          "Accuracy",           ACCENT_LONG,  "dot"),
        ("balanced_accuracy", "Balanced Acc",       "#f59e0b",    "dash"),
    ]
    for col, name, color, dash in series:
        if col not in ml.columns:
            continue
        sub = ml.dropna(subset=[col])
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["ts"], y=sub[col].astype(float),
            mode="lines+markers", name=name,
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=5, color=color),
            hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>",
        ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(148,163,184,0.3)",
                  annotation_text="0.5 baseline", annotation_font_size=10)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.3)", height=320,
        margin=dict(l=8, r=8, t=32, b=8),
        legend=dict(orientation="h", y=1.08, font=dict(size=11, color="#94a3b8")),
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#64748b")),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)",
                   tickfont=dict(size=10, color="#64748b"), range=[0.4, 0.75]),
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=11),
    )
    return fig


def threshold_sweep_chart(ts: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Sharpe Ratio by Threshold", "Hit Rate by Threshold"],
        horizontal_spacing=0.12,
    )
    if "long_threshold" in ts.columns and "sharpe" in ts.columns:
        fig.add_trace(go.Scatter(
            x=ts["long_threshold"], y=ts["sharpe"],
            mode="lines+markers", name="Sharpe",
            line=dict(color=ACCENT_CYAN, width=2),
            marker=dict(size=4, color=ACCENT_CYAN),
            hovertemplate="Threshold: %{x:.2f}<br>Sharpe: %{y:.3f}<extra></extra>",
        ), row=1, col=1)
    if "long_threshold" in ts.columns and "hit_rate_when_long" in ts.columns:
        fig.add_trace(go.Scatter(
            x=ts["long_threshold"], y=ts["hit_rate_when_long"],
            mode="lines+markers", name="Hit Rate",
            line=dict(color=ACCENT_LONG, width=2),
            marker=dict(size=4, color=ACCENT_LONG),
            hovertemplate="Threshold: %{x:.2f}<br>Hit Rate: %{y:.3f}<extra></extra>",
        ), row=1, col=2)
        fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(148,163,184,0.3)",
                      row=1, col=2)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.3)", height=280,
        margin=dict(l=8, r=8, t=40, b=8), showlegend=False,
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=10),
    )
    for ax in ["xaxis", "xaxis2"]:
        fig.update_layout(**{ax: dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)",
                                      tickfont=dict(size=10))})
    for ax in ["yaxis", "yaxis2"]:
        fig.update_layout(**{ax: dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)")})
    for ann in fig.layout.annotations:
        ann.font.size = 11
        ann.font.color = "#64748b"
    return fig


def signal_history_heatmap(plog: pd.DataFrame, ticker: str) -> go.Figure | None:
    sub = plog[plog["ticker"] == ticker].copy()
    if sub.empty:
        return None
    sub["dt"] = pd.to_datetime(sub["scored_date"], errors="coerce")
    sub = sub.dropna(subset=["dt"]).sort_values("dt")
    sub["dow"] = sub["dt"].dt.dayofweek
    sub["week"] = sub["dt"].dt.isocalendar().week.astype(int)
    sub["year"] = sub["dt"].dt.year
    sub["week_label"] = sub["year"].astype(str) + "-W" + sub["week"].astype(str).str.zfill(2)

    signal_map = {"long": 1, "flat": 0, "short": -1}
    sub["sig_num"] = sub["signal"].map(signal_map).fillna(0).astype(int)

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    week_labels = sub["week_label"].unique().tolist()

    z = []
    for d in range(7):
        row = []
        for wl in week_labels:
            match = sub[(sub["dow"] == d) & (sub["week_label"] == wl)]
            row.append(int(match["sig_num"].iloc[0]) if len(match) else None)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=week_labels, y=dow_labels[:7],
        colorscale=[
            [0.0, ACCENT_SHORT], [0.5, ACCENT_FLAT], [1.0, ACCENT_LONG],
        ],
        zmin=-1, zmax=1, showscale=False,
        hovertemplate="Week: %{x}<br>Day: %{y}<br>Signal: %{z}<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.3)", height=200,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(showgrid=False, tickfont=dict(size=8, color="#64748b"), side="top"),
        yaxis=dict(showgrid=False, tickfont=dict(size=9, color="#64748b"), autorange="reversed"),
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=10),
    )
    return fig


def equity_curve_chart(trades_df: pd.DataFrame) -> go.Figure | None:
    """Build a cumulative P&L line from closed trades."""
    closed = trades_df[trades_df["action"].str.lower() == "sell"].copy()
    if closed.empty:
        return None
    closed["date"] = pd.to_datetime(closed["date"], errors="coerce")
    closed = closed.dropna(subset=["date"]).sort_values("date")
    closed["pnl"] = pd.to_numeric(closed.get("pnl", 0), errors="coerce").fillna(0)
    closed["cum_pnl"] = closed["pnl"].cumsum()

    fig = go.Figure(go.Scatter(
        x=closed["date"], y=closed["cum_pnl"],
        mode="lines+markers",
        line=dict(color=ACCENT_CYAN, width=2),
        marker=dict(size=4),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.07)",
        hovertemplate="%{x|%b %d}<br>Cumul P&L: $%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.3)", height=260,
        margin=dict(l=8, r=8, t=20, b=8),
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#64748b")),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)",
                   tickfont=dict(size=10, color="#64748b"), tickprefix="$"),
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=10),
    )
    return fig


# ─────────────────────── pipeline runner ───────────────────────

def _pipeline_python() -> str:
    """
    Python used for **Run pipeline** subprocess.

    Prefer `project/.venv` so scripts see the same packages as `requirements.txt`, even when
    Streamlit was started with another interpreter (e.g. conda on PATH or the IDE default).

    IMPORTANT: do NOT resolve() — the venv python is typically a symlink to the base interpreter,
    and resolving it would bypass the venv's site-packages entirely.
    """
    if os.name == "nt":
        win = ROOT / ".venv" / "Scripts" / "python.exe"
        if win.exists():
            return str(win)
    nix = ROOT / ".venv" / "bin" / "python"
    if nix.exists():
        return str(nix)
    return sys.executable


def run_pipeline_ui(fetch_prices: bool, fetch_news: bool, robust: bool, skip_train: bool) -> None:
    """Run run_pipeline.py in a subprocess and stream output into the app."""
    st.markdown(
        '<style>[data-testid="stStatusWidget"]::after{content:"scoring..."!important;}</style>',
        unsafe_allow_html=True,
    )
    py = _pipeline_python()
    cmd = [py, str(ROOT / "scripts" / "run_pipeline.py")]
    if fetch_prices:
        cmd.append("--fetch-prices")
    if fetch_news:
        cmd.append("--fetch-news")
    if robust:
        cmd.append("--robust")
    if skip_train:
        cmd.append("--skip-train")
        cmd.append("--skip-backtest")

    st.markdown(
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;color:#64748b;margin-bottom:0.5rem">'
        f'<span style="opacity:0.85">Python:</span> <code style="font-size:0.72rem">{py}</code><br/>'
        f'$ {" ".join(cmd[1:])}</p>',
        unsafe_allow_html=True,
    )

    output_box  = st.empty()
    status_box  = st.empty()
    lines: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "OMP_NUM_THREADS": "1", "KMP_USE_SHM": "0"},
        )
        for line in proc.stdout:          # type: ignore[union-attr]
            lines.append(line.rstrip())
            output_box.code("\n".join(lines[-40:]), language="bash")
        proc.wait()
    except Exception as exc:
        status_box.error(f"Failed to start pipeline: {exc}")
        return

    if proc.returncode == 0:
        status_box.success("Pipeline finished — clearing cache and reloading…")
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    else:
        status_box.error(f"Pipeline exited with code {proc.returncode}. See output above.")


# ─────────────────────── sidebar ───────────────────────

def build_sidebar(bundle, p_up_arr: np.ndarray):
    """Build sidebar. p_up_arr is the array of current model probabilities (may be empty)."""
    with st.sidebar:
        st.markdown("## ◈ Olympus")
        st.caption("Model-audit dashboard · not financial advice")

        # ── conviction preset ──
        PRESETS = {
            "Conservative": {"long": 0.58, "short": 0.42, "desc": "Fewer trades, higher confidence required"},
            "Balanced":     {"long": 0.53, "short": 0.45, "desc": "Middle threshold preset for inspection"},
            "Aggressive":   {"long": 0.50, "short": 0.48, "desc": "More trades, lower bar to act"},
        }

        st.markdown('<p class="sb-title">Conviction level</p>', unsafe_allow_html=True)
        preset = st.radio(
            "How aggressive should signals be?",
            list(PRESETS.keys()),
            index=1,
            label_visibility="collapsed",
            horizontal=True,
        )

        long_th = PRESETS[preset]["long"]
        short_th = PRESETS[preset]["short"]
        st.caption(PRESETS[preset]["desc"])

        with st.expander("Custom thresholds", expanded=False):
            long_th = st.slider("Buy if confidence above", 0.50, 0.80, long_th, 0.01)
            short_th = st.slider("Avoid/short if below", 0.20, 0.50, short_th, 0.01)
            if short_th >= long_th:
                st.warning("Short threshold must be below long threshold.")

        # ── model info ──
        st.markdown('<p class="sb-title">Model</p>', unsafe_allow_html=True)
        if bundle:
            for k, v in {
                "File":        MODEL_PATH.name,
                "Features":    bundle.get("feature_set_version", "?"),
                "Mode":        bundle.get("model_kind", "global").replace("_", " "),
                "Task":        bundle.get("task", "classification"),
                "Train cutoff": bundle.get("train_cutoff_date", "?"),
            }.items():
                st.caption(f"**{k}:** `{v}`")
            xe = bundle.get("xgb_extra")
            if xe:
                with st.expander("XGB params"):
                    for k, v in xe.items():
                        st.caption(f"`{k}` = `{v}`")
        else:
            st.warning("No model loaded.")

        # ── last train metrics ──
        ml = load_metrics_log()
        last_metrics = latest_metrics_for_bundle(ml, bundle)
        if last_metrics is not None:
            st.markdown('<p class="sb-title">Last train run</p>', unsafe_allow_html=True)
            for col, label in [("accuracy", "Accuracy"), ("roc_auc", "ROC AUC"),
                                ("balanced_accuracy", "Bal. acc.")]:
                val = last_metrics.get(col)
                if val is not None and pd.notna(val):
                    st.caption(f"**{label}:** `{float(val):.4f}`")

        # ── pipeline runner ──
        st.markdown('<p class="sb-title">Refresh & retrain</p>', unsafe_allow_html=True)
        fetch_prices = st.checkbox("Fetch latest prices",  value=True)
        fetch_news   = st.checkbox("Fetch latest news",    value=True)
        robust       = st.checkbox("--robust preset",      value=True)
        skip_train   = st.checkbox("Data only (skip train + backtest)", value=False)

        run_clicked = st.button(
            "▶ Run pipeline",
            use_container_width=True,
            type="primary",
        )

        if run_clicked:
            st.session_state["pipeline_running"] = True

        if st.session_state.get("pipeline_running"):
            st.session_state["pipeline_running"] = False
            run_pipeline_ui(fetch_prices, fetch_news, robust, skip_train)

        st.caption(
            "Presets update the board instantly. "
            "Use **▶ Run pipeline** to pull fresh data + retrain."
        )

    return long_th, short_th


# ─────────────────────── main ───────────────────────

def main():
    st.set_page_config(
        page_title="Olympus", page_icon="◈",
        layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown(STYLES, unsafe_allow_html=True)

    bundle = load_bundle()
    df_raw = latest_feature_rows()

    # ── score FIRST so sidebar can show distribution ──
    rows: list[dict] = []
    p_up_arr = np.array([], dtype=float)
    is_reg = False
    score_error: str | None = None

    if bundle is not None and df_raw is not None:
        feature_names = bundle.get("feature_names") or []
        missing = [f for f in feature_names if f not in df_raw.columns]
        if not missing:
            try:
                with st.spinner("Scoring…"):
                    latest = score_latest_per_ticker(df_raw, bundle)
                latest["p_up"] = latest["pred_prob"].values
                is_reg = bundle.get("task") == "regression"
                for _, r in latest.iterrows():
                    row = {
                        "ticker": str(r["ticker"]),
                        "date":   r["Date"].strftime("%Y-%m-%d") if pd.notna(r["Date"]) else "—",
                        "p_up":   float(r["p_up"]),
                    }
                    if is_reg and "pred_return" in latest.columns:
                        row["pred_ret"] = float(r["pred_return"])
                    rows.append(row)
                p_up_arr = np.array([r["p_up"] for r in rows])
            except Exception as e:
                score_error = f"{type(e).__name__}: {e}"

    long_th, short_th = build_sidebar(bundle, p_up_arr)

    # ── persist signals for scorecard / history ──
    if rows:
        sig_rows = [
            {**r, "signal": signal_key(r["p_up"], long_th, short_th)}
            for r in rows
        ]
        try:
            append_signals(sig_rows, long_th, short_th, PREDICTION_LOG)
        except Exception:
            pass

    # ── counts ──
    counts: dict[str, int] = {"long": 0, "short": 0, "flat": 0}
    for r in rows:
        counts[signal_key(r["p_up"], long_th, short_th)] += 1

    # ── hero ──
    is_ready = bundle is not None and df_raw is not None
    refreshed = datetime.now(timezone.utc).strftime("%H:%M UTC")
    as_of_max = max((r["date"] for r in rows), default="—")
    fresh_cls, fresh_label, _fresh_days = data_freshness(as_of_max)
    st.markdown(
        f'<div class="hero">'
        f'<span class="hero-badge">{"MODEL READY" if is_ready else "NO MODEL"}</span>'
        f'<span class="{fresh_cls}">{fresh_label}</span>'
        f'<span class="hero-badge-warn">NOT FINANCIAL ADVICE</span>'
        f'<h1>Olympus</h1>'
        f'<p>Model-audit dashboard for an experimental XGBoost market-signal pipeline. '
        f'The current signal is evaluated against realistic baselines and may be rejected for deployment. '
        f'Scores use the last complete feature row per ticker, not a live feed. '
        f'Page loaded {refreshed}.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if bundle is None:
        st.error(f"No model at `{MODEL_PATH}`. Run `python scripts/train_model.py` first.")
        st.stop()
    if df_raw is None:
        st.error(f"No `{FEATURES_CSV}`. Run `python scripts/build_features.py` first.")
        st.stop()

    feature_names = bundle.get("feature_names") or []
    missing = [f for f in feature_names if f not in df_raw.columns]
    if missing:
        st.error(f"features.csv is missing columns: {missing[:5]}…")
        st.stop()

    if score_error:
        st.error(
            "**Scoring failed** (signals and charts stay empty until this is fixed):\n\n"
            f"`{score_error}`\n\n"
            "Common fixes: install matching `xgboost` in the same environment as Streamlit; "
            "set `OMP_NUM_THREADS=1` and `KMP_USE_SHM=0`; rerun from project root after "
            "`build_features.py` + `train_model.py`."
        )

    # ── key terms glossary ──
    with st.expander("📖  Key terms — what do these numbers mean?", expanded=False):
        st.markdown(
            '<div class="glossary-grid">'
            '<div class="glossary-card">'
            '  <p class="glossary-term">P(up)</p>'
            '  <p class="glossary-def">The model\'s predicted probability for its stored target label. Check the Model Health tab for the exact target column.</p>'
            '</div>'
            '<div class="glossary-card">'
            '  <p class="glossary-term">Signal</p>'
            '  <p class="glossary-def"><b>Long</b> — P(up) is above the buy threshold; model favors the stock.<br/>'
            '  <b>Short</b> — P(up) is below the avoid threshold; model sees weakness.<br/>'
            '  <b>Flat</b> — in between; no strong opinion.</p>'
            '</div>'
            '<div class="glossary-card">'
            '  <p class="glossary-term">Edge</p>'
            '  <p class="glossary-def">How far P(up) is past the threshold. Bigger edge = stronger conviction. Think of it as the model\'s confidence margin.</p>'
            '</div>'
            '<div class="glossary-card">'
            '  <p class="glossary-term">ROC AUC</p>'
            '  <p class="glossary-def">Area Under the ROC Curve (0.5 = random, 1.0 = perfect). Measures how well the model separates winners from losers.</p>'
            '</div>'
            '<div class="glossary-card">'
            '  <p class="glossary-term">Accuracy</p>'
            '  <p class="glossary-def">% of rows where the direction label was correct. Compare it to the majority-class baseline before trusting it.</p>'
            '</div>'
            '<div class="glossary-card">'
            '  <p class="glossary-term">Sharpe Ratio</p>'
            '  <p class="glossary-def">Risk-adjusted return: higher is better. Above 1.0 is good; above 2.0 is excellent. Compares return earned to the volatility experienced.</p>'
            '</div>'
            '<div class="glossary-card">'
            '  <p class="glossary-term">Max Drawdown</p>'
            '  <p class="glossary-def">Largest peak-to-trough portfolio drop during the backtest. Shows the worst-case pain you\'d have endured.</p>'
            '</div>'
            '<div class="glossary-card">'
            '  <p class="glossary-term">Conviction Level</p>'
            '  <p class="glossary-def"><b>Conservative</b> — only the strongest signals (fewer trades).<br/>'
            '  <b>Balanced</b> — middle preset for inspecting scores.<br/>'
            '  <b>Aggressive</b> — lower threshold, more signals, higher false-positive risk.</p>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── KPI cards ──
    ml   = load_metrics_log()
    last_auc = "—"
    last_metrics = latest_metrics_for_bundle(ml, bundle)
    if last_metrics is not None:
        val = last_metrics.get("roc_auc")
        if val is not None and pd.notna(val):
            last_auc = f"{float(val):.4f}"

    p_mean_str = f"{float(p_up_arr.mean()):.3f}" if len(p_up_arr) else "—"
    kpi_data = [
        ("Tickers",      str(len(rows)),          f"as of {as_of_max}"),
        ("Long signals", str(counts["long"]),      f"P(up) > {long_th:.2f}"),
        ("Short signals",str(counts["short"]),     f"P(up) < {short_th:.2f}"),
        ("Avg P(up)",    p_mean_str,               "across tickers"),
        ("ROC AUC",      last_auc,                 "last train run"),
    ]
    cards = '<div class="stat-row">'
    for label, val, sub in kpi_data:
        cards += (f'<div class="stat-card"><p class="stat-label">{label}</p>'
                  f'<p class="stat-value">{val}</p><p class="stat-sub">{sub}</p></div>')
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)

    # ── tabs ──
    tab_audit, tab_sig, tab_chart, tab_health, tab_bt, tab_paper = st.tabs(
        ["✅  Audit", "📊  Signals", "📈  Chart", "🧪  Model Health", "📋  Backtest", "📝  Paper Trading"]
    )

    # ════════ TAB 0: Audit ════════
    with tab_audit:
        st.markdown('<p class="section-heading">Deployment audit</p>', unsafe_allow_html=True)
        st.caption(
            "This is the audit centerpiece: the system summarizes whether the current model "
            "survives leakage-aware evaluation, realistic baselines, ranking-alpha checks, and walk-forward tests."
        )
        audit_df = load_model_audit_summary()
        audit_meta = load_model_audit_meta()

        if audit_meta:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="audit-verdict"><p class="audit-verdict-title">Deployment verdict</p>'
                    f'<p class="audit-verdict-value">{audit_meta.get("deploy_verdict", "UNKNOWN")}</p></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="audit-verdict"><p class="audit-verdict-title">Portfolio verdict</p>'
                    f'<p class="audit-verdict-value">{audit_meta.get("project_verdict", "UNKNOWN")}</p></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="audit-verdict"><p class="audit-verdict-title">Target</p>'
                    f'<p class="audit-verdict-value">{audit_meta.get("target_column", "UNKNOWN")}</p></div>',
                    unsafe_allow_html=True,
                )

        if audit_df is not None and len(audit_df):
            status_counts = audit_df["status"].value_counts().to_dict() if "status" in audit_df.columns else {}
            st.caption(
                f"Checks: {len(audit_df)} total · "
                f"{status_counts.get('pass', 0)} pass · "
                f"{status_counts.get('monitor', 0)} monitor · "
                f"{status_counts.get('fail', 0)} fail"
            )
            st.markdown(audit_table_html(audit_df), unsafe_allow_html=True)

            md_path = ROOT / "portfolio" / "olympus_model_audit_report.md"
            html_path = ROOT / "portfolio" / "olympus_model_audit_report.html"
            dl_cols = st.columns([1, 1, 3])
            if md_path.exists():
                dl_cols[0].download_button(
                    "Download Markdown",
                    data=md_path.read_text(),
                    file_name=md_path.name,
                    mime="text/markdown",
                    use_container_width=True,
                )
            if html_path.exists():
                dl_cols[1].download_button(
                    "Download HTML",
                    data=html_path.read_text(),
                    file_name=html_path.name,
                    mime="text/html",
                    use_container_width=True,
                )
        else:
            st.info("No audit report yet. Run `python scripts/generate_model_audit_report.py`.")

    # ════════ TAB 1: Signals ════════
    with tab_sig:
        # ── Yesterday's scorecard ──
        plog = load_prediction_log(PREDICTION_LOG)
        if plog is not None and df_raw is not None and "target_return_1d" in df_raw.columns:
            scored_dates = sorted(plog["scored_date"].dropna().unique())
            if len(scored_dates) >= 2:
                yesterday_date = scored_dates[-2]
                yest_preds = plog[plog["scored_date"] == yesterday_date].copy()

                feat_dated = df_raw[["ticker", "Date", "target_return_1d", "target_direction"]].copy()
                feat_dated["date_str"] = feat_dated["Date"].dt.strftime("%Y-%m-%d")
                actuals = feat_dated[feat_dated["date_str"] == yesterday_date]

                if not yest_preds.empty and not actuals.empty:
                    merged = yest_preds.merge(
                        actuals[["ticker", "target_return_1d", "target_direction"]],
                        on="ticker", how="inner",
                    )
                    if not merged.empty:
                        merged["pred_dir"] = (merged["p_up"].astype(float) >= 0.5).astype(int)
                        merged["actual_dir"] = merged["target_direction"].astype(int)
                        merged["correct"] = merged["pred_dir"] == merged["actual_dir"]
                        n_correct = int(merged["correct"].sum())
                        n_total = len(merged)
                        hit_pct = n_correct / n_total * 100 if n_total else 0

                        with st.expander(
                            f"Yesterday's scorecard ({yesterday_date}) — "
                            f"{n_correct}/{n_total} correct ({hit_pct:.0f}%)",
                            expanded=False,
                        ):
                            sc_html = '<table class="signal-table"><thead><tr>'
                            sc_html += '<th>Ticker</th><th>Signal</th><th>P(up)</th><th>Actual Return</th><th>Result</th>'
                            sc_html += '</tr></thead><tbody>'
                            for _, mr in merged.sort_values("p_up", ascending=False).iterrows():
                                sig = str(mr.get("signal", "flat"))
                                ret = float(mr["target_return_1d"]) if pd.notna(mr["target_return_1d"]) else 0.0
                                ok = bool(mr["correct"])
                                result_cls = "sc-correct" if ok else "sc-wrong"
                                result_sym = "✓" if ok else "✗"
                                sc_html += (
                                    f'<tr>'
                                    f'<td><span class="ticker-chip">{mr["ticker"]}</span></td>'
                                    f'<td>{badge_html(sig)}</td>'
                                    f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:0.8rem">'
                                    f'{float(mr["p_up"]):.1%}</td>'
                                    f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:0.8rem">'
                                    f'{ret:+.2%}</td>'
                                    f'<td><span class="{result_cls}">{result_sym}</span></td>'
                                    f'</tr>'
                                )
                            sc_html += '</tbody></table>'
                            st.markdown(sc_html, unsafe_allow_html=True)

        sig_col, dist_col = st.columns([2.4, 1], gap="large")
        with sig_col:
            st.markdown('<p class="section-heading">Signal board</p>', unsafe_allow_html=True)
            st.caption(
                "Stocks ranked by **P(up)** for the model's stored target label. "
                "**Edge** shows how far past the threshold each signal is. Use this as inspection context, not a trade recommendation."
            )
            if rows:
                st.markdown(signal_table_html(rows, long_th, short_th), unsafe_allow_html=True)

                # Download button
                dl_rows = [
                    {"ticker": r["ticker"], "as_of": r["date"],
                     "p_up": round(r["p_up"], 6),
                     "signal": signal_key(r["p_up"], long_th, short_th),
                     "edge_vs_threshold": round(
                         r["p_up"] - long_th if signal_key(r["p_up"], long_th, short_th) == "long"
                         else short_th - r["p_up"] if signal_key(r["p_up"], long_th, short_th) == "short"
                         else 0.0, 4)}
                    for r in rows
                ]
                csv_str = pd.DataFrame(dl_rows).to_csv(index=False)
                st.download_button(
                    "⬇ Download signals CSV",
                    data=csv_str,
                    file_name=f"olympus_signals_{as_of_max}.csv",
                    mime="text/csv",
                    use_container_width=False,
                )

                # Paper trade quick-log
                with st.expander("Log a paper trade", expanded=False):
                    with st.form("paper_trade_form", clear_on_submit=True):
                        pt_cols = st.columns([1, 1, 1, 1, 2])
                        pt_ticker = pt_cols[0].selectbox(
                            "Ticker", [r["ticker"] for r in rows], key="pt_ticker",
                        )
                        pt_action = pt_cols[1].selectbox(
                            "Action", ["buy", "sell"], key="pt_action",
                        )
                        pt_shares = pt_cols[2].number_input(
                            "Shares", min_value=1, value=10, key="pt_shares",
                        )
                        pt_price = pt_cols[3].number_input(
                            "Price", min_value=0.01, value=100.0, step=0.01,
                            key="pt_price",
                        )
                        pt_notes = pt_cols[4].text_input("Notes", key="pt_notes")
                        pt_submit = st.form_submit_button("Save trade")
                        if pt_submit:
                            save_paper_trade({
                                "date": datetime.now().strftime("%Y-%m-%d"),
                                "ticker": pt_ticker,
                                "action": pt_action,
                                "shares": pt_shares,
                                "price": pt_price,
                                "notes": pt_notes,
                            })
                            st.success(f"Logged: {pt_action.upper()} {pt_shares} {pt_ticker} @ ${pt_price:.2f}")
            else:
                st.info("No signals — check model and features files.")

        with dist_col:
            st.markdown('<p class="section-heading">Distribution</p>', unsafe_allow_html=True)
            st.plotly_chart(distribution_donut(counts), use_container_width=True,
                            config={"displayModeBar": False})
            st.caption(
                "How today's signals break down. A healthy market typically shows more Longs; "
                "lots of Shorts may signal broad weakness."
            )

        # ── News sentiment panel ──
        if df_raw is not None:
            sent_cols_needed = ["ticker", "Date", "news_count_3d", "sentiment_mean_3d"]
            if all(c in df_raw.columns for c in sent_cols_needed):
                latest_sent = (
                    df_raw.sort_values("Date")
                    .groupby("ticker").tail(1)
                    [["ticker", "news_count_3d", "sentiment_mean_3d"]]
                    .sort_values("sentiment_mean_3d", ascending=False)
                )
                has_any_news = (latest_sent["news_count_3d"].fillna(0) > 0).any()
                if has_any_news:
                    st.markdown('<p class="section-heading">News sentiment (3-day)</p>',
                                unsafe_allow_html=True)
                    st.caption("Recent news tone per ticker. Green = positive, red = negative, gray = no news.")
                    sent_html = '<div style="overflow-x:auto"><table class="signal-table"><thead><tr>'
                    sent_html += '<th>Ticker</th><th>Articles</th><th>Sentiment</th></tr></thead><tbody>'
                    for _, sr in latest_sent.iterrows():
                        nc = int(sr["news_count_3d"]) if pd.notna(sr["news_count_3d"]) else 0
                        sv = float(sr["sentiment_mean_3d"]) if pd.notna(sr["sentiment_mean_3d"]) else 0.0
                        if nc == 0:
                            bar_color = ACCENT_FLAT
                        elif sv > 0.05:
                            bar_color = ACCENT_LONG
                        elif sv < -0.05:
                            bar_color = ACCENT_SHORT
                        else:
                            bar_color = ACCENT_FLAT
                        pct = max(0.0, min(1.0, (sv + 1) / 2)) * 100
                        sent_html += (
                            f'<tr>'
                            f'<td><span class="ticker-chip">{sr["ticker"]}</span></td>'
                            f'<td style="font-family:\'JetBrains Mono\',monospace;font-size:0.8rem">{nc}</td>'
                            f'<td><div class="sent-bar-wrap">'
                            f'<div class="sent-bar-bg"><div class="sent-bar-fill" '
                            f'style="width:{pct:.0f}%;background:{bar_color}"></div></div>'
                            f'<span class="sent-val">{sv:+.3f}</span>'
                            f'</div></td></tr>'
                        )
                    sent_html += '</tbody></table></div>'
                    st.markdown(sent_html, unsafe_allow_html=True)

    # ════════ TAB 2: Chart ════════
    with tab_chart:
        st.markdown('<p class="section-heading">Price context</p>', unsafe_allow_html=True)
        # Prefer scored tickers; if scoring failed, still list names from features + disk CSVs.
        tickers_from_rows = sorted({str(r["ticker"]) for r in rows})
        if tickers_from_rows:
            tickers_sorted = tickers_from_rows
        elif df_raw is not None and "ticker" in df_raw.columns:
            tickers_sorted = sorted(
                df_raw["ticker"].astype(str).str.upper().dropna().unique().tolist()
            )
        else:
            tickers_sorted = []
        tickers_with_price = [t for t in tickers_sorted if (DATA_DIR / f"{t}_daily.csv").exists()]
        if not tickers_sorted:
            st.info("No tickers found in `features.csv`. Run `build_features.py` first.")
        else:
            if not tickers_with_price:
                st.warning(
                    f"No `data/<TICKER>_daily.csv` files for: {', '.join(tickers_sorted)}. "
                    "Run `fetch_price_data.py` (or add daily CSVs) to enable charts."
                )
            default_idx = next(
                (i for i, t in enumerate(tickers_sorted) if (DATA_DIR / f"{t}_daily.csv").exists()),
                0,
            )
            pick = st.selectbox("Ticker", tickers_sorted, index=min(default_idx, len(tickers_sorted) - 1),
                                label_visibility="collapsed")
            pick_signal = signal_key(
                next((r["p_up"] for r in rows if str(r["ticker"]) == pick), 0.5),
                long_th, short_th,
            )
            hist = load_price_history(pick)
            if hist is None or hist.empty:
                st.info(f"No `data/{pick}_daily.csv` — add price data to see the chart.")
            else:
                chart_col, detail_col = st.columns([2.4, 1], gap="large")
                with chart_col:
                    st.plotly_chart(
                        candlestick_chart(hist, pick, signal=pick_signal, initial_days=183),
                        use_container_width=True,
                        config={"scrollZoom": False, "displayModeBar": True,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d", "toImage", "autoScale2d"],
                                "displaylogo": False},
                    )

                with detail_col:
                    # Signal card
                    pick_pup = next((r["p_up"] for r in rows if str(r["ticker"]) == pick), None)
                    if pick_pup is not None:
                        pick_edge = (
                            pick_pup - long_th if pick_signal == "long"
                            else short_th - pick_pup if pick_signal == "short"
                            else 0.0
                        )
                        sig_colors = {"long": ACCENT_LONG, "short": ACCENT_SHORT, "flat": ACCENT_FLAT}
                        st.markdown(
                            f'<div style="background:rgba(15,23,42,0.55);border:1px solid rgba(148,163,184,0.12);'
                            f'border-radius:14px;padding:1rem 1.2rem;margin-bottom:0.6rem">'
                            f'<p class="stat-label">SIGNAL</p>'
                            f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:1.4rem;'
                            f'font-weight:600;color:{sig_colors[pick_signal]};margin:0.2rem 0">'
                            f'{pick_signal.upper()}</p>'
                            f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.85rem;'
                            f'color:#cbd5e1;margin:0">P(up): {pick_pup:.1%} &nbsp; Edge: {pick_edge:+.1%}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # Key features for this ticker
                    if df_raw is not None and "ticker" in df_raw.columns:
                        ticker_latest = (
                            df_raw[df_raw["ticker"] == pick]
                            .sort_values("Date")
                            .tail(1)
                        )
                        if not ticker_latest.empty:
                            feat_display = {
                                "rsi_14": "RSI (14)",
                                "momentum_20d": "Mom 20d",
                                "vol_5d": "Vol 5d",
                                "sentiment_mean_3d": "Sentiment",
                                "return_1d": "Return 1d",
                                "volume_ratio": "Vol Ratio",
                                "dist_ma_20": "Dist MA20",
                                "bollinger_pctb": "Boll %B",
                            }
                            available_feats = {k: v for k, v in feat_display.items()
                                               if k in ticker_latest.columns}
                            if available_feats:
                                st.markdown('<p class="section-heading">Key features</p>',
                                            unsafe_allow_html=True)
                                feat_html = '<div class="feat-grid">'
                                r_latest = ticker_latest.iloc[0]
                                for col, label in available_feats.items():
                                    val = r_latest[col]
                                    val_str = f"{float(val):.4f}" if pd.notna(val) else "—"
                                    feat_html += (
                                        f'<div class="feat-card">'
                                        f'<p class="feat-name">{label}</p>'
                                        f'<p class="feat-val">{val_str}</p></div>'
                                    )
                                feat_html += '</div>'
                                st.markdown(feat_html, unsafe_allow_html=True)

                    # Top feature contributions
                    feature_imp = bundle.get("feature_importance")
                    if feature_imp and df_raw is not None:
                        ticker_latest = (
                            df_raw[df_raw["ticker"] == pick]
                            .sort_values("Date")
                            .tail(1)
                        )
                        if not ticker_latest.empty:
                            imp_s = pd.Series(feature_imp)
                            avail = [c for c in imp_s.index if c in ticker_latest.columns]
                            if avail:
                                r_latest = ticker_latest.iloc[0]
                                feat_means = df_raw[avail].mean()
                                weighted = {}
                                for c in avail:
                                    v = r_latest[c]
                                    m = feat_means[c]
                                    if pd.notna(v) and pd.notna(m) and m != 0:
                                        weighted[c] = abs(imp_s[c]) * abs(v - m) / (abs(m) + 1e-9)
                                if weighted:
                                    top5 = sorted(weighted.items(), key=lambda x: x[1], reverse=True)[:5]
                                    st.markdown('<p class="section-heading">Top drivers</p>',
                                                unsafe_allow_html=True)
                                    st.caption("Features most influencing this ticker's score vs the average.")
                                    for feat, w in top5:
                                        v = r_latest[feat]
                                        m = feat_means[feat]
                                        direction = "above" if v > m else "below"
                                        st.caption(f"`{feat}` = {float(v):.4f} ({direction} avg {float(m):.4f})")

            # ── Signal history heatmap ──
            if plog is not None:
                hm_fig = signal_history_heatmap(plog, pick)
                if hm_fig is not None:
                    st.markdown('<p class="section-heading">Signal history</p>',
                                unsafe_allow_html=True)
                    st.caption(
                        f"How the model's signal for **{pick}** has evolved. "
                        "Green = Long, Red = Short, Gray = Flat."
                    )
                    st.plotly_chart(hm_fig, use_container_width=True,
                                    config={"displayModeBar": False})

    # ════════ TAB 3: Model Health ════════
    with tab_health:
        h1, h2 = st.columns(2, gap="large")

        with h1:
            st.markdown('<p class="section-heading">Training metrics history</p>', unsafe_allow_html=True)
            st.caption(
                "**ROC AUC** measures ranking quality (>0.5 is better than random). "
                "**Accuracy** is the % of correct direction calls. "
                "Both should stay above the 0.5 baseline."
            )
            if ml is not None and len(ml) >= 2:
                st.plotly_chart(metrics_history_chart(ml), use_container_width=True,
                                config={"displayModeBar": False})
            elif ml is not None and len(ml) == 1:
                st.caption("Only one training run logged — retrain to see a trend.")
            else:
                st.caption("No metrics log yet — run `python scripts/train_model.py`.")

            st.markdown('<p class="section-heading">Feature importances</p>', unsafe_allow_html=True)
            imp = bundle.get("feature_importance")
            if imp:
                st.plotly_chart(importance_chart(imp), use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.caption("Retrain to embed importances.")

        with h2:
            st.markdown('<p class="section-heading">Walk-forward folds</p>', unsafe_allow_html=True)
            st.caption(
                "Each fold trains on past data and tests on future data — simulating real trading. "
                "Consistent metrics across folds means the model generalises well."
            )
            wf = load_walk_forward()
            if wf is not None and len(wf):
                display_cols = [c for c in
                    ["fold", "test_start", "test_end", "roc_auc", "accuracy_0p5",
                     "portfolio_mode", "total_return", "sharpe", "max_drawdown"]
                    if c in wf.columns]
                fmt: dict = {}
                for c in ["roc_auc", "accuracy_0p5", "total_return", "sharpe", "max_drawdown"]:
                    if c in display_cols:
                        fmt[c] = "{:.4f}"
                st.dataframe(
                    wf[display_cols].style.format(fmt),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("No walk-forward results yet — run `python scripts/walk_forward_eval.py`.")

            st.markdown('<p class="section-heading">Model bundle details</p>', unsafe_allow_html=True)
            bundle_info = {k: v for k, v in bundle.items()
                           if k not in ("model", "models_by_ticker", "feature_importance",
                                        "feature_names", "xgb_extra")}
            for k, v in bundle_info.items():
                st.caption(f"**{k}:** `{v}`")

        # ── Regime, robustness, threshold sweep (full width below) ──
        reg_col, rob_col = st.columns(2, gap="large")

        with reg_col:
            st.markdown('<p class="section-heading">Regime analysis</p>', unsafe_allow_html=True)
            st.caption("How the model performs under different market conditions.")
            regime_df = load_regime_analysis()
            if regime_df is not None and len(regime_df):
                rfmt = {c: "{:.4f}" for c in ["accuracy", "roc_auc"] if c in regime_df.columns}
                st.dataframe(regime_df.style.format(rfmt), use_container_width=True, hide_index=True)
            else:
                st.caption("Run `python scripts/evaluate_robustness.py` to generate regime data.")

        with rob_col:
            st.markdown('<p class="section-heading">Robustness (bootstrap CIs)</p>', unsafe_allow_html=True)
            st.caption("Per-fold metrics with confidence intervals from bootstrap resampling.")
            robust_df = load_robustness_report()
            if robust_df is not None and len(robust_df):
                rcols = [c for c in ["fold", "test_start", "test_end", "accuracy",
                                     "roc_auc", "roc_auc_ci_lo", "roc_auc_ci_hi",
                                     "brier"] if c in robust_df.columns]
                rfmt2 = {c: "{:.4f}" for c in rcols if c not in ("fold", "test_start", "test_end")}
                st.dataframe(robust_df[rcols].style.format(rfmt2),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("Run `python scripts/evaluate_robustness.py` to generate robustness data.")

        st.markdown('<p class="section-heading">Threshold sweep</p>', unsafe_allow_html=True)
        st.caption(
            "How Sharpe ratio and hit rate change as the long threshold moves. "
            "Helps justify the preset conviction levels."
        )
        ts_df = load_threshold_sweep()
        if ts_df is not None and len(ts_df):
            st.plotly_chart(threshold_sweep_chart(ts_df), use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.caption("Run `python scripts/walk_forward_eval.py` to generate threshold sweep data.")

    # ════════ TAB 4: Backtest ════════
    with tab_bt:
        bt1, bt2 = st.columns(2, gap="large")
        with bt1:
            st.markdown('<p class="section-heading">Strategy vs baselines</p>', unsafe_allow_html=True)
            st.caption(
                "**Total Return** — cumulative profit over the test period. "
                "**Sharpe** — return per unit of risk (higher is better). "
                "A model should not be considered deployable unless it beats passive baselines."
            )
            bdf = load_backtest_comparison()
            if bdf is not None and len(bdf):
                st.plotly_chart(backtest_bar(bdf), use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.caption("Run `python scripts/evaluate_backtest.py` to generate backtest data.")

        with bt2:
            st.markdown('<p class="section-heading">Per-strategy summary</p>', unsafe_allow_html=True)
            st.caption(
                "**CAGR** — annualised growth rate. "
                "**Max Drawdown** — worst peak-to-trough drop (closer to 0 is better). "
                "**Days** — how many trading days the strategy covered."
            )
            if bdf is not None and len(bdf):
                show = [c for c in ["strategy", "total_return", "cagr", "sharpe", "max_drawdown", "days"]
                        if c in bdf.columns]
                fmt2 = {c: "{:.4f}" for c in ["total_return", "cagr", "sharpe", "max_drawdown"] if c in show}
                st.dataframe(bdf[show].style.format(fmt2), use_container_width=True, hide_index=True)

    # ════════ TAB 5: Paper Trading ════════
    with tab_paper:
        st.markdown('<p class="section-heading">Paper trading journal</p>', unsafe_allow_html=True)
        st.caption(
            "Track simulated trades to build confidence before using real money. "
            "Log trades from the Signals tab or directly here."
        )

        pt_df = load_paper_trades()

        # Quick-add form
        with st.expander("Log a new trade", expanded=False):
            with st.form("paper_trade_form_tab", clear_on_submit=True):
                ptc = st.columns([1, 1, 1, 1, 2])
                t_ticker = ptc[0].text_input("Ticker", value="AAPL")
                t_action = ptc[1].selectbox("Action", ["buy", "sell"])
                t_shares = ptc[2].number_input("Shares", min_value=1, value=10)
                t_price = ptc[3].number_input("Price", min_value=0.01, value=100.0, step=0.01)
                t_notes = ptc[4].text_input("Notes")
                if st.form_submit_button("Save trade"):
                    save_paper_trade({
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "ticker": t_ticker.upper(),
                        "action": t_action,
                        "shares": t_shares,
                        "price": t_price,
                        "notes": t_notes,
                    })
                    st.success(f"Logged: {t_action.upper()} {t_shares} {t_ticker.upper()} @ ${t_price:.2f}")
                    st.rerun()

        if pt_df is not None and len(pt_df):
            paper_c1, paper_c2 = st.columns(2, gap="large")

            with paper_c1:
                # Open positions (buys without matching sells)
                st.markdown('<p class="section-heading">Open positions</p>', unsafe_allow_html=True)
                positions: dict[str, dict] = {}
                for _, tr in pt_df.iterrows():
                    tk = str(tr["ticker"]).upper()
                    shares = int(tr.get("shares", 0))
                    price = float(tr.get("price", 0))
                    action = str(tr.get("action", "")).lower()
                    if tk not in positions:
                        positions[tk] = {"shares": 0, "cost_basis": 0.0}
                    if action == "buy":
                        total_cost = positions[tk]["cost_basis"] * positions[tk]["shares"] + price * shares
                        positions[tk]["shares"] += shares
                        if positions[tk]["shares"] > 0:
                            positions[tk]["cost_basis"] = total_cost / positions[tk]["shares"]
                    elif action == "sell":
                        positions[tk]["shares"] -= shares

                open_pos = {k: v for k, v in positions.items() if v["shares"] > 0}
                if open_pos:
                    pos_html = '<table class="signal-table"><thead><tr>'
                    pos_html += '<th>Ticker</th><th>Shares</th><th>Avg Cost</th><th>Current</th><th>P&L</th>'
                    pos_html += '</tr></thead><tbody>'
                    for tk, pos in sorted(open_pos.items()):
                        hist_tk = load_price_history(tk, max_rows=2)
                        cur_price = float(hist_tk.iloc[-1]["Close"]) if hist_tk is not None and len(hist_tk) else 0.0
                        pnl = (cur_price - pos["cost_basis"]) * pos["shares"]
                        pnl_cls = "pnl-pos" if pnl >= 0 else "pnl-neg"
                        pos_html += (
                            f'<tr>'
                            f'<td><span class="ticker-chip">{tk}</span></td>'
                            f'<td style="font-family:\'JetBrains Mono\',monospace">{pos["shares"]}</td>'
                            f'<td style="font-family:\'JetBrains Mono\',monospace">${pos["cost_basis"]:.2f}</td>'
                            f'<td style="font-family:\'JetBrains Mono\',monospace">'
                            f'{"${:.2f}".format(cur_price) if cur_price else "—"}</td>'
                            f'<td><span class="{pnl_cls}">${pnl:+,.2f}</span></td>'
                            f'</tr>'
                        )
                    pos_html += '</tbody></table>'
                    st.markdown(pos_html, unsafe_allow_html=True)
                else:
                    st.caption("No open positions. Log a buy trade to get started.")

                # Performance summary
                buys_df = pt_df[pt_df["action"].str.lower() == "buy"].copy()
                sells_df = pt_df[pt_df["action"].str.lower() == "sell"].copy()
                if not sells_df.empty and not buys_df.empty:
                    st.markdown('<p class="section-heading">Performance summary</p>',
                                unsafe_allow_html=True)
                    total_bought = (
                        pd.to_numeric(buys_df["shares"], errors="coerce").fillna(0)
                        * pd.to_numeric(buys_df["price"], errors="coerce").fillna(0)
                    ).sum()
                    total_sold = (
                        pd.to_numeric(sells_df["shares"], errors="coerce").fillna(0)
                        * pd.to_numeric(sells_df["price"], errors="coerce").fillna(0)
                    ).sum()
                    realized = total_sold - total_bought
                    n_sells = len(sells_df)
                    pnl_cls = "pnl-pos" if realized >= 0 else "pnl-neg"
                    st.markdown(
                        f'<div class="feat-grid">'
                        f'<div class="feat-card"><p class="feat-name">Total Trades</p>'
                        f'<p class="feat-val">{len(pt_df)}</p></div>'
                        f'<div class="feat-card"><p class="feat-name">Closed Trades</p>'
                        f'<p class="feat-val">{n_sells}</p></div>'
                        f'<div class="feat-card"><p class="feat-name">Realized P&L</p>'
                        f'<p class="feat-val {pnl_cls}">${realized:+,.2f}</p></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with paper_c2:
                st.markdown('<p class="section-heading">Trade history</p>', unsafe_allow_html=True)
                display_pt = pt_df.copy()
                display_pt = display_pt.sort_index(ascending=False)
                show_cols = [c for c in ["date", "ticker", "action", "shares", "price", "notes"]
                             if c in display_pt.columns]
                st.dataframe(display_pt[show_cols], use_container_width=True, hide_index=True,
                             height=400)
        else:
            st.info(
                "No paper trades yet. Use the **Log a paper trade** form in the Signals tab "
                "or above to record your first simulated trade."
            )

    st.markdown("---")
    st.caption(
        "Signals use the **last complete date** per ticker in `features.csv`. "
        "Refresh data with your scripts then use **Rerun** (⋮ menu) to reload. "
        "This model predicts whether a stock will **beat SPY** tomorrow — not whether it goes up in absolute terms."
    )


if __name__ == "__main__":
    main()
