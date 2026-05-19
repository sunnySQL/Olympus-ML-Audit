#!/usr/bin/env python3
"""
Run build_features → train_model → evaluate_backtest in sequence.
Stops on first failure. Optional price/news refresh first.

Usage (from project root):
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --fetch-prices --fetch-news
  python scripts/run_pipeline.py --no-per-ticker
  python scripts/run_pipeline.py --skip-backtest
  python scripts/run_pipeline.py --walk-forward
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _project_python() -> str:
    """
    Interpreter for pipeline subprocesses. Prefer repo `.venv` so every step uses the same
    packages as `requirements.txt`, even when this script was started with conda/system Python.

    Do NOT resolve() — the venv python is a symlink to the base interpreter, and resolving it
    would bypass the venv's site-packages entirely.
    """
    if os.name == "nt":
        win = ROOT / ".venv" / "Scripts" / "python.exe"
        if win.exists():
            return str(win)
    nix = ROOT / ".venv" / "bin" / "python"
    if nix.exists():
        return str(nix)
    return sys.executable


_PY = _project_python()


def run_step(argv: list[str]) -> None:
    print("\n" + "=" * 60)
    print("$", _PY, " ".join(argv))
    print("=" * 60 + "\n")
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("KMP_USE_SHM", "0")
    r = subprocess.run([_PY, *argv], cwd=ROOT, env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description="Run feature build, train, and backtest in one go.")
    p.add_argument("--fetch-prices", action="store_true", help="Run fetch_price_data.py first")
    p.add_argument("--fetch-news", action="store_true", help="Run fetch_news.py first")
    p.add_argument(
        "--fmp-stock-backfill-days",
        type=int,
        default=0,
        metavar="N",
        help="When used with --fetch-news, pass to fetch_news.py (FMP Search Stock News; 0=skip)",
    )
    p.add_argument("--skip-features", action="store_true", help="Skip build_features.py")
    p.add_argument("--skip-train", action="store_true", help="Skip train_model.py")
    p.add_argument("--skip-backtest", action="store_true", help="Skip evaluate_backtest.py")
    p.add_argument(
        "--walk-forward",
        action="store_true",
        help="After training, run walk_forward_eval.py (OOS folds; pass extra flags after --)",
    )
    p.add_argument(
        "--no-per-ticker",
        action="store_true",
        help="Pass through to train_model.py (single global model)",
    )
    args, train_extra = p.parse_known_args()

    if args.fetch_prices:
        run_step([str(ROOT / "scripts" / "fetch_price_data.py")])
    if args.fetch_news:
        fn_cmd = [str(ROOT / "scripts" / "fetch_news.py")]
        if args.fmp_stock_backfill_days > 0:
            fn_cmd.extend(["--fmp-stock-backfill-days", str(args.fmp_stock_backfill_days)])
        run_step(fn_cmd)

    if not args.skip_features:
        run_step([str(ROOT / "scripts" / "build_features.py")])

    if not args.skip_train:
        train_cmd = [str(ROOT / "scripts" / "train_model.py")]
        if args.no_per_ticker:
            train_cmd.append("--no-per-ticker")
        train_cmd.extend(train_extra)
        run_step(train_cmd)

    if not args.skip_backtest:
        run_step([str(ROOT / "scripts" / "evaluate_backtest.py")])

    if args.walk_forward:
        wf_cmd = [str(ROOT / "scripts" / "walk_forward_eval.py")]
        if not args.no_per_ticker:
            wf_cmd.append("--per-ticker")
        run_step(wf_cmd)

    print("\nPipeline finished OK.")


if __name__ == "__main__":
    main()
