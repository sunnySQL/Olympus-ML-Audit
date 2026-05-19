#!/usr/bin/env python3
"""Assemble curated Olympus portfolio artifacts."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORTFOLIO_DIR = ROOT / "portfolio"
ARTIFACTS_DIR = PORTFOLIO_DIR / "artifacts"
SCREENSHOTS_DIR = PORTFOLIO_DIR / "screenshots"
REPORTS_DIR = ROOT / "reports"


KEY_REPORTS = [
    "model_audit_summary.csv",
    "model_audit_summary.json",
    "model_target_benchmark.csv",
    "backtest_comparison.csv",
    "ranking_alpha_summary.csv",
    "rank_feature_sweep.csv",
    "walk_forward.csv",
    "walk_forward_next3d.csv",
]


def run_audit() -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "generate_model_audit_report.py")]
    subprocess.run(cmd, cwd=ROOT, check=True)


def copy_reports() -> list[str]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in KEY_REPORTS:
        src = REPORTS_DIR / name
        if not src.exists():
            continue
        dst = ARTIFACTS_DIR / name
        shutil.copy2(src, dst)
        copied.append(name)
    return copied


def ensure_screenshot_placeholders() -> None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    gitkeep = SCREENSHOTS_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")
    guide = SCREENSHOTS_DIR / "README.md"
    guide.write_text(
        """# Screenshot Checklist

Add these images before publishing the portfolio page:

- `audit_tab.png` - dashboard audit verdict and checklist.
- `model_health_tab.png` - metrics history, feature importance, and walk-forward section.
- `backtest_tab.png` - strategy vs baseline comparison.
- `audit_report_html.png` - generated HTML model audit report.

Recommended browser width: 1440px or wider.
""",
        encoding="utf-8",
    )


def load_audit_meta() -> dict:
    path = REPORTS_DIR / "model_audit_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_portfolio_index(copied: list[str]) -> None:
    meta = load_audit_meta()
    generated = datetime.now(timezone.utc).isoformat()
    artifact_lines = "\n".join(f"- `artifacts/{name}`" for name in copied) or "- No report artifacts copied."
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    (PORTFOLIO_DIR / "README.md").write_text(
        f"""# Olympus Portfolio Export

Generated: `{generated}`

## Verdict

- Deployment: `{meta.get("deploy_verdict", "UNKNOWN")}`
- Project: `{meta.get("project_verdict", "UNKNOWN")}`
- Target: `{meta.get("target_column", "UNKNOWN")}`
- Feature set: `{meta.get("feature_set_version", "UNKNOWN")}`

## Main Artifacts

- `olympus_case_study.md`
- `olympus_model_audit_report.md`
- `olympus_model_audit_report.html`

## Result Tables

{artifact_lines}

## Screenshots

Add your dashboard screenshots to `screenshots/`:

- `screenshots/audit_tab.png`
- `screenshots/model_health_tab.png`
- `screenshots/backtest_tab.png`
- `screenshots/audit_report_html.png`

## Suggested Portfolio Blurb

Olympus is an ML research platform for market signals. The current model is rejected for trading, but the system is portfolio-ready because it demonstrates realistic target design, leakage checks, baseline comparisons, ranking-alpha evaluation, walk-forward testing, and a generated deployment audit.
""",
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Export curated Olympus portfolio artifacts")
    p.add_argument(
        "--skip-audit",
        action="store_true",
        help="Do not regenerate the model audit before copying artifacts",
    )
    args = p.parse_args()

    if not args.skip_audit:
        run_audit()
    copied = copy_reports()
    ensure_screenshot_placeholders()
    write_portfolio_index(copied)

    print(f"Portfolio export complete: {PORTFOLIO_DIR}")
    print(f"Copied {len(copied)} report artifacts.")
    print(f"Add screenshots to: {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()
