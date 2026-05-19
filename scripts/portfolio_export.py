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


SCREENSHOT_LABELS = {
    "landingpage.png": "Dashboard landing page",
    "auditlog.png": "Audit checklist",
    "modelhealth.png": "Model health",
    "backtest.png": "Backtest vs baselines",
    "chart.png": "Ticker chart and signal context",
}


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
    screenshots = list_screenshots()
    found_lines = "\n".join(
        f"- `{path.name}` - {SCREENSHOT_LABELS.get(path.name, 'Project screenshot')}"
        for path in screenshots
    )
    if not found_lines:
        found_lines = "- No screenshots found yet."
    guide = SCREENSHOTS_DIR / "README.md"
    guide.write_text(
        f"""# Screenshot Checklist

Current screenshots:

{found_lines}

Recommended set before publishing:

- `landingpage.png` - dashboard landing page.
- `auditlog.png` - dashboard audit verdict and checklist.
- `modelhealth.png` - metrics history, feature importance, and walk-forward section.
- `backtest.png` - strategy vs baseline comparison.
- `chart.png` - ticker chart and signal context.

Recommended browser width: 1440px or wider.
""",
        encoding="utf-8",
    )


def list_screenshots() -> list[Path]:
    if not SCREENSHOTS_DIR.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(p for p in SCREENSHOTS_DIR.iterdir() if p.suffix.lower() in exts)


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
    screenshots = list_screenshots()
    screenshot_lines = "\n".join(
        f"- `screenshots/{path.name}` - {SCREENSHOT_LABELS.get(path.name, 'Project screenshot')}"
        for path in screenshots
    )
    if not screenshot_lines:
        screenshot_lines = "- Add screenshots to `screenshots/`."
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

{screenshot_lines}

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
