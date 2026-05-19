#!/usr/bin/env python3
"""Run the unittest suite from project root: python scripts/run_tests.py [--coverage]"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    extra = list(sys.argv[1:])
    cov = False
    if "--coverage" in extra:
        extra.remove("--coverage")
        cov = True
    if cov:
        cmd = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--source=utils,scripts",
            "--omit=*/venv/*",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-v",
            *extra,
        ]
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc == 0:
            subprocess.call([sys.executable, "-m", "coverage", "report", "-m"], cwd=ROOT)
        return rc
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v", *extra]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
