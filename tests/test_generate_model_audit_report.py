"""Model audit report helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import generate_model_audit_report as audit  # noqa: E402


class TestAuditHelpers(unittest.TestCase):
    def test_latest_metrics_prefers_matching_target(self):
        df = pd.DataFrame(
            {
                "target": ["a", "b", "a"],
                "roc_auc": [0.40, 0.60, 0.55],
            }
        )
        row = audit.latest_metrics(df, target_column="b")
        self.assertEqual(row["target"], "b")
        self.assertEqual(row["roc_auc"], 0.60)

    def test_status_thresholds(self):
        self.assertEqual(audit.status_for(0.56, pass_at=0.55, monitor_at=0.52), "pass")
        self.assertEqual(audit.status_for(0.53, pass_at=0.55, monitor_at=0.52), "monitor")
        self.assertEqual(audit.status_for(0.50, pass_at=0.55, monitor_at=0.52), "fail")

    def test_markdown_table_escapes_pipe(self):
        df = pd.DataFrame({"area": ["a|b"], "status": ["pass"]})
        out = audit.markdown_table(df, ["area", "status"])
        self.assertIn("a\\|b", out)


if __name__ == "__main__":
    unittest.main()
