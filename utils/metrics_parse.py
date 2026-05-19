"""Parse printed metrics from train_model.py stdout (used by compare_v5_v6 and tests)."""
from __future__ import annotations

import re
from typing import Any


def parse_train_stdout(text: str) -> dict[str, Any]:
    """
    Extract Accuracy, ROC AUC, Log loss, Brier, and feature count from train_model output.
    """
    m: dict[str, Any] = {}
    for key, pat in [
        ("accuracy", r"Accuracy:\s*([\d.]+)"),
        ("roc_auc", r"ROC AUC:\s*([\d.]+)"),
        ("log_loss", r"Log loss:\s*([\d.]+)"),
        ("brier", r"Brier:\s*([\d.]+)"),
        ("n_features", r"Features after variance prune:\s*(\d+)"),
    ]:
        g = re.search(pat, text)
        if g:
            try:
                m[key] = float(g.group(1)) if key != "n_features" else int(g.group(1))
            except ValueError:
                m[key] = g.group(1)
    return m
