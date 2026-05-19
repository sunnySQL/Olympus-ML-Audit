"""Pickle-stable dummy classifier for integration tests (importable from scripts)."""
from __future__ import annotations

import numpy as np


class DummyBinaryClassifier:
    """Returns fixed positive-class probability for every row."""

    def __init__(self, p_pos: float = 0.62):
        self.p_pos = float(p_pos)

    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, self.p_pos)
        return np.column_stack([1.0 - p, p])
