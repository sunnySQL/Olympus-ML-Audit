"""Expanding-window date splits for walk-forward evaluation."""
from __future__ import annotations

import numpy as np


def expanding_splits(
    dates_u: np.ndarray,
    min_train_days: int,
    n_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    dates_u: sorted unique trading dates (ascending).
    Returns list of (train_dates, test_dates) where train grows and each test block is disjoint.
    """
    d = len(dates_u)
    if d < min_train_days + n_folds:
        raise ValueError(
            f"Need at least min_train_days + folds unique dates ({min_train_days + n_folds}), got {d}"
        )
    remainder = dates_u[min_train_days:]
    chunks = np.array_split(remainder, n_folds)
    out = []
    offset = 0
    for _j, test_block in enumerate(chunks):
        if len(test_block) == 0:
            continue
        train_end = min_train_days + offset
        train_dates = dates_u[:train_end]
        out.append((train_dates, test_block))
        offset += len(test_block)
    return out
