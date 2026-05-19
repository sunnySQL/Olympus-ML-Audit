"""
Ensemble utilities for blending multiple model signals.

Supports:
  1. Global + per-ticker weighted blend (learned on calibration set)
  2. Classification + regression stacking via a lightweight meta-learner
  3. Multi-target consensus voting
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils.predict_bundle import add_pred_prob


class BlendedPredictor:
    """
    Weighted average of global model probability and per-ticker probability,
    with the blend weight learned on a calibration set.
    """

    def __init__(self, bundle: dict, blend_weight: float = 0.5):
        self.bundle = bundle
        self.blend_weight = blend_weight

    def fit_blend_weight(self, cal_df: pd.DataFrame, y_col: str) -> "BlendedPredictor":
        feats = self.bundle["feature_names"]
        global_m = self.bundle["model"]
        mbt: dict = self.bundle.get("models_by_ticker") or {}

        p_global = global_m.predict_proba(cal_df[feats])[:, 1]
        p_ticker = np.full_like(p_global, np.nan)
        for t, g in cal_df.groupby("ticker"):
            key = str(t).upper()
            m = mbt.get(key)
            if m is not None:
                p_ticker[g.index.get_indexer(g.index)] = m.predict_proba(g[feats])[:, 1]
            else:
                p_ticker[g.index.get_indexer(g.index)] = p_global[cal_df.index.get_indexer(g.index)]

        mask = ~np.isnan(p_ticker)
        if mask.sum() < 20:
            self.blend_weight = 0.5
            return self

        y = cal_df[y_col].values[mask]
        X = np.column_stack([p_global[mask], p_ticker[mask]])
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        try:
            lr.fit(X, y)
            w = lr.coef_[0]
            total = abs(w[0]) + abs(w[1])
            self.blend_weight = abs(w[1]) / total if total > 0 else 0.5
        except Exception:
            self.blend_weight = 0.5

        return self

    def predict_proba_blended(self, df: pd.DataFrame) -> np.ndarray:
        feats = self.bundle["feature_names"]
        global_m = self.bundle["model"]
        mbt: dict = self.bundle.get("models_by_ticker") or {}

        p_global = global_m.predict_proba(df[feats])[:, 1]
        p_ticker = p_global.copy()
        for t, g in df.groupby("ticker"):
            key = str(t).upper()
            m = mbt.get(key)
            if m is not None:
                idx = df.index.get_indexer(g.index)
                p_ticker[idx] = m.predict_proba(g[feats])[:, 1]

        w = self.blend_weight
        return (1 - w) * p_global + w * p_ticker


class StackedEnsemble:
    """
    Stacks classification probability + regression prediction into a meta-learner.
    """

    def __init__(self):
        self.meta: LogisticRegression | None = None

    def fit(
        self,
        clf_bundle: dict,
        reg_bundle: dict,
        cal_df: pd.DataFrame,
        y_col: str,
    ) -> "StackedEnsemble":
        scored_clf = add_pred_prob(cal_df, clf_bundle)
        scored_reg = add_pred_prob(cal_df, reg_bundle)

        p_clf = scored_clf["pred_prob"].values
        p_reg = scored_reg["pred_prob"].values
        if "pred_return" in scored_reg.columns:
            ret_pred = scored_reg["pred_return"].values
        else:
            ret_pred = np.zeros_like(p_reg)

        X = np.column_stack([p_clf, p_reg, ret_pred])
        y = cal_df[y_col].values
        mask = np.isfinite(X).all(axis=1)

        self.meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self.meta.fit(X[mask], y[mask])
        return self

    def predict_proba(
        self, df: pd.DataFrame, clf_bundle: dict, reg_bundle: dict
    ) -> np.ndarray:
        if self.meta is None:
            raise ValueError("Call .fit() before .predict_proba()")
        scored_clf = add_pred_prob(df, clf_bundle)
        scored_reg = add_pred_prob(df, reg_bundle)
        p_clf = scored_clf["pred_prob"].values
        p_reg = scored_reg["pred_prob"].values
        ret_pred = scored_reg.get("pred_return", pd.Series(np.zeros(len(df)))).values
        X = np.column_stack([p_clf, p_reg, ret_pred])
        return self.meta.predict_proba(X)[:, 1]


def multi_target_consensus(
    bundles: dict[str, dict],
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Score df with each bundle, compute a consensus probability (mean) and
    a vote count (how many targets predict > threshold).
    """
    probs = {}
    for name, bundle in bundles.items():
        scored = add_pred_prob(df, bundle)
        probs[name] = scored["pred_prob"].values

    out = df.copy()
    prob_matrix = np.column_stack(list(probs.values()))
    out["consensus_prob"] = prob_matrix.mean(axis=1)
    out["consensus_votes"] = (prob_matrix > threshold).sum(axis=1)
    out["consensus_signal"] = np.where(
        out["consensus_votes"] > len(bundles) / 2, 1, 0
    )
    for name, p in probs.items():
        out[f"pred_prob_{name}"] = p
    return out
