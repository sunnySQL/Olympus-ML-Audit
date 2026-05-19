"""Apply saved model bundle predictions to a feature dataframe (any row order)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from utils.sklearn_compat import patch_bundle_models


def _require_features(df: pd.DataFrame, feats: list) -> None:
    missing = [c for c in feats if c not in df.columns]
    if missing:
        tail = "..." if len(missing) > 15 else ""
        raise ValueError(
            f"DataFrame missing columns required by bundle: {missing[:15]}{tail}"
        )


def add_pred_prob(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Returns copy with pred_prob; regression bundles also set pred_return and map pred_prob via expit."""
    patch_bundle_models(bundle)
    if bundle.get("task") == "regression":
        return _add_regression(df, bundle)
    out = df.copy()
    feats = bundle["feature_names"]
    _require_features(out, list(feats))
    out["pred_prob"] = np.nan

    kind = bundle.get("model_kind", "global")
    if kind == "per_ticker":
        global_m = bundle["model"]
        mbt: dict = bundle.get("models_by_ticker") or {}
        for t, g in out.groupby("ticker"):
            key = str(t).upper()
            m = mbt.get(key)
            model = m if m is not None else global_m
            out.loc[g.index, "pred_prob"] = model.predict_proba(g[feats])[:, 1]
    else:
        out["pred_prob"] = bundle["model"].predict_proba(out[feats])[:, 1]

    return out


def _add_regression(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    out = df.copy()
    feats = bundle["feature_names"]
    _require_features(out, list(feats))
    scale = float(bundle.get("return_prob_scale", 30.0))
    kind = bundle.get("model_kind", "global")
    out["pred_return"] = np.nan
    if kind == "per_ticker":
        global_m = bundle["model"]
        mbt: dict = bundle.get("models_by_ticker") or {}
        for t, g in out.groupby("ticker"):
            key = str(t).upper()
            m = mbt.get(key)
            model = m if m is not None else global_m
            out.loc[g.index, "pred_return"] = model.predict(g[feats])
    else:
        out["pred_return"] = bundle["model"].predict(out[feats])
    out["pred_prob"] = expit(out["pred_return"].astype(float) * scale)
    return out
