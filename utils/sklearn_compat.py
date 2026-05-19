"""Shims for sklearn version mismatches on pickled models."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def _patch_logistic_regression(lr: LogisticRegression) -> None:
    # sklearn 1.8+ dropped multi_class; sklearn <1.8 predict_proba still reads it.
    if not hasattr(lr, "multi_class"):
        lr.multi_class = "ovr"


def patch_sklearn_estimators(est) -> None:
    """Patch estimators in a loaded bundle for older sklearn runtimes."""
    if est is None:
        return
    if isinstance(est, LogisticRegression):
        _patch_logistic_regression(est)
        return
    calibrator = getattr(est, "calibrator_", None)
    if isinstance(calibrator, LogisticRegression):
        _patch_logistic_regression(calibrator)
    for cal in getattr(est, "calibrators_", None) or []:
        if isinstance(cal, LogisticRegression):
            _patch_logistic_regression(cal)
    meta = getattr(est, "meta", None)
    if isinstance(meta, LogisticRegression):
        _patch_logistic_regression(meta)


def patch_bundle_models(bundle: dict) -> None:
    """Patch all models referenced by a direction-model bundle dict."""
    if not isinstance(bundle, dict):
        return
    patch_sklearn_estimators(bundle.get("model"))
    for model in (bundle.get("models_by_ticker") or {}).values():
        patch_sklearn_estimators(model)
