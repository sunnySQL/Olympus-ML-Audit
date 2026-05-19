"""Calibration wrappers for binary classifiers: Platt (sigmoid) and isotonic."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


class PlattCalibratedBinaryClassifier:
    def __init__(self, base, method: str = "platt"):
        self.base = base
        self.method = method
        self.calibrator_: LogisticRegression | IsotonicRegression | None = None
        self.classes_ = np.array([0, 1])

    def fit_calibrator(self, X, y) -> "PlattCalibratedBinaryClassifier":
        p = self.base.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self.calibrator_.fit(p, y)
        else:
            self.calibrator_ = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
            self.calibrator_.fit(p.reshape(-1, 1), y)
        return self

    def predict_proba(self, X):
        p = self.base.predict_proba(X)[:, 1]
        if isinstance(self.calibrator_, IsotonicRegression):
            cal_p = np.clip(self.calibrator_.predict(p), 0, 1)
        else:
            cal_p = self.calibrator_.predict_proba(p.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - cal_p, cal_p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return self.base.feature_importances_


class CrossFitCalibratedClassifier:
    """
    K-fold cross-fit calibration: train K calibrators on out-of-fold predictions,
    then average at inference. Reduces overfitting vs a single calibration window.
    """

    def __init__(self, base, method: str = "platt", n_folds: int = 3):
        self.base = base
        self.method = method
        self.n_folds = n_folds
        self.calibrators_: list = []
        self.classes_ = np.array([0, 1])

    def fit_calibrator(self, X, y) -> "CrossFitCalibratedClassifier":
        p = self.base.predict_proba(X)[:, 1]
        y_arr = np.asarray(y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        self.calibrators_ = []
        for train_idx, _ in kf.split(p):
            if self.method == "isotonic":
                cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                cal.fit(p[train_idx], y_arr[train_idx])
            else:
                cal = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
                cal.fit(p[train_idx].reshape(-1, 1), y_arr[train_idx])
            self.calibrators_.append(cal)
        return self

    def predict_proba(self, X):
        p = self.base.predict_proba(X)[:, 1]
        preds = []
        for cal in self.calibrators_:
            if isinstance(cal, IsotonicRegression):
                preds.append(np.clip(cal.predict(p), 0, 1))
            else:
                preds.append(cal.predict_proba(p.reshape(-1, 1))[:, 1])
        avg = np.mean(preds, axis=0)
        return np.column_stack([1 - avg, avg])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return self.base.feature_importances_
