import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import pearsonr
from utils.platt_calibration import (
    CrossFitCalibratedClassifier,
    PlattCalibratedBinaryClassifier,
)
from utils.predict_bundle import add_pred_prob
from xgboost import XGBClassifier, XGBRegressor

DATA_PATH = ROOT / "data" / "features.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "direction_model.pkl"
REPORT_DIR = ROOT / "reports"
METRICS_LOG = REPORT_DIR / "metrics_log.csv"

FEATURE_COLUMNS = [
    # returns & momentum
    "return_1d",
    "return_5d",
    "return_10d",
    "gap_pct",
    "momentum_3d",
    "momentum_10d",
    "momentum_20d",
    # realized vol
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "vol_5_over_20",
    "vol_10d_over_20",
    "vol_ratio_5_10",
    "vol_of_vol_20d",
    # technical
    "macd_line",
    "hl_range_mean_5d",
    "atr_14",
    "bollinger_pctb",
    # lagged returns
    "return_1d_lag1",
    "return_1d_lag2",
    "return_1d_lag3",
    "return_5d_lag1",
    # MAs & position vs trend
    "ma_5",
    "ma_20",
    "ma_50",
    "dist_ma_20",
    "dist_ma_50",
    # oscillators & range
    "rsi_14",
    "volume_ratio",
    "hl_range_pct",
    "intraday_return",
    "zscore_close_20d",
    # microstructure
    "overnight_return",
    "close_vs_range",
    # relative strength
    "rel_ratio",
    "rel_mom_20",
    # cross-sectional ranks
    "cs_rank_return_5d",
    "cs_rank_momentum_20d",
    "cs_rank_volume_ratio",
    "cs_rank_rsi",
    "cs_rank_vol",
    "cs_rank_sentiment",
    # calendar
    "day_of_week",
    "month_of_year",
    "quarter",
    "is_month_start",
    "is_month_end",
    # cross-asset
    "spy_return_1d",
    "spy_vol_10d",
    "excess_return_1d",
    "gld_slv_ratio",
    # news / sentiment
    "news_count_6h",
    "news_count_24h",
    "news_count_3d",
    "sentiment_mean_6h",
    "sentiment_mean_24h",
    "sentiment_mean_3d",
    "sentiment_max_6h",
    "sentiment_min_6h",
    "sentiment_max_24h",
    "sentiment_min_24h",
    "weighted_sentiment_6h",
    "weighted_sentiment_24h",
    "weighted_sentiment_3d",
    "has_news",
    # interactions
    "sentiment_x_vol",
    "excess_x_volume",
    "return_over_vol",
]

TARGET_CHOICES = {
    "next_intraday": "target_intraday_next_direction",
    "next_3d": "target_direction_next_open_to_close_3d",
    "next_5d": "target_direction_next_open_to_close_5d",
    "direction": "target_direction",
    "excess": "target_excess_up",
    "direction_5d": "target_direction_5d",
}
REG_TARGET_CHOICES = (
    "target_return_next_open_to_close",
    "target_return_next_open_to_close_3d",
    "target_return_next_open_to_close_5d",
    "target_return_1d",
    "target_return_5d",
)


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values(["Date", "ticker"])
    dates = pd.Series(df["Date"].unique()).sort_values()
    n = len(dates)
    if n < 5:
        raise ValueError("Not enough unique dates to split")
    k = int(n * (1.0 - test_frac))
    if k <= 0:
        k = 1
    if k >= n:
        k = n - 1
    split_date = dates.iloc[k]
    train = df[df["Date"] < split_date]
    test = df[df["Date"] >= split_date]
    return train, test, split_date


def train_triple_split(
    train_df: pd.DataFrame,
    fit_frac: float = 0.62,
    es_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Disjoint train windows by calendar date: fit, early-stop eval, calibration."""
    dates = pd.Series(train_df["Date"].unique()).sort_values()
    n = len(dates)
    if n < 18:
        i1 = max(2, int(n * 0.7))
        i2 = max(i1 + 1, int(n * 0.88))
        d_fit = set(dates[:i1])
        d_es = set(dates[i1:i2])
        d_cal = set(dates[i2:])
    else:
        i1 = int(n * fit_frac)
        i2 = int(n * (fit_frac + es_frac))
        i1 = max(1, min(i1, n - 3))
        i2 = max(i1 + 1, min(i2, n - 1))
        d_fit = set(dates[:i1])
        d_es = set(dates[i1:i2])
        d_cal = set(dates[i2:])
    df_f = train_df[train_df["Date"].isin(d_fit)]
    df_e = train_df[train_df["Date"].isin(d_es)]
    df_c = train_df[train_df["Date"].isin(d_cal)]
    return df_f, df_e, df_c


def variance_prune(cols: list[str], X: pd.DataFrame, min_std: float = 1e-12) -> list[str]:
    keep = []
    for c in cols:
        s = X[c].std()
        if pd.isna(s) or float(s) < min_std:
            continue
        keep.append(c)
    return keep if keep else cols[:]


def sparsity_prune(
    cols: list[str],
    X: pd.DataFrame,
    max_zero_frac: float | None = 0.995,
    zero_epsilon: float = 1e-12,
) -> list[str]:
    """Drop features that are almost always zero in the fit window."""
    if max_zero_frac is None or max_zero_frac >= 1.0:
        return cols[:]
    if max_zero_frac < 0:
        raise ValueError("max_zero_frac must be non-negative")

    keep: list[str] = []
    for c in cols:
        s = pd.to_numeric(X[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        zero_frac = float((s.abs() <= zero_epsilon).mean())
        if zero_frac > max_zero_frac:
            continue
        keep.append(c)
    return keep if keep else cols[:]


def shap_select(
    cols: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    min_features: int = 10,
    top_k: int | None = None,
) -> list[str]:
    """
    Train a quick XGBoost, compute mean |SHAP| on eval set, keep features above the mean
    threshold (or top_k if specified). Falls back to full list on failure.
    """
    try:
        import shap
    except ImportError:
        return cols

    from xgboost import XGBClassifier

    quick = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=1,
        eval_metric="logloss",
    )
    try:
        quick.fit(X_train[cols], y_train, verbose=False)
    except Exception:
        return cols

    try:
        explainer = shap.TreeExplainer(quick)
        sv = explainer.shap_values(X_eval[cols])
        if isinstance(sv, list):
            sv = sv[1]
        importance = np.abs(sv).mean(axis=0)
    except Exception:
        return cols

    imp_series = pd.Series(importance, index=cols).sort_values(ascending=False)

    if top_k is not None:
        keep = list(imp_series.head(max(top_k, min_features)).index)
    else:
        threshold = float(imp_series.mean())
        keep = list(imp_series[imp_series >= threshold].index)
        if len(keep) < min_features:
            keep = list(imp_series.head(min_features).index)

    return keep if keep else cols


def effective_scale_pos_weight(
    y: pd.Series,
    override: float | None,
    max_auto: float | None = None,
) -> float:
    """
    XGBoost scale_pos_weight for binary classification.
    If override is set, use it. Otherwise max(1, n0/n1), optionally capped by max_auto.
    """
    if override is not None:
        return float(override)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n1 == 0:
        return 1.0
    w = max(1.0, n0 / n1)
    if max_auto is not None:
        w = min(w, float(max_auto))
    return w


def recency_sample_weight(dates: pd.Series, half_life_days: float) -> np.ndarray:
    mx = pd.Timestamp(dates.max()).normalize()
    age = (mx - pd.to_datetime(dates).dt.normalize()).dt.days.astype(float)
    return np.exp(-np.log(2) * age / max(half_life_days, 1e-6)).values


def _grid_best_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, lo: float = 0.48, hi: float = 0.62, n: int = 29
) -> tuple[float, float]:
    """Maximize accuracy vs a single probability threshold (exploratory; holdout-only)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(lo, hi, n):
        pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = float(acc)
            best_t = float(t)
    return best_t, best_acc


def build_xgb_core_params(light: bool, xgb_extra: dict | None) -> dict:
    """Merge defaults with CLI overrides (used for both classifier and regressor)."""
    n_est = 320 if light else 600
    depth = 3 if light else 4
    params = {
        "n_estimators": n_est,
        "max_depth": depth,
        "learning_rate": 0.03,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "colsample_bylevel": 0.8,
        "min_child_weight": 6.0,
        "reg_lambda": 2.5,
        "reg_alpha": 0.15,
        "gamma": 0.1,
        "random_state": 42,
        "n_jobs": 1,
    }
    if xgb_extra:
        params.update(xgb_extra)
    return params


def build_classifier(
    scale_pw: float,
    early_rounds: int | None,
    light: bool,
    xgb_extra: dict | None,
) -> XGBClassifier:
    params = build_xgb_core_params(light, xgb_extra)
    params["eval_metric"] = "logloss"
    params["scale_pos_weight"] = scale_pw
    if early_rounds is not None:
        params["early_stopping_rounds"] = early_rounds
    return XGBClassifier(**params)


def build_regressor(
    early_rounds: int | None,
    light: bool,
    xgb_extra: dict | None,
) -> XGBRegressor:
    params = build_xgb_core_params(light, xgb_extra)
    if early_rounds is not None:
        params["early_stopping_rounds"] = early_rounds
    return XGBRegressor(**params)


def fit_xgb(
    scale_pw: float,
    X_tr,
    y_tr,
    X_es,
    y_es,
    sample_weight=None,
    light: bool = False,
    xgb_extra: dict | None = None,
    early_stopping_rounds: int | None = 45,
):
    if len(X_es) >= 80:
        clf = build_classifier(scale_pw, early_stopping_rounds, light, xgb_extra)
        clf.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weight,
            eval_set=[(X_es, y_es)],
            verbose=False,
        )
    else:
        clf = build_classifier(scale_pw, None, light, xgb_extra)
        clf.fit(X_tr, y_tr, sample_weight=sample_weight, verbose=False)
    return clf


def fit_xgb_reg(
    X_tr,
    y_tr,
    X_es,
    y_es,
    sample_weight=None,
    light: bool = False,
    xgb_extra: dict | None = None,
    early_stopping_rounds: int | None = 45,
):
    if len(X_es) >= 80:
        reg = build_regressor(early_stopping_rounds, light, xgb_extra)
        reg.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weight,
            eval_set=[(X_es, y_es)],
            verbose=False,
        )
    else:
        reg = build_regressor(None, light, xgb_extra)
        reg.fit(X_tr, y_tr, sample_weight=sample_weight, verbose=False)
    return reg


def maybe_calibrate(
    clf, X_cal, y_cal, min_rows: int, enabled: bool,
    method: str = "platt", cross_fit: bool = False,
):
    if not enabled or len(X_cal) < min_rows:
        return clf, False
    if y_cal.nunique() < 2:
        return clf, False
    try:
        if cross_fit and len(X_cal) >= min_rows * 2:
            wrapped = CrossFitCalibratedClassifier(clf, method=method).fit_calibrator(X_cal, y_cal)
        else:
            wrapped = PlattCalibratedBinaryClassifier(clf, method=method).fit_calibrator(X_cal, y_cal)
        return wrapped, True
    except (ValueError, Exception):
        return clf, False


def booster_for_importance(model):
    if isinstance(model, PlattCalibratedBinaryClassifier):
        return model.base
    return model


def train_one_stack(
    train_df: pd.DataFrame,
    cols: list[str],
    calibrate: bool,
    cal_min_rows: int,
    use_recency: bool,
    recency_hl: float,
    y_col: str,
    task: str,
    light: bool,
    xgb_extra: dict | None,
    early_stopping_rounds: int | None,
    scale_pos_weight_override: float | None = None,
    max_scale_pos_weight: float | None = None,
    cal_method: str = "platt",
    cal_cross_fit: bool = False,
):
    df_f, df_e, df_c = train_triple_split(train_df)
    sw = None
    if use_recency:
        sw = recency_sample_weight(df_f["Date"], recency_hl)
    if task == "regression":
        reg = fit_xgb_reg(
            df_f[cols],
            df_f[y_col],
            df_e[cols],
            df_e[y_col],
            sample_weight=sw,
            light=light,
            xgb_extra=xgb_extra,
            early_stopping_rounds=early_stopping_rounds,
        )
        return reg, False
    pw = effective_scale_pos_weight(df_f[y_col], scale_pos_weight_override, max_scale_pos_weight)
    clf = fit_xgb(
        pw,
        df_f[cols],
        df_f[y_col],
        df_e[cols],
        df_e[y_col],
        sample_weight=sw,
        light=light,
        xgb_extra=xgb_extra,
        early_stopping_rounds=early_stopping_rounds,
    )
    final_m, did_cal = maybe_calibrate(
        clf, df_c[cols], df_c[y_col], cal_min_rows, calibrate,
        method=cal_method, cross_fit=cal_cross_fit,
    )
    return final_m, did_cal


def train_per_ticker(
    train_df: pd.DataFrame,
    cols: list[str],
    min_rows: int,
    calibrate: bool,
    cal_min_rows_global: int,
    cal_min_rows_ticker: int,
    use_recency: bool,
    recency_hl: float,
    y_col: str,
    task: str,
    light: bool,
    xgb_extra: dict | None,
    early_stopping_rounds: int | None,
    scale_pos_weight_override: float | None = None,
    max_scale_pos_weight: float | None = None,
    cal_method: str = "platt",
    cal_cross_fit: bool = False,
):
    global_model, g_cal = train_one_stack(
        train_df,
        cols,
        calibrate,
        cal_min_rows_global,
        use_recency,
        recency_hl,
        y_col,
        task,
        light,
        xgb_extra,
        early_stopping_rounds,
        scale_pos_weight_override,
        max_scale_pos_weight,
        cal_method=cal_method,
        cal_cross_fit=cal_cross_fit,
    )
    models_by_ticker: dict = {}

    for t in sorted(train_df["ticker"].astype(str).str.upper().unique()):
        sub = train_df[train_df["ticker"].astype(str).str.upper() == t]
        if len(sub) < min_rows:
            models_by_ticker[t] = None
            continue
        df_f, df_e, df_c = train_triple_split(sub)
        if len(df_f) < min_rows // 2:
            models_by_ticker[t] = None
            continue
        sw = None
        if use_recency:
            sw = recency_sample_weight(df_f["Date"], recency_hl)
        if task == "regression":
            reg = fit_xgb_reg(
                df_f[cols],
                df_f[y_col],
                df_e[cols],
                df_e[y_col],
                sample_weight=sw,
                light=light,
                xgb_extra=xgb_extra,
                early_stopping_rounds=early_stopping_rounds,
            )
            models_by_ticker[t] = reg
            continue
        pw = effective_scale_pos_weight(df_f[y_col], scale_pos_weight_override, max_scale_pos_weight)
        clf = fit_xgb(
            pw,
            df_f[cols],
            df_f[y_col],
            df_e[cols],
            df_e[y_col],
            sample_weight=sw,
            light=light,
            xgb_extra=xgb_extra,
            early_stopping_rounds=early_stopping_rounds,
        )
        final_m, _ = maybe_calibrate(
            clf, df_c[cols], df_c[y_col], cal_min_rows_ticker, calibrate,
            method=cal_method, cross_fit=cal_cross_fit,
        )
        models_by_ticker[t] = final_m

    return global_model, models_by_ticker, g_cal


def append_metrics_log(row: dict):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_LOG
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train direction or return model")
    parser.add_argument(
        "--no-per-ticker",
        action="store_true",
        help="Single global model only",
    )
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--no-calibration", action="store_true", help="Skip sigmoid calibration on hold-in cal window")
    parser.add_argument(
        "--cal-method",
        choices=["platt", "isotonic"],
        default="platt",
        help="Calibration method: platt (sigmoid, default) or isotonic (non-parametric)",
    )
    parser.add_argument(
        "--cal-cross-fit",
        action="store_true",
        help="Use K-fold cross-fit calibration (reduces overfitting on cal window)",
    )
    parser.add_argument(
        "--cal-min-rows",
        type=int,
        default=450,
        help="Min rows in calibration window for global calibration",
    )
    parser.add_argument(
        "--cal-min-rows-ticker",
        type=int,
        default=180,
        help="Min rows in calibration window for per-ticker calibration",
    )
    parser.add_argument("--no-recency", action="store_true", help="Disable exponential recency sample weights on fit window")
    parser.add_argument(
        "--recency-half-life-days",
        type=float,
        default=252.0,
        help="Half-life (trading days) for recency weights on oldest fit rows",
    )
    parser.add_argument("--min-rows-per-ticker", type=int, default=380)
    parser.add_argument(
        "--target",
        choices=list(TARGET_CHOICES.keys()),
        default="next_intraday",
        help=(
            "Classification label: next_intraday (next open->close, default), "
            "next_3d/next_5d (next open->future close), direction (close->next close), "
            "excess (beat SPY next day; SPY excluded), direction_5d"
        ),
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="classification (default) or regression on next-day / 5d return",
    )
    parser.add_argument(
        "--reg-target",
        choices=list(REG_TARGET_CHOICES),
        default="target_return_next_open_to_close",
        help="Regression label column (when --task regression)",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Smaller / faster XGBoost (fewer trees, shallower)",
    )
    parser.add_argument(
        "--features-csv",
        type=str,
        default=str(DATA_PATH),
        help="Path to features CSV (default: data/features.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH),
        help="Where to save the joblib bundle (default: models/direction_model.pkl)",
    )
    parser.add_argument("--learning-rate", type=float, default=0.03, help="XGBoost eta")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Override tree count (default: 600, or 320 with --light)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Override max depth (default: 4, or 3 with --light)",
    )
    parser.add_argument("--min-child-weight", type=float, default=6.0)
    parser.add_argument("--reg-lambda", type=float, default=2.5, help="XGBoost reg_lambda (L2)")
    parser.add_argument("--reg-alpha", type=float, default=0.15, help="XGBoost reg_alpha (L1)")
    parser.add_argument("--gamma", type=float, default=0.1, help="XGBoost min split gain")
    parser.add_argument("--subsample", type=float, default=0.75)
    parser.add_argument("--colsample-bytree", type=float, default=0.75, dest="colsample_bytree")
    parser.add_argument("--colsample-bylevel", type=float, default=0.8, dest="colsample_bylevel")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=45,
        dest="early_stopping_rounds",
        help="Early stopping on eval split; 0 disables when eval set exists",
    )
    parser.add_argument(
        "--variance-min-std",
        type=float,
        default=1e-12,
        help="Drop features with std below this in the fit window (variance prune)",
    )
    parser.add_argument(
        "--max-zero-frac",
        type=float,
        default=0.995,
        help=(
            "Drop features with a zero fraction above this in the fit window "
            "(default 0.995; use 1.0 to disable sparse-feature pruning)"
        ),
    )
    parser.add_argument(
        "--zero-epsilon",
        type=float,
        default=1e-12,
        help="Absolute value treated as zero for --max-zero-frac pruning",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Preset: slower learning, stronger regularization, shallower trees (often better OOS; overrides matching defaults)",
    )
    parser.add_argument(
        "--feature-select",
        action="store_true",
        help="SHAP-based feature selection: train a quick model, keep features with above-average |SHAP|",
    )
    parser.add_argument(
        "--feature-select-top-k",
        type=int,
        default=None,
        metavar="K",
        help="When --feature-select, keep top K features by SHAP importance (default: mean threshold)",
    )
    parser.add_argument(
        "--scale-pos-weight",
        type=str,
        default="auto",
        metavar="AUTO|FLOAT",
        help="XGBoost scale_pos_weight: 'auto' (default: max(1,n0/n1), see --max-scale-pos-weight) or explicit positive float (e.g. 1.0 = no imbalance reweighting)",
    )
    parser.add_argument(
        "--max-scale-pos-weight",
        type=float,
        default=None,
        metavar="FLOAT",
        help="When --scale-pos-weight is auto, cap the computed weight (e.g. 3.0)",
    )
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        metavar="PATH",
        help="JSON file from tune_model.py; XGB params override CLI defaults (--robust still wins if both set)",
    )
    args = parser.parse_args()

    task = args.task
    y_col = args.reg_target if task == "regression" else TARGET_CHOICES[args.target]
    light = args.light

    xgb_extra: dict = {
        "learning_rate": args.learning_rate,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "gamma": args.gamma,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "colsample_bylevel": args.colsample_bylevel,
    }
    if args.params_json:
        pj = json.loads(Path(args.params_json).read_text())
        tuned = pj.get("params", pj)
        xgb_extra.update({k: v for k, v in tuned.items() if k in (
            "max_depth", "learning_rate", "n_estimators", "subsample",
            "colsample_bytree", "colsample_bylevel", "min_child_weight",
            "reg_lambda", "reg_alpha", "gamma",
        )})
        print("Loaded tuned params from", args.params_json)

    if args.n_estimators is not None:
        xgb_extra["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        xgb_extra["max_depth"] = args.max_depth

    if args.robust:
        xgb_extra.update(
            {
                "learning_rate": 0.02,
                "min_child_weight": 10.0,
                "reg_lambda": 4.0,
                "reg_alpha": 0.2,
                "gamma": 0.25,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.75,
            }
        )
        if args.n_estimators is None:
            xgb_extra["n_estimators"] = 400 if light else 700
        if args.max_depth is None:
            xgb_extra["max_depth"] = 3

    es_rounds: int | None = None if args.early_stopping_rounds == 0 else args.early_stopping_rounds

    spw_s = (args.scale_pos_weight or "").strip().lower()
    if spw_s == "auto":
        spw_override: float | None = None
    else:
        try:
            spw_override = float(spw_s)
        except ValueError:
            parser.error("--scale-pos-weight must be 'auto' or a positive float")
        if spw_override <= 0:
            parser.error("--scale-pos-weight must be positive")

    data_path = Path(args.features_csv).resolve()
    model_out = Path(args.model_path).resolve()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path, parse_dates=["Date"])
    if y_col not in df.columns:
        raise SystemExit(f"Missing target column {y_col}; run scripts/build_features.py with the current code first.")

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    need_cols = available + ["ticker", y_col]
    df = df.dropna(subset=[c for c in need_cols if c in df.columns])

    train_df, test_df, cutoff = time_based_split(df, test_frac=args.test_frac)

    df_f0, _, _ = train_triple_split(train_df)
    cols = variance_prune(available, df_f0[available], min_std=args.variance_min_std)
    n_before_sparse = len(cols)
    cols = sparsity_prune(
        cols,
        df_f0[cols],
        max_zero_frac=args.max_zero_frac,
        zero_epsilon=args.zero_epsilon,
    )
    n_after_sparse = len(cols)
    n_before_shap = len(cols)
    if args.feature_select and task == "classification":
        _, df_e_sel, _ = train_triple_split(train_df)
        if len(df_e_sel) > 30:
            cols = shap_select(
                cols, df_f0[cols], df_f0[y_col], df_e_sel[cols],
                top_k=args.feature_select_top_k,
            )
            print(f"SHAP feature selection: {n_before_shap} → {len(cols)}")
    print("Target:", y_col, "| task:", task, "| light:", light)
    print("Features CSV:", data_path)
    print("Features after variance prune:", n_before_sparse, "(dropped", len(available) - n_before_sparse, ")")
    print("Sparse-feature prune:", f"max_zero_frac={args.max_zero_frac}", "(dropped", n_before_sparse - n_after_sparse, ")")
    print("XGB overrides:", {k: v for k, v in xgb_extra.items()}, "| early_stopping_rounds:", es_rounds)
    if args.robust:
        print("Preset: --robust (stronger regularization, tuned for out-of-sample stability)")
    spw_note = "auto"
    if spw_override is not None:
        spw_note = str(spw_override)
    elif args.max_scale_pos_weight is not None:
        spw_note = f"auto capped at {args.max_scale_pos_weight}"
    print("scale_pos_weight:", spw_note)

    X_test = test_df[cols]
    y_test = test_df[y_col]

    use_per_ticker = not args.no_per_ticker
    cal_on = not args.no_calibration and task == "classification"
    rec_on = not args.no_recency

    if use_per_ticker:
        global_model, models_by_ticker, g_cal = train_per_ticker(
            train_df,
            cols,
            args.min_rows_per_ticker,
            cal_on,
            args.cal_min_rows,
            args.cal_min_rows_ticker,
            rec_on,
            args.recency_half_life_days,
            y_col,
            task,
            light,
            xgb_extra,
            es_rounds,
            spw_override,
            args.max_scale_pos_weight,
            cal_method=args.cal_method,
            cal_cross_fit=args.cal_cross_fit,
        )
        bundle = {
            "model": global_model,
            "model_kind": "per_ticker",
            "models_by_ticker": models_by_ticker,
            "feature_names": cols,
            "feature_set_version": str(df["feature_set_version"].iloc[0])
            if "feature_set_version" in df.columns
            else "v7",
            "train_cutoff_date": str(cutoff.date()),
            "test_first_date": str(test_df["Date"].min().date()) if len(test_df) else None,
            "training_notes": f"per_ticker_{task}_{y_col}",
            "scale_pos_weight_override": spw_override,
            "max_scale_pos_weight": args.max_scale_pos_weight,
            "xgb_extra": dict(xgb_extra),
            "early_stopping_rounds": es_rounds,
            "max_zero_frac": args.max_zero_frac,
            "zero_epsilon": args.zero_epsilon,
            "calibrated_global": g_cal,
            "recency_weights": rec_on,
            "task": task,
            "target_column": y_col,
            "return_prob_scale": 30.0,
        }
        scored = add_pred_prob(test_df, bundle)
        y_prob = scored["pred_prob"].values
        imp_src = booster_for_importance(global_model)
    else:
        global_model, g_cal = train_one_stack(
            train_df,
            cols,
            cal_on,
            args.cal_min_rows,
            rec_on,
            args.recency_half_life_days,
            y_col,
            task,
            light,
            xgb_extra,
            es_rounds,
            spw_override,
            args.max_scale_pos_weight,
            cal_method=args.cal_method,
            cal_cross_fit=args.cal_cross_fit,
        )
        bundle = {
            "model": global_model,
            "model_kind": "global",
            "models_by_ticker": None,
            "feature_names": cols,
            "feature_set_version": str(df["feature_set_version"].iloc[0])
            if "feature_set_version" in df.columns
            else "v7",
            "train_cutoff_date": str(cutoff.date()),
            "test_first_date": str(test_df["Date"].min().date()) if len(test_df) else None,
            "training_notes": f"global_{task}_{y_col}",
            "scale_pos_weight_override": spw_override,
            "max_scale_pos_weight": args.max_scale_pos_weight,
            "xgb_extra": dict(xgb_extra),
            "early_stopping_rounds": es_rounds,
            "max_zero_frac": args.max_zero_frac,
            "zero_epsilon": args.zero_epsilon,
            "calibrated_global": g_cal,
            "recency_weights": rec_on,
            "task": task,
            "target_column": y_col,
            "return_prob_scale": 30.0,
        }
        scored = add_pred_prob(test_df, bundle)
        y_prob = scored["pred_prob"].values
        imp_src = booster_for_importance(global_model)

    if task == "regression":
        y_hat = scored["pred_return"].values
        rmse = float(np.sqrt(mean_squared_error(y_test, y_hat)))
        corr, _ = pearsonr(y_test, y_hat) if len(y_test) > 5 else (float("nan"), None)
        print("Train rows:", len(train_df), "Test rows:", len(test_df))
        print("Split date (train < / test >=):", cutoff.date())
        print("RMSE:", rmse, " Pearson r:", corr)
        y_pred = (y_prob >= 0.5).astype(int)
        y_bin = (y_test > 0).astype(int) if y_col.startswith("target_return") else None
        if y_bin is not None:
            print("Accuracy (sign match):", accuracy_score(y_bin, (y_hat > 0).astype(int)))
    else:
        y_pred = (y_prob >= 0.5).astype(int)
        print("Train rows:", len(train_df), "Test rows:", len(test_df))
        print("Split date (train < / test >=):", cutoff.date())
        print("Calibration (global stack):", g_cal if use_per_ticker else bundle["calibrated_global"])
        print("Recency sample weights:", rec_on)
        maj = float(max((y_test == 0).mean(), (y_test == 1).mean()))
        print("Accuracy @0.5:", accuracy_score(y_test, y_pred))
        print("Balanced accuracy @0.5:", balanced_accuracy_score(y_test, y_pred))
        print("F1 @0.5:", f1_score(y_test, y_pred, zero_division=0))
        print("Precision / Recall @0.5:", precision_score(y_test, y_pred, zero_division=0), "/", recall_score(y_test, y_pred, zero_division=0))
        print("Majority-class baseline accuracy:", round(maj, 4), "(always predict the more common class)")
        print("ROC AUC:", roc_auc_score(y_test, y_prob))
        try:
            print("Log loss:", log_loss(y_test, y_prob))
            print("Brier:", brier_score_loss(y_test, y_prob))
        except Exception:
            pass
        best_t, best_acc = _grid_best_threshold(y_test.values, y_prob)
        print(
            f"Best single threshold on holdout (exploratory only): t={best_t:.3f} → accuracy {best_acc:.4f} "
            "(use walk-forward / fresh data to choose rules; see scripts/suggest_thresholds.py)"
        )

    imp = pd.Series(imp_src.feature_importances_, index=cols).sort_values(ascending=False)
    print("\nFeature importance (top 15):")
    for name, val in imp.head(15).items():
        print(f"  {name}: {val:.5f}")

    bundle["feature_importance"] = imp.to_dict()
    bundle["features_csv"] = str(data_path)
    joblib.dump(bundle, model_out)
    print(f"\nSaved bundle to {model_out}")

    log_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_kind": bundle["model_kind"],
        "train_cutoff": bundle["train_cutoff_date"],
        "n_features": len(cols),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "task": task,
        "target": y_col,
        "calibrated_global": bool(bundle.get("calibrated_global")) if task == "classification" else False,
        "recency_weights": bool(rec_on),
    }
    if task == "regression":
        y_hat = scored["pred_return"].values
        rmse = float(np.sqrt(mean_squared_error(y_test, y_hat)))
        try:
            corr, _ = pearsonr(y_test, y_hat) if len(y_test) > 5 else (float("nan"), None)
        except Exception:
            corr = float("nan")
        log_row.update(
            {
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
                "f1": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "majority_baseline": float("nan"),
                "roc_auc": float("nan"),
                "brier": float("nan"),
                "log_loss": float("nan"),
                "rmse": rmse,
                "pearson_r": float(corr),
            }
        )
    else:
        try:
            ll = float(log_loss(y_test, np.clip(y_prob, 1e-15, 1 - 1e-15)))
        except Exception:
            ll = float("nan")
        try:
            br = float(brier_score_loss(y_test, y_prob))
        except Exception:
            br = float("nan")
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            auc = float("nan")
        maj = float(max((y_test == 0).mean(), (y_test == 1).mean()))
        log_row.update(
            {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "majority_baseline": maj,
                "roc_auc": auc,
                "brier": br,
                "log_loss": ll,
                "rmse": float("nan"),
                "pearson_r": float("nan"),
            }
        )
    append_metrics_log(log_row)
    print(f"Appended run to {METRICS_LOG}")


if __name__ == "__main__":
    main()
