# Olympus Model Audit Report

Generated: `2026-05-19T03:32:02.590614+00:00`

## Executive Verdict

**Deployment verdict:** `REJECT_FOR_TRADING`

**Portfolio verdict:** `PORTFOLIO_READY_MODEL_AUDIT_PLATFORM`

Olympus is currently best presented as an **honest ML research and model-audit platform**, not as a production trading system. The current signal does not show stable selection alpha, but the project now demonstrates the stronger engineering story: leakage-aware labels, realistic execution assumptions, baseline comparisons, cross-sectional ranking evaluation, raw-feature sanity checks, and walk-forward testing.

## System Snapshot

- Model: `/Users/ken/Documents/Olympus Project/models/direction_model.pkl`
- Model kind: `global`
- Target column: `target_intraday_next_direction`
- Feature set: `v10`
- Model features: `52`
- Dataset: `42,245` rows, `35` tickers, `1207` trading dates
- Date range: `2021-07-27` to `2026-05-15`
- Latest holdout AUC: `0.4780`
- Best target benchmark: `next_3d` at AUC `0.5034`

## Audit Checklist

| area | check | value | status | evidence |
| --- | --- | --- | --- | --- |
| Data coverage | Universe has enough rows and assets for cross-sectional tests | 42,245 rows / 35 tickers / 1207 dates | pass | 2021-07-27 to 2026-05-15 \| features v10 |
| Label integrity | Model uses realistic next-open target metadata | target_intraday_next_direction | pass | Bundle stores target_column and feature_set_version. |
| Feature health | Sparse news features are detected instead of trusted blindly | mean news zero frac 99.92% | monitor | Sparse-feature pruning is enabled in train_model.py with --max-zero-frac. |
| Holdout model quality | Latest model beats random ranking quality | ROC AUC 0.4780 | fail | Latest chronological holdout run from reports/metrics_log.csv. |
| Target benchmark | Best tested target shows meaningful separation | next_3d AUC 0.5034 | fail | All targets are trained on the same chronological split. |
| Backtest baseline | Top-ranked model basket beats SPY on holdout | model -1.34% vs SPY -3.98% | pass | Uses next-open-to-close execution and holdout-only scoring. |
| Ranking alpha | Top-ranked names beat SPY and bottom-ranked names | top-SPY Sharpe -2.706; top-bottom Sharpe -1.410 | fail | Cross-sectional ranking alpha report excludes SPY from the traded universe. |
| Raw feature sanity | Standalone feature sweep finds a robust simple signal | hl_range_mean_5d:high spread Sharpe -0.693 | fail | Every numeric model feature is tested both high and low. |
| Walk-forward | Retrained folds remain above random | default mean AUC 0.5078 | fail | Each fold trains on past data and scores the next unseen block. |
| Longer-horizon research | 3-day model improves enough to justify further research | 3d WF mean AUC 0.5170 | fail | 3d returns are overlapping diagnostics, not final portfolio stats. |

## Status Counts

- Pass: `3`
- Monitor: `1`
- Missing: `0`
- Fail: `6`

## Interpretation

The project is compelling because it shows the full lifecycle of an ML system under pressure:

1. Build an end-to-end prediction pipeline.
2. Discover that attractive initial metrics can be misleading.
3. Add safeguards for target leakage, execution realism, benchmark fairness, and walk-forward validation.
4. Reject the current signal when it fails those tests.
5. Produce reproducible reports that explain exactly why the model is or is not deployable.

That is the differentiated portfolio story: **the tool tells the truth even when the model is not ready.**

## Recommended Next Research

- Add sector-relative and beta-adjusted features.
- Add VIX, rates, credit-spread, and market-regime context.
- Build a non-overlapping multi-day portfolio simulator.
- Replace sparse news input with broader timestamped coverage or remove news features from the active feature list.
- Use the audit checklist as the promotion gate before any paper-trading or live-trading claim.
