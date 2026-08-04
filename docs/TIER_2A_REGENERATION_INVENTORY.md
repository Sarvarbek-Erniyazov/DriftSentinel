# Tier 2A.1 — target switch regeneration inventory

**Change:** `readmitted_binary` moves from the merged target
(`{"NO":0, "<30":1, ">30":1}`, 46.1% prevalence) to **30-day readmission**
(`{"NO":0, "<30":1, ">30":0}`, **11.2% prevalence**).

Single source of truth: `TARGET_BINARY_MAP` in `src/data/preprocessor.py:122`.
Every module reads the derived column rather than re-deriving it, so the map is
the only definition that changes. `readmitted_multi` is unchanged and remains the
3-class view.

**Purpose of this inventory:** the switch changes essentially every number in the
repository, including ones corrected in Tiers 1.1–1.5. Nothing may be left
carrying a merged-target number under a 30-day label.

---

## A. Artifacts that MUST be regenerated (carry merged-target numbers)

Listed in dependency order. Each stage consumes the previous stage's output.

| # | Producer | Artifacts | Why |
|---|---|---|---|
| 1 | `pipelines/pipeline.py` | `data/train/{train,val,test}_fs.parquet`, `data/production/*.parquet`, `outputs/artifacts/*` (encoders, FE stats, **selector**, `selected_features.json`, `feature_scores.csv`), `outputs/log/consistency_report.json`, **`pipeline_summary.json`** | The parquets embed the target column. **Feature selection is target-dependent** — MI, SHAP and Boruta are all computed against `readmitted_binary`, so the 53 selected features will change |
| 2 | `models/trainer.py` | `lgbm_v1.pkl`, `logreg_v1.pkl`, `logreg_scaler.pkl`, `training_summary.json`, `lgbm_feature_importance.csv` | Trained on the new target and the new feature set |
| 3 | `models/evaluator.py` | `evaluation_report.json`, figures 20–24 | All metrics; also supplies the operating threshold |
| 4 | `uncertainty/calibration.py` | `calibrator_isotonic_*.pkl`, `calibrator_temperature_*.pkl`, figures 21/25 | Calibration is target-conditional |
| 5 | `uncertainty/quantifier.py` | `conformal_predictor_*.pkl`, `conformal_report_*.json`, figures 26–27 | Prediction sets change with prevalence |
| 6 | `models/registry.py` | `model_registry.json`, `registry_history.csv`, `model_comparison.json`, `lgbm_v2.pkl` | **Tier 1.1** — DeLong/bootstrap must be recomputed on the new target |
| 7 | `drift/data_drift.py`, `feature_drift.py` | `data_drift_*.csv/json`, `feature_drift_*` | Feature set changed; impact scores are target-weighted |
| 8 | `drift/concept_drift.py` | `concept_drift_val_test.json`, `sliding_windows_*.csv` | **Tier 1.4** — the label-drift rule is prevalence-scaled; this is the case it was rebuilt for |
| 9 | `drift/alerting.py` | `alert_report_val_test.json`, `alert_summary_val_test.csv` | **Tier 1.5** — evidence count and severities |
| 10 | `uncertainty/threshold.py` + `threshold_policy.py` | `threshold_report_*.json`, **`threshold_policy_lgbm_v1.json`**, figures 28–29, 44 | **Tier 1.3** — and the rate-form defect matters *here*: at 11.2% prevalence it inflates 5:1 into an effective 39.8:1 |
| 11 | `adversarial/{attacks,robustness,defense}.py` | `attack_report`, `robustness_report`, **`defense_report_lgbm_v1.json`**, figures 30–31, 43 | **Tier 1.2** — confusion matrices and ROC are computed against the target |
| 12 | `monitoring/health_check.py` | `health_check_report.json` | **Phase 1.0** — reads all of the above |

## B. Artifacts that do NOT need regeneration

| Artifact | Why it is safe |
|---|---|
| `outputs/reports/temporal_validity.json`, figures 32–39 | Operates on the **raw CSV**, independent of the pipeline. Already uses `<30` as the primary target (`configs/temporal_validity.yaml: primary_label: lt30`); the merged target appears only as an explicitly-labelled secondary contrast |
| `outputs/reports/regime_*.json`, `regime_matrix.csv`, figures 40–42 | `split_regimes.py` builds its **own** features and models from raw data and already runs on `<30` (`configs/split_regimes.yaml: target: lt30`). It does not consume the pipeline parquets |
| `outputs/reports/language_audit_plan.*` | Text scan, target-independent |
| `outputs/reports/superseded/**` | Preserved evidence — must never be regenerated |

> Note: the regime study will still need a re-run *if* `concept_drift.py` changes
> again, since it imports the live detector. It does not need one for the target
> switch itself.

## C. Reproducibility controls verified before the re-run

| Component | Seed | Status |
|---|---|---|
| `splitter.py` | `RANDOM_SEED = 42` | pinned |
| `selector.py` | `RANDOM_SEED = 42` (RandomForest, Boruta shadow RNG, stability bootstrap `SEED + i`) | pinned |
| `trainer.py` | `RANDOM_SEED = 42` (LGBM, LogReg, StratifiedKFold shuffle) | pinned |
| `preprocessor.py`, `engineer.py` | no stochastic component | n/a |

Residual risk: `n_jobs=-1` in the selector and trainer. LightGBM and some sklearn
estimators can produce thread-count-dependent floating-point results even with a
fixed seed. **Verified empirically after the re-run rather than assumed** — this
is the third determinism check in the remediation (previous two: the numpy
2.5.1 → 2.4.6 change, verified on Phase 0.1 and on the regime sweep).

## D. Expected direction of change

Stated in advance so the results cannot be rationalised afterwards:

- **Prevalence** 46.1% → 11.2%
- **AUC** expected to *fall*. Published 30-day models on this dataset sit at
  roughly 0.63–0.70; the merged-target AUC was 0.6865 (val) / 0.6560 (test)
- **F1 / precision** expected to fall sharply — a 4× rarer positive class
- **Selected features** expected to change; prior-utilisation features may weaken
- **PSI / data drift** largely unchanged (feature-side, target-independent)
- **`label_drift`** now uses the Tier 1.4 relative rule — the fixed 0.05 absolute
  rule would have been ~45% relative at this base rate and would have gone blind
- **Threshold PPR** expected to change substantially; the corrected
  population-weighted cost function matters far more at 11.2% than at 46.1%
