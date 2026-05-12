<div align="center">

# 🛡️ DriftSentinel

### Production ML Reliability Toolkit —  
### Automated drift detection, uncertainty quantification, and adversarial robustness for clinical prediction models

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)](https://lightgbm.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.44-red)](https://shap.readthedocs.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Conformal](https://img.shields.io/badge/Conformal_Prediction-THR%2FLAC%2FAPS-purple)](https://angelopoulos.ai/blog/simple/)
[![CUSUM](https://img.shields.io/badge/Sequential-CUSUM%20%7C%20PageHinkley-blue)](https://en.wikipedia.org/wiki/CUSUM)
[![Dataset](https://img.shields.io/badge/Dataset-UCI_Diabetes_130--US-yellow)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</div>

---

## The Problem

Deploying an ML model is only the beginning. In real clinical environments, **the world keeps changing**:

- Patient demographics shift over time → feature distributions drift
- Hospital billing codes change → `payer_code` PSI jumps from 0.0 to **0.93**
- Readmission rates drop 14 percentage points → the model was never told
- Precision collapses from **52.9% → 37.3%** — silently, without any alert

A model that worked last quarter may be actively harmful today. Most teams find out when a clinician stops trusting it. DriftSentinel finds out automatically, quantifies the damage, and triggers retraining.

---

## What We Built

| | 📊 Details |
|---|---|
| **Dataset** | UCI Diabetes 130-US Hospitals (1999–2008) |
| **Scale** | 101,766 encounters · 50 raw features · 71,518 unique patients |
| **Target** | Hospital readmission within 30 days (binary) |
| **Split strategy** | Patient-level temporal split — zero leakage |
| **Features engineered** | 38 clinical FE_ features (7 groups) |
| **Final features** | 53 (29 FE_ + 24 raw) via 7-stage selection |
| **Primary model** | LightGBM (val AUC=0.6865) |
| **Drift detected** | CRITICAL — 8/8 evidence signals fired |
| **Concept drift** | F1: 0.6735 → 0.5299 (−0.1436) under temporal shift |
| **After retraining** | lgbm_v2 test AUC: 0.6560 → 0.6648 (+0.0088) |
| **Conformal coverage** | 91.4% empirical (target 90%) — guarantee satisfied |
| **Robustness** | Score=0.8954 (ROBUST tier) |
| **Health check** | 31/32 PASS — DEGRADED (drift active, expected) |

---

## Drift Story — The Core Narrative

This is not about maximizing AUC. It is about **detecting when a deployed model stops being reliable**.

```
STAGE 1 ─ Train
  lgbm_v1 trained on 63,492 patients
  Val AUC = 0.6865  |  F1 = 0.6735  |  Precision = 0.5286

STAGE 2 ─ Deploy
  Model served on val window (20,949 patients) as reference production

STAGE 3 ─ Drift Begins
  Test window arrives (17,325 newer patients)
  payer_code PSI = 0.93   ← CRITICAL
  number_diagnoses PSI = 0.26  ← CRITICAL
  Readmission rate: 47.6% → 33.6%  (−14pp label shift)

STAGE 4 ─ DriftSentinel Fires
  8/8 concept drift evidence signals triggered:
    ✓ AUC drop (−0.0305)        ✓ F1 drop (−0.1436)
    ✓ Brier increase             ✓ CUSUM (109 alarms, first at 3.92%)
    ✓ Page-Hinkley (6 alarms)   ✓ Prediction distribution shift
    ✓ Label drift (−14pp)        ✓ AUC slope negative (−0.00123/window)
  SYSTEM STATUS: CRITICAL — IMMEDIATE ACTION REQUIRED

STAGE 5 ─ Retraining Triggered
  Registry detects drift_alert → triggers lgbm_v2
  lgbm_v2 trained on train + val (84,441 patients, more recent data)
  Test AUC: 0.6560 → 0.6648  (+0.0088)
  Verdict: PROMOTE → lgbm_v2 becomes active production model
```

---

## Key Results

### Model Performance Under Drift

| Metric | Train | Val (reference) | Test (drifted) | Val→Test Δ |
|---|---|---|---|---|
| **AUC** | 0.7891 | 0.6865 | 0.6560 | −0.0305 |
| **F1** | 0.7292 | 0.6735 | 0.5299 | −0.1436 |
| **Precision** | 0.6322 | 0.5286 | 0.3726 | −0.1560 |
| **Recall** | 0.8613 | 0.9278 | 0.9171 | −0.0107 |
| **Brier** | 0.1904 | 0.2301 | 0.2520 | +0.0219 |

### After Retraining (lgbm_v2)

| Model | Trigger | Train rows | Test AUC | Test F1 | Status |
|---|---|---|---|---|---|
| lgbm_v1 | manual | 63,492 | 0.6560 | 0.5299 | superseded |
| **lgbm_v2** | **drift_alert** | **84,441** | **0.6648** | **0.5368** | **active** |

### Data Drift (train → test)

| Feature | PSI | Level |
|---|---|---|
| payer_code | 0.84 | 🔴 CRITICAL |
| medical_specialty | 0.44 | 🔴 CRITICAL |
| number_diagnoses | 0.26 | 🔴 CRITICAL |
| FE_labs_per_day_x_comorbidity | 0.21 | 🔴 CRITICAL |
| admission_type_id | 0.19 | 🟡 MODERATE |

### Uncertainty Quantification

| Method | ECE before | ECE after | Improvement |
|---|---|---|---|
| Isotonic (best) | 0.2054 | 0.1173 | **−43%** |
| Temperature (T=1.305) | 0.2054 | 0.1958 | −5% |

| Conformal predictor | Target coverage | Empirical coverage | Satisfied |
|---|---|---|---|
| THR α=5% | 95.0% | 97.2% | ✓ |
| THR α=10% | 90.0% | 91.4% | ✓ |
| THR α=20% | 80.0% | 82.1% | ✓ |

### Adaptive Threshold (Clinical Cost: FN×5, FP×1)

| Threshold | F1 | Recall | Missed readmissions |
|---|---|---|---|
| Fixed 0.50 | 0.4683 | 45.3% | 3,158 |
| **Cost-sensitive 0.128** | **0.5126** | **99.5%** | **29** |

### Adversarial Robustness

| Attack | ASR | ΔAUC |
|---|---|---|
| FGSM | 0.29% | −0.0004 |
| PGD | 0.20% | −0.0003 |
| RANDOM (ε=0.1) | 1.93% | −0.0441 |
| MASK top-5 | **4.51%** | +0.0004 |
| BOUNDARY | 0.26% | −0.0002 |
| **Overall score** | **0.8954** | **ROBUST** |

### System Health Check

| Category | Checks | Status |
|---|---|---|
| DATA | 10/10 | ✓ PASS |
| MODELS | 8/8 | ✓ PASS |
| DRIFT | 4/5 | ⚠ WARN (drift active — expected) |
| UNCERTAINTY | 7/7 | ✓ PASS |
| ADVERSARIAL | 2/2 | ✓ PASS |
| **Total** | **31/32** | **DEGRADED** |

---

## Documentation

| Section | Description |
|---|---|
| [EDA & Data Pipeline](docs/01_eda_and_data_pipeline.md) | Raw data analysis, 10-check validation, preprocessing, 7-stage feature selection |
| [Feature Engineering](docs/02_feature_engineering.md) | 7 clinical feature groups, MI scores, SHAP importance, FE_ vs raw comparison |
| [Model Training](docs/03_model_training.md) | LightGBM + LogReg, 5-fold CV, performance degradation, model registry |
| [Drift Detection](docs/04_drift_detection.md) | Data drift (PSI/KS/Chi²), feature impact, CUSUM, Page-Hinkley, 8/8 evidence |
| [Uncertainty Quantification](docs/05_uncertainty.md) | Calibration, Conformal Prediction, adaptive threshold, clinical cost analysis |
| [Adversarial Robustness](docs/06_adversarial.md) | 6 attack methods, robustness score, 5-layer defense system |

## Architecture

```
DriftSentinel/
├── src/
│   ├── data/
│   │   ├── loader.py          # sentinel replacement, schema check
│   │   ├── validator.py       # 10-check validation suite (PASS=81, WARN=2)
│   │   ├── splitter.py        # patient-level temporal split, leakage-free
│   │   └── preprocessor.py   # ICD-9 grouping, ordinal encoding, log1p
│   │
│   ├── features/
│   │   ├── engineer.py        # 38 FE_ features across 7 clinical groups
│   │   ├── selector.py        # 7-stage: variance→corr→MI→Boruta→SHAP→stability→consensus
│   │   └── consistency.py     # PSI/KS/leakage/null consistency checks
│   │
│   ├── models/
│   │   ├── trainer.py         # LightGBM + LogReg, 5-fold CV, early stopping
│   │   ├── evaluator.py       # AUC/F1/calibration/degradation report
│   │   └── registry.py        # model versioning, drift-triggered retraining, promotion
│   │
│   ├── drift/
│   │   ├── data_drift.py      # PSI/KS/Chi2/JS/Mann-Whitney per feature
│   │   ├── feature_drift.py   # SHAP shift × drift score → impact ranking
│   │   ├── concept_drift.py   # CUSUM + Page-Hinkley + sliding window AUC
│   │   └── alerting.py        # multi-source alert engine, severity tiers
│   │
│   ├── uncertainty/
│   │   ├── calibration.py     # ECE/MCE, isotonic + temperature scaling
│   │   ├── quantifier.py      # Conformal Prediction (THR/LAC/APS, multi-alpha)
│   │   └── threshold.py       # F1-max / Youden / cost-sensitive threshold search
│   │
│   ├── adversarial/
│   │   ├── attacks.py         # FGSM / PGD / Random / Mask / Boundary
│   │   ├── robustness.py      # ASR/AUC/proba robustness score, epsilon sensitivity
│   │   └── defense.py         # 5-layer defense: validation+anomaly+consistency+smoothing+ensemble
│   │
│   ├── monitoring/
│   │   ├── logger.py          # centralized structured logger
│   │   └── health_check.py    # 32-check system health
│   │
│   └── pipelines/
│       └── pipeline.py        # 8-stage orchestration (54s end-to-end)
│
├── docs/
│   ├── 01_eda_and_data_pipeline.md
│   ├── 02_feature_engineering.md
│   ├── 03_model_training.md
│   ├── 04_drift_detection.md
│   ├── 05_uncertainty.md
│   └── 06_adversarial.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_preprocessing_audit.ipynb
│
├── outputs/
│   ├── figure/                # 31 figures (01–31)
│   ├── models/                # lgbm_v1.pkl, lgbm_v2.pkl, logreg_v1.pkl
│   ├── artifacts/             # encoders, calibrators, conformal predictor
│   ├── registry/              # model_registry.json, registry_history.csv
│   ├── alerts/                # alert_report_val_test.json
│   └── log/                   # drift reports, health_check.json
│
└── data/
    ├── raw/                   # diabetic_data.csv, IDS_mapping.csv
    ├── train/                 # train/val/test_fs.parquet (53 features)
    └── production/            # production_val/test.parquet
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Sarvarbek13/DriftSentinel.git
cd DriftSentinel
pip install -r requirements.txt

# 2. Run full training pipeline (54s)
python src/pipelines/pipeline.py

# 3. Train models
python src/models/trainer.py
python src/models/evaluator.py
python src/models/registry.py

# 4. Run drift detection
python src/drift/data_drift.py
python src/drift/feature_drift.py
python src/drift/concept_drift.py
python src/drift/alerting.py

# 5. Uncertainty quantification
python src/uncertainty/calibration.py
python src/uncertainty/quantifier.py
python src/uncertainty/threshold.py

# 6. Adversarial robustness
python src/adversarial/attacks.py
python src/adversarial/robustness.py
python src/adversarial/defense.py

# 7. System health check
python src/monitoring/health_check.py
```

---

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction*. arXiv:2107.07511.
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100–115.
- Gama, J. et al. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 1–37.
- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. NeurIPS.
- Strack, B. et al. (2014). Impact of HbA1c measurement on hospital readmission rates. *BioMed Research International*.

