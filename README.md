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

- **Patient demographics shift** over time → feature distributions drift.
- **Hospital billing codes change** → `payer_code` PSI jumps from 0.0 to **0.93**.
- **Readmission rates drop** 14 percentage points → the model was never told.
- **Precision collapses** from **52.9% → 37.3%** — silently, without any alert.

A model that worked last quarter may be actively harmful today. Most teams find out when a clinician stops trusting it. **DriftSentinel** finds out automatically, quantifies the damage, and triggers retraining.

---

## What We Built

| Feature | 📊 Details |
|---|---|
| **Dataset** | UCI Diabetes 130-US Hospitals (1999–2008) |
| **Scale** | 101,766 encounters · 50 raw features · 71,518 unique patients |
| **Target** | Hospital readmission within 30 days (binary) |
| **Split strategy** | Patient-level temporal split — zero leakage |
| **Features engineered** | 38 clinical `FE_` features (7 groups) |
| **Final features** | 53 (29 `FE_` + 24 raw) via 7-stage selection |
| **Primary model** | LightGBM (val AUC=0.6865) |
| **Drift detected** | **CRITICAL** — 8/8 evidence signals fired |
| **Concept drift** | F1: 0.6735 → 0.5299 (−0.1436) under temporal shift |
| **After retraining** | `lgbm_v2` test AUC: 0.6560 → 0.6648 (+0.0088) |
| **Conformal coverage** | 91.4% empirical (target 90%) — guarantee satisfied |
| **Robustness** | Score=0.8954 (**ROBUST** tier) |
| **Health check** | 31/32 PASS — **DEGRADED** (drift active, expected) |

---

## Drift Story — The Core Narrative

This is not about maximizing AUC. It is about **detecting when a deployed model stops being reliable**.

1.  **STAGE 1 ─ Train:** `lgbm_v1` trained on 63,492 patients (Val AUC = 0.6865).
2.  **STAGE 2 ─ Deploy:** Model served on val window as reference production.
3.  **STAGE 3 ─ Drift Begins:** Test window arrives. `payer_code` PSI = 0.93 (**CRITICAL**). Readmission rate: 47.6% → 33.6%.
4.  **STAGE 4 ─ DriftSentinel Fires:** 8/8 signals triggered (AUC drop, CUSUM alarms, Label shift). **SYSTEM STATUS: CRITICAL**.
5.  **STAGE 5 ─ Retraining:** Registry triggers `lgbm_v2`. Test AUC improves (+0.0088). `lgbm_v2` promoted to production.

---

## Key Results

### Performance Under Drift & Retraining
| Model | Status | Train Rows | Test AUC | Test F1 | Precision |
|---|---|---|---|---|---|
| `lgbm_v1` | Superseded | 63,492 | 0.6560 | 0.5299 | 0.3726 |
| **`lgbm_v2`** | **Active** | **84,441** | **0.6648** | **0.5368** | **0.3841** |

### Uncertainty & Robustness
*   **Calibration:** Isotonic scaling improved ECE by **43%** (0.2054 → 0.1173).
*   **Conformal Prediction:** 91.4% empirical coverage for a 90% target.
*   **Clinical Cost Tuning:** By adjusting threshold (0.128), missed readmissions dropped from **3,158 to 29**.
*   **Adversarial Score:** **0.8954 (ROBUST)** against FGSM, PGD, and Boundary attacks.

---

## Architecture

```text
DriftSentinel/
├── src/
│   ├── data/          # Loader, Validator (10-check suite), Splitter
│   ├── features/      # Engineering (38 features), 7-stage Selection
│   ├── models/        # Trainer, Evaluator, Model Registry
│   ├── drift/         # Data Drift (PSI/KS), Concept Drift (CUSUM/Page-Hinkley)
│   ├── uncertainty/   # Calibration, Conformal Prediction, Cost-thresholds
│   ├── adversarial/   # Attacks (FGSM/PGD) & 5-layer Defense
│   └── monitoring/    # Health Check (32-point system audit)
├── docs/              # Detailed module documentation
├── outputs/           # Figures, Models (v1/v2), Alerts, Logs
└── data/              # Raw and Processed Parquet files