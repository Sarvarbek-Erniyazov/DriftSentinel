"""
DriftSentinel — Concept Drift Detector
Monitors model performance degradation over time windows.
Concept drift = input distribution stable but P(Y|X) changed.

Detection methods:
    1. Sliding window performance monitoring (AUC, F1, Brier)
    2. CUSUM (Cumulative Sum) — sequential change detection
    3. Page-Hinkley test — sequential drift detection
    4. Error rate drift — prediction confidence shift
    5. Label distribution shift — P(Y) change over windows
    6. Prediction drift — P(Ŷ) change (model output distribution)
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, brier_score_loss,
    precision_score, recall_score
)
from scipy.stats import ks_2samp, mannwhitneyu
import sys

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("concept_drift")

ROOT       = Path(__file__).resolve().parents[2]
TRAIN_DIR  = ROOT / "data"    / "train"
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR= ROOT / "outputs" / "log"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = {"readmitted_binary", "readmitted_multi"}

# ── CUSUM & Page-Hinkley parameters ───────────────────────────────────────
CUSUM_THRESHOLD   = 5.0
CUSUM_DRIFT_DELTA = 0.005
PH_DELTA          = 0.005
PH_LAMBDA         = 50.0
MIN_WINDOW_SIZE   = 200


# ══════════════════════════════════════════════════════════════════════════
# Sequential drift tests
# ══════════════════════════════════════════════════════════════════════════

def cusum_test(
    errors: np.ndarray,
    threshold: float = CUSUM_THRESHOLD,
    delta: float     = CUSUM_DRIFT_DELTA,
) -> dict:
    """
    CUSUM (Cumulative Sum Control Chart).
    Detects sustained shift in prediction error stream.

    Parameters
    ----------
    errors    : per-sample prediction errors (|y_true - y_proba|)
    threshold : alarm threshold
    delta     : allowable slack (sensitivity parameter)
    """
    n           = len(errors)
    mean_ref    = errors[:MIN_WINDOW_SIZE].mean()
    cusum_pos   = np.zeros(n)
    cusum_neg   = np.zeros(n)
    alarms      = []

    for i in range(1, n):
        deviation     = errors[i] - mean_ref
        cusum_pos[i]  = max(0, cusum_pos[i-1] + deviation - delta)
        cusum_neg[i]  = max(0, cusum_neg[i-1] - deviation - delta)

        if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
            alarms.append(i)
            cusum_pos[i] = 0
            cusum_neg[i] = 0

    drift_detected = len(alarms) > 0
    first_alarm    = int(alarms[0]) if alarms else None

    return {
        "method"          : "CUSUM",
        "drift_detected"  : drift_detected,
        "n_alarms"        : len(alarms),
        "first_alarm_idx" : first_alarm,
        "first_alarm_pct" : round(first_alarm / n * 100, 2) if first_alarm else None,
        "cusum_pos_final" : round(float(cusum_pos[-1]), 4),
        "cusum_neg_final" : round(float(cusum_neg[-1]), 4),
        "mean_ref"        : round(float(mean_ref), 4),
        "mean_prod"       : round(float(errors[MIN_WINDOW_SIZE:].mean()), 4),
    }


def page_hinkley_test(
    errors: np.ndarray,
    delta:  float = PH_DELTA,
    lambda_: float = PH_LAMBDA,
) -> dict:
    """
    Page-Hinkley sequential drift detection test.
    More sensitive to gradual drift than CUSUM.
    """
    n         = len(errors)
    mean_ref  = errors[:MIN_WINDOW_SIZE].mean()
    m_t       = 0.0
    ph_vals   = np.zeros(n)
    min_val   = float("inf")
    alarms    = []

    for i in range(n):
        m_t      += errors[i] - mean_ref - delta
        min_val   = min(min_val, m_t)
        ph_t      = m_t - min_val
        ph_vals[i]= ph_t

        if ph_t > lambda_ and i >= MIN_WINDOW_SIZE:
            alarms.append(i)
            m_t     = 0.0
            min_val = float("inf")

    drift_detected = len(alarms) > 0
    first_alarm    = int(alarms[0]) if alarms else None

    return {
        "method"          : "PageHinkley",
        "drift_detected"  : drift_detected,
        "n_alarms"        : len(alarms),
        "first_alarm_idx" : first_alarm,
        "first_alarm_pct" : round(first_alarm / n * 100, 2) if first_alarm else None,
        "ph_final"        : round(float(ph_vals[-1]), 4),
        "mean_ref"        : round(float(mean_ref), 4),
        "mean_prod"       : round(float(errors[MIN_WINDOW_SIZE:].mean()), 4),
    }


# ══════════════════════════════════════════════════════════════════════════
# Sliding window evaluator
# ══════════════════════════════════════════════════════════════════════════

def sliding_window_performance(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
    n_windows: int = 10,
    threshold:  float = 0.5,
) -> pd.DataFrame:
    """
    Split data into n_windows and compute metrics per window.
    Simulates temporal performance monitoring.
    """
    n        = len(y_true)
    win_size = n // n_windows
    records  = []

    for i in range(n_windows):
        start = i * win_size
        end   = start + win_size if i < n_windows - 1 else n
        y_w   = y_true[start:end]
        p_w   = y_proba[start:end]
        pred_w = (p_w >= threshold).astype(int)

        if len(np.unique(y_w)) < 2:
            continue

        records.append({
            "window"      : i + 1,
            "start_idx"   : start,
            "end_idx"     : end,
            "n_samples"   : end - start,
            "pos_rate"    : round(float(y_w.mean()), 4),
            "mean_proba"  : round(float(p_w.mean()), 4),
            "auc"         : round(float(roc_auc_score(y_w, p_w)), 4),
            "f1"          : round(float(f1_score(y_w, pred_w, zero_division=0)), 4),
            "brier"       : round(float(brier_score_loss(y_w, p_w)), 4),
            "precision"   : round(float(precision_score(y_w, pred_w, zero_division=0)), 4),
            "recall"      : round(float(recall_score(y_w, pred_w, zero_division=0)), 4),
            "pred_rate"   : round(float(pred_w.mean()), 4),
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════
# Main concept drift detector
# ══════════════════════════════════════════════════════════════════════════

class ConceptDriftDetector:
    """
    Monitors concept drift by tracking model performance
    across reference and production windows.
    """

    def __init__(self, model_name: str = "lgbm_v1"):
        self.model_name  = model_name
        self.report_: dict = {}

    def detect(
        self,
        ref_df:     pd.DataFrame,
        prod_df:    pd.DataFrame,
        feat_cols:  list[str],
        predict_fn,
        ref_name:   str   = "val",
        prod_name:  str   = "test",
        threshold:  float = 0.5,
        n_windows:  int   = 10,
    ) -> dict:
        """
        Full concept drift detection pipeline.
        """
        logger.info("=" * 70)
        logger.info("DriftSentinel — Concept Drift Detector")
        logger.info(f"  Model      : {self.model_name}")
        logger.info(f"  Reference  : {ref_name}  ({len(ref_df):,} rows)")
        logger.info(f"  Production : {prod_name}  ({len(prod_df):,} rows)")
        logger.info("=" * 70)

        # ── Predictions ────────────────────────────────────────────────────
        X_ref   = ref_df[feat_cols].values
        X_prod  = prod_df[feat_cols].values
        y_ref   = ref_df["readmitted_binary"].values
        y_prod  = prod_df["readmitted_binary"].values

        p_ref   = predict_fn(X_ref)
        p_prod  = predict_fn(X_prod)

        err_ref  = np.abs(y_ref  - p_ref)
        err_prod = np.abs(y_prod - p_prod)

        # ── Step 1: Reference vs production metrics ────────────────────────
        logger.info("-" * 50)
        logger.info("Step 1: Reference vs Production Performance")

        def _metrics(y, p, name):
            pred = (p >= threshold).astype(int)
            return {
                "split"      : name,
                "n"          : int(len(y)),
                "pos_rate"   : round(float(y.mean()),   4),
                "mean_proba" : round(float(p.mean()),   4),
                "mean_error" : round(float(np.abs(y-p).mean()), 4),
                "auc"        : round(float(roc_auc_score(y, p)), 4),
                "f1"         : round(float(f1_score(y, pred, zero_division=0)), 4),
                "brier"      : round(float(brier_score_loss(y, p)), 4),
                "precision"  : round(float(precision_score(y, pred, zero_division=0)), 4),
                "recall"     : round(float(recall_score(y, pred, zero_division=0)), 4),
            }

        ref_metrics  = _metrics(y_ref,  p_ref,  ref_name)
        prod_metrics = _metrics(y_prod, p_prod, prod_name)

        logger.info(f"  {'Metric':<15} {'Reference':>12} {'Production':>12} {'Delta':>10}")
        logger.info(f"  {'-'*52}")
        for key in ["auc", "f1", "brier", "mean_error",
                    "pos_rate", "mean_proba", "precision", "recall"]:
            delta = prod_metrics[key] - ref_metrics[key]
            sign  = "↓" if delta < 0 else "↑"
            logger.info(
                f"  {key:<15} "
                f"{ref_metrics[key]:>12.4f} "
                f"{prod_metrics[key]:>12.4f} "
                f"{delta:>+9.4f} {sign}"
            )

        # ── Step 2: Sliding window monitoring ─────────────────────────────
        logger.info("-" * 50)
        logger.info(f"Step 2: Sliding Window Performance ({n_windows} windows)")

        y_combined = np.concatenate([y_ref,  y_prod])
        p_combined = np.concatenate([p_ref,  p_prod])

        windows_df = sliding_window_performance(
            y_combined, p_combined,
            n_windows=n_windows * 2,
            threshold=threshold
        )

        ref_win_auc  = windows_df.iloc[:n_windows]["auc"].mean()
        prod_win_auc = windows_df.iloc[n_windows:]["auc"].mean()
        auc_trend    = windows_df["auc"].values

        logger.info(f"  Reference windows mean AUC  : {ref_win_auc:.4f}")
        logger.info(f"  Production windows mean AUC : {prod_win_auc:.4f}")
        logger.info(f"  AUC trend across windows    : "
                    + " → ".join([f"{v:.3f}" for v in auc_trend]))

        auc_slope = np.polyfit(range(len(auc_trend)), auc_trend, 1)[0]
        logger.info(f"  AUC slope (linear trend)    : {auc_slope:+.6f} "
                    f"({'declining' if auc_slope < 0 else 'improving'})")

        # ── Step 3: CUSUM test ─────────────────────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 3: CUSUM Sequential Test")

        err_combined = np.concatenate([err_ref, err_prod])
        cusum_result = cusum_test(err_combined)

        logger.info(f"  Drift detected   : {cusum_result['drift_detected']}")
        logger.info(f"  N alarms         : {cusum_result['n_alarms']}")
        logger.info(f"  First alarm at   : idx={cusum_result['first_alarm_idx']} "
                    f"({cusum_result['first_alarm_pct']}% of stream)")
        logger.info(f"  Mean error ref   : {cusum_result['mean_ref']:.4f}")
        logger.info(f"  Mean error prod  : {cusum_result['mean_prod']:.4f}")

        # ── Step 4: Page-Hinkley test ──────────────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 4: Page-Hinkley Sequential Test")

        ph_result = page_hinkley_test(err_combined)

        logger.info(f"  Drift detected   : {ph_result['drift_detected']}")
        logger.info(f"  N alarms         : {ph_result['n_alarms']}")
        logger.info(f"  First alarm at   : idx={ph_result['first_alarm_idx']} "
                    f"({ph_result['first_alarm_pct']}% of stream)")
        logger.info(f"  PH final value   : {ph_result['ph_final']:.4f}")

        # ── Step 5: Prediction distribution shift ─────────────────────────
        logger.info("-" * 50)
        logger.info("Step 5: Prediction Distribution Shift")

        ks_stat, ks_pval = ks_2samp(p_ref, p_prod)
        mw_stat, mw_pval = mannwhitneyu(p_ref, p_prod, alternative="two-sided")

        pred_shift = {
            "ref_mean_proba"  : round(float(p_ref.mean()),  4),
            "prod_mean_proba" : round(float(p_prod.mean()), 4),
            "delta_mean_proba": round(float(p_prod.mean() - p_ref.mean()), 4),
            "ref_std_proba"   : round(float(p_ref.std()),   4),
            "prod_std_proba"  : round(float(p_prod.std()),  4),
            "ks_stat"         : round(float(ks_stat), 4),
            "ks_pval"         : round(float(ks_pval), 6),
            "mw_pval"         : round(float(mw_pval), 6),
            "proba_drift"     : ks_pval < 0.01,
        }

        logger.info(f"  Ref   mean proba : {pred_shift['ref_mean_proba']}")
        logger.info(f"  Prod  mean proba : {pred_shift['prod_mean_proba']}")
        logger.info(f"  Delta mean proba : {pred_shift['delta_mean_proba']:+.4f}")
        logger.info(f"  KS stat/pval     : {pred_shift['ks_stat']} / {pred_shift['ks_pval']}")
        logger.info(f"  Prediction drift : {pred_shift['proba_drift']}")

        # ── Step 6: Label distribution shift ──────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 6: Label Distribution Shift (P(Y) change)")

        label_shift = {
            "ref_pos_rate"   : round(float(y_ref.mean()),  4),
            "prod_pos_rate"  : round(float(y_prod.mean()), 4),
            "delta_pos_rate" : round(float(y_prod.mean() - y_ref.mean()), 4),
            "label_drift"    : abs(y_prod.mean() - y_ref.mean()) > 0.05,
        }

        logger.info(f"  Ref  pos rate    : {label_shift['ref_pos_rate']}")
        logger.info(f"  Prod pos rate    : {label_shift['prod_pos_rate']}")
        logger.info(f"  Delta pos rate   : {label_shift['delta_pos_rate']:+.4f}")
        logger.info(f"  Label drift      : {label_shift['label_drift']}")

        # ── Step 7: Concept drift verdict ─────────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 7: Concept Drift Verdict")

        auc_degradation   = ref_metrics["auc"] - prod_metrics["auc"]
        f1_degradation    = ref_metrics["f1"]  - prod_metrics["f1"]
        brier_degradation = prod_metrics["brier"] - ref_metrics["brier"]

        evidence = {
            "auc_drop"           : auc_degradation    > 0.02,
            "f1_drop"            : f1_degradation     > 0.05,
            "brier_increase"     : brier_degradation  > 0.01,
            "cusum_alarm"        : cusum_result["drift_detected"],
            "ph_alarm"           : ph_result["drift_detected"],
            "prediction_drift"   : pred_shift["proba_drift"],
            "label_drift"        : label_shift["label_drift"],
            "auc_slope_negative" : auc_slope < -0.001,
        }

        n_evidence = sum(evidence.values())
        severity = (
            "CRITICAL" if n_evidence >= 5 else
            "MODERATE" if n_evidence >= 3 else
            "MILD"     if n_evidence >= 1 else
            "NONE"
        )

        logger.info(f"  Evidence signals ({n_evidence}/8):")
        for signal, fired in evidence.items():
            status = "✓ FIRED" if fired else "✗ not fired"
            logger.info(f"    {signal:<30} {status}")

        logger.info(f"\n  CONCEPT DRIFT VERDICT : {severity}")
        logger.info(f"  AUC degradation       : {auc_degradation:+.4f}")
        logger.info(f"  F1  degradation       : {f1_degradation:+.4f}")
        logger.info(f"  Brier increase        : {brier_degradation:+.4f}")

        # ── Save report ────────────────────────────────────────────────────
        self.report_ = {
            "model_name"     : self.model_name,
            "ref_name"       : ref_name,
            "prod_name"      : prod_name,
            "ref_metrics"    : ref_metrics,
            "prod_metrics"   : prod_metrics,
            "auc_degradation": round(float(auc_degradation),   4),
            "f1_degradation" : round(float(f1_degradation),    4),
            "brier_change"   : round(float(brier_degradation), 4),
            "auc_slope"      : round(float(auc_slope),         6),
            "cusum"          : cusum_result,
            "page_hinkley"   : ph_result,
            "prediction_shift": pred_shift,
            "label_shift"    : label_shift,
            "evidence"       : {k: bool(v) for k, v in evidence.items()},
            "n_evidence"     : int(n_evidence),
            "severity"       : severity,
            "windows"        : windows_df.to_dict("records"),
        }

        report_path = REPORTS_DIR / f"concept_drift_{ref_name}_{prod_name}.json"
        
        def _make_serializable(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_make_serializable(i) for i in obj]
            return obj

        with open(report_path, "w") as f:
            json.dump(_make_serializable(self.report_), f, indent=2)
            logger.info(f"\n  Concept drift report: {report_path}")

        windows_path = REPORTS_DIR / f"sliding_windows_{ref_name}_{prod_name}.csv"
        windows_df.to_csv(windows_path, index=False)
        logger.info(f"  Sliding windows CSV : {windows_path}")

        logger.info("=" * 70)
        logger.info(f"Concept Drift Detection Complete — {severity}")
        logger.info("=" * 70)

        return self.report_


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_concept_drift() -> dict:
    logger.info("=" * 70)
    logger.info("DriftSentinel — Concept Drift Detection Run")
    logger.info("=" * 70)

    val  = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")

    feat_cols = [c for c in val.columns if c not in TARGET_COLS]

    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        lgbm_model = pickle.load(f)

    def lgbm_predict(X: np.ndarray) -> np.ndarray:
        df_X = pd.DataFrame(X, columns=feat_cols)
        return lgbm_model.predict_proba(df_X)[:, 1]

    eval_path = MODELS_DIR / "evaluation_report.json"
    threshold = 0.5
    if eval_path.exists():
        with open(eval_path) as f:
            eval_report = json.load(f)
        threshold = eval_report.get("lgbm", {}).get("threshold", 0.5)
        logger.info(f"Using threshold from evaluation report: {threshold:.4f}")

    detector = ConceptDriftDetector(model_name="lgbm_v1")
    report   = detector.detect(
        ref_df    = val,
        prod_df   = test,
        feat_cols = feat_cols,
        predict_fn= lgbm_predict,
        ref_name  = "val",
        prod_name = "test",
        threshold = threshold,
        n_windows = 10,
    )

    print(f"\nConcept Drift Verdict : {report['severity']}")
    print(f"Evidence signals      : {report['n_evidence']}/8")
    print(f"AUC degradation       : {report['auc_degradation']:+.4f}")
    print(f"F1  degradation       : {report['f1_degradation']:+.4f}")
    print(f"CUSUM alarm           : {report['cusum']['drift_detected']}")
    print(f"Page-Hinkley alarm    : {report['page_hinkley']['drift_detected']}")
    print(f"Label drift           : {report['label_shift']['label_drift']}")

    return report


if __name__ == "__main__":
    run_concept_drift()