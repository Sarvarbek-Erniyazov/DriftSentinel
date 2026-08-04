"""
DriftSentinel — Concept Drift Detector
Monitors model performance degradation across sequential evaluation windows.
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

# ── Config (Tier 1.4/1.5) ─────────────────────────────────────────────────
import yaml

CONFIG_PATH = ROOT / "configs" / "drift_config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)["concept_drift"]

VOTING_SIGNALS     = list(_CFG["voting_signals"])
DIAGNOSTIC_SIGNALS = list(_CFG["diagnostic_only_signals"])
SEVERITY_FRACTIONS = _CFG["severity_fractions"]
LABEL_DRIFT_CFG    = _CFG["label_drift"]


def set_reports_dir(path) -> None:
    """
    Redirect this module's report output (Tier 1.7).

    Explicit, supported API. The Tier 0 regime sweep previously reassigned the
    module global directly — a monkey-patch that worked but was invisible to
    anyone reading this module. A sweep of ~200 detector runs must not scatter
    throwaway artifacts through outputs/log/, so redirection is legitimate; doing
    it by attribute assignment from another file was not.
    """
    global REPORTS_DIR
    from pathlib import Path as _P
    REPORTS_DIR = _P(path)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


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
    Simulates sequential performance monitoring. Windows are INDEX-based
    slices of the concatenated stream, not calendar intervals.
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
        threshold_source: str = "unspecified — caller did not declare the source",
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

        # Tier 1.4: prevalence-scaled rule. A fixed ABSOLUTE threshold means a
        # different thing at every base rate — 0.05 was ~11% relative under the
        # merged target and ~45% relative under <30, where it stopped detecting
        # real shifts. Replaced by significance AND a minimum relative effect.
        p_ref_, p_prod_ = float(y_ref.mean()), float(y_prod.mean())
        n_ref_, n_prod_ = len(y_ref), len(y_prod)
        pooled = (p_ref_ * n_ref_ + p_prod_ * n_prod_) / (n_ref_ + n_prod_)
        se = np.sqrt(pooled * (1 - pooled) * (1 / n_ref_ + 1 / n_prod_))
        z_stat = float((p_prod_ - p_ref_) / se) if se > 0 else 0.0
        from scipy.stats import norm as _norm
        p_value = float(2 * _norm.sf(abs(z_stat)))
        rel_change = float((p_prod_ - p_ref_) / p_ref_) if p_ref_ > 0 else 0.0

        alpha = LABEL_DRIFT_CFG["alpha"]
        min_rel = LABEL_DRIFT_CFG["min_relative_change"]
        fires = bool(p_value < alpha and abs(rel_change) >= min_rel)

        label_shift = {
            "ref_pos_rate"   : round(p_ref_,  4),
            "prod_pos_rate"  : round(p_prod_, 4),
            "delta_pos_rate" : round(p_prod_ - p_ref_, 4),
            "relative_change": round(rel_change, 4),
            "z_stat"         : round(z_stat, 4),
            "p_value"        : p_value,
            "alpha"          : alpha,
            "min_relative_change": min_rel,
            "label_drift"    : fires,
            "rule"           : (f"two-proportion test p < {alpha} AND "
                                f"|relative change| >= {min_rel}"),
            # Kept so the superseded verdict stays reportable and the change is
            # auditable rather than asserted.
            "legacy_absolute_rule": {
                "threshold": LABEL_DRIFT_CFG["legacy_absolute_threshold"],
                "would_fire": bool(abs(p_prod_ - p_ref_) >
                                   LABEL_DRIFT_CFG["legacy_absolute_threshold"]),
                "note": ("fixed absolute threshold; equivalent to "
                         f"{LABEL_DRIFT_CFG['legacy_absolute_threshold'] / p_ref_:.0%} "
                         "relative at this base rate"),
            },
        }

        logger.info(f"  Ref  pos rate    : {label_shift['ref_pos_rate']}")
        logger.info(f"  Prod pos rate    : {label_shift['prod_pos_rate']}")
        logger.info(f"  Delta pos rate   : {label_shift['delta_pos_rate']:+.4f} "
                    f"({label_shift['relative_change']:+.1%} relative)")
        logger.info(f"  Two-proportion   : z={z_stat:+.3f} p={p_value:.3e}")
        logger.info(f"  Label drift      : {fires}  "
                    f"(legacy absolute rule would say "
                    f"{label_shift['legacy_absolute_rule']['would_fire']})")

        # ── Step 7: Concept drift verdict ─────────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 7: Concept Drift Verdict")

        auc_degradation   = ref_metrics["auc"] - prod_metrics["auc"]
        f1_degradation    = ref_metrics["f1"]  - prod_metrics["f1"]
        brier_degradation = prod_metrics["brier"] - ref_metrics["brier"]

        all_signals = {
            "auc_drop"           : auc_degradation    > 0.02,
            "f1_drop"            : f1_degradation     > 0.05,
            "brier_increase"     : brier_degradation  > 0.01,
            "cusum_alarm"        : cusum_result["drift_detected"],
            "ph_alarm"           : ph_result["drift_detected"],
            "prediction_drift"   : pred_shift["proba_drift"],
            "label_drift"        : label_shift["label_drift"],
            "auc_slope_negative" : auc_slope < -0.001,
        }

        # Tier 1.5: the evidence count is over VOTING signals only. cusum_alarm
        # and ph_alarm are retained and reported, but do not vote — they are
        # structurally broken, not merely mis-tuned (see configs/drift_config.yaml
        # and outputs/reports/regime_random.json -> sequential_detector_diagnosis).
        # They are NOT silently dropped: a signal removed without evidence is
        # just as unaccountable as a signal counted without evidence.
        evidence   = {k: bool(v) for k, v in all_signals.items() if k in VOTING_SIGNALS}
        diagnostics = {k: bool(v) for k, v in all_signals.items() if k in DIAGNOSTIC_SIGNALS}

        n_voting   = len(evidence)
        n_evidence = sum(evidence.values())
        crit_n = int(np.ceil(SEVERITY_FRACTIONS["critical"] * n_voting))
        mod_n  = int(np.ceil(SEVERITY_FRACTIONS["moderate"] * n_voting))
        severity = (
            "CRITICAL" if n_evidence >= crit_n else
            "MODERATE" if n_evidence >= mod_n  else
            "MILD"     if n_evidence >= 1      else
            "NONE"
        )

        logger.info(f"  Evidence signals ({n_evidence}/{n_voting} voting):")
        for signal, fired in evidence.items():
            logger.info(f"    {signal:<30} {'✓ FIRED' if fired else '✗ not fired'}")
        logger.info(f"  Diagnostics (NOT evidence — structurally broken, Tier 1.5):")
        for signal, fired in diagnostics.items():
            logger.info(f"    {signal:<30} {'fired' if fired else 'silent'}  "
                        f"[excluded from the count]")
        logger.info(f"  Severity boundaries: CRITICAL >= {crit_n}, MODERATE >= {mod_n} "
                    f"(fractions {SEVERITY_FRACTIONS} of {n_voting} voting signals)")

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
            "n_voting_signals": int(n_voting),
            "diagnostics_not_evidence": diagnostics,
            "retired_signals": {
                "signals": DIAGNOSTIC_SIGNALS,
                "reason": ("structurally broken, not mis-tuned: CUSUM is saturated "
                           "(fires in 100% of runs in every regime including the "
                           "no-drift control, 101-207 alarms/run) and Page-Hinkley's "
                           "verdict is set by whether the first MIN_WINDOW_SIZE=200 "
                           "rows happen to sit above or below the stream mean. "
                           "Neither responded to any synthetic shift at any "
                           "magnitude."),
                "evidence": "outputs/reports/regime_random.json -> sequential_detector_diagnosis",
                "status": "RETAINED in code and reported as diagnostics; NOT counted",
            },
            "severity_boundaries": {"critical_min": crit_n, "moderate_min": mod_n,
                                    "fractions": SEVERITY_FRACTIONS},
            "threshold_used": round(float(threshold), 5),
            "threshold_source": threshold_source,
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

        # Tier 1.7: preserve any existing artifact before replacing it.
        from src.monitoring.artifact_io import write_artifact, write_dataframe
        write_artifact(report_path, _make_serializable(self.report_),
                       overwrite=True, preserve=True)
        logger.info(f"  Concept drift report: {report_path}")

        windows_path = REPORTS_DIR / f"sliding_windows_{ref_name}_{prod_name}.csv"
        write_dataframe(windows_path, windows_df, overwrite=True, preserve=True)
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

    # ── Threshold source (Tier 2A.4 / audit F13) ──────────────────────────
    #
    # The operating threshold must NOT come from the reference window. The
    # shipped code read it from evaluation_report.json, where it was chosen by
    # F1-max ON VAL — and `val` is then used here as the drift REFERENCE. The
    # reference window therefore carried a threshold fitted to itself while the
    # production window did not, and part of the reported degradation was
    # threshold optimism rather than drift.
    #
    # Tier 2A.4 measured the size of that artifact: with the threshold selected
    # on val, F1 fell 0.2627 -> 0.2200 (drop +0.0427); with it selected on a
    # held-out slice of TRAIN, F1 went 0.1908 -> 0.2122 (drop -0.0214). The
    # optimism was +0.0641 — LARGER than the entire reported drop, so the
    # "degradation" reversed sign once measured honestly.
    #
    # The threshold is now selected on a patient-disjoint slice of TRAIN, which
    # neither the reference nor the production window has seen.
    threshold = 0.5
    threshold_source = "fallback 0.5"
    try:
        from src.models.repeated_eval import recover_patient_ids
        from src.uncertainty.decontamination import (f1_max_threshold,
                                                     patient_halves)
        train = pd.read_parquet(TRAIN_DIR / "train_fs.parquet")
        y_tr = train["readmitted_binary"].to_numpy()
        g_tr, note = recover_patient_ids("train", y_tr)
        if g_tr is None:
            raise RuntimeError(note)
        hold_m, _ = patient_halves(g_tr, seed=43)
        p_tr = lgbm_predict(train[feat_cols].values)
        threshold = f1_max_threshold(y_tr[hold_m], p_tr[hold_m])
        threshold_source = ("F1-max on a patient-disjoint slice of TRAIN "
                            "(decontaminated — audit F13)")
        logger.info(f"Threshold {threshold:.4f} from {threshold_source}")
    except Exception as e:
        eval_path = MODELS_DIR / "evaluation_report.json"
        if eval_path.exists():
            with open(eval_path) as f:
                threshold = json.load(f).get("lgbm", {}).get("threshold", 0.5)
            threshold_source = (f"CONTAMINATED fallback: evaluation_report "
                                f"(fitted on the reference window) — {type(e).__name__}")
            logger.warning(f"Decontaminated threshold unavailable ({e}); "
                           f"falling back to the CONTAMINATED value {threshold:.4f}. "
                           f"The reported degradation will include threshold optimism.")

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
        threshold_source = threshold_source,
    )

    print(f"\nConcept Drift Verdict : {report['severity']}")
    print(f"Evidence signals      : {report['n_evidence']}/{report['n_voting_signals']} voting")
    print(f"AUC degradation       : {report['auc_degradation']:+.4f}")
    print(f"F1  degradation       : {report['f1_degradation']:+.4f}")
    print(f"CUSUM alarm           : {report['cusum']['drift_detected']}")
    print(f"Page-Hinkley alarm    : {report['page_hinkley']['drift_detected']}")
    print(f"Label drift           : {report['label_shift']['label_drift']}")

    return report


if __name__ == "__main__":
    run_concept_drift()