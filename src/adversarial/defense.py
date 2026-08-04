"""
DriftSentinel — Adversarial Defense System
Detects and mitigates adversarial inputs before model inference.

Defense layers:
    Layer 1 — Input Validation    : range check per feature
    Layer 2 — Anomaly Detection   : Isolation Forest on input space
    Layer 3 — Prediction Consistency: multiple perturbations consistency
    Layer 4 — Feature Smoothing   : median filter on sensitive features
    Layer 5 — Ensemble Agreement  : lgbm_v1 vs lgbm_v2 disagreement flag

Defense verdict:
    CLEAN      — no defense triggered
    SUSPICIOUS — 1-2 layers triggered
    ADVERSARIAL— 3+ layers triggered → reject or flag for review
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib
matplotlib.use("Agg")   # headless: figures save to disk only, no GUI window
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger
from src.uncertainty.calibration import IsotonicCalibrator

logger = get_logger("defense")

ROOT          = Path(__file__).resolve().parents[2]
TRAIN_DIR     = ROOT / "data"    / "train"
MODELS_DIR    = ROOT / "outputs" / "models"
FIGURE_DIR    = ROOT / "outputs" / "figure"
REPORTS_DIR   = ROOT / "outputs" / "log"
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_binary"

PROTECTED_FEATURES = {
    "readmitted_binary", "readmitted_multi",
    "weight_missing", "max_glu_serum_missing", "A1Cresult_missing",
}


# ══════════════════════════════════════════════════════════════════════════
# Defense layers
# ══════════════════════════════════════════════════════════════════════════

class InputValidator:
    """
    Layer 1: Range validation.
    Flags samples with feature values outside training distribution.
    Uses [Q1 - 3*IQR, Q3 + 3*IQR] bounds (robust outlier detection).
    """

    def __init__(self, iqr_multiplier: float = 3.0):
        self.iqr_multiplier = iqr_multiplier
        self.bounds_: dict  = {}
        self.fitted_: bool  = False
        self.calibration_: dict = {}

    def calibrate(
        self,
        X_train:      np.ndarray,
        X_clean_held: np.ndarray,
        feat_cols:    list[str],
        target_rate:  float = 0.05,
        grid:         tuple = tuple(np.arange(1.5, 30.01, 0.5)),
    ) -> "InputValidator":
        """
        Choose `iqr_multiplier` so the trigger rate on HELD-OUT CLEAN data meets
        a target (default 5%).

        WHY: at the shipped multiplier of 3.0 this layer fired on **93.6% of
        clean production traffic**. A detector that flags almost everything has
        no operating value, and the reported "false positive rate 0.014" hid it
        by counting only the ADVERSARIAL verdict (audit F4).

        The bounds are fit on TRAIN and the rate is measured on a SEPARATE clean
        window, so the multiplier is not tuned on the data it is evaluated on.
        """
        best = None
        for m in grid:
            self.iqr_multiplier = float(m)
            self.fit(X_train, feat_cols)
            rate = float(self.detect(X_clean_held, feat_cols).mean())
            if best is None or abs(rate - target_rate) < abs(best[1] - target_rate):
                best = (float(m), rate)
            if rate <= target_rate:
                break
        self.iqr_multiplier = best[0]
        self.fit(X_train, feat_cols)
        achieved = float(self.detect(X_clean_held, feat_cols).mean())
        self.calibration_ = {
            "target_clean_trigger_rate": target_rate,
            "selected_iqr_multiplier": best[0],
            "achieved_clean_trigger_rate": achieved,
            "target_met": bool(achieved <= target_rate * 1.5),
            "calibrated_on": "held-out clean window (not the evaluation set)",
            "shipped_default_was": 3.0,
            "n_degenerate_zero_iqr_features": len(getattr(self, "degenerate_features_", [])),
            "degenerate_zero_iqr_features": list(getattr(self, "degenerate_features_", [])),
            "zero_iqr_repair": (
                "features with Q1 == Q3 (binary / zero-inflated) get bounds from "
                "the observed training range instead of the degenerate IQR rule; "
                "without this the multiplier has no effect on the trigger rate"),
        }
        return self

    def fit(
        self, X_train: np.ndarray, feat_cols: list[str]
    ) -> "InputValidator":
        self.degenerate_features_ = []
        for i, col in enumerate(feat_cols):
            if col in PROTECTED_FEATURES:
                continue
            vals = X_train[:, i]
            q1   = np.percentile(vals, 25)
            q3   = np.percentile(vals, 75)
            iqr  = q3 - q1
            if iqr == 0:
                # ZERO-IQR REPAIR. 23 of the 53 features here are binary or
                # heavily zero-inflated, so Q1 == Q3 and the IQR rule collapses
                # to the single point [Q1, Q3]: EVERY value away from it is
                # flagged, at every multiplier. This is why the layer fired on
                # 93.6% of clean traffic and why widening the multiplier from 3
                # to 12 changed nothing — the rate was never multiplier-driven.
                # Fall back to the observed training range, so only genuinely
                # unseen values flag.
                lower, upper = float(vals.min()), float(vals.max())
                self.degenerate_features_.append(col)
            else:
                lower = q1 - self.iqr_multiplier * iqr
                upper = q3 + self.iqr_multiplier * iqr
            self.bounds_[col] = {"lower": lower, "upper": upper, "idx": i}
        self.fitted_ = True
        return self

    def detect(
        self, X: np.ndarray, feat_cols: list[str]
    ) -> np.ndarray:
        """Returns boolean mask: True = suspicious sample."""
        flags = np.zeros(len(X), dtype=bool)
        for col, b in self.bounds_.items():
            i      = b["idx"]
            vals   = X[:, i]
            oob    = (vals < b["lower"]) | (vals > b["upper"])
            flags |= oob
        return flags

    def violation_count(
        self, X: np.ndarray, feat_cols: list[str]
    ) -> np.ndarray:
        """Number of violated bounds per sample."""
        counts = np.zeros(len(X), dtype=int)
        for col, b in self.bounds_.items():
            i       = b["idx"]
            vals    = X[:, i]
            oob     = (vals < b["lower"]) | (vals > b["upper"])
            counts += oob.astype(int)
        return counts


class AnomalyDetector:
    """
    Layer 2: Isolation Forest anomaly detection.
    Fitted on clean training data.
    Flags samples that are anomalous in the input space.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators:  int   = 100,
        random_state:  int   = 42,
    ):
        self.iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.fitted_ = False

    def fit(self, X_train: np.ndarray) -> "AnomalyDetector":
        self.iso_forest.fit(X_train)
        self.fitted_ = True
        return self

    def detect(self, X: np.ndarray) -> np.ndarray:
        """Returns boolean mask: True = anomalous."""
        preds = self.iso_forest.predict(X)
        return preds == -1

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly score (lower = more anomalous)."""
        return self.iso_forest.score_samples(X)


class PredictionConsistencyChecker:
    """
    Layer 3: Prediction consistency under small perturbations.
    If prediction flips with tiny noise → suspicious.
    Robust samples should have stable predictions.
    """

    def __init__(
        self,
        n_perturbations: int   = 10,
        noise_scale:     float = 0.01,
        flip_threshold:  float = 0.30,
    ):
        self.n_perturbations = n_perturbations
        self.noise_scale     = noise_scale
        self.flip_threshold  = flip_threshold

    def detect(
        self,
        X:          np.ndarray,
        predict_fn,
        feat_cols:  list[str],
        threshold:  float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            suspicious : bool mask
            flip_rates : fraction of perturbations that flipped prediction
        """
        p_orig     = predict_fn(pd.DataFrame(X, columns=feat_cols))
        pred_orig  = (p_orig >= threshold).astype(int)

        flip_counts = np.zeros(len(X))

        rng = np.random.default_rng(42)
        for _ in range(self.n_perturbations):
            noise    = rng.normal(0, self.noise_scale, size=X.shape)
            X_noisy  = X + noise
            p_noisy  = predict_fn(pd.DataFrame(X_noisy, columns=feat_cols))
            pred_noisy = (p_noisy >= threshold).astype(int)
            flip_counts += (pred_noisy != pred_orig).astype(int)

        flip_rates = flip_counts / self.n_perturbations
        suspicious = flip_rates >= self.flip_threshold

        return suspicious, flip_rates


class FeatureSmoother:
    """
    Layer 4: Smooth sensitive features using training distribution statistics.
    Clips extreme values to percentile bounds.
    Reduces effectiveness of adversarial perturbations.
    """

    def __init__(
        self,
        sensitive_features: list[str] = None,
        clip_percentile:    float     = 99.0,
    ):
        self.sensitive_features = sensitive_features or []
        self.clip_percentile    = clip_percentile
        self.clip_bounds_: dict = {}
        self.fitted_: bool      = False

    def fit(
        self, X_train: np.ndarray, feat_cols: list[str]
    ) -> "FeatureSmoother":
        for i, col in enumerate(feat_cols):
            if col not in self.sensitive_features:
                continue
            self.clip_bounds_[col] = {
                "lower": float(np.percentile(X_train[:, i], 100 - self.clip_percentile)),
                "upper": float(np.percentile(X_train[:, i], self.clip_percentile)),
                "idx"  : i,
            }
        self.fitted_ = True
        return self

    def smooth(
        self, X: np.ndarray, feat_cols: list[str]
    ) -> np.ndarray:
        X_smooth = X.copy().astype(float)
        for col, b in self.clip_bounds_.items():
            i = b["idx"]
            X_smooth[:, i] = np.clip(X_smooth[:, i], b["lower"], b["upper"])
        return X_smooth


class EnsembleAgreementChecker:
    """
    Layer 5: Check agreement between lgbm_v1 and lgbm_v2.
    Large disagreement = potentially adversarial input.
    """

    def __init__(self, disagreement_threshold: float = 0.20):
        self.threshold = disagreement_threshold

    def detect(
        self,
        p_v1: np.ndarray,
        p_v2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            suspicious   : bool mask (True = large disagreement)
            disagreement : absolute difference between v1 and v2
        """
        disagreement = np.abs(p_v1 - p_v2)
        suspicious   = disagreement >= self.threshold
        return suspicious, disagreement


# ══════════════════════════════════════════════════════════════════════════
# Defense pipeline
# ══════════════════════════════════════════════════════════════════════════

class AdversarialDefenseSystem:
    """
    Multi-layer adversarial defense.
    Combines all defense layers into unified detection pipeline.
    """

    def __init__(self, model_name: str = "lgbm_v1"):
        self.model_name         = model_name
        self.validator          = InputValidator(iqr_multiplier=3.0)
        self.anomaly_detector   = AnomalyDetector(contamination=0.05)
        self.consistency_checker = PredictionConsistencyChecker(
            n_perturbations=10, noise_scale=0.01, flip_threshold=0.30
        )
        self.smoother           = None
        self.ensemble_checker   = EnsembleAgreementChecker(
            disagreement_threshold=0.20
        )
        self.fitted_: bool      = False

    # ──────────────────────────────────────────────────────────────────────
    def fit(
        self,
        X_train:    np.ndarray,
        feat_cols:  list[str],
        sensitive_features: list[str] = None,
    ) -> "AdversarialDefenseSystem":
        """Fit all defense components on clean training data."""
        logger.info("Fitting defense system on training data...")

        self.feat_cols_ = feat_cols
        self.validator.fit(X_train, feat_cols)
        logger.info("  Layer 1 (InputValidator) fitted")

        self.anomaly_detector.fit(X_train)
        logger.info("  Layer 2 (AnomalyDetector) fitted")

        if sensitive_features:
            self.smoother = FeatureSmoother(
                sensitive_features=sensitive_features,
                clip_percentile=99.0
            )
            self.smoother.fit(X_train, feat_cols)
            logger.info(f"  Layer 4 (FeatureSmoother) fitted "
                        f"on {len(sensitive_features)} features")

        self.fitted_ = True
        return self

    # ──────────────────────────────────────────────────────────────────────
    def defend(
        self,
        X:          np.ndarray,
        predict_fn_v1,
        predict_fn_v2 = None,
        threshold:  float = 0.5,
        apply_smoothing: bool = True,
    ) -> dict:
        """
        Run full defense pipeline on input batch.

        Returns
        -------
        dict with per-layer flags and overall verdict per sample.
        """
        n = len(X)
        layer_flags = np.zeros((n, 5), dtype=bool)

        # Layer 1: Input validation
        l1_flags   = self.validator.detect(X, self.feat_cols_)
        l1_counts  = self.validator.violation_count(X, self.feat_cols_)
        layer_flags[:, 0] = l1_flags

        # Layer 2: Anomaly detection
        l2_flags   = self.anomaly_detector.detect(X)
        l2_scores  = self.anomaly_detector.anomaly_score(X)
        layer_flags[:, 1] = l2_flags

        # Layer 3: Prediction consistency
        l3_flags, flip_rates = self.consistency_checker.detect(
            X, predict_fn_v1, self.feat_cols_, threshold
        )
        layer_flags[:, 2] = l3_flags

        # Layer 4: Feature smoothing (apply and get predictions)
        if self.smoother and apply_smoothing:
            X_smooth   = self.smoother.smooth(X, self.feat_cols_)
            p_smooth   = predict_fn_v1(
                pd.DataFrame(X_smooth, columns=self.feat_cols_)
            )
            p_orig     = predict_fn_v1(
                pd.DataFrame(X, columns=self.feat_cols_)
            )
            smooth_delta = np.abs(p_smooth - p_orig)
            l4_flags   = smooth_delta > 0.10
        else:
            p_orig     = predict_fn_v1(
                pd.DataFrame(X, columns=self.feat_cols_)
            )
            l4_flags   = np.zeros(n, dtype=bool)
            smooth_delta = np.zeros(n)
        layer_flags[:, 3] = l4_flags

        # Layer 5: Ensemble agreement
        if predict_fn_v2 is not None:
            p_v2 = predict_fn_v2(
                pd.DataFrame(X, columns=self.feat_cols_)
            )
            l5_flags, disagreement = self.ensemble_checker.detect(
                p_orig, p_v2
            )
        else:
            l5_flags     = np.zeros(n, dtype=bool)
            disagreement = np.zeros(n)
        layer_flags[:, 4] = l5_flags

        # Verdict
        n_triggered = layer_flags.sum(axis=1)
        verdicts    = np.where(
            n_triggered >= 3, "ADVERSARIAL",
            np.where(n_triggered >= 1, "SUSPICIOUS", "CLEAN")
        )

        return {
            "layer_flags"    : layer_flags,
            "n_triggered"    : n_triggered,
            "verdicts"       : verdicts,
            "l1_violation_count": l1_counts,
            "l2_anomaly_score"  : l2_scores,
            "l3_flip_rates"     : flip_rates,
            "l4_smooth_delta"   : smooth_delta,
            "l5_disagreement"   : disagreement,
            "p_orig"            : p_orig,
            "n_clean"           : int((verdicts == "CLEAN").sum()),
            "n_suspicious"      : int((verdicts == "SUSPICIOUS").sum()),
            "n_adversarial"     : int((verdicts == "ADVERSARIAL").sum()),
        }


# ══════════════════════════════════════════════════════════════════════════
# Evaluation — confusion matrix, ROC, matched-FPR detection, layer disposition
# ══════════════════════════════════════════════════════════════════════════

LAYER_NAMES = ["L1_InputValidation", "L2_AnomalyDetection", "L3_Consistency",
               "L4_Smoothing", "L5_EnsembleAgreement"]

# Continuous per-layer statistics, used to build a score with more than the six
# operating points a 0-5 trigger COUNT can express. A ROC needs a ranking, not a
# handful of steps.
LAYER_SCORE_KEYS = {
    "L1_InputValidation":   ("l1_violation_count", +1),
    "L2_AnomalyDetection":  ("l2_anomaly_score",   -1),   # lower = more anomalous
    "L3_Consistency":       ("l3_flip_rates",      +1),
    "L4_Smoothing":         ("l4_smooth_delta",    +1),
    "L5_EnsembleAgreement": ("l5_disagreement",    +1),
}


def defense_confusion_matrix(clean_verdicts: np.ndarray,
                             attack_verdicts: np.ndarray) -> dict:
    """
    Full 3x2 confusion matrix (verdict x actual) — the PRIMARY artifact (R3).

    The shipped report published "detection rate 1.000, false positive rate
    0.014" while 941 of 1000 clean samples were flagged SUSPICIOUS. Both numbers
    were true and together they were misleading, because the FP rate counted only
    the ADVERSARIAL cell. A single cell is not a result.
    """
    classes = ["CLEAN", "SUSPICIOUS", "ADVERSARIAL"]
    n_c, n_a = len(clean_verdicts), len(attack_verdicts)
    matrix = {c: {"actual_clean": int((clean_verdicts == c).sum()),
                  "actual_attacked": int((attack_verdicts == c).sum())}
              for c in classes}
    flagged_clean = n_c - matrix["CLEAN"]["actual_clean"]
    flagged_attack = n_a - matrix["CLEAN"]["actual_attacked"]
    return {
        "matrix": matrix,
        "n_clean": n_c, "n_attacked": n_a,
        "rates": {
            "clean_flagged_any": round(flagged_clean / n_c, 4),
            "clean_flagged_adversarial_only": round(
                matrix["ADVERSARIAL"]["actual_clean"] / n_c, 4),
            "attacked_flagged_any": round(flagged_attack / n_a, 4),
            "attacked_flagged_adversarial_only": round(
                matrix["ADVERSARIAL"]["actual_attacked"] / n_a, 4),
        },
        "interpretation": (
            "'clean_flagged_any' is the operationally relevant false-alarm rate: "
            "a SUSPICIOUS verdict still costs a human review. Reporting only "
            "'clean_flagged_adversarial_only' understates the burden."),
    }


def defense_roc(score_clean: np.ndarray, score_attacked: np.ndarray,
                fpr_targets=(0.01, 0.05, 0.10, 0.20)) -> dict:
    """
    ROC over the continuous defense score, plus detection at MATCHED FPR.

    A detection rate quoted without its false-positive rate is meaningless: any
    detector reaches 100% detection by flagging everything, which is close to
    what the shipped configuration did.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.concatenate([np.zeros(len(score_clean)), np.ones(len(score_attacked))])
    s = np.concatenate([score_clean, score_attacked])
    auc = float(roc_auc_score(y, s))
    fpr, tpr, thr = roc_curve(y, s)

    matched = {}
    for t in fpr_targets:
        idx = int(np.searchsorted(fpr, t, side="right") - 1)
        idx = max(idx, 0)
        matched[f"tpr_at_fpr_{t:.2f}"] = round(float(tpr[idx]), 4)
        matched[f"threshold_at_fpr_{t:.2f}"] = round(float(thr[idx]), 6)
    return {
        "auc": round(auc, 4),
        "detection_at_matched_fpr": matched,
        "roc_curve": {"fpr": [round(float(v), 5) for v in fpr],
                      "tpr": [round(float(v), 5) for v in tpr]},
        "note": ("AUC ~0.5 means the score cannot separate attacked from clean "
                 "inputs at ANY operating point, regardless of how the verdict "
                 "thresholds are set."),
    }


def layer_disposition(clean_result: dict, attack_result: dict,
                      min_lift: float = 0.02) -> dict:
    """
    Decide, per layer, whether it carries usable signal — with the evidence.

    Rules (fixed before looking at the numbers):
      DROP_ANTI_INFORMATIVE  lift < 0        fires MORE on clean than attacked
      DROP_NEVER_FIRES       never fires anywhere
      DROP_NEGLIGIBLE        0 <= lift < min_lift
      KEEP                   lift >= min_lift
    """
    out = {}
    for i, name in enumerate(LAYER_NAMES):
        c = float(clean_result["layer_flags"][:, i].mean())
        a = float(attack_result["layer_flags"][:, i].mean())
        lift = a - c
        if c == 0.0 and a == 0.0:
            verdict, why = "DROP_NEVER_FIRES", "never fired on clean or attacked data"
        elif lift < 0:
            verdict, why = ("DROP_ANTI_INFORMATIVE",
                            f"fires MORE on clean ({c:.3f}) than attacked ({a:.3f})")
        elif lift < min_lift:
            verdict, why = ("DROP_NEGLIGIBLE", f"lift {lift:+.3f} below {min_lift}")
        else:
            verdict, why = "KEEP", f"lift {lift:+.3f}"
        out[name] = {"clean_rate": round(c, 4), "attack_rate": round(a, 4),
                     "lift": round(lift, 4), "disposition": verdict, "evidence": why}
    out["_summary"] = {
        "n_layers_implemented": len(LAYER_NAMES),
        "n_layers_carrying_signal": sum(1 for k, v in out.items()
                                        if k != "_summary" and v["disposition"] == "KEEP"),
        "voting_layers": [k for k, v in out.items()
                          if k != "_summary" and v["disposition"] == "KEEP"],
        "rule": ("the system is described by the number of layers that carry "
                 "signal, not the number implemented"),
    }
    return out


def combined_defense_score(result: dict, voting_layers: list[str],
                           ref: dict = None) -> np.ndarray:
    """
    Continuous defense score: mean of z-scored per-layer statistics over the
    VOTING layers only. Standardisation uses the clean reference batch so the
    score is comparable across batches.
    """
    cols = []
    for name in voting_layers:
        key, sign = LAYER_SCORE_KEYS[name]
        v = np.asarray(result[key], dtype=float) * sign
        base = np.asarray((ref or result)[key], dtype=float) * sign
        sd = base.std()
        cols.append((v - base.mean()) / sd if sd > 0 else np.zeros_like(v))
    return np.mean(cols, axis=0) if cols else np.zeros(len(result["n_triggered"]))


def recompute_verdicts(result: dict, voting_layers: list[str],
                       adversarial_min: int = 2) -> np.ndarray:
    """Re-derive verdicts counting ONLY the layers that carry signal."""
    idx = [LAYER_NAMES.index(n) for n in voting_layers]
    if not idx:
        return np.full(len(result["n_triggered"]), "CLEAN")
    n_trig = result["layer_flags"][:, idx].sum(axis=1)
    return np.where(n_trig >= adversarial_min, "ADVERSARIAL",
                    np.where(n_trig >= 1, "SUSPICIOUS", "CLEAN"))


# ══════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════

def _plot_defense_results(
    clean_result:  dict,
    attack_result: dict,
    model_name:    str,
):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    layer_names = [
        "L1: Input\nValidation",
        "L2: Anomaly\nDetection",
        "L3: Prediction\nConsistency",
        "L4: Feature\nSmoothing",
        "L5: Ensemble\nAgreement",
    ]

    # Row 1: Clean data
    # Left: Layer trigger rates on clean
    ax = axes[0, 0]
    clean_rates = clean_result["layer_flags"].mean(axis=0)
    colors = ["#2ecc71" if r < 0.10 else "#f39c12"
              if r < 0.20 else "#e74c3c" for r in clean_rates]
    ax.bar(range(5), clean_rates, color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(layer_names, fontsize=7)
    ax.set_ylabel("Trigger Rate")
    ax.set_title("Defense Trigger Rate — CLEAN data",
                 fontweight="bold")
    ax.set_ylim(0, 0.5)
    ax.grid(axis="y", alpha=0.3)
    for i, r in enumerate(clean_rates):
        ax.text(i, r + 0.005, f"{r:.2f}", ha="center", fontsize=8)

    # Middle: Verdict distribution clean
    ax = axes[0, 1]
    verdicts_clean = ["CLEAN", "SUSPICIOUS", "ADVERSARIAL"]
    counts_clean   = [
        clean_result["n_clean"],
        clean_result["n_suspicious"],
        clean_result["n_adversarial"],
    ]
    colors_v = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars     = ax.bar(verdicts_clean, counts_clean,
                      color=colors_v, edgecolor="black", linewidth=0.5)
    for bar, cnt in zip(bars, counts_clean):
        ax.text(bar.get_x() + bar.get_width()/2,
                cnt + 5, str(cnt), ha="center", fontsize=9)
    ax.set_title("Verdict Distribution — CLEAN data", fontweight="bold")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    # Right: Anomaly scores clean
    ax = axes[0, 2]
    ax.hist(clean_result["l2_anomaly_score"], bins=40,
            color="#3498db", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Anomaly Score (lower = more anomalous)")
    ax.set_ylabel("Count")
    ax.set_title("Anomaly Score Distribution — CLEAN", fontweight="bold")
    ax.grid(alpha=0.3)

    # Row 2: Attacked data
    ax = axes[1, 0]
    attack_rates = attack_result["layer_flags"].mean(axis=0)
    colors_a = ["#2ecc71" if r < 0.10 else "#f39c12"
                if r < 0.20 else "#e74c3c" for r in attack_rates]
    ax.bar(range(5), attack_rates, color=colors_a,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(layer_names, fontsize=7)
    ax.set_ylabel("Trigger Rate")
    ax.set_title("Defense Trigger Rate — ATTACKED data",
                 fontweight="bold")
    ax.set_ylim(0, max(0.5, max(attack_rates) + 0.05))
    ax.grid(axis="y", alpha=0.3)
    for i, r in enumerate(attack_rates):
        ax.text(i, r + 0.005, f"{r:.2f}", ha="center", fontsize=8)

    ax = axes[1, 1]
    counts_attack = [
        attack_result["n_clean"],
        attack_result["n_suspicious"],
        attack_result["n_adversarial"],
    ]
    bars = ax.bar(verdicts_clean, counts_attack,
                  color=colors_v, edgecolor="black", linewidth=0.5)
    for bar, cnt in zip(bars, counts_attack):
        ax.text(bar.get_x() + bar.get_width()/2,
                cnt + 5, str(cnt), ha="center", fontsize=9)
    ax.set_title("Verdict Distribution — ATTACKED data", fontweight="bold")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    # Right: N triggered comparison
    ax = axes[1, 2]
    ax.hist(clean_result["n_triggered"],  bins=range(7),
            alpha=0.6, color="#2ecc71", label="Clean",
            edgecolor="black", linewidth=0.5, align="left")
    ax.hist(attack_result["n_triggered"], bins=range(7),
            alpha=0.6, color="#e74c3c", label="Attacked",
            edgecolor="black", linewidth=0.5, align="left")
    ax.set_xlabel("Number of Defense Layers Triggered")
    ax.set_ylabel("Count")
    ax.set_title("Layers Triggered: Clean vs Attacked",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axvline(3, color="black", linestyle="--",
               linewidth=1.5, label="Adversarial threshold")

    plt.suptitle(
        f"Adversarial Defense System — {model_name}\n"
        f"5-layer defense: Input Validation | Anomaly | Consistency | "
        f"Smoothing | Ensemble",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = FIGURE_DIR / f"31_defense_results_{model_name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close("all")   # free the figure (long runs accumulate)
    logger.info(f"  Saved: {path.name}")



def _plot_defense_evaluation(cm_orig: dict, cm_fixed: dict, roc: dict,
                             disposition: dict, l1_cal: dict,
                             score_clean, score_attacked, model_name: str):
    """Corrected defense dashboard: confusion matrices, ROC, layer disposition."""
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    classes = ["CLEAN", "SUSPICIOUS", "ADVERSARIAL"]

    for ax, cm, title in [(axes[0, 0], cm_orig, "5-layers-vote rule (with repaired L1)"),
                          (axes[0, 1], cm_fixed, "CORRECTED (signal-carrying layers only)")]:
        M = np.array([[cm["matrix"][c]["actual_clean"] for c in classes],
                      [cm["matrix"][c]["actual_attacked"] for c in classes]], dtype=float)
        Mn = M / M.sum(axis=1, keepdims=True)
        ax.imshow(Mn, cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(classes, rotation=15)
        ax.set_yticks(range(2)); ax.set_yticklabels(["actual CLEAN", "actual ATTACKED"])
        for i in range(2):
            for j in range(3):
                ax.text(j, i, f"{int(M[i, j])}\n{Mn[i, j]:.1%}", ha="center", va="center",
                        fontsize=9, color="white" if Mn[i, j] > 0.55 else "black")
        ax.set_title(f"{title}\nclean flagged (any): "
                     f"{cm['rates']['clean_flagged_any']:.1%}", fontsize=9)
        ax.grid(False)

    ax = axes[0, 2]
    ax.plot(roc["roc_curve"]["fpr"], roc["roc_curve"]["tpr"], color="#2b6cb0", lw=1.8)
    ax.plot([0, 1], [0, 1], ls="--", color="#a0aec0", lw=1)
    for t in (0.05, 0.10):
        k = f"tpr_at_fpr_{t:.2f}"
        ax.plot([t], [roc["detection_at_matched_fpr"][k]], "o", color="#c05621", ms=7)
        ax.annotate(f"  TPR={roc['detection_at_matched_fpr'][k]:.2f} @ FPR={t:.0%}",
                    (t, roc["detection_at_matched_fpr"][k]), fontsize=8)
    ax.set_xlabel("false positive rate (clean flagged)")
    ax.set_ylabel("true positive rate (attacked detected)")
    ax.set_title(f"ROC over the continuous defense score\nAUC = {roc['auc']:.3f}", fontsize=9)

    ax = axes[1, 0]
    names = list(LAYER_NAMES)
    lifts = [disposition[n]["lift"] for n in names]
    cols = ["#2f855a" if disposition[n]["disposition"] == "KEEP" else "#9b2c2c"
            for n in names]
    ax.barh(range(len(names)), lifts, color=cols)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}\n[{disposition[n]['disposition']}]" for n in names],
                       fontsize=7)
    ax.set_xlabel("lift (attacked rate - clean rate)")
    ax.set_title("Layer disposition - green = carries signal", fontsize=9)

    ax = axes[1, 1]
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [disposition[n]["clean_rate"] for n in names], w,
           label="clean", color="#718096")
    ax.bar(x + w / 2, [disposition[n]["attack_rate"] for n in names], w,
           label="attacked", color="#c05621")
    ax.set_xticks(x); ax.set_xticklabels([n.split("_")[0] for n in names])
    ax.set_ylabel("trigger rate"); ax.legend(fontsize=8)
    ax.set_title(f"L1 recalibrated: IQR x{l1_cal['shipped_default_was']:.1f} -> "
                 f"x{l1_cal['selected_iqr_multiplier']:.1f}\n"
                 f"clean trigger {l1_cal['achieved_clean_trigger_rate']:.1%} "
                 f"(target {l1_cal['target_clean_trigger_rate']:.0%})", fontsize=9)

    ax = axes[1, 2]
    ax.hist(score_clean, bins=40, alpha=0.6, color="#2f855a", label="clean")
    ax.hist(score_attacked, bins=40, alpha=0.6, color="#c05621", label="attacked")
    ax.set_xlabel("combined defense score"); ax.set_ylabel("count")
    ax.set_title("Score separation\n(overlap = no usable operating point)", fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle(f"Adversarial Defense - corrected evaluation ({model_name})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = FIGURE_DIR / f"43_defense_evaluation_{model_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path.name}")
    return path


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_defense(target_clean_trigger_rate: float = 0.05) -> dict:
    """
    Corrected defense evaluation (Tier 1.2).

    Changes from the shipped version:
      * the full 3x2 confusion matrix is the primary artifact, not one cell
      * detection reported AT MATCHED FPR, with a ROC over a continuous score
      * L1 recalibrated on a held-out clean window to a target trigger rate
      * layers kept or dropped on evidence; the system is described by the
        number of layers that carry signal, not the number implemented
    """
    logger.info("=" * 70)
    logger.info("DriftSentinel - Adversarial Defense (corrected evaluation)")
    logger.info("=" * 70)

    train = pd.read_parquet(TRAIN_DIR / "train_fs.parquet")
    val   = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test  = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")
    feat_cols = [c for c in train.columns
                 if c not in {"readmitted_binary", "readmitted_multi"}]

    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model_v1 = pickle.load(f)
    with open(MODELS_DIR / "lgbm_v2.pkl", "rb") as f:
        model_v2 = pickle.load(f)
    with open(ARTIFACTS_DIR / "calibrator_isotonic_lgbm_v1.pkl", "rb") as f:
        calibrator = pickle.load(f)
    with open(MODELS_DIR / "evaluation_report.json") as f:
        threshold = json.load(f)["lgbm"]["threshold"]

    X_train = train[feat_cols].values.astype(float)
    X_val   = val[feat_cols].values.astype(float)
    X_test  = test[feat_cols].values.astype(float)

    def predict_v1(X_df):
        return calibrator.transform(model_v1.predict_proba(X_df)[:, 1])

    def predict_v2(X_df):
        return model_v2.predict_proba(X_df)[:, 1]

    sensitive_features = ["admission_source_id", "discharge_disposition_id",
                          "FE_multi_channel_utilizer", "FE_has_prior_inpatient",
                          "age", "medical_specialty"]

    # ── Step 1: fit, with L1 calibrated on held-out clean data ────────────
    logger.info("Step 1: Fit defense; calibrate L1 on the val window")
    defense = AdversarialDefenseSystem(model_name="lgbm_v1")
    defense.feat_cols_ = feat_cols
    defense.validator.calibrate(X_train, X_val, feat_cols,
                                target_rate=target_clean_trigger_rate)
    l1_cal = defense.validator.calibration_
    logger.info(f"  L1 IQR multiplier {l1_cal['shipped_default_was']} -> "
                f"{l1_cal['selected_iqr_multiplier']}  "
                f"(clean trigger {l1_cal['achieved_clean_trigger_rate']:.3f}, "
                f"target {target_clean_trigger_rate})")
    defense.anomaly_detector.fit(X_train)
    defense.smoother = FeatureSmoother(sensitive_features=sensitive_features,
                                       clip_percentile=99.0)
    defense.smoother.fit(X_train, feat_cols)
    defense.fitted_ = True

    # ── Step 2: clean and attacked batches ────────────────────────────────
    n_eval = 1000
    logger.info(f"Step 2: Defend {n_eval} clean and {n_eval} attacked samples")
    clean_result = defense.defend(X_test[:n_eval], predict_v1, predict_v2, threshold)

    from src.adversarial.attacks import RandomNoiseAttack, _get_feature_ranges
    bounds = _get_feature_ranges(X_test, feat_cols)
    X_attacked = RandomNoiseAttack(epsilon=0.20).attack(
        X_test[:n_eval], predict_v1, feat_cols, bounds)
    attack_result = defense.defend(X_attacked, predict_v1, predict_v2, threshold)

    # ── Step 3: layer disposition ─────────────────────────────────────────
    logger.info("Step 3: Layer disposition (kept or dropped on evidence)")
    disposition = layer_disposition(clean_result, attack_result)
    for name in LAYER_NAMES:
        d = disposition[name]
        logger.info(f"  {name:<24} clean={d['clean_rate']:.3f} "
                    f"attacked={d['attack_rate']:.3f} lift={d['lift']:+.3f} "
                    f"-> {d['disposition']} ({d['evidence']})")
    voting = disposition["_summary"]["voting_layers"]
    logger.info(f"  Implemented: {len(LAYER_NAMES)} | carrying signal: "
                f"{disposition['_summary']['n_layers_carrying_signal']} -> {voting}")

    # ── Step 4: confusion matrices ────────────────────────────────────────
    logger.info("Step 4: Confusion matrices (3x2, primary artifact)")
    cm_orig = defense_confusion_matrix(clean_result["verdicts"],
                                       attack_result["verdicts"])
    v_clean_fixed = recompute_verdicts(clean_result, voting)
    v_attack_fixed = recompute_verdicts(attack_result, voting)
    cm_fixed = defense_confusion_matrix(v_clean_fixed, v_attack_fixed)
    for tag, cm in [("SHIPPED  ", cm_orig), ("CORRECTED", cm_fixed)]:
        r = cm["rates"]
        logger.info(f"  {tag}: clean flagged(any)={r['clean_flagged_any']:.3f} "
                    f"clean ADVERSARIAL={r['clean_flagged_adversarial_only']:.3f} | "
                    f"attacked flagged(any)={r['attacked_flagged_any']:.3f}")

    # ── Step 5: ROC and detection at matched FPR ──────────────────────────
    logger.info("Step 5: ROC over the continuous score")
    score_layers = voting or list(LAYER_NAMES)
    s_clean = combined_defense_score(clean_result, score_layers)
    s_attack = combined_defense_score(attack_result, score_layers, ref=clean_result)
    roc = defense_roc(s_clean, s_attack)
    logger.info(f"  AUC = {roc['auc']:.4f}")
    for k, v in roc["detection_at_matched_fpr"].items():
        if k.startswith("tpr"):
            logger.info(f"    {k} = {v:.4f}")

    # Per-layer ROC. Reported for ALL five layers so the headline AUC cannot be
    # an artifact of which layers were chosen to vote. L1's binary FLAG has zero
    # lift after the zero-IQR repair, but its continuous violation COUNT may
    # still separate — those are different statistics and both are shown.
    per_layer = {}
    for name in LAYER_NAMES:
        key, sign = LAYER_SCORE_KEYS[name]
        c = np.asarray(clean_result[key], dtype=float) * sign
        a = np.asarray(attack_result[key], dtype=float) * sign
        if np.std(np.concatenate([c, a])) == 0:
            per_layer[name] = {"auc": None, "note": "constant statistic — no ranking exists"}
        else:
            per_layer[name] = {
                "auc": defense_roc(c, a)["auc"],
                "statistic": key,
            }
    all_layers_score_clean = combined_defense_score(clean_result, list(LAYER_NAMES))
    all_layers_score_attack = combined_defense_score(attack_result, list(LAYER_NAMES),
                                                     ref=clean_result)
    roc_all = defense_roc(all_layers_score_clean, all_layers_score_attack)
    logger.info("  Per-layer ROC AUC (continuous statistic):")
    for name, v in per_layer.items():
        logger.info(f"    {name:<24} {v.get('auc')}")
    logger.info(f"    {'ALL 5 layers combined':<24} {roc_all['auc']}")

    fig_path = _plot_defense_evaluation(cm_orig, cm_fixed, roc, disposition,
                                        l1_cal, s_clean, s_attack, "lgbm_v1")

    report = {
        "model_name": "lgbm_v1",
        "tier": "1.2 - corrected defense evaluation",
        "threshold": threshold,
        "n_clean": n_eval, "n_attacked": n_eval,
        "attack": "RandomNoiseAttack(epsilon=0.20)",
        "confusion_matrix_five_voting_layers": cm_orig,
        "confusion_matrix_five_voting_layers_note": (
            "This is the ORIGINAL 5-layers-vote verdict rule, but computed with "
            "the REPAIRED L1 (zero-IQR fix + recalibration). It is not the "
            "as-shipped result. The true as-shipped numbers — 45 CLEAN / 941 "
            "SUSPICIOUS / 14 ADVERSARIAL, i.e. 94.5% of clean traffic flagged — "
            "are preserved in the superseded artifact and are not reproducible "
            "here because the defect that produced them has been fixed."),
        "confusion_matrix_corrected": cm_fixed,
        "roc": roc,
        "roc_per_layer": per_layer,
        "roc_all_five_layers": roc_all,
        "layer_disposition": disposition,
        "l1_calibration": l1_cal,
        "supersedes": "outputs/reports/superseded/defense_report_lgbm_v1.json",
        # Tier 2C.3 correction. These strings previously carried HAND-TYPED
        # numbers, written when the module was first run and never updated. The
        # Tier 2A.1 target switch changed the model, the threshold and therefore
        # every value below, and the prose silently kept the old ones: F1 said
        # "23 of 53" where the calibration reported 20, F3 said L4 AUC 0.602
        # where it computed 0.582, and F4 said "all five reach 0.617 ... 0.064 at
        # 5% FPR", which was both stale (0.651) and a conflation of the all-five
        # ROC with the KEPT-layers ROC (0.064 is the kept-layers figure; all five
        # give 0.129). Nothing was wrong with the computation — the defect was
        # hand-typed results inside a generated artifact, which is exactly what
        # R4 forbids. Every number here is now interpolated from the values
        # computed above, so a regeneration cannot leave the prose behind.
        "findings": {
            "F1_zero_iqr_made_calibration_impossible": (
                f"{l1_cal['n_degenerate_zero_iqr_features']} of {len(feat_cols)} "
                "features are binary or zero-inflated, so Q1 == Q3 and the IQR "
                "rule collapsed to a single point: every value away from it was "
                "flagged at EVERY multiplier. Raising the multiplier left the "
                "clean trigger rate unchanged. The clean-trigger rate reported "
                "before repair was never a tuning problem — the layer was "
                "mis-specified for the feature types it was applied to."),
            "F2_the_original_detection_was_a_type_violation_artifact": (
                "Before the repair, the L1 violation COUNT separated clean from "
                "attacked at AUC 0.94 — but only because the attack adds "
                "continuous noise to binary columns, so 'detection' amounted to "
                "noticing that a binary feature held a non-integer value. That is "
                "a data-type validity check, not adversarial detection, and it "
                "says nothing about robustness to an adversary who respects the "
                f"schema. After repair, L1 separates at AUC "
                f"{per_layer['L1_InputValidation'].get('auc')}."),
            "F3_L4_is_not_dead_its_threshold_is_unreachable": (
                f"L4 Smoothing's FLAG fires at "
                f"{disposition['L4_Smoothing']['clean_rate']:.3f} on clean and "
                f"{disposition['L4_Smoothing']['attack_rate']:.3f} on attacked "
                "input, which the audit read as a dead layer. Its continuous "
                f"statistic separates at AUC "
                f"{per_layer['L4_Smoothing'].get('auc')}, so the flag is dead "
                "because its threshold sits beyond the range the statistic takes "
                "— a repairable threshold, not a dead layer, and a correction to "
                "audit F16."),
            "F4_ceiling": (
                "Combining all five continuous statistics reaches AUC "
                f"{roc_all['auc']}, with detection "
                f"{roc_all['detection_at_matched_fpr']['tpr_at_fpr_0.05']:.3f} at "
                "a 5% false-positive rate. The kept-layer score reaches AUC "
                f"{roc['auc']} and detection "
                f"{roc['detection_at_matched_fpr']['tpr_at_fpr_0.05']:.3f} at the "
                "same 5% FPR. These are DIFFERENT scores and must not be quoted "
                "against each other. On either, there is no operating point at "
                "which this system usefully detects this attack. The shipped "
                "'detection rate 1.000, false positive rate 0.014' was an "
                "artifact of flagging 94.5% of clean traffic."),
        },
        "corrections": [
            "Primary artifact is the full 3x2 confusion matrix. The shipped "
            "report published 'false positive rate 0.014' by counting only the "
            "ADVERSARIAL cell while 941/1000 clean samples were SUSPICIOUS.",
            "Detection is reported at matched FPR with a ROC over a continuous "
            "score; a bare detection rate is unfalsifiable.",
            "L1 recalibrated on a held-out clean window against a target trigger "
            "rate, replacing the shipped IQR multiplier of 3.0.",
            "Layers kept or dropped on evidence; the system is described by the "
            "number of layers that carry signal.",
        ],
        "figure": str(fig_path.relative_to(ROOT).as_posix()),
    }

    # Tier 1.7: route through the write guard so regenerating this report
    # preserves the previous version instead of destroying it.
    from src.monitoring.artifact_io import write_artifact
    out = REPORTS_DIR / "defense_report_lgbm_v1.json"
    write_artifact(out, report, overwrite=True, preserve=True)
    logger.info(f"  Report: {out}")
    logger.info("=" * 70)
    return report


if __name__ == "__main__":
    r = run_defense()
    cm = r["confusion_matrix_corrected"]
    print("\nCorrected defense evaluation")
    print(f"  clean flagged (any)   : {cm['rates']['clean_flagged_any']:.3f}")
    print(f"  attacked flagged (any): {cm['rates']['attacked_flagged_any']:.3f}")
    print(f"  ROC AUC               : {r['roc']['auc']:.4f}")
    print(f"  layers carrying signal: "
          f"{r['layer_disposition']['_summary']['n_layers_carrying_signal']}/5")
