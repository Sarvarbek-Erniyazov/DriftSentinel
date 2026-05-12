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

    def fit(
        self, X_train: np.ndarray, feat_cols: list[str]
    ) -> "InputValidator":
        for i, col in enumerate(feat_cols):
            if col in PROTECTED_FEATURES:
                continue
            vals = X_train[:, i]
            q1   = np.percentile(vals, 25)
            q3   = np.percentile(vals, 75)
            iqr  = q3 - q1
            self.bounds_[col] = {
                "lower": q1 - self.iqr_multiplier * iqr,
                "upper": q3 + self.iqr_multiplier * iqr,
                "idx"  : i,
            }
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
    plt.show()
    logger.info(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_defense() -> dict:
    logger.info("=" * 70)
    logger.info("DriftSentinel — Adversarial Defense System")
    logger.info("=" * 70)

    # Load data
    train = pd.read_parquet(TRAIN_DIR / "train_fs.parquet")
    test  = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")

    feat_cols = [c for c in train.columns
                 if c not in {"readmitted_binary", "readmitted_multi"}]

    # Load models
    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model_v1 = pickle.load(f)
    with open(MODELS_DIR / "lgbm_v2.pkl", "rb") as f:
        model_v2 = pickle.load(f)
    with open(ARTIFACTS_DIR / "calibrator_isotonic_lgbm_v1.pkl", "rb") as f:
        calibrator = pickle.load(f)

    eval_path = MODELS_DIR / "evaluation_report.json"
    with open(eval_path) as f:
        threshold = json.load(f)["lgbm"]["threshold"]

    X_train = train[feat_cols].values.astype(float)
    X_test  = test[feat_cols].values.astype(float)
    y_test  = test[TARGET].values

    def predict_v1(X_df: pd.DataFrame) -> np.ndarray:
        p = model_v1.predict_proba(X_df)[:, 1]
        return calibrator.transform(p)

    def predict_v2(X_df: pd.DataFrame) -> np.ndarray:
        return model_v2.predict_proba(X_df)[:, 1]

    # Sensitive features from robustness report
    sensitive_features = [
        "admission_source_id", "discharge_disposition_id",
        "FE_multi_channel_utilizer", "FE_has_prior_inpatient",
        "age", "medical_specialty",
    ]

    # ── Step 1: Fit defense system ─────────────────────────────────────────
    logger.info("\nStep 1: Fitting defense system")
    defense = AdversarialDefenseSystem(model_name="lgbm_v1")
    defense.fit(X_train, feat_cols, sensitive_features=sensitive_features)

    # ── Step 2: Defend clean test data ─────────────────────────────────────
    logger.info("\nStep 2: Defense on CLEAN test data")
    clean_result = defense.defend(
        X_test[:1000], predict_v1, predict_v2, threshold
    )

    logger.info(f"  Clean samples   : {clean_result['n_clean']}")
    logger.info(f"  Suspicious      : {clean_result['n_suspicious']}")
    logger.info(f"  Adversarial flag: {clean_result['n_adversarial']}")
    logger.info(f"  False positive rate: "
                f"{clean_result['n_adversarial']/1000:.3f}")

    # ── Step 3: Generate attacked data ────────────────────────────────────
    logger.info("\nStep 3: Defense on ATTACKED test data")

    from src.adversarial.attacks import (
        RandomNoiseAttack, FeatureMaskAttack, _get_feature_ranges
    )

    bounds   = _get_feature_ranges(X_test, feat_cols)
    attacker = RandomNoiseAttack(epsilon=0.20)
    X_attacked = attacker.attack(
        X_test[:1000], predict_v1, feat_cols, bounds
    )

    attack_result = defense.defend(
        X_attacked, predict_v1, predict_v2, threshold
    )

    logger.info(f"  Clean samples   : {attack_result['n_clean']}")
    logger.info(f"  Suspicious      : {attack_result['n_suspicious']}")
    logger.info(f"  Adversarial flag: {attack_result['n_adversarial']}")
    logger.info(f"  Detection rate  : "
                f"{(attack_result['n_suspicious'] + attack_result['n_adversarial'])/1000:.3f}")

    # ── Step 4: Defense effectiveness ─────────────────────────────────────
    logger.info("\nStep 4: Defense effectiveness summary")
    logger.info("-" * 50)

    layer_names = [
        "L1_InputValidation",
        "L2_AnomalyDetection",
        "L3_Consistency",
        "L4_Smoothing",
        "L5_EnsembleAgreement",
    ]

    logger.info(f"  {'Layer':<25} {'Clean':>8} {'Attacked':>10} {'Lift':>8}")
    logger.info("  " + "-" * 55)
    for i, name in enumerate(layer_names):
        clean_rate  = clean_result["layer_flags"][:, i].mean()
        attack_rate = attack_result["layer_flags"][:, i].mean()
        lift        = attack_rate - clean_rate
        logger.info(
            f"  {name:<25} {clean_rate:>8.3f} "
            f"{attack_rate:>10.3f} {lift:>+8.3f}"
        )

    logger.info("-" * 50)
    clean_detection  = (clean_result["n_suspicious"] +
                        clean_result["n_adversarial"]) / 1000
    attack_detection = (attack_result["n_suspicious"] +
                        attack_result["n_adversarial"]) / 1000
    logger.info(f"  Overall detection rate (clean)  : {clean_detection:.3f}")
    logger.info(f"  Overall detection rate (attacked): {attack_detection:.3f}")
    logger.info(f"  Detection lift                   : {attack_detection-clean_detection:+.3f}")

    # ── Step 5: Plot ───────────────────────────────────────────────────────
    logger.info("\nStep 5: Generating defense dashboard")
    _plot_defense_results(clean_result, attack_result, "lgbm_v1")

    # ── Save ───────────────────────────────────────────────────────────────
    report = {
        "model_name"     : "lgbm_v1",
        "threshold"      : threshold,
        "n_samples"      : 1000,
        "clean"          : {
            "n_clean"      : clean_result["n_clean"],
            "n_suspicious" : clean_result["n_suspicious"],
            "n_adversarial": clean_result["n_adversarial"],
            "false_positive_rate": round(
                clean_result["n_adversarial"] / 1000, 4
            ),
        },
        "attacked"       : {
            "n_clean"      : attack_result["n_clean"],
            "n_suspicious" : attack_result["n_suspicious"],
            "n_adversarial": attack_result["n_adversarial"],
            "detection_rate": round(attack_detection, 4),
        },
        "detection_lift" : round(attack_detection - clean_detection, 4),
        "layer_performance": {
            name: {
                "clean_rate" : round(
                    float(clean_result["layer_flags"][:, i].mean()), 4
                ),
                "attack_rate": round(
                    float(attack_result["layer_flags"][:, i].mean()), 4
                ),
            }
            for i, name in enumerate(layer_names)
        },
    }

    report_path = REPORTS_DIR / "defense_report_lgbm_v1.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Defense report saved: {report_path}")

    logger.info("=" * 70)
    logger.info("Defense System Complete")
    logger.info(f"  False positive rate : {report['clean']['false_positive_rate']:.3f}")
    logger.info(f"  Detection rate      : {report['attacked']['detection_rate']:.3f}")
    logger.info(f"  Detection lift      : {report['detection_lift']:+.3f}")
    logger.info("  Next: README.md")
    logger.info("=" * 70)

    print(f"\nDefense Results:")
    print(f"  False positive rate : {report['clean']['false_positive_rate']:.3f}")
    print(f"  Detection rate      : {report['attacked']['detection_rate']:.3f}")
    print(f"  Lift                : {report['detection_lift']:+.3f}")

    return report


if __name__ == "__main__":
    run_defense()