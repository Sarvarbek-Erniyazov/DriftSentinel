"""
DriftSentinel — Adversarial Attack Generator
Generates adversarial perturbations for tabular medical data.

Context: Hospital readmission prediction
Adversarial goal: flip model prediction from READMIT → NO READMIT
(e.g., patient or billing department manipulates features to avoid
 readmission flag and associated costs/interventions)

Attack methods:
    FGSM   — Fast Gradient Sign Method (gradient-based)
    PGD    — Projected Gradient Descent (iterative FGSM)
    RANDOM — Random noise injection
    MASK   — Zero out top-k important features
    BOUNDARY — Move sample toward decision boundary
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
import sys

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger
from src.uncertainty.calibration import IsotonicCalibrator

logger = get_logger("attacks")

ROOT          = Path(__file__).resolve().parents[2]
TRAIN_DIR     = ROOT / "data"    / "train"
MODELS_DIR    = ROOT / "outputs" / "models"
REPORTS_DIR   = ROOT / "outputs" / "log"
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"

TARGET = "readmitted_binary"

# Feature perturbation bounds — clinically plausible ranges
# Max delta per feature (absolute units)
FEATURE_BOUNDS = {
    "num_medications"    : 3,
    "number_diagnoses"   : 2,
    "time_in_hospital"   : 1,
    "number_outpatient"  : 1,
    "number_inpatient"   : 1,
    "number_emergency"   : 1,
    "num_lab_procedures" : 5,
    "num_procedures"     : 1,
    "age"                : 1,
}

# Features that should NOT be perturbed (IDs, binary flags, targets)
PROTECTED_FEATURES = {
    "readmitted_binary", "readmitted_multi",
    "weight_missing", "max_glu_serum_missing", "A1Cresult_missing",
}


# ══════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════

def _get_feature_ranges(
    X_ref: np.ndarray,
    feat_cols: list[str],
) -> dict:
    """Compute min/max bounds per feature from reference data."""
    bounds = {}
    for i, col in enumerate(feat_cols):
        bounds[col] = {
            "min": float(X_ref[:, i].min()),
            "max": float(X_ref[:, i].max()),
            "std": float(X_ref[:, i].std()),
        }
    return bounds


def _clip_to_bounds(
    X_adv:     np.ndarray,
    X_orig:    np.ndarray,
    feat_cols: list[str],
    bounds:    dict,
    epsilon:   float,
) -> np.ndarray:
    """
    Clip adversarial samples to:
    1. Feature valid range [min, max]
    2. L-inf ball around original: |x_adv - x_orig| <= epsilon * std
    """
    X_clipped = X_adv.copy()
    for i, col in enumerate(feat_cols):
        if col in PROTECTED_FEATURES:
            X_clipped[:, i] = X_orig[:, i]
            continue
        # Feature valid range
        X_clipped[:, i] = np.clip(
            X_clipped[:, i],
            bounds[col]["min"],
            bounds[col]["max"]
        )
        # L-inf ball
        max_delta = epsilon * bounds[col]["std"]
        delta     = X_clipped[:, i] - X_orig[:, i]
        delta     = np.clip(delta, -max_delta, max_delta)
        X_clipped[:, i] = X_orig[:, i] + delta

    return X_clipped


# ══════════════════════════════════════════════════════════════════════════
# Attack implementations
# ══════════════════════════════════════════════════════════════════════════

class FGSMAttack:
    """
    Fast Gradient Sign Method for tabular data.
    Approximates gradient using finite differences (model-agnostic).

    For each feature i:
        x_adv[i] = x[i] - epsilon * sign(∂L/∂x[i])

    Goal: minimize P(readmit) → flip prediction to NO READMIT
    """

    def __init__(
        self,
        epsilon:   float = 0.1,
        n_samples: int   = 500,
    ):
        self.epsilon   = epsilon
        self.n_samples = n_samples

    def attack(
        self,
        X:         np.ndarray,
        predict_fn,
        feat_cols: list[str],
        bounds:    dict,
    ) -> np.ndarray:
        """
        Generate adversarial examples using finite-difference gradient.

        Parameters
        ----------
        X          : original samples (n, d)
        predict_fn : callable(X_df) -> proba array
        """
        n, d   = X.shape
        X_adv  = X.copy().astype(float)
        h      = 1e-3  # finite difference step

        # Use subset for efficiency
        idx    = np.random.choice(n, min(self.n_samples, n), replace=False)
        X_sub  = X[idx].copy().astype(float)

        # Compute gradient approximation
        gradients = np.zeros_like(X_sub)
        X_df      = pd.DataFrame(X_sub, columns=feat_cols)
        p_orig    = predict_fn(X_df)

        for j in range(d):
            if feat_cols[j] in PROTECTED_FEATURES:
                continue
            X_plus        = X_sub.copy()
            X_plus[:, j] += h
            p_plus        = predict_fn(pd.DataFrame(X_plus, columns=feat_cols))
            gradients[:, j] = (p_plus - p_orig) / h

        # FGSM: subtract gradient to reduce readmit probability
        X_sub_adv = X_sub - self.epsilon * np.sign(gradients)
        X_sub_adv = _clip_to_bounds(X_sub_adv, X_sub, feat_cols, bounds, self.epsilon)

        X_adv[idx] = X_sub_adv
        return X_adv


class PGDAttack:
    """
    Projected Gradient Descent — iterative FGSM.
    Stronger attack: multiple small steps with projection.
    """

    def __init__(
        self,
        epsilon:    float = 0.1,
        alpha:      float = 0.02,
        n_iter:     int   = 10,
        n_samples:  int   = 300,
    ):
        self.epsilon   = epsilon
        self.alpha     = alpha
        self.n_iter    = n_iter
        self.n_samples = n_samples

    def attack(
        self,
        X:         np.ndarray,
        predict_fn,
        feat_cols: list[str],
        bounds:    dict,
    ) -> np.ndarray:
        n, d  = X.shape
        X_adv = X.copy().astype(float)
        h     = 1e-3

        idx   = np.random.choice(n, min(self.n_samples, n), replace=False)
        X_sub = X[idx].copy().astype(float)
        X_orig_sub = X_sub.copy()

        for step in range(self.n_iter):
            gradients = np.zeros_like(X_sub)
            p_curr    = predict_fn(pd.DataFrame(X_sub, columns=feat_cols))

            for j in range(d):
                if feat_cols[j] in PROTECTED_FEATURES:
                    continue
                X_plus        = X_sub.copy()
                X_plus[:, j] += h
                p_plus        = predict_fn(
                    pd.DataFrame(X_plus, columns=feat_cols)
                )
                gradients[:, j] = (p_plus - p_curr) / h

            # Step
            X_sub = X_sub - self.alpha * np.sign(gradients)
            # Project back to epsilon ball
            X_sub = _clip_to_bounds(
                X_sub, X_orig_sub, feat_cols, bounds, self.epsilon
            )

        X_adv[idx] = X_sub
        return X_adv


class RandomNoiseAttack:
    """
    Random Gaussian noise injection.
    Baseline attack — tests basic robustness to noise.
    """

    def __init__(self, epsilon: float = 0.1, seed: int = 42):
        self.epsilon = epsilon
        self.seed    = seed

    def attack(
        self,
        X:         np.ndarray,
        predict_fn,
        feat_cols: list[str],
        bounds:    dict,
    ) -> np.ndarray:
        rng   = np.random.default_rng(self.seed)
        X_adv = X.copy().astype(float)

        for i, col in enumerate(feat_cols):
            if col in PROTECTED_FEATURES:
                continue
            std   = bounds[col]["std"]
            noise = rng.normal(0, self.epsilon * std, size=len(X))
            X_adv[:, i] += noise

        X_adv = _clip_to_bounds(X_adv, X, feat_cols, bounds, self.epsilon * 2)
        return X_adv


class FeatureMaskAttack:
    """
    Zero out top-k most important features.
    Tests model dependence on key features.
    Simulates missing data attack or data poisoning.
    """

    def __init__(self, k: int = 5):
        self.k = k

    def attack(
        self,
        X:              np.ndarray,
        predict_fn,
        feat_cols:      list[str],
        bounds:         dict,
        feature_importance: np.ndarray = None,
    ) -> np.ndarray:
        X_adv = X.copy().astype(float)

        if feature_importance is not None:
            top_k_idx = np.argsort(feature_importance)[::-1][:self.k]
        else:
            top_k_idx = np.arange(self.k)

        for idx in top_k_idx:
            col = feat_cols[idx]
            if col in PROTECTED_FEATURES:
                continue
            # Replace with median (not zero — more realistic)
            median_val   = float(np.median(X[:, idx]))
            X_adv[:, idx] = median_val

        return X_adv


class BoundaryAttack:
    """
    Move samples toward decision boundary.
    Finds minimum perturbation to flip prediction.
    Simulates targeted manipulation to avoid readmission flag.
    """

    def __init__(
        self,
        n_steps:   int   = 20,
        step_size: float = 0.05,
        n_samples: int   = 200,
    ):
        self.n_steps   = n_steps
        self.step_size = step_size
        self.n_samples = n_samples

    def attack(
        self,
        X:         np.ndarray,
        predict_fn,
        feat_cols: list[str],
        bounds:    dict,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        For each positive sample (predicted readmit),
        iteratively move toward negative region.
        """
        n, d  = X.shape
        X_adv = X.copy().astype(float)
        h     = 1e-3

        # Only attack positive predictions
        p_orig = predict_fn(pd.DataFrame(X, columns=feat_cols))
        pos_idx = np.where(p_orig >= threshold)[0]
        pos_idx = pos_idx[:self.n_samples]

        for i in pos_idx:
            x_i = X[i].copy().astype(float)

            for step in range(self.n_steps):
                p_i = predict_fn(
                    pd.DataFrame(x_i.reshape(1, -1), columns=feat_cols)
                )[0]

                if p_i < threshold:
                    break  # Flipped successfully

                # Gradient toward boundary
                grad = np.zeros(d)
                for j in range(d):
                    if feat_cols[j] in PROTECTED_FEATURES:
                        continue
                    x_plus    = x_i.copy()
                    x_plus[j] += h
                    p_plus    = predict_fn(
                        pd.DataFrame(x_plus.reshape(1, -1), columns=feat_cols)
                    )[0]
                    grad[j]   = (p_plus - p_i) / h

                x_i = x_i - self.step_size * grad
                x_i = _clip_to_bounds(
                    x_i.reshape(1, -1),
                    X[i].reshape(1, -1),
                    feat_cols, bounds,
                    self.step_size * self.n_steps
                )[0]

            X_adv[i] = x_i

        return X_adv


# ══════════════════════════════════════════════════════════════════════════
# Attack evaluator
# ══════════════════════════════════════════════════════════════════════════

def evaluate_attack(
    y_true:     np.ndarray,
    p_orig:     np.ndarray,
    p_adv:      np.ndarray,
    attack_name: str,
    threshold:  float = 0.5,
) -> dict:
    """
    Evaluate attack effectiveness.

    Metrics:
        ASR   — Attack Success Rate (fraction of predictions flipped)
        ΔAUC  — AUC degradation under attack
        ΔF1   — F1 degradation under attack
        Δproba — Mean probability shift
    """
    pred_orig = (p_orig >= threshold).astype(int)
    pred_adv  = (p_adv  >= threshold).astype(int)

    # Attack success: positive → negative flip
    pos_mask  = pred_orig == 1
    flipped   = (pred_orig == 1) & (pred_adv == 0)
    asr       = flipped.sum() / (pos_mask.sum() + 1e-8)

    # Performance degradation
    auc_orig  = roc_auc_score(y_true, p_orig)
    auc_adv   = roc_auc_score(y_true, p_adv)
    f1_orig   = f1_score(y_true, pred_orig, zero_division=0)
    f1_adv    = f1_score(y_true, pred_adv,  zero_division=0)

    return {
        "attack_name"    : attack_name,
        "asr"            : round(float(asr),            4),
        "n_flipped"      : int(flipped.sum()),
        "n_positive_orig": int(pos_mask.sum()),
        "auc_orig"       : round(float(auc_orig),       4),
        "auc_adv"        : round(float(auc_adv),        4),
        "auc_delta"      : round(float(auc_adv - auc_orig), 4),
        "f1_orig"        : round(float(f1_orig),        4),
        "f1_adv"         : round(float(f1_adv),         4),
        "f1_delta"       : round(float(f1_adv - f1_orig), 4),
        "proba_delta"    : round(float((p_adv - p_orig).mean()), 4),
        "proba_std_delta": round(float(
            p_adv.std() - p_orig.std()
        ), 4),
    }


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_attacks() -> dict:
    logger.info("=" * 70)
    logger.info("DriftSentinel — Adversarial Attack Generator")
    logger.info("=" * 70)

    # Load data
    val  = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")

    feat_cols = [c for c in val.columns
                 if c not in {"readmitted_binary", "readmitted_multi"}]

    # Load model + calibrator
    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "calibrator_isotonic_lgbm_v1.pkl", "rb") as f:
        calibrator = pickle.load(f)

    # Load threshold
    eval_path = MODELS_DIR / "evaluation_report.json"
    with open(eval_path) as f:
        threshold = json.load(f)["lgbm"]["threshold"]

    # Use test split for attacks (production window)
    X_test = test[feat_cols].values.astype(float)
    y_test = test[TARGET].values

    def predict_fn(X_df: pd.DataFrame) -> np.ndarray:
        p_raw = model.predict_proba(X_df)[:, 1]
        return calibrator.transform(p_raw)

    # Reference predictions
    p_orig = predict_fn(pd.DataFrame(X_test, columns=feat_cols))

    logger.info(f"  Test samples      : {len(X_test):,}")
    logger.info(f"  Pos rate          : {y_test.mean():.4f}")
    logger.info(f"  Mean proba (orig) : {p_orig.mean():.4f}")
    logger.info(f"  Threshold         : {threshold:.4f}")

    # Feature bounds from test data
    bounds = _get_feature_ranges(X_test, feat_cols)

    # Feature importance for mask attack
    importance = model.feature_importances_

    # ── Run attacks ────────────────────────────────────────────────────────
    results = {}

    attacks = [
        ("FGSM",     FGSMAttack(epsilon=0.1, n_samples=500)),
        ("PGD",      PGDAttack(epsilon=0.1, alpha=0.02,
                               n_iter=10, n_samples=300)),
        ("RANDOM",   RandomNoiseAttack(epsilon=0.1)),
        ("MASK_k5",  FeatureMaskAttack(k=5)),
        ("MASK_k10", FeatureMaskAttack(k=10)),
        ("BOUNDARY", BoundaryAttack(n_steps=20, step_size=0.05,
                                    n_samples=200)),
    ]

    logger.info("\n" + "=" * 70)
    logger.info("Running attacks on test split")
    logger.info("=" * 70)
    logger.info(
        f"\n  {'Attack':<12} {'ASR':>7} {'Flipped':>9} "
        f"{'ΔAUC':>8} {'ΔF1':>8} {'Δproba':>8}"
    )
    logger.info("  " + "-" * 58)

    for name, attacker in attacks:
        logger.info(f"  Running {name}...")

        if name == "MASK_k5":
            X_adv = attacker.attack(
                X_test, predict_fn, feat_cols, bounds,
                feature_importance=importance
            )
        elif name == "MASK_k10":
            X_adv = attacker.attack(
                X_test, predict_fn, feat_cols, bounds,
                feature_importance=importance
            )
        elif name == "BOUNDARY":
            X_adv = attacker.attack(
                X_test, predict_fn, feat_cols, bounds,
                threshold=threshold
            )
        else:
            X_adv = attacker.attack(
                X_test, predict_fn, feat_cols, bounds
            )

        p_adv  = predict_fn(pd.DataFrame(X_adv, columns=feat_cols))
        result = evaluate_attack(y_test, p_orig, p_adv, name, threshold)
        results[name] = result

        logger.info(
            f"  {name:<12} "
            f"{result['asr']:>7.4f} "
            f"{result['n_flipped']:>9} "
            f"{result['auc_delta']:>+8.4f} "
            f"{result['f1_delta']:>+8.4f} "
            f"{result['proba_delta']:>+8.4f}"
        )

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Attack Summary")
    logger.info("=" * 70)

    strongest = max(results, key=lambda k: results[k]["asr"])
    logger.info(f"  Strongest attack  : {strongest} "
                f"(ASR={results[strongest]['asr']:.4f})")
    logger.info(f"  Mean ASR          : "
                f"{np.mean([r['asr'] for r in results.values()]):.4f}")
    logger.info(f"  Max AUC drop      : "
                f"{min(r['auc_delta'] for r in results.values()):+.4f}")

    # Save
    report_path = REPORTS_DIR / "attack_report_test.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Attack report saved: {report_path}")

    csv_path = REPORTS_DIR / "attack_results_test.csv"
    pd.DataFrame(results.values()).to_csv(csv_path, index=False)
    logger.info(f"  Attack results CSV : {csv_path}")

    logger.info("=" * 70)
    logger.info("Next: robustness.py → defense.py")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    results = run_attacks()
    print(f"\n{'='*55}")
    print("ATTACK RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Attack':<12} {'ASR':>7} {'ΔAUC':>8} {'ΔF1':>8}")
    print(f"  {'-'*38}")
    for name, r in results.items():
        print(
            f"  {name:<12} "
            f"{r['asr']:>7.4f} "
            f"{r['auc_delta']:>+8.4f} "
            f"{r['f1_delta']:>+8.4f}"
        )