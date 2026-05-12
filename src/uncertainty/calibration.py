"""
DriftSentinel — Model Calibration
Evaluates and improves probability calibration of trained models.
Well-calibrated probabilities are prerequisite for Conformal Prediction.

Methods:
    ECE  — Expected Calibration Error (primary metric)
    MCE  — Maximum Calibration Error
    Brier Score — proper scoring rule
    Reliability diagram — visual calibration assessment
    Isotonic regression — non-parametric recalibration
    Temperature scaling — parametric recalibration
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("calibration")

ROOT        = Path(__file__).resolve().parents[2]
TRAIN_DIR   = ROOT / "data"    / "train"
MODELS_DIR  = ROOT / "outputs" / "models"
FIGURE_DIR  = ROOT / "outputs" / "figure"
REPORTS_DIR = ROOT / "outputs" / "log"
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_binary"


# ══════════════════════════════════════════════════════════════════════════
# Calibration metrics
# ══════════════════════════════════════════════════════════════════════════

def expected_calibration_error(
    y_true:   np.ndarray,
    y_proba:  np.ndarray,
    n_bins:   int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    Measures weighted average gap between confidence and accuracy.
    Perfect calibration: ECE = 0.
    """
    bins     = np.linspace(0, 1, n_bins + 1)
    ece      = 0.0
    n        = len(y_true)

    for i in range(n_bins):
        mask  = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc   = y_true[mask].mean()
        conf  = y_proba[mask].mean()
        ece  += mask.sum() / n * abs(acc - conf)

    return float(ece)


def maximum_calibration_error(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
    n_bins:  int = 10,
) -> float:
    """
    Maximum Calibration Error (MCE).
    Worst-case calibration gap across all bins.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    mce  = 0.0

    for i in range(n_bins):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_proba[mask].mean()
        mce  = max(mce, abs(acc - conf))

    return float(mce)


def calibration_metrics(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
    name:    str = "model",
) -> dict:
    """Full calibration metric suite."""
    return {
        "name"  : name,
        "ece"   : round(expected_calibration_error(y_true, y_proba), 4),
        "mce"   : round(maximum_calibration_error(y_true, y_proba),  4),
        "brier" : round(brier_score_loss(y_true, y_proba),           4),
        "mean_proba"   : round(float(y_proba.mean()), 4),
        "mean_label"   : round(float(y_true.mean()),  4),
        "proba_label_gap": round(float(y_proba.mean() - y_true.mean()), 4),
    }


# ══════════════════════════════════════════════════════════════════════════
# Recalibration methods
# ══════════════════════════════════════════════════════════════════════════

class IsotonicCalibrator:
    """
    Isotonic regression recalibration.
    Non-parametric — no assumptions about probability distribution.
    Fitted on calibration set (val), applied to test.
    """

    def __init__(self):
        self.iso_reg = IsotonicRegression(out_of_bounds="clip")
        self.fitted  = False

    def fit(self, y_cal: np.ndarray, p_cal: np.ndarray) -> "IsotonicCalibrator":
        self.iso_reg.fit(p_cal, y_cal)
        self.fitted = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        return self.iso_reg.predict(p)

    def fit_transform(
        self, y_cal: np.ndarray, p_cal: np.ndarray
    ) -> np.ndarray:
        self.fit(y_cal, p_cal)
        return self.transform(p_cal)


class TemperatureScaler:
    """
    Temperature scaling recalibration.
    Single parameter T: p_calibrated = sigmoid(logit(p) / T).
    T > 1 → softer (less confident), T < 1 → sharper.
    Fitted by minimizing NLL on calibration set.
    """

    def __init__(self):
        self.temperature = 1.0
        self.fitted      = False

    def fit(
        self, y_cal: np.ndarray, p_cal: np.ndarray
    ) -> "TemperatureScaler":
        p_cal   = np.clip(p_cal, 1e-7, 1 - 1e-7)
        logits  = np.log(p_cal / (1 - p_cal))

        def nll(T):
            T    = max(T, 1e-3)
            p_t  = 1 / (1 + np.exp(-logits / T))
            p_t  = np.clip(p_t, 1e-7, 1 - 1e-7)
            return -np.mean(
                y_cal * np.log(p_t) + (1 - y_cal) * np.log(1 - p_t)
            )

        result           = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        self.fitted      = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        p      = np.clip(p, 1e-7, 1 - 1e-7)
        logits = np.log(p / (1 - p))
        return 1 / (1 + np.exp(-logits / self.temperature))

    def fit_transform(
        self, y_cal: np.ndarray, p_cal: np.ndarray
    ) -> np.ndarray:
        self.fit(y_cal, p_cal)
        return self.transform(p_cal)


# ══════════════════════════════════════════════════════════════════════════
# Main calibrator
# ══════════════════════════════════════════════════════════════════════════

class ModelCalibrator:
    """
    Full calibration pipeline.
    Fits on val split, evaluates on test split.
    Selects best calibration method by ECE.
    """

    def __init__(self, model_name: str = "lgbm_v1"):
        self.model_name    = model_name
        self.iso_calibrator = IsotonicCalibrator()
        self.temp_scaler    = TemperatureScaler()
        self.best_method    = None
        self.report_: dict  = {}

    def calibrate(
        self,
        y_val:  np.ndarray,
        p_val:  np.ndarray,
        y_test: np.ndarray,
        p_test: np.ndarray,
    ) -> dict:
        """
        Fit calibrators on val, evaluate on test.

        Parameters
        ----------
        y_val, p_val   : ground truth and probabilities on val (calibration set)
        y_test, p_test : ground truth and probabilities on test (evaluation set)
        """
        logger.info("=" * 60)
        logger.info(f"DriftSentinel — Model Calibrator [{self.model_name}]")
        logger.info("=" * 60)

        # ── Step 1: Pre-calibration metrics ───────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 1: Pre-calibration metrics")

        pre_val  = calibration_metrics(y_val,  p_val,  "pre_val")
        pre_test = calibration_metrics(y_test, p_test, "pre_test")

        logger.info(
            f"  {'Split':<10} {'ECE':>7} {'MCE':>7} {'Brier':>8} "
            f"{'MeanP':>8} {'MeanY':>8} {'Gap':>8}"
        )
        logger.info("  " + "-" * 58)
        for m in [pre_val, pre_test]:
            logger.info(
                f"  {m['name']:<10} {m['ece']:>7.4f} {m['mce']:>7.4f} "
                f"{m['brier']:>8.4f} {m['mean_proba']:>8.4f} "
                f"{m['mean_label']:>8.4f} {m['proba_label_gap']:>+8.4f}"
            )

        # ── Step 2: Isotonic calibration ───────────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 2: Isotonic regression calibration")

        self.iso_calibrator.fit(y_val, p_val)
        p_test_iso  = self.iso_calibrator.transform(p_test)
        iso_test    = calibration_metrics(y_test, p_test_iso, "isotonic_test")

        logger.info(
            f"  ECE: {pre_test['ece']:.4f} → {iso_test['ece']:.4f} "
            f"({iso_test['ece'] - pre_test['ece']:+.4f})"
        )
        logger.info(
            f"  Brier: {pre_test['brier']:.4f} → {iso_test['brier']:.4f}"
        )

        # ── Step 3: Temperature scaling ────────────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 3: Temperature scaling")

        self.temp_scaler.fit(y_val, p_val)
        p_test_temp = self.temp_scaler.transform(p_test)
        temp_test   = calibration_metrics(y_test, p_test_temp, "temperature_test")

        logger.info(f"  Temperature T = {self.temp_scaler.temperature:.4f}")
        logger.info(
            f"  {'T > 1 → softer predictions' if self.temp_scaler.temperature > 1 else 'T < 1 → sharper predictions'}"
        )
        logger.info(
            f"  ECE: {pre_test['ece']:.4f} → {temp_test['ece']:.4f} "
            f"({temp_test['ece'] - pre_test['ece']:+.4f})"
        )
        logger.info(
            f"  Brier: {pre_test['brier']:.4f} → {temp_test['brier']:.4f}"
        )

        # ── Step 4: Select best method ─────────────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 4: Best calibration method selection")

        methods = {
            "uncalibrated" : (pre_test["ece"],  p_test),
            "isotonic"     : (iso_test["ece"],  p_test_iso),
            "temperature"  : (temp_test["ece"], p_test_temp),
        }
        self.best_method = min(methods, key=lambda k: methods[k][0])
        p_test_best      = methods[self.best_method][1]
        best_ece         = methods[self.best_method][0]

        logger.info(f"  {'Method':<15} {'ECE':>8}")
        logger.info("  " + "-" * 25)
        for method, (ece, _) in sorted(methods.items(), key=lambda x: x[1][0]):
            marker = " ← BEST" if method == self.best_method else ""
            logger.info(f"  {method:<15} {ece:>8.4f}{marker}")

        # ── Step 5: Plot reliability diagrams ──────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 5: Reliability diagrams")
        self._plot_reliability(
            y_test, p_test, p_test_iso, p_test_temp, self.model_name
        )

        # ── Build report ───────────────────────────────────────────────────
        self.report_ = {
            "model_name"        : self.model_name,
            "best_method"       : self.best_method,
            "temperature"       : round(self.temp_scaler.temperature, 4),
            "pre_calibration"   : {
                "val"  : pre_val,
                "test" : pre_test,
            },
            "post_calibration"  : {
                "isotonic"   : iso_test,
                "temperature": temp_test,
            },
            "ece_improvement"   : round(
                pre_test["ece"] - best_ece, 4
            ),
        }

        logger.info("=" * 60)
        logger.info(f"Calibration complete")
        logger.info(f"  Best method  : {self.best_method}")
        logger.info(f"  ECE (test)   : {pre_test['ece']:.4f} → {best_ece:.4f}")
        logger.info(f"  ECE improvement: {self.report_['ece_improvement']:+.4f}")
        logger.info("=" * 60)

        return {
            "p_test_calibrated": p_test_best,
            "p_val_calibrated" : self.iso_calibrator.transform(p_val)
                                 if self.best_method == "isotonic"
                                 else self.temp_scaler.transform(p_val),
            "report"           : self.report_,
        }

    # ──────────────────────────────────────────────────────────────────────
    def _plot_reliability(
        self,
        y_test:     np.ndarray,
        p_raw:      np.ndarray,
        p_iso:      np.ndarray,
        p_temp:     np.ndarray,
        model_name: str,
    ):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        configs = [
            ("Uncalibrated",  p_raw,  "#e74c3c"),
            ("Isotonic",      p_iso,  "#2ecc71"),
            ("Temperature",   p_temp, "#3498db"),
        ]

        for ax, (title, proba, color) in zip(axes, configs):
            frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
            ece = expected_calibration_error(y_test, proba)
            brier = brier_score_loss(y_test, proba)

            ax.plot([0, 1], [0, 1], "k--", linewidth=1.5,
                    alpha=0.6, label="Perfect")
            ax.plot(mean_pred, frac_pos, marker="o", linewidth=2,
                    color=color, markersize=7, label=f"Model")
            ax.fill_between(mean_pred, mean_pred, frac_pos,
                            alpha=0.15, color=color)

            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title(
                f"{title}\nECE={ece:.4f}  Brier={brier:.4f}",
                fontweight="bold"
            )
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        plt.suptitle(
            f"Reliability Diagrams — {model_name}\n"
            f"(closer to diagonal = better calibrated)",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        path = FIGURE_DIR / f"25_reliability_{model_name}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.show()
        logger.info(f"  Saved: {path.name}")

    # ──────────────────────────────────────────────────────────────────────
    def save_artifacts(self):
        """Save calibrators for use in Conformal Prediction pipeline."""
        iso_path  = ARTIFACTS_DIR / f"calibrator_isotonic_{self.model_name}.pkl"
        temp_path = ARTIFACTS_DIR / f"calibrator_temperature_{self.model_name}.pkl"

        with open(iso_path, "wb") as f:
            pickle.dump(self.iso_calibrator, f)
        with open(temp_path, "wb") as f:
            pickle.dump(self.temp_scaler, f)

        report_path = REPORTS_DIR / f"calibration_report_{self.model_name}.json"
        with open(report_path, "w") as f:
            json.dump(self.report_, f, indent=2)

        logger.info(f"  Isotonic calibrator saved  : {iso_path}")
        logger.info(f"  Temperature scaler saved   : {temp_path}")
        logger.info(f"  Calibration report saved   : {report_path}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_calibration() -> dict:
    logger.info("=" * 70)
    logger.info("DriftSentinel — Calibration Run")
    logger.info("=" * 70)

    val  = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")

    feat_cols = [c for c in val.columns
                 if c not in {"readmitted_binary", "readmitted_multi"}]

    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model = pickle.load(f)

    X_val  = pd.DataFrame(val[feat_cols].values,  columns=feat_cols)
    X_test = pd.DataFrame(test[feat_cols].values, columns=feat_cols)
    y_val  = val[TARGET].values
    y_test = test[TARGET].values

    p_val  = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    calibrator = ModelCalibrator(model_name="lgbm_v1")
    result     = calibrator.calibrate(y_val, p_val, y_test, p_test)
    calibrator.save_artifacts()

    print(f"\nBest method      : {result['report']['best_method']}")
    print(f"ECE before       : {result['report']['pre_calibration']['test']['ece']}")
    print(f"ECE after        : {result['report']['post_calibration'][result['report']['best_method']]['ece']}")
    print(f"ECE improvement  : {result['report']['ece_improvement']:+.4f}")

    return result


if __name__ == "__main__":
    run_calibration()