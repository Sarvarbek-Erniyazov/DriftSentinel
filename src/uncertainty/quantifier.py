"""
DriftSentinel — Conformal Prediction Quantifier
Provides distribution-free uncertainty quantification with coverage guarantee.

Theory:
    Conformal Prediction (Vovk et al., 2005) guarantees that for any
    significance level alpha, the prediction set contains the true label
    with probability >= 1 - alpha, regardless of data distribution.

    Coverage guarantee: P(Y ∈ C(X)) >= 1 - alpha

Methods implemented:
    RAPS  — Regularized Adaptive Prediction Sets (Angelopoulos et al., 2021)
    LAC   — Least Ambiguous Set-valued Classifiers
    APS   — Adaptive Prediction Sets
    THR   — Threshold Conformal Prediction (binary)

For DriftSentinel:
    - Calibrate on val split (calibration set)
    - Generate prediction sets on test split (production)
    - Measure coverage degradation under drift
    - Coverage drop = additional evidence of concept drift
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib
matplotlib.use("Agg")   # headless: figures save to disk only, no GUI window
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger
from src.uncertainty.calibration import IsotonicCalibrator

logger = get_logger("quantifier")

ROOT          = Path(__file__).resolve().parents[2]
TRAIN_DIR     = ROOT / "data"    / "train"
MODELS_DIR    = ROOT / "outputs" / "models"
FIGURE_DIR    = ROOT / "outputs" / "figure"
REPORTS_DIR   = ROOT / "outputs" / "log"
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_binary"


# ══════════════════════════════════════════════════════════════════════════
# Nonconformity scores
# ══════════════════════════════════════════════════════════════════════════

def _nonconformity_score_thr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> np.ndarray:
    """
    THR: score = 1 - p(y_true).
    For binary: score = 1 - p if y=1, else score = p.
    Higher score = more nonconforming (less expected).
    """
    scores = np.where(y_true == 1, 1 - y_proba, y_proba)
    return scores


def _nonconformity_score_lac(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
) -> np.ndarray:
    """
    LAC: score = 1 - p(y_true).
    Same as THR for binary — returns probability of true class.
    """
    p_true = np.where(y_true == 1, y_proba, 1 - y_proba)
    return 1 - p_true


def _nonconformity_score_aps(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
) -> np.ndarray:
    """
    APS: cumulative probability up to and including true class.
    For binary: score = sum of probabilities in decreasing order
    until true class is included.
    """
    scores = np.zeros(len(y_true))
    for i in range(len(y_true)):
        p1 = y_proba[i]
        p0 = 1 - y_proba[i]
        if y_true[i] == 1:
            if p1 >= p0:
                scores[i] = p1
            else:
                scores[i] = p0 + p1
        else:
            if p0 >= p1:
                scores[i] = p0
            else:
                scores[i] = p1 + p0
    return scores


# ══════════════════════════════════════════════════════════════════════════
# Core conformal predictor
# ══════════════════════════════════════════════════════════════════════════

class ConformalPredictor:
    """
    Split Conformal Prediction for binary classification.

    Protocol:
        1. fit(y_cal, p_cal)  — compute nonconformity scores on calibration set
                                find quantile q_hat at level ceil((n+1)(1-alpha))/n
        2. predict(p_test)    — return prediction sets for each test point
        3. evaluate(y_test)   — measure empirical coverage and efficiency
    """

    def __init__(
        self,
        alpha:  float = 0.10,
        method: str   = "thr",
    ):
        """
        Parameters
        ----------
        alpha  : significance level (1-alpha = target coverage)
                 alpha=0.10 → 90% coverage guarantee
        method : 'thr' / 'lac' / 'aps'
        """
        self.alpha    = alpha
        self.method   = method
        self.q_hat_   = None
        self.n_cal_   = None
        self.scores_  = None
        self.fitted_  = False

    # ──────────────────────────────────────────────────────────────────────
    def fit(
        self,
        y_cal:   np.ndarray,
        p_cal:   np.ndarray,
    ) -> "ConformalPredictor":
        """
        Compute calibration quantile q_hat.

        Parameters
        ----------
        y_cal : true labels on calibration set
        p_cal : model probabilities on calibration set
        """
        self.n_cal_ = len(y_cal)

        # Nonconformity scores
        if self.method == "thr":
            self.scores_ = _nonconformity_score_thr(y_cal, p_cal)
        elif self.method == "lac":
            self.scores_ = _nonconformity_score_lac(y_cal, p_cal)
        elif self.method == "aps":
            self.scores_ = _nonconformity_score_aps(y_cal, p_cal)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Conformal quantile — finite sample correction
        level   = np.ceil((self.n_cal_ + 1) * (1 - self.alpha)) / self.n_cal_
        level   = min(level, 1.0)
        self.q_hat_ = float(np.quantile(self.scores_, level))
        self.fitted_ = True

        return self

    # ──────────────────────────────────────────────────────────────────────
    def predict(self, p_test: np.ndarray) -> np.ndarray:
        """
        Generate prediction sets for test points.

        Returns
        -------
        sets : array of shape (n_test, 2) — boolean [include_0, include_1]
               True = class is in prediction set
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() first")

        n     = len(p_test)
        sets  = np.zeros((n, 2), dtype=bool)

        for i in range(n):
            p1 = p_test[i]
            p0 = 1 - p_test[i]

            if self.method == "thr":
                # Include class c if nonconformity score <= q_hat
                sets[i, 1] = (1 - p1) <= self.q_hat_
                sets[i, 0] = p1       <= self.q_hat_

            elif self.method in ("lac", "aps"):
                sets[i, 1] = (1 - p1) <= self.q_hat_
                sets[i, 0] = (1 - p0) <= self.q_hat_

        return sets

    # ──────────────────────────────────────────────────────────────────────
    def predict_with_scores(
        self, p_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns prediction sets and nonconformity scores for test points.
        """
        sets        = self.predict(p_test)
        test_scores = np.where(
            np.arange(len(p_test))[:, None] >= 0,
            np.column_stack([p_test, 1 - p_test]),
            0
        )
        # Score = nonconformity of predicted positive class
        scores = np.where(p_test >= 0.5, 1 - p_test, p_test)
        return sets, scores

    # ──────────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        y_test: np.ndarray,
        p_test: np.ndarray,
    ) -> dict:
        """
        Evaluate empirical coverage and efficiency on test set.

        Coverage guarantee: empirical coverage should be >= 1 - alpha.
        Under drift: coverage may drop below guarantee → drift signal.
        """
        sets = self.predict(p_test)

        # Coverage: fraction of test points where true label is in set
        covered    = np.array([
            sets[i, int(y_test[i])] for i in range(len(y_test))
        ])
        coverage   = float(covered.mean())

        # Efficiency metrics
        set_sizes  = sets.sum(axis=1)
        empty_sets = (set_sizes == 0).sum()
        singleton  = (set_sizes == 1).sum()
        both_class = (set_sizes == 2).sum()

        # Coverage gap from guarantee
        target_coverage = 1 - self.alpha
        coverage_gap    = coverage - target_coverage

        # Conditional coverage per class
        mask_pos   = y_test == 1
        mask_neg   = y_test == 0
        cov_pos    = float(covered[mask_pos].mean()) if mask_pos.sum() > 0 else np.nan
        cov_neg    = float(covered[mask_neg].mean()) if mask_neg.sum() > 0 else np.nan

        return {
            "alpha"           : self.alpha,
            "target_coverage" : round(target_coverage, 4),
            "empirical_coverage": round(coverage, 4),
            "coverage_gap"    : round(coverage_gap, 4),
            "coverage_satisfied": coverage >= target_coverage,
            "cov_positive_class": round(cov_pos, 4),
            "cov_negative_class": round(cov_neg, 4),
            "n_test"          : int(len(y_test)),
            "n_covered"       : int(covered.sum()),
            "set_size_mean"   : round(float(set_sizes.mean()), 4),
            "n_empty_sets"    : int(empty_sets),
            "n_singleton"     : int(singleton),
            "n_both_classes"  : int(both_class),
            "q_hat"           : round(float(self.q_hat_), 4),
            "method"          : self.method,
        }


# ══════════════════════════════════════════════════════════════════════════
# Multi-alpha conformal predictor
# ══════════════════════════════════════════════════════════════════════════

class MultiAlphaConformalPredictor:
    """
    Runs conformal prediction at multiple alpha levels.
    Provides full uncertainty profile per prediction.

    Alpha levels: 0.05 (95%), 0.10 (90%), 0.20 (80%)
    """

    ALPHA_LEVELS = [0.05, 0.10, 0.20]
    METHODS      = ["thr", "lac", "aps"]

    def __init__(self):
        self.predictors: dict = {}
        self.fitted_: bool    = False

    def fit(
        self,
        y_cal: np.ndarray,
        p_cal: np.ndarray,
    ) -> "MultiAlphaConformalPredictor":
        """Fit conformal predictors for all alpha × method combinations."""
        logger.info(f"  Fitting {len(self.ALPHA_LEVELS)} alpha × "
                    f"{len(self.METHODS)} method = "
                    f"{len(self.ALPHA_LEVELS)*len(self.METHODS)} predictors")

        for method in self.METHODS:
            for alpha in self.ALPHA_LEVELS:
                key = f"{method}_a{int(alpha*100):02d}"
                cp  = ConformalPredictor(alpha=alpha, method=method)
                cp.fit(y_cal, p_cal)
                self.predictors[key] = cp
                logger.info(
                    f"    [{key}] q_hat={cp.q_hat_:.4f}  "
                    f"target_cov={1-alpha:.0%}"
                )

        self.fitted_ = True
        return self

    def evaluate_all(
        self,
        y_test: np.ndarray,
        p_test: np.ndarray,
        split_name: str = "test",
    ) -> pd.DataFrame:
        """Evaluate all predictors on test set."""
        records = []
        for key, cp in self.predictors.items():
            res = cp.evaluate(y_test, p_test)
            res["predictor_key"] = key
            res["split"]         = split_name
            records.append(res)

        df = pd.DataFrame(records)
        df = df.sort_values(["method", "alpha"]).reset_index(drop=True)
        return df

    def predict_sample(
        self,
        p_single: float,
        alpha: float = 0.10,
        method: str  = "thr",
    ) -> dict:
        """
        Generate prediction set for a single patient.
        Returns human-readable uncertainty report.
        """
        key = f"{method}_a{int(alpha*100):02d}"
        if key not in self.predictors:
            raise KeyError(f"Predictor {key} not fitted")

        cp   = self.predictors[key]
        sets = cp.predict(np.array([p_single]))[0]

        classes  = {0: "NO readmission", 1: "READMISSION"}
        in_set   = [classes[c] for c in range(2) if sets[c]]
        certain  = len(in_set) == 1

        return {
            "p_readmit"      : round(p_single, 4),
            "prediction_set" : in_set,
            "certain"        : certain,
            "coverage_level" : f"{(1-alpha)*100:.0f}%",
            "q_hat"          : round(cp.q_hat_, 4),
            "interpretation" : (
                f"With {(1-alpha)*100:.0f}% coverage guarantee: "
                f"{'CERTAIN — ' + in_set[0] if certain else 'UNCERTAIN — both outcomes possible'}"
            ),
        }


# ══════════════════════════════════════════════════════════════════════════
# Coverage drift analysis
# ══════════════════════════════════════════════════════════════════════════

def coverage_drift_analysis(
    cp:       ConformalPredictor,
    y_val:    np.ndarray,
    p_val:    np.ndarray,
    y_test:   np.ndarray,
    p_test:   np.ndarray,
) -> dict:
    """
    Compare coverage between reference (val) and production (test).
    Coverage drop under drift = additional concept drift evidence.

    Under distribution shift, the conformal guarantee may not hold.
    This analysis quantifies the coverage degradation.
    """
    val_eval  = cp.evaluate(y_val,  p_val)
    test_eval = cp.evaluate(y_test, p_test)

    cov_delta     = test_eval["empirical_coverage"] - val_eval["empirical_coverage"]
    size_delta    = test_eval["set_size_mean"]       - val_eval["set_size_mean"]
    guarantee_held= test_eval["coverage_satisfied"]

    return {
        "val_coverage"    : val_eval["empirical_coverage"],
        "test_coverage"   : test_eval["empirical_coverage"],
        "coverage_delta"  : round(cov_delta,  4),
        "target_coverage" : val_eval["target_coverage"],
        "guarantee_held"  : guarantee_held,
        "val_set_size"    : val_eval["set_size_mean"],
        "test_set_size"   : test_eval["set_size_mean"],
        "set_size_delta"  : round(size_delta, 4),
        "drift_signal"    : not guarantee_held or cov_delta < -0.05,
    }


# ══════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════

def _plot_coverage_profile(
    results_df: pd.DataFrame,
    model_name: str,
):
    """Coverage profile across alpha levels and methods."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    methods = results_df["method"].unique()
    colors  = {"thr": "#e74c3c", "lac": "#2ecc71", "aps": "#3498db"}
    alphas  = sorted(results_df["alpha"].unique())

    # Left: Empirical vs target coverage
    ax = axes[0]
    for method in methods:
        sub = results_df[results_df["method"] == method].sort_values("alpha")
        ax.plot(
            1 - sub["alpha"],
            sub["empirical_coverage"],
            marker="o", linewidth=2,
            color=colors.get(method, "gray"),
            label=method.upper()
        )
    ax.plot([0.75, 1.0], [0.75, 1.0], "k--",
            linewidth=1.5, alpha=0.5, label="Perfect coverage")
    ax.set_xlabel("Target Coverage (1-α)")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Coverage: Empirical vs Target", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Middle: Coverage gap
    ax = axes[1]
    for method in methods:
        sub = results_df[results_df["method"] == method].sort_values("alpha")
        ax.bar(
            [f"{method}\nα={a:.0%}" for a in sub["alpha"]],
            sub["coverage_gap"],
            color=colors.get(method, "gray"),
            alpha=0.8, edgecolor="black", linewidth=0.5
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(-0.05, color="red", linestyle="--",
               linewidth=1, alpha=0.7, label="Critical gap (-5%)")
    ax.set_ylabel("Coverage Gap (empirical - target)")
    ax.set_title("Coverage Gap\n(positive = over-covered)", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: Prediction set sizes
    ax = axes[2]
    for method in methods:
        sub = results_df[results_df["method"] == method].sort_values("alpha")
        ax.plot(
            1 - sub["alpha"],
            sub["set_size_mean"],
            marker="s", linewidth=2,
            color=colors.get(method, "gray"),
            label=method.upper()
        )
    ax.axhline(1.0, color="gray", linestyle="--",
               linewidth=1, alpha=0.5, label="Singleton sets")
    ax.set_xlabel("Target Coverage (1-α)")
    ax.set_ylabel("Mean Prediction Set Size")
    ax.set_title("Efficiency: Mean Set Size\n(smaller = more informative)",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 2.1)

    plt.suptitle(
        f"Conformal Prediction Profile — {model_name} [test split]",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = FIGURE_DIR / f"26_conformal_coverage_{model_name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close("all")   # free the figure (long runs accumulate)
    logger.info(f"  Saved: {path.name}")


def _plot_coverage_drift(
    drift_results: dict,
    model_name: str,
):
    """Coverage comparison val vs test across alpha levels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    alphas         = [0.05, 0.10, 0.20]
    target_covs    = [0.95, 0.90, 0.80]
    val_covs       = [drift_results[f"thr_a{int(a*100):02d}"]["val_coverage"]
                      for a in alphas]
    test_covs      = [drift_results[f"thr_a{int(a*100):02d}"]["test_coverage"]
                      for a in alphas]
    drift_signals  = [drift_results[f"thr_a{int(a*100):02d}"]["drift_signal"]
                      for a in alphas]

    x     = np.arange(len(alphas))
    width = 0.25

    # Coverage comparison
    ax = axes[0]
    ax.bar(x - width, target_covs, width, label="Target",
           color="#95a5a6", edgecolor="black", linewidth=0.5)
    ax.bar(x,          val_covs,   width, label="Val (reference)",
           color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x + width,  test_covs,  width, label="Test (production)",
           color="#e74c3c", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"α={a:.0%}\n({t:.0%} target)"
                        for a, t in zip(alphas, target_covs)])
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage: Val vs Test vs Target\n(THR method)",
                 fontweight="bold")
    ax.legend()
    ax.set_ylim(0.6, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for i, (tc, vc, tec, ds) in enumerate(
        zip(target_covs, val_covs, test_covs, drift_signals)
    ):
        ax.text(i - width, tc + 0.005, f"{tc:.2f}",
                ha="center", fontsize=7)
        ax.text(i,          vc + 0.005, f"{vc:.2f}",
                ha="center", fontsize=7)
        color = "#e74c3c" if ds else "#27ae60"
        ax.text(i + width, tec + 0.005, f"{tec:.2f}",
                ha="center", fontsize=7, color=color, fontweight="bold")

    # Coverage delta
    ax2 = axes[1]
    deltas = [test_covs[i] - target_covs[i] for i in range(len(alphas))]
    colors_bar = ["#e74c3c" if d < 0 else "#2ecc71" for d in deltas]
    bars = ax2.bar([f"α={a:.0%}" for a in alphas], deltas,
                   color=colors_bar, edgecolor="black", linewidth=0.6)
    ax2.axhline(0,    color="black",  linewidth=1)
    ax2.axhline(-0.05, color="red", linestyle="--",
                linewidth=1, alpha=0.7, label="Critical (-5%)")
    ax2.set_ylabel("Test Coverage − Target Coverage")
    ax2.set_title("Coverage Deficit Under Drift\n(negative = guarantee violated)",
                  fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    for bar, delta, ds in zip(bars, deltas, drift_signals):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            delta + (0.002 if delta >= 0 else -0.005),
            f"{delta:+.3f}",
            ha="center", fontsize=9, fontweight="bold",
            color="#e74c3c" if ds else "#27ae60"
        )

    plt.suptitle(
        f"Coverage Drift Analysis — {model_name}\n"
        f"(red = coverage guarantee violated under drift)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = FIGURE_DIR / f"27_coverage_drift_{model_name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close("all")   # free the figure (long runs accumulate)
    logger.info(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_quantifier() -> dict:
    logger.info("=" * 70)
    logger.info("DriftSentinel — Conformal Prediction Quantifier")
    logger.info("=" * 70)

    # Load data
    val  = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")

    feat_cols = [c for c in val.columns
                 if c not in {"readmitted_binary", "readmitted_multi"}]

    # Load model + calibrator
    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model = pickle.load(f)
    from src.uncertainty.calibration import IsotonicCalibrator
    with open(ARTIFACTS_DIR / "calibrator_isotonic_lgbm_v1.pkl", "rb") as f:
        calibrator = pickle.load(f)

    X_val  = pd.DataFrame(val[feat_cols].values,  columns=feat_cols)
    X_test = pd.DataFrame(test[feat_cols].values, columns=feat_cols)
    y_val  = val[TARGET].values
    y_test = test[TARGET].values

    p_val_raw  = model.predict_proba(X_val)[:, 1]
    p_test_raw = model.predict_proba(X_test)[:, 1]

    # Use calibrated probabilities
    p_val  = calibrator.transform(p_val_raw)
    p_test = calibrator.transform(p_test_raw)

    logger.info(f"  Calibrated probabilities loaded")
    logger.info(f"  Val  mean proba: {p_val.mean():.4f}")
    logger.info(f"  Test mean proba: {p_test.mean():.4f}")

    # ── Step 1: Multi-alpha conformal prediction ───────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Step 1: Multi-alpha Conformal Prediction")
    logger.info("=" * 50)

    mcp = MultiAlphaConformalPredictor()
    mcp.fit(y_val, p_val)

    results_df = mcp.evaluate_all(y_test, p_test, split_name="test")

    logger.info("\n  Coverage results on TEST split:")
    logger.info(
        f"  {'Key':<12} {'Target':>8} {'Empirical':>10} "
        f"{'Gap':>8} {'SetSize':>8} {'Satisfied':>10}"
    )
    logger.info("  " + "-" * 60)
    for _, row in results_df.iterrows():
        satisfied = "✓" if row["coverage_satisfied"] else "✗ VIOLATED"
        logger.info(
            f"  {row['predictor_key']:<12} "
            f"{row['target_coverage']:>8.3f} "
            f"{row['empirical_coverage']:>10.3f} "
            f"{row['coverage_gap']:>+8.3f} "
            f"{row['set_size_mean']:>8.3f} "
            f"{satisfied:>10}"
        )

    # ── Step 2: Coverage drift analysis ───────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Step 2: Coverage Drift Analysis (val vs test)")
    logger.info("=" * 50)

    drift_results = {}
    for key, cp in mcp.predictors.items():
        drift_results[key] = coverage_drift_analysis(
            cp, y_val, p_val, y_test, p_test
        )

    logger.info(
        f"\n  {'Key':<12} {'Val_cov':>8} {'Test_cov':>9} "
        f"{'Delta':>8} {'Guaranteed':>11} {'DriftSignal':>12}"
    )
    logger.info("  " + "-" * 65)
    for key, res in drift_results.items():
        logger.info(
            f"  {key:<12} "
            f"{res['val_coverage']:>8.3f} "
            f"{res['test_coverage']:>9.3f} "
            f"{res['coverage_delta']:>+8.3f} "
            f"{'YES' if res['guarantee_held'] else 'NO VIOLATED':>11} "
            f"{'DRIFT' if res['drift_signal'] else 'stable':>12}"
        )

    # ── Step 3: Single patient example ────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Step 3: Single Patient Prediction Example")
    logger.info("=" * 50)

    example_probas = [0.25, 0.45, 0.55, 0.75, 0.90]
    for p in example_probas:
        result = mcp.predict_sample(p, alpha=0.10, method="thr")
        logger.info(
            f"  p={p:.2f} → set={result['prediction_set']}  "
            f"certain={result['certain']}  "
            f"{result['interpretation']}"
        )

    # ── Step 4: Plots ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Step 4: Generating plots")
    logger.info("=" * 50)

    _plot_coverage_profile(results_df, "lgbm_v1")
    _plot_coverage_drift(drift_results, "lgbm_v1")

    # ── Save artifacts ─────────────────────────────────────────────────────
    cp_path = ARTIFACTS_DIR / "conformal_predictor_lgbm_v1.pkl"
    with open(cp_path, "wb") as f:
        pickle.dump(mcp, f)
    logger.info(f"\n  Conformal predictor saved: {cp_path}")

    results_df.to_csv(
        REPORTS_DIR / "conformal_results_lgbm_v1.csv", index=False
    )

    # Summary
    thr_90  = results_df[
        (results_df["method"] == "thr") & (results_df["alpha"] == 0.10)
    ].iloc[0]
    n_drift = sum(1 for r in drift_results.values() if r["drift_signal"])

    report = {
        "model_name"          : "lgbm_v1",
        "n_predictors"        : len(mcp.predictors),
        "thr_90_coverage"     : thr_90["empirical_coverage"],
        "thr_90_gap"          : thr_90["coverage_gap"],
        "thr_90_set_size"     : thr_90["set_size_mean"],
        "thr_90_satisfied"    : bool(thr_90["coverage_satisfied"]),
        "n_drift_signals"     : n_drift,
        "drift_results"       : {
            k: {kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else vv)
                for kk, vv in v.items()}
            for k, v in drift_results.items()
        },
    }

    report_path = REPORTS_DIR / "conformal_report_lgbm_v1.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Conformal report saved: {report_path}")

    logger.info("=" * 70)
    logger.info("Conformal Prediction Complete")
    logger.info(f"  THR 90% coverage  : {thr_90['empirical_coverage']:.4f} "
                f"(target={thr_90['target_coverage']:.2f})")
    logger.info(f"  Coverage gap      : {thr_90['coverage_gap']:+.4f}")
    logger.info(f"  Mean set size     : {thr_90['set_size_mean']:.4f}")
    logger.info(f"  Drift signals     : {n_drift}/{len(drift_results)}")
    logger.info("  Next: threshold.py")
    logger.info("=" * 70)

    print(f"\nTHR 90% coverage : {thr_90['empirical_coverage']:.4f}")
    print(f"Coverage gap     : {thr_90['coverage_gap']:+.4f}")
    print(f"Mean set size    : {thr_90['set_size_mean']:.4f}")
    print(f"Drift signals    : {n_drift}/{len(drift_results)}")

    return report


if __name__ == "__main__":
    run_quantifier()