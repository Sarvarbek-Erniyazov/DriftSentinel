"""
DriftSentinel — Adaptive Threshold Optimizer
Under concept drift, the optimal classification threshold shifts.
Val-optimal threshold may be suboptimal on drifted test distribution.

Methods:
    1. F1-max threshold        — maximize F1 on calibration set
    2. Cost-sensitive threshold — clinical cost matrix (FP vs FN)
    3. Youden's J threshold    — maximize sensitivity + specificity
    4. Precision-recall tradeoff — per operating point
    5. Drift-aware recalibration — window-based threshold adaptation

Clinical context (hospital readmission):
    FN (missed readmission) cost >> FP (unnecessary intervention) cost
    → threshold should favor recall over precision
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib
matplotlib.use("Agg")   # headless: figures save to disk only, no GUI window
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve,
    roc_curve, confusion_matrix, brier_score_loss
)
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger
from src.uncertainty.calibration import IsotonicCalibrator

logger = get_logger("threshold")

ROOT          = Path(__file__).resolve().parents[2]
TRAIN_DIR     = ROOT / "data"    / "train"
MODELS_DIR    = ROOT / "outputs" / "models"
FIGURE_DIR    = ROOT / "outputs" / "figure"
REPORTS_DIR   = ROOT / "outputs" / "log"
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_binary"

# Clinical cost matrix
# FN: missed readmission → patient re-hospitalized without preparation
# FP: unnecessary intervention → wasted resources
COST_FN = 5.0   # FN 5x more costly than FP (clinical judgment)
COST_FP = 1.0


# ══════════════════════════════════════════════════════════════════════════
# Threshold search methods
# ══════════════════════════════════════════════════════════════════════════

def threshold_f1_max(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    """Find threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s     = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s[:-1])
    return float(thresholds[best_idx]), float(f1s[best_idx])


def threshold_youden(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    """
    Youden's J statistic: J = sensitivity + specificity - 1.
    Maximizes balanced performance regardless of class imbalance.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def threshold_cost_sensitive(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> tuple[float, float]:
    """
    Cost-sensitive threshold optimization.

    Minimises the POPULATION-WEIGHTED expected cost per patient:

        (FN * cost_fn + FP * cost_fp) / N

    CORRECTED (Tier 1.3). The previous implementation normalised each error type
    by its OWN class size — `FN/n_pos * cost_fn + FP/n_neg * cost_fp` — which
    divides prevalence out and silently multiplies the requested FN:FP ratio by
    the inverse odds of the positive class. At the merged target's 47.6%
    prevalence that turned 5:1 into 5.5:1 (minor); at the new `<30` target's
    11.16% prevalence it turns 5:1 into 39.8:1 (serious).

    NOTE: this correction does NOT explain the 97% alert rate. That is what an
    optimal 5:1 rule does on this probability distribution — see
    `src/uncertainty/threshold_policy.py` for the analysis and for the alert
    budget, cost-ratio sweep and decision-curve tooling that make the operating
    point defensible.
    """
    from src.uncertainty.threshold_policy import threshold_cost_sensitive as _tcs

    res = _tcs(y_true, y_proba, cost_fn=cost_fn, cost_fp=cost_fp)
    if res["degenerate"]:
        logger.warning(
            f"DEGENERATE operating point: threshold {res['threshold']:.4f} flags "
            f"{res['predicted_positive_rate']:.1%} of patients. This is not a "
            f"deployable recommendation — see threshold_policy.py for the "
            f"budget-constrained alternatives.")
    return float(res["threshold"]), float(res["expected_cost_per_patient"])


def threshold_precision_target(
    y_true:    np.ndarray,
    y_proba:   np.ndarray,
    min_prec:  float = 0.50,
) -> tuple[float, float]:
    """
    Find highest-recall threshold that meets minimum precision requirement.
    Clinical use: ensure at least 50% precision to avoid alert fatigue.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    valid = [(t, r) for p, r, t in zip(precisions, recalls, thresholds)
             if p >= min_prec]
    if not valid:
        return 0.5, 0.0
    best = max(valid, key=lambda x: x[1])
    return float(best[0]), float(best[1])


# ══════════════════════════════════════════════════════════════════════════
# Per-window threshold analysis
# ══════════════════════════════════════════════════════════════════════════

def window_threshold_analysis(
    y_true:    np.ndarray,
    y_proba:   np.ndarray,
    n_windows: int   = 10,
    method:    str   = "f1_max",
) -> pd.DataFrame:
    """
    Compute optimal threshold per time window.
    Shows how threshold needs to adapt under drift.
    """
    n        = len(y_true)
    win_size = n // n_windows
    records  = []

    for i in range(n_windows):
        start = i * win_size
        end   = start + win_size if i < n_windows - 1 else n
        y_w   = y_true[start:end]
        p_w   = y_proba[start:end]

        if len(np.unique(y_w)) < 2:
            continue

        if method == "f1_max":
            thr, score = threshold_f1_max(y_w, p_w)
        elif method == "youden":
            thr, score = threshold_youden(y_w, p_w)
        else:
            thr, score = threshold_f1_max(y_w, p_w)

        y_pred = (p_w >= thr).astype(int)
        records.append({
            "window"    : i + 1,
            "start_idx" : start,
            "end_idx"   : end,
            "pos_rate"  : round(float(y_w.mean()),  4),
            "opt_threshold": round(thr, 4),
            "score"     : round(score, 4),
            "f1"        : round(float(f1_score(y_w, y_pred, zero_division=0)), 4),
            "precision" : round(float(precision_score(y_w, y_pred, zero_division=0)), 4),
            "recall"    : round(float(recall_score(y_w, y_pred, zero_division=0)), 4),
            "auc"       : round(float(roc_auc_score(y_w, p_w)), 4),
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════
# Threshold comparator
# ══════════════════════════════════════════════════════════════════════════

def compare_thresholds(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
    thresholds: dict,
    split_name: str = "test",
) -> pd.DataFrame:
    """
    Compare multiple thresholds on same split.
    Shows what each threshold optimizes and its tradeoffs.
    """
    records = []
    for name, thr in thresholds.items():
        y_pred = (y_proba >= thr).astype(int)
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        records.append({
            "threshold_name": name,
            "threshold"     : round(thr, 4),
            "split"         : split_name,
            "f1"            : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "precision"     : round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall"        : round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "auc"           : round(float(roc_auc_score(y_true, y_proba)), 4),
            "tp"            : int(tp),
            "fp"            : int(fp),
            "tn"            : int(tn),
            "fn"            : int(fn),
            "fn_rate"       : round(fn / (tp + fn + 1e-8), 4),
            "fp_rate"       : round(fp / (tn + fp + 1e-8), 4),
            "cost"          : round(
                fn / (tp + fn + 1e-8) * COST_FN +
                fp / (tn + fp + 1e-8) * COST_FP, 4
            ),
        })

    return pd.DataFrame(records).sort_values("f1", ascending=False)


# ══════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════

def _plot_threshold_analysis(
    y_val:   np.ndarray,
    p_val:   np.ndarray,
    y_test:  np.ndarray,
    p_test:  np.ndarray,
    thresholds_val:  dict,
    thresholds_test: dict,
    model_name: str,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    splits = [
        ("Val (reference)", y_val,  p_val,  thresholds_val),
        ("Test (drift)",    y_test, p_test, thresholds_test),
    ]

    for row, (split_name, y, p, thrs) in enumerate(splits):
        precisions, recalls, thr_vals = precision_recall_curve(y, p)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        fpr_arr, tpr_arr, roc_thrs    = roc_curve(y, p)

        # Left: Precision-Recall curve with threshold markers
        ax = axes[row, 0]
        ax.plot(recalls, precisions, color="#3498db", linewidth=2, label="PR curve")
        colors_thr = {
            "f1_max"    : "#e74c3c",
            "youden"    : "#2ecc71",
            "cost"      : "#f39c12",
            "prec50"    : "#9b59b6",
            "fixed_05"  : "#1abc9c",
        }
        for thr_name, thr_val in thrs.items():
            y_pred  = (p >= thr_val).astype(int)
            prec    = precision_score(y, y_pred, zero_division=0)
            rec     = recall_score(y, y_pred, zero_division=0)
            color   = colors_thr.get(thr_name, "gray")
            ax.scatter(rec, prec, s=100, color=color,
                       zorder=5, label=f"{thr_name}={thr_val:.3f}")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — {split_name}", fontweight="bold")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.3)

        # Middle: F1 vs threshold
        ax = axes[row, 1]
        thr_range = np.linspace(0.1, 0.9, 100)
        f1_vals   = [f1_score(y, (p >= t).astype(int), zero_division=0)
                     for t in thr_range]
        prec_vals = [precision_score(y, (p >= t).astype(int), zero_division=0)
                     for t in thr_range]
        rec_vals  = [recall_score(y, (p >= t).astype(int), zero_division=0)
                     for t in thr_range]

        ax.plot(thr_range, f1_vals,   color="#e74c3c", linewidth=2, label="F1")
        ax.plot(thr_range, prec_vals, color="#3498db", linewidth=1.5,
                linestyle="--", label="Precision", alpha=0.8)
        ax.plot(thr_range, rec_vals,  color="#2ecc71", linewidth=1.5,
                linestyle="--", label="Recall", alpha=0.8)

        for thr_name, thr_val in thrs.items():
            color = colors_thr.get(thr_name, "gray")
            ax.axvline(thr_val, color=color, linestyle=":",
                       linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"Metrics vs Threshold — {split_name}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Right: Cost vs threshold
        ax = axes[row, 2]
        costs = []
        for t in thr_range:
            y_pred = (p >= t).astype(int)
            cm     = confusion_matrix(y, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn_c, fp_c, fn_c, tp_c = cm.ravel()
                n_pos = y.sum()
                n_neg = len(y) - n_pos
                cost  = (fn_c / (n_pos + 1e-8)) * COST_FN + \
                        (fp_c / (n_neg + 1e-8)) * COST_FP
            else:
                cost = float("inf")
            costs.append(cost)

        ax.plot(thr_range, costs, color="#9b59b6", linewidth=2, label="Total cost")
        cost_thr = thrs.get("cost", 0.5)
        ax.axvline(cost_thr, color="#f39c12", linestyle="--",
                   linewidth=2, label=f"Cost-opt={cost_thr:.3f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(f"Cost (FN×{COST_FN} + FP×{COST_FP})")
        ax.set_title(f"Clinical Cost vs Threshold — {split_name}",
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"Adaptive Threshold Analysis — {model_name}\n"
        f"(clinical cost: FN={COST_FN}x, FP={COST_FP}x)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = FIGURE_DIR / f"28_threshold_analysis_{model_name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close("all")   # free the figure (long runs accumulate)
    logger.info(f"  Saved: {path.name}")


def _plot_threshold_drift(
    window_df_val:  pd.DataFrame,
    window_df_test: pd.DataFrame,
    model_name: str,
):
    """Show threshold drift across sequential evaluation windows."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Combined windows
    window_df_val  = window_df_val.copy()
    window_df_test = window_df_test.copy()
    window_df_val["split"]  = "val"
    window_df_test["split"] = "test"
    combined = pd.concat([window_df_val, window_df_test], ignore_index=True)
    combined["window_abs"] = range(len(combined))

    # Left: Optimal threshold per window
    ax = axes[0]
    ax.plot(window_df_val["window"],  window_df_val["opt_threshold"],
            marker="o", color="#2ecc71", linewidth=2, label="Val windows")
    ax.plot(window_df_test["window"], window_df_test["opt_threshold"],
            marker="s", color="#e74c3c", linewidth=2, label="Test windows")
    ax.axvline(len(window_df_val) + 0.5, color="gray",
               linestyle="--", alpha=0.5, label="Drift boundary")
    ax.set_xlabel("Window")
    ax.set_ylabel("Optimal Threshold (F1-max)")
    ax.set_title("Threshold Shift Across Windows", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Middle: Positive rate vs threshold
    ax = axes[1]
    ax.plot(window_df_val["window"],  window_df_val["pos_rate"],
            marker="o", color="#2ecc71", linewidth=2, label="Val pos rate")
    ax.plot(window_df_test["window"], window_df_test["pos_rate"],
            marker="s", color="#e74c3c", linewidth=2, label="Test pos rate")
    ax.set_xlabel("Window")
    ax.set_ylabel("Positive Rate (readmission rate)")
    ax.set_title("Label Distribution per Window\n(concept drift evidence)",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: F1 per window at optimal threshold
    ax = axes[2]
    ax.plot(window_df_val["window"],  window_df_val["f1"],
            marker="o", color="#2ecc71", linewidth=2, label="Val F1")
    ax.plot(window_df_test["window"], window_df_test["f1"],
            marker="s", color="#e74c3c", linewidth=2, label="Test F1")
    ax.set_xlabel("Window")
    ax.set_ylabel("F1 at Optimal Threshold")
    ax.set_title("Per-Window F1 Performance", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"Threshold Adaptation Under Drift — {model_name}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = FIGURE_DIR / f"29_threshold_drift_{model_name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close("all")   # free the figure (long runs accumulate)
    logger.info(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_threshold() -> dict:
    logger.info("=" * 70)
    logger.info("DriftSentinel — Adaptive Threshold Optimizer")
    logger.info("=" * 70)

    val  = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")

    feat_cols = [c for c in val.columns
                 if c not in {"readmitted_binary", "readmitted_multi"}]

    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "calibrator_isotonic_lgbm_v1.pkl", "rb") as f:
        calibrator = pickle.load(f)

    X_val  = pd.DataFrame(val[feat_cols].values,  columns=feat_cols)
    X_test = pd.DataFrame(test[feat_cols].values, columns=feat_cols)
    y_val  = val[TARGET].values
    y_test = test[TARGET].values

    p_val_raw  = model.predict_proba(X_val)[:, 1]
    p_test_raw = model.predict_proba(X_test)[:, 1]

    p_val  = calibrator.transform(p_val_raw)
    p_test = calibrator.transform(p_test_raw)

    logger.info(f"  Val  pos rate: {y_val.mean():.4f}")
    logger.info(f"  Test pos rate: {y_test.mean():.4f}")

    # ── Step 1: Find optimal thresholds on val ─────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Step 1: Optimal thresholds on VAL (calibration set)")
    logger.info("=" * 50)

    f1_thr_val,   f1_score_val  = threshold_f1_max(y_val, p_val)
    youden_thr_val, youden_j_val = threshold_youden(y_val, p_val)
    cost_thr_val, cost_val       = threshold_cost_sensitive(y_val, p_val)
    prec50_thr_val, rec50_val    = threshold_precision_target(
        y_val, p_val, min_prec=0.50
    )

    thresholds_val = {
        "f1_max"  : f1_thr_val,
        "youden"  : youden_thr_val,
        "cost"    : cost_thr_val,
        "prec50"  : prec50_thr_val,
        "fixed_05": 0.50,
    }

    logger.info(f"  {'Method':<15} {'Threshold':>10} {'Score':>8}")
    logger.info("  " + "-" * 36)
    logger.info(f"  {'f1_max':<15} {f1_thr_val:>10.4f} {f1_score_val:>8.4f}")
    logger.info(f"  {'youden':<15} {youden_thr_val:>10.4f} {youden_j_val:>8.4f}")
    logger.info(f"  {'cost_sensitive':<15} {cost_thr_val:>10.4f} {cost_val:>8.4f}")
    logger.info(f"  {'prec50':<15} {prec50_thr_val:>10.4f} {rec50_val:>8.4f}")
    logger.info(f"  {'fixed_0.5':<15} {0.50:>10.4f}")

    # ── Step 2: Apply val thresholds to test ───────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Step 2: Apply val thresholds to TEST (drift window)")
    logger.info("=" * 50)

    comparison_df = compare_thresholds(y_test, p_test, thresholds_val, "test")

    logger.info(f"\n  {'Name':<15} {'Thr':>6} {'F1':>7} {'Prec':>7} "
                f"{'Recall':>8} {'FN_rate':>9} {'Cost':>7}")
    logger.info("  " + "-" * 65)
    for _, row in comparison_df.iterrows():
        logger.info(
            f"  {row['threshold_name']:<15} "
            f"{row['threshold']:>6.4f} "
            f"{row['f1']:>7.4f} "
            f"{row['precision']:>7.4f} "
            f"{row['recall']:>8.4f} "
            f"{row['fn_rate']:>9.4f} "
            f"{row['cost']:>7.4f}"
        )

    # ── Step 3: Find optimal thresholds directly on test ──────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Step 3: Oracle thresholds on TEST (what we could achieve)")
    logger.info("=" * 50)

    f1_thr_test,  f1_score_test  = threshold_f1_max(y_test, p_test)
    youden_thr_test, youden_j_test = threshold_youden(y_test, p_test)
    cost_thr_test, cost_test      = threshold_cost_sensitive(y_test, p_test)

    thresholds_test = {
        "f1_max"  : f1_thr_test,
        "youden"  : youden_thr_test,
        "cost"    : cost_thr_test,
        "prec50"  : threshold_precision_target(y_test, p_test, 0.50)[0],
        "fixed_05": 0.50,
    }

    logger.info(f"  {'Method':<15} {'Val_thr':>9} {'Test_thr':>10} {'Shift':>8}")
    logger.info("  " + "-" * 44)
    for method, val_t in [
        ("f1_max", f1_thr_val),
        ("youden",  youden_thr_val),
        ("cost",    cost_thr_val),
    ]:
        test_t = thresholds_test[method]
        shift  = test_t - val_t
        logger.info(
            f"  {method:<15} {val_t:>9.4f} {test_t:>10.4f} {shift:>+8.4f}"
        )

    # ── Step 4: Window analysis ────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Step 4: Window-based threshold analysis")
    logger.info("=" * 50)

    y_combined = np.concatenate([y_val,  y_test])
    p_combined = np.concatenate([p_val,  p_test])

    window_val  = window_threshold_analysis(y_val,  p_val,  n_windows=10)
    window_test = window_threshold_analysis(y_test, p_test, n_windows=10)

    logger.info(f"  Val  threshold range: "
                f"[{window_val['opt_threshold'].min():.4f}, "
                f"{window_val['opt_threshold'].max():.4f}]  "
                f"mean={window_val['opt_threshold'].mean():.4f}")
    logger.info(f"  Test threshold range: "
                f"[{window_test['opt_threshold'].min():.4f}, "
                f"{window_test['opt_threshold'].max():.4f}]  "
                f"mean={window_test['opt_threshold'].mean():.4f}")
    logger.info(f"  Threshold shift val→test: "
                f"{window_test['opt_threshold'].mean() - window_val['opt_threshold'].mean():+.4f}")

    # ── Step 5: Best threshold recommendation ─────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Step 5: Recommended threshold for drifted production")
    logger.info("=" * 50)

    # Clinical recommendation: use cost-sensitive under drift
    recommended_thr = cost_thr_val
    y_rec  = (p_test >= recommended_thr).astype(int)
    y_fixed= (p_test >= 0.50).astype(int)

    rec_f1    = f1_score(y_test,   y_rec,   zero_division=0)
    fixed_f1  = f1_score(y_test,   y_fixed, zero_division=0)
    rec_rec   = recall_score(y_test, y_rec,   zero_division=0)
    fixed_rec = recall_score(y_test, y_fixed, zero_division=0)

    logger.info(f"  Fixed 0.50     : F1={fixed_f1:.4f}  Recall={fixed_rec:.4f}")
    logger.info(f"  Cost-sensitive : F1={rec_f1:.4f}  "
                f"Recall={rec_rec:.4f}  thr={recommended_thr:.4f}")
    logger.info(f"  F1 improvement : {rec_f1 - fixed_f1:+.4f}")
    logger.info(f"  Recall gain    : {rec_rec - fixed_rec:+.4f}")

    if rec_rec > fixed_rec:
        logger.info(
            f"  → Cost-sensitive threshold captures "
            f"{int((rec_rec - fixed_rec) * y_test.sum())} "
            f"more true readmissions"
        )

    # ── Plots ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Step 6: Generating plots")
    logger.info("=" * 50)

    _plot_threshold_analysis(
        y_val, p_val, y_test, p_test,
        thresholds_val, thresholds_test, "lgbm_v1"
    )
    _plot_threshold_drift(window_val, window_test, "lgbm_v1")

    # ── Save ───────────────────────────────────────────────────────────────
    comparison_df.to_csv(
        REPORTS_DIR / "threshold_comparison_lgbm_v1.csv", index=False
    )

    report = {
        "model_name"        : "lgbm_v1",
        "val_thresholds"    : {k: round(v, 4) for k, v in thresholds_val.items()},
        "test_oracle_thresholds": {k: round(v, 4) for k, v in thresholds_test.items()},
        "threshold_shift"   : {
            "f1_max": round(f1_thr_test - f1_thr_val, 4),
            "youden": round(youden_thr_test - youden_thr_val, 4),
            "cost"  : round(cost_thr_test - cost_thr_val, 4),
        },
        "recommended_threshold": round(recommended_thr, 4),
        "recommended_method"   : "cost_sensitive",
        "test_performance"  : {
            "fixed_05" : {
                "f1"    : round(fixed_f1,  4),
                "recall": round(fixed_rec, 4),
            },
            "cost_sensitive": {
                "f1"    : round(rec_f1,  4),
                "recall": round(rec_rec, 4),
            },
        },
        "window_analysis"   : {
            "val_mean_threshold" : round(
                float(window_val["opt_threshold"].mean()),  4
            ),
            "test_mean_threshold": round(
                float(window_test["opt_threshold"].mean()), 4
            ),
            "threshold_drift"    : round(
                float(window_test["opt_threshold"].mean()
                      - window_val["opt_threshold"].mean()), 4
            ),
        },
    }

    report_path = REPORTS_DIR / "threshold_report_lgbm_v1.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Threshold report saved: {report_path}")

    logger.info("=" * 70)
    logger.info("Threshold Optimization Complete")
    logger.info(f"  Recommended threshold : {recommended_thr:.4f} (cost-sensitive)")
    logger.info(f"  vs fixed 0.50         : F1 {rec_f1-fixed_f1:+.4f}  "
                f"Recall {rec_rec-fixed_rec:+.4f}")
    logger.info("  Next: adversarial/ → README")
    logger.info("=" * 70)

    print(f"\nRecommended threshold : {recommended_thr:.4f}")
    print(f"Fixed 0.50  F1={fixed_f1:.4f}  Recall={fixed_rec:.4f}")
    print(f"Cost-opt    F1={rec_f1:.4f}  Recall={rec_rec:.4f}")
    print(f"F1 gain     : {rec_f1-fixed_f1:+.4f}")
    print(f"Recall gain : {rec_rec-fixed_rec:+.4f}")

    return report


if __name__ == "__main__":
    run_threshold()