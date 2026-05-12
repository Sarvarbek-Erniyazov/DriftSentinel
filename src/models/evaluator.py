"""
DriftSentinel — Model Evaluator
Evaluates trained models across train / val / test splits.
Primary purpose: demonstrate performance degradation under concept drift.

Evaluation suite:
    - AUC, F1, Precision, Recall, Brier Score
    - Calibration curve (reliability diagram)
    - Confusion matrix per split
    - Performance degradation report (val -> test delta)
    - Threshold analysis
    - Per-feature-group performance breakdown
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report
)
from sklearn.calibration import calibration_curve
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("evaluator")

ROOT       = Path(__file__).resolve().parents[2]
TRAIN_DIR  = ROOT / "data" / "train"
MODELS_DIR = ROOT / "outputs" / "models"
FIGURE_DIR = ROOT / "outputs" / "figure"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_binary"


# ── Loaders ────────────────────────────────────────────────────────────────
def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(TRAIN_DIR / "train_fs.parquet")
    val   = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test  = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")
    return train, val, test


def _load_lgbm() -> object:
    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        return pickle.load(f)


def _load_logreg() -> tuple[object, object]:
    with open(MODELS_DIR / "logreg_v1.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"]


def _get_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in {"readmitted_binary", "readmitted_multi"}]


# ── Core metrics ───────────────────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    split_name: str = "unknown"
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "split"      : split_name,
        "n_samples"  : int(len(y_true)),
        "pos_rate"   : round(float(y_true.mean()), 4),
        "pred_rate"  : round(float(y_pred.mean()), 4),
        "mean_proba" : round(float(y_proba.mean()), 4),
        "auc"        : round(float(roc_auc_score(y_true, y_proba)), 4),
        "avg_prec"   : round(float(average_precision_score(y_true, y_proba)), 4),
        "f1"         : round(float(f1_score(y_true, y_pred)), 4),
        "precision"  : round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall"     : round(float(recall_score(y_true, y_pred)), 4),
        "brier"      : round(float(brier_score_loss(y_true, y_proba)), 4),
        "threshold"  : threshold,
    }


# ── Optimal threshold (F1 maximization on val) ────────────────────────────
def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_thr = float(thresholds[min(best_idx, len(thresholds) - 1)])
    logger.info(f"  Optimal threshold (F1 max on val): {best_thr:.4f}  F1={f1s[best_idx]:.4f}")
    return best_thr


# ── ROC + PR curves ────────────────────────────────────────────────────────
def _plot_roc_pr(
    results: dict,
    model_name: str
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"train": "#3498db", "val": "#2ecc71", "test": "#e74c3c"}

    for split_name, res in results.items():
        y_true  = res["y_true"]
        y_proba = res["y_proba"]
        color   = colors.get(split_name, "gray")

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_val = res["metrics"]["auc"]
        axes[0].plot(fpr, tpr, color=color, linewidth=2,
                     label=f"{split_name} (AUC={auc_val:.4f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ap_val = res["metrics"]["avg_prec"]
        axes[1].plot(rec, prec, color=color, linewidth=2,
                     label=f"{split_name} (AP={ap_val:.4f})")

    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'ROC Curve — {model_name}', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve — {model_name}', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = FIGURE_DIR / f"20_roc_pr_{model_name}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    logger.info(f"  Saved: {path.name}")


# ── Calibration curves ─────────────────────────────────────────────────────
def _plot_calibration(results: dict, model_name: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"train": "#3498db", "val": "#2ecc71", "test": "#e74c3c"}

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5,
            alpha=0.6, label='Perfect calibration')

    for split_name, res in results.items():
        y_true  = res["y_true"]
        y_proba = res["y_proba"]
        color   = colors.get(split_name, "gray")

        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
        brier = res["metrics"]["brier"]
        ax.plot(mean_pred, frac_pos, marker='o', linewidth=2,
                color=color, label=f"{split_name} (Brier={brier:.4f})")

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Curve (Reliability Diagram) — {model_name}',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGURE_DIR / f"21_calibration_{model_name}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    logger.info(f"  Saved: {path.name}")


# ── Confusion matrices ─────────────────────────────────────────────────────
def _plot_confusion_matrices(results: dict, model_name: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    splits = ["train", "val", "test"]

    for ax, split_name in zip(axes, splits):
        if split_name not in results:
            ax.set_visible(False)
            continue
        res    = results[split_name]
        y_true = res["y_true"]
        y_pred = (res["y_proba"] >= res["threshold"]).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
        cm_pct = cm / cm.sum() * 100

        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                    ax=ax, linewidths=0.5,
                    xticklabels=['Pred NO', 'Pred Readmit'],
                    yticklabels=['True NO', 'True Readmit'])
        auc = res["metrics"]["auc"]
        ax.set_title(f'{split_name.upper()}\nAUC={auc:.4f}',
                     fontweight='bold')

    plt.suptitle(f'Confusion Matrices (%) — {model_name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = FIGURE_DIR / f"22_confusion_matrices_{model_name}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    logger.info(f"  Saved: {path.name}")


# ── Performance degradation ────────────────────────────────────────────────
def _degradation_report(
    metrics_by_split: dict,
    model_name: str
) -> dict:
    """
    Key output for DriftSentinel story:
    Shows how much performance drops from val to test (drift window).
    """
    logger.info("-" * 60)
    logger.info(f"Performance Degradation Report — {model_name}")
    logger.info("-" * 60)

    header = f"{'Metric':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Val→Test Δ':>12}"
    logger.info(header)
    logger.info("-" * 60)

    degradation = {}
    metrics_list = ["auc", "f1", "precision", "recall", "brier", "mean_proba"]

    for metric in metrics_list:
        tr_val  = metrics_by_split["train"].get(metric, np.nan)
        v_val   = metrics_by_split["val"].get(metric, np.nan)
        te_val  = metrics_by_split["test"].get(metric, np.nan)
        delta   = round(te_val - v_val, 4)
        sign    = "↓" if delta < 0 else "↑" if delta > 0 else "—"

        degradation[metric] = {
            "train": tr_val,
            "val"  : v_val,
            "test" : te_val,
            "delta": delta,
            "direction": sign,
        }
        logger.info(
            f"{metric:<15} {tr_val:>8.4f} {v_val:>8.4f} {te_val:>8.4f} "
            f"{delta:>+10.4f} {sign}"
        )

    # Drift severity assessment
    auc_delta = abs(degradation["auc"]["delta"])
    severity  = (
        "CRITICAL" if auc_delta > 0.10 else
        "MODERATE" if auc_delta > 0.05 else
        "MILD"
    )
    logger.info("-" * 60)
    logger.info(f"AUC degradation  : {degradation['auc']['delta']:+.4f}")
    logger.info(f"Drift severity   : {severity}")
    logger.info(f"DriftSentinel    : This degradation is what we detect and alert on")

    return {"degradation": degradation, "severity": severity}


# ── Degradation bar chart ──────────────────────────────────────────────────
def _plot_degradation(degradation_report: dict, model_name: str):
    deg  = degradation_report["degradation"]
    metrics = ["auc", "f1", "precision", "recall"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: metric across splits
    x     = np.arange(len(metrics))
    width = 0.25
    splits_colors = {"train": "#3498db", "val": "#2ecc71", "test": "#e74c3c"}

    for i, (split, color) in enumerate(splits_colors.items()):
        vals = [deg[m][split] for m in metrics]
        axes[0].bar(x + i*width, vals, width, label=split.upper(),
                    color=color, edgecolor='black', linewidth=0.5)

    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([m.upper() for m in metrics])
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'Metrics Across Temporal Windows\n{model_name}',
                      fontweight='bold')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)

    # Right: val→test delta
    deltas = [deg[m]["delta"] for m in metrics]
    colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in deltas]
    bars   = axes[1].bar(metrics, deltas, color=colors,
                          edgecolor='black', linewidth=0.6)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_ylabel('Val → Test Delta')
    axes[1].set_title(f'Performance Degradation (Val→Test)\n{model_name}',
                      fontweight='bold')
    axes[1].set_xticklabels([m.upper() for m in metrics])

    severity = degradation_report["severity"]
    color_sev = '#e74c3c' if severity == 'CRITICAL' else \
                '#f39c12' if severity == 'MODERATE' else '#2ecc71'
    axes[1].text(0.98, 0.05, f"Drift: {severity}",
                 transform=axes[1].transAxes, ha='right',
                 fontsize=12, fontweight='bold', color=color_sev,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for bar, delta in zip(bars, deltas):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     delta + (0.002 if delta >= 0 else -0.005),
                     f'{delta:+.4f}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = FIGURE_DIR / f"23_degradation_{model_name}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    logger.info(f"  Saved: {path.name}")


# ── Main evaluation function ───────────────────────────────────────────────
def evaluate_model(
    model_name: str,
    predict_fn,
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    feat_cols: list[str],
    threshold: float = None,
) -> dict:
    """
    Full evaluation suite for one model across all three splits.

    Parameters
    ----------
    model_name : display name
    predict_fn : callable(X) -> probabilities array
    threshold  : if None, computed from val split (F1 max)
    """
    logger.info("=" * 60)
    logger.info(f"Evaluating: {model_name}")
    logger.info("=" * 60)

    splits = {
        "train": train,
        "val"  : val,
        "test" : test,
    }

    # Compute predictions
    all_results = {}
    for split_name, df in splits.items():
        X      = df[feat_cols].values
        y_true = df[TARGET].values
        y_proba = predict_fn(X)
        all_results[split_name] = {
            "y_true" : y_true,
            "y_proba": y_proba,
        }

    # Optimal threshold from val
    if threshold is None:
        threshold = find_optimal_threshold(
            all_results["val"]["y_true"],
            all_results["val"]["y_proba"]
        )

    # Compute metrics
    metrics_by_split = {}
    for split_name, res in all_results.items():
        m = compute_metrics(
            res["y_true"], res["y_proba"],
            threshold=threshold,
            split_name=split_name
        )
        all_results[split_name]["metrics"]   = m
        all_results[split_name]["threshold"] = threshold
        metrics_by_split[split_name]         = m

        logger.info(
            f"  {split_name:<6} AUC={m['auc']:.4f}  "
            f"F1={m['f1']:.4f}  "
            f"Prec={m['precision']:.4f}  "
            f"Recall={m['recall']:.4f}  "
            f"Brier={m['brier']:.4f}  "
            f"mean_p={m['mean_proba']:.4f}"
        )

    # Plots
    _plot_roc_pr(all_results, model_name)
    _plot_calibration(all_results, model_name)
    _plot_confusion_matrices(all_results, model_name)

    # Degradation report
    deg_report = _degradation_report(metrics_by_split, model_name)
    _plot_degradation(deg_report, model_name)

    return {
        "model_name"      : model_name,
        "threshold"       : threshold,
        "metrics_by_split": metrics_by_split,
        "degradation"     : deg_report,
    }


# ── Comparative summary plot ───────────────────────────────────────────────
def _plot_model_comparison(lgbm_report: dict, lr_report: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    splits  = ["train", "val", "test"]
    metrics = ["auc", "f1"]
    colors  = {"lgbm_v1": "#e74c3c", "logreg_v1": "#3498db"}

    for ax_i, metric in enumerate(metrics):
        ax = axes[ax_i]
        x  = np.arange(len(splits))
        width = 0.35

        for m_i, report in enumerate([lgbm_report, lr_report]):
            vals  = [report["metrics_by_split"][s][metric] for s in splits]
            name  = report["model_name"]
            color = colors.get(name, "gray")
            bars  = ax.bar(x + m_i*width, vals, width,
                           label=name, color=color,
                           edgecolor='black', linewidth=0.5, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        val + 0.005, f'{val:.3f}',
                        ha='center', fontsize=8)

        ax.set_xticks(x + width/2)
        ax.set_xticklabels([s.upper() for s in splits])
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} — LGBM vs LogReg', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        # Highlight test (drift window)
        ax.axvspan(1.5, 2.5, alpha=0.08, color='red', label='drift window')

    plt.suptitle('Model Comparison Across Temporal Windows\n'
                 '(red zone = concept drift window)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = FIGURE_DIR / "24_model_comparison.png"
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    logger.info(f"  Saved: {path.name}")


# ── Entry point ────────────────────────────────────────────────────────────
def run_evaluation() -> dict:
    logger.info("=" * 70)
    logger.info("DriftSentinel — Model Evaluator")
    logger.info("=" * 70)

    train, val, test = _load_splits()
    feat_cols        = _get_features(train)
    lgbm_model       = _load_lgbm()
    lr_model, scaler = _load_logreg()

    # Predict functions
    def lgbm_predict(X):
        df_X = pd.DataFrame(X, columns=feat_cols)
        return lgbm_model.predict_proba(df_X)[:, 1]

    def lr_predict(X):
        X_sc = scaler.transform(X)
        return lr_model.predict_proba(X_sc)[:, 1]

    # Evaluate both
    lgbm_report = evaluate_model(
        "lgbm_v1", lgbm_predict, train, val, test, feat_cols
    )
    lr_report = evaluate_model(
        "logreg_v1", lr_predict, train, val, test, feat_cols
    )

    # Comparative plot
    _plot_model_comparison(lgbm_report, lr_report)

    # Save evaluation report
    report = {
        "lgbm"  : {
            "threshold"       : lgbm_report["threshold"],
            "metrics_by_split": lgbm_report["metrics_by_split"],
            "degradation"     : lgbm_report["degradation"],
        },
        "logreg": {
            "threshold"       : lr_report["threshold"],
            "metrics_by_split": lr_report["metrics_by_split"],
            "degradation"     : lr_report["degradation"],
        },
    }

    report_path = MODELS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved -> {report_path}")

    # Summary
    logger.info("=" * 70)
    logger.info("Evaluation Summary")
    logger.info("=" * 70)
    for model_key, rep in [("LGBM", lgbm_report), ("LogReg", lr_report)]:
        deg = rep["degradation"]
        logger.info(f"{model_key}:")
        logger.info(f"  Val  AUC : {rep['metrics_by_split']['val']['auc']}")
        logger.info(f"  Test AUC : {rep['metrics_by_split']['test']['auc']}")
        logger.info(f"  AUC Δ    : {deg['degradation']['auc']['delta']:+.4f}")
        logger.info(f"  Severity : {deg['severity']}")
    logger.info("=" * 70)
    logger.info("Next: src/drift/ — drift detection pipeline")
    logger.info("=" * 70)

    return report


if __name__ == "__main__":
    report = run_evaluation()
    print(f"\nLGBM  : Val={report['lgbm']['metrics_by_split']['val']['auc']} "
          f"Test={report['lgbm']['metrics_by_split']['test']['auc']} "
          f"Δ={report['lgbm']['degradation']['degradation']['auc']['delta']:+.4f}")
    print(f"LogReg: Val={report['logreg']['metrics_by_split']['val']['auc']} "
          f"Test={report['logreg']['metrics_by_split']['test']['auc']} "
          f"Δ={report['logreg']['degradation']['degradation']['auc']['delta']:+.4f}")