"""
DriftSentinel — Model Robustness Evaluator
Aggregates attack results into comprehensive robustness score.
Compares robustness across: clean data, drifted data, attacked data.

Robustness Score [0-1]:
    1.0 = perfectly robust (attacks have zero effect)
    0.0 = completely vulnerable

Components:
    ASR robustness    — resistance to prediction flipping
    AUC robustness    — performance stability under attack
    Proba robustness  — prediction confidence stability
    Drift robustness  — combined drift + attack resistance
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger
from src.uncertainty.calibration import IsotonicCalibrator
from src.adversarial.attacks import (
    FGSMAttack, RandomNoiseAttack, FeatureMaskAttack,
    evaluate_attack, _get_feature_ranges
)

logger = get_logger("robustness")

ROOT          = Path(__file__).resolve().parents[2]
TRAIN_DIR     = ROOT / "data"    / "train"
MODELS_DIR    = ROOT / "outputs" / "models"
FIGURE_DIR    = ROOT / "outputs" / "figure"
REPORTS_DIR   = ROOT / "outputs" / "log"
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_binary"


# ══════════════════════════════════════════════════════════════════════════
# Robustness score computation
# ══════════════════════════════════════════════════════════════════════════

def compute_robustness_score(attack_results: dict) -> dict:
    """
    Aggregate per-attack metrics into overall robustness score.

    Formula:
        asr_rob   = 1 - mean(ASR across attacks)
        auc_rob   = 1 - mean(|ΔAUC| across attacks)
        proba_rob = 1 - mean(|Δproba| across attacks)
        overall   = 0.40 * asr_rob + 0.40 * auc_rob + 0.20 * proba_rob
    """
    asrs       = [r["asr"]            for r in attack_results.values()]
    auc_deltas = [abs(r["auc_delta"]) for r in attack_results.values()]
    proba_deltas=[abs(r["proba_delta"])for r in attack_results.values()]

    asr_rob   = max(0.0, 1.0 - np.mean(asrs))
    auc_rob   = max(0.0, 1.0 - np.mean(auc_deltas) / 0.10)
    proba_rob = max(0.0, 1.0 - np.mean(proba_deltas) / 0.10)
    overall   = 0.40 * asr_rob + 0.40 * auc_rob + 0.20 * proba_rob

    tier = (
        "ROBUST"     if overall >= 0.80 else
        "MODERATE"   if overall >= 0.60 else
        "VULNERABLE" if overall >= 0.40 else
        "CRITICAL"
    )

    return {
        "asr_robustness"  : round(asr_rob,  4),
        "auc_robustness"  : round(auc_rob,  4),
        "proba_robustness": round(proba_rob, 4),
        "overall_score"   : round(overall,  4),
        "tier"            : tier,
        "mean_asr"        : round(np.mean(asrs),        4),
        "max_asr"         : round(max(asrs),             4),
        "mean_auc_delta"  : round(np.mean(auc_deltas),  4),
        "max_auc_delta"   : round(max(auc_deltas),      4),
    }


# ══════════════════════════════════════════════════════════════════════════
# Epsilon sensitivity analysis
# ══════════════════════════════════════════════════════════════════════════

def epsilon_sensitivity(
    X:         np.ndarray,
    y:         np.ndarray,
    predict_fn,
    feat_cols: list[str],
    bounds:    dict,
    threshold: float,
    epsilons:  list = None,
) -> pd.DataFrame:
    """
    How does robustness change with attack strength (epsilon)?
    Shows at what epsilon model becomes vulnerable.
    """
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

    p_orig  = predict_fn(pd.DataFrame(X, columns=feat_cols))
    records = []

    for eps in epsilons:
        attacker = RandomNoiseAttack(epsilon=eps)
        X_adv    = attacker.attack(X, predict_fn, feat_cols, bounds)
        p_adv    = predict_fn(pd.DataFrame(X_adv, columns=feat_cols))
        result   = evaluate_attack(y, p_orig, p_adv, f"RANDOM_e{eps}", threshold)

        records.append({
            "epsilon"       : eps,
            "asr"           : result["asr"],
            "auc_delta"     : result["auc_delta"],
            "f1_delta"      : result["f1_delta"],
            "proba_delta"   : result["proba_delta"],
            "n_flipped"     : result["n_flipped"],
        })
        logger.info(
            f"  ε={eps:.2f}  ASR={result['asr']:.4f}  "
            f"ΔAUC={result['auc_delta']:+.4f}  "
            f"ΔF1={result['f1_delta']:+.4f}"
        )

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════
# Per-feature sensitivity
# ══════════════════════════════════════════════════════════════════════════

def feature_sensitivity(
    X:         np.ndarray,
    y:         np.ndarray,
    predict_fn,
    feat_cols: list[str],
    bounds:    dict,
    threshold: float,
    top_n:     int = 15,
) -> pd.DataFrame:
    """
    Sensitivity of model to perturbation of individual features.
    Perturb one feature at a time, measure AUC change.
    """
    p_orig   = predict_fn(pd.DataFrame(X, columns=feat_cols))
    records  = []

    from src.adversarial.attacks import PROTECTED_FEATURES

    for i, col in enumerate(feat_cols):
        if col in PROTECTED_FEATURES:
            continue

        std    = bounds[col]["std"]
        X_adv  = X.copy().astype(float)

        # Perturb single feature by 1 std
        X_adv[:, i] += std
        X_adv[:, i]  = np.clip(
            X_adv[:, i],
            bounds[col]["min"],
            bounds[col]["max"]
        )

        p_adv    = predict_fn(pd.DataFrame(X_adv, columns=feat_cols))
        auc_orig = float(np.abs(p_orig - y).mean())
        auc_adv  = float(np.abs(p_adv  - y).mean())

        pred_orig = (p_orig >= threshold).astype(int)
        pred_adv  = (p_adv  >= threshold).astype(int)
        flip_rate = float((pred_orig != pred_adv).mean())

        records.append({
            "feature"       : col,
            "is_fe"         : col.startswith("FE_"),
            "flip_rate"     : round(flip_rate, 4),
            "proba_delta"   : round(float((p_adv - p_orig).mean()), 4),
            "proba_std"     : round(float((p_adv - p_orig).std()),  4),
            "error_delta"   : round(auc_adv - auc_orig, 4),
            "feature_std"   : round(std, 4),
        })

    df = pd.DataFrame(records).sort_values(
        "flip_rate", ascending=False
    ).reset_index(drop=True)

    logger.info(f"\n  Top {top_n} sensitive features:")
    logger.info(
        f"  {'Feature':<45} {'FlipRate':>9} {'Δproba':>8} {'Δerror':>8}"
    )
    logger.info("  " + "-" * 73)
    for _, row in df.head(top_n).iterrows():
        logger.info(
            f"  {row['feature']:<45} "
            f"{row['flip_rate']:>9.4f} "
            f"{row['proba_delta']:>+8.4f} "
            f"{row['error_delta']:>+8.4f}"
        )

    return df


# ══════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════

def _plot_robustness_dashboard(
    attack_results:   dict,
    robustness_score: dict,
    epsilon_df:       pd.DataFrame,
    sensitivity_df:   pd.DataFrame,
    model_name:       str,
):
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Robustness score gauge ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    score = robustness_score["overall_score"]
    tier  = robustness_score["tier"]
    color = ("#27ae60" if score >= 0.80 else
             "#f39c12" if score >= 0.60 else
             "#e74c3c")

    ax1.bar(["Robustness\nScore"], [score],
            color=color, edgecolor="black", linewidth=0.8, width=0.4)
    ax1.set_ylim(0, 1)
    ax1.axhline(0.80, color="#27ae60", linestyle="--",
                linewidth=1, alpha=0.7, label="Robust (0.80)")
    ax1.axhline(0.60, color="#f39c12", linestyle="--",
                linewidth=1, alpha=0.7, label="Moderate (0.60)")
    ax1.text(0, score + 0.03, f"{score:.3f}\n{tier}",
             ha="center", fontsize=11, fontweight="bold", color=color)
    ax1.set_title("Overall Robustness Score", fontweight="bold")
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # ── 2. Component scores ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    components = {
        "ASR\nRobustness"  : robustness_score["asr_robustness"],
        "AUC\nRobustness"  : robustness_score["auc_robustness"],
        "Proba\nRobustness": robustness_score["proba_robustness"],
    }
    colors_comp = ["#3498db", "#2ecc71", "#9b59b6"]
    bars = ax2.bar(
        components.keys(), components.values(),
        color=colors_comp, edgecolor="black", linewidth=0.6
    )
    for bar, val in zip(bars, components.values()):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 val + 0.02, f"{val:.3f}",
                 ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Robustness Components", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # ── 3. ASR per attack ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    names  = list(attack_results.keys())
    asrs   = [attack_results[n]["asr"] for n in names]
    colors_att = ["#e74c3c" if a > 0.02 else "#3498db" for a in asrs]
    ax3.barh(names, asrs, color=colors_att,
             edgecolor="black", linewidth=0.5)
    ax3.axvline(0.02, color="orange", linestyle="--",
                linewidth=1.5, label="Concern threshold (2%)")
    ax3.set_xlabel("Attack Success Rate (ASR)")
    ax3.set_title("ASR per Attack Method", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(axis="x", alpha=0.3)
    for i, (name, asr) in enumerate(zip(names, asrs)):
        ax3.text(asr + 0.001, i, f"{asr:.3f}",
                 va="center", fontsize=8)

    # ── 4. AUC delta per attack ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    auc_deltas = [attack_results[n]["auc_delta"] for n in names]
    colors_auc = ["#e74c3c" if d < -0.01 else "#2ecc71" for d in auc_deltas]
    bars = ax4.bar(names, auc_deltas,
                   color=colors_auc, edgecolor="black", linewidth=0.5)
    ax4.axhline(0, color="black", linewidth=1)
    ax4.axhline(-0.05, color="red", linestyle="--",
                linewidth=1, alpha=0.7, label="Critical (-0.05)")
    ax4.set_ylabel("ΔAUC")
    ax4.set_title("AUC Change per Attack", fontweight="bold")
    ax4.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    # ── 5. Epsilon sensitivity ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epsilon_df["epsilon"], epsilon_df["asr"],
             marker="o", color="#e74c3c", linewidth=2, label="ASR")
    ax5_r = ax5.twinx()
    ax5_r.plot(epsilon_df["epsilon"], epsilon_df["auc_delta"].abs(),
               marker="s", color="#3498db", linewidth=2,
               linestyle="--", label="|ΔAUC|")
    ax5.set_xlabel("Epsilon (attack strength)")
    ax5.set_ylabel("ASR", color="#e74c3c")
    ax5_r.set_ylabel("|ΔAUC|", color="#3498db")
    ax5.set_title("Sensitivity to Attack Strength", fontweight="bold")
    ax5.grid(alpha=0.3)
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_r.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # ── 6. F1 delta per attack ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    f1_deltas = [attack_results[n]["f1_delta"] for n in names]
    colors_f1 = ["#e74c3c" if d < -0.01 else "#2ecc71" for d in f1_deltas]
    ax6.bar(names, f1_deltas, color=colors_f1,
            edgecolor="black", linewidth=0.5)
    ax6.axhline(0, color="black", linewidth=1)
    ax6.set_ylabel("ΔF1")
    ax6.set_title("F1 Change per Attack", fontweight="bold")
    ax6.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax6.grid(axis="y", alpha=0.3)

    # ── 7. Feature sensitivity (top 15) ───────────────────────────────────
    ax7 = fig.add_subplot(gs[2, :])
    top15 = sensitivity_df.head(15)
    colors_feat = [
        "#e74c3c" if fe else "#3498db"
        for fe in top15["is_fe"]
    ]
    bars = ax7.barh(
        top15["feature"][::-1],
        top15["flip_rate"][::-1],
        color=colors_feat[::-1],
        edgecolor="black", linewidth=0.4
    )
    ax7.set_xlabel("Flip Rate (fraction of predictions changed)")
    ax7.set_title(
        "Feature Sensitivity: Top 15 Most Sensitive Features\n"
        "(red=FE_ engineered, blue=raw | perturb by +1 std)",
        fontweight="bold"
    )
    ax7.axvline(0.02, color="orange", linestyle="--",
                linewidth=1.5, alpha=0.8, label="Concern (2%)")
    ax7.legend(fontsize=8)
    ax7.grid(axis="x", alpha=0.3)

    plt.suptitle(
        f"Robustness Dashboard — {model_name}\n"
        f"Overall Score: {score:.3f} ({tier})",
        fontsize=13, fontweight="bold"
    )
    path = FIGURE_DIR / f"30_robustness_dashboard_{model_name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    logger.info(f"  Saved: {path.name}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_robustness() -> dict:
    logger.info("=" * 70)
    logger.info("DriftSentinel — Robustness Evaluator")
    logger.info("=" * 70)

    # Load data
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")

    feat_cols = [c for c in test.columns
                 if c not in {"readmitted_binary", "readmitted_multi"}]

    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "calibrator_isotonic_lgbm_v1.pkl", "rb") as f:
        calibrator = pickle.load(f)

    eval_path = MODELS_DIR / "evaluation_report.json"
    with open(eval_path) as f:
        threshold = json.load(f)["lgbm"]["threshold"]

    X_test = test[feat_cols].values.astype(float)
    y_test = test[TARGET].values

    def predict_fn(X_df: pd.DataFrame) -> np.ndarray:
        p_raw = model.predict_proba(X_df)[:, 1]
        return calibrator.transform(p_raw)

    bounds     = _get_feature_ranges(X_test, feat_cols)
    importance = model.feature_importances_

    # ── Step 1: Load existing attack results ───────────────────────────────
    logger.info("\nStep 1: Loading attack results")
    attack_path = REPORTS_DIR / "attack_report_test.json"

    if attack_path.exists():
        with open(attack_path) as f:
            attack_results = json.load(f)
        logger.info(f"  Loaded {len(attack_results)} attack results")
    else:
        logger.warning("  Attack report not found — running quick attacks")
        p_orig = predict_fn(pd.DataFrame(X_test, columns=feat_cols))
        attack_results = {}
        for name, eps in [("FGSM", 0.1), ("RANDOM", 0.1), ("MASK_k5", None)]:
            if name == "MASK_k5":
                attacker = FeatureMaskAttack(k=5)
                X_adv = attacker.attack(
                    X_test, predict_fn, feat_cols, bounds,
                    feature_importance=importance
                )
            else:
                attacker = RandomNoiseAttack(epsilon=eps)
                X_adv = attacker.attack(X_test, predict_fn, feat_cols, bounds)
            p_adv = predict_fn(pd.DataFrame(X_adv, columns=feat_cols))
            attack_results[name] = evaluate_attack(
                y_test, p_orig, p_adv, name, threshold
            )

    # ── Step 2: Robustness score ───────────────────────────────────────────
    logger.info("\nStep 2: Computing robustness score")
    robustness_score = compute_robustness_score(attack_results)

    logger.info(f"  ASR robustness    : {robustness_score['asr_robustness']:.4f}")
    logger.info(f"  AUC robustness    : {robustness_score['auc_robustness']:.4f}")
    logger.info(f"  Proba robustness  : {robustness_score['proba_robustness']:.4f}")
    logger.info(f"  Overall score     : {robustness_score['overall_score']:.4f}")
    logger.info(f"  Tier              : {robustness_score['tier']}")

    # ── Step 3: Epsilon sensitivity ────────────────────────────────────────
    logger.info("\nStep 3: Epsilon sensitivity analysis")
    epsilon_df = epsilon_sensitivity(
        X_test[:3000], y_test[:3000],
        predict_fn, feat_cols, bounds, threshold,
        epsilons=[0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
    )

    # ── Step 4: Feature sensitivity ────────────────────────────────────────
    logger.info("\nStep 4: Feature sensitivity analysis")
    sensitivity_df = feature_sensitivity(
        X_test[:3000], y_test[:3000],
        predict_fn, feat_cols, bounds, threshold
    )

    # ── Step 5: Plot ───────────────────────────────────────────────────────
    logger.info("\nStep 5: Generating robustness dashboard")
    _plot_robustness_dashboard(
        attack_results, robustness_score,
        epsilon_df, sensitivity_df, "lgbm_v1"
    )

    # ── Save ───────────────────────────────────────────────────────────────
    epsilon_df.to_csv(
        REPORTS_DIR / "epsilon_sensitivity_lgbm_v1.csv", index=False
    )
    sensitivity_df.to_csv(
        REPORTS_DIR / "feature_sensitivity_lgbm_v1.csv", index=False
    )

    report = {
        "model_name"      : "lgbm_v1",
        "robustness_score": robustness_score,
        "epsilon_sensitivity": epsilon_df.to_dict("records"),
        "top10_sensitive_features": sensitivity_df.head(10)[
            ["feature", "flip_rate", "proba_delta"]
        ].to_dict("records"),
    }

    report_path = REPORTS_DIR / "robustness_report_lgbm_v1.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Robustness report: {report_path}")

    logger.info("=" * 70)
    logger.info("Robustness Evaluation Complete")
    logger.info(f"  Overall score : {robustness_score['overall_score']:.4f}")
    logger.info(f"  Tier          : {robustness_score['tier']}")
    logger.info("  Next: defense.py")
    logger.info("=" * 70)

    print(f"\nRobustness Score : {robustness_score['overall_score']:.4f}")
    print(f"Tier             : {robustness_score['tier']}")
    print(f"Mean ASR         : {robustness_score['mean_asr']:.4f}")
    print(f"Max AUC drop     : {robustness_score['max_auc_delta']:.4f}")

    return report


if __name__ == "__main__":
    import json
    run_robustness()