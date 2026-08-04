"""
DriftSentinel — Tier 1.3: operating-point policy

WHY THIS EXISTS
    The shipped "cost-optimal" threshold 0.1282 was reported as a headline win
    (F1 0.5126, recall 99.5%, 29 missed readmissions). It flags **97% of all
    patients**. A clinical decision-support tool that alerts on 97% of
    admissions supplies no triage information and is switched off within a week
    (audit F5).

WAS IT STRUCTURAL, LIKE THE LAYER-1 IQR COLLAPSE?  No — and that was worth
testing rather than assuming.

    The shipped cost function normalises each error type by ITS OWN class size:

        cost = FN/n_pos * c_fn + FP/n_neg * c_fp          (rate form)

    which divides prevalence out and silently multiplies the FN:FP ratio by the
    inverse odds of the positive class. That IS a defect, and it is fixed here.
    But at the merged target's 47.6% prevalence the inverse odds are only 1.10,
    so 5:1 becomes an effective 5.5:1 and the threshold barely moves
    (0.1282 rate form vs 0.1627 population form; PPR 0.970 vs 0.954).

    The real cause is simpler and worse: **the asserted 5:1 cost ratio is doing
    all the work.** For a calibrated model the optimal threshold is
    c_fp/(c_fp+c_fn) = 1/6 = 0.167, and 95.4% of predicted probabilities exceed
    it because the model's outputs concentrate near the 47.6% base rate (p5 =
    0.20, p25 = 0.36). The threshold is CORRECT for 5:1. The alert volume is
    what an optimal 5:1 rule does on this probability distribution — so the
    conclusion rests entirely on a cost ratio asserted with no citation.

    The rate-form defect becomes serious under the NEW primary target: at 11.16%
    prevalence for `<30`, it inflates 5:1 into an effective 39.8:1.

CONTENTS
    predicted_positive_rate    PPR, reported with every recommendation (R3)
    operating_point            full metric set + DEGENERATE flag
    threshold_cost_sensitive   population-weighted expected cost (corrected)
    threshold_under_budget     maximise utility subject to PPR <= B
    cost_ratio_sweep           1:1 -> 100:1, with the feasible region
    decision_curve_analysis    Vickers & Elkin (2006) net benefit
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
# Imported at MODULE level, not inside the runner: the shipped calibrator was
# pickled from a __main__ script, so unpickling resolves `IsotonicCalibrator`
# against whichever module is __main__. A function-local import is not an
# attribute of that module and the load fails. (Another symptom of pickled
# model artifacts — see audit "pickle for model artifacts (fragile)".)
from src.uncertainty.calibration import IsotonicCalibrator  # noqa: F401

DEGENERATE_PPR = 0.60          # audit F5: auto-flag anything above this
DEFAULT_BUDGETS = (0.10, 0.20, 0.30)


def predicted_positive_rate(y_proba: np.ndarray, threshold: float) -> float:
    """Share of the population the rule would flag. The number the audit found missing."""
    return float((np.asarray(y_proba) >= threshold).mean())


def operating_point(y_true: np.ndarray, y_proba: np.ndarray, threshold: float,
                    cost_fn: float = 5.0, cost_fp: float = 1.0) -> dict:
    """
    Every metric for one threshold, PPR included, with a degeneracy flag.

    R3: no "optimal" threshold is reported without its predicted-positive-rate.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba, dtype=float)
    pred = (p >= threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n = len(y)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ppr = (tp + fp) / n

    return {
        "threshold": float(threshold),
        "predicted_positive_rate": round(ppr, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "expected_cost_per_patient": round((fn * cost_fn + fp * cost_fp) / n, 5),
        "missed_positives": fn,
        "alerts_per_1000_patients": int(round(ppr * 1000)),
        "degenerate": bool(ppr > DEGENERATE_PPR),
        "degenerate_reason": (
            f"flags {ppr:.1%} of all patients (> {DEGENERATE_PPR:.0%}); provides no "
            f"triage information and would be disabled for alert fatigue"
            if ppr > DEGENERATE_PPR else None),
    }


def threshold_cost_sensitive(y_true: np.ndarray, y_proba: np.ndarray,
                             cost_fn: float = 5.0, cost_fp: float = 1.0,
                             n_grid: int = 200, legacy_rate_form: bool = False) -> dict:
    """
    Minimise expected cost per patient.

    CORRECTED to the population-weighted form
        (FN * c_fn + FP * c_fp) / N
    rather than the shipped per-class-rate form
        FN/n_pos * c_fn + FP/n_neg * c_fp
    which divides prevalence out and inflates the effective ratio by the inverse
    odds of the positive class. `legacy_rate_form=True` reproduces the shipped
    behaviour so the difference can be quantified rather than asserted.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba, dtype=float)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())

    best_thr, best_cost = 0.5, float("inf")
    for t in np.linspace(0.01, 0.99, n_grid):
        pred = (p >= t).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        if legacy_rate_form:
            cost = fn / (n_pos + 1e-8) * cost_fn + fp / (n_neg + 1e-8) * cost_fp
        else:
            cost = (fn * cost_fn + fp * cost_fp) / len(y)
        if cost < best_cost:
            best_thr, best_cost = float(t), float(cost)

    prevalence = n_pos / len(y)
    out = operating_point(y, p, best_thr, cost_fn, cost_fp)
    out.update({
        "objective": "rate-form (legacy, defective)" if legacy_rate_form
                     else "population-weighted expected cost",
        "cost_ratio_requested": cost_fn / cost_fp,
        "prevalence": round(prevalence, 4),
        "effective_cost_ratio": round(
            (cost_fn / cost_fp) * (1 - prevalence) / prevalence, 3) if legacy_rate_form
            else round(cost_fn / cost_fp, 3),
        "theoretical_optimal_threshold_if_calibrated": round(
            cost_fp / (cost_fp + cost_fn), 4),
    })
    return out


def threshold_under_budget(y_true: np.ndarray, y_proba: np.ndarray,
                           budget: float, cost_fn: float = 5.0,
                           cost_fp: float = 1.0, n_grid: int = 400) -> dict:
    """
    Maximise utility subject to an ALERT BUDGET: flag at most `budget` of patients.

    This is the constraint the audit asked for. Without it the optimiser is free
    to return a degenerate rule, and it does.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba, dtype=float)

    feasible = []
    for t in np.linspace(0.001, 0.999, n_grid):
        if predicted_positive_rate(p, t) <= budget:
            feasible.append(t)
    if not feasible:
        return {"budget": budget, "feasible": False,
                "reason": "no threshold satisfies this budget on this distribution"}

    best = min(feasible,
               key=lambda t: operating_point(y, p, t, cost_fn, cost_fp)["expected_cost_per_patient"])
    out = operating_point(y, p, best, cost_fn, cost_fp)
    out.update({"budget": budget, "feasible": True,
                "constraint": f"predicted_positive_rate <= {budget}"})
    return out


def cost_ratio_sweep(y_true: np.ndarray, y_proba: np.ndarray,
                     ratios=(1, 2, 3, 5, 10, 20, 50, 100)) -> dict:
    """
    Sweep FN:FP over two orders of magnitude and report the FEASIBLE region.

    No conclusion may rest on the single asserted 5:1, which carries no citation
    and no sensitivity analysis in the original.
    """
    rows = []
    for r in ratios:
        res = threshold_cost_sensitive(y_true, y_proba, cost_fn=float(r), cost_fp=1.0)
        rows.append({"cost_ratio": r, "threshold": res["threshold"],
                     "predicted_positive_rate": res["predicted_positive_rate"],
                     "precision": res["precision"], "recall": res["recall"],
                     "f1": res["f1"], "degenerate": res["degenerate"]})
    feasible = [r for r in rows if not r["degenerate"]]
    return {
        "sweep": rows,
        "feasible_ratios": [r["cost_ratio"] for r in feasible],
        "degenerate_ratios": [r["cost_ratio"] for r in rows if r["degenerate"]],
        "max_feasible_ratio": max([r["cost_ratio"] for r in feasible], default=None),
        "interpretation": (
            "ratios above max_feasible_ratio produce degenerate rules on this "
            "probability distribution: the 'optimal' threshold falls below most "
            "predicted probabilities, so nearly everyone is flagged"),
    }


def decision_curve_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                            thresholds=None) -> dict:
    """
    Decision curve analysis (Vickers & Elkin, Med Decis Making 2006).

    Net benefit at threshold probability pt:
        NB = TP/N - FP/N * pt/(1-pt)
    compared against treat-all and treat-none. The standard clinical framing for
    threshold utility, and the one health-informatics reviewers expect. A model
    is only useful where its curve sits ABOVE both default strategies.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba, dtype=float)
    n = len(y)
    prevalence = float(y.mean())
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.81, 0.02), 3)

    rows = []
    for pt in thresholds:
        pred = (p >= pt).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        w = pt / (1 - pt)
        nb_model = tp / n - (fp / n) * w
        nb_all = prevalence - (1 - prevalence) * w
        rows.append({
            "threshold_probability": float(pt),
            "net_benefit_model": round(float(nb_model), 6),
            "net_benefit_treat_all": round(float(nb_all), 6),
            "net_benefit_treat_none": 0.0,
            "model_beats_defaults": bool(nb_model > max(nb_all, 0.0)),
            "predicted_positive_rate": round(float(pred.mean()), 4),
        })

    useful = [r["threshold_probability"] for r in rows if r["model_beats_defaults"]]
    return {
        "reference": "Vickers & Elkin (2006), Decision Curve Analysis",
        "prevalence": round(prevalence, 4),
        "curve": rows,
        "useful_threshold_range": ([min(useful), max(useful)] if useful else None),
        "interpretation": (
            "the model adds value only where its net benefit exceeds BOTH "
            "treat-all and treat-none; outside that range a default strategy is "
            "at least as good and the model should not drive the decision"),
    }


# ══════════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════════

def run_threshold_policy(model_name: str = "lgbm_v1") -> dict:
    """
    Regenerate the operating-point report with PPR, budget constraints, a cost
    ratio sweep and decision curve analysis.
    """
    import json
    import pickle
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    import sys
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT))
    from src.monitoring.logger import get_logger
    from src.uncertainty.calibration import IsotonicCalibrator  # noqa: F401 (unpickling)

    logger = get_logger("threshold_policy")
    REPORTS = ROOT / "outputs" / "reports"
    FIGDIR = ROOT / "outputs" / "figure"
    REPORTS.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DriftSentinel - Operating-point policy (Tier 1.3)")
    logger.info("=" * 70)

    val = pd.read_parquet(ROOT / "data" / "train" / "val_fs.parquet")
    test = pd.read_parquet(ROOT / "data" / "train" / "test_fs.parquet")
    fc = [c for c in val.columns if c not in {"readmitted_binary", "readmitted_multi"}]
    with open(ROOT / "outputs" / "models" / f"{model_name}.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ROOT / "outputs" / "artifacts" / f"calibrator_isotonic_{model_name}.pkl", "rb") as f:
        cal = pickle.load(f)

    def proba(df):
        return cal.transform(model.predict_proba(df[fc])[:, 1])

    y_val, p_val = val["readmitted_binary"].values, proba(val)
    y_test, p_test = test["readmitted_binary"].values, proba(test)

    # ── the shipped recommendation, now with its PPR ──────────────────────
    legacy = threshold_cost_sensitive(y_val, p_val, legacy_rate_form=True)
    fixed = threshold_cost_sensitive(y_val, p_val)
    logger.info(f"  shipped (rate form)      thr={legacy['threshold']:.4f} "
                f"PPR={legacy['predicted_positive_rate']:.4f} "
                f"degenerate={legacy['degenerate']}")
    logger.info(f"  corrected (population)   thr={fixed['threshold']:.4f} "
                f"PPR={fixed['predicted_positive_rate']:.4f} "
                f"degenerate={fixed['degenerate']}")

    # ── budget-constrained operating points ───────────────────────────────
    budgets = {}
    logger.info("  Alert-budget constrained operating points (val):")
    for b in DEFAULT_BUDGETS:
        r = threshold_under_budget(y_val, p_val, b)
        budgets[f"budget_{b:.2f}"] = r
        if r.get("feasible"):
            logger.info(f"    PPR<={b:.0%}: thr={r['threshold']:.4f} "
                        f"precision={r['precision']:.4f} recall={r['recall']:.4f} "
                        f"F1={r['f1']:.4f} missed={r['missed_positives']}")

    # ── cost-ratio sensitivity ────────────────────────────────────────────
    sweep = cost_ratio_sweep(y_val, p_val)
    logger.info(f"  Cost-ratio sweep: feasible {sweep['feasible_ratios']} | "
                f"degenerate {sweep['degenerate_ratios']}")

    # ── decision curve analysis ───────────────────────────────────────────
    dca = decision_curve_analysis(y_val, p_val)
    logger.info(f"  DCA: model beats both defaults over "
                f"{dca['useful_threshold_range']}")

    # ── figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)

    ax = axes[0]
    thrs = np.linspace(0.02, 0.95, 120)
    pprs = [predicted_positive_rate(p_val, t) for t in thrs]
    ax.plot(thrs, pprs, color="#2b6cb0", lw=1.8)
    ax.axhline(DEGENERATE_PPR, color="#9b2c2c", ls="--", lw=1.3,
               label=f"degenerate above {DEGENERATE_PPR:.0%}")
    ax.axvline(legacy["threshold"], color="#c05621", ls=":", lw=1.6,
               label=f"shipped 'cost-optimal' {legacy['threshold']:.3f}")
    for b in DEFAULT_BUDGETS:
        r = budgets[f"budget_{b:.2f}"]
        if r.get("feasible"):
            ax.plot([r["threshold"]], [r["predicted_positive_rate"]], "o",
                    color="#2f855a", ms=6)
    ax.set_xlabel("threshold"); ax.set_ylabel("predicted-positive rate")
    ax.set_title("Alert volume vs threshold\n(green = budget-constrained points)",
                 fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    rs = [r["cost_ratio"] for r in sweep["sweep"]]
    ax.plot(rs, [r["predicted_positive_rate"] for r in sweep["sweep"]],
            "o-", color="#2b6cb0", lw=1.6)
    ax.axhline(DEGENERATE_PPR, color="#9b2c2c", ls="--", lw=1.3)
    ax.set_xscale("log"); ax.set_xlabel("FN:FP cost ratio (log)")
    ax.set_ylabel("predicted-positive rate")
    ax.set_title("Cost-ratio sensitivity\nthe 5:1 assumption was never tested",
                 fontsize=9)
    ax.axvline(5, color="#c05621", ls=":", lw=1.6)

    ax = axes[2]
    pts = [r["threshold_probability"] for r in dca["curve"]]
    ax.plot(pts, [r["net_benefit_model"] for r in dca["curve"]],
            color="#2b6cb0", lw=1.8, label="model")
    ax.plot(pts, [r["net_benefit_treat_all"] for r in dca["curve"]],
            color="#718096", ls="--", lw=1.3, label="treat all")
    ax.axhline(0, color="#a0aec0", lw=1, label="treat none")
    ax.set_ylim(min(-0.05, min(r["net_benefit_model"] for r in dca["curve"])), None)
    ax.set_xlabel("threshold probability"); ax.set_ylabel("net benefit")
    ax.set_title("Decision curve analysis\n(Vickers & Elkin 2006)", fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle(f"Operating-point policy - {model_name}", fontweight="bold")
    figpath = FIGDIR / f"44_threshold_policy_{model_name}.png"
    fig.savefig(figpath, bbox_inches="tight")
    plt.close(fig)

    report = {
        "model_name": model_name,
        "tier": "1.3 - operating-point policy",
        "target_note": ("computed on the MERGED target still stored in the "
                        "feature parquets; re-run after the Tier 2A.1 switch to <30"),
        "shipped_recommendation_rate_form": legacy,
        "corrected_recommendation_population_form": fixed,
        "budget_constrained": budgets,
        "cost_ratio_sweep": sweep,
        "decision_curve_analysis": dca,
        "test_split_check": {
            "corrected": threshold_cost_sensitive(y_test, p_test),
            "budget_0.20": threshold_under_budget(y_test, p_test, 0.20),
        },
        "findings": {
            "not_structural": (
                "Tested the hypothesis that the 97% alert rate had a structural "
                "cause like the Layer-1 IQR collapse. It does not. The per-class "
                "rate normalisation IS a defect and is fixed, but at 47.6% "
                "prevalence it only inflates 5:1 to 5.5:1 and moves the threshold "
                "0.1282 -> 0.1627 (PPR 0.970 -> 0.954)."),
            "actual_cause": (
                "The asserted 5:1 cost ratio does all the work. For a calibrated "
                "model the optimum is c_fp/(c_fp+c_fn) = 0.167, and 95.4% of "
                "predicted probabilities exceed it because the model's outputs "
                "concentrate near the 47.6% base rate (p5=0.20, p25=0.36). The "
                "threshold is CORRECT for 5:1; the alert volume is what an "
                "optimal 5:1 rule does on this distribution."),
            "rate_form_becomes_serious_under_the_new_target": (
                "At the <30 prevalence of 11.16% the same rate-form defect "
                "inflates 5:1 into an effective 39.8:1. Fixed before the target "
                "switch rather than after."),
        },
        "figure": str(figpath.relative_to(ROOT).as_posix()),
    }
    out = REPORTS / f"threshold_policy_{model_name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"  Report: {out}")
    logger.info("=" * 70)
    return report


if __name__ == "__main__":
    r = run_threshold_policy()
    lg = r["shipped_recommendation_rate_form"]
    fx = r["corrected_recommendation_population_form"]
    print("\nOperating-point policy")
    print(f"  shipped   : thr {lg['threshold']:.4f}  PPR {lg['predicted_positive_rate']:.1%}"
          f"  degenerate={lg['degenerate']}")
    print(f"  corrected : thr {fx['threshold']:.4f}  PPR {fx['predicted_positive_rate']:.1%}"
          f"  degenerate={fx['degenerate']}")
    for k, v in r["budget_constrained"].items():
        if v.get("feasible"):
            print(f"  {k}: thr {v['threshold']:.4f}  precision {v['precision']:.3f}"
                  f"  recall {v['recall']:.3f}  missed {v['missed_positives']}")
    print(f"  feasible cost ratios: {r['cost_ratio_sweep']['feasible_ratios']}")
    print(f"  DCA useful range    : {r['decision_curve_analysis']['useful_threshold_range']}")
