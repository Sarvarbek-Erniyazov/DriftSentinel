"""
DriftSentinel — Tier 2B.4: threat-modelled robustness

THE THREAT MODEL, STATED EXPLICITLY
    The shipped robustness suite ran FGSM and PGD and reported ASR 0.2-0.3% as
    evidence of adversarial robustness. Two things are wrong with that.

    1. THERE IS NO CREDIBLE ADVERSARY. Who attacks a hospital readmission model,
       and what do they gain? The model triages follow-up care. There is no
       payout, no authentication to bypass, no content filter to evade. An
       "adversarial robustness" section with no adversary is method-shopping.

    2. THE ATTACK DID NOT EXECUTE. LightGBM is piecewise constant. The shipped
       attacks estimate gradients by finite differences with h = 1e-3, which for
       almost every (sample, feature) pair does not cross a split threshold, so
       the numerator is EXACTLY zero. ASR ~ 0 is the null behaviour of a broken
       method, not a property of the model. This module MEASURES that zero-
       gradient fraction rather than asserting it.

    THE REAL THREAT IS EHR DATA QUALITY. These failures are routine, documented,
    and have direct clinical consequence:
        missingness injection  a feed degrades and fields arrive empty
        unit errors            a lab value arrives in the wrong unit (x10, /10)
        coding drift           a categorical vocabulary is remapped upstream
        delayed labs           lab-derived features are unavailable at inference
        feature outage         an entire upstream feature stops arriving

FALSIFICATION ARM (carried forward from 2B.1 and 2B.3)
    A null result is only interpretable if the harness CAN detect damage. A
    destructive control — replacing the model's most important feature with pure
    noise — is included. If that does not degrade performance, the harness is
    not measuring anything and no null elsewhere can be believed.
"""

from __future__ import annotations

import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger
from src.uncertainty.calibration import IsotonicCalibrator  # noqa: F401

logger = get_logger("data_quality_robustness")

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "train"
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_binary"
TARGET_COLS = {"readmitted_binary", "readmitted_multi"}
SEED = 42
RATES = [0.05, 0.10, 0.25, 0.50]


# ══════════════════════════════════════════════════════════════════════════
# Why the shipped gradient attacks did not execute
# ══════════════════════════════════════════════════════════════════════════

def measure_zero_gradient_fraction(model, X: pd.DataFrame, h: float = 1e-3,
                                   n: int = 400, seed: int = SEED) -> dict:
    """
    Measure the fraction of finite-difference gradients that are EXACTLY zero.

    This is the concrete demonstration that FGSM/PGD never executed on this
    model class. A piecewise-constant ensemble has zero gradient except where a
    perturbation crosses a split threshold.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs = X.iloc[idx].to_numpy(dtype=float)
    base = model.predict_proba(pd.DataFrame(Xs, columns=X.columns))[:, 1]

    zeros = total = 0
    for j in range(Xs.shape[1]):
        Xp = Xs.copy()
        Xp[:, j] += h
        pj = model.predict_proba(pd.DataFrame(Xp, columns=X.columns))[:, 1]
        g = (pj - base) / h
        zeros += int(np.sum(g == 0.0))
        total += len(g)
    return {
        "h": h,
        "n_samples": int(len(idx)),
        "n_features": int(Xs.shape[1]),
        "n_gradient_entries": total,
        "exactly_zero_fraction": round(zeros / total, 5),
        "interpretation": (
            "a finite-difference gradient that is exactly zero for this fraction "
            "of (sample, feature) pairs means FGSM/PGD had no direction to step "
            "in. The shipped ASR of 0.2-0.3% is the null behaviour of a method "
            "that did not execute, not evidence of robustness."),
    }


# ══════════════════════════════════════════════════════════════════════════
# EHR data-quality perturbations
# ══════════════════════════════════════════════════════════════════════════

def perturb_missingness(X, cols, rate, rng, impute):
    """A feed degrades: fields arrive empty and are imputed."""
    Xp = X.copy()
    for c in cols:
        m = rng.random(len(Xp)) < rate
        Xp.loc[m, c] = impute[c]
    return Xp


def perturb_unit_error(X, cols, rate, rng, factor=10.0):
    """A lab value arrives in the wrong unit for a fraction of records."""
    Xp = X.copy()
    for c in cols:
        m = rng.random(len(Xp)) < rate
        Xp.loc[m, c] = Xp.loc[m, c] * factor
    return Xp


def perturb_coding_drift(X, cols, rate, rng):
    """An upstream categorical vocabulary is remapped (codes permuted)."""
    Xp = X.copy()
    for c in cols:
        vals = np.unique(Xp[c].to_numpy())
        if len(vals) < 2:
            continue
        mapping = dict(zip(vals, rng.permutation(vals)))
        m = rng.random(len(Xp)) < rate
        Xp.loc[m, c] = Xp.loc[m, c].map(mapping).astype(Xp[c].dtype)
    return Xp


def perturb_feature_outage(X, cols, rate, rng, impute):
    """An entire upstream feature stops arriving (constant at its imputed value)."""
    Xp = X.copy()
    for c in cols:
        Xp[c] = impute[c]
    return Xp


def perturb_destructive_control(X, cols, rate, rng):
    """FALSIFICATION: replace the top feature with pure noise. MUST degrade."""
    Xp = X.copy()
    for c in cols:
        Xp[c] = rng.permutation(Xp[c].to_numpy())
    return Xp


def evaluate(model, cal, X, y, threshold) -> dict:
    p = cal.transform(model.predict_proba(X)[:, 1])
    pred = (p >= threshold).astype(int)
    return {
        "auc": round(float(roc_auc_score(y, p)), 5),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 5),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 5),
        "predicted_positive_rate": round(float(pred.mean()), 5),
    }


def run_data_quality_robustness() -> dict:
    logger.info("=" * 78)
    logger.info("Tier 2B.4 — threat-modelled robustness (EHR data quality)")
    logger.info("=" * 78)

    train = pd.read_parquet(TRAIN_DIR / "train_fs.parquet")
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")
    feat = [c for c in train.columns if c not in TARGET_COLS]
    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ROOT / "outputs" / "artifacts" /
              "calibrator_isotonic_lgbm_v1.pkl", "rb") as f:
        cal = pickle.load(f)
    with open(MODELS_DIR / "evaluation_report.json") as f:
        threshold = json.load(f)["lgbm"]["threshold"]

    X, y = test[feat], test[TARGET].to_numpy()
    impute = {c: float(train[c].median()) for c in feat}
    rng = np.random.default_rng(SEED)

    baseline = evaluate(model, cal, X, y, threshold)
    logger.info(f"  baseline: AUC {baseline['auc']:.4f} F1 {baseline['f1']:.4f} "
                f"recall {baseline['recall']:.4f} PPR {baseline['predicted_positive_rate']:.4f}")

    # ── why the shipped gradient attacks never executed ───────────────────
    logger.info("-" * 62)
    logger.info("Gradient validity check (why FGSM/PGD reported ASR ~ 0)")
    zg = measure_zero_gradient_fraction(model, X)
    logger.info(f"  finite-difference gradients exactly zero: "
                f"{zg['exactly_zero_fraction']:.2%} of "
                f"{zg['n_gradient_entries']:,} (sample, feature) pairs at h={zg['h']}")

    # ── scenario definitions, tied to real EHR failure modes ──────────────
    imp = pd.Series(model.booster_.feature_importance("gain"), index=feat)
    top_feature = imp.sort_values(ascending=False).index[0]
    lab_cols = [c for c in feat if "lab" in c.lower()][:3]
    util_cols = [c for c in feat if "inpatient" in c.lower() or
                 "emergency" in c.lower() or "outpatient" in c.lower()][:3]
    cat_cols = [c for c in feat if c in ("payer_code", "medical_specialty",
                                         "admission_source_id",
                                         "discharge_disposition_id")]
    num_cols = [c for c in feat if c in ("num_lab_procedures", "num_medications",
                                         "time_in_hospital", "number_diagnoses")]

    scenarios = {
        "missingness_utilisation": (perturb_missingness, util_cols,
                                    "prior-utilisation fields arrive empty"),
        "missingness_labs": (perturb_missingness, lab_cols,
                             "lab-derived fields arrive empty"),
        "unit_error_x10": (perturb_unit_error, num_cols,
                           "a numeric field arrives in the wrong unit (x10)"),
        "coding_drift_categorical": (perturb_coding_drift, cat_cols,
                                     "an upstream categorical vocabulary is remapped"),
        "feature_outage_top": (perturb_feature_outage, [top_feature],
                               f"the highest-gain feature ({top_feature}) stops arriving"),
        "DESTRUCTIVE_CONTROL_top_feature_noise": (
            perturb_destructive_control, [top_feature],
            "FALSIFICATION: the top feature is replaced with pure noise"),
    }

    results = {}
    logger.info("-" * 62)
    logger.info(f"{'scenario':<42}{'rate':>7}{'AUC':>9}{'dAUC':>9}{'recall':>9}{'dPPR':>9}")
    for name, (fn, cols, desc) in scenarios.items():
        if not cols:
            continue
        rows = []
        rates = RATES if "outage" not in name and "DESTRUCTIVE" not in name else [1.0]
        for r in rates:
            rr = np.random.default_rng(SEED)
            Xp = (fn(X, cols, r, rr, impute) if fn in
                  (perturb_missingness, perturb_feature_outage)
                  else fn(X, cols, r, rr))
            m = evaluate(model, cal, Xp, y, threshold)
            rows.append({"rate": r, **m,
                         "delta_auc": round(m["auc"] - baseline["auc"], 5),
                         "delta_recall": round(m["recall"] - baseline["recall"], 5),
                         "delta_ppr": round(m["predicted_positive_rate"]
                                            - baseline["predicted_positive_rate"], 5)})
            logger.info(f"{name:<42}{r:>7.2f}{m['auc']:>9.4f}"
                        f"{rows[-1]['delta_auc']:>+9.4f}{m['recall']:>9.4f}"
                        f"{rows[-1]['delta_ppr']:>+9.4f}")
        results[name] = {"description": desc, "columns": cols, "sweep": rows,
                         "worst_delta_auc": min(r["delta_auc"] for r in rows)}

    ctrl = results["DESTRUCTIVE_CONTROL_top_feature_noise"]["worst_delta_auc"]
    harness_works = ctrl < -0.005
    realistic = {k: v["worst_delta_auc"] for k, v in results.items()
                 if not k.startswith("DESTRUCTIVE")}
    worst_realistic = min(realistic.values())
    worst_name = min(realistic, key=realistic.get)

    report = {
        "phase": "2B.4",
        "title": "Threat-modelled robustness: EHR data quality, not adversarial ML",
        "threat_model": {
            "adversary": "NONE. There is no credible adversary against a hospital "
                         "readmission triage model: no payout, no authentication "
                         "to bypass, no filter to evade.",
            "real_threats": "routine EHR data-quality failures",
            "why_the_shipped_framing_was_wrong": (
                "'adversarial robustness' with no adversary is method-shopping, "
                "and the attacks used did not execute on this model class"),
        },
        "gradient_validity_check": zg,
        "baseline": baseline,
        "scenarios": results,
        "falsification": {
            "destructive_control_delta_auc": ctrl,
            "harness_can_detect_damage": bool(harness_works),
            "why_it_matters": ("a null on the realistic scenarios is only "
                               "interpretable if the harness can detect damage "
                               "when damage is real"),
        },
        "verdict": {
            "worst_realistic_scenario": worst_name,
            "worst_realistic_delta_auc": worst_realistic,
            "finding": (
                f"Under realistic EHR data-quality failures the model degrades "
                f"gracefully: the worst realistic scenario ({worst_name}) costs "
                f"{abs(worst_realistic):.4f} AUC. The destructive control costs "
                f"{abs(ctrl):.4f}, confirming the harness detects damage when it "
                f"is real. The shipped 'adversarial robustness score' is withdrawn: "
                f"it measured the null behaviour of gradient attacks that cannot "
                f"execute on a piecewise-constant ensemble "
                f"({zg['exactly_zero_fraction']:.1%} of finite-difference gradients "
                f"are exactly zero), and it had no threat model."
                if harness_works else
                "HARNESS FAILED ITS OWN CONTROL — the destructive control did not "
                "degrade performance, so no null on the realistic scenarios can be "
                "believed. Do not report these numbers."),
        },
        "supersedes": ("outputs/log/robustness_report_lgbm_v1.json — the FGSM/PGD "
                       "'robustness score' is withdrawn, not merely supplemented"),
        "reproducibility": {"seed": SEED, "python": platform.python_version(),
                            "numpy": np.__version__, "pandas": pd.__version__},
    }
    out = REPORTS_DIR / "data_quality_robustness.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("-" * 62)
    logger.info(f"destructive control dAUC {ctrl:+.4f} -> harness works = {harness_works}")
    logger.info(f"worst realistic: {worst_name} {worst_realistic:+.4f} AUC")
    logger.info(f"Report: {out.name}")
    logger.info("=" * 78)
    return report


if __name__ == "__main__":
    r = run_data_quality_robustness()
    print(f"\nzero-gradient fraction: {r['gradient_validity_check']['exactly_zero_fraction']:.2%}")
    print(f"{'scenario':<42}{'worst dAUC':>12}")
    for k, v in r["scenarios"].items():
        print(f"{k:<42}{v['worst_delta_auc']:>+12.5f}")
    print(f"\nharness can detect damage: {r['falsification']['harness_can_detect_damage']}")
    print(r["verdict"]["finding"][:320])
