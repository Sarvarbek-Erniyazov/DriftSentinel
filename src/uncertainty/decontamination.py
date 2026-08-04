"""
DriftSentinel — Tier 2A.4: decontaminating fit-and-evaluate-on-the-same-data

THREE CIRCULARITIES, all of the same shape: something was FITTED on `val` and
then EVALUATED on `val`, and the resulting number was reported as evidence.

  1. THRESHOLD (audit F13)
     The operating threshold is chosen by F1-max ON VAL, and `val` is then used
     as the drift REFERENCE WINDOW against which test degradation is measured.
     The reference window has a threshold fitted to itself; the production
     window does not. Part of the reported F1 collapse is threshold optimism,
     not drift.

  2. CONFORMAL COVERAGE (audit F14)
     The conformal predictor is calibrated on `val`, and val coverage is then
     reported next to test coverage to conclude "no drift". Calibration-set
     coverage equals the target BY CONSTRUCTION — it is what the algorithm
     solves for. Comparing a fitted quantity against a held-out one and
     concluding "stable" is circular.

  3. ISOTONIC CALIBRATION / ECE (found in Tier 2A.2)
     The isotonic calibrator is fitted on `val`, and Tier 2A.2 measured
     val ECE = 0.0000. That is not excellent calibration; it is the fit.

FIX
    `val` is split IN HALF BY PATIENT into a calibration half and an audit half.
    Everything that was fitted on val is refitted on the calibration half only,
    and every reported number is measured on the audit half. The in-sample
    number is retained alongside so the size of the optimism is visible rather
    than quietly removed.

    The threshold additionally gets a variant selected on a held-out slice of
    TRAIN, so the drift reference window carries no fitted quantity at all.

R3: every coverage and calibration number below is labelled in-sample or
held-out. R6: the check is the honest measurement itself, not the presence of
a decontamination step.
"""

from __future__ import annotations

import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger
from src.models.repeated_eval import _ece, recover_patient_ids
from src.uncertainty.calibration import IsotonicCalibrator  # noqa: F401

logger = get_logger("decontamination")

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "train"
MODELS_DIR = ROOT / "outputs" / "models"
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"
REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_binary"
TARGET_COLS = {"readmitted_binary", "readmitted_multi"}
SEED = 42
CONFORMAL_ALPHA = 0.10          # target 90% coverage


# ══════════════════════════════════════════════════════════════════════════
# Split conformal (implemented explicitly so the calibration set is auditable)
# ══════════════════════════════════════════════════════════════════════════

def conformal_quantile(p_cal: np.ndarray, y_cal: np.ndarray,
                       alpha: float = CONFORMAL_ALPHA) -> float:
    """
    Split-conformal threshold on the nonconformity score s = 1 - p(true class).

    Uses the finite-sample corrected level ceil((n+1)(1-alpha))/n, which is what
    gives the marginal coverage guarantee.
    """
    p_true = np.where(y_cal == 1, p_cal, 1.0 - p_cal)
    scores = 1.0 - p_true
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def conformal_sets(p: np.ndarray, q: float) -> np.ndarray:
    """Return an (n, 2) boolean array: whether each class is in the set."""
    return np.column_stack([(1.0 - (1.0 - p)) <= q, (1.0 - p) <= q])


def conformal_metrics(p: np.ndarray, y: np.ndarray, q: float) -> dict:
    """Empirical coverage and set size for a given conformal quantile."""
    sets = conformal_sets(p, q)
    covered = sets[np.arange(len(y)), y.astype(int)]
    size = sets.sum(axis=1)
    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(size.mean()),
        "share_singleton": float((size == 1).mean()),
        "share_both_labels": float((size == 2).mean()),
        "share_empty": float((size == 0).mean()),
        "n": int(len(y)),
    }


# ══════════════════════════════════════════════════════════════════════════
# Patient-level halving
# ══════════════════════════════════════════════════════════════════════════

def patient_halves(groups: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    Split rows into two halves BY PATIENT.

    Splitting by row would put the same patient's encounters on both sides,
    leaking the calibration set into the audit set and reproducing in miniature
    the very contamination this module exists to remove.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    perm = rng.permutation(uniq)
    half = set(perm[: len(perm) // 2].tolist())
    mask_cal = np.array([g in half for g in groups])
    return mask_cal, ~mask_cal


def f1_max_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 197):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def op_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p >= t).astype(int)
    return {
        "threshold": round(t, 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "predicted_positive_rate": round(float(pred.mean()), 4),
    }


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_decontamination() -> dict:
    logger.info("=" * 78)
    logger.info("Tier 2A.4 — decontamination of fit-and-evaluate-on-the-same-data")
    logger.info("=" * 78)

    train = pd.read_parquet(TRAIN_DIR / "train_fs.parquet")
    val = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")
    feat = [c for c in train.columns if c not in TARGET_COLS]

    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model = pickle.load(f)

    y_tr = train[TARGET].to_numpy()
    y_va = val[TARGET].to_numpy()
    y_te = test[TARGET].to_numpy()
    raw_tr = model.predict_proba(train[feat])[:, 1]
    raw_va = model.predict_proba(val[feat])[:, 1]
    raw_te = model.predict_proba(test[feat])[:, 1]

    g_va, note_va = recover_patient_ids("val", y_va)
    g_tr, note_tr = recover_patient_ids("train", y_tr)
    logger.info(f"  val patient recovery : {note_va[:70]}")
    logger.info(f"  train patient recovery: {note_tr[:70]}")
    if g_va is None:
        raise RuntimeError("cannot halve val by patient without verified ids")

    cal_m, aud_m = patient_halves(g_va)
    logger.info(f"  val halved by patient: calibration {cal_m.sum():,} rows / "
                f"audit {aud_m.sum():,} rows "
                f"({len(np.unique(g_va[cal_m]))} vs {len(np.unique(g_va[aud_m]))} patients)")
    assert not (set(g_va[cal_m]) & set(g_va[aud_m])), "patient leak between halves"

    # ── 1. ISOTONIC CALIBRATION / ECE ─────────────────────────────────────
    logger.info("-" * 62)
    logger.info("1/3 Isotonic calibration — ECE")
    iso_contaminated = IsotonicRegression(out_of_bounds="clip").fit(raw_va, y_va)
    ece_val_insample = _ece(y_va, iso_contaminated.predict(raw_va))
    ece_test_contam = _ece(y_te, iso_contaminated.predict(raw_te))

    iso_clean = IsotonicRegression(out_of_bounds="clip").fit(raw_va[cal_m], y_va[cal_m])
    ece_audit_heldout = _ece(y_va[aud_m], iso_clean.predict(raw_va[aud_m]))
    ece_cal_insample = _ece(y_va[cal_m], iso_clean.predict(raw_va[cal_m]))
    ece_test_clean = _ece(y_te, iso_clean.predict(raw_te))

    calibration = {
        "contaminated_fit_on_full_val": {
            "val_ece_IN_SAMPLE": round(ece_val_insample, 5),
            "test_ece_held_out": round(ece_test_contam, 5),
            "label": "val ECE is IN-SAMPLE — the calibrator was fitted on it",
        },
        "decontaminated_fit_on_val_calibration_half": {
            "calibration_half_ece_IN_SAMPLE": round(ece_cal_insample, 5),
            "audit_half_ece_HELD_OUT": round(ece_audit_heldout, 5),
            "test_ece_held_out": round(ece_test_clean, 5),
        },
        "optimism_val_ece": round(ece_audit_heldout - ece_val_insample, 5),
    }
    logger.info(f"  contaminated: val ECE {ece_val_insample:.5f} (IN-SAMPLE)")
    logger.info(f"  decontaminated: audit-half ECE {ece_audit_heldout:.5f} (HELD-OUT)"
                f" | calib-half {ece_cal_insample:.5f} (in-sample)")
    logger.info(f"  optimism hidden by the contaminated number: "
                f"{ece_audit_heldout - ece_val_insample:+.5f}")

    # ── 2. CONFORMAL COVERAGE ─────────────────────────────────────────────
    logger.info("-" * 62)
    logger.info(f"2/3 Conformal coverage (target {1 - CONFORMAL_ALPHA:.0%})")
    p_va_c = iso_contaminated.predict(raw_va)
    p_te_c = iso_contaminated.predict(raw_te)
    q_contam = conformal_quantile(p_va_c, y_va)
    cov_val_insample = conformal_metrics(p_va_c, y_va, q_contam)
    cov_test_contam = conformal_metrics(p_te_c, y_te, q_contam)

    p_cal = iso_clean.predict(raw_va[cal_m])
    p_aud = iso_clean.predict(raw_va[aud_m])
    p_te_k = iso_clean.predict(raw_te)
    q_clean = conformal_quantile(p_cal, y_va[cal_m])
    cov_cal_insample = conformal_metrics(p_cal, y_va[cal_m], q_clean)
    cov_audit_heldout = conformal_metrics(p_aud, y_va[aud_m], q_clean)
    cov_test_clean = conformal_metrics(p_te_k, y_te, q_clean)

    conformal = {
        "target_coverage": 1 - CONFORMAL_ALPHA,
        "contaminated_calibrated_on_full_val": {
            "quantile": round(q_contam, 5),
            "val_IN_SAMPLE": cov_val_insample,
            "test_held_out": cov_test_contam,
            "label": ("val coverage equals the target BY CONSTRUCTION — this is "
                      "what the algorithm solves for, not evidence of stability"),
        },
        "decontaminated_calibrated_on_val_calibration_half": {
            "quantile": round(q_clean, 5),
            "calibration_half_IN_SAMPLE": cov_cal_insample,
            "audit_half_HELD_OUT": cov_audit_heldout,
            "test_held_out": cov_test_clean,
        },
        "honest_coverage_gap_audit_minus_target": round(
            cov_audit_heldout["coverage"] - (1 - CONFORMAL_ALPHA), 5),
    }
    logger.info(f"  contaminated : val coverage {cov_val_insample['coverage']:.4f} "
                f"(IN-SAMPLE) | test {cov_test_contam['coverage']:.4f}")
    logger.info(f"  decontaminated: calib-half {cov_cal_insample['coverage']:.4f} "
                f"(in-sample) | AUDIT-HALF {cov_audit_heldout['coverage']:.4f} "
                f"(HELD-OUT) | test {cov_test_clean['coverage']:.4f}")
    logger.info(f"  mean set size: audit {cov_audit_heldout['mean_set_size']:.3f} "
                f"| both-labels {cov_audit_heldout['share_both_labels']:.3f}")

    # ── 3. THRESHOLD ──────────────────────────────────────────────────────
    logger.info("-" * 62)
    logger.info("3/3 Operating threshold")
    t_contam = f1_max_threshold(y_va, p_va_c)
    contam_val = op_metrics(y_va, p_va_c, t_contam)
    contam_test = op_metrics(y_te, p_te_c, t_contam)

    # (a) selected on the val calibration half, applied to the audit half
    t_valhalf = f1_max_threshold(y_va[cal_m], p_cal)
    clean_audit = op_metrics(y_va[aud_m], p_aud, t_valhalf)

    # (b) selected on a held-out slice of TRAIN, so the reference window carries
    #     no fitted quantity at all.
    #
    #     Tier 2C.6 CORRECTION — this block previously mixed THREE probability
    #     scales in one comparison: the threshold was selected under `iso_clean`,
    #     applied to val under `iso_contaminated` (`p_va_c`) and to test under
    #     `iso_clean` (`p_te_k`). A threshold is a cut point on a probability
    #     scale, so scoring the two windows on different calibrators makes the
    #     val->test difference partly an artifact of which calibrator each side
    #     got. It reported a val->test F1 IMPROVEMENT that did not survive being
    #     recomputed on one scale, and `threshold_optimism_...` was derived from
    #     it. Found when Tier 2C.6 reconciled `headline_metrics_ci.json` to this
    #     threshold and got a different test F1 than this file reports.
    #
    #     Both scales are now reported, each internally consistent:
    #       DEPLOYMENT scale  — everything under the SHIPPED calibrator (fitted
    #         on full val). This is the scale `repeated_eval.py`,
    #         `fairness_audit.py` and the served model actually use, so it is the
    #         one the headline metrics must match. Val is still contaminated as a
    #         CALIBRATION set here, which is stated separately in section 1; what
    #         this variant removes is the THRESHOLD contamination.
    #       RESEARCH scale    — everything under `iso_clean`, with val evaluated
    #         on the AUDIT half only, since the calibration half fitted it. Fully
    #         held out on both axes, and not the scale that ships.
    if g_tr is None:
        raise RuntimeError(
            "cannot select a threshold on a held-out train slice without "
            "verified train patient ids; refusing to fall back to the val "
            "calibration half, which would silently reintroduce val-fitted "
            "quantities into the reference window (R6)")

    tr_cal_m, _ = patient_halves(g_tr, seed=SEED + 1)
    train_src = "held-out slice of train (patient-level)"

    # DEPLOYMENT scale — one calibrator (the shipped one) throughout.
    p_tr_deploy = iso_contaminated.predict(raw_tr)
    t_deploy = f1_max_threshold(y_tr[tr_cal_m], p_tr_deploy[tr_cal_m])
    deploy_val = op_metrics(y_va, p_va_c, t_deploy)
    deploy_test = op_metrics(y_te, p_te_c, t_deploy)
    f1_drop_deploy = deploy_val["f1"] - deploy_test["f1"]

    # RESEARCH scale — one calibrator (iso_clean) throughout; val = audit half.
    p_tr_clean = iso_clean.predict(raw_tr)
    t_research = f1_max_threshold(y_tr[tr_cal_m], p_tr_clean[tr_cal_m])
    research_val = op_metrics(y_va[aud_m], p_aud, t_research)
    research_test = op_metrics(y_te, p_te_k, t_research)
    f1_drop_research = research_val["f1"] - research_test["f1"]

    f1_drop_contam = contam_val["f1"] - contam_test["f1"]

    threshold = {
        "contaminated_selected_on_val": {
            "threshold": t_contam, "val_IN_SAMPLE": contam_val,
            "test_held_out": contam_test,
            "f1_drop_val_to_test": round(f1_drop_contam, 4),
            "probability_scale": "shipped calibrator (fitted on full val)",
            "label": ("the reference window has a threshold fitted to itself; "
                      "the production window does not (audit F13)"),
        },
        "decontaminated_selected_on_val_calibration_half": {
            "threshold": t_valhalf, "audit_half_HELD_OUT": clean_audit,
            "probability_scale": "iso_clean (fitted on the val calibration half)"},
        "decontaminated_selected_on_train_holdout": {
            "threshold": t_deploy, "source": train_src,
            "probability_scale": ("shipped calibrator throughout — selection, "
                                  "val and test all on one scale"),
            "scale_variant": "DEPLOYMENT",
            "why_this_is_the_one_downstream_modules_use": (
                "it is the scale the served model and every metrics module "
                "actually operate on; a threshold from another scale would not "
                "mean the same thing when applied to deployed probabilities"),
            "val_held_out": deploy_val, "test_held_out": deploy_test,
            "f1_drop_val_to_test": round(f1_drop_deploy, 4)},
        "decontaminated_selected_on_train_holdout_research_scale": {
            "threshold": t_research, "source": train_src,
            "probability_scale": ("iso_clean throughout; val is the AUDIT half "
                                  "only, because the calibration half fitted it"),
            "scale_variant": "RESEARCH",
            "val_audit_half_HELD_OUT": research_val, "test_held_out": research_test,
            "f1_drop_val_to_test": round(f1_drop_research, 4)},
        "threshold_optimism_in_the_reported_f1_drop": round(
            f1_drop_contam - f1_drop_deploy, 4),
        "threshold_optimism_note": (
            "contaminated drop minus DEPLOYMENT-scale decontaminated drop. Both "
            "terms are now computed on the same probability scale; before the "
            "Tier 2C.6 correction the second term mixed two calibrators and the "
            "difference was not a like-for-like comparison."),
        "threshold_optimism_research_scale": round(
            f1_drop_contam - f1_drop_research, 4),
    }
    logger.info(f"  contaminated  thr {t_contam:.4f}: val F1 {contam_val['f1']:.4f} "
                f"-> test {contam_test['f1']:.4f} (drop {f1_drop_contam:+.4f})")
    logger.info(f"  train-holdout thr {t_deploy:.4f} [DEPLOYMENT scale]: "
                f"val F1 {deploy_val['f1']:.4f} -> test {deploy_test['f1']:.4f} "
                f"(drop {f1_drop_deploy:+.4f})")
    logger.info(f"  train-holdout thr {t_research:.4f} [RESEARCH scale]: "
                f"val(audit) F1 {research_val['f1']:.4f} -> "
                f"test {research_test['f1']:.4f} (drop {f1_drop_research:+.4f})")
    logger.info(f"  threshold optimism in the reported drop: "
                f"{f1_drop_contam - f1_drop_deploy:+.4f} (deployment scale)")

    suspicion = (
        "Coverage did NOT degrade materially on the audit half. Checked rather "
        "than assumed: the calibration and audit halves are disjoint BY PATIENT "
        "(verified by assertion), and the quantile was refitted on the "
        "calibration half only. Split conformal's marginal guarantee is "
        "distribution-free and holds on any exchangeable sample, so near-target "
        "coverage on a held-out half of the SAME window is the expected result, "
        "not a sign of leakage. The meaningful test of conformal under shift is "
        "the TEST window, where exchangeability is violated — that number is "
        "reported above and is the one that matters."
        if abs(cov_audit_heldout["coverage"] - (1 - CONFORMAL_ALPHA)) < 0.02
        else "Coverage degraded on the audit half, as expected once measured honestly.")

    report = {
        "phase": "2A.4",
        "title": "Decontamination of threshold, conformal and calibration",
        "val_split_by_patient": {
            "calibration_rows": int(cal_m.sum()), "audit_rows": int(aud_m.sum()),
            "calibration_patients": int(len(np.unique(g_va[cal_m]))),
            "audit_patients": int(len(np.unique(g_va[aud_m]))),
            "patient_disjoint": True,
            "why_by_patient": ("row-level halving would place a patient's "
                               "encounters on both sides, leaking calibration "
                               "into audit"),
        },
        "calibration_ece": calibration,
        "conformal": conformal,
        "threshold": threshold,
        "coverage_suspicion_check": suspicion,
        "patient_recovery": {"val": note_va, "train": note_tr},
        "reproducibility": {"seed": SEED, "python": platform.python_version(),
                            "numpy": np.__version__, "pandas": pd.__version__},
    }
    out = REPORTS_DIR / "decontamination.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("-" * 62)
    logger.info(f"Report: {out.name}")
    logger.info("=" * 78)
    return report


if __name__ == "__main__":
    r = run_decontamination()
    c = r["conformal"]
    print("\nConformal coverage (target 0.90)")
    print(f"  contaminated  val  : {c['contaminated_calibrated_on_full_val']['val_IN_SAMPLE']['coverage']:.4f}  IN-SAMPLE")
    print(f"  decontaminated audit: {c['decontaminated_calibrated_on_val_calibration_half']['audit_half_HELD_OUT']['coverage']:.4f}  HELD-OUT")
    print(f"  decontaminated test : {c['decontaminated_calibrated_on_val_calibration_half']['test_held_out']['coverage']:.4f}  HELD-OUT")
    t = r["threshold"]
    print(f"\nThreshold optimism in the reported F1 drop: "
          f"{t['threshold_optimism_in_the_reported_f1_drop']:+.4f}")
    e = r["calibration_ece"]
    print(f"ECE: val {e['contaminated_fit_on_full_val']['val_ece_IN_SAMPLE']:.5f} (in-sample) "
          f"-> audit {e['decontaminated_fit_on_val_calibration_half']['audit_half_ece_HELD_OUT']:.5f} (held-out)")
