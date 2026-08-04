"""
DriftSentinel — Tier 2A.2: repeated evaluation with confidence intervals

WHY
    Every headline number in the repository came from a single split with
    `random_state=42`, with no variance, no interval, and no way to tell whether
    a reported difference exceeded noise (audit F8). No headline metric ships
    as a single number: each is reported over repeated splits or seeds with a
    bootstrap interval, or is marked single-run with the reason.

THREE SOURCES OF VARIANCE, REPORTED SEPARATELY
    They answer different questions and collapsing them would be dishonest.

    1. ESTIMATION variance on the deployed split
       Patient-clustered bootstrap over the evaluation rows. Answers: "how
       precisely is this metric estimated on this data?" This is the right
       interval for the entry-cohort split, which is DETERMINISTIC — it is the
       split under study, not one draw from a distribution of splits.

    2. MODEL variance
       Refit with N different model seeds on the same split. Answers: "how much
       of this number is the learner's own stochasticity?"

    3. SPLIT variance
       N random patient-level splits. Answers: "how much would this move if the
       cohort had been drawn differently?" Reported for reference only: a random
       split is a DIFFERENT REGIME from the entry-cohort split (Tier 0 showed
       0.25/6 vs 3.30/6 signals), so its spread is not an error bar on the
       deployed number.

CLUSTERING
    46.2% of rows come from multi-visit patients. Row-level resampling is
    anti-conservative (audit F28), so every bootstrap here resamples PATIENTS.
    Patient ids are recovered from split_index.json and VERIFIED against the
    label sequence before use — if verification fails the interval is reported
    as row-level with that fact attached, never silently.

OUTPUT
    outputs/reports/headline_metrics_ci.json — the canonical source for every
    number in the final README (R4). Nothing enters the README that is not here.
"""

from __future__ import annotations

import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger
from src.uncertainty.calibration import IsotonicCalibrator  # noqa: F401 (unpickling)

logger = get_logger("repeated_eval")

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "train"
MODELS_DIR = ROOT / "outputs" / "models"
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"
REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_binary"
TARGET_COLS = {"readmitted_binary", "readmitted_multi"}

N_BOOT = 2000
N_SEEDS = 20
SEED = 42


# ══════════════════════════════════════════════════════════════════════════
# Patient recovery (shared; generalises the registry's test-only version)
# ══════════════════════════════════════════════════════════════════════════

def recover_patient_ids(split: str, y_expected: np.ndarray) -> tuple[np.ndarray | None, str]:
    """
    Recover `patient_nbr` for a split's rows so bootstraps can cluster on it.

    The feature parquets do not carry patient ids. The splitter sorts each split
    by encounter_id, so the raw rows for that split's patients in the same order
    should align 1:1. That is an ASSUMPTION, so it is verified against the label
    sequence (R6: check the property, not a proxy). On failure the caller falls
    back to row-level resampling WITH the caveat attached.
    """
    try:
        from src.data.preprocessor import TARGET_BINARY_MAP
        with open(ARTIFACTS_DIR / "split_index.json") as f:
            idx = json.load(f)
        raw = pd.read_csv(ROOT / "data" / "raw" / "diabetes_hospital" / "diabetic_data.csv",
                          usecols=["encounter_id", "patient_nbr", "readmitted"],
                          na_values=["?"], keep_default_na=False)
        pats = set(idx[f"{split}_patient_ids"])
        sub = raw[raw["patient_nbr"].isin(pats)].sort_values("encounter_id")
        if len(sub) != len(y_expected):
            return None, f"FAILED: row count {len(sub)} != {len(y_expected)}"
        rec = sub["readmitted"].map(TARGET_BINARY_MAP).to_numpy().astype(int)
        if not np.array_equal(rec, np.asarray(y_expected).astype(int)):
            return None, "FAILED: recovered label sequence does not match the parquet target"
        return sub["patient_nbr"].to_numpy(), (
            f"OK: verified against the label sequence "
            f"({sub['patient_nbr'].nunique():,} patients / {len(sub):,} rows)")
    except Exception as e:                                    # pragma: no cover
        return None, f"FAILED: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════

def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(e)


def core_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """The headline metric set for one split at one operating threshold."""
    pred = (p >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "ece": _ece(y, p),
        "predicted_positive_rate": float(pred.mean()),
        "prevalence": float(y.mean()),
    }


def clustered_bootstrap(y: np.ndarray, p: np.ndarray, threshold: float,
                        groups: np.ndarray | None, n_boot: int,
                        seed: int = SEED) -> dict:
    """
    Patient-clustered bootstrap CI for every core metric.

    Resamples PATIENTS, not rows. Returns point estimate, mean, std and a
    percentile 95% interval per metric, plus the resampling unit so the reader
    can see whether clustering was actually achieved.
    """
    rng = np.random.default_rng(seed)
    point = core_metrics(y, p, threshold)

    if groups is None:
        pool, clustered = None, False
    else:
        uniq, inv = np.unique(groups, return_inverse=True)
        pool = [np.flatnonzero(inv == g) for g in range(len(uniq))]
        clustered = True

    draws = {k: [] for k in point}
    for _ in range(n_boot):
        if clustered:
            pick = rng.integers(0, len(pool), size=len(pool))
            idx = np.concatenate([pool[i] for i in pick])
        else:
            idx = rng.integers(0, len(y), size=len(y))
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        m = core_metrics(yb, pb, threshold)
        for k, v in m.items():
            draws[k].append(v)

    out = {}
    for k, v in draws.items():
        arr = np.asarray(v, dtype=float)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        out[k] = {
            "point": round(point[k], 5),
            "mean": round(float(np.nanmean(arr)), 5),
            "std": round(float(np.nanstd(arr, ddof=1)), 5),
            "ci95": [round(float(lo), 5), round(float(hi), 5)],
            "n_boot_effective": int(len(arr)),
        }
    return {
        "metrics": out,
        "resampling_unit": "patient (cluster)" if clustered else "row",
        "cluster_robust": clustered,
        "caveat": None if clustered else (
            "row-level resampling: intervals are anti-conservative because "
            "patients contribute multiple encounters (audit F28)"),
    }


# ══════════════════════════════════════════════════════════════════════════
# Variance components
# ══════════════════════════════════════════════════════════════════════════

def model_seed_variance(X_tr, y_tr, evals: dict, threshold: float,
                        n_seeds: int = N_SEEDS) -> dict:
    """
    Refit the learner with n_seeds different seeds on the SAME split.

    Isolates the learner's own stochasticity from split variance. Uses the
    shipped LGBM_PARAMS so this measures the deployed configuration.
    """
    import lightgbm as lgb
    from src.models.trainer import LGBM_PARAMS

    per_split = {k: {m: [] for m in
                     ("auc", "precision", "recall", "f1", "brier", "ece")}
                 for k in evals}
    for s in range(n_seeds):
        params = {**LGBM_PARAMS, "random_state": s}
        model = lgb.LGBMClassifier(**params).fit(X_tr, y_tr)
        for name, (Xe, ye) in evals.items():
            p = model.predict_proba(Xe)[:, 1]
            m = core_metrics(ye, p, threshold)
            for k in per_split[name]:
                per_split[name][k].append(m[k])

    out = {}
    for name, mm in per_split.items():
        out[name] = {k: {"mean": round(float(np.mean(v)), 5),
                         "std": round(float(np.std(v, ddof=1)), 5),
                         "min": round(float(np.min(v)), 5),
                         "max": round(float(np.max(v)), 5),
                         "n_seeds": n_seeds}
                     for k, v in mm.items()}
    return out


def split_variance(n_seeds: int = N_SEEDS) -> dict:
    """
    N random patient-level splits, using the fixed simple feature pipeline from
    the Tier 0 regime harness.

    REPORTED FOR REFERENCE ONLY. A random split is a DIFFERENT REGIME from the
    entry-cohort split — Tier 0 measured 0.25/6 signals under random splitting
    versus 3.30/6 under entry-cohort — so this spread is NOT an error bar on the
    deployed number. It answers "how much would this move under a different
    cohort draw", which is a different question.
    """
    import yaml
    from src.investigation.split_regimes import (encode, load_and_prepare,
                                                 split_random, _fit_model,
                                                 _pick_threshold)

    with open(ROOT / "configs" / "split_regimes.yaml", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    df = load_and_prepare(conf)
    tcol = "y_lt30"

    rows = []
    for s in range(n_seeds):
        sp = split_random(df, conf, s)
        tr, va, te = sp["train"], sp["val"], sp["test"]
        Xtr, Xo, cols = encode(tr, {"val": va, "test": te}, conf)
        ytr = tr[tcol].to_numpy()
        model = _fit_model(Xtr, ytr, conf, s)
        thr = _pick_threshold(ytr, model.predict_proba(Xtr)[:, 1])
        for name, d in (("val", va), ("test", te)):
            p = model.predict_proba(Xo[name])[:, 1]
            m = core_metrics(d[tcol].to_numpy(), p, thr)
            rows.append({"seed": s, "split": name, **m})

    t = pd.DataFrame(rows)
    out = {}
    for name in ("val", "test"):
        sub = t[t["split"] == name]
        out[name] = {c: {"mean": round(float(sub[c].mean()), 5),
                         "std": round(float(sub[c].std(ddof=1)), 5),
                         "n_seeds": n_seeds}
                     for c in ("auc", "precision", "recall", "f1", "brier",
                               "ece", "prevalence")}
    return out


# ══════════════════════════════════════════════════════════════════════════
# Operating threshold
# ══════════════════════════════════════════════════════════════════════════

def load_operating_threshold() -> tuple[float, dict]:
    """
    Return the DECONTAMINATED operating threshold and its provenance.

    Tier 2C.6 correction. This module previously read the threshold from
    `evaluation_report.json`, which is the F1-max threshold fitted ON VAL — and
    val is also the drift reference window, so the reference window carried a
    threshold tuned to itself while the production window did not. Tier 2A.4
    measured the resulting optimism at **0.0641 of the reported F1 drop** and
    produced a replacement selected on a held-out slice of TRAIN.

    Leaving the contaminated value here meant the file that calls itself "the
    canonical source for every number in the README" held the number a whole
    phase was spent proving wrong.

    Raises rather than falling back (R6): a default here would be
    indistinguishable from a real measurement, and would silently restore
    exactly the contamination this function exists to remove.
    """
    path = REPORTS_DIR / "decontamination.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required for the decontaminated operating threshold. "
            "Run `python src/uncertainty/decontamination.py` first. This does "
            "NOT fall back to the val-fitted threshold in evaluation_report.json "
            "— that value is contaminated (Tier 2A.4) and silently substituting "
            "it would reintroduce the defect.")
    with open(path, encoding="utf-8") as f:
        deco = json.load(f)
    node = deco["threshold"]["decontaminated_selected_on_train_holdout"]

    with open(MODELS_DIR / "evaluation_report.json", encoding="utf-8") as f:
        superseded = json.load(f)["lgbm"]["threshold"]

    provenance = {
        "value": node["threshold"],
        "selected_on": node["source"],
        "source_artifact": "outputs/reports/decontamination.json",
        "source_path": "threshold/decontaminated_selected_on_train_holdout/threshold",
        "supersedes": {
            "value": superseded,
            "selected_on": "val, by F1-max",
            "source_artifact": "outputs/models/evaluation_report.json",
            "why_superseded": (
                "val is BOTH the threshold-selection set and the drift reference "
                "window, so the reference window's F1 was optimistic by "
                f"{deco['threshold']['threshold_optimism_in_the_reported_f1_drop']} "
                "while the production window's was not. Every threshold-dependent "
                "number computed at that threshold inherited the optimism."),
        },
        "unaffected_metrics": ["auc", "brier", "prevalence"],
        "affected_metrics": ["precision", "recall", "f1", "predicted_positive_rate"],
    }
    return node["threshold"], provenance


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_repeated_eval() -> dict:
    logger.info("=" * 78)
    logger.info("Tier 2A.2 — repeated evaluation with confidence intervals")
    logger.info("=" * 78)

    train = pd.read_parquet(TRAIN_DIR / "train_fs.parquet")
    val = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")
    feat = [c for c in train.columns if c not in TARGET_COLS]

    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "calibrator_isotonic_lgbm_v1.pkl", "rb") as f:
        cal = pickle.load(f)
    threshold, threshold_provenance = load_operating_threshold()
    logger.info(f"Operating threshold: {threshold} "
                f"({threshold_provenance['selected_on']})")

    from src.models.trainer import LGBM_PARAMS
    with open(MODELS_DIR / "training_summary.json", encoding="utf-8") as f:
        deployed_n_trees = json.load(f)["lgbm"]["best_iteration"]

    def proba(d):
        return cal.transform(model.predict_proba(d[feat])[:, 1])

    splits = {"train": train, "val": val, "test": test}
    y = {k: d[TARGET].to_numpy() for k, d in splits.items()}
    p = {k: proba(d) for k, d in splits.items()}

    # ── 1. estimation variance (patient-clustered bootstrap) ──────────────
    logger.info("-" * 62)
    logger.info(f"1/3 Estimation variance — patient-clustered bootstrap "
                f"({N_BOOT} draws)")
    estimation, recovery = {}, {}
    for k in splits:
        g, note = recover_patient_ids(k, y[k])
        recovery[k] = note
        logger.info(f"  {k:<6} patient recovery: {note[:70]}")
        estimation[k] = clustered_bootstrap(y[k], p[k], threshold, g, N_BOOT)
        m = estimation[k]["metrics"]
        logger.info(f"  {k:<6} AUC {m['auc']['point']:.4f} "
                    f"[{m['auc']['ci95'][0]:.4f}, {m['auc']['ci95'][1]:.4f}] | "
                    f"F1 {m['f1']['point']:.4f} "
                    f"[{m['f1']['ci95'][0]:.4f}, {m['f1']['ci95'][1]:.4f}] | "
                    f"ECE {m['ece']['point']:.4f}")

    # ── 2. model-seed variance ────────────────────────────────────────────
    logger.info("-" * 62)
    logger.info(f"2/3 Model variance — {N_SEEDS} model seeds, same split")
    seedvar = model_seed_variance(
        train[feat], y["train"],
        {"val": (val[feat], y["val"]), "test": (test[feat], y["test"])},
        threshold, N_SEEDS)
    for k, mm in seedvar.items():
        logger.info(f"  {k:<6} AUC {mm['auc']['mean']:.4f} +- {mm['auc']['std']:.4f} "
                    f"(min {mm['auc']['min']:.4f}, max {mm['auc']['max']:.4f})")

    # ── 3. split variance (reference only) ────────────────────────────────
    logger.info("-" * 62)
    logger.info(f"3/3 Split variance — {N_SEEDS} random patient splits "
                f"(REFERENCE ONLY, different regime)")
    splitvar = split_variance(N_SEEDS)
    for k, mm in splitvar.items():
        logger.info(f"  {k:<6} AUC {mm['auc']['mean']:.4f} +- {mm['auc']['std']:.4f} "
                    f"| prevalence {mm['prevalence']['mean']:.4f}")

    # ── downstream artifacts that already carry their own intervals ───────
    linked = {}
    for name, path, note in [
        ("model_comparison_delong_bootstrap", "outputs/registry/model_comparison.json",
         "v1 vs v2 on the only comparable split, DeLong + paired patient-clustered bootstrap"),
        ("threshold_policy", "outputs/reports/threshold_policy_lgbm_v1.json",
         "operating points incl. PPR, budget-constrained points, cost-ratio sweep, DCA"),
        ("regime_signal_firing_rates", "outputs/reports/regime_random.json",
         "drift-signal firing rates over 20 seeds per regime (Tier 0)"),
    ]:
        pth = ROOT / path
        linked[name] = {"path": path, "exists": pth.exists(), "note": note}

    contamination_flags = {
        "val_ece_is_in_sample": (
            "ECE on `val` is ~0 BY CONSTRUCTION: the isotonic calibrator was FIT "
            "on val, so measuring calibration error on val measures the fit, not "
            "generalisation (R3). The val ECE reported here is IN-SAMPLE and is "
            "not evidence of calibration quality. Decontaminated in Tier 2A.4; "
            "the test-split ECE is the honest number."),
        # Tier 2C.6: the tree count here was hand-typed as 173 and went stale at
        # the Tier 2A.1 target switch — the same defect as registry.py. Read it.
        "model_seed_variance_uses_a_different_configuration": (
            f"The seed-variance refits use LGBM_PARAMS as shipped "
            f"(n_estimators={LGBM_PARAMS.get('n_estimators')}, no early "
            f"stopping), whereas the DEPLOYED lgbm_v1 stopped early at "
            f"{deployed_n_trees} trees. The seed-variance means are therefore "
            "NOT directly comparable to the point estimates — they quantify "
            "learner stochasticity within that configuration, not the deployed "
            "model's spread."),
        "threshold_dependent_metrics_are_now_decontaminated": (
            "Precision, recall, F1 and predicted-positive rate are computed at "
            f"the decontaminated threshold {threshold} "
            f"({threshold_provenance['selected_on']}), NOT at the val-fitted "
            f"{threshold_provenance['supersedes']['value']} used before Tier "
            "2C.6. AUC, Brier and prevalence are threshold-free and are "
            "unchanged. See `operating_threshold_provenance`."),
    }

    single_run = {
        "conformal_coverage": (
            "reported single-run pending Tier 2A.4: the shipped coverage is "
            "measured on the SAME data used to calibrate the predictor, so it is "
            "guaranteed by construction and an interval around it would not make "
            "it evidence (R3). Decontaminated and given an interval in 2A.4."),
        "feature_selection": (
            "single-run: the selector is deterministic given seed 42 and its "
            "output did not change under a 4x prevalence shift. Variance is "
            "characterised by the Tier 2A.5 ablation instead of a CI."),
        "entry_cohort_split_itself": (
            "the entry-cohort split is DETERMINISTIC — it is the object of study, "
            "not a draw. Its uncertainty is estimation variance (component 1), "
            "not split variance."),
    }

    report = {
        "phase": "2A.2",
        "title": "Repeated evaluation with confidence intervals",
        "target": "readmitted_binary = 30-day readmission (<30)",
        "operating_threshold": threshold,
        "operating_threshold_provenance": threshold_provenance,
        "rule": "no single-number claims — every headline metric carries variance and a bootstrap interval, or is explicitly marked single-run",
        "variance_components": {
            "1_estimation_patient_clustered_bootstrap": {
                "n_boot": N_BOOT,
                "description": ("precision of each metric on the DEPLOYED split; "
                                "the correct interval for the entry-cohort split"),
                "patient_id_recovery": recovery,
                "by_split": estimation,
            },
            "2_model_seed_variance": {
                "n_seeds": N_SEEDS,
                "description": "learner stochasticity, same split",
                "by_split": seedvar,
            },
            "3_split_variance_reference_only": {
                "n_seeds": N_SEEDS,
                "description": ("random patient splits — a DIFFERENT REGIME "
                                "(Tier 0: 0.25/6 signals vs 3.30/6 entry-cohort). "
                                "NOT an error bar on the deployed number."),
                "by_split": splitvar,
            },
        },
        "linked_artifacts_with_own_intervals": linked,
        "explicitly_single_run": single_run,
        "contamination_flags": contamination_flags,
        "canonical_source_rule": (
            "R4 — this file is the canonical source for every number in the "
            "README. A number not present here does not ship."),
        "reproducibility": {
            "seed": SEED, "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
        },
    }

    out = REPORTS_DIR / "headline_metrics_ci.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("-" * 62)
    logger.info(f"Report: {out}")
    logger.info("=" * 78)
    return report


if __name__ == "__main__":
    r = run_repeated_eval()
    est = r["variance_components"]["1_estimation_patient_clustered_bootstrap"]["by_split"]
    print("\nEstimation variance (patient-clustered bootstrap):")
    for k, v in est.items():
        m = v["metrics"]
        print(f"  {k:<6} AUC {m['auc']['point']:.4f} "
              f"[{m['auc']['ci95'][0]:.4f}, {m['auc']['ci95'][1]:.4f}]  "
              f"({v['resampling_unit']})")
