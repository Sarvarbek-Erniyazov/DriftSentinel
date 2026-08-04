"""
DriftSentinel — Phases 0.2 / 0.3 / 0.4: ground-truth detector benchmark

WHAT
    Runs the repository's ACTUAL drift detectors under split regimes whose
    ground truth is known, and reports which signals fire in which regime.

      Phase 0.2  random split          -> negative control  (no drift exists)
      Phase 0.3  synthetic shifts      -> positive controls (one mechanism each)
      Phase 0.4  regime x signal matrix

WHY
    The central claim was "DriftSentinel detects drift", evidenced by 8/8
    signals firing on one split. That is unfalsifiable without a no-drift
    control (audit F6). A detector that never stays silent is not a detector.

REGIMES
    entry_cohort  the repository's existing split (patients ordered by first
                  encounter_id) — the setup every current headline came from
    random        negative control
    temporal      TRUE chronological split of ENCOUNTERS, newly defensible
                  because Phase 0.1 verified encounter_id chronology against
                  external anchors. Leakage-controlled: a patient seen in an
                  earlier split is removed from later splits.
    synthetic     covariate / label / concept shift, swept over magnitude

DESIGN COMMITMENTS
  * The detectors under test are IMPORTED, not reimplemented — this measures
    `src/drift/*.py` as shipped, including its thresholds and its verdict logic.
  * The model is identical in every regime: fixed hyperparameters, fixed tree
    count, no early stopping anywhere. Regimes differ only in their DATA
    (the lgbm_v1/v2 comparison failed this; see audit F3).
  * The operating threshold is selected on a held-out slice of TRAIN, never on
    val, because val is the drift reference window (audit F13). This
    decontamination is applied here from the start.
  * Feature construction is deliberately simple and refit per regime
    (fit-on-train-only). SCOPE NOTE: the 7-stage selector is NOT re-run per
    seed. The object of study is detector calibration, not feature selection,
    and holding the feature pipeline fixed is what makes regimes comparable.

OUTPUTS
    outputs/reports/regime_random.json      (Phase 0.2)
    outputs/reports/regime_synthetic.json   (Phase 0.3)
    outputs/reports/regime_matrix.csv/json  (Phase 0.4)
    outputs/figure/40..42_*.png
    outputs/log/split_regimes.log
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
from src.monitoring.logger import get_logger

# The detectors under test — imported as shipped.
from src.drift import alerting as al_mod
from src.drift import concept_drift as cd_mod
from src.drift import data_drift as dd_mod
from src.drift.alerting import AlertEngine
from src.drift.concept_drift import ConceptDriftDetector
from src.drift.data_drift import DataDriftDetector

logger = get_logger("split_regimes")

ROOT        = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "split_regimes.yaml"
RAW_CSV     = ROOT / "data" / "raw" / "diabetes_hospital" / "diabetic_data.csv"
REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR  = ROOT / "outputs" / "figure"
SCRATCH_DIR = ROOT / "outputs" / "reports" / "_regime_raw"
for _d in (REPORTS_DIR, FIGURE_DIR, SCRATCH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# The detectors write per-call artifacts and log ~60 lines each. Across ~200
# runs that would bury the phase log, so their output is redirected to a scratch
# directory and their loggers quieted. Their BEHAVIOUR is untouched.
# Redirect detector output to a scratch directory for the duration of the sweep.
#
# Tier 1.7 replaced the previous raw attribute assignment (`cd_mod.REPORTS_DIR =
# ...`) with the explicit `set_reports_dir()` / `set_alerts_dir()` API each
# module now exposes. The redirection itself is still correct and still wanted:
# this sweep performs ~200 detector runs, and scattering that many throwaway
# artifacts through outputs/log/ and outputs/alerts/ would bury the real evidence
# trail. What changed is that redirection is now a supported, documented call
# rather than a monkey-patch invisible to anyone reading the detector modules.
#
# The overwrite hazard that originally motivated it is separately handled by
# src/monitoring/artifact_io.py, which refuses to clobber an existing artifact.
cd_mod.set_reports_dir(SCRATCH_DIR)
dd_mod.set_reports_dir(SCRATCH_DIR)
al_mod.set_alerts_dir(SCRATCH_DIR)
for _name in ("concept_drift", "data_drift", "alerting"):
    get_logger(_name).setLevel(logging.WARNING)

# Tier 1.5: the matrix is built over VOTING signals only, read from the detector
# rather than hardcoded here — otherwise this harness and the detector could
# disagree about what counts as evidence. cusum_alarm and ph_alarm are still
# collected, as diagnostics, so their behaviour stays visible in the regime
# study that condemned them.
SIGNALS = list(cd_mod.VOTING_SIGNALS)
DIAGNOSTIC_SIGNALS = list(cd_mod.DIAGNOSTIC_SIGNALS)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})
C_MAIN, C_WARN, C_OK, C_NULL = "#2b6cb0", "#c05621", "#2f855a", "#718096"


# ══════════════════════════════════════════════════════════════════════════
# Data preparation
# ══════════════════════════════════════════════════════════════════════════

def _icd9_chapter(code: str) -> str:
    """
    Map an ICD-9 code to its chapter. Standard clinical practice, and it also
    keeps `data_drift._chi2` tractable: that implementation loops over unique
    values, so raw diag codes (~700 levels) would dominate the entire runtime.
    """
    if code in ("", "?", "nan", "None") or pd.isna(code):
        return "MISSING"
    s = str(code)
    if s.startswith("V"):
        return "V_supplementary"
    if s.startswith("E"):
        return "E_external"
    try:
        v = float(s)
    except ValueError:
        return "MISSING"
    bounds = [(1, 139, "infectious"), (140, 239, "neoplasm"), (240, 279, "endocrine"),
              (280, 289, "blood"), (290, 319, "mental"), (320, 389, "nervous"),
              (390, 459, "circulatory"), (460, 519, "respiratory"),
              (520, 579, "digestive"), (580, 629, "genitourinary"),
              (630, 679, "pregnancy"), (680, 709, "skin"),
              (710, 739, "musculoskeletal"), (740, 759, "congenital"),
              (760, 779, "perinatal"), (780, 799, "symptoms"), (800, 999, "injury")]
    for lo, hi, name in bounds:
        if lo <= v <= hi:
            return name
    return "other"


def load_and_prepare(conf: dict) -> pd.DataFrame:
    """Load raw data and derive labels plus the raw fields the encoder needs."""
    df = pd.read_csv(RAW_CSV, na_values=["?"], keep_default_na=False, low_memory=False)
    fc = conf["features"]

    for c in fc["diag_columns"]:
        df[c] = df[c].astype(str).map(_icd9_chapter)

    # Rare specialties collapsed (see medical_specialty_top_k in config).
    k = fc["medical_specialty_top_k"]
    top = df["medical_specialty"].value_counts().head(k).index
    df["medical_specialty"] = df["medical_specialty"].where(
        df["medical_specialty"].isin(top), other="Other")

    df["y_lt30"] = (df["readmitted"] == "<30").astype(int)
    df["y_merged"] = (df["readmitted"] != "NO").astype(int)

    # Observability marker from Phase 0.1: a row can only carry a positive label
    # if the patient has a later encounter in the extract. Tracked per split so
    # the temporal regime's label behaviour is interpretable rather than magic.
    mx = df.groupby("patient_nbr")["encounter_id"].transform("max")
    df["is_last_encounter"] = (df["encounter_id"] == mx).astype(int)
    return df


def encode(train: pd.DataFrame, others: dict[str, pd.DataFrame],
           conf: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    """
    Fixed feature pipeline, FIT ON TRAIN ONLY.

    Categories are learned from train; unseen levels in val/test map to -1,
    which is the honest representation of "a level the model never saw".
    """
    fc = conf["features"]
    cat_cols = fc["categorical"] + fc["diag_columns"]

    def _base(d: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=d.index)
        for c in fc["numeric"]:
            out[c] = d[c].astype(float)
        for c in fc["missing_indicator_columns"]:
            out[f"{c}_missing"] = d[c].isna().astype(int)
        for c in fc["none_indicator_columns"]:
            out[f"{c}_measured"] = (d[c].astype(str) != "None").astype(int)
        return out

    cats = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in cat_cols}

    def _full(d: pd.DataFrame) -> pd.DataFrame:
        out = _base(d)
        for c in cat_cols:
            out[c] = cats[c].get_indexer(d[c].astype(str)).astype(float)
        return out

    X_train = _full(train)
    X_others = {k: _full(v) for k, v in others.items()}
    return X_train, X_others, list(X_train.columns)


# ══════════════════════════════════════════════════════════════════════════
# Split regimes
# ══════════════════════════════════════════════════════════════════════════

def split_entry_cohort(df: pd.DataFrame, conf: dict, seed: int) -> dict:
    """The repository's existing split — patients ordered by first encounter_id."""
    order = df.groupby("patient_nbr")["encounter_id"].min().sort_values().index
    return _cut_patients(df, order, conf, extra={"deterministic": True})


def split_random(df: pd.DataFrame, conf: dict, seed: int) -> dict:
    """Negative control — patient-level random assignment, same proportions."""
    rng = np.random.default_rng(seed)
    pats = df["patient_nbr"].unique()
    order = pd.Index(rng.permutation(pats))
    return _cut_patients(df, order, conf, extra={"deterministic": False})


def split_temporal(df: pd.DataFrame, conf: dict, seed: int) -> dict:
    """
    TRUE temporal split — chronological cut of ENCOUNTERS, leakage-controlled.

    Newly defensible: Phase 0.1 verified encounter_id chronology against
    troglitazone withdrawal (2000-03-21), ICD-9 V85 introduction (2005-10-01)
    and the rosiglitazone safety changepoint (2007-05-21), with the anchor-
    derived calendar map reproducing the dataset's own 30-day readmission
    boundary to within days.

    A chronological cut necessarily straddles patients, so patients appearing in
    an earlier split are DROPPED from later splits. That preserves the
    prospective reading — train on everything up to time T, evaluate on patients
    first seen after T — and keeps the evaluation leakage-free.
    """
    r = conf["split_ratios"]
    d = df.sort_values("encounter_id")
    n = len(d)
    i1, i2 = int(n * r["train"]), int(n * (r["train"] + r["val"]))
    tr, va, te = d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]

    p_tr = set(tr["patient_nbr"])
    va_clean = va[~va["patient_nbr"].isin(p_tr)]
    p_seen = p_tr | set(va_clean["patient_nbr"]) | set(va["patient_nbr"])
    te_clean = te[~te["patient_nbr"].isin(p_seen)]

    out = {"train": tr, "val": va_clean, "test": te_clean,
           "meta": {"deterministic": True,
                    "val_rows_dropped_for_leakage": int(len(va) - len(va_clean)),
                    "test_rows_dropped_for_leakage": int(len(te) - len(te_clean)),
                    "val_rows_kept": int(len(va_clean)),
                    "test_rows_kept": int(len(te_clean))}}
    _assert_disjoint(out)
    return out


def _cut_patients(df: pd.DataFrame, order: pd.Index, conf: dict, extra: dict) -> dict:
    r = conf["split_ratios"]
    n = len(order)
    n_tr, n_va = int(np.floor(n * r["train"])), int(np.floor(n * r["val"]))
    sets = {"train": set(order[:n_tr]),
            "val": set(order[n_tr:n_tr + n_va]),
            "test": set(order[n_tr + n_va:])}
    out = {k: df[df["patient_nbr"].isin(v)] for k, v in sets.items()}
    out["meta"] = extra
    _assert_disjoint(out)
    return out


def _assert_disjoint(splits: dict) -> None:
    """Patient-level leakage guard. Encoded as an assertion, not as prose."""
    a, b, c = (set(splits[k]["patient_nbr"]) for k in ("train", "val", "test"))
    assert not (a & b) and not (a & c) and not (b & c), "patient leakage in split"


# ══════════════════════════════════════════════════════════════════════════
# Model + detector run
# ══════════════════════════════════════════════════════════════════════════

def _fit_model(X: pd.DataFrame, y: np.ndarray, conf: dict, seed: int):
    params = dict(conf["model"])
    return lgb.LGBMClassifier(random_state=seed, **params).fit(X, y)


def _pick_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """
    F1-max threshold, selected on a held-out slice of TRAIN.

    NOT on val: val is the drift reference window, and fitting the threshold on
    the reference window then measuring degradation against it is circular
    (audit F13). Decontaminated here from the outset.
    """
    best, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (p >= t).astype(int)
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return best


def run_detectors(val_X: pd.DataFrame, val_y: np.ndarray,
                  test_X: pd.DataFrame, test_y: np.ndarray,
                  feat_cols: list[str], predict_fn, threshold: float,
                  tag: str) -> dict:
    """Run the shipped detectors and return the signals they produce."""
    ref = val_X.copy();  ref["readmitted_binary"] = val_y
    prod = test_X.copy(); prod["readmitted_binary"] = test_y

    dd = DataDriftDetector(reference_name="val")
    dd.fit(ref, feat_cols)
    drift_df = dd.detect(prod, production_name="test")
    dd_summary = dd.summary_["test"]      # AlertEngine expects the inner summary

    cdd = ConceptDriftDetector(model_name=tag)
    cd_report = cdd.detect(ref_df=ref, prod_df=prod, feat_cols=feat_cols,
                           predict_fn=predict_fn, ref_name=f"{tag}_val",
                           prod_name=f"{tag}_test", threshold=threshold,
                           n_windows=10)

    alert = AlertEngine(model_name=tag).run(
        concept_report=cd_report, data_drift_summary=dd_summary,
        feature_report={}, ref_name=f"{tag}_val", prod_name=f"{tag}_test")

    return {
        "evidence": {k: bool(v) for k, v in cd_report["evidence"].items()},
        "diagnostics_not_evidence": {
            k: bool(v) for k, v in cd_report.get("diagnostics_not_evidence", {}).items()},
        "n_evidence": int(cd_report["n_evidence"]),
        "n_voting_signals": int(cd_report.get("n_voting_signals", len(SIGNALS))),
        "label_relative_change": cd_report["label_shift"].get("relative_change"),
        "label_legacy_rule_would_fire": cd_report["label_shift"].get(
            "legacy_absolute_rule", {}).get("would_fire"),
        "severity": cd_report["severity"],
        "auc_ref": cd_report["ref_metrics"]["auc"],
        "auc_prod": cd_report["prod_metrics"]["auc"],
        "auc_degradation": cd_report["auc_degradation"],
        "f1_degradation": cd_report["f1_degradation"],
        "label_delta": cd_report["label_shift"]["delta_pos_rate"],
        "pred_delta": cd_report["prediction_shift"]["delta_mean_proba"],
        "n_features": int(len(drift_df)),
        "n_drifted_features": int(drift_df["drift_detected"].sum()),
        "n_psi_critical": int((drift_df["psi_level"] == "CRITICAL").sum()),
        "n_psi_moderate": int((drift_df["psi_level"] == "MODERATE").sum()),
        "alert_status": alert.get("system_status", "UNKNOWN"),
        "n_alerts": int(alert.get("total_alerts", 0)),
        "alert_counts": alert.get("alert_counts", {}),
    }


def run_one(df: pd.DataFrame, conf: dict, regime: str, seed: int,
            split_fn, target_col: str = None) -> dict:
    """Train a model and run every detector for one (regime, seed)."""
    target_col = target_col or f"y_{conf['target']}"
    sp = split_fn(df, conf, seed)
    train, val, test = sp["train"], sp["val"], sp["test"]

    X_train, X_other, feat_cols = encode(train, {"val": val, "test": test}, conf)
    y_train = train[target_col].to_numpy()

    # Threshold-selection slice, held out of model fitting, split by PATIENT.
    rng = np.random.default_rng(seed)
    pats = train["patient_nbr"].unique()
    n_hold = int(len(pats) * conf["threshold_holdout_frac"])
    hold = set(rng.permutation(pats)[:n_hold])
    m_hold = train["patient_nbr"].isin(hold).to_numpy()

    model = _fit_model(X_train[~m_hold], y_train[~m_hold], conf, seed)
    p_hold = model.predict_proba(X_train[m_hold])[:, 1]
    threshold = _pick_threshold(y_train[m_hold], p_hold)

    def predict_fn(X):
        return model.predict_proba(pd.DataFrame(X, columns=feat_cols))[:, 1]

    res = run_detectors(X_other["val"], val[target_col].to_numpy(),
                        X_other["test"], test[target_col].to_numpy(),
                        feat_cols, predict_fn, threshold, f"{regime}_s{seed}")
    res.update({
        "regime": regime, "seed": seed, "threshold": threshold,
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "prevalence_train": float(train[target_col].mean()),
        "prevalence_val": float(val[target_col].mean()),
        "prevalence_test": float(test[target_col].mean()),
        "share_last_encounter_val": float(val["is_last_encounter"].mean()),
        "share_last_encounter_test": float(test["is_last_encounter"].mean()),
        "split_meta": sp["meta"],
    })
    return res, (model, X_other, val, test, feat_cols, threshold, target_col)


# ══════════════════════════════════════════════════════════════════════════
# Phase 0.3 — synthetic shift constructors
# ══════════════════════════════════════════════════════════════════════════

def shift_covariate(X: pd.DataFrame, y: np.ndarray, driver: str,
                    alpha: float, rng: np.random.Generator):
    """
    PURE covariate shift. Importance-resamples REAL rows so P(X) moves while
    P(Y|X) is untouched — no feature value is edited and no label is altered.
    """
    z = X[driver].to_numpy(dtype=float)
    z = (z - z.mean()) / (z.std() + 1e-9)
    w = np.exp(alpha * z)
    w = w / w.sum()
    idx = rng.choice(len(X), size=len(X), replace=True, p=w)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def shift_label(X: pd.DataFrame, y: np.ndarray, delta: float,
                rng: np.random.Generator):
    """
    PURE label shift. Resamples positives and negatives at different rates to
    move P(Y) by `delta`, preserving P(X|Y) exactly.
    """
    n = len(y)
    p_target = float(np.clip(y.mean() + delta, 0.01, 0.99))
    n_pos = int(round(n * p_target))
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    take = np.concatenate([rng.choice(pos_idx, n_pos, replace=True),
                           rng.choice(neg_idx, n - n_pos, replace=True)])
    rng.shuffle(take)
    return X.iloc[take].reset_index(drop=True), y[take]


def shift_concept(X: pd.DataFrame, y: np.ndarray, subgroup: np.ndarray,
                  frac: float, rng: np.random.Generator):
    """
    PURE concept shift. Flips the label mechanism inside a fixed subgroup,
    swapping EQUAL counts of 0->1 and 1->0 so P(X) is preserved exactly and
    marginal P(Y) is held approximately constant — isolating a P(Y|X) change
    from a label shift.
    """
    y2 = y.copy()
    sg_pos = np.flatnonzero(subgroup & (y == 1))
    sg_neg = np.flatnonzero(subgroup & (y == 0))
    k = int(round(frac * min(len(sg_pos), len(sg_neg))))
    if k > 0:
        y2[rng.choice(sg_pos, k, replace=False)] = 0
        y2[rng.choice(sg_neg, k, replace=False)] = 1
    return X, y2


# ══════════════════════════════════════════════════════════════════════════
# Phases
# ══════════════════════════════════════════════════════════════════════════

def phase_0_2(df: pd.DataFrame, conf: dict) -> dict:
    """Random-split falsification control plus the two real deterministic splits."""
    logger.info("=" * 78)
    logger.info("PHASE 0.2 — split regimes (entry_cohort / random / temporal)")
    logger.info("=" * 78)

    runs = {"entry_cohort": [], "random": [], "temporal": []}
    fns = {"entry_cohort": split_entry_cohort, "random": split_random,
           "temporal": split_temporal}
    n_seeds = {"entry_cohort": conf["n_seeds_deterministic"],
               "random": conf["n_seeds_random"],
               "temporal": conf["n_seeds_deterministic"]}

    for regime, fn in fns.items():
        logger.info("-" * 62)
        logger.info(f"REGIME: {regime}   ({n_seeds[regime]} seeds)")
        for seed in range(n_seeds[regime]):
            res, _ = run_one(df, conf, regime, seed, fn)
            runs[regime].append(res)
            if seed == 0:
                logger.info(f"  n_train={res['n_train']:,} n_val={res['n_val']:,} "
                            f"n_test={res['n_test']:,} | prev "
                            f"{res['prevalence_train']:.4f}/{res['prevalence_val']:.4f}/"
                            f"{res['prevalence_test']:.4f} | thr={res['threshold']:.3f}")
                if regime == "temporal":
                    logger.info(f"  leakage control dropped "
                                f"{res['split_meta']['val_rows_dropped_for_leakage']:,} val and "
                                f"{res['split_meta']['test_rows_dropped_for_leakage']:,} test rows")
            logger.info(f"  seed {seed:>2}: {res['n_evidence']}/8 signals, "
                        f"{res['severity']:<8} alert={res['alert_status']:<8} "
                        f"AUC {res['auc_ref']:.4f}->{res['auc_prod']:.4f} "
                        f"({res['auc_degradation']:+.4f}) psi_crit={res['n_psi_critical']}")

    summary = {r: _summarise(v, conf) for r, v in runs.items()}

    logger.info("-" * 62)
    logger.info("FIRING RATES BY REGIME")
    logger.info(f"  {'signal':<22} " + "".join(f"{r:>15}" for r in runs))
    for s in SIGNALS:
        row = "".join(f"{summary[r]['firing_rates'][s]:>15.2f}" for r in runs)
        logger.info(f"  {s:<22} " + row)
    logger.info(f"  {'n_evidence mean':<22} " +
                "".join(f"{summary[r]['n_evidence_mean']:>15.2f}" for r in runs))

    mis = summary["random"]["miscalibrated_signals"]
    logger.info("-" * 62)
    logger.info(f"MIS-CALIBRATED under the negative control "
                f"(fires in >{conf['miscalibration_threshold']:.0%} of no-drift seeds): "
                f"{mis if mis else 'none'}")

    out = {
        "phase": "0.2",
        "regimes": {r: {"description": conf["regimes"][r]["description"].strip(),
                        "n_seeds": n_seeds[r], "summary": summary[r],
                        "runs": runs[r]} for r in runs},
        "miscalibration_threshold": conf["miscalibration_threshold"],
        "interpretation": _interpret_0_2(summary, conf),
    }
    with open(REPORTS_DIR / "regime_random.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Report written: regime_random.json")
    return out


def _summarise(runs: list[dict], conf: dict) -> dict:
    n = len(runs)
    rates = {s: float(np.mean([r["evidence"][s] for r in runs])) for s in SIGNALS}
    thr = conf["miscalibration_threshold"]
    def _ms(key):
        v = np.array([r[key] for r in runs], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if n > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max())}
    diag_rates = {s: float(np.mean([r.get("diagnostics_not_evidence", {}).get(s, False)
                                    for r in runs])) for s in DIAGNOSTIC_SIGNALS}
    return {
        "n_runs": n,
        "firing_rates": rates,
        "diagnostic_firing_rates_NOT_EVIDENCE": diag_rates,
        "legacy_label_rule_firing_rate": float(np.mean(
            [bool(r.get("label_legacy_rule_would_fire")) for r in runs])),
        "family_firing_rates": {
            fam: float(np.mean([max(runs[i]["evidence"][s] for s in sigs)
                                for i in range(n)]))
            for fam, sigs in conf["signal_families"].items()},
        "n_evidence_mean": float(np.mean([r["n_evidence"] for r in runs])),
        "n_evidence_std": float(np.std([r["n_evidence"] for r in runs], ddof=1)) if n > 1 else 0.0,
        "severity_counts": pd.Series([r["severity"] for r in runs]).value_counts().to_dict(),
        "alert_status_counts": pd.Series([r["alert_status"] for r in runs]).value_counts().to_dict(),
        "auc_ref": _ms("auc_ref"), "auc_prod": _ms("auc_prod"),
        "auc_degradation": _ms("auc_degradation"),
        "n_psi_critical": _ms("n_psi_critical"),
        "n_drifted_features": _ms("n_drifted_features"),
        "miscalibrated_signals": [s for s, v in rates.items() if v > thr],
    }


def _interpret_0_2(summary: dict, conf: dict) -> dict:
    rnd = summary["random"]
    return {
        "negative_control_result": (
            f"Under a no-drift random split the detector fires "
            f"{rnd['n_evidence_mean']:.2f} +/- {rnd['n_evidence_std']:.2f} of 8 signals "
            f"on average; severities {rnd['severity_counts']}, "
            f"alert statuses {rnd['alert_status_counts']}."),
        "miscalibrated_signals": rnd["miscalibrated_signals"],
        "required_statement": (
            "Any signal firing under random splitting is detector "
            "mis-calibration, not evidence of drift. The original 8/8 CRITICAL "
            "result must be discounted by exactly this baseline."),
    }


def phase_0_3(df: pd.DataFrame, conf: dict) -> dict:
    """Synthetic positive controls with detection-power curves."""
    logger.info("=" * 78)
    logger.info("PHASE 0.3 — synthetic positive controls")
    logger.info("=" * 78)

    syn = conf["synthetic"]
    n_seeds = conf["n_seeds_synthetic"]
    grids = {"covariate_shift": syn["covariate_shift"]["alphas"],
             "label_shift": syn["label_shift"]["deltas"],
             "concept_shift": syn["concept_shift"]["flip_fractions"]}
    results = {k: {str(m): [] for m in v} for k, v in grids.items()}

    for seed in range(n_seeds):
        # One model per seed on the random (no-drift) base; every shift is then
        # applied to that seed's test set, so shift magnitude is the only thing
        # that varies within a seed.
        _, ctx = run_one(df, conf, "synthetic_base", seed, split_random)
        model, X_other, val, test, feat_cols, threshold, tcol = ctx
        rng = np.random.default_rng(1000 + seed)

        def predict_fn(X):
            return model.predict_proba(pd.DataFrame(X, columns=feat_cols))[:, 1]

        val_X, val_y = X_other["val"], val[tcol].to_numpy()
        test_X, test_y = X_other["test"].reset_index(drop=True), test[tcol].to_numpy()
        sub_col = syn["concept_shift"]["subgroup_column"]
        subgroup = (test[sub_col].to_numpy() > 0)

        for kind, grid in grids.items():
            for mag in grid:
                if kind == "covariate_shift":
                    Xs, ys = shift_covariate(test_X, test_y,
                                             syn["covariate_shift"]["driver"], mag, rng)
                elif kind == "label_shift":
                    Xs, ys = shift_label(test_X, test_y, mag, rng)
                else:
                    Xs, ys = shift_concept(test_X, test_y, subgroup, mag, rng)
                res = run_detectors(val_X, val_y, Xs, ys, feat_cols, predict_fn,
                                    threshold, f"{kind}_{mag}_s{seed}")
                res.update({"magnitude": float(mag), "seed": seed, "kind": kind})
                results[kind][str(mag)].append(res)
        logger.info(f"  seed {seed:>2} complete ({sum(len(g) for g in grids.values())} shift points)")

    power = {}
    for kind, grid in grids.items():
        power[kind] = {
            "magnitudes": [float(m) for m in grid],
            "curves": {s: [float(np.mean([r["evidence"][s]
                                          for r in results[kind][str(m)]]))
                           for m in grid] for s in SIGNALS},
            "n_evidence_mean": [float(np.mean([r["n_evidence"]
                                               for r in results[kind][str(m)]]))
                                for m in grid],
        }
        logger.info("-" * 62)
        logger.info(f"POWER CURVE — {kind}   magnitudes {power[kind]['magnitudes']}")
        for s in SIGNALS:
            logger.info(f"  {s:<22} " +
                        "".join(f"{v:>8.2f}" for v in power[kind]["curves"][s]))

    diagnostic = _diagnosticity(power, grids)
    logger.info("-" * 62)
    logger.info("DIAGNOSTICITY (which signal identifies which mechanism)")
    for s, v in diagnostic.items():
        logger.info(f"  {s:<22} responds to: {v['responds_to'] or 'nothing'}  "
                    f"-> {v['verdict']}")

    out = {"phase": "0.3", "n_seeds": n_seeds,
           "construction_notes": {k: syn[k]["note"].strip() for k in grids},
           "power": power, "diagnosticity": diagnostic,
           "raw": {k: {m: v for m, v in g.items()} for k, g in results.items()}}
    with open(REPORTS_DIR / "regime_synthetic.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Report written: regime_synthetic.json")
    return out


def _diagnosticity(power: dict, grids: dict) -> dict:
    """
    A signal that fires under every mechanism is a general alarm, not evidence
    for a specific one, and the plan requires it to be described that way.
    Response = firing rate at maximum magnitude exceeds that at zero by >= 0.25.
    """
    out = {}
    for s in SIGNALS:
        responds = []
        for kind in grids:
            c = power[kind]["curves"][s]
            if c[-1] - c[0] >= 0.25:
                responds.append(kind)
        out[s] = {
            "responds_to": responds,
            "verdict": ("DIAGNOSTIC" if len(responds) == 1 else
                        "GENERAL_ALARM" if len(responds) >= 3 else
                        "PARTIAL" if len(responds) == 2 else "UNRESPONSIVE"),
            "baseline_firing_rate": {k: power[k]["curves"][s][0] for k in grids},
            "max_magnitude_firing_rate": {k: power[k]["curves"][s][-1] for k in grids},
        }
    return out


def phase_0_4(r02: dict, r03: dict, conf: dict) -> dict:
    """Consolidate everything into the regime x signal matrix."""
    logger.info("=" * 78)
    logger.info("PHASE 0.4 — regime x signal matrix")
    logger.info("=" * 78)

    cols = {
        "entry_cohort": r02["regimes"]["entry_cohort"]["summary"]["firing_rates"],
        "random_control": r02["regimes"]["random"]["summary"]["firing_rates"],
        "temporal": r02["regimes"]["temporal"]["summary"]["firing_rates"],
        "covariate_shift_max": {s: r03["power"]["covariate_shift"]["curves"][s][-1]
                                for s in SIGNALS},
        "label_shift_max": {s: r03["power"]["label_shift"]["curves"][s][-1]
                            for s in SIGNALS},
        "concept_shift_max": {s: r03["power"]["concept_shift"]["curves"][s][-1]
                              for s in SIGNALS},
    }
    fam_of = {s: fam for fam, sigs in conf["signal_families"].items() for s in sigs}
    rows = [{"family": fam_of[s], "signal": s,
             **{c: cols[c][s] for c in cols}} for s in SIGNALS]
    matrix = pd.DataFrame(rows).sort_values(["family", "signal"]).reset_index(drop=True)
    matrix.to_csv(REPORTS_DIR / "regime_matrix.csv", index=False)

    logger.info(f"  {'family':<13}{'signal':<22}" + "".join(f"{c:>21}" for c in cols))
    for _, r in matrix.iterrows():
        logger.info(f"  {r['family']:<13}{r['signal']:<22}" +
                    "".join(f"{r[c]:>21.2f}" for c in cols))

    answer = _answer_what_was_the_drift(r02, r03, conf)
    logger.info("-" * 62)
    for k, v in answer.items():
        logger.info(f"  {k}: {v}")

    out = {"phase": "0.4", "matrix": matrix.to_dict("records"),
           "columns": list(cols), "families": conf["signal_families"],
           "what_was_the_original_drift": answer}
    with open(REPORTS_DIR / "regime_matrix.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Reports written: regime_matrix.csv / regime_matrix.json")
    return out, matrix


def _answer_what_was_the_drift(r02: dict, r03: dict, conf: dict) -> dict:
    """
    Phase 0.4 requires a plain answer to 'what was the original drift actually?'
    — stated even where it weakens the original claim.
    """
    ec = r02["regimes"]["entry_cohort"]["summary"]
    rnd = r02["regimes"]["random"]["summary"]
    tmp = r02["regimes"]["temporal"]["summary"]
    excess = {s: ec["firing_rates"][s] - rnd["firing_rates"][s] for s in SIGNALS}
    real = [s for s, v in excess.items() if v > 0.5]
    baseline = [s for s, v in rnd["firing_rates"].items() if v > conf["miscalibration_threshold"]]
    return {
        "entry_cohort_signals_mean": round(ec["n_evidence_mean"], 3),
        "random_control_signals_mean": round(rnd["n_evidence_mean"], 3),
        "temporal_signals_mean": round(tmp["n_evidence_mean"], 3),
        "signals_firing_above_the_no_drift_baseline": real,
        "signals_that_fire_even_with_no_drift": baseline,
        "entry_cohort_auc_degradation": round(ec["auc_degradation"]["mean"], 4),
        "random_auc_degradation": round(rnd["auc_degradation"]["mean"], 4),
        "temporal_auc_degradation": round(tmp["auc_degradation"]["mean"], 4),
    }


def summarise_sequential_detectors() -> dict:
    """
    Post-hoc diagnosis of the sequential family, read from the per-run detector
    artifacts in SCRATCH_DIR.

    WHY THIS EXISTS: cusum_alarm and ph_alarm fire under the NO-DRIFT control
    and are unresponsive to every synthetic shift, so the matrix alone shows
    they carry no information but not WHY. Both detectors estimate their
    reference mean from the first MIN_WINDOW_SIZE (=200) samples of the
    concatenated error stream, an arbitrary constant with no ARL calibration
    (audit F25). This extracts the internals that show the consequence.
    """
    import glob
    out = {}
    for reg in ("entry_cohort", "random", "temporal"):
        rows = []
        for f in sorted(glob.glob(str(SCRATCH_DIR / f"concept_drift_{reg}_s*.json"))):
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            rows.append({"cusum_alarms": d["cusum"]["n_alarms"],
                         "ph_alarms": d["page_hinkley"]["n_alarms"],
                         "mean_ref": d["page_hinkley"]["mean_ref"],
                         "mean_prod": d["page_hinkley"]["mean_prod"]})
        if not rows:
            continue
        t = pd.DataFrame(rows)
        out[reg] = {
            "n_runs": int(len(t)),
            "cusum_alarms_mean": float(t["cusum_alarms"].mean()),
            "cusum_alarms_min": int(t["cusum_alarms"].min()),
            "ph_alarms_mean": float(t["ph_alarms"].mean()),
            "mean_ref": float(t["mean_ref"].mean()),
            "mean_prod": float(t["mean_prod"].mean()),
            "ref_exceeds_prod": bool(t["mean_ref"].mean() > t["mean_prod"].mean()),
        }
    out["diagnosis"] = {
        "cusum": ("fires in 100% of runs in EVERY regime including the no-drift "
                  "control, with 110-142 alarms per run. With threshold=5.0 and "
                  "delta=0.005 against a mean absolute error of ~0.18, the "
                  "statistic is saturated: it is a constant TRUE and carries zero "
                  "information about drift."),
        "page_hinkley": ("fires or stays silent according to whether the first 200 "
                         "samples of the stream happen to have HIGHER or LOWER mean "
                         "error than the remainder. Where mean_ref < mean_prod it "
                         "alarms; in the temporal regime mean_ref (0.185) exceeds "
                         "mean_prod (0.172), the cumulative sum drifts negative and "
                         "it never alarms. Its output is set by an arbitrary "
                         "MIN_WINDOW_SIZE constant, not by drift."),
        "consequence": ("the entire 'sequential' family — 2 of the 8 advertised "
                        "evidence signals — must be excluded from any evidence "
                        "count until ARL-calibrated. Reporting them as 2 of 8 "
                        "inflates every severity verdict."),
    }
    return out


# ══════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════

def _fig_firing_rates(r02: dict, conf: dict, path: Path) -> None:
    regs = ["entry_cohort", "random", "temporal"]
    x = np.arange(len(SIGNALS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    for i, (r, c) in enumerate(zip(regs, [C_MAIN, C_NULL, C_OK])):
        vals = [r02["regimes"][r]["summary"]["firing_rates"][s] for s in SIGNALS]
        ax.bar(x + (i - 1) * w, vals, w, label=r, color=c)
    ax.axhline(conf["miscalibration_threshold"], color=C_WARN, ls="--", lw=1.3,
               label=f"mis-calibration threshold ({conf['miscalibration_threshold']:.0%})")
    ax.set_xticks(x)
    ax.set_xticklabels(SIGNALS, rotation=25, ha="right")
    ax.set_ylabel("firing rate across seeds")
    ax.set_ylim(0, 1.08)
    ax.set_title("Phase 0.2 — signal firing rate by split regime\n"
                 "'random' is the negative control: anything it fires is mis-calibration, not drift")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_power(r03: dict, path: Path) -> None:
    kinds = list(r03["power"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for ax, kind in zip(axes, kinds):
        mags = r03["power"][kind]["magnitudes"]
        for i, s in enumerate(SIGNALS):
            ax.plot(mags, r03["power"][kind]["curves"][s], "o-", ms=4, lw=1.4,
                    color=cmap(i % 10), label=s)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("shift magnitude")
        ax.set_ylabel("firing rate")
        ax.set_title(f"{kind}\n(detection power)")
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Phase 0.3 — detection-power curves per signal per shift mechanism", y=1.04)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _fig_matrix(matrix: pd.DataFrame, cols: list[str], path: Path) -> None:
    M = matrix[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right")
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels([f"[{r['family'][:4]}] {r['signal']}" for _, r in matrix.iterrows()])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 else "black")
    ax.set_title("Phase 0.4 — regime x signal matrix (firing rate)\n"
                 "column 2 is the no-drift control: it is the baseline every other column "
                 "must be read against")
    fig.colorbar(im, ax=ax, shrink=0.8, label="firing rate")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_split_regimes(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    logger.info("=" * 78)
    logger.info("DriftSentinel — Phases 0.2 / 0.3 / 0.4")
    logger.info(f"target = {conf['target']}   detectors imported from src/drift/ as shipped")
    logger.info("=" * 78)

    df = load_and_prepare(conf)
    logger.info(f"Rows {len(df):,} | patients {df['patient_nbr'].nunique():,} | "
                f"prevalence(<30) {df['y_lt30'].mean():.4f} | "
                f"prevalence(merged) {df['y_merged'].mean():.4f}")

    r02 = phase_0_2(df, conf)
    r03 = phase_0_3(df, conf)
    r04, matrix = phase_0_4(r02, r03, conf)

    figs = {"firing": FIGURE_DIR / "40_regime_firing_rates.png",
            "power": FIGURE_DIR / "41_detection_power_curves.png",
            "matrix": FIGURE_DIR / "42_regime_signal_matrix.png"}
    _fig_firing_rates(r02, conf, figs["firing"])
    _fig_power(r03, figs["power"])
    _fig_matrix(matrix, r04["columns"], figs["matrix"])
    logger.info("Figures: " + ", ".join(p.name for p in figs.values()))

    repro = {"seed": conf["seed"], "python": platform.python_version(),
             "numpy": np.__version__, "pandas": pd.__version__,
             "scipy": scipy.__version__, "lightgbm": lgb.__version__,
             "matplotlib": matplotlib.__version__, "platform": platform.platform()}
    for name, rep in [("regime_random.json", r02), ("regime_synthetic.json", r03),
                      ("regime_matrix.json", r04)]:
        rep["reproducibility"] = repro
        rep["figures"] = {k: str(v.relative_to(ROOT).as_posix()) for k, v in figs.items()}
        with open(REPORTS_DIR / name, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, default=str)

    logger.info("=" * 78)
    logger.info("Phases 0.2 / 0.3 / 0.4 complete")
    logger.info("=" * 78)
    return {"phase_0_2": r02, "phase_0_3": r03, "phase_0_4": r04}


if __name__ == "__main__":
    out = run_split_regimes()
    a = out["phase_0_4"]["what_was_the_original_drift"]
    print("\nentry_cohort signals :", a["entry_cohort_signals_mean"], "/8")
    print("random control       :", a["random_control_signals_mean"], "/8")
    print("temporal split       :", a["temporal_signals_mean"], "/8")
    print("fires with NO drift  :", a["signals_that_fire_even_with_no_drift"])
