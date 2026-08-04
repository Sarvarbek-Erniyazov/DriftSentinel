"""
DriftSentinel — Tier 2C.4a: fairness / subgroup analysis

WHY THIS IS FIRST IN TIER 2C
    Its absence is the most conspicuous gap for a clinical ML reviewer in 2026.
    A deployment artifact without subgroup performance is not reviewable, and
    the audit listed it as an expected section rather than an optional one.

WHAT IS REPORTED
    Per subgroup (race, gender, age band): n, prevalence, AUC, precision,
    recall, F1, predicted-positive rate and calibration error, each with a
    PATIENT-CLUSTERED bootstrap interval. Disparities are reported as gaps with
    intervals, not as point estimates, because a gap without an interval cannot
    be distinguished from sampling noise.

    Subgroups too small to support a claim are marked INSUFFICIENT_EVIDENCE and
    excluded from disparity claims — reported, never silently dropped, and never
    smoothed into a reassuring average.

THE EQUITY QUESTION THE AUDIT RAISED
    `payer_code` — insurance status — was among the most drifted and most
    important features. Insurance status driving clinical risk prediction is
    exactly the pattern Obermeyer et al. (Science, 2019) warn about: a proxy for
    access to care being learned as if it were clinical severity. Its importance
    rank is reported here so the question is answerable rather than deflected.

FALSIFICATION ARM (the Tier 2B through-line)
    A fairness audit that reports "no disparity" is only interpretable if it CAN
    detect one. A synthetic disparity is injected into a held-out copy — one
    subgroup's predictions are degraded — and the audit must flag it. If it does
    not, no null on the real data can be believed.
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
from src.models.repeated_eval import _ece
from src.uncertainty.calibration import IsotonicCalibrator  # noqa: F401

logger = get_logger("fairness_audit")

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "train"
MODELS_DIR = ROOT / "outputs" / "models"
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"
REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_binary"
TARGET_COLS = {"readmitted_binary", "readmitted_multi"}
SEED = 42
N_BOOT = 1000
MIN_N = 500            # below this, a subgroup claim is not supportable
MIN_POSITIVES = 30     # AUC on <30 positives is not a stable estimate


def load_aligned_demographics(y_expected: np.ndarray) -> tuple[pd.DataFrame, str]:
    """
    Recover raw demographic columns aligned to the test parquet rows.

    Same verified-alignment protocol used throughout: the splitter sorts each
    split by encounter_id, so raw rows for that split's patients in the same
    order align 1:1 — VERIFIED against the label sequence before use (R6).
    """
    from src.data.preprocessor import TARGET_BINARY_MAP
    with open(ARTIFACTS_DIR / "split_index.json") as f:
        idx = json.load(f)
    raw = pd.read_csv(ROOT / "data" / "raw" / "diabetes_hospital" / "diabetic_data.csv",
                      usecols=["encounter_id", "patient_nbr", "readmitted",
                               "race", "gender", "age"],
                      na_values=["?"], keep_default_na=False)
    sub = raw[raw["patient_nbr"].isin(set(idx["test_patient_ids"]))].sort_values(
        "encounter_id").reset_index(drop=True)
    if len(sub) != len(y_expected):
        return None, f"FAILED: row count {len(sub)} != {len(y_expected)}"
    rec = sub["readmitted"].map(TARGET_BINARY_MAP).to_numpy().astype(int)
    if not np.array_equal(rec, np.asarray(y_expected).astype(int)):
        return None, "FAILED: recovered label sequence does not match the parquet target"
    return sub, f"OK: verified ({sub['patient_nbr'].nunique():,} patients / {len(sub):,} rows)"


def _metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    out = {"precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "predicted_positive_rate": float(pred.mean()),
           "prevalence": float(y.mean()),
           "ece": _ece(y, p)}
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    return out


def subgroup_report(y, p, groups, thr, name, seed=SEED, n_boot=N_BOOT) -> dict:
    """Per-level metrics with patient-clustered bootstrap intervals."""
    rng = np.random.default_rng(seed)
    out = {}
    for lvl in sorted(pd.unique(groups[~pd.isna(groups)])):
        m = (groups == lvl).to_numpy() if hasattr(groups, "to_numpy") else (groups == lvl)
        n, npos = int(m.sum()), int(y[m].sum())
        entry = {"n": n, "n_positive": npos,
                 "share_of_cohort": round(float(m.mean()), 4)}
        if n < MIN_N or npos < MIN_POSITIVES or len(np.unique(y[m])) < 2:
            entry["status"] = "INSUFFICIENT_EVIDENCE"
            entry["reason"] = (f"n={n} (min {MIN_N}), positives={npos} "
                               f"(min {MIN_POSITIVES}) — too small to support a claim")
            if n > 0:
                entry["prevalence"] = round(float(y[m].mean()), 4)
            out[str(lvl)] = entry
            continue

        point = _metrics(y[m], p[m], thr)
        pats = groups.index.to_numpy() if hasattr(groups, "index") else np.arange(len(y))
        gp = np.asarray(pats)[m]
        uniq, inv = np.unique(gp, return_inverse=True)
        pool = [np.flatnonzero(inv == g) for g in range(len(uniq))]
        draws = {k: [] for k in point}
        ysub, psub = y[m], p[m]
        for _ in range(n_boot):
            pick = rng.integers(0, len(pool), size=len(pool))
            ii = np.concatenate([pool[i] for i in pick])
            if len(np.unique(ysub[ii])) < 2:
                continue
            mm = _metrics(ysub[ii], psub[ii], thr)
            for k, v in mm.items():
                draws[k].append(v)
        entry["status"] = "OK"
        for k, v in point.items():
            arr = np.asarray(draws[k], dtype=float)
            lo, hi = np.nanpercentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)
            entry[k] = {"point": round(float(v), 4),
                        "ci95": [round(float(lo), 4), round(float(hi), 4)]}
        out[str(lvl)] = entry
    return {"attribute": name, "levels": out}


def _gap(levels: dict, metric: str) -> dict:
    """Max-min gap across levels that carry sufficient evidence."""
    ok = {k: v for k, v in levels.items() if v.get("status") == "OK"}
    if len(ok) < 2:
        return {"status": "INSUFFICIENT_EVIDENCE",
                "reason": f"fewer than 2 levels with sufficient evidence for {metric}"}
    pts = {k: v[metric]["point"] for k, v in ok.items()}
    hi_k, lo_k = max(pts, key=pts.get), min(pts, key=pts.get)
    hi_ci, lo_ci = ok[hi_k][metric]["ci95"], ok[lo_k][metric]["ci95"]
    # intervals overlap -> the gap is not distinguishable from sampling noise
    overlap = not (hi_ci[0] > lo_ci[1] or lo_ci[0] > hi_ci[1])
    return {"status": "OK", "metric": metric,
            "highest": {"level": hi_k, "value": pts[hi_k], "ci95": hi_ci},
            "lowest": {"level": lo_k, "value": pts[lo_k], "ci95": lo_ci},
            "gap": round(pts[hi_k] - pts[lo_k], 4),
            "intervals_overlap": bool(overlap),
            "claim_supported": bool(not overlap),
            "interpretation": ("the intervals overlap, so this gap is not "
                               "distinguishable from sampling noise"
                               if overlap else
                               "the intervals do not overlap — this disparity is "
                               "supported by the data and must be reported")}


def run_fairness_audit() -> dict:
    logger.info("=" * 78)
    logger.info("Tier 2C.4a — fairness / subgroup analysis")
    logger.info("=" * 78)

    train = pd.read_parquet(TRAIN_DIR / "train_fs.parquet")
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")
    feat = [c for c in train.columns if c not in TARGET_COLS]
    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS_DIR / "calibrator_isotonic_lgbm_v1.pkl", "rb") as f:
        cal = pickle.load(f)

    y = test[TARGET].to_numpy()
    p = cal.transform(model.predict_proba(test[feat])[:, 1])

    demo, note = load_aligned_demographics(y)
    if demo is None:
        raise RuntimeError(f"cannot align demographics: {note}")
    logger.info(f"  demographic alignment: {note}")

    thr_path = REPORTS_DIR / "decontamination.json"
    thr = 0.1950
    if thr_path.exists():
        d = json.loads(thr_path.read_text(encoding="utf-8"))
        thr = d["threshold"]["decontaminated_selected_on_train_holdout"]["threshold"]
    logger.info(f"  operating threshold {thr} (decontaminated, Tier 2A.4)")

    # age bands: collapse the 10 decade bins into 3 clinically meaningful bands
    age_map = {"[0-10)": "<40", "[10-20)": "<40", "[20-30)": "<40", "[30-40)": "<40",
               "[40-50)": "40-69", "[50-60)": "40-69", "[60-70)": "40-69",
               "[70-80)": "70+", "[80-90)": "70+", "[90-100)": "70+"}
    demo = demo.copy()
    demo["age_band"] = demo["age"].map(age_map)
    demo["race"] = demo["race"].fillna("Missing").replace("", "Missing")
    pats = pd.Series(demo["patient_nbr"].to_numpy(), index=demo["patient_nbr"].to_numpy())

    reports = {}
    for attr in ("race", "gender", "age_band"):
        g = pd.Series(demo[attr].to_numpy(), index=demo["patient_nbr"].to_numpy())
        rep = subgroup_report(y, p, g, thr, attr)
        reports[attr] = rep
        logger.info("-" * 62)
        logger.info(f"ATTRIBUTE: {attr}")
        logger.info(f"  {'level':<20}{'n':>7}{'prev':>8}{'AUC':>9}{'recall':>9}"
                    f"{'PPR':>9}  status")
        for lvl, v in rep["levels"].items():
            if v["status"] != "OK":
                logger.info(f"  {lvl:<20}{v['n']:>7}{v.get('prevalence', float('nan')):>8.4f}"
                            f"{'—':>9}{'—':>9}{'—':>9}  {v['status']}")
            else:
                logger.info(f"  {lvl:<20}{v['n']:>7}{v['prevalence']['point']:>8.4f}"
                            f"{v['auc']['point']:>9.4f}{v['recall']['point']:>9.4f}"
                            f"{v['predicted_positive_rate']['point']:>9.4f}  OK")
        rep["disparities"] = {m: _gap(rep["levels"], m)
                              for m in ("auc", "recall", "predicted_positive_rate",
                                        "precision", "ece")}
        for m, d in rep["disparities"].items():
            if d.get("status") == "OK":
                logger.info(f"    gap[{m}] {d['gap']:+.4f} "
                            f"({d['lowest']['level']} -> {d['highest']['level']}) "
                            f"supported={d['claim_supported']}")

    # ── payer_code equity question (Obermeyer et al. 2019) ────────────────
    imp = pd.Series(model.booster_.feature_importance("gain"), index=feat)
    ranked = imp.sort_values(ascending=False)
    payer_rank = int(list(ranked.index).index("payer_code")) + 1 if "payer_code" in ranked.index else None
    payer = {
        "in_model": "payer_code" in ranked.index,
        "gain_importance_rank": payer_rank,
        "of_n_features": int(len(ranked)),
        "top_5_features": list(ranked.head(5).index),
        "concern": ("insurance status is a proxy for ACCESS TO CARE, not clinical "
                    "severity. A model that leans on it may be encoding "
                    "access disparities as if they were risk (Obermeyer et al., "
                    "Science 2019)."),
    }
    logger.info("-" * 62)
    logger.info(f"payer_code gain-importance rank: {payer_rank}/{len(ranked)} | "
                f"top5 {payer['top_5_features']}")

    # ── FALSIFICATION ARM ─────────────────────────────────────────────────
    # Inject a synthetic disparity: degrade predictions for one gender level.
    # The audit MUST flag it, or no null on the real data is interpretable.
    rng = np.random.default_rng(SEED)
    g_gender = pd.Series(demo["gender"].to_numpy(), index=demo["patient_nbr"].to_numpy())
    biggest = demo["gender"].value_counts().index[0]
    m_inj = (demo["gender"] == biggest).to_numpy()
    p_inj = p.copy()
    p_inj[m_inj] = rng.permutation(p_inj[m_inj])       # destroy signal in that group
    rep_inj = subgroup_report(y, p_inj, g_gender, thr, "gender_INJECTED")
    gap_inj = _gap(rep_inj["levels"], "auc")
    gap_real = reports["gender"]["disparities"]["auc"]
    falsification = {
        "design": (f"predictions for gender='{biggest}' are shuffled, destroying "
                   f"the model's signal in that subgroup only"),
        "injected_auc_gap": gap_inj.get("gap"),
        "injected_claim_supported": gap_inj.get("claim_supported"),
        "real_auc_gap": gap_real.get("gap"),
        "audit_can_detect_disparity": bool(gap_inj.get("claim_supported")),
    }
    logger.info(f"FALSIFICATION: injected AUC gap {gap_inj.get('gap')} "
                f"detected={gap_inj.get('claim_supported')} | "
                f"real gap {gap_real.get('gap')}")

    supported = {a: [m for m, d in r["disparities"].items()
                     if d.get("claim_supported")] for a, r in reports.items()}
    insufficient = {a: [l for l, v in r["levels"].items()
                        if v["status"] != "OK"] for a, r in reports.items()}

    report = {
        "phase": "2C.4a",
        "title": "Fairness / subgroup performance with intervals",
        "cohort": "test split (entry-cohort), 30-day readmission",
        "operating_threshold": thr,
        "alignment": note,
        "minimum_evidence_rule": {"min_n": MIN_N, "min_positives": MIN_POSITIVES,
                                  "rationale": ("AUC on very few positives is not "
                                                "a stable estimate; a claim about "
                                                "such a subgroup is not supportable")},
        "subgroups": reports,
        "supported_disparities": supported,
        "insufficient_evidence_levels": insufficient,
        "payer_code_equity_question": payer,
        "falsification_arm": falsification,
        "reproducibility": {"seed": SEED, "n_boot": N_BOOT,
                            "python": platform.python_version(),
                            "numpy": np.__version__, "pandas": pd.__version__},
    }
    out = REPORTS_DIR / "fairness_audit.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("-" * 62)
    logger.info(f"supported disparities: {supported}")
    logger.info(f"Report: {out.name}")
    logger.info("=" * 78)
    return report


if __name__ == "__main__":
    r = run_fairness_audit()
    print("\nSupported disparities (intervals do not overlap):")
    for a, ms in r["supported_disparities"].items():
        print(f"  {a:<10} {ms if ms else '(none)'}")
    print("\nInsufficient evidence:")
    for a, ls in r["insufficient_evidence_levels"].items():
        print(f"  {a:<10} {ls if ls else '(none)'}")
    f = r["falsification_arm"]
    print(f"\nAudit can detect an injected disparity: {f['audit_can_detect_disparity']} "
          f"(injected gap {f['injected_auc_gap']}, real gap {f['real_auc_gap']})")
    print(f"payer_code importance rank: {r['payer_code_equity_question']['gain_importance_rank']}"
          f"/{r['payer_code_equity_question']['of_n_features']}")
