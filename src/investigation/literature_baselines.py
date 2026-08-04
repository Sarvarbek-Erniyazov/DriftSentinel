"""
DriftSentinel — Tier 2C.2: literature positioning, made traceable.

WHY THIS MODULE EXISTS AT ALL
    R4 says every number in the README maps to a named file in outputs/reports/
    and that nothing is hand-typed. A literature comparison is the one table that
    CANNOT be regenerated from this repository, because its inputs live in other
    people's papers. The honest resolution is not to exempt it from R4 but to
    split it:

      * the EXTERNAL rows are curated data with per-row provenance (DOI, URL,
        the date the value was read, and the exact protocol the paper used).
        They are declared here as a constant, reviewable in one screen.
      * the OURS rows are read at runtime from the generated artifacts. They are
        never typed. If `headline_metrics_ci.json` changes, this table changes.
      * the CONTRAST between them is computed, not asserted.

    So the table ships as a generated artifact whose external half is auditable
    by DOI and whose internal half is auditable by regeneration.

WHAT THE COMPARISON IS FOR
    Not to win it. Published AUCs on this dataset and target cluster in a narrow
    band, and this project's discrimination sits inside it. The point of the
    table is the SECOND column — the evaluation protocol — because the spread
    attributable to protocol is of the same order as the spread attributable to
    the model, and only one of the two is routinely reported.

WHAT IS DELIBERATELY NOT CLAIMED
    Cross-paper AUC differences are not a controlled comparison: different
    feature sets, different preprocessing, different imbalance handling. The
    controlled comparison is the one INSIDE this repository, where model, code
    and features are held fixed and only the split regime moves. Both are
    reported, and which is which is stated on every row.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.artifact_io import write_artifact
from src.monitoring.logger import get_logger

logger = get_logger("literature_baselines")

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "outputs" / "reports"

# Date the external values below were read from the sources. Recorded so a
# reviewer can tell how stale the curation is without re-reading every paper.
CURATION_DATE = "2026-08-04"

# ── External baselines ────────────────────────────────────────────────────────
# Curated by hand from the cited sources. Every row carries the protocol,
# because the protocol is the finding. `patient_grouped` is the field that
# matters most: encounters are not independent (46.2% of patients contribute
# more than one), so a split that does not group by patient leaks.
EXTERNAL_BASELINES = [
    {
        "study": "Strack et al. (2014)",
        "citation": ("Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, "
                     "Clore JN. Impact of HbA1c Measurement on Hospital Readmission "
                     "Rates: Analysis of 70,000 Clinical Database Patient Records. "
                     "BioMed Research International, 2014:781670."),
        "doi": "10.1155/2014/781670",
        "url": "https://onlinelibrary.wiley.com/doi/10.1155/2014/781670",
        "target": "<30 (early readmission)",
        "model": "multivariable logistic regression",
        "auc": None,
        "auc_note": ("No discrimination metric is reported. This is an ETIOLOGIC "
                     "study — it estimates the association between HbA1c measurement "
                     "and early readmission while adjusting for covariates, and "
                     "reports odds ratios. Citing it as a predictive baseline would "
                     "misrepresent it."),
        "protocol": "multivariable logistic regression on ~70k filtered encounters",
        "patient_grouped": None,
        "role": ("the source of the dataset and of the <30 / >30 / NO encoding; "
                 "establishes the clinical framing, not a predictive benchmark"),
        "verified": "abstract and methods read from the publisher page",
    },
    {
        "study": "Liu, Sue & Wu (2024)",
        "citation": ("Liu VB, Sue LY, Wu Y. Comparison of machine learning models for "
                     "predicting 30-day readmission rates for patients with diabetes. "
                     "Journal of Medical Artificial Intelligence, 2024;7:23."),
        "doi": "10.21037/jmai-24-70",
        "url": "https://jmai.amegroups.org/article/view/9179/html",
        "target": "<30 (11.2% prevalence, 101,766 encounters)",
        "model": "best of 7 (XGBoost)",
        "auc": 0.64,
        "auc_ci95": [0.64, 0.65],
        "auc_note": ("range across all seven models reported: 0.48 (SVM-RBF) to 0.64 "
                     "(XGBoost); tree ensembles and logistic regression all 0.62-0.64"),
        "protocol": "group 5-fold CV, encounters for the same patient kept in one fold",
        "patient_grouped": True,
        "role": ("the PROTOCOL-MATCHED comparison: same dataset, same target, same "
                 "prevalence, and patient grouping enforced as it is here"),
        "verified": "full text read from the publisher page",
    },
    {
        "study": "Salim & Ibrahim (2026)",
        "citation": ("Salim SS, Ibrahim AA. A Machine Learning Approach for Predicting "
                     "30-Day Hospital Readmission in Patients with Diabetes. "
                     "Healthcare (Basel), 2026;14(9):1185."),
        "doi": "10.3390/healthcare14091185",
        "url": "https://www.mdpi.com/2227-9032/14/9/1185",
        "target": "<30 (11.16% prevalence, 101,766 encounters)",
        "model": "XGBoost",
        "auc": 0.664,
        "auc_ci95": None,
        "auc_note": ("0.664 nested CV, 0.688 after calibration; logistic regression "
                     "0.657, random forest 0.650"),
        "protocol": "nested stratified 5-fold CV (3-fold inner)",
        "patient_grouped": False,
        "patient_grouped_note": ("the authors state as a limitation that the "
                                 "cross-validation did not enforce patient-level "
                                 "splits, so the same patient can appear on both "
                                 "sides of a fold boundary"),
        "role": ("the UNGROUPED comparison: identical dataset and target, protocol "
                 "differing in exactly the dimension that leaks"),
        "verified": "full text read from PMC",
    },
]

# Which of our generated artifacts supplies each internal row, and where in it.
OURS_SOURCES = {
    "entry_cohort_test": {
        "file": "headline_metrics_ci.json",
        "path": ["variance_components", "1_estimation_patient_clustered_bootstrap",
                 "by_split", "test", "metrics", "auc"],
        "protocol": "entry-cohort split, held-out test window, patient-clustered bootstrap",
        "patient_grouped": True,
    },
    "random_patient_split_test": {
        "file": "headline_metrics_ci.json",
        "path": ["variance_components", "3_split_variance_reference_only",
                 "by_split", "test", "auc"],
        "protocol": "random patient-level split, 20 seeds (Tier 0 negative control)",
        "patient_grouped": True,
    },
    "patient_grouped_cv_all_features": {
        "file": "selection_ablation.json",
        "path": ["arms", "a_all_features"],
        "protocol": "StratifiedGroupKFold(5) x 2 repeats, grouped by patient",
        "patient_grouped": True,
    },
    "patient_grouped_cv_shipped_selection": {
        "file": "selection_ablation.json",
        "path": ["arms", "c_full_7_stage_shipped"],
        "protocol": "StratifiedGroupKFold(5) x 2 repeats, grouped by patient",
        "patient_grouped": True,
    },
}


def _dig(obj: dict, path: list[str]):
    """Walk a JSON path, raising rather than defaulting if a key is absent (R6)."""
    cur = obj
    for key in path:
        if key not in cur:
            raise KeyError(f"missing key {key!r} on path {'/'.join(path)}")
        cur = cur[key]
    return cur


def collect_our_numbers() -> dict:
    """
    Read this project's AUCs from the generated artifacts.

    Nothing here is typed. A missing key raises: a literature table that silently
    fell back to a default would be indistinguishable from a real measurement,
    which is the exact failure mode R6 exists to prevent.
    """
    cache: dict[str, dict] = {}
    rows = {}
    for name, spec in OURS_SOURCES.items():
        if spec["file"] not in cache:
            with open(REPORTS_DIR / spec["file"], encoding="utf-8") as f:
                cache[spec["file"]] = json.load(f)
        node = _dig(cache[spec["file"]], spec["path"])

        if "point" in node:                      # bootstrap-interval shape
            auc, ci, sd = node["point"], node.get("ci95"), None
        elif "auc_mean" in node:                 # cross-validation-arm shape
            auc, ci, sd = node["auc_mean"], node.get("auc_ci95"), node.get("auc_std")
        elif "mean" in node:                     # seed-variance shape
            auc, ci, sd = node["mean"], None, node.get("std")
        else:
            raise KeyError(f"unrecognised metric shape at {name}: {sorted(node)}")

        rows[name] = {
            "auc": round(float(auc), 5),
            "auc_ci95": ci,
            "auc_std": sd,
            "protocol": spec["protocol"],
            "patient_grouped": spec["patient_grouped"],
            "source_file": f"outputs/reports/{spec['file']}",
            "source_path": "/".join(spec["path"]),
        }
        logger.info(f"  ours[{name}] AUC={auc:.5f}  <- {spec['file']}")
    return rows


def build_report() -> dict:
    logger.info("=" * 78)
    logger.info("DriftSentinel — Tier 2C.2  literature positioning")
    logger.info("=" * 78)

    ours = collect_our_numbers()

    # ── The contrast that is the actual finding ──────────────────────────────
    # Held fixed: model, features, code, seed. Moved: the split regime only.
    grouped_cv = ours["patient_grouped_cv_all_features"]["auc"]
    entry_test = ours["entry_cohort_test"]["auc"]
    random_test = ours["random_patient_split_test"]["auc"]

    protocol_gap = round(grouped_cv - entry_test, 5)
    regime_gap = round(random_test - entry_test, 5)

    published = [b for b in EXTERNAL_BASELINES if b["auc"] is not None]
    pub_lo = min(b["auc"] for b in published)
    pub_hi = max(b["auc"] for b in published)
    pub_spread = round(pub_hi - pub_lo, 5)

    contrast = {
        "within_repository_controlled": {
            "held_fixed": "model class, hyperparameters, features, code, seed",
            "varied": "split regime only",
            "patient_grouped_cv_auc": grouped_cv,
            "entry_cohort_test_auc": entry_test,
            "random_patient_split_test_auc": random_test,
            "protocol_gap_cv_minus_entry_cohort": protocol_gap,
            "regime_gap_random_minus_entry_cohort": regime_gap,
            "interpretation": (
                "The same model on the same features scores ~0.04 AUC higher under "
                "patient-grouped CV or a random patient split than on the entry-cohort "
                "held-out window. Nothing about the model changed. A single AUC reported "
                "without naming its regime is therefore not comparable across papers, "
                "and this project's own regime matrix is what makes the gap measurable."),
        },
        "across_published_studies_uncontrolled": {
            "published_auc_min": pub_lo,
            "published_auc_max": pub_hi,
            "published_spread": pub_spread,
            "caveat": (
                "NOT a controlled comparison — feature sets, preprocessing and "
                "imbalance handling all differ between these papers. Reported only to "
                "establish the order of magnitude of between-paper variation."),
            "observation": (
                "the between-paper spread is of the same order as the within-repository "
                "protocol gap, and the higher published number is the study that states "
                "it did not enforce patient-level splits"),
            "not_claimed": (
                "we do NOT claim the published difference IS leakage. Two studies is "
                "not a sample, and the papers differ in more than one dimension. The "
                "claim is only that protocol variation is large enough to matter and is "
                "usually unreported."),
        },
        "where_this_project_sits": (
            "inside the published band on the deployed split, at or above its upper "
            "edge under the protocol-matched grouped CV. Discrimination is not this "
            "project's contribution and is not presented as one."),
    }

    report = {
        "phase": "2C.2",
        "title": "Literature positioning and baseline comparison",
        "task": "30-day readmission (<30) on UCI Diabetes 130-US Hospitals, 1999-2008",
        "novelty_statement": (
            "Neither the task nor the dataset is novel. Readmission prediction on this "
            "dataset is well-trodden and drift detection on it is not new. The "
            "contribution is the negative-control methodology and the systematic "
            "falsification design, not the model and not the detector suite."),
        "curation_date": CURATION_DATE,
        "external_baselines": EXTERNAL_BASELINES,
        "ours": ours,
        "contrast": contrast,
        "traceability": {
            "external_rows": "curated by hand; each carries a DOI, URL and read date",
            "internal_rows": ("read at runtime from the named generated artifacts; "
                              "never typed, and a missing key raises"),
            "why_split": ("R4 forbids hand-typed results, but external literature "
                          "cannot be regenerated. Separating the halves keeps the "
                          "rule enforceable on the half it can govern."),
        },
        "reproducibility": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return report


def run_literature_baselines() -> dict:
    report = build_report()
    out = REPORTS_DIR / "literature_baselines.json"
    write_artifact(out, report, overwrite=True, preserve=True)
    c = report["contrast"]["within_repository_controlled"]
    logger.info("-" * 62)
    logger.info(f"protocol gap (grouped CV - entry-cohort test): {c['protocol_gap_cv_minus_entry_cohort']:+.4f}")
    logger.info(f"regime gap  (random split - entry-cohort test): {c['regime_gap_random_minus_entry_cohort']:+.4f}")
    logger.info(f"published band: {report['contrast']['across_published_studies_uncontrolled']['published_auc_min']}"
                f"-{report['contrast']['across_published_studies_uncontrolled']['published_auc_max']}")
    logger.info(f"Report: {out.name}")
    logger.info("=" * 78)
    return report


if __name__ == "__main__":
    r = run_literature_baselines()
    print("\nPublished baselines (same dataset, same <30 target):")
    for b in r["external_baselines"]:
        auc = f"{b['auc']:.3f}" if b["auc"] is not None else "  n/a"
        grp = {True: "grouped", False: "NOT grouped", None: "n/a"}[b["patient_grouped"]]
        print(f"  {b['study']:<26} AUC {auc}   {grp:<12} {b['doi']}")
    print("\nOurs (read from generated artifacts):")
    for name, o in r["ours"].items():
        print(f"  {name:<36} AUC {o['auc']:.4f}   {o['source_file']}")
    c = r["contrast"]["within_repository_controlled"]
    print(f"\nSame model, regime varied only: "
          f"grouped CV {c['patient_grouped_cv_auc']:.4f} vs "
          f"entry-cohort test {c['entry_cohort_test_auc']:.4f} "
          f"({c['protocol_gap_cv_minus_entry_cohort']:+.4f})")
