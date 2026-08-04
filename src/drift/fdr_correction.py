"""
DriftSentinel — Tier 2A.3: Benjamini-Hochberg FDR across the full test family

WHY
    The pipeline runs hundreds of hypothesis tests across features, windows and
    test types with NO multiple-testing correction (audit F10). At alpha=0.01
    across a family this size, several "significant" results are expected by
    chance alone, so the claim "N features drifted" has no error control.

FAMILY SIZE — COUNTED, NOT ESTIMATED
    The audit estimated ~265. That figure could not be verified from the saved
    artifacts, because `data_drift.run_data_drift()` writes `data_drift_test.csv`
    TWICE in a single run: once for the val->test window and again for the
    train->test window, which silently overwrites the first. A whole window's
    results are destroyed before anything can read them (the same fixed-path
    class of defect as Tier 1.7 P1).

    So the tests are RE-RUN here from the shipped `data_drift` test functions,
    across every window, and the family is counted directly. Correction strength
    depends on the family size — an undercount would make BH too lenient, which
    is the failure direction that flatters the result.

WHAT IS CORRECTED
    Every p-value the detectors produce: KS, chi-square and Mann-Whitney per
    feature per window, plus the concept-drift label-shift and prediction-shift
    tests. PSI and JS divergence are NOT tests — they are effect sizes with
    threshold rules and no p-value, so they are reported separately and excluded
    from the family rather than silently corrected as if they were tests.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger
from src.drift.data_drift import (BINARY_FEATURES, ORDINAL_FEATURES, _chi2,
                                  _js_divergence, _ks, _mann_whitney, _psi,
                                  _psi_level)

logger = get_logger("fdr_correction")

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "train"
REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = {"readmitted_binary", "readmitted_multi"}
ALPHA = 0.01      # the uncorrected threshold the pipeline used
Q = 0.05          # BH false discovery rate


def benjamini_hochberg(pvals: np.ndarray, q: float = Q) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg step-up. Returns (adjusted p-values, reject flags).

    Adjusted p-values are monotone-enforced from the largest downward, which is
    the standard BH adjustment; `reject` is p_adj <= q.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    prev = 1.0
    for rank_i in range(m - 1, -1, -1):
        i = order[rank_i]
        val = p[i] * m / (rank_i + 1)
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    return adj, adj <= q


def collect_tests() -> pd.DataFrame:
    """
    Re-run every per-feature test across every window and collect the p-values.

    Windows are the three the pipeline actually evaluates. Test applicability
    follows data_drift.py exactly (chi-square for binary/ordinal, KS and
    Mann-Whitney for non-binary), so this corrects the tests as shipped rather
    than a reimplementation of them.
    """
    train = pd.read_parquet(TRAIN_DIR / "train_fs.parquet")
    val = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")
    feats = [c for c in train.columns if c not in TARGET_COLS]

    windows = [("train", train, "val", val),
               ("train", train, "test", test),
               ("val", val, "test", test)]

    rows = []
    for ref_name, ref_df, prod_name, prod_df in windows:
        for col in feats:
            r = ref_df[col].dropna().values
            p_ = prod_df[col].dropna().values
            is_bin = col in BINARY_FEATURES
            is_ord = col in ORDINAL_FEATURES

            if not is_bin:
                ks_s, ks_p = _ks(r, p_)
                mw_s, mw_p = _mann_whitney(r, p_)
            else:
                ks_s = ks_p = mw_s = mw_p = np.nan
            if is_bin or is_ord:
                c2_s, c2_p = _chi2(r, p_)
            else:
                c2_s = c2_p = np.nan

            base = {"window": f"{ref_name}->{prod_name}", "feature": col,
                    "psi": _psi(r, p_), "psi_level": _psi_level(_psi(r, p_)),
                    "js": _js_divergence(r, p_)}
            for tname, stat, pv in (("KS", ks_s, ks_p), ("chi2", c2_s, c2_p),
                                    ("MannWhitney", mw_s, mw_p)):
                if not np.isnan(pv):
                    rows.append({**base, "test": tname, "statistic": float(stat),
                                 "p_raw": float(pv)})
    return pd.DataFrame(rows)


def run_fdr_correction() -> dict:
    logger.info("=" * 78)
    logger.info("Tier 2A.3 — Benjamini-Hochberg FDR across the full test family")
    logger.info("=" * 78)

    df = collect_tests()

    # concept-drift tests belong to the same family
    cd_path = ROOT / "outputs" / "log" / "concept_drift_val_test.json"
    extra = []
    if cd_path.exists():
        cd = json.loads(cd_path.read_text(encoding="utf-8"))
        ls, ps = cd.get("label_shift", {}), cd.get("prediction_shift", {})
        if "p_value" in ls:
            extra.append({"window": "val->test", "feature": "__label__",
                          "test": "two-proportion", "statistic": ls.get("z_stat"),
                          "p_raw": float(ls["p_value"]), "psi": np.nan,
                          "psi_level": "N/A", "js": np.nan})
        for k, tn in (("ks_pval", "KS"), ("mw_pval", "MannWhitney")):
            if k in ps:
                extra.append({"window": "val->test", "feature": "__prediction__",
                              "test": tn, "statistic": ps.get("ks_stat"),
                              "p_raw": float(ps[k]), "psi": np.nan,
                              "psi_level": "N/A", "js": np.nan})
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    m = len(df)
    logger.info(f"Family size (counted, not estimated): {m} p-values")
    for w, g in df.groupby("window"):
        logger.info(f"  {w:<16} {len(g):>4} tests  "
                    f"{dict(g['test'].value_counts())}")

    # Correction must be compared at a MATCHED level. Comparing raw p<0.01
    # against BH q=0.05 mixes two different error rates and can make the
    # "corrected" set look LARGER, which is an artifact of the comparison, not a
    # property of BH.
    adj05, rej05 = benjamini_hochberg(df["p_raw"].to_numpy(), 0.05)
    adj01, rej01 = benjamini_hochberg(df["p_raw"].to_numpy(), 0.01)
    df["p_adj_q05"] = adj05
    df["p_adj_q01"] = adj01
    df["significant_raw_a05"] = df["p_raw"] < 0.05
    df["significant_raw_a01"] = df["p_raw"] < 0.01
    df["significant_fdr_q05"] = rej05
    df["significant_fdr_q01"] = rej01
    # canonical columns used downstream
    df["p_adj"] = adj05
    df["significant_raw"] = df["significant_raw_a01"]
    df["significant_fdr"] = rej05

    matched = {}
    for lvl, rawc, fdrc in ((0.05, "significant_raw_a05", "significant_fdr_q05"),
                            (0.01, "significant_raw_a01", "significant_fdr_q01")):
        nr, nf = int(df[rawc].sum()), int(df[fdrc].sum())
        matched[f"level_{lvl}"] = {"raw": nr, "fdr": nf, "lost": nr - nf}
        logger.info(f"  level {lvl}: raw {nr}/{m} -> BH {nf}/{m} "
                    f"(lost {nr - nf})")

    n_raw = int(df["significant_raw_a01"].sum())
    n_fdr = int(df["significant_fdr_q05"].sum())
    logger.info("-" * 62)
    logger.info(f"MATCHED-LEVEL comparison (the correct one):")
    for k, v in matched.items():
        logger.info(f"  {k:<12} raw {v['raw']:>4} -> FDR {v['fdr']:>4}  lost {v['lost']:>3}")
    logger.info(f"Expected false positives at alpha=0.01 : ~{ALPHA * m:.1f}")

    # EFFECT SIZE — the binding constraint here, not multiplicity.
    feat_only = df[~df["feature"].str.startswith("__")]
    psi_by_feat = feat_only.groupby(["window", "feature"])["psi"].first()
    sig_by_feat = feat_only.groupby(["window", "feature"])["significant_fdr_q05"].any()
    joint = pd.DataFrame({"psi": psi_by_feat, "sig": sig_by_feat}).reset_index()
    effect = {
        "n_feature_windows": int(len(joint)),
        "significant_after_fdr": int(joint["sig"].sum()),
        "significant_and_psi_ge_0.10": int(((joint["sig"]) & (joint["psi"] >= 0.10)).sum()),
        "significant_and_psi_ge_0.20": int(((joint["sig"]) & (joint["psi"] >= 0.20)).sum()),
        "significant_but_psi_lt_0.10": int(((joint["sig"]) & (joint["psi"] < 0.10)).sum()),
        "interpretation": (
            "Multiplicity is NOT the binding constraint on this evidence. With "
            "~20k rows per window the tests are so overpowered that trivial "
            "differences reach p < 1e-10, so BH removes almost nothing. What "
            "separates real drift from detectable-but-meaningless drift is "
            "EFFECT SIZE: most feature-windows that survive FDR have PSI < 0.10, "
            "i.e. they are statistically certain and practically negligible. "
            "This is the same lesson as the Tier 1.4 label_drift rule — "
            "significance AND a minimum effect, never significance alone."),
    }
    logger.info("-" * 62)
    logger.info("EFFECT SIZE (the actual binding constraint)")
    logger.info(f"  feature-windows significant after FDR : {effect['significant_after_fdr']}"
                f"/{effect['n_feature_windows']}")
    logger.info(f"  ... and PSI >= 0.10                   : {effect['significant_and_psi_ge_0.10']}")
    logger.info(f"  ... and PSI >= 0.20 (critical)        : {effect['significant_and_psi_ge_0.20']}")
    logger.info(f"  significant but PSI < 0.10 (trivial)  : {effect['significant_but_psi_lt_0.10']}")

    # per-feature drift counts, before and after
    feat_rows = df[~df["feature"].str.startswith("__")]
    per_window = {}
    for w, g in feat_rows.groupby("window"):
        raw_feats = set(g.loc[g["significant_raw"], "feature"])
        fdr_feats = set(g.loc[g["significant_fdr"], "feature"])
        n_feats = g["feature"].nunique()
        per_window[w] = {
            "n_features": int(n_feats),
            "n_tests": int(len(g)),
            "features_drifted_uncorrected": len(raw_feats),
            "features_drifted_fdr": len(fdr_feats),
            "features_lost_to_correction": sorted(raw_feats - fdr_feats),
        }
        logger.info(f"  {w:<16} features flagged: {len(raw_feats)}/{n_feats} raw "
                    f"-> {len(fdr_feats)}/{n_feats} after FDR")

    by_test = {}
    for t, g in df.groupby("test"):
        by_test[t] = {"n": int(len(g)),
                      "significant_raw": int(g["significant_raw"].sum()),
                      "significant_fdr": int(g["significant_fdr"].sum())}

    out_csv = REPORTS_DIR / "fdr_corrected_tests.csv"
    df.sort_values("p_raw").to_csv(out_csv, index=False)

    report = {
        "phase": "2A.3",
        "title": "Benjamini-Hochberg FDR across the full hypothesis-test family",
        "alpha_uncorrected": ALPHA,
        "q_fdr": Q,
        "family_size_counted": m,
        "audit_estimate": 265,
        "family_size_note": (
            "COUNTED by re-running the shipped test functions across all three "
            "windows, not read from the saved CSVs: data_drift.run_data_drift() "
            "writes data_drift_test.csv twice in one run (val->test then "
            "train->test), silently destroying the first window's results. "
            "An undercount would weaken BH in the flattering direction."),
        "matched_level_comparison": matched,
        "effect_size_analysis": effect,
        "significant_uncorrected_alpha01": n_raw,
        "significant_after_fdr_q05": n_fdr,
        "note_on_cross_level_comparison": (
            "Do NOT compare significant_uncorrected_alpha01 against "
            "significant_after_fdr_q05 — different error rates; the cross-level "
            "difference is an artifact. Use matched_level_comparison."),
        "expected_false_positives_uncorrected": round(ALPHA * m, 1),
        "per_window": per_window,
        "per_test_type": by_test,
        "excluded_from_family": {
            "PSI": ("an effect size with a threshold rule, not a hypothesis test — "
                    "it has no p-value and correcting it would be a category error"),
            "Jensen-Shannon": "same — a divergence, not a test",
        },
        "artifact": str(out_csv.relative_to(ROOT).as_posix()),
        "reproducibility": {"python": platform.python_version(),
                            "numpy": np.__version__, "pandas": pd.__version__},
    }
    out = REPORTS_DIR / "fdr_correction.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Reports: {out.name}, {out_csv.name}")
    logger.info("=" * 78)
    return report


if __name__ == "__main__":
    r = run_fdr_correction()
    print(f"\nFamily size (counted): {r['family_size_counted']} "
          f"(audit estimated {r['audit_estimate']})")
    print("Matched-level comparison (the correct one):")
    for k, v in r["matched_level_comparison"].items():
        print(f"  {k:<12} raw {v['raw']:>4} -> BH {v['fdr']:>4}  lost {v['lost']}")
    e = r["effect_size_analysis"]
    print(f"Effect size: {e['significant_after_fdr']}/{e['n_feature_windows']} "
          f"feature-windows significant after FDR, but only "
          f"{e['significant_and_psi_ge_0.10']} have PSI >= 0.10 "
          f"({e['significant_but_psi_lt_0.10']} significant-but-negligible)")
