"""
DriftSentinel — Feature Consistency Checker
Post-selection validation across train / val / test splits.
Runs before any model training to surface data quality and leakage issues.
All findings logged to outputs/log/consistency.log.

Checks performed:
    Check 1  — Schema consistency        : column names and dtypes match across splits
    Check 2  — Shape sanity              : row/column counts are plausible
    Check 3  — Null consistency          : null rates do not spike in val/test
    Check 4  — Target leakage           : no feature correlated > threshold with target
    Check 5  — FE_ feature leakage      : engineered features checked separately
    Check 6  — Distribution shift (PSI) : Population Stability Index per feature
    Check 7  — Distribution shift (KS)  : Kolmogorov-Smirnov test per feature
    Check 8  — Encoding consistency     : value ranges match train bounds
    Check 9  — Target distribution      : class balance stable across splits
    Check 10 — Duplicate row check      : no identical rows leaked across splits
    Check 11 — Feature variance parity  : variance collapse in val/test flagged
    Check 12 — Constant prediction risk : features near-constant in val/test
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import ks_2samp
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("consistency")

# Tier 2C.6 reproducibility: this was a HARDCODED ABSOLUTE PATH to one
# developer's machine, so `pipeline.py` did not reproduce anything from raw
# data on a clean clone -- it read from and wrote to a directory that exists
# nowhere else. On Linux CI the same literal resolves to a RELATIVE folder
# whose name contains backslashes, so artifacts land somewhere harmless-
# looking and the run still 'succeeds'. It worked on exactly one machine,
# which is why nothing caught it. Now derived from this file's location.
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"
REPORT_DIR    = ROOT / "outputs" / "log"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Thresholds ─────────────────────────────────────────────────────────────
TARGET_COLS             = {"readmitted_binary", "readmitted_multi", "readmitted"}
LEAKAGE_CORR_THRESHOLD  = 0.80
NULL_SPIKE_THRESHOLD    = 0.05
PSI_WARNING_THRESHOLD   = 0.10
PSI_CRITICAL_THRESHOLD  = 0.20
KS_ALPHA                = 0.01
VARIANCE_RATIO_MIN      = 0.10
PSI_N_BINS              = 10


# ── PSI helper ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════
# Failure classification (Phase 1.0 — audit F2)
# ══════════════════════════════════════════════════════════════════════════
#
# The pipeline previously set `summary["pipeline_ready"] = True` unconditionally,
# over a check reporting 12 FAILs. The INTENT was defensible — this project
# studies distribution shift, so shift between splits is the subject matter, not
# a defect — but a blanket assignment cannot distinguish "expected shift" from
# "the schema is broken", and to a reader it looks like a disabled quality gate.
#
# Failures are now classified by check name. Anything not matched here is
# UNEXPECTED and blocks the pipeline.

EXPECTED_DRIFT_CHECK_PREFIXES = (
    "psi_",            # population stability between splits — the object of study
    "target_dist_",    # label prevalence differs between entry cohorts
)

# These must NEVER fail. They are integrity properties, not observations about
# the data distribution, and a failure means the pipeline itself is wrong.
INTEGRITY_CHECK_PREFIXES = (
    "schema_", "column_", "dtype_", "total_row_coverage", "leakage_",
    "target_leakage", "fe_", "null_spike", "constant_", "duplicate_",
)


def classify_failure(check_name: str) -> str:
    """Return 'expected_drift' or 'unexpected' for a failing check."""
    if check_name.startswith(EXPECTED_DRIFT_CHECK_PREFIXES):
        return "expected_drift"
    return "unexpected"


def evaluate_gate(report: dict, drift_expected: bool = True) -> dict:
    """
    Evaluate the pipeline-readiness gate and return its full reasoning.

    Replaces `summary["pipeline_ready"] = True`. The gate is now an expression
    over named categories, and it logs why it decided what it decided:

        ready = no unexpected failures  AND  (expected failures are permitted
                only when drift_expected is True)

    Every failure is named and attributed, so "we allowed 12 failures" becomes
    "we allowed these 12, each of which is a distribution-shift observation,
    and we would NOT have allowed a leakage or schema failure."
    """
    fails = [f for f in report.get("findings", []) if f["level"] == "FAIL"]
    expected, unexpected = [], []
    for f in fails:
        entry = {"check": f["check"], "split": f["split"], "detail": f["detail"]}
        (expected if classify_failure(f["check"]) == "expected_drift"
         else unexpected).append(entry)

    ready = (not unexpected) and (drift_expected or not expected)

    if not ready:
        if unexpected:
            reason = (f"{len(unexpected)} UNEXPECTED failure(s) that are not "
                      f"distribution-shift observations: "
                      f"{[u['check'] for u in unexpected]}")
        else:
            reason = (f"{len(expected)} drift-related failure(s) present and "
                      f"drift_expected=False")
    elif expected:
        reason = (f"{len(expected)} drift-related failure(s) permitted because "
                  f"drift_expected=True; 0 unexpected failures. Distribution "
                  f"shift between splits is this project's subject matter, not "
                  f"a pipeline defect.")
    else:
        reason = "no failures"

    return {
        "ready": bool(ready),
        "reason": reason,
        "drift_expected": bool(drift_expected),
        "n_expected_failures": len(expected),
        "n_unexpected_failures": len(unexpected),
        "expected_failures": expected,
        "unexpected_failures": unexpected,
        "rule": ("ready = (no unexpected failures) and "
                 "(drift_expected or no expected failures)"),
        "expected_categories": list(EXPECTED_DRIFT_CHECK_PREFIXES),
        "integrity_categories": list(INTEGRITY_CHECK_PREFIXES),
    }


def _psi(reference: np.ndarray, production: np.ndarray, n_bins: int = PSI_N_BINS) -> float:
    """
    Population Stability Index.
    PSI < 0.10  : no shift
    PSI 0.10-0.20 : moderate shift — investigate
    PSI > 0.20  : significant shift — action required
    """
    ref  = reference[~np.isnan(reference)]
    prod = production[~np.isnan(production)]

    if len(ref) == 0 or len(prod) == 0:
        return np.nan

    breakpoints = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    breakpoints  = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return np.nan

    ref_counts  = np.histogram(ref,  bins=breakpoints)[0]
    prod_counts = np.histogram(prod, bins=breakpoints)[0]

    ref_pct  = ref_counts  / len(ref)  + 1e-8
    prod_pct = prod_counts / len(prod) + 1e-8

    psi_val = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return round(float(psi_val), 6)


class ConsistencyChecker:
    """
    Validates feature consistency across train / val / test splits.
    Must be run after FeatureSelector.transform() on all three splits.
    """

    def __init__(self):
        self.report: dict = {}

    # ──────────────────────────────────────────────────────────────────────
    def run(
        self,
        train: pd.DataFrame,
        val:   pd.DataFrame,
        test:  pd.DataFrame,
    ) -> dict:
        """
        Run all consistency checks.

        Parameters
        ----------
        train : selected feature train split
        val   : selected feature val split
        test  : selected feature test split

        Returns
        -------
        report : dict with all check results and overall pass/fail
        """
        logger.info("=" * 70)
        logger.info("DriftSentinel — Feature Consistency Checker")
        logger.info("=" * 70)
        logger.info(f"Train : {train.shape}")
        logger.info(f"Val   : {val.shape}")
        logger.info(f"Test  : {test.shape}")

        findings = []
        passed   = 0
        warned   = 0
        failed   = 0

        def record(level: str, check: str, split: str, detail: str):
            nonlocal passed, warned, failed
            entry = {
                "level"  : level,
                "check"  : check,
                "split"  : split,
                "detail" : detail,
            }
            findings.append(entry)
            if level == "PASS":
                passed += 1
                logger.info(f"  [PASS]  {check:<40} [{split}] {detail}")
            elif level == "WARN":
                warned += 1
                logger.warning(f"  [WARN]  {check:<40} [{split}] {detail}")
            elif level == "FAIL":
                failed += 1
                logger.error(f"  [FAIL]  {check:<40} [{split}] {detail}")

        feat_cols = self._feature_cols(train)
        targets   = self._target_cols(train)

        # ── Check 1 — Schema consistency ──────────────────────────────────
        logger.info("-" * 50)
        logger.info("Check 1: Schema Consistency")

        train_cols = set(train.columns)
        for name, split_df in [("val", val), ("test", test)]:
            split_cols   = set(split_df.columns)
            missing_cols = train_cols - split_cols
            extra_cols   = split_cols - train_cols

            record(
                "PASS" if not missing_cols else "FAIL",
                "schema_missing_cols", name,
                f"missing={list(missing_cols) if missing_cols else 'none'}"
            )
            record(
                "PASS" if not extra_cols else "WARN",
                "schema_extra_cols", name,
                f"extra={list(extra_cols) if extra_cols else 'none'}"
            )

            for col in feat_cols:
                if col not in split_df.columns:
                    continue
                dtype_match = train[col].dtype == split_df[col].dtype
                record(
                    "PASS" if dtype_match else "WARN",
                    f"dtype_{col}", name,
                    f"train={train[col].dtype} {name}={split_df[col].dtype}"
                )

        # ── Check 2 — Shape sanity ─────────────────────────────────────────
        logger.info("-" * 50)
        logger.info("Check 2: Shape Sanity")

        total_rows = len(train) + len(val) + len(test)
        record("PASS", "total_row_coverage", "all",
               f"train={len(train):,} val={len(val):,} test={len(test):,} total={total_rows:,}")

        for name, split_df in [("val", val), ("test", test)]:
            col_match = split_df.shape[1] == train.shape[1]
            record(
                "PASS" if col_match else "FAIL",
                "column_count_match", name,
                f"train={train.shape[1]} {name}={split_df.shape[1]}"
            )

        # ── Check 3 — Null consistency ─────────────────────────────────────
        logger.info("-" * 50)
        logger.info("Check 3: Null Rate Consistency")

        train_null_rates = train[feat_cols].isna().mean()
        for name, split_df in [("val", val), ("test", test)]:
            for col in feat_cols:
                if col not in split_df.columns:
                    continue
                split_null = split_df[col].isna().mean()
                train_null = train_null_rates.get(col, 0)
                delta      = abs(split_null - train_null)
                level = (
                    "FAIL" if delta > NULL_SPIKE_THRESHOLD * 2 else
                    "WARN" if delta > NULL_SPIKE_THRESHOLD else
                    "PASS"
                )
                if level != "PASS":
                    record(level, f"null_spike_{col}", name,
                           f"train_null={train_null:.4f} {name}_null={split_null:.4f} delta={delta:.4f}")

        passing_null = sum(1 for f in findings if f["check"].startswith("null_spike") and f["level"] == "PASS")
        logger.info(f"  Null consistency: {len(feat_cols)} features checked")

        # ── Check 4 — Target leakage ───────────────────────────────────────
        logger.info("-" * 50)
        logger.info(f"Check 4: Target Leakage (threshold=|r|>{LEAKAGE_CORR_THRESHOLD})")

        present_targets = [c for c in targets if c in train.columns]
        suspicious      = []

        for col in feat_cols:
            for tgt in present_targets:
                try:
                    r = train[col].corr(train[tgt])
                    if abs(r) > LEAKAGE_CORR_THRESHOLD:
                        suspicious.append((col, tgt, round(r, 4)))
                        record("FAIL", f"leakage_{col}_vs_{tgt}", "train",
                               f"|r|={abs(r):.4f} exceeds {LEAKAGE_CORR_THRESHOLD}")
                except Exception:
                    pass

        if not suspicious:
            record("PASS", "target_leakage_all_features", "train",
                   f"no |r| > {LEAKAGE_CORR_THRESHOLD} among {len(feat_cols)} features")

        # ── Check 5 — FE_ feature leakage ─────────────────────────────────
        logger.info("-" * 50)
        logger.info("Check 5: FE_ Feature Leakage (separate audit)")

        fe_cols     = [c for c in feat_cols if c.startswith("FE_")]
        fe_sus      = []
        for col in fe_cols:
            for tgt in present_targets:
                try:
                    r = train[col].corr(train[tgt])
                    if abs(r) > LEAKAGE_CORR_THRESHOLD:
                        fe_sus.append((col, tgt, round(r, 4)))
                except Exception:
                    pass

        record(
            "PASS" if not fe_sus else "FAIL",
            "fe_feature_leakage", "train",
            f"suspicious FE_ pairs={fe_sus if fe_sus else 'none'}"
        )
        logger.info(f"  FE_ features audited: {len(fe_cols)}")

        # ── Check 6 — PSI per feature ──────────────────────────────────────
        logger.info("-" * 50)
        logger.info(f"Check 6: PSI Distribution Shift "
                    f"(warn>{PSI_WARNING_THRESHOLD}, critical>{PSI_CRITICAL_THRESHOLD})")

        psi_results = {}
        for col in feat_cols:
            psi_vals = {}
            for name, split_df in [("val", val), ("test", test)]:
                if col not in split_df.columns:
                    continue
                psi_val = _psi(
                    train[col].values,
                    split_df[col].values
                )
                psi_vals[name] = psi_val
                level = (
                    "FAIL" if psi_val is not None and psi_val > PSI_CRITICAL_THRESHOLD else
                    "WARN" if psi_val is not None and psi_val > PSI_WARNING_THRESHOLD  else
                    "PASS"
                )
                if level != "PASS":
                    record(level, f"psi_{col}", name,
                           f"PSI={psi_val:.4f} "
                           f"({'CRITICAL' if psi_val > PSI_CRITICAL_THRESHOLD else 'MODERATE'})")
            psi_results[col] = psi_vals

        # Log top drifted features
        psi_summary = {
            col: max(v.values()) if v else 0
            for col, v in psi_results.items()
            if psi_results[col]
        }
        top_psi = sorted(psi_summary.items(), key=lambda x: -x[1])[:15]
        logger.info("  Top 15 PSI scores (train vs max(val,test)):")
        for col, psi_val in top_psi:
            status = (
                "CRITICAL" if psi_val > PSI_CRITICAL_THRESHOLD else
                "MODERATE" if psi_val > PSI_WARNING_THRESHOLD  else
                "STABLE"
            )
            logger.info(f"    {col:<45} PSI={psi_val:.4f}  [{status}]")

        self.report["psi_results"] = psi_results

        # ── Check 7 — KS Test per feature ─────────────────────────────────
        logger.info("-" * 50)
        logger.info(f"Check 7: Kolmogorov-Smirnov Test (alpha={KS_ALPHA})")

        ks_results  = {}
        ks_drifted  = []
        for col in feat_cols:
            ks_vals = {}
            for name, split_df in [("val", val), ("test", test)]:
                if col not in split_df.columns:
                    continue
                try:
                    stat, p_val = ks_2samp(
                        train[col].dropna().values,
                        split_df[col].dropna().values
                    )
                    ks_vals[name] = {"stat": round(stat, 4), "p_value": round(p_val, 6)}
                    if p_val < KS_ALPHA:
                        ks_drifted.append((col, name, round(stat, 4), round(p_val, 6)))
                except Exception:
                    pass
            ks_results[col] = ks_vals

        logger.info(f"  Features with KS drift (p < {KS_ALPHA}): {len(ks_drifted)}")
        for col, split, stat, p in sorted(ks_drifted, key=lambda x: -x[2])[:15]:
            logger.info(f"    {col:<40} [{split}] KS={stat:.4f}  p={p:.6f}")
            record("WARN", f"ks_drift_{col}", split,
                   f"KS={stat:.4f} p={p:.6f} — distribution shifted")

        self.report["ks_results"] = ks_results

        # ── Check 8 — Encoding consistency ────────────────────────────────
        logger.info("-" * 50)
        logger.info("Check 8: Encoding Consistency (value range bounds)")

        for col in feat_cols:
            train_min = train[col].min()
            train_max = train[col].max()
            for name, split_df in [("val", val), ("test", test)]:
                if col not in split_df.columns:
                    continue
                split_min = split_df[col].min()
                split_max = split_df[col].max()
                out_of_bounds = split_min < train_min or split_max > train_max
                if out_of_bounds:
                    record("WARN", f"range_oob_{col}", name,
                           f"train=[{train_min},{train_max}] "
                           f"{name}=[{split_min},{split_max}]")

        # ── Check 9 — Target distribution ─────────────────────────────────
        logger.info("-" * 50)
        logger.info("Check 9: Target Distribution Stability")

        for tgt in present_targets:
            train_dist = train[tgt].value_counts(normalize=True).sort_index()
            for name, split_df in [("val", val), ("test", test)]:
                if tgt not in split_df.columns:
                    continue
                split_dist = split_df[tgt].value_counts(normalize=True).sort_index()
                for cls in train_dist.index:
                    t_pct = train_dist.get(cls, 0)
                    s_pct = split_dist.get(cls, 0)
                    delta = abs(t_pct - s_pct)
                    level = "FAIL" if delta > 0.10 else "WARN" if delta > 0.05 else "PASS"
                    if level != "PASS":
                        record(level, f"target_dist_{tgt}_class{cls}", name,
                               f"train={t_pct:.3f} {name}={s_pct:.3f} delta={delta:.3f}")
                logger.info(
                    f"  {tgt} [{name}]: " +
                    "  ".join([f"cls{k}={v:.3f}" for k, v in split_dist.items()])
                )

        # ── Check 10 — Cross-split duplicate rows ──────────────────────────
        logger.info("-" * 50)
        logger.info("Check 10: Cross-split Duplicate Row Detection")

        train_hash = set(
            pd.util.hash_pandas_object(train[feat_cols], index=False).values
        )
        for name, split_df in [("val", val), ("test", test)]:
            split_hash  = pd.util.hash_pandas_object(
                split_df[[c for c in feat_cols if c in split_df.columns]],
                index=False
            ).values
            leaked_rows = sum(1 for h in split_hash if h in train_hash)
            pct         = leaked_rows / len(split_df) * 100
            level       = "FAIL" if pct > 1.0 else "WARN" if pct > 0.1 else "PASS"
            record(level, "cross_split_duplicate_rows", name,
                   f"leaked_rows={leaked_rows:,} ({pct:.3f}%)")

        # ── Check 11 — Feature variance parity ────────────────────────────
        logger.info("-" * 50)
        logger.info(f"Check 11: Feature Variance Parity "
                    f"(min_ratio={VARIANCE_RATIO_MIN})")

        for col in feat_cols:
            train_var = train[col].var()
            if train_var == 0:
                continue
            for name, split_df in [("val", val), ("test", test)]:
                if col not in split_df.columns:
                    continue
                split_var = split_df[col].var()
                ratio     = split_var / train_var if train_var > 0 else 1.0
                if ratio < VARIANCE_RATIO_MIN:
                    record("WARN", f"variance_collapse_{col}", name,
                           f"train_var={train_var:.4f} "
                           f"{name}_var={split_var:.4f} ratio={ratio:.4f}")

        # ── Check 12 — Constant prediction risk ───────────────────────────
        logger.info("-" * 50)
        logger.info("Check 12: Constant Prediction Risk in Val/Test")

        for name, split_df in [("val", val), ("test", test)]:
            for col in feat_cols:
                if col not in split_df.columns:
                    continue
                n_unique = split_df[col].nunique()
                if n_unique == 1:
                    record("WARN", f"constant_feature_{col}", name,
                           f"only 1 unique value in {name} — "
                           f"model will predict constant for this feature")

        # ── Summary ────────────────────────────────────────────────────────
        logger.info("=" * 70)
        logger.info("Consistency Check Summary")
        logger.info(f"  PASS  : {passed}")
        logger.info(f"  WARN  : {warned}")
        logger.info(f"  FAIL  : {failed}")

        drifted_psi = [
            col for col, v in psi_summary.items()
            if v > PSI_WARNING_THRESHOLD
        ]
        logger.info(f"  PSI drift detected   : {len(drifted_psi)} features")
        logger.info(f"  KS  drift detected   : {len(ks_drifted)} feature-split pairs")

        if failed > 0:
            logger.error(f"CONSISTENCY FAILED — {failed} critical issues. "
                         f"Do not proceed to model training.")
        elif warned > 0:
            logger.warning(f"CONSISTENCY PASSED WITH WARNINGS — {warned} items to review.")
        else:
            logger.info("CONSISTENCY FULLY PASSED — safe to proceed to modeling.")

        logger.info("=" * 70)

        self.report.update({
            "passed"       : passed,
            "warned"       : warned,
            "failed"       : failed,
            "findings"     : findings,
            "ready"        : failed == 0,
            "drifted_psi"  : drifted_psi,
            "ks_drifted"   : [(c, s, st, p) for c, s, st, p in ks_drifted],
        })

        self._save_report()
        return self.report

    # ──────────────────────────────────────────────────────────────────────
    def _feature_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in TARGET_COLS]

    def _target_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c in TARGET_COLS]

    # ──────────────────────────────────────────────────────────────────────
    def _save_report(self):
        """Save full consistency report as JSON."""
        report_path = REPORT_DIR / "consistency_report.json"

        serializable = {
            k: v for k, v in self.report.items()
            if k not in ("psi_results", "ks_results")
        }
        serializable["psi_results"] = {
            col: {s: round(v, 6) if v is not None else None
                  for s, v in vals.items()}
            for col, vals in self.report.get("psi_results", {}).items()
        }
        serializable["ks_results"] = {
            col: {
                s: {"stat": v["stat"], "p_value": v["p_value"]}
                for s, v in splits.items()
            }
            for col, splits in self.report.get("ks_results", {}).items()
        }

        with open(report_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Consistency report saved -> {report_path}")


if __name__ == "__main__":
    from src.data.loader        import load_raw
    from src.data.validator     import validate
    from src.data.splitter      import split
    from src.data.preprocessor  import Preprocessor
    from src.features.engineer  import FeatureEngineer
    from src.features.selector  import FeatureSelector

    df, ids_df, _        = load_raw()
    report               = validate(df)
    if not report["ready"]:
        raise RuntimeError("Validation failed")

    train, val, test, _  = split(df)

    prep                 = Preprocessor()
    train_clean          = prep.fit_transform(train)
    val_clean            = prep.transform(val,  split_name="val")
    test_clean           = prep.transform(test, split_name="test")

    eng                  = FeatureEngineer()
    train_fe             = eng.fit_transform(train_clean)
    val_fe               = eng.transform(val_clean,  split_name="val")
    test_fe              = eng.transform(test_clean, split_name="test")

    selector             = FeatureSelector()
    selector.fit(train_fe)
    train_fs             = selector.transform(train_fe, split_name="train")
    val_fs               = selector.transform(val_fe,   split_name="val")
    test_fs              = selector.transform(test_fe,  split_name="test")

    checker              = ConsistencyChecker()
    report               = checker.run(train_fs, val_fs, test_fs)

    print(f"\nPASS={report['passed']}  WARN={report['warned']}  FAIL={report['failed']}")
    print(f"Ready for modeling: {report['ready']}")
    print(f"PSI drifted features: {len(report['drifted_psi'])}")