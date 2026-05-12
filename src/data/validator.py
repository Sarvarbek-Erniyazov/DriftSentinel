"""
DriftSentinel — Data Validator
Validates raw DataFrame produced by loader.py before any transformation.
Checks schema, dtypes, value ranges, target integrity, and clinical constraints.
All findings logged to outputs/log/validator.log.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("validator")


# ── Expected schema ────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "encounter_id", "patient_nbr", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

CATEGORICAL_COLS = [
    "race", "gender", "age", "weight", "payer_code", "medical_specialty",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed", "readmitted",
    "diag_1", "diag_2", "diag_3"
]

MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone"
]

VALID_MEDICATION_VALUES = {"No", "Steady", "Up", "Down"}
VALID_TARGET_VALUES     = {"NO", "<30", ">30"}
VALID_GENDER_VALUES     = {"Male", "Female", "Unknown/Invalid"}
VALID_CHANGE_VALUES     = {"No", "Ch"}
VALID_DIABETESMED       = {"Yes", "No"}

# Clinical range constraints (min, max)
CLINICAL_RANGES = {
    "time_in_hospital"   : (1,   14),
    "num_lab_procedures" : (1,  132),
    "num_procedures"     : (0,    6),
    "num_medications"    : (1,   81),
    "number_outpatient"  : (0,   42),
    "number_emergency"   : (0,   76),
    "number_inpatient"   : (0,   21),
    "number_diagnoses"   : (1,   16),
    "admission_type_id"  : (1,    8),
    "discharge_disposition_id": (1, 28),
    "admission_source_id": (1,   26),
}

VALID_AGE_BINS = {
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
}


def validate(df: pd.DataFrame) -> dict:
    """
    Run full validation suite on raw DataFrame.

    Parameters
    ----------
    df : raw DataFrame from loader.load_raw()

    Returns
    -------
    report : dict with pass/fail counts and all findings
    """
    logger.info("=" * 70)
    logger.info("DriftSentinel — Data Validator")
    logger.info("=" * 70)

    findings = []
    passed   = 0
    failed   = 0
    warned   = 0

    def record(level: str, check: str, detail: str):
        nonlocal passed, failed, warned
        entry = {"level": level, "check": check, "detail": detail}
        findings.append(entry)
        if level == "PASS":
            passed += 1
            logger.info(f"  [PASS]  {check:<45} {detail}")
        elif level == "FAIL":
            failed += 1
            logger.error(f"  [FAIL]  {check:<45} {detail}")
        elif level == "WARN":
            warned += 1
            logger.warning(f"  [WARN]  {check:<45} {detail}")

    # ── 1. Shape ───────────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("1. Shape & Memory")

    record("PASS" if df.shape[0] == 101_766 else "FAIL",
           "row_count",
           f"{df.shape[0]:,} rows (expected 101,766)")

    record("PASS" if df.shape[1] == 50 else "FAIL",
           "column_count",
           f"{df.shape[1]} columns (expected 50)")

    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    record("PASS", "memory_usage", f"{mem_mb:.2f} MB")

    # ── 2. Column presence ─────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("2. Column Presence")

    all_expected = set(NUMERIC_COLS + CATEGORICAL_COLS + MEDICATION_COLS)
    missing = all_expected - set(df.columns)
    extra   = set(df.columns) - all_expected - {"readmitted"}

    record("PASS" if not missing else "FAIL",
           "required_columns_present",
           f"missing={missing if missing else 'none'}")

    record("PASS" if not extra else "WARN",
           "unexpected_columns",
           f"extra={extra if extra else 'none'}")

    # ── 3. Dtype checks ────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("3. Dtype Validation")

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        ok = pd.api.types.is_integer_dtype(df[col])
        record("PASS" if ok else "FAIL",
               f"dtype_{col}",
               f"{df[col].dtype} ({'int64 expected' if not ok else 'OK'})")

    for col in CATEGORICAL_COLS + MEDICATION_COLS:
        if col not in df.columns:
            continue
        ok = pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
        record("PASS" if ok else "WARN",
           f"dtype_{col}",
           f"{df[col].dtype} (OK)" if ok else f"{df[col].dtype} (object/str expected)")

    # ── 4. Null audit ──────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("4. Null Value Audit")

    null_counts = df.isna().sum()

    # Target must never be null
    target_nulls = null_counts["readmitted"]
    record("PASS" if target_nulls == 0 else "FAIL",
           "target_null_free",
           f"readmitted nulls={target_nulls}")

    # Medication cols must never be null
    med_nulls = null_counts[MEDICATION_COLS].sum()
    record("PASS" if med_nulls == 0 else "FAIL",
           "medication_null_free",
           f"total medication nulls={med_nulls}")

    # Numeric cols must never be null
    num_nulls = null_counts[NUMERIC_COLS].sum()
    record("PASS" if num_nulls == 0 else "FAIL",
           "numeric_null_free",
           f"total numeric nulls={num_nulls}")

    # High-missing columns — expected
    for col, expected_pct in [("weight", 96.0), ("max_glu_serum", 94.0), ("A1Cresult", 83.0)]:
        if col not in df.columns:
            continue
        actual_pct = null_counts[col] / len(df) * 100
        ok = actual_pct >= expected_pct - 2.0
        record("PASS" if ok else "WARN",
               f"expected_missing_{col}",
               f"null={actual_pct:.1f}% (expected ~{expected_pct:.0f}%)")

    # ── 5. Clinical range checks ───────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("5. Clinical Range Validation")

    for col, (lo, hi) in CLINICAL_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        record("PASS" if out_of_range == 0 else "WARN",
               f"range_{col}",
               f"[{lo}, {hi}]  out_of_range={out_of_range}")

    # ── 6. Categorical value sets ──────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("6. Categorical Value Integrity")

    # Target
    actual_target = set(df["readmitted"].dropna().unique())
    record("PASS" if actual_target <= VALID_TARGET_VALUES else "FAIL",
           "target_values",
           f"found={actual_target}")

    # Gender
    actual_gender = set(df["gender"].dropna().unique())
    record("PASS" if actual_gender <= VALID_GENDER_VALUES else "WARN",
           "gender_values",
           f"found={actual_gender}")

    # Change
    actual_change = set(df["change"].dropna().unique())
    record("PASS" if actual_change <= VALID_CHANGE_VALUES else "WARN",
           "change_values",
           f"found={actual_change}")

    # diabetesMed
    actual_dm = set(df["diabetesMed"].dropna().unique())
    record("PASS" if actual_dm <= VALID_DIABETESMED else "WARN",
           "diabetesMed_values",
           f"found={actual_dm}")

    # Age bins
    actual_age = set(df["age"].dropna().unique())
    unexpected_age = actual_age - VALID_AGE_BINS
    record("PASS" if not unexpected_age else "WARN",
           "age_bin_values",
           f"unexpected={unexpected_age if unexpected_age else 'none'}")

    # Medication value sets
    med_violations = {}
    for col in MEDICATION_COLS:
        if col not in df.columns:
            continue
        bad = set(df[col].dropna().unique()) - VALID_MEDICATION_VALUES
        if bad:
            med_violations[col] = bad

    record("PASS" if not med_violations else "FAIL",
           "medication_value_sets",
           f"violations={med_violations if med_violations else 'none'}")

    # ── 7. Zero-variance columns ───────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("7. Zero-Variance Detection")

    zero_var = [c for c in MEDICATION_COLS if df[c].nunique() <= 1]
    record("WARN" if zero_var else "PASS",
           "zero_variance_medication_cols",
           f"cols={zero_var if zero_var else 'none'}  count={len(zero_var)}")

    # ── 8. Duplicate check ─────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("8. Duplicate Integrity")

    dup_enc = df.duplicated("encounter_id").sum()
    record("PASS" if dup_enc == 0 else "FAIL",
           "encounter_id_unique",
           f"duplicates={dup_enc}")

    uniq_patients = df["patient_nbr"].nunique()
    multi_visit   = (df["patient_nbr"].value_counts() > 1).sum()
    record("PASS",
           "patient_nbr_multi_visit",
           f"unique_patients={uniq_patients:,}  multi_visit={multi_visit:,} ({multi_visit/uniq_patients*100:.1f}%)")

    # ── 9. encounter_id ordering check ────────────────────────────────────
    logger.info("-" * 50)
    logger.info("9. encounter_id Monotonicity (drift split prerequisite)")

    is_monotonic = df["encounter_id"].is_monotonic_increasing
    record("PASS" if is_monotonic else "WARN",
           "encounter_id_monotonic",
           f"monotonic_increasing={is_monotonic} (required for temporal split)")

    enc_min = df["encounter_id"].min()
    enc_max = df["encounter_id"].max()
    logger.info(f"  encounter_id range: {enc_min:,} — {enc_max:,}")

    # ── 10. Target class balance ───────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("10. Target Class Balance")

    dist = df["readmitted"].value_counts(normalize=True) * 100
    for label, pct in dist.items():
        logger.info(f"  {label:<5} : {pct:.2f}%")

    minority_pct = dist.min()
    record("PASS" if minority_pct > 5 else "WARN",
           "target_minority_class",
           f"minority={minority_pct:.2f}% ({'severe imbalance' if minority_pct < 5 else 'manageable'})")

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Validation Summary")
    logger.info(f"  PASS  : {passed}")
    logger.info(f"  WARN  : {warned}")
    logger.info(f"  FAIL  : {failed}")

    if failed > 0:
        logger.error(f"Validation FAILED with {failed} critical issue(s) — pipeline should not proceed")
    elif warned > 0:
        logger.warning(f"Validation passed with {warned} warning(s) — review before proceeding")
    else:
        logger.info("Validation PASSED — data is clean and ready for splitting")

    logger.info("=" * 70)

    report = {
        "passed"   : passed,
        "warned"   : warned,
        "failed"   : failed,
        "findings" : findings,
        "ready"    : failed == 0,
    }

    return report


if __name__ == "__main__":
    from src.data.loader import load_raw
    df, ids_df, meta = load_raw()
    report = validate(df)
    print(f"\nReady for split: {report['ready']}")
    print(f"PASS={report['passed']}  WARN={report['warned']}  FAIL={report['failed']}")