"""
DriftSentinel — Data Loader
Raw ingestion of Diabetes 130-US Hospitals dataset.
Sentinel replacement (? -> NaN) applied at ingestion time only.
No transformations performed — raw fidelity preserved for validator.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("loader")

DATA_DIR = Path(r"C:\Users\sharg\Desktop\github\DriftSentinel\data\raw\diabetes_hospital")

SENTINEL_VALUE = "?"
EXPECTED_ROWS  = 101_766
EXPECTED_COLS  = 50

REQUIRED_COLUMNS = {
    "encounter_id", "patient_nbr", "race", "gender", "age",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "readmitted",
    "diag_1", "diag_2", "diag_3",
    "weight", "payer_code", "medical_specialty",
    "max_glu_serum", "A1Cresult",
    "metformin", "insulin", "change", "diabetesMed"
}


def load_raw(
    data_path: Path = DATA_DIR / "diabetic_data.csv",
    map_path:  Path = DATA_DIR / "IDS_mapping.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load raw dataset and IDS mapping.

    Parameters
    ----------
    data_path : path to diabetic_data.csv
    map_path  : path to IDS_mapping.csv

    Returns
    -------
    df      : raw DataFrame with ? replaced by NaN
    ids_df  : IDS mapping DataFrame
    meta    : load diagnostics dict
    """
    logger.info("=" * 70)
    logger.info("DriftSentinel — Data Loader")
    logger.info("=" * 70)

    # ── Load main dataset ──────────────────────────────────────────────────
    logger.info(f"Source : {data_path}")
    t0 = time.perf_counter()

    df = pd.read_csv(data_path, na_values=SENTINEL_VALUE, low_memory=False)

    elapsed = time.perf_counter() - t0
    logger.info(f"Loaded in {elapsed:.3f}s")
    logger.info(f"Shape         : {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"Memory        : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # ── Dtype summary ──────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Dtype summary")
    for dtype, count in df.dtypes.value_counts().items():
        logger.info(f"  {str(dtype):<12} : {count} columns")

    # ── Schema check ───────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Schema check")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        logger.error(f"MISSING REQUIRED COLUMNS: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")
    logger.info(f"All {len(REQUIRED_COLUMNS)} required columns present — OK")

    if df.shape[0] != EXPECTED_ROWS:
        logger.warning(f"Row count: expected {EXPECTED_ROWS:,}, got {df.shape[0]:,}")
    else:
        logger.info(f"Row count {df.shape[0]:,} — OK")

    if df.shape[1] != EXPECTED_COLS:
        logger.warning(f"Column count: expected {EXPECTED_COLS}, got {df.shape[1]}")
    else:
        logger.info(f"Column count {df.shape[1]} — OK")

    # ── Sentinel audit ─────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Sentinel replacement audit (? -> NaN)")

    null_summary = df.isna().sum()
    null_cols    = null_summary[null_summary > 0].sort_values(ascending=False)
    total_nulls  = null_summary.sum()

    logger.info(f"Columns with NaN : {len(null_cols)}")
    for col, cnt in null_cols.items():
        logger.info(f"  {col:<30} null={cnt:>6,}  ({cnt/len(df)*100:.2f}%)")
    logger.info(f"Total null cells : {total_nulls:,} / {df.size:,} ({total_nulls/df.size*100:.3f}%)")

    # ── Duplicate check ────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Duplicate integrity")

    dup_enc     = df.duplicated("encounter_id").sum()
    uniq_pat    = df["patient_nbr"].nunique()
    multi_rows  = df[df["patient_nbr"].duplicated(keep=False)].shape[0]

    logger.info(f"Duplicate encounter_id : {dup_enc}")
    logger.info(f"Unique patient_nbr     : {uniq_pat:,}")
    logger.info(f"Multi-visit rows       : {multi_rows:,} ({multi_rows/len(df)*100:.1f}%)")

    if dup_enc > 0:
        logger.error(f"DUPLICATE encounter_ids: {dup_enc} — pipeline halted")
        raise ValueError(f"Duplicate encounter_ids detected: {dup_enc}")
    logger.info("encounter_id uniqueness — OK")

    # ── Target audit ───────────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Target column: readmitted")

    target_dist = df["readmitted"].value_counts()
    for label, cnt in target_dist.items():
        logger.info(f"  {label:<5} : {cnt:>6,}  ({cnt/len(df)*100:.2f}%)")

    if df["readmitted"].isna().any():
        logger.error("NULL values in target column — pipeline halted")
        raise ValueError("Target column contains NaN")
    logger.info("Target column integrity — OK")

    # ── Load IDS mapping ───────────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info(f"Loading IDS mapping: {map_path}")

    ids_df = pd.read_csv(map_path, header=None)
    logger.info(f"IDS mapping shape: {ids_df.shape}")

    # ── Meta ───────────────────────────────────────────────────────────────
    meta = {
        "shape"           : df.shape,
        "memory_mb"       : round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "null_cols"       : null_cols.to_dict(),
        "total_nulls"     : int(total_nulls),
        "dup_encounters"  : int(dup_enc),
        "unique_patients" : int(uniq_pat),
        "multi_visit_rows": int(multi_rows),
        "target_dist"     : target_dist.to_dict(),
        "load_time_s"     : round(elapsed, 3),
    }

    logger.info("=" * 70)
    logger.info("Loader completed successfully")
    logger.info("=" * 70)

    return df, ids_df, meta


if __name__ == "__main__":
    df, ids_df, meta = load_raw()
    print(f"\nDataFrame shape : {df.shape}")
    print(f"IDS mapping     : {ids_df.shape}")
    print(f"Load time       : {meta['load_time_s']}s")