"""
DriftSentinel — Data Splitter
Patient-level stratified split to prevent label leakage.
encounter_id ordering preserved for drift simulation window.
All split artifacts logged to outputs/log/splitter.log.

Split strategy:
    - Group by patient_nbr (patient-level, not encounter-level)
    - Order patients by their FIRST encounter_id (temporal proxy)
    - 60% reference window  -> train
    - 20% production window -> val
    - 20% production window -> test
    - Same patient NEVER appears in more than one split
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("splitter")

ARTIFACTS_DIR = Path(r"C:\Users\sharg\Desktop\github\DriftSentinel\outputs\artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20
RANDOM_SEED = 42


def _get_patient_order(df: pd.DataFrame) -> pd.Series:
    """
    Order patients by their first encounter_id.
    This preserves temporal structure without requiring explicit timestamps.

    Returns
    -------
    pd.Series : patient_nbr sorted by first encounter_id ascending
    """
    return (
        df.groupby("patient_nbr")["encounter_id"]
        .min()
        .sort_values(ascending=True)
        .index
    )


def _compute_drift_stats(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> dict:
    """
    Compute basic distributional statistics across splits.
    Used to verify drift simulation is realistic.
    """
    stats = {}
    numeric_cols = [
        "time_in_hospital", "num_lab_procedures", "num_medications",
        "number_inpatient", "number_emergency", "number_outpatient"
    ]

    for col in numeric_cols:
        if col not in train.columns:
            continue
        stats[col] = {
            "train_mean" : round(train[col].mean(), 4),
            "val_mean"   : round(val[col].mean(),   4),
            "test_mean"  : round(test[col].mean(),  4),
            "train_std"  : round(train[col].std(),  4),
            "val_std"    : round(val[col].std(),    4),
            "test_std"   : round(test[col].std(),   4),
        }

    return stats


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Patient-level temporal split of raw DataFrame.

    Parameters
    ----------
    df : validated raw DataFrame from validator.validate()

    Returns
    -------
    train : 60% of patients (reference window)
    val   : 20% of patients (first production window)
    test  : 20% of patients (second production window)
    meta  : split diagnostics dict
    """
    logger.info("=" * 70)
    logger.info("DriftSentinel — Data Splitter")
    logger.info("=" * 70)
    logger.info(f"Strategy       : patient-level temporal split")
    logger.info(f"Ordering       : first encounter_id per patient (ascending)")
    logger.info(f"Ratios         : train={TRAIN_RATIO} / val={VAL_RATIO} / test={TEST_RATIO}")
    logger.info(f"Random seed    : {RANDOM_SEED}")

    # ── Step 1: patient ordering ───────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Step 1: Patient ordering by first encounter_id")

    ordered_patients = _get_patient_order(df)
    n_patients       = len(ordered_patients)

    logger.info(f"Total unique patients : {n_patients:,}")
    logger.info(f"First patient enc_id  : {df[df['patient_nbr'] == ordered_patients[0]]['encounter_id'].min():,}")
    logger.info(f"Last patient enc_id   : {df[df['patient_nbr'] == ordered_patients[-1]]['encounter_id'].min():,}")

    # ── Step 2: patient-level cutoffs ──────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Step 2: Computing patient-level split cutoffs")

    n_train = int(np.floor(n_patients * TRAIN_RATIO))
    n_val   = int(np.floor(n_patients * VAL_RATIO))
    n_test  = n_patients - n_train - n_val

    logger.info(f"Train patients : {n_train:,} ({n_train/n_patients*100:.1f}%)")
    logger.info(f"Val patients   : {n_val:,}   ({n_val/n_patients*100:.1f}%)")
    logger.info(f"Test patients  : {n_test:,}   ({n_test/n_patients*100:.1f}%)")

    train_patients = set(ordered_patients[:n_train])
    val_patients   = set(ordered_patients[n_train:n_train + n_val])
    test_patients  = set(ordered_patients[n_train + n_val:])

    # ── Step 3: leakage verification ───────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Step 3: Patient leakage verification")

    train_val_overlap  = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap   = val_patients   & test_patients

    logger.info(f"Train ∩ Val  : {len(train_val_overlap)}  (must be 0)")
    logger.info(f"Train ∩ Test : {len(train_test_overlap)}  (must be 0)")
    logger.info(f"Val   ∩ Test : {len(val_test_overlap)}  (must be 0)")

    if any([train_val_overlap, train_test_overlap, val_test_overlap]):
        logger.error("PATIENT LEAKAGE DETECTED — pipeline halted")
        raise ValueError("Patient leakage detected in split")
    logger.info("Patient leakage check — PASSED")

    # ── Step 4: row assignment ─────────────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Step 4: Row assignment to splits")

    train = df[df["patient_nbr"].isin(train_patients)].copy()
    val   = df[df["patient_nbr"].isin(val_patients)].copy()
    test  = df[df["patient_nbr"].isin(test_patients)].copy()

    total_assigned = len(train) + len(val) + len(test)

    logger.info(f"Train rows : {len(train):,}  ({len(train)/len(df)*100:.1f}%)")
    logger.info(f"Val rows   : {len(val):,}   ({len(val)/len(df)*100:.1f}%)")
    logger.info(f"Test rows  : {len(test):,}   ({len(test)/len(df)*100:.1f}%)")
    logger.info(f"Total      : {total_assigned:,} / {len(df):,} (all rows accounted: {total_assigned == len(df)})")

    if total_assigned != len(df):
        logger.error(f"Row count mismatch: {total_assigned} != {len(df)}")
        raise ValueError("Row count mismatch after split")
    logger.info("Row count integrity — PASSED")

    # ── Step 5: encounter_id ordering within splits ────────────────────────
    logger.info("-" * 50)
    logger.info("Step 5: Preserving encounter_id order within splits")

    train = train.sort_values("encounter_id").reset_index(drop=True)
    val   = val.sort_values("encounter_id").reset_index(drop=True)
    test  = test.sort_values("encounter_id").reset_index(drop=True)

    logger.info(f"Train enc_id range : {train['encounter_id'].min():,} — {train['encounter_id'].max():,}")
    logger.info(f"Val   enc_id range : {val['encounter_id'].min():,} — {val['encounter_id'].max():,}")
    logger.info(f"Test  enc_id range : {test['encounter_id'].min():,} — {test['encounter_id'].max():,}")

    # ── Step 6: target distribution across splits ──────────────────────────
    logger.info("-" * 50)
    logger.info("Step 6: Target distribution across splits")

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        dist = split_df["readmitted"].value_counts(normalize=True) * 100
        logger.info(f"  {name:<6} : " + "  ".join(
            [f"{k}={v:.1f}%" for k, v in dist.items()]
        ))

    # ── Step 7: drift simulation stats ────────────────────────────────────
    logger.info("-" * 50)
    logger.info("Step 7: Distributional shift across splits (drift simulation check)")

    drift_stats = _compute_drift_stats(train, val, test)
    for col, s in drift_stats.items():
        delta_val  = abs(s["val_mean"]  - s["train_mean"])
        delta_test = abs(s["test_mean"] - s["train_mean"])
        logger.info(
            f"  {col:<25} "
            f"train_mean={s['train_mean']:.3f}  "
            f"val_mean={s['val_mean']:.3f} (Δ={delta_val:.3f})  "
            f"test_mean={s['test_mean']:.3f} (Δ={delta_test:.3f})"
        )

    # ── Step 8: save split index artifacts ────────────────────────────────
    logger.info("-" * 50)
    logger.info("Step 8: Saving split index artifacts")

    split_index = {
        "train_patient_ids" : sorted(list(train_patients)),
        "val_patient_ids"   : sorted(list(val_patients)),
        "test_patient_ids"  : sorted(list(test_patients)),
        "n_train_patients"  : n_train,
        "n_val_patients"    : n_val,
        "n_test_patients"   : n_test,
        "n_train_rows"      : len(train),
        "n_val_rows"        : len(val),
        "n_test_rows"       : len(test),
        "train_ratio"       : TRAIN_RATIO,
        "val_ratio"         : VAL_RATIO,
        "test_ratio"        : TEST_RATIO,
        "random_seed"       : RANDOM_SEED,
    }

    index_path = ARTIFACTS_DIR / "split_index.json"
    with open(index_path, "w") as f:
        json.dump(split_index, f, indent=2)
    logger.info(f"Split index saved: {index_path}")

    # ── Meta ───────────────────────────────────────────────────────────────
    meta = {
        "n_patients"       : n_patients,
        "n_train_patients" : n_train,
        "n_val_patients"   : n_val,
        "n_test_patients"  : n_test,
        "n_train_rows"     : len(train),
        "n_val_rows"       : len(val),
        "n_test_rows"      : len(test),
        "leakage_free"     : True,
        "drift_stats"      : drift_stats,
    }

    logger.info("=" * 70)
    logger.info("Splitter completed successfully")
    logger.info(f"Artifacts saved to: {ARTIFACTS_DIR}")
    logger.info("=" * 70)

    return train, val, test, meta


if __name__ == "__main__":
    from src.data.loader    import load_raw
    from src.data.validator import validate

    df, ids_df, loader_meta  = load_raw()
    report                   = validate(df)

    if not report["ready"]:
        raise RuntimeError("Validation failed — cannot split")

    train, val, test, meta = split(df)

    print(f"\nTrain : {meta['n_train_rows']:,} rows  /  {meta['n_train_patients']:,} patients")
    print(f"Val   : {meta['n_val_rows']:,} rows  /  {meta['n_val_patients']:,} patients")
    print(f"Test  : {meta['n_test_rows']:,} rows  /  {meta['n_test_patients']:,} patients")
    print(f"Leakage free: {meta['leakage_free']}")