"""
DriftSentinel — Data Preprocessor
Fits all transformation artifacts on train split only.
Applies fitted artifacts to val and test splits.
Prevents data leakage by strict fit/transform separation.
All decisions grounded in EDA findings from 01_eda.ipynb.

Pipeline order per EDA findings:
    1.  Drop zero-variance and ID columns
    2.  Encode missingness flags (weight, max_glu_serum, A1Cresult)
    3.  ICD-9 chapter grouping (diag_1, diag_2, diag_3)
    4.  Age ordinal encoding
    5.  Medication encoding (ordinal / binary by prescription rate)
    6.  Categorical imputation (medical_specialty, payer_code, race)
    7.  Categorical label encoding
    8.  Log1p transformation (number_outpatient, number_emergency)
    9.  Target encoding (binary + multiclass)
    10. Artifact persistence
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("preprocessor")

# Tier 2C.6 reproducibility: this was a HARDCODED ABSOLUTE PATH to one
# developer's machine, so `pipeline.py` did not reproduce anything from raw
# data on a clean clone -- it read from and wrote to a directory that exists
# nowhere else. On Linux CI the same literal resolves to a RELATIVE folder
# whose name contains backslashes, so artifacts land somewhere harmless-
# looking and the run still 'succeeds'. It worked on exactly one machine,
# which is why nothing caught it. Now derived from this file's location.
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Columns to drop immediately ────────────────────────────────────────────
# ID columns: no predictive value
# Zero-variance medications: EDA confirmed 100% "No"
ID_COLS = ["encounter_id", "patient_nbr"]

ZERO_VAR_MED_COLS = [
    "examide", "citoglipton", "acetohexamide", "troglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
    "glimepiride-pioglitazone"
]

DROP_COLS = ID_COLS + ZERO_VAR_MED_COLS

# ── High-missing columns: encode missingness then drop original ────────────
# EDA: weight delta=12.71pp, max_glu_serum delta=2.63pp, A1Cresult delta=2.59pp
# All three cross the 2pp signal threshold
HIGH_MISS_COLS = ["weight", "max_glu_serum", "A1Cresult"]

# ── Log1p transformation targets ──────────────────────────────────────────
# EDA: number_outpatient skew=8.83 kurt=147.9
#      number_emergency  skew=22.86 kurt=1191.7
LOG1P_COLS = ["number_outpatient", "number_emergency"]

# ── Age ordinal map ────────────────────────────────────────────────────────
# EDA: ordinal structure confirmed, chi2 p=9.3e-56
AGE_ORDINAL = {
    "[0-10)"  : 0, "[10-20)" : 1, "[20-30)" : 2,
    "[30-40)" : 3, "[40-50)" : 4, "[50-60)" : 5,
    "[60-70)" : 6, "[70-80)" : 7, "[80-90)" : 8,
    "[90-100)": 9
}

# ── Medication encoding ────────────────────────────────────────────────────
# EDA prescribed_pct threshold: >= 1% -> ordinal, < 1% -> binary
# Ordinal: No=0, Steady=1, Up=2, Down=3
MED_ORDINAL_MAP = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}

MED_ORDINAL_COLS = [
    "metformin", "repaglinide", "nateglinide", "glimepiride",
    "glipizide", "glyburide", "pioglitazone", "rosiglitazone",
    "insulin"
]

MED_BINARY_COLS = [
    "chlorpropamide", "tolbutamide", "acarbose", "miglitol",
    "tolazamide", "glyburide-metformin", "glipizide-metformin"
]

# ── ICD-9 chapter grouping ─────────────────────────────────────────────────
# Maps raw ICD-9 codes to 18 clinical chapters
# Reduces cardinality from ~800 to 18 interpretable groups
def _icd9_chapter(code: str) -> str:
    if pd.isna(code) or code in ("", "Unknown"):
        return "Unknown"
    code = str(code).strip()
    # V and E codes
    if code.startswith("V"):
        return "Supplementary"
    if code.startswith("E"):
        return "External"
    try:
        num = float(code)
    except ValueError:
        return "Unknown"
    if   1   <= num <= 139:  return "Infectious"
    elif 140 <= num <= 239:  return "Neoplasms"
    elif 240 <= num <= 279:  return "Endocrine"
    elif 280 <= num <= 289:  return "Blood"
    elif 290 <= num <= 319:  return "Mental"
    elif 320 <= num <= 389:  return "Nervous"
    elif 390 <= num <= 459:  return "Circulatory"
    elif 460 <= num <= 519:  return "Respiratory"
    elif 520 <= num <= 579:  return "Digestive"
    elif 580 <= num <= 629:  return "Genitourinary"
    elif 630 <= num <= 679:  return "Pregnancy"
    elif 680 <= num <= 709:  return "Skin"
    elif 710 <= num <= 739:  return "Musculoskeletal"
    elif 740 <= num <= 759:  return "Congenital"
    elif 760 <= num <= 779:  return "Perinatal"
    elif 780 <= num <= 799:  return "Symptoms"
    elif 800 <= num <= 999:  return "Injury"
    else:                    return "Unknown"


# ── Target encoding ────────────────────────────────────────────────────────
# ── Target definition (Tier 2A.1) ─────────────────────────────────────────
#
# PRIMARY TARGET: 30-day readmission.
#
#     {"NO": 0, "<30": 1, ">30": 0}      prevalence 11.16%
#
# This was previously the MERGED target `{"NO":0, "<30":1, ">30":1}` at 46.1%
# prevalence, which is not the clinical task. 30-day readmission is the
# CMS-penalised outcome, the endpoint in Strack et al. (2014), and the endpoint
# in essentially every published model on this dataset. ">30" includes a patient
# readmitted two years later, which carries no operational readmission-risk
# signal (audit F7).
#
# Consequences accepted deliberately: AUC will fall (published 30-day models on
# this dataset sit at roughly 0.63-0.70), and the positive class becomes 4x
# rarer. A defensible result on the real clinical task beats an unremarkable
# result on an easier one nobody studies.
#
# NOTE: the README previously described the target as "readmission within 30
# days" while the code computed the merged version — the label and the
# computation disagreed. They now agree.
#
# The merged target is retained as a documented SECONDARY analysis below.
TARGET_BINARY_MAP = {"NO": 0, "<30": 1, ">30": 0}

# Secondary/legacy view. Deliberately NOT materialised as its own column:
# `readmitted_multi` (0=NO, 1=<30, 2=>30) already carries the full information,
# so the merged target is exactly `readmitted_multi > 0`.
#
# Adding a `readmitted_merged` column would have required updating 20 separate
# target-exclusion sets across 15 modules, and missing ONE would have leaked a
# variant of the old target in as a FEATURE predicting the new one — producing a
# spectacular and entirely fake AUC. Derivation costs nothing and has no leakage
# surface.
TARGET_MERGED_MAP = {"NO": 0, "<30": 1, ">30": 1}   # reference only


def merged_target_from_multi(multi_series):
    """Secondary (superseded) merged target, derived without a new column."""
    return (multi_series > 0).astype(int)

TARGET_MULTI_MAP  = {"NO": 0, "<30": 1, ">30": 2}

# ── Categorical cols for LabelEncoder ─────────────────────────────────────
LABEL_ENCODE_COLS = [
    "race", "gender", "payer_code", "medical_specialty",
    "change", "diabetesMed",
    "diag_1", "diag_2", "diag_3"
]


class Preprocessor:
    """
    Stateful preprocessor.
    fit_transform(train) -> fits all artifacts, transforms train.
    transform(df)         -> applies fitted artifacts only, no re-fitting.
    """

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.impute_modes:   dict[str, str]           = {}
        self.fitted:         bool                     = False

    # ──────────────────────────────────────────────────────────────────────
    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit all artifacts on train split and return transformed train.

        Parameters
        ----------
        train : raw train DataFrame from splitter.split()

        Returns
        -------
        Transformed train DataFrame
        """
        logger.info("=" * 70)
        logger.info("DriftSentinel — Preprocessor  [fit_transform on TRAIN]")
        logger.info("=" * 70)

        df = train.copy()

        # Step 1: Drop columns
        logger.info("-" * 50)
        logger.info("Step 1: Dropping zero-variance and ID columns")
        existing_drop = [c for c in DROP_COLS if c in df.columns]
        df.drop(columns=existing_drop, inplace=True)
        logger.info(f"Dropped {len(existing_drop)} columns: {existing_drop}")
        logger.info(f"Remaining columns: {df.shape[1]}")

        # Step 2: Missingness flags
        logger.info("-" * 50)
        logger.info("Step 2: Encoding missingness flags")
        for col in HIGH_MISS_COLS:
            if col not in df.columns:
                continue
            flag_col = f"{col}_missing"
            df[flag_col] = df[col].isna().astype(int)
            n_missing = df[flag_col].sum()
            logger.info(f"  {flag_col:<35} flagged={n_missing:,} ({n_missing/len(df)*100:.1f}%)")
            df.drop(columns=[col], inplace=True)
            logger.info(f"  Original column '{col}' dropped after flag creation")

        # Step 3: ICD-9 chapter grouping
        logger.info("-" * 50)
        logger.info("Step 3: ICD-9 chapter grouping (diag_1 / diag_2 / diag_3)")
        for col in ["diag_1", "diag_2", "diag_3"]:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna("Unknown").apply(_icd9_chapter)
            dist = df[col].value_counts()
            logger.info(f"  {col} -> {df[col].nunique()} chapters")
            for chapter, cnt in dist.head(5).items():
                logger.info(f"    {chapter:<20} : {cnt:,}")

        # Step 4: Age ordinal encoding
        logger.info("-" * 50)
        logger.info("Step 4: Age ordinal encoding")
        if "age" in df.columns:
            before = df["age"].nunique()
            df["age"] = df["age"].map(AGE_ORDINAL)
            unmapped = df["age"].isna().sum()
            logger.info(f"  age: {before} bins -> ordinal [0-9]  unmapped={unmapped}")

        # Step 5: Medication encoding
        logger.info("-" * 50)
        logger.info("Step 5: Medication encoding")

        for col in MED_ORDINAL_COLS:
            if col not in df.columns:
                continue
            df[col] = df[col].map(MED_ORDINAL_MAP)
            logger.info(f"  {col:<35} ordinal encoded [No=0,Steady=1,Up=2,Down=3]")

        for col in MED_BINARY_COLS:
            if col not in df.columns:
                continue
            df[col] = (df[col] != "No").astype(int)
            prescribed = df[col].sum()
            logger.info(f"  {col:<35} binary encoded  prescribed={prescribed:,} ({prescribed/len(df)*100:.2f}%)")

        # Step 6: Categorical imputation — FIT on train
        logger.info("-" * 50)
        logger.info("Step 6: Categorical imputation (fit on train)")

        impute_cols = {
            "race"               : "mode",
            "medical_specialty" : "Unknown",
            "payer_code"        : "Unknown",
        }

        for col, strategy in impute_cols.items():
            if col not in df.columns:
                continue
            if strategy == "mode":
                mode_val = df[col].mode()[0]
                self.impute_modes[col] = mode_val
                df[col] = df[col].fillna(mode_val)
                logger.info(f"  {col:<25} mode imputed with '{mode_val}'  (fitted on train)")
            else:
                self.impute_modes[col] = strategy
                df[col] = df[col].fillna(strategy)
                logger.info(f"  {col:<25} filled with '{strategy}'")

        # Step 7: LabelEncoder — FIT on train
        logger.info("-" * 50)
        logger.info("Step 7: LabelEncoder (fit on train)")

        for col in LABEL_ENCODE_COLS:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"  {col:<25} {len(le.classes_)} classes  (fitted on train)")

        # Step 8: Log1p transformation
        logger.info("-" * 50)
        logger.info("Step 8: Log1p transformation")

        for col in LOG1P_COLS:
            if col not in df.columns:
                continue
            skew_before = df[col].skew()
            df[col] = np.log1p(df[col])
            skew_after = df[col].skew()
            logger.info(f"  {col:<25} skew: {skew_before:.3f} -> {skew_after:.3f}")

        # Step 9: Target encoding
        logger.info("-" * 50)
        logger.info("Step 9: Target encoding")

        if "readmitted" in df.columns:
            df["readmitted_binary"] = df["readmitted"].map(TARGET_BINARY_MAP)
            df["readmitted_multi"]  = df["readmitted"].map(TARGET_MULTI_MAP)
            df.drop(columns=["readmitted"], inplace=True)

            binary_dist = df["readmitted_binary"].value_counts()
            multi_dist  = df["readmitted_multi"].value_counts()

            logger.info(f"  readmitted_binary: {dict(binary_dist)}")
            logger.info(f"  readmitted_multi : {dict(multi_dist)}")

        self.fitted = True

        # Step 10: Save artifacts
        logger.info("-" * 50)
        logger.info("Step 10: Saving fitted artifacts")
        self._save_artifacts()

        logger.info("=" * 70)
        logger.info(f"fit_transform complete — shape: {df.shape}")
        logger.info("=" * 70)

        return df

    # ──────────────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame, split_name: str = "unknown") -> pd.DataFrame:
        """
        Apply fitted artifacts to val or test split.
        No re-fitting occurs — prevents data leakage.

        Parameters
        ----------
        df         : raw val or test DataFrame from splitter.split()
        split_name : 'val' or 'test' for logging

        Returns
        -------
        Transformed DataFrame
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform(train) first.")

        logger.info("=" * 70)
        logger.info(f"DriftSentinel — Preprocessor  [transform on {split_name.upper()}]")
        logger.info("=" * 70)

        df = df.copy()

        # Step 1: Drop
        existing_drop = [c for c in DROP_COLS if c in df.columns]
        df.drop(columns=existing_drop, inplace=True)
        logger.info(f"Step 1: Dropped {len(existing_drop)} columns")

        # Step 2: Missingness flags
        for col in HIGH_MISS_COLS:
            if col not in df.columns:
                continue
            df[f"{col}_missing"] = df[col].isna().astype(int)
            df.drop(columns=[col], inplace=True)
        logger.info("Step 2: Missingness flags applied")

        # Step 3: ICD-9 grouping
        for col in ["diag_1", "diag_2", "diag_3"]:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna("Unknown").apply(_icd9_chapter)
        logger.info("Step 3: ICD-9 chapter grouping applied")

        # Step 4: Age ordinal
        if "age" in df.columns:
            df["age"] = df["age"].map(AGE_ORDINAL)
        logger.info("Step 4: Age ordinal encoding applied")

        # Step 5: Medication
        for col in MED_ORDINAL_COLS:
            if col not in df.columns:
                continue
            df[col] = df[col].map(MED_ORDINAL_MAP)

        for col in MED_BINARY_COLS:
            if col not in df.columns:
                continue
            df[col] = (df[col] != "No").astype(int)
        logger.info("Step 5: Medication encoding applied")

        # Step 6: Imputation — use TRAIN fitted values
        for col, fill_val in self.impute_modes.items():
            if col not in df.columns:
                continue
            df[col] = df[col].fillna(fill_val)
        logger.info("Step 6: Imputation applied (train-fitted values)")

        # Step 7: LabelEncoder — use TRAIN fitted encoders
        unseen_total = 0
        for col, le in self.label_encoders.items():
            if col not in df.columns:
                continue
            known     = set(le.classes_)
            col_str   = df[col].astype(str)
            unseen_mask = ~col_str.isin(known)
            unseen_cnt  = unseen_mask.sum()

            if unseen_cnt > 0:
                unseen_total += unseen_cnt
                logger.warning(f"  {col}: {unseen_cnt} unseen categories -> mapped to fallback '{le.classes_[0]}'")
                col_str = col_str.where(~unseen_mask, le.classes_[0])

            df[col] = le.transform(col_str)
        
        logger.info(f"Step 7: LabelEncoder applied (train-fitted)  unseen_total={unseen_total}")

        # Step 8: Log1p
        for col in LOG1P_COLS:
            if col not in df.columns:
                continue
            df[col] = np.log1p(df[col])
        logger.info("Step 8: Log1p applied")

        # Step 9: Target
        if "readmitted" in df.columns:
            df["readmitted_binary"] = df["readmitted"].map(TARGET_BINARY_MAP)
            df["readmitted_multi"]  = df["readmitted"].map(TARGET_MULTI_MAP)
            df.drop(columns=["readmitted"], inplace=True)
        logger.info("Step 9: Target encoding applied")

        logger.info("=" * 70)
        logger.info(f"transform complete [{split_name}] — shape: {df.shape}")
        logger.info("=" * 70)

        return df

    # ──────────────────────────────────────────────────────────────────────
    def _save_artifacts(self):
        """Persist fitted artifacts for reproducibility and serving."""

        # LabelEncoders
        le_path = ARTIFACTS_DIR / "label_encoders.pkl"
        with open(le_path, "wb") as f:
            pickle.dump(self.label_encoders, f)
        logger.info(f"  label_encoders.pkl saved -> {le_path}")

        # Impute modes
        mode_path = ARTIFACTS_DIR / "impute_modes.json"
        with open(mode_path, "w") as f:
            json.dump(self.impute_modes, f, indent=2)
        logger.info(f"  impute_modes.json saved -> {mode_path}")

        # Age map
        age_path = ARTIFACTS_DIR / "age_ordinal_map.json"
        with open(age_path, "w") as f:
            json.dump(AGE_ORDINAL, f, indent=2)
        logger.info(f"  age_ordinal_map.json saved -> {age_path}")

        # Column config
        config = {
            "drop_cols"         : DROP_COLS,
            "high_miss_cols"    : HIGH_MISS_COLS,
            "log1p_cols"        : LOG1P_COLS,
            "med_ordinal_cols"  : MED_ORDINAL_COLS,
            "med_binary_cols"   : MED_BINARY_COLS,
            "label_encode_cols" : LABEL_ENCODE_COLS,
            "target_binary_map" : TARGET_BINARY_MAP,
            "target_multi_map"  : TARGET_MULTI_MAP,
        }
        config_path = ARTIFACTS_DIR / "preprocessor_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"  preprocessor_config.json saved -> {config_path}")


if __name__ == "__main__":
    from src.data.loader     import load_raw
    from src.data.validator  import validate
    from src.data.splitter   import split

    df, ids_df, _      = load_raw()
    report             = validate(df)

    if not report["ready"]:
        raise RuntimeError("Validation failed")

    train, val, test, _ = split(df)

    preprocessor = Preprocessor()
    train_clean  = preprocessor.fit_transform(train)
    val_clean    = preprocessor.transform(val,  split_name="val")
    test_clean   = preprocessor.transform(test, split_name="test")

    print(f"\nTrain : {train_clean.shape}")
    print(f"Val   : {val_clean.shape}")
    print(f"Test  : {test_clean.shape}")
    print(f"Columns: {list(train_clean.columns)}")