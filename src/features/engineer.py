"""
DriftSentinel — Feature Engineer
Creates domain-grounded features from preprocessed Diabetes 130-US data.
All engineered features prefixed with FE_ for downstream traceability.
Fitted on train only — transform applied to val/test without re-fitting.
No target column used in any feature construction (leakage-free by design).

Clinical domain rationale per feature group:
    - Utilization intensity  : how heavily patient uses healthcare system
    - Medication complexity  : polypharmacy burden and treatment aggressiveness
    - Diagnosis burden       : comorbidity load
    - Lab & procedure load   : diagnostic intensity during admission
    - Prior utilization      : historical contact pattern before this encounter
    - Diabetes management    : insulin/medication change signals
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("engineer")

ARTIFACTS_DIR = Path(r"C:\Users\sharg\Desktop\github\DriftSentinel\outputs\artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Target columns — NEVER used in feature construction ───────────────────
TARGET_COLS = {"readmitted_binary", "readmitted_multi", "readmitted"}

# ── Medication ordinal columns (after preprocessor encoding) ───────────────
# Values: No=0, Steady=1, Up=2, Down=3
MED_ORDINAL_COLS = [
    "metformin", "repaglinide", "nateglinide", "glimepiride",
    "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "insulin"
]

# ── Medication binary columns (after preprocessor encoding) ───────────────
MED_BINARY_COLS = [
    "chlorpropamide", "tolbutamide", "acarbose", "miglitol",
    "tolazamide", "glyburide-metformin", "glipizide-metformin"
]


class FeatureEngineer:
    """
    Stateful feature engineer.
    fit_transform(train) -> fits any train-derived statistics, transforms train.
    transform(df)        -> applies fitted statistics only, no re-fitting.
    """

    def __init__(self):
        self.train_stats: dict = {}
        self.fitted: bool      = False
        self.feature_names: list[str] = []

    # ──────────────────────────────────────────────────────────────────────
    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on train and return engineered train DataFrame.

        Parameters
        ----------
        train : preprocessed train DataFrame

        Returns
        -------
        DataFrame with FE_ features appended
        """
        logger.info("=" * 70)
        logger.info("DriftSentinel — Feature Engineer  [fit_transform on TRAIN]")
        logger.info("=" * 70)
        logger.info(f"Input shape  : {train.shape}")

        df = train.copy()
        fe_cols_before = [c for c in df.columns if c.startswith("FE_")]
        logger.info(f"Existing FE_ columns: {len(fe_cols_before)}")

        df = self._group_A_utilization_intensity(df, fit=True)
        df = self._group_B_medication_complexity(df, fit=True)
        df = self._group_C_diagnosis_burden(df, fit=True)
        df = self._group_D_lab_procedure_load(df, fit=True)
        df = self._group_E_prior_utilization(df, fit=True)
        df = self._group_F_diabetes_management(df, fit=True)
        df = self._group_G_interaction_features(df, fit=True)

        self.fitted = True
        self.feature_names = [c for c in df.columns if c.startswith("FE_")]

        self._null_audit(df, split="train")
        self._leakage_guard(df)
        self._save_artifacts()

        logger.info("=" * 70)
        logger.info(f"fit_transform complete — shape: {df.shape}")
        logger.info(f"FE_ features created: {len(self.feature_names)}")
        logger.info("=" * 70)

        return df

    # ──────────────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame, split_name: str = "unknown") -> pd.DataFrame:
        """
        Apply fitted artifacts to val or test split.
        No statistics recomputed from df — train stats used throughout.

        Parameters
        ----------
        df         : preprocessed val or test DataFrame
        split_name : 'val' or 'test' for logging

        Returns
        -------
        DataFrame with FE_ features appended
        """
        if not self.fitted:
            raise RuntimeError("FeatureEngineer not fitted. Call fit_transform(train) first.")

        logger.info("=" * 70)
        logger.info(f"DriftSentinel — Feature Engineer  [transform on {split_name.upper()}]")
        logger.info("=" * 70)
        logger.info(f"Input shape  : {df.shape}")

        df = df.copy()
        df = self._group_A_utilization_intensity(df, fit=False)
        df = self._group_B_medication_complexity(df, fit=False)
        df = self._group_C_diagnosis_burden(df, fit=False)
        df = self._group_D_lab_procedure_load(df, fit=False)
        df = self._group_E_prior_utilization(df, fit=False)
        df = self._group_F_diabetes_management(df, fit=False)
        df = self._group_G_interaction_features(df, fit=False)

        self._null_audit(df, split=split_name)

        logger.info("=" * 70)
        logger.info(f"transform complete [{split_name}] — shape: {df.shape}")
        logger.info("=" * 70)

        return df

    # ══════════════════════════════════════════════════════════════════════
    # GROUP A — Utilization Intensity
    # Clinical rationale: patients who stay longer, take more medications,
    # and undergo more procedures have higher readmission risk.
    # All inputs are numeric — no target information used.
    # ══════════════════════════════════════════════════════════════════════
    def _group_A_utilization_intensity(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        logger.info("-" * 50)
        logger.info("Group A: Utilization Intensity")

        # Total clinical contacts per admission
        # time_in_hospital + num_procedures + num_lab_procedures
        df["FE_total_clinical_contacts"] = (
            df["time_in_hospital"] +
            df["num_procedures"] +
            df["num_lab_procedures"]
        )
        logger.info("  FE_total_clinical_contacts : time + procedures + labs")

        # Lab procedures per hospital day
        # Measures diagnostic intensity normalized by stay length
        df["FE_labs_per_day"] = (
            df["num_lab_procedures"] /
            df["time_in_hospital"].replace(0, 1)
        ).round(4)
        logger.info("  FE_labs_per_day            : num_lab_procedures / time_in_hospital")

        # Medications per hospital day
        # Polypharmacy burden normalized by stay duration
        df["FE_meds_per_day"] = (
            df["num_medications"] /
            df["time_in_hospital"].replace(0, 1)
        ).round(4)
        logger.info("  FE_meds_per_day            : num_medications / time_in_hospital")

        # Procedure density
        # num_procedures relative to num_lab_procedures
        # High ratio: more invasive procedures per diagnostic test
        df["FE_procedure_density"] = (
            df["num_procedures"] /
            (df["num_lab_procedures"] + 1)
        ).round(4)
        logger.info("  FE_procedure_density       : num_procedures / (num_lab_procedures + 1)")

        # High utilization flag
        # Patients in top clinical contact tier
        if fit:
            threshold = df["FE_total_clinical_contacts"].quantile(0.75)
            self.train_stats["utilization_high_threshold"] = threshold
            logger.info(f"  FE_high_utilization threshold (Q75, train) = {threshold:.2f}")
        else:
            threshold = self.train_stats["utilization_high_threshold"]

        df["FE_high_utilization"] = (
            df["FE_total_clinical_contacts"] >= threshold
        ).astype(int)
        logger.info(f"  FE_high_utilization        : >= {threshold:.2f}  n={df['FE_high_utilization'].sum():,}")

        return df

    # ══════════════════════════════════════════════════════════════════════
    # GROUP B — Medication Complexity
    # Clinical rationale: number of active medications and presence of
    # dose changes indicate treatment instability — a known readmission risk.
    # Medication change (Up/Down) vs Steady is clinically meaningful.
    # ══════════════════════════════════════════════════════════════════════
    def _group_B_medication_complexity(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        logger.info("-" * 50)
        logger.info("Group B: Medication Complexity")

        # Count of active medications (ordinal > 0 means prescribed)
        ordinal_present = [c for c in MED_ORDINAL_COLS if c in df.columns]
        binary_present  = [c for c in MED_BINARY_COLS  if c in df.columns]

        df["FE_n_active_ordinal_meds"] = (
            df[ordinal_present].gt(0).sum(axis=1)
        )
        logger.info(f"  FE_n_active_ordinal_meds   : sum of {len(ordinal_present)} ordinal meds > 0")

        df["FE_n_active_binary_meds"] = df[binary_present].sum(axis=1)
        logger.info(f"  FE_n_active_binary_meds    : sum of {len(binary_present)} binary meds")

        df["FE_total_active_meds"] = (
            df["FE_n_active_ordinal_meds"] + df["FE_n_active_binary_meds"]
        )
        logger.info("  FE_total_active_meds       : ordinal_active + binary_active")

        # Dose change count — Up=2 or Down=3 in ordinal encoding
        df["FE_n_med_changes"] = (
            df[ordinal_present].isin([2, 3]).sum(axis=1)
        )
        logger.info("  FE_n_med_changes           : meds with Up or Down dose")

        # Dose increase count (Up=2)
        df["FE_n_med_increases"] = (
            df[ordinal_present].eq(2).sum(axis=1)
        )
        logger.info("  FE_n_med_increases         : meds with Up dose only")

        # Dose decrease count (Down=3)
        df["FE_n_med_decreases"] = (
            df[ordinal_present].eq(3).sum(axis=1)
        )
        logger.info("  FE_n_med_decreases         : meds with Down dose only")

        # Any medication changed flag
        df["FE_any_med_changed"] = (
            df["FE_n_med_changes"] > 0
        ).astype(int)
        logger.info("  FE_any_med_changed         : binary flag, at least one dose change")

        # Polypharmacy flag — clinical threshold >= 5 active meds
        df["FE_polypharmacy"] = (
            df["FE_total_active_meds"] >= 5
        ).astype(int)
        logger.info("  FE_polypharmacy            : total_active_meds >= 5 (clinical threshold)")

        # Medication to num_medications ratio
        # Cross-check between raw count and engineered count
        df["FE_med_coverage_ratio"] = (
            df["FE_total_active_meds"] /
            (df["num_medications"] + 1)
        ).round(4)
        logger.info("  FE_med_coverage_ratio      : total_active_meds / (num_medications + 1)")

        return df

    # ══════════════════════════════════════════════════════════════════════
    # GROUP C — Diagnosis Burden
    # Clinical rationale: multiple concurrent diagnoses (comorbidities)
    # increase complexity of care and readmission probability.
    # number_diagnoses is the raw count — augmented with derived signals.
    # ══════════════════════════════════════════════════════════════════════
    def _group_C_diagnosis_burden(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        logger.info("-" * 50)
        logger.info("Group C: Diagnosis Burden")

        # High comorbidity flag
        # number_diagnoses >= 7 indicates complex multi-morbid patient
        if fit:
            threshold = df["number_diagnoses"].quantile(0.75)
            self.train_stats["diagnosis_high_threshold"] = threshold
            logger.info(f"  FE_high_comorbidity threshold (Q75, train) = {threshold:.2f}")
        else:
            threshold = self.train_stats["diagnosis_high_threshold"]

        df["FE_high_comorbidity"] = (
            df["number_diagnoses"] >= threshold
        ).astype(int)
        logger.info(f"  FE_high_comorbidity        : number_diagnoses >= {threshold:.0f}")

        # Diagnosis density per hospital day
        df["FE_diagnoses_per_day"] = (
            df["number_diagnoses"] /
            df["time_in_hospital"].replace(0, 1)
        ).round(4)
        logger.info("  FE_diagnoses_per_day       : number_diagnoses / time_in_hospital")

        # Diagnosis to medication ratio
        # High ratio: many diagnoses managed with few meds -> under-treatment signal
        df["FE_diag_med_ratio"] = (
            df["number_diagnoses"] /
            (df["num_medications"] + 1)
        ).round(4)
        logger.info("  FE_diag_med_ratio          : number_diagnoses / (num_medications + 1)")

        return df

    # ══════════════════════════════════════════════════════════════════════
    # GROUP D — Lab & Procedure Load
    # Clinical rationale: high lab count indicates diagnostic uncertainty
    # or monitoring of unstable conditions.
    # ══════════════════════════════════════════════════════════════════════
    def _group_D_lab_procedure_load(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        logger.info("-" * 50)
        logger.info("Group D: Lab & Procedure Load")

        # High lab flag — above train Q75
        if fit:
            threshold = df["num_lab_procedures"].quantile(0.75)
            self.train_stats["lab_high_threshold"] = threshold
            logger.info(f"  FE_high_lab_load threshold (Q75, train) = {threshold:.2f}")
        else:
            threshold = self.train_stats["lab_high_threshold"]

        df["FE_high_lab_load"] = (
            df["num_lab_procedures"] >= threshold
        ).astype(int)
        logger.info(f"  FE_high_lab_load           : num_lab_procedures >= {threshold:.0f}")

        # Procedure-free flag — no procedures performed
        # May indicate observation-only admission
        df["FE_no_procedures"] = (
            df["num_procedures"] == 0
        ).astype(int)
        logger.info("  FE_no_procedures           : num_procedures == 0")

        # Lab to procedure ratio
        # Very high ratio: heavy diagnostic, low intervention
        df["FE_lab_to_procedure_ratio"] = (
            df["num_lab_procedures"] /
            (df["num_procedures"] + 1)
        ).round(4)
        logger.info("  FE_lab_to_procedure_ratio  : num_lab_procedures / (num_procedures + 1)")

        return df

    # ══════════════════════════════════════════════════════════════════════
    # GROUP E — Prior Utilization
    # Clinical rationale: patients with prior emergency and inpatient
    # visits have demonstrated healthcare-seeking patterns that predict
    # future readmission. EDA: number_inpatient H=5662 (highest KW stat).
    # ══════════════════════════════════════════════════════════════════════
    def _group_E_prior_utilization(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        logger.info("-" * 50)
        logger.info("Group E: Prior Utilization")

        # Total prior contacts across all channels
        # log1p already applied to outpatient and emergency in preprocessor
        df["FE_total_prior_contacts"] = (
            df["number_inpatient"] +
            df["number_outpatient"] +
            df["number_emergency"]
        )
        logger.info("  FE_total_prior_contacts    : inpatient + outpatient + emergency")

        # Prior emergency flag — any emergency visit in prior year
        # EDA: number_emergency KW H=1651 — strong signal
        df["FE_has_prior_emergency"] = (
            df["number_emergency"] > 0
        ).astype(int)
        logger.info("  FE_has_prior_emergency     : number_emergency > 0")

        # Prior inpatient flag — any inpatient visit in prior year
        # EDA: number_inpatient KW H=5662 — strongest numeric signal
        df["FE_has_prior_inpatient"] = (
            df["number_inpatient"] > 0
        ).astype(int)
        logger.info("  FE_has_prior_inpatient     : number_inpatient > 0")

        # High prior utilization flag
        if fit:
            threshold = df["FE_total_prior_contacts"].quantile(0.75)
            self.train_stats["prior_util_high_threshold"] = threshold
            logger.info(f"  FE_high_prior_utilization threshold (Q75, train) = {threshold:.2f}")
        else:
            threshold = self.train_stats["prior_util_high_threshold"]

        df["FE_high_prior_utilization"] = (
            df["FE_total_prior_contacts"] >= threshold
        ).astype(int)
        logger.info(f"  FE_high_prior_utilization  : total_prior_contacts >= {threshold:.2f}")

        # Multi-channel utilizer — uses 2+ channels
        df["FE_multi_channel_utilizer"] = (
            (df["number_inpatient"] > 0).astype(int) +
            (df["number_outpatient"] > 0).astype(int) +
            (df["number_emergency"] > 0).astype(int)
        )
        logger.info("  FE_multi_channel_utilizer  : count of channels with > 0 visits")

        return df

    # ══════════════════════════════════════════════════════════════════════
    # GROUP F — Diabetes Management
    # Clinical rationale: insulin usage and medication change patterns
    # are direct indicators of glycemic control quality.
    # Poor glycemic control is the primary driver of readmission.
    # ══════════════════════════════════════════════════════════════════════
    def _group_F_diabetes_management(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        logger.info("-" * 50)
        logger.info("Group F: Diabetes Management")

        # Insulin prescribed flag
        # EDA: insulin prescribed_pct=53.4% — most common medication
        if "insulin" in df.columns:
            df["FE_insulin_prescribed"] = (
                df["insulin"] > 0
            ).astype(int)
            logger.info("  FE_insulin_prescribed      : insulin > 0 (Steady/Up/Down)")

            # Insulin dose changed flag
            df["FE_insulin_changed"] = (
                df["insulin"].isin([2, 3])
            ).astype(int)
            logger.info("  FE_insulin_changed         : insulin Up or Down")

            # Insulin dose increased
            df["FE_insulin_increased"] = (
                df["insulin"] == 2
            ).astype(int)
            logger.info("  FE_insulin_increased       : insulin Up only")

        # Metformin prescribed flag
        # EDA: metformin prescribed_pct=19.6%
        if "metformin" in df.columns:
            df["FE_metformin_prescribed"] = (
                df["metformin"] > 0
            ).astype(int)
            logger.info("  FE_metformin_prescribed    : metformin > 0")

            df["FE_metformin_changed"] = (
                df["metformin"].isin([2, 3])
            ).astype(int)
            logger.info("  FE_metformin_changed       : metformin Up or Down")

        # Diabetes medication flag — from raw preprocessed col
        if "diabetesMed" in df.columns:
            df["FE_on_diabetes_med"] = df["diabetesMed"]
            logger.info("  FE_on_diabetes_med         : diabetesMed (pass-through alias)")

        # Medication change flag — from raw preprocessed col
        if "change" in df.columns:
            df["FE_med_regimen_changed"] = df["change"]
            logger.info("  FE_med_regimen_changed     : change (pass-through alias)")

        # Glycemic risk score
        # Composite: insulin + missing A1C + missing glucose
        components = []
        if "FE_insulin_prescribed" in df.columns:
            components.append(df["FE_insulin_prescribed"])
        if "A1Cresult_missing" in df.columns:
            components.append(df["A1Cresult_missing"])
        if "max_glu_serum_missing" in df.columns:
            components.append(df["max_glu_serum_missing"])

        if components:
            df["FE_glycemic_risk_score"] = sum(components)
            logger.info(f"  FE_glycemic_risk_score     : sum of {len(components)} glycemic indicators")

        return df

    # ══════════════════════════════════════════════════════════════════════
    # GROUP G — Interaction Features
    # Clinical rationale: combined signals capture non-linear relationships
    # that individual features miss. All interactions are clinically motivated.
    # ══════════════════════════════════════════════════════════════════════
    def _group_G_interaction_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        logger.info("-" * 50)
        logger.info("Group G: Interaction Features")

        # Prior inpatient × medication complexity
        # High prior visits + high med count = highest risk profile
        if "FE_has_prior_inpatient" in df.columns:
            df["FE_inpatient_x_polypharmacy"] = (
                df["FE_has_prior_inpatient"] * df["FE_polypharmacy"]
            )
            logger.info("  FE_inpatient_x_polypharmacy : prior_inpatient × polypharmacy")

        # High utilization × medication change
        if "FE_high_utilization" in df.columns and "FE_any_med_changed" in df.columns:
            df["FE_high_util_x_med_change"] = (
                df["FE_high_utilization"] * df["FE_any_med_changed"]
            )
            logger.info("  FE_high_util_x_med_change  : high_utilization × any_med_changed")

        # High comorbidity × high lab load
        if "FE_high_comorbidity" in df.columns and "FE_high_lab_load" in df.columns:
            df["FE_comorbidity_x_lab_load"] = (
                df["FE_high_comorbidity"] * df["FE_high_lab_load"]
            )
            logger.info("  FE_comorbidity_x_lab_load  : high_comorbidity × high_lab_load")

        # Glycemic risk × prior emergency
        if "FE_glycemic_risk_score" in df.columns and "FE_has_prior_emergency" in df.columns:
            df["FE_glycemic_x_emergency"] = (
                df["FE_glycemic_risk_score"] * df["FE_has_prior_emergency"]
            )
            logger.info("  FE_glycemic_x_emergency    : glycemic_risk_score × has_prior_emergency")

        # Labs per day × high comorbidity
        if "FE_labs_per_day" in df.columns and "FE_high_comorbidity" in df.columns:
            df["FE_labs_per_day_x_comorbidity"] = (
                df["FE_labs_per_day"] * df["FE_high_comorbidity"]
            ).round(4)
            logger.info("  FE_labs_per_day_x_comorbidity : labs_per_day × high_comorbidity")

        return df

    # ──────────────────────────────────────────────────────────────────────
    def _null_audit(self, df: pd.DataFrame, split: str):
        """Verify no FE_ column introduces unexpected nulls."""
        logger.info("-" * 50)
        logger.info(f"Null audit on FE_ columns [{split}]")

        fe_cols   = [c for c in df.columns if c.startswith("FE_")]
        null_fe   = df[fe_cols].isna().sum()
        null_fe   = null_fe[null_fe > 0]

        if null_fe.empty:
            logger.info(f"  All {len(fe_cols)} FE_ columns are null-free — OK")
        else:
            for col, cnt in null_fe.items():
                logger.warning(f"  {col}: {cnt} nulls ({cnt/len(df)*100:.2f}%)")

    # ──────────────────────────────────────────────────────────────────────
    def _leakage_guard(self, df: pd.DataFrame):
        """
        Verify no FE_ feature is constructed from target columns.
        Checks Pearson correlation between FE_ features and targets.
        Flags any feature with |r| > 0.80 as suspicious.
        """
        logger.info("-" * 50)
        logger.info("Leakage guard: FE_ features vs target correlation")

        fe_cols     = [c for c in df.columns if c.startswith("FE_")]
        target_cols = [c for c in TARGET_COLS if c in df.columns]

        if not target_cols:
            logger.info("  No target columns found in DataFrame — skip")
            return

        suspicious = []
        for fe_col in fe_cols:
            for tgt_col in target_cols:
                try:
                    r = df[fe_col].corr(df[tgt_col])
                    if abs(r) > 0.80:
                        suspicious.append((fe_col, tgt_col, round(r, 4)))
                        logger.warning(f"  SUSPICIOUS: {fe_col} vs {tgt_col} |r|={abs(r):.4f}")
                except Exception:
                    pass

        if not suspicious:
            logger.info(f"  Leakage guard passed — no |r| > 0.80 among {len(fe_cols)} FE_ features")
        else:
            logger.error(f"  {len(suspicious)} suspicious correlations detected — review before modeling")

    # ──────────────────────────────────────────────────────────────────────
    def _save_artifacts(self):
        """Save train statistics and feature names for reproducibility."""

        stats_path = ARTIFACTS_DIR / "fe_train_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.train_stats, f, indent=2)
        logger.info(f"  fe_train_stats.json saved -> {stats_path}")

        names_path = ARTIFACTS_DIR / "fe_feature_names.json"
        with open(names_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"  fe_feature_names.json saved -> {names_path}")

        obj_path = ARTIFACTS_DIR / "feature_engineer.pkl"
        with open(obj_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"  feature_engineer.pkl saved -> {obj_path}")


if __name__ == "__main__":
    from src.data.loader      import load_raw
    from src.data.validator   import validate
    from src.data.splitter    import split
    from src.data.preprocessor import Preprocessor

    df, ids_df, _       = load_raw()
    report              = validate(df)
    if not report["ready"]:
        raise RuntimeError("Validation failed")

    train, val, test, _ = split(df)

    prep         = Preprocessor()
    train_clean  = prep.fit_transform(train)
    val_clean    = prep.transform(val,  split_name="val")
    test_clean   = prep.transform(test, split_name="test")

    eng          = FeatureEngineer()
    train_fe     = eng.fit_transform(train_clean)
    val_fe       = eng.transform(val_clean,  split_name="val")
    test_fe      = eng.transform(test_clean, split_name="test")

    fe_cols = [c for c in train_fe.columns if c.startswith("FE_")]
    print(f"\nTrain  : {train_fe.shape}")
    print(f"Val    : {val_fe.shape}")
    print(f"Test   : {test_fe.shape}")
    print(f"FE_ features ({len(fe_cols)}): {fe_cols}")