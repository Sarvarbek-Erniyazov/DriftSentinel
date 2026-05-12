"""
DriftSentinel — Data Drift Detector
Detects distributional shift between reference (train/val) and
production (test) windows using multiple statistical tests.

Methods implemented:
    PSI  — Population Stability Index (industry standard)
    KS   — Kolmogorov-Smirnov test (continuous features)
    Chi2 — Chi-squared test (categorical/binary features)
    JS   — Jensen-Shannon divergence (information-theoretic)
    MW   — Mann-Whitney U test (non-parametric location shift)

Each method returns a per-feature score + significance flag.
Final drift score = weighted ensemble of all applicable tests.
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from scipy.stats import (
    ks_2samp, chi2_contingency, mannwhitneyu
)
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
import sys

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("data_drift")

ROOT          = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"
REPORTS_DIR   = ROOT / "outputs" / "log"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Thresholds ─────────────────────────────────────────────────────────────
PSI_STABLE    = 0.10
PSI_MODERATE  = 0.20
KS_ALPHA      = 0.01
CHI2_ALPHA    = 0.01
JS_MODERATE   = 0.05
JS_CRITICAL   = 0.10
MW_ALPHA      = 0.01
PSI_N_BINS    = 10

# ── Feature type classification ────────────────────────────────────────────
BINARY_FEATURES = {
    "weight_missing", "max_glu_serum_missing", "A1Cresult_missing",
    "FE_high_utilization", "FE_any_med_changed", "FE_high_lab_load",
    "FE_no_procedures", "FE_has_prior_inpatient", "FE_high_prior_utilization",
    "FE_insulin_prescribed", "FE_insulin_changed", "FE_insulin_increased",
    "FE_metformin_prescribed", "FE_metformin_changed", "FE_on_diabetes_med",
    "FE_med_regimen_changed", "FE_high_util_x_med_change",
    "FE_comorbidity_x_lab_load",
}

ORDINAL_FEATURES = {
    "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "race", "gender", "payer_code",
    "medical_specialty", "diag_1", "diag_2", "diag_3",
    "repaglinide", "glimepiride", "glipizide", "glyburide",
    "pioglitazone", "rosiglitazone",
}


# ══════════════════════════════════════════════════════════════════════════
# Statistical test implementations
# ══════════════════════════════════════════════════════════════════════════

def _psi(ref: np.ndarray, prod: np.ndarray, n_bins: int = PSI_N_BINS) -> float:
    ref  = ref[~np.isnan(ref)]
    prod = prod[~np.isnan(prod)]
    if len(ref) == 0 or len(prod) == 0:
        return np.nan
    bp = np.unique(np.percentile(ref, np.linspace(0, 100, n_bins + 1)))
    if len(bp) < 2:
        return np.nan
    r_pct = np.histogram(ref,  bins=bp)[0] / len(ref)  + 1e-8
    p_pct = np.histogram(prod, bins=bp)[0] / len(prod) + 1e-8
    return float(np.sum((p_pct - r_pct) * np.log(p_pct / r_pct)))


def _ks(ref: np.ndarray, prod: np.ndarray) -> tuple[float, float]:
    ref  = ref[~np.isnan(ref)]
    prod = prod[~np.isnan(prod)]
    if len(ref) < 5 or len(prod) < 5:
        return np.nan, np.nan
    stat, pval = ks_2samp(ref, prod)
    return float(stat), float(pval)


def _chi2(ref: np.ndarray, prod: np.ndarray) -> tuple[float, float]:
    ref  = ref[~np.isnan(ref)]
    prod = prod[~np.isnan(prod)]
    all_vals = np.unique(np.concatenate([ref, prod]))
    if len(all_vals) < 2:
        return np.nan, np.nan
    ref_counts  = np.array([np.sum(ref  == v) for v in all_vals]) + 1e-8
    prod_counts = np.array([np.sum(prod == v) for v in all_vals]) + 1e-8
    contingency = np.array([ref_counts, prod_counts])
    try:
        chi2, pval, _, _ = chi2_contingency(contingency)
        return float(chi2), float(pval)
    except Exception:
        return np.nan, np.nan


def _js_divergence(ref: np.ndarray, prod: np.ndarray,
                   n_bins: int = PSI_N_BINS) -> float:
    ref  = ref[~np.isnan(ref)]
    prod = prod[~np.isnan(prod)]
    if len(ref) < 5 or len(prod) < 5:
        return np.nan
    combined = np.concatenate([ref, prod])
    bp = np.unique(np.percentile(combined, np.linspace(0, 100, n_bins + 1)))
    if len(bp) < 2:
        return np.nan
    r_pct = np.histogram(ref,  bins=bp)[0] / len(ref)  + 1e-8
    p_pct = np.histogram(prod, bins=bp)[0] / len(prod) + 1e-8
    r_pct /= r_pct.sum()
    p_pct /= p_pct.sum()
    return float(jensenshannon(r_pct, p_pct))


def _mann_whitney(ref: np.ndarray, prod: np.ndarray) -> tuple[float, float]:
    ref  = ref[~np.isnan(ref)]
    prod = prod[~np.isnan(prod)]
    if len(ref) < 5 or len(prod) < 5:
        return np.nan, np.nan
    try:
        stat, pval = mannwhitneyu(ref, prod, alternative="two-sided")
        return float(stat), float(pval)
    except Exception:
        return np.nan, np.nan


# ══════════════════════════════════════════════════════════════════════════
# Drift level classification
# ══════════════════════════════════════════════════════════════════════════

def _psi_level(psi_val: float) -> str:
    if np.isnan(psi_val):
        return "UNKNOWN"
    if psi_val < PSI_STABLE:
        return "STABLE"
    if psi_val < PSI_MODERATE:
        return "MODERATE"
    return "CRITICAL"


def _composite_drift_score(
    psi: float,
    ks_stat: float,
    js: float,
    ks_pval: float,
    mw_pval: float,
) -> float:
    """
    Weighted composite drift score [0, 1].
    Combines PSI, KS statistic, and JS divergence.
    Higher score = more drift.
    """
    scores = []

    if not np.isnan(psi):
        psi_norm = min(psi / 0.5, 1.0)
        scores.append(("psi", psi_norm, 0.40))

    if not np.isnan(ks_stat):
        scores.append(("ks", min(ks_stat, 1.0), 0.35))

    if not np.isnan(js):
        js_norm = min(js / 0.20, 1.0)
        scores.append(("js", js_norm, 0.25))

    if not scores:
        return np.nan

    total_weight = sum(w for _, _, w in scores)
    composite    = sum(s * w for _, s, w in scores) / total_weight
    return round(float(composite), 4)


# ══════════════════════════════════════════════════════════════════════════
# Main detector
# ══════════════════════════════════════════════════════════════════════════

class DataDriftDetector:
    """
    Multi-method data drift detector.
    Compares reference window vs production window per feature.
    """

    def __init__(self, reference_name: str = "train"):
        self.reference_name = reference_name
        self.results_: dict  = {}
        self.summary_: dict  = {}
        self.fitted_:  bool  = False

    # ──────────────────────────────────────────────────────────────────────
    def fit(self, reference: pd.DataFrame, feat_cols: list[str]) -> "DataDriftDetector":
        """
        Store reference distribution statistics.

        Parameters
        ----------
        reference  : reference DataFrame (train or val split)
        feat_cols  : feature columns to monitor
        """
        logger.info("=" * 70)
        logger.info("DriftSentinel — Data Drift Detector  [fit reference]")
        logger.info("=" * 70)
        logger.info(f"Reference window : {self.reference_name}  shape={reference.shape}")
        logger.info(f"Features to monitor: {len(feat_cols)}")

        self._reference    = reference[feat_cols].copy()
        self._feat_cols    = feat_cols
        self.fitted_       = True

        # Reference statistics
        self._ref_stats = {}
        for col in feat_cols:
            vals = reference[col].dropna().values
            self._ref_stats[col] = {
                "mean"   : float(np.mean(vals)),
                "std"    : float(np.std(vals)),
                "median" : float(np.median(vals)),
                "q25"    : float(np.percentile(vals, 25)),
                "q75"    : float(np.percentile(vals, 75)),
                "n"      : int(len(vals)),
            }

        logger.info(f"Reference statistics computed for {len(feat_cols)} features")
        return self

    # ──────────────────────────────────────────────────────────────────────
    def detect(
        self,
        production: pd.DataFrame,
        production_name: str = "production"
    ) -> pd.DataFrame:
        """
        Run all drift tests between reference and production window.

        Parameters
        ----------
        production      : production DataFrame
        production_name : label for logging

        Returns
        -------
        drift_df : DataFrame with per-feature drift scores and flags
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before detect()")

        logger.info("=" * 70)
        logger.info(f"DriftSentinel — Data Drift Detection")
        logger.info(f"  Reference  : {self.reference_name}  ({len(self._reference):,} rows)")
        logger.info(f"  Production : {production_name}  ({len(production):,} rows)")
        logger.info("=" * 70)

        records = []

        for col in self._feat_cols:
            if col not in production.columns:
                logger.warning(f"  Column {col} missing from production — skip")
                continue

            ref_vals  = self._reference[col].dropna().values
            prod_vals = production[col].dropna().values
            is_binary = col in BINARY_FEATURES
            is_ord    = col in ORDINAL_FEATURES

            # ── PSI ───────────────────────────────────────────────────────
            psi_val   = _psi(ref_vals, prod_vals)
            psi_lvl   = _psi_level(psi_val)

            # ── KS (continuous + ordinal) ─────────────────────────────────
            if not is_binary:
                ks_stat, ks_pval = _ks(ref_vals, prod_vals)
            else:
                ks_stat, ks_pval = np.nan, np.nan

            # ── Chi2 (categorical / binary) ────────────────────────────────
            if is_binary or is_ord:
                chi2_stat, chi2_pval = _chi2(ref_vals, prod_vals)
            else:
                chi2_stat, chi2_pval = np.nan, np.nan

            # ── JS divergence ─────────────────────────────────────────────
            js_val = _js_divergence(ref_vals, prod_vals)

            # ── Mann-Whitney ──────────────────────────────────────────────
            if not is_binary:
                mw_stat, mw_pval = _mann_whitney(ref_vals, prod_vals)
            else:
                mw_stat, mw_pval = np.nan, np.nan

            # ── Composite score ───────────────────────────────────────────
            composite = _composite_drift_score(
                psi_val, ks_stat, js_val, ks_pval, mw_pval
            )

            # ── Significance flags ────────────────────────────────────────
            ks_drift   = (not np.isnan(ks_pval))   and (ks_pval   < KS_ALPHA)
            chi2_drift = (not np.isnan(chi2_pval)) and (chi2_pval < CHI2_ALPHA)
            mw_drift   = (not np.isnan(mw_pval))   and (mw_pval   < MW_ALPHA)
            psi_drift  = psi_lvl in ("MODERATE", "CRITICAL")
            js_drift   = (not np.isnan(js_val))    and (js_val    > JS_MODERATE)

            n_tests_fired = sum([ks_drift, chi2_drift, mw_drift, psi_drift, js_drift])
            drift_detected = n_tests_fired >= 2

            # ── Mean shift ────────────────────────────────────────────────
            ref_mean  = float(np.mean(ref_vals))  if len(ref_vals)  > 0 else np.nan
            prod_mean = float(np.mean(prod_vals)) if len(prod_vals) > 0 else np.nan
            mean_shift = round(prod_mean - ref_mean, 4) if not np.isnan(ref_mean) else np.nan

            records.append({
                "feature"        : col,
                "is_fe"          : col.startswith("FE_"),
                "is_binary"      : is_binary,
                "psi"            : round(psi_val,   4) if not np.isnan(psi_val)   else None,
                "psi_level"      : psi_lvl,
                "ks_stat"        : round(ks_stat,   4) if not np.isnan(ks_stat)   else None,
                "ks_pval"        : round(ks_pval,   6) if not np.isnan(ks_pval)   else None,
                "ks_drift"       : ks_drift,
                "chi2_stat"      : round(chi2_stat, 4) if not np.isnan(chi2_stat) else None,
                "chi2_pval"      : round(chi2_pval, 6) if not np.isnan(chi2_pval) else None,
                "chi2_drift"     : chi2_drift,
                "js_divergence"  : round(js_val,    4) if not np.isnan(js_val)    else None,
                "js_drift"       : js_drift,
                "mw_stat"        : round(mw_stat,   4) if not np.isnan(mw_stat)   else None,
                "mw_pval"        : round(mw_pval,   6) if not np.isnan(mw_pval)   else None,
                "mw_drift"       : mw_drift,
                "composite_score": composite,
                "n_tests_fired"  : n_tests_fired,
                "drift_detected" : drift_detected,
                "ref_mean"       : round(ref_mean,  4) if not np.isnan(ref_mean)  else None,
                "prod_mean"      : round(prod_mean, 4) if not np.isnan(prod_mean) else None,
                "mean_shift"     : mean_shift,
            })

        drift_df = pd.DataFrame(records).sort_values(
            "composite_score", ascending=False
        ).reset_index(drop=True)

        # ── Summary ────────────────────────────────────────────────────────
        n_total    = len(drift_df)
        n_drifted  = drift_df["drift_detected"].sum()
        n_critical = (drift_df["psi_level"] == "CRITICAL").sum()
        n_moderate = (drift_df["psi_level"] == "MODERATE").sum()
        n_stable   = (drift_df["psi_level"] == "STABLE").sum()

        logger.info("-" * 70)
        logger.info(f"Drift Detection Results: {self.reference_name} → {production_name}")
        logger.info("-" * 70)
        logger.info(f"  Total features     : {n_total}")
        logger.info(f"  Drift detected     : {n_drifted} ({n_drifted/n_total*100:.1f}%)")
        logger.info(f"  PSI Critical       : {n_critical}")
        logger.info(f"  PSI Moderate       : {n_moderate}")
        logger.info(f"  PSI Stable         : {n_stable}")
        logger.info("-" * 70)
        logger.info(f"{'Feature':<45} {'PSI':>7} {'Level':>10} {'KS':>6} {'JS':>6} {'Score':>7} {'Drift':>6}")
        logger.info("-" * 70)

        for _, row in drift_df.head(20).iterrows():
            flag = "🔴" if row["drift_detected"] else "🟢"
            logger.info(
                f"  {row['feature']:<43} "
                f"{str(row['psi'] or 'N/A'):>7} "
                f"{row['psi_level']:>10} "
                f"{str(row['ks_stat'] or 'N/A'):>6} "
                f"{str(row['js_divergence'] or 'N/A'):>6} "
                f"{str(row['composite_score'] or 'N/A'):>7} "
                f"{'YES' if row['drift_detected'] else 'NO':>6}"
            )

        self.results_[production_name] = drift_df
        self.summary_[production_name] = {
            "reference"      : self.reference_name,
            "production"     : production_name,
            "n_features"     : n_total,
            "n_drifted"      : int(n_drifted),
            "n_critical_psi" : int(n_critical),
            "n_moderate_psi" : int(n_moderate),
            "n_stable_psi"   : int(n_stable),
            "top_drifted"    : drift_df[drift_df["drift_detected"]][
                ["feature", "composite_score", "psi_level"]
            ].head(10).to_dict("records"),
        }

        return drift_df

    # ──────────────────────────────────────────────────────────────────────
    def save_report(self, path: Path = None) -> Path:
        if path is None:
            path = REPORTS_DIR / "data_drift_report.json"

        serializable = {}
        for prod_name, summary in self.summary_.items():
            serializable[prod_name] = summary

        # Add full results as CSV
        for prod_name, drift_df in self.results_.items():
            csv_path = REPORTS_DIR / f"data_drift_{prod_name}.csv"
            drift_df.to_csv(csv_path, index=False)
            logger.info(f"  Drift results CSV: {csv_path}")

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"  Data drift report saved: {path}")
        return path


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_data_drift() -> dict:
    logger.info("=" * 70)
    logger.info("DriftSentinel — Data Drift Detection Run")
    logger.info("=" * 70)

    # Load splits
    train = pd.read_parquet(ROOT / "data" / "train" / "train_fs.parquet")
    val   = pd.read_parquet(ROOT / "data" / "train" / "val_fs.parquet")
    test  = pd.read_parquet(ROOT / "data" / "train" / "test_fs.parquet")

    target_cols = {"readmitted_binary", "readmitted_multi"}
    feat_cols   = [c for c in train.columns if c not in target_cols]

    # Detector: reference = val (deployed model window)
    detector = DataDriftDetector(reference_name="val")
    detector.fit(val, feat_cols)

    # Detect: val → test (simulates production drift)
    logger.info("\n>>> Drift window: val (reference) → test (production)")
    drift_val_test = detector.detect(test, production_name="test")

    # Detect: train → val (warm-up check)
    detector_train = DataDriftDetector(reference_name="train")
    detector_train.fit(train, feat_cols)

    logger.info("\n>>> Drift window: train (reference) → val (production)")
    drift_train_val = detector_train.detect(val, production_name="val")

    logger.info("\n>>> Drift window: train (reference) → test (production)")
    drift_train_test = detector_train.detect(test, production_name="test")

    # Save reports
    detector.save_report()
    detector_train.save_report(REPORTS_DIR / "data_drift_train_reference.json")

    # Final summary
    logger.info("=" * 70)
    logger.info("Data Drift Detection Complete")
    logger.info("=" * 70)
    for name, df in [("val→test", drift_val_test),
                     ("train→val", drift_train_val),
                     ("train→test", drift_train_test)]:
        n_d = df["drift_detected"].sum()
        n_c = (df["psi_level"] == "CRITICAL").sum()
        logger.info(f"  {name:<15}: drifted={n_d}/{len(df)}  critical={n_c}")

    logger.info("  Next: feature_drift.py → concept_drift.py → alerting.py")
    logger.info("=" * 70)

    return {
        "val_test"   : detector.summary_,
        "train_ref"  : detector_train.summary_,
    }


if __name__ == "__main__":
    results = run_data_drift()
    print("\nDrift Detection Summary:")
    for window, summary in results["val_test"].items():
        print(f"  val→{window}: drifted={summary['n_drifted']}/{summary['n_features']}")
    for window, summary in results["train_ref"].items():
        print(f"  train→{window}: drifted={summary['n_drifted']}/{summary['n_features']}")