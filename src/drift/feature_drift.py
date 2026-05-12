"""
DriftSentinel — Feature Drift Analyzer
Links data drift signals to model performance impact.
Answers: which drifted features actually hurt model predictions?

Analysis pipeline:
    1. Load drift scores from data_drift.py results
    2. Compute per-feature SHAP values on reference (val) and production (test)
    3. Measure SHAP distribution shift per feature
    4. Rank features by: drift_score × shap_importance (impact score)
    5. Identify features that are both drifted AND important
    6. Generate feature drift impact report
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger
from src.drift.data_drift  import DataDriftDetector, _psi, _ks, _js_divergence

logger = get_logger("feature_drift")

ROOT          = Path(__file__).resolve().parents[2]
TRAIN_DIR     = ROOT / "data"   / "train"
MODELS_DIR    = ROOT / "outputs" / "models"
REPORTS_DIR   = ROOT / "outputs" / "log"
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"

TARGET_COLS = {"readmitted_binary", "readmitted_multi"}


# ══════════════════════════════════════════════════════════════════════════
# SHAP shift detector
# ══════════════════════════════════════════════════════════════════════════

def _compute_shap_shift(
    model,
    ref_X:  np.ndarray,
    prod_X: np.ndarray,
    feat_cols: list[str],
) -> pd.DataFrame:
    """
    Compute mean |SHAP| on reference and production.
    SHAP shift = how much each feature's contribution changed.
    """
    import shap

    explainer = shap.TreeExplainer(model)

    ref_shap  = explainer.shap_values(ref_X)
    prod_shap = explainer.shap_values(prod_X)

    # Handle list output (binary classifier)
    if isinstance(ref_shap, list):
        ref_shap  = ref_shap[1]
        prod_shap = prod_shap[1]
    elif ref_shap.ndim == 3:
        ref_shap  = ref_shap[:, :, 1]
        prod_shap = prod_shap[:, :, 1]

    ref_mean_shap  = np.abs(ref_shap).mean(axis=0)
    prod_mean_shap = np.abs(prod_shap).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature"        : feat_cols,
        "shap_ref"       : ref_mean_shap,
        "shap_prod"      : prod_mean_shap,
        "shap_delta"     : prod_mean_shap - ref_mean_shap,
        "shap_delta_pct" : (prod_mean_shap - ref_mean_shap) /
                           (ref_mean_shap + 1e-8) * 100,
        "shap_ref_rank"  : pd.Series(ref_mean_shap).rank(ascending=False).values,
        "shap_prod_rank" : pd.Series(prod_mean_shap).rank(ascending=False).values,
    })

    shap_df["rank_shift"] = (
        shap_df["shap_prod_rank"] - shap_df["shap_ref_rank"]
    )

    return shap_df.sort_values("shap_ref", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# Feature drift impact scorer
# ══════════════════════════════════════════════════════════════════════════

def _compute_impact_score(
    drift_score:    float,
    shap_ref:       float,
    shap_delta_pct: float,
) -> float:
    """
    Impact score = drift severity × model importance × contribution change.

    Range [0, 1]. Higher = more dangerous for model reliability.
    """
    drift_component  = min(drift_score, 1.0)
    importance_norm  = min(shap_ref / 0.05, 1.0)
    change_component = min(abs(shap_delta_pct) / 50.0, 1.0)

    impact = (
        0.45 * drift_component  +
        0.35 * importance_norm  +
        0.20 * change_component
    )
    return round(float(impact), 4)


# ══════════════════════════════════════════════════════════════════════════
# Main analyzer
# ══════════════════════════════════════════════════════════════════════════

class FeatureDriftAnalyzer:
    """
    Links data drift to model performance impact via SHAP.
    Produces ranked feature impact report for alerting pipeline.
    """

    def __init__(self):
        self.report_: dict = {}

    def analyze(
        self,
        ref_df:    pd.DataFrame,
        prod_df:   pd.DataFrame,
        feat_cols: list[str],
        model,
        ref_name:  str = "val",
        prod_name: str = "test",
    ) -> pd.DataFrame:
        """
        Full feature drift impact analysis.

        Parameters
        ----------
        ref_df    : reference DataFrame (val split)
        prod_df   : production DataFrame (test split)
        feat_cols : feature columns list
        model     : fitted LightGBM model
        """
        logger.info("=" * 70)
        logger.info("DriftSentinel — Feature Drift Analyzer")
        logger.info(f"  Reference  : {ref_name}  ({len(ref_df):,} rows)")
        logger.info(f"  Production : {prod_name}  ({len(prod_df):,} rows)")
        logger.info("=" * 70)

        # ── Step 1: Data drift scores ──────────────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 1: Computing data drift scores")

        detector = DataDriftDetector(reference_name=ref_name)
        detector.fit(ref_df, feat_cols)
        drift_df = detector.detect(prod_df, production_name=prod_name)

        drift_lookup = drift_df.set_index("feature")[
            ["composite_score", "psi", "psi_level",
             "ks_stat", "js_divergence", "drift_detected",
             "mean_shift", "n_tests_fired"]
        ].to_dict("index")

        logger.info(f"  Drifted features: {drift_df['drift_detected'].sum()}/{len(drift_df)}")

        # ── Step 2: SHAP shift ─────────────────────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 2: Computing SHAP distribution shift")

        ref_X  = pd.DataFrame(ref_df[feat_cols].values,  columns=feat_cols)
        prod_X = pd.DataFrame(prod_df[feat_cols].values, columns=feat_cols)

        shap_df = _compute_shap_shift(
            model,
            ref_X.values,
            prod_X.values,
            feat_cols
        )

        shap_lookup = shap_df.set_index("feature").to_dict("index")

        logger.info(
            f"  Top SHAP shift (ref→prod):\n"
            + "\n".join([
                f"    {row['feature']:<40} "
                f"ref={row['shap_ref']:.4f}  "
                f"prod={row['shap_prod']:.4f}  "
                f"Δ={row['shap_delta']:+.4f}  "
                f"({row['shap_delta_pct']:+.1f}%)"
                for _, row in shap_df.head(10).iterrows()
            ])
        )

        # ── Step 3: Impact score ───────────────────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 3: Computing feature drift impact scores")

        records = []
        for feat in feat_cols:
            d = drift_lookup.get(feat, {})
            s = shap_lookup.get(feat, {})

            drift_score    = d.get("composite_score") or 0.0
            shap_ref       = s.get("shap_ref")        or 0.0
            shap_delta_pct = s.get("shap_delta_pct")  or 0.0

            impact = _compute_impact_score(drift_score, shap_ref, shap_delta_pct)

            # Risk tier
            if impact >= 0.30:
                risk = "HIGH"
            elif impact >= 0.15:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            records.append({
                "feature"         : feat,
                "is_fe"           : feat.startswith("FE_"),
                "impact_score"    : impact,
                "risk_tier"       : risk,
                "drift_score"     : d.get("composite_score"),
                "psi"             : d.get("psi"),
                "psi_level"       : d.get("psi_level"),
                "drift_detected"  : d.get("drift_detected"),
                "ks_stat"         : d.get("ks_stat"),
                "js_divergence"   : d.get("js_divergence"),
                "mean_shift"      : d.get("mean_shift"),
                "n_tests_fired"   : d.get("n_tests_fired"),
                "shap_ref"        : round(s.get("shap_ref")        or 0, 6),
                "shap_prod"       : round(s.get("shap_prod")       or 0, 6),
                "shap_delta"      : round(s.get("shap_delta")      or 0, 6),
                "shap_delta_pct"  : round(s.get("shap_delta_pct")  or 0, 2),
                "shap_ref_rank"   : s.get("shap_ref_rank"),
                "shap_prod_rank"  : s.get("shap_prod_rank"),
                "rank_shift"      : s.get("rank_shift"),
            })

        impact_df = pd.DataFrame(records).sort_values(
            "impact_score", ascending=False
        ).reset_index(drop=True)

        # ── Step 4: Summary report ─────────────────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 4: Feature drift impact report")
        logger.info(f"{'Feature':<45} {'Impact':>7} {'Risk':>7} {'Drift':>6} {'SHAP_ref':>9} {'SHAP_Δ%':>8}")
        logger.info("-" * 70)

        for _, row in impact_df.head(20).iterrows():
            logger.info(
                f"  {row['feature']:<43} "
                f"{row['impact_score']:>7.4f} "
                f"{row['risk_tier']:>7} "
                f"{'YES' if row['drift_detected'] else 'NO':>6} "
                f"{row['shap_ref']:>9.6f} "
                f"{row['shap_delta_pct']:>+8.1f}%"
            )

        high_risk  = impact_df[impact_df["risk_tier"] == "HIGH"]
        medium_risk = impact_df[impact_df["risk_tier"] == "MEDIUM"]
        low_risk   = impact_df[impact_df["risk_tier"] == "LOW"]

        logger.info("-" * 70)
        logger.info(f"  HIGH risk features   : {len(high_risk)}")
        logger.info(f"  MEDIUM risk features : {len(medium_risk)}")
        logger.info(f"  LOW risk features    : {len(low_risk)}")

        # ── Step 5: SHAP rank stability ────────────────────────────────────
        logger.info("-" * 50)
        logger.info("Step 5: SHAP rank stability analysis")

        rank_shifts = impact_df[["feature", "shap_ref_rank",
                                  "shap_prod_rank", "rank_shift"]].copy()
        rank_shifts = rank_shifts.sort_values(
            "rank_shift", key=abs, ascending=False
        )

        logger.info("  Top rank shifts (ref→prod):")
        for _, row in rank_shifts.head(10).iterrows():
            direction = "↑" if row["rank_shift"] < 0 else "↓"
            logger.info(
                f"    {row['feature']:<43} "
                f"ref_rank={int(row['shap_ref_rank']):>3}  "
                f"prod_rank={int(row['shap_prod_rank']):>3}  "
                f"shift={row['rank_shift']:>+4.0f} {direction}"
            )

        # ── Step 6: FE_ vs raw drift comparison ───────────────────────────
        logger.info("-" * 50)
        logger.info("Step 6: FE_ vs raw feature drift comparison")

        fe_df  = impact_df[impact_df["is_fe"]]
        raw_df = impact_df[~impact_df["is_fe"]]

        logger.info(f"  FE_ features  — mean impact={fe_df['impact_score'].mean():.4f}  "
                    f"high_risk={len(fe_df[fe_df['risk_tier']=='HIGH'])}")
        logger.info(f"  Raw features  — mean impact={raw_df['impact_score'].mean():.4f}  "
                    f"high_risk={len(raw_df[raw_df['risk_tier']=='HIGH'])}")

        # ── Save ───────────────────────────────────────────────────────────
        csv_path = REPORTS_DIR / f"feature_drift_{ref_name}_{prod_name}.csv"
        impact_df.to_csv(csv_path, index=False)
        logger.info(f"  Feature drift CSV: {csv_path}")

        self.report_ = {
            "ref_name"       : ref_name,
            "prod_name"      : prod_name,
            "n_features"     : len(feat_cols),
            "n_high_risk"    : int(len(high_risk)),
            "n_medium_risk"  : int(len(medium_risk)),
            "n_low_risk"     : int(len(low_risk)),
            "top_high_risk"  : high_risk[
                ["feature", "impact_score", "risk_tier",
                 "drift_score", "shap_ref", "shap_delta_pct"]
            ].head(10).to_dict("records"),
            "fe_mean_impact" : round(float(fe_df["impact_score"].mean()),  4),
            "raw_mean_impact": round(float(raw_df["impact_score"].mean()), 4),
        }

        report_path = REPORTS_DIR / f"feature_drift_report_{ref_name}_{prod_name}.json"
        with open(report_path, "w") as f:
            json.dump(self.report_, f, indent=2, default=str)
        logger.info(f"  Feature drift report: {report_path}")

        logger.info("=" * 70)
        logger.info("Feature Drift Analysis Complete")
        logger.info(f"  HIGH risk  : {len(high_risk)}")
        logger.info(f"  MEDIUM risk: {len(medium_risk)}")
        logger.info(f"  LOW risk   : {len(low_risk)}")
        logger.info("  Next: concept_drift.py → alerting.py")
        logger.info("=" * 70)

        return impact_df


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_feature_drift() -> dict:
    logger.info("=" * 70)
    logger.info("DriftSentinel — Feature Drift Detection Run")
    logger.info("=" * 70)

    # Load splits
    val  = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")

    feat_cols = [c for c in val.columns if c not in TARGET_COLS]

    # Load LGBM model
    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model = pickle.load(f)

    # Analyze
    analyzer   = FeatureDriftAnalyzer()
    impact_df  = analyzer.analyze(
        ref_df    = val,
        prod_df   = test,
        feat_cols = feat_cols,
        model     = model,
        ref_name  = "val",
        prod_name = "test",
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("FEATURE DRIFT IMPACT SUMMARY")
    print("=" * 60)
    print(f"{'Feature':<45} {'Impact':>7} {'Risk':>7}")
    print("-" * 60)
    for _, row in impact_df.head(15).iterrows():
        print(
            f"{row['feature']:<45} "
            f"{row['impact_score']:>7.4f} "
            f"{row['risk_tier']:>7}"
        )

    return analyzer.report_


if __name__ == "__main__":
    report = run_feature_drift()
    print(f"\nHIGH risk features : {report['n_high_risk']}")
    print(f"MEDIUM risk        : {report['n_medium_risk']}")
    print(f"LOW risk           : {report['n_low_risk']}")