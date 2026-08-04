"""
DriftSentinel — Drift Alert System
Consolidates all drift signals into actionable alerts.
Reads outputs from data_drift, feature_drift, concept_drift modules.
Produces structured alert report with severity, root cause, and recommendations.

Alert levels:
    CRITICAL — immediate action required, model unreliable
    HIGH     — significant drift, schedule retraining
    MEDIUM   — drift detected, monitor closely
    LOW      — minor drift, within acceptable range
    STABLE   — no drift detected

Alert types:
    DATA_DRIFT     — feature distribution shift detected
    CONCEPT_DRIFT  — model performance degradation detected
    FEATURE_IMPACT — high-impact feature drifted
    LABEL_SHIFT    — target distribution changed
    PREDICTION_SHIFT — model output distribution changed
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("alerting")

ROOT        = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "outputs" / "log"
MODELS_DIR  = ROOT / "outputs" / "models"
ALERTS_DIR  = ROOT / "outputs" / "alerts"
ALERTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Alert thresholds ───────────────────────────────────────────────────────
THRESHOLDS = {
    "auc_drop_critical"    : 0.05,
    "auc_drop_high"        : 0.03,
    "auc_drop_medium"      : 0.01,
    "f1_drop_critical"     : 0.10,
    "f1_drop_high"         : 0.05,
    "psi_critical"         : 0.20,
    "psi_high"             : 0.10,
    "n_drifted_critical"   : 0.40,
    "n_drifted_high"       : 0.25,
    "label_shift_critical" : 0.10,
    "label_shift_high"     : 0.05,
    "proba_shift_critical" : 0.05,
    "impact_high"          : 0.35,
    "impact_medium"        : 0.20,
    "evidence_critical"    : 6,
    "evidence_high"        : 4,
    "evidence_medium"      : 2,
}


def set_alerts_dir(path) -> None:
    """
    Redirect this module's alert output (Tier 1.7).

    Explicit, supported API. The Tier 0 regime sweep previously reassigned the
    module global directly — a monkey-patch that worked but was invisible to
    anyone reading this module. A sweep of ~200 detector runs must not scatter
    throwaway artifacts through outputs/log/, so redirection is legitimate; doing
    it by attribute assignment from another file was not.
    """
    global ALERTS_DIR
    from pathlib import Path as _P
    ALERTS_DIR = _P(path)
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# Alert builder
# ══════════════════════════════════════════════════════════════════════════

class Alert:
    def __init__(
        self,
        alert_id:    str,
        alert_type:  str,
        level:       str,
        title:       str,
        description: str,
        metric:      str,
        value:       float,
        threshold:   float,
        features:    list[str] = None,
        action:      str       = None,
    ):
        self.alert_id    = alert_id
        self.alert_type  = alert_type
        self.level       = level
        self.title       = title
        self.description = description
        self.metric      = metric
        self.value       = value
        self.threshold   = threshold
        self.features    = features or []
        self.action      = action or ""
        self.timestamp   = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "alert_id"   : self.alert_id,
            "alert_type" : self.alert_type,
            "level"      : self.level,
            "title"      : self.title,
            "description": self.description,
            "metric"     : self.metric,
            "value"      : round(float(self.value), 4),
            "threshold"  : round(float(self.threshold), 4),
            "features"   : self.features,
            "action"     : self.action,
            "timestamp"  : self.timestamp,
        }

    def __repr__(self):
        return (f"[{self.level}] {self.alert_type} | "
                f"{self.title} | {self.metric}={self.value:.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Alert generators per drift type
# ══════════════════════════════════════════════════════════════════════════

def _seq_level(concept_report: dict) -> str:
    """
    Alert level for the sequential detectors (CUSUM / Page-Hinkley).

    Returns LOW when the detector has been retired from the evidence count, so
    it stays visible without influencing `system_status` (which counts only
    CRITICAL and HIGH). Falls back to HIGH for reports produced before Tier 1.5
    so old artifacts still render with their original semantics.
    """
    retired = set(concept_report.get("retired_signals", {}).get("signals", []))
    return "LOW" if {"cusum_alarm", "ph_alarm"} & retired else "HIGH"


def _concept_drift_alerts(concept_report: dict) -> list[Alert]:
    alerts = []
    auc_deg  = concept_report.get("auc_degradation", 0)
    f1_deg   = concept_report.get("f1_degradation", 0)
    n_ev     = concept_report.get("n_evidence", 0)
    severity = concept_report.get("severity", "NONE")

    # AUC degradation alert
    if auc_deg >= THRESHOLDS["auc_drop_critical"]:
        level = "CRITICAL"
    elif auc_deg >= THRESHOLDS["auc_drop_high"]:
        level = "HIGH"
    elif auc_deg >= THRESHOLDS["auc_drop_medium"]:
        level = "MEDIUM"
    else:
        level = None

    if level:
        alerts.append(Alert(
            alert_id    = "CD-001",
            alert_type  = "CONCEPT_DRIFT",
            level       = level,
            title       = "Model AUC Degradation Detected",
            description = (
                f"Model AUC dropped from "
                f"{concept_report['ref_metrics']['auc']:.4f} (reference) to "
                f"{concept_report['prod_metrics']['auc']:.4f} (production). "
                f"Degradation: {auc_deg:.4f}."
            ),
            metric      = "auc_degradation",
            value       = auc_deg,
            threshold   = THRESHOLDS["auc_drop_high"],
            action      = (
                "Trigger model retraining on recent data. "
                "Investigate feature drift in high-impact features."
            )
        ))

    # F1 degradation alert
    if f1_deg >= THRESHOLDS["f1_drop_critical"]:
        f1_level = "CRITICAL"
    elif f1_deg >= THRESHOLDS["f1_drop_high"]:
        f1_level = "HIGH"
    else:
        f1_level = None

    if f1_level:
        ref_m  = concept_report["ref_metrics"]
        prod_m = concept_report["prod_metrics"]
        alerts.append(Alert(
            alert_id    = "CD-002",
            alert_type  = "CONCEPT_DRIFT",
            level       = f1_level,
            title       = "Model F1 Score Degradation",
            description = (
                f"F1 dropped {f1_deg:.4f} points. "
                f"Precision: {ref_m['precision']:.4f} → "
                f"{prod_m['precision']:.4f} "
                f"({prod_m['precision'] - ref_m['precision']:+.4f}). "
                f"Recall: {ref_m['recall']:.4f} → "
                f"{prod_m['recall']:.4f}."
            ),
            metric      = "f1_degradation",
            value       = f1_deg,
            threshold   = THRESHOLDS["f1_drop_high"],
            action      = (
                "Review precision-recall tradeoff. "
                "Consider threshold recalibration before retraining."
            )
        ))

    # Evidence count alert.
    # Tier 1.5: thresholds are FRACTIONS of the voting-signal count, which is
    # read from the report rather than hardcoded to 8. Two signals were retired
    # from voting, and leaving absolute thresholds behind would have silently
    # made every severity level harder to reach.
    n_voting = int(concept_report.get("n_voting_signals", 8))
    _frac = {"critical": THRESHOLDS["evidence_critical"] / 8,
             "high":     THRESHOLDS["evidence_high"] / 8,
             "medium":   THRESHOLDS["evidence_medium"] / 8}
    ev_crit = int(np.ceil(_frac["critical"] * n_voting))
    ev_high = int(np.ceil(_frac["high"] * n_voting))
    ev_med  = int(np.ceil(_frac["medium"] * n_voting))

    if n_ev >= ev_crit:
        ev_level = "CRITICAL"
    elif n_ev >= ev_high:
        ev_level = "HIGH"
    elif n_ev >= ev_med:
        ev_level = "MEDIUM"
    else:
        ev_level = None

    if ev_level:
        evidence = concept_report.get("evidence", {})
        fired    = [k for k, v in evidence.items() if v]
        # R5: these signals are NOT independent. auc_drop, f1_drop,
        # brier_increase and auc_slope_negative are four views of one
        # degradation; prediction_drift and label_drift are mechanically coupled.
        # The word "independent" was the most quotable overclaim in the README.
        families = {"performance": {"auc_drop", "f1_drop", "brier_increase",
                                    "auc_slope_negative"},
                    "distribution": {"prediction_drift", "label_drift"}}
        fam_hit = sorted({f for f, sigs in families.items() if sigs & set(fired)})
        alerts.append(Alert(
            alert_id    = "CD-003",
            alert_type  = "CONCEPT_DRIFT",
            level       = ev_level,
            title       = f"Concept Drift Evidence: {n_ev}/{n_voting} Voting Signals Fired",
            description = (
                f"{n_ev} of {n_voting} voting signals fired, spanning "
                f"{len(fam_hit)} correlated signal families ({', '.join(fam_hit) or 'none'}): "
                f"{', '.join(fired)}. These signals are NOT independent. "
                f"Sequential detectors (CUSUM, Page-Hinkley) are excluded from "
                f"the count as structurally broken — see Tier 1.5."
            ),
            metric      = "n_evidence",
            value       = float(n_ev),
            threshold   = float(ev_high),
            action      = (
                "Initiate drift investigation protocol. "
                "Run feature_drift analysis to identify root cause features."
            )
        ))

    # CUSUM alert
    if concept_report["cusum"]["drift_detected"]:
        alerts.append(Alert(
            alert_id    = "CD-004",
            alert_type  = "CONCEPT_DRIFT",
            # Tier 1.5 (completion): demoted from HIGH to LOW. Retiring these
            # signals from the EVIDENCE COUNT was not enough — this alert reads
            # `cusum.drift_detected` directly, so a saturated detector that fires
            # in 100% of runs in every regime was still producing a HIGH alert,
            # and one HIGH alert alone sets system_status to MODERATE. Measured
            # on the no-drift control: 38 HIGH alerts across 20 seeds, and
            # MODERATE status on all 20. A retirement that leaves the signal
            # driving the alert system is cosmetic.
            level       = _seq_level(concept_report),
            title       = "CUSUM Sequential Alarm [DIAGNOSTIC — not evidence]",
            description = (
                f"CUSUM detected {concept_report['cusum']['n_alarms']} alarms. "
                f"First alarm at index {concept_report['cusum']['first_alarm_idx']} "
                f"({concept_report['cusum']['first_alarm_pct']}% into stream). "
                f"Mean error shift: "
                f"{concept_report['cusum']['mean_ref']:.4f} → "
                f"{concept_report['cusum']['mean_prod']:.4f}."
            ),
            metric    = "cusum_alarms",
            value     = float(concept_report["cusum"]["n_alarms"]),
            threshold = 1.0,
            action    = (
                "Deploy monitoring on incoming predictions. "
                "CUSUM suggests sustained error increase from early in stream."
            )
        ))

    # Page-Hinkley alert
    if concept_report["page_hinkley"]["drift_detected"]:
        alerts.append(Alert(
            alert_id    = "CD-005",
            alert_type  = "CONCEPT_DRIFT",
            level       = _seq_level(concept_report),   # Tier 1.5: see CD-004
            title       = "Page-Hinkley Sequential Alarm [DIAGNOSTIC — not evidence]",
            description = (
                f"Page-Hinkley detected gradual drift. "
                f"First alarm at {concept_report['page_hinkley']['first_alarm_pct']}% "
                f"of stream. PH statistic: "
                f"{concept_report['page_hinkley']['ph_final']:.4f}."
            ),
            metric    = "ph_statistic",
            value     = float(concept_report["page_hinkley"]["ph_final"]),
            threshold = 50.0,
            action    = (
                "Gradual concept drift confirmed. "
                "Schedule periodic model retraining every N production batches."
            )
        ))

    return alerts


def _data_drift_alerts(drift_summary: dict) -> list[Alert]:
    alerts      = []
    n_features  = drift_summary.get("n_features",     0)
    n_drifted   = drift_summary.get("n_drifted",       0)
    n_critical  = drift_summary.get("n_critical_psi",  0)
    drift_rate  = n_drifted / n_features if n_features > 0 else 0
    top_drifted = drift_summary.get("top_drifted",     [])

    # Overall drift rate alert
    if drift_rate >= THRESHOLDS["n_drifted_critical"]:
        rate_level = "CRITICAL"
    elif drift_rate >= THRESHOLDS["n_drifted_high"]:
        rate_level = "HIGH"
    else:
        rate_level = "MEDIUM"

    drifted_names = [d["feature"] for d in top_drifted[:5]]
    alerts.append(Alert(
        alert_id    = "DD-001",
        alert_type  = "DATA_DRIFT",
        level       = rate_level,
        title       = f"Data Drift: {n_drifted}/{n_features} Features Drifted",
        description = (
            f"{drift_rate*100:.1f}% of monitored features show distribution shift. "
            f"{n_critical} features with PSI > 0.20 (critical). "
            f"Top drifted: {', '.join(drifted_names)}."
        ),
        metric      = "drift_rate",
        value       = drift_rate,
        threshold   = THRESHOLDS["n_drifted_high"],
        features    = drifted_names,
        action      = (
            "Investigate data pipeline for source distribution changes. "
            "Check for data quality issues in production ingestion."
        )
    ))

    # Critical PSI features
    if n_critical > 0:
        critical_feats = [
            d["feature"] for d in top_drifted
            if d.get("psi_level") == "CRITICAL"
        ]
        alerts.append(Alert(
            alert_id    = "DD-002",
            alert_type  = "DATA_DRIFT",
            level       = "CRITICAL",
            title       = f"{n_critical} Features with Critical PSI (>0.20)",
            description = (
                f"Features with PSI > 0.20: "
                f"{', '.join(critical_feats[:5])}. "
                f"These features require immediate investigation as "
                f"their distributions have significantly shifted."
            ),
            metric      = "n_critical_psi",
            value       = float(n_critical),
            threshold   = 1.0,
            features    = critical_feats[:10],
            action      = (
                "Retrain preprocessing encoders on recent data. "
                "Verify data collection pipeline for these features."
            )
        ))

    return alerts


def _label_shift_alert(concept_report: dict) -> list[Alert]:
    alerts     = []
    label_info = concept_report.get("label_shift", {})
    delta      = abs(label_info.get("delta_pos_rate", 0))

    if delta >= THRESHOLDS["label_shift_critical"]:
        level = "CRITICAL"
    elif delta >= THRESHOLDS["label_shift_high"]:
        level = "HIGH"
    else:
        return alerts

    alerts.append(Alert(
        alert_id    = "LS-001",
        alert_type  = "LABEL_SHIFT",
        level       = level,
        title       = "Target Label Distribution Shift Detected",
        description = (
            f"Readmission rate shifted from "
            f"{label_info['ref_pos_rate']:.4f} (reference) to "
            f"{label_info['prod_pos_rate']:.4f} (production). "
            f"Delta: {label_info['delta_pos_rate']:+.4f} "
            f"({delta*100:.1f}pp change). "
            f"This indicates real-world outcome distribution has changed."
        ),
        metric      = "label_shift_delta",
        value       = delta,
        threshold   = THRESHOLDS["label_shift_high"],
        action      = (
            "Investigate clinical/operational changes in the data source. "
            "Label shift may require re-stratification of training data. "
            "Review class weights and decision threshold."
        )
    ))

    return alerts


def _prediction_shift_alert(concept_report: dict) -> list[Alert]:
    alerts    = []
    pred_info = concept_report.get("prediction_shift", {})
    delta     = abs(pred_info.get("delta_mean_proba", 0))

    if pred_info.get("proba_drift") and delta >= THRESHOLDS["proba_shift_critical"]:
        alerts.append(Alert(
            alert_id    = "PS-001",
            alert_type  = "PREDICTION_SHIFT",
            level       = "HIGH",
            title       = "Model Prediction Distribution Shifted",
            description = (
                f"Mean predicted probability changed from "
                f"{pred_info['ref_mean_proba']:.4f} to "
                f"{pred_info['prod_mean_proba']:.4f} "
                f"(delta={pred_info['delta_mean_proba']:+.4f}). "
                f"KS test: stat={pred_info['ks_stat']:.4f}, "
                f"p={pred_info['ks_pval']:.2e}. "
                f"Model is systematically over/under-predicting."
            ),
            metric    = "proba_shift_delta",
            value     = delta,
            threshold = THRESHOLDS["proba_shift_critical"],
            action    = (
                "Recalibrate model probability outputs. "
                "Apply isotonic regression or Platt scaling on recent data."
            )
        ))

    return alerts


def _feature_impact_alerts(feature_report: dict) -> list[Alert]:
    alerts   = []
    high_risk = feature_report.get("top_high_risk", [])

    if not high_risk:
        return alerts

    # Top high-impact drifted feature
    top = high_risk[0]
    if top["impact_score"] >= THRESHOLDS["impact_high"]:
        alerts.append(Alert(
            alert_id    = "FI-001",
            alert_type  = "FEATURE_IMPACT",
            level       = "HIGH",
            title       = f"High-Impact Feature Drift: {top['feature']}",
            description = (
                f"Feature '{top['feature']}' has highest impact score "
                f"({top['impact_score']:.4f}). "
                f"Drift score: {top['drift_score']:.4f}. "
                f"SHAP contribution: {top['shap_ref']:.6f} (reference). "
                f"SHAP Δ: {top['shap_delta_pct']:+.1f}%."
            ),
            metric      = "impact_score",
            value       = top["impact_score"],
            threshold   = THRESHOLDS["impact_high"],
            features    = [f["feature"] for f in high_risk[:5]],
            action      = (
                f"Feature '{top['feature']}' is both drifted and important. "
                f"Retrain with recent data emphasizing this feature. "
                f"Consider feature-specific monitoring dashboard."
            )
        ))

    # Count of high-risk features
    n_high = feature_report.get("n_high_risk", 0)
    if n_high >= 5:
        alerts.append(Alert(
            alert_id    = "FI-002",
            alert_type  = "FEATURE_IMPACT",
            level       = "HIGH" if n_high >= 10 else "MEDIUM",
            title       = f"{n_high} High-Risk Features Identified",
            description = (
                f"{n_high} features classified as HIGH risk "
                f"(drift × importance score >= {THRESHOLDS['impact_high']}). "
                f"FE_ features: {feature_report['fe_mean_impact']:.4f} mean impact. "
                f"Raw features: {feature_report['raw_mean_impact']:.4f} mean impact."
            ),
            metric      = "n_high_risk_features",
            value       = float(n_high),
            threshold   = 5.0,
            features    = [f["feature"] for f in high_risk[:10]],
            action      = (
                "Prioritize retraining pipeline. "
                "Focus data collection on high-risk drifted features."
            )
        ))

    return alerts


# ══════════════════════════════════════════════════════════════════════════
# Alert engine
# ══════════════════════════════════════════════════════════════════════════

class AlertEngine:
    """
    Consolidates all drift signals into structured alerts.
    Aggregates, deduplicates, and ranks alerts by severity.
    """

    LEVEL_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "STABLE": 4}

    def __init__(self, model_name: str = "lgbm_v1"):
        self.model_name = model_name
        self.alerts: list[Alert] = []

    def run(
        self,
        concept_report:  dict,
        data_drift_summary: dict,
        feature_report:  dict,
        ref_name:  str = "val",
        prod_name: str = "test",
    ) -> dict:
        """
        Run full alert generation pipeline.

        Parameters
        ----------
        concept_report      : output from ConceptDriftDetector.detect()
        data_drift_summary  : output from DataDriftDetector.summary_
        feature_report      : output from FeatureDriftAnalyzer.report_
        """
        logger.info("=" * 70)
        logger.info("DriftSentinel — Alert Engine")
        logger.info(f"  Model      : {self.model_name}")
        logger.info(f"  Reference  : {ref_name}")
        logger.info(f"  Production : {prod_name}")
        logger.info("=" * 70)

        # Generate alerts from each source
        logger.info("-" * 50)
        logger.info("Generating alerts from concept drift signals")
        cd_alerts = _concept_drift_alerts(concept_report)
        logger.info(f"  Generated: {len(cd_alerts)} concept drift alerts")

        logger.info("-" * 50)
        logger.info("Generating alerts from data drift signals")
        dd_alerts = _data_drift_alerts(data_drift_summary)
        logger.info(f"  Generated: {len(dd_alerts)} data drift alerts")

        logger.info("-" * 50)
        logger.info("Generating alerts from label shift signals")
        ls_alerts = _label_shift_alert(concept_report)
        logger.info(f"  Generated: {len(ls_alerts)} label shift alerts")

        logger.info("-" * 50)
        logger.info("Generating alerts from prediction shift signals")
        ps_alerts = _prediction_shift_alert(concept_report)
        logger.info(f"  Generated: {len(ps_alerts)} prediction shift alerts")

        logger.info("-" * 50)
        logger.info("Generating alerts from feature impact signals")
        fi_alerts = _feature_impact_alerts(feature_report)
        logger.info(f"  Generated: {len(fi_alerts)} feature impact alerts")

        # Combine and sort
        self.alerts = (
            cd_alerts + dd_alerts +
            ls_alerts + ps_alerts + fi_alerts
        )
        self.alerts.sort(
            key=lambda a: self.LEVEL_ORDER.get(a.level, 99)
        )

        # ── Summary ────────────────────────────────────────────────────────
        logger.info("=" * 70)
        logger.info("ALERT SUMMARY")
        logger.info("=" * 70)

        level_counts = {}
        for a in self.alerts:
            level_counts[a.level] = level_counts.get(a.level, 0) + 1

        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            cnt = level_counts.get(level, 0)
            if cnt > 0:
                logger.info(f"  {level:<10}: {cnt} alert(s)")

        logger.info("-" * 70)
        logger.info("ALL ALERTS:")
        logger.info("-" * 70)

        for a in self.alerts:
            logger.info(f"\n  [{a.level}] {a.alert_id} — {a.title}")
            logger.info(f"  Type       : {a.alert_type}")
            logger.info(f"  Metric     : {a.metric} = {a.value:.4f} (threshold: {a.threshold:.4f})")
            logger.info(f"  Description: {a.description}")
            logger.info(f"  Action     : {a.action}")
            if a.features:
                logger.info(f"  Features   : {', '.join(a.features[:5])}")

        # ── Overall system status ──────────────────────────────────────────
        critical_count = level_counts.get("CRITICAL", 0)
        high_count     = level_counts.get("HIGH",     0)

        if critical_count >= 2:
            system_status = "CRITICAL"
            recommendation = (
                "IMMEDIATE ACTION REQUIRED. "
                "Model is unreliable in production. "
                "Halt predictions or apply fallback model. "
                "Trigger emergency retraining pipeline."
            )
        elif critical_count >= 1 or high_count >= 3:
            system_status = "HIGH"
            recommendation = (
                "Model performance significantly degraded. "
                "Schedule retraining within 24 hours. "
                "Increase monitoring frequency."
            )
        elif high_count >= 1:
            system_status = "MODERATE"
            recommendation = (
                "Drift detected. Monitor closely. "
                "Plan retraining in next maintenance window."
            )
        else:
            system_status = "STABLE"
            recommendation = "No significant drift. Continue normal monitoring."

        logger.info("=" * 70)
        logger.info(f"SYSTEM STATUS    : {system_status}")
        logger.info(f"TOTAL ALERTS     : {len(self.alerts)}")
        logger.info(f"CRITICAL         : {critical_count}")
        logger.info(f"HIGH             : {high_count}")
        logger.info(f"RECOMMENDATION   : {recommendation}")
        logger.info("=" * 70)

        # ── Build full report ──────────────────────────────────────────────
        report = {
            "timestamp"      : datetime.now().isoformat(),
            "model_name"     : self.model_name,
            "ref_name"       : ref_name,
            "prod_name"      : prod_name,
            "system_status"  : system_status,
            "recommendation" : recommendation,
            "alert_counts"   : level_counts,
            "total_alerts"   : len(self.alerts),
            "alerts"         : [a.to_dict() for a in self.alerts],
            "drift_summary"  : {
                "concept_severity"  : concept_report.get("severity"),
                "n_evidence"        : concept_report.get("n_evidence"),
                "auc_degradation"   : concept_report.get("auc_degradation"),
                "f1_degradation"    : concept_report.get("f1_degradation"),
                "n_data_drifted"    : data_drift_summary.get("n_drifted"),
                "n_data_critical"   : data_drift_summary.get("n_critical_psi"),
                "n_feature_high"    : feature_report.get("n_high_risk"),
                "label_shift_delta" : concept_report.get(
                    "label_shift", {}
                ).get("delta_pos_rate"),
            },
        }

        # ── Save ───────────────────────────────────────────────────────────
        # Tier 1.7: this fixed path would have destroyed the original
        # entry-cohort alert artifacts during the Tier 0 sweep.
        from src.monitoring.artifact_io import write_artifact
        alert_path = ALERTS_DIR / f"alert_report_{ref_name}_{prod_name}.json"
        write_artifact(alert_path, report, overwrite=True, preserve=True)
        logger.info(f"Alert report saved: {alert_path}")

        # Summary CSV
        summary_rows = []
        for a in self.alerts:
            summary_rows.append({
                "alert_id"  : a.alert_id,
                "level"     : a.level,
                "type"      : a.alert_type,
                "title"     : a.title,
                "metric"    : a.metric,
                "value"     : a.value,
                "threshold" : a.threshold,
                "action"    : a.action,
                "timestamp" : a.timestamp,
            })
        summary_df  = pd.DataFrame(summary_rows)
        csv_path    = ALERTS_DIR / f"alert_summary_{ref_name}_{prod_name}.csv"
        from src.monitoring.artifact_io import write_dataframe
        write_dataframe(csv_path, summary_df, overwrite=True, preserve=True)
        logger.info(f"Alert summary CSV : {csv_path}")

        return report


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_alerting() -> dict:
    logger.info("=" * 70)
    logger.info("DriftSentinel — Alert Engine Run")
    logger.info("=" * 70)

    # ── Load concept drift report ──────────────────────────────────────────
    concept_path = REPORTS_DIR / "concept_drift_val_test.json"
    with open(concept_path) as f:
        concept_report = json.load(f)
    logger.info(f"Loaded concept drift report: {concept_path}")

    # ── Load data drift report ─────────────────────────────────────────────
    data_drift_path = REPORTS_DIR / "data_drift_report.json"
    with open(data_drift_path) as f:
        data_drift_full = json.load(f)
    data_drift_summary = data_drift_full.get("test", {})
    logger.info(f"Loaded data drift report  : {data_drift_path}")

    # ── Load feature drift report ──────────────────────────────────────────
    feat_path = REPORTS_DIR / "feature_drift_report_val_test.json"
    with open(feat_path) as f:
        feature_report = json.load(f)
    logger.info(f"Loaded feature drift report: {feat_path}")

    # ── Run alert engine ───────────────────────────────────────────────────
    engine = AlertEngine(model_name="lgbm_v1")
    report = engine.run(
        concept_report     = concept_report,
        data_drift_summary = data_drift_summary,
        feature_report     = feature_report,
        ref_name           = "val",
        prod_name          = "test",
    )

    # ── Print final dashboard ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DRIFTSENTINEL ALERT DASHBOARD")
    print("=" * 65)
    print(f"  System Status    : {report['system_status']}")
    print(f"  Total Alerts     : {report['total_alerts']}")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        cnt = report["alert_counts"].get(level, 0)
        if cnt > 0:
            print(f"  {level:<12}   : {cnt}")
    print("-" * 65)
    print(f"  AUC degradation  : {report['drift_summary']['auc_degradation']:+.4f}")
    print(f"  F1  degradation  : {report['drift_summary']['f1_degradation']:+.4f}")
    print(f"  Concept severity : {report['drift_summary']['concept_severity']}")
    print(f"  Data drifted     : {report['drift_summary']['n_data_drifted']}/53")
    print(f"  Feature HIGH risk: {report['drift_summary']['n_feature_high']}")
    print(f"  Label shift Δ    : {report['drift_summary']['label_shift_delta']:+.4f}")
    print("-" * 65)
    print(f"  RECOMMENDATION   : {report['recommendation']}")
    print("=" * 65)
    print(f"\n  Alert report     : outputs/alerts/")

    return report


if __name__ == "__main__":
    run_alerting()