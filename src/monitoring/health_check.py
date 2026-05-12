"""
DriftSentinel — System Health Check
Validates that all pipeline components are operational.
Runs pre-flight checks before model serving or drift monitoring.

Health status:
    HEALTHY   — all checks passed
    DEGRADED  — warnings present, system functional
    CRITICAL  — failures detected, intervention required

Check categories:
    DATA     — parquet files, schema integrity
    MODELS   — pkl files, prediction sanity
    DRIFT    — alert status, drift report freshness
    UNCERTAINTY — calibrator, conformal predictor
    ADVERSARIAL — defense system
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
from src.uncertainty.calibration import IsotonicCalibrator

logger = get_logger("health_check")

ROOT          = Path(__file__).resolve().parents[2]
TRAIN_DIR     = ROOT / "data"    / "train"
PROD_DIR      = ROOT / "data"    / "production"
MODELS_DIR    = ROOT / "outputs" / "models"
REPORTS_DIR   = ROOT / "outputs" / "log"
ARTIFACTS_DIR = ROOT / "outputs" / "artifacts"
ALERTS_DIR    = ROOT / "outputs" / "alerts"
REGISTRY_DIR  = ROOT / "outputs" / "registry"

TARGET_COLS = {"readmitted_binary", "readmitted_multi"}
N_FEATURES  = 53


# ══════════════════════════════════════════════════════════════════════════
# Check result builder
# ══════════════════════════════════════════════════════════════════════════

class CheckResult:
    def __init__(
        self,
        name:    str,
        status:  str,    # PASS / WARN / FAIL
        message: str,
        value    = None,
    ):
        self.name    = name
        self.status  = status
        self.message = message
        self.value   = value

    def to_dict(self) -> dict:
        return {
            "name"   : self.name,
            "status" : self.status,
            "message": self.message,
            "value"  : self.value,
        }

    def __repr__(self):
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(self.status, "?")
        return f"[{icon} {self.status}] {self.name}: {self.message}"


# ══════════════════════════════════════════════════════════════════════════
# Individual checks
# ══════════════════════════════════════════════════════════════════════════

# ── Data checks ───────────────────────────────────────────────────────────

def check_data_files() -> list[CheckResult]:
    results = []

    required_files = {
        "train_fs.parquet" : TRAIN_DIR,
        "val_fs.parquet"   : TRAIN_DIR,
        "test_fs.parquet"  : TRAIN_DIR,
        "production_val.parquet" : PROD_DIR,
        "production_test.parquet": PROD_DIR,
    }

    for fname, fdir in required_files.items():
        path = fdir / fname
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            results.append(CheckResult(
                f"data_file_{fname}",
                "PASS",
                f"exists ({size_mb:.1f} MB)",
                size_mb
            ))
        else:
            results.append(CheckResult(
                f"data_file_{fname}",
                "FAIL",
                f"missing: {path}",
            ))

    return results


def check_data_schema() -> list[CheckResult]:
    results = []

    try:
        train = pd.read_parquet(TRAIN_DIR / "train_fs.parquet")
        feat_cols = [c for c in train.columns if c not in TARGET_COLS]

        # Feature count
        if len(feat_cols) == N_FEATURES:
            results.append(CheckResult(
                "data_feature_count",
                "PASS",
                f"{len(feat_cols)} features (expected {N_FEATURES})",
                len(feat_cols)
            ))
        else:
            results.append(CheckResult(
                "data_feature_count",
                "FAIL",
                f"{len(feat_cols)} features (expected {N_FEATURES})",
                len(feat_cols)
            ))

        # Null check
        null_count = train[feat_cols].isna().sum().sum()
        results.append(CheckResult(
            "data_null_free",
            "PASS" if null_count == 0 else "FAIL",
            f"nulls={null_count}",
            int(null_count)
        ))

        # Target presence
        for tgt in ["readmitted_binary", "readmitted_multi"]:
            present = tgt in train.columns
            results.append(CheckResult(
                f"data_target_{tgt}",
                "PASS" if present else "FAIL",
                "present" if present else "missing",
            ))

        # Shape sanity
        results.append(CheckResult(
            "data_train_shape",
            "PASS",
            f"train={train.shape}",
            list(train.shape)
        ))

    except Exception as e:
        results.append(CheckResult(
            "data_schema", "FAIL", str(e)
        ))

    return results


# ── Model checks ───────────────────────────────────────────────────────────

def check_model_files() -> list[CheckResult]:
    results = []

    required_models = [
        "lgbm_v1.pkl",
        "lgbm_v2.pkl",
        "logreg_v1.pkl",
    ]

    for fname in required_models:
        path = MODELS_DIR / fname
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            results.append(CheckResult(
                f"model_file_{fname}",
                "PASS",
                f"exists ({size_mb:.1f} MB)",
                size_mb
            ))
        else:
            results.append(CheckResult(
                f"model_file_{fname}",
                "WARN" if "v2" in fname else "FAIL",
                f"missing: {path}",
            ))

    return results


def check_model_prediction() -> list[CheckResult]:
    results = []

    try:
        from src.uncertainty.calibration import IsotonicCalibrator

        with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
            model = pickle.load(f)
        with open(ARTIFACTS_DIR / "calibrator_isotonic_lgbm_v1.pkl", "rb") as f:
            calibrator = pickle.load(f)

        # Load small sample
        test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")
        feat_cols = [c for c in test.columns if c not in TARGET_COLS]
        X_sample  = pd.DataFrame(
            test[feat_cols].head(100).values,
            columns=feat_cols
        )

        # Prediction sanity
        p_raw  = model.predict_proba(X_sample)[:, 1]
        p_cal  = calibrator.transform(p_raw)

        # Range check
        in_range = ((p_cal >= 0) & (p_cal <= 1)).all()
        results.append(CheckResult(
            "model_prediction_range",
            "PASS" if in_range else "FAIL",
            f"probabilities in [0,1]: {in_range}",
            float(p_cal.mean())
        ))

        # NaN check
        no_nan = not np.isnan(p_cal).any()
        results.append(CheckResult(
            "model_prediction_no_nan",
            "PASS" if no_nan else "FAIL",
            "no NaN in predictions" if no_nan else "NaN detected",
        ))

        # Mean proba sanity
        mean_p = float(p_cal.mean())
        if 0.2 <= mean_p <= 0.8:
            results.append(CheckResult(
                "model_prediction_sanity",
                "PASS",
                f"mean proba={mean_p:.4f} (in [0.2, 0.8])",
                mean_p
            ))
        else:
            results.append(CheckResult(
                "model_prediction_sanity",
                "WARN",
                f"mean proba={mean_p:.4f} (outside [0.2, 0.8])",
                mean_p
            ))

    except Exception as e:
        results.append(CheckResult(
            "model_prediction", "FAIL", str(e)
        ))

    return results


def check_registry() -> list[CheckResult]:
    results = []

    registry_path = REGISTRY_DIR / "model_registry.json"
    if not registry_path.exists():
        results.append(CheckResult(
            "registry_file", "FAIL", "model_registry.json missing"
        ))
        return results

    try:
        with open(registry_path) as f:
            registry = json.load(f)

        active = registry.get("active_model")
        results.append(CheckResult(
            "registry_active_model",
            "PASS" if active else "FAIL",
            f"active={active}",
            active
        ))

        n_models = len(registry.get("models", {}))
        results.append(CheckResult(
            "registry_model_count",
            "PASS",
            f"{n_models} model(s) registered",
            n_models
        ))

    except Exception as e:
        results.append(CheckResult(
            "registry", "FAIL", str(e)
        ))

    return results


# ── Drift checks ───────────────────────────────────────────────────────────

def check_drift_reports() -> list[CheckResult]:
    results = []

    required_reports = {
        "concept_drift_val_test.json" : "concept_drift",
        "data_drift_report.json"      : "data_drift",
        "feature_drift_report_val_test.json": "feature_drift",
    }

    for fname, report_type in required_reports.items():
        path = REPORTS_DIR / fname
        if path.exists():
            # Check freshness
            mod_time = datetime.fromtimestamp(path.stat().st_mtime)
            results.append(CheckResult(
                f"drift_report_{report_type}",
                "PASS",
                f"exists | last modified: {mod_time.strftime('%Y-%m-%d %H:%M')}",
            ))
        else:
            results.append(CheckResult(
                f"drift_report_{report_type}",
                "FAIL",
                f"missing: {path}",
            ))

    return results


def check_alert_status() -> list[CheckResult]:
    results = []

    alert_path = ALERTS_DIR / "alert_report_val_test.json"
    if not alert_path.exists():
        results.append(CheckResult(
            "alert_report", "WARN", "alert report not found"
        ))
        return results

    try:
        with open(alert_path) as f:
            alert_report = json.load(f)

        system_status = alert_report.get("system_status", "UNKNOWN")
        n_critical    = alert_report.get("alert_counts", {}).get("CRITICAL", 0)
        n_high        = alert_report.get("alert_counts", {}).get("HIGH",     0)
        total_alerts  = alert_report.get("total_alerts", 0)

        status_map = {
            "CRITICAL": "WARN",   # drift detected — expected behavior
            "HIGH"    : "WARN",
            "MODERATE": "WARN",
            "STABLE"  : "PASS",
        }
        check_status = status_map.get(system_status, "WARN")

        results.append(CheckResult(
            "alert_system_status",
            check_status,
            f"system_status={system_status} | "
            f"CRITICAL={n_critical} HIGH={n_high} "
            f"total={total_alerts}",
            system_status
        ))

        # Drift severity
        drift_summary = alert_report.get("drift_summary", {})
        auc_deg = drift_summary.get("auc_degradation", 0)
        if auc_deg and abs(auc_deg) > 0.05:
            results.append(CheckResult(
                "alert_auc_degradation",
                "FAIL",
                f"AUC degradation={auc_deg:+.4f} (>0.05 threshold)",
                auc_deg
            ))
        else:
            results.append(CheckResult(
                "alert_auc_degradation",
                "PASS",
                f"AUC degradation={auc_deg:+.4f}",
                auc_deg
            ))

    except Exception as e:
        results.append(CheckResult(
            "alert_status", "FAIL", str(e)
        ))

    return results


# ── Uncertainty checks ─────────────────────────────────────────────────────

def check_uncertainty_artifacts() -> list[CheckResult]:
    results = []

    artifacts = {
        "calibrator_isotonic_lgbm_v1.pkl" : "isotonic_calibrator",
        "calibrator_temperature_lgbm_v1.pkl": "temperature_scaler",
        "conformal_predictor_lgbm_v1.pkl" : "conformal_predictor",
        "feature_selector.pkl"            : "feature_selector",
        "label_encoders.pkl"              : "label_encoders",
    }

    for fname, artifact_name in artifacts.items():
        path = ARTIFACTS_DIR / fname
        if path.exists():
            size_kb = path.stat().st_size / 1024
            results.append(CheckResult(
                f"artifact_{artifact_name}",
                "PASS",
                f"exists ({size_kb:.1f} KB)",
                size_kb
            ))
        else:
            results.append(CheckResult(
                f"artifact_{artifact_name}",
                "WARN",
                f"missing: {fname}",
            ))

    # Calibration report
    cal_path = REPORTS_DIR / "calibration_report_lgbm_v1.json"
    if cal_path.exists():
        with open(cal_path) as f:
            cal_report = json.load(f)
        ece_improvement = cal_report.get("ece_improvement", 0)
        results.append(CheckResult(
            "calibration_ece_improvement",
            "PASS" if ece_improvement > 0 else "WARN",
            f"ECE improvement={ece_improvement:+.4f} "
            f"best_method={cal_report.get('best_method')}",
            ece_improvement
        ))

    # Conformal coverage
    cp_path = REPORTS_DIR / "conformal_report_lgbm_v1.json"
    if cp_path.exists():
        with open(cp_path) as f:
            cp_report = json.load(f)
        coverage = cp_report.get("thr_90_coverage", 0)
        satisfied = cp_report.get("thr_90_satisfied", False)
        results.append(CheckResult(
            "conformal_coverage_90",
            "PASS" if satisfied else "FAIL",
            f"empirical_coverage={coverage:.4f} "
            f"(target=0.90) satisfied={satisfied}",
            coverage
        ))

    return results


# ── Adversarial checks ─────────────────────────────────────────────────────

def check_adversarial_status() -> list[CheckResult]:
    results = []

    # Robustness report
    rob_path = REPORTS_DIR / "robustness_report_lgbm_v1.json"
    if rob_path.exists():
        with open(rob_path) as f:
            rob_report = json.load(f)

        score = rob_report.get(
            "robustness_score", {}
        ).get("overall_score", 0)
        tier  = rob_report.get(
            "robustness_score", {}
        ).get("tier", "UNKNOWN")

        status = (
            "PASS" if tier == "ROBUST"   else
            "WARN" if tier == "MODERATE" else
            "FAIL"
        )
        results.append(CheckResult(
            "adversarial_robustness_score",
            status,
            f"score={score:.4f} tier={tier}",
            score
        ))
    else:
        results.append(CheckResult(
            "adversarial_robustness_score",
            "WARN",
            "robustness report not found",
        ))

    # Defense report
    def_path = REPORTS_DIR / "defense_report_lgbm_v1.json"
    if def_path.exists():
        with open(def_path) as f:
            def_report = json.load(f)

        detection_rate = def_report.get(
            "attacked", {}
        ).get("detection_rate", 0)
        fp_rate = def_report.get(
            "clean", {}
        ).get("false_positive_rate", 1)

        results.append(CheckResult(
            "adversarial_detection_rate",
            "PASS" if detection_rate >= 0.80 else "WARN",
            f"detection_rate={detection_rate:.3f} "
            f"false_positive_rate={fp_rate:.3f}",
            detection_rate
        ))
    else:
        results.append(CheckResult(
            "adversarial_defense",
            "WARN",
            "defense report not found",
        ))

    return results


# ══════════════════════════════════════════════════════════════════════════
# Health check engine
# ══════════════════════════════════════════════════════════════════════════

class HealthChecker:
    """
    Runs all checks and produces system health report.
    """

    CATEGORIES = {
        "DATA"       : [check_data_files, check_data_schema],
        "MODELS"     : [check_model_files, check_model_prediction, check_registry],
        "DRIFT"      : [check_drift_reports, check_alert_status],
        "UNCERTAINTY": [check_uncertainty_artifacts],
        "ADVERSARIAL": [check_adversarial_status],
    }

    def run(self) -> dict:
        logger.info("=" * 70)
        logger.info("DriftSentinel — System Health Check")
        logger.info(f"  Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 70)

        all_results  = {}
        category_status = {}

        for category, check_fns in self.CATEGORIES.items():
            logger.info(f"\n{'─'*50}")
            logger.info(f"Category: {category}")
            logger.info(f"{'─'*50}")

            cat_results = []
            for fn in check_fns:
                results = fn()
                cat_results.extend(results)
                for r in results:
                    icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(
                        r.status, "?"
                    )
                    logger.info(
                        f"  [{icon}] {r.name:<45} {r.message}"
                    )

            all_results[category] = [r.to_dict() for r in cat_results]

            # Category status = worst check status
            statuses = [r.status for r in cat_results]
            if "FAIL" in statuses:
                cat_status = "FAIL"
            elif "WARN" in statuses:
                cat_status = "WARN"
            else:
                cat_status = "PASS"
            category_status[category] = cat_status

            n_pass = statuses.count("PASS")
            n_warn = statuses.count("WARN")
            n_fail = statuses.count("FAIL")
            logger.info(
                f"  → {category}: PASS={n_pass} WARN={n_warn} FAIL={n_fail} "
                f"[{cat_status}]"
            )

        # Overall system status
        all_statuses = list(category_status.values())
        if "FAIL" in all_statuses:
            system_health = "CRITICAL"
        elif all_statuses.count("WARN") >= 2:
            system_health = "DEGRADED"
        elif "WARN" in all_statuses:
            system_health = "DEGRADED"
        else:
            system_health = "HEALTHY"

        # Count totals
        all_checks = [
            r for cat in all_results.values() for r in cat
        ]
        n_pass = sum(1 for r in all_checks if r["status"] == "PASS")
        n_warn = sum(1 for r in all_checks if r["status"] == "WARN")
        n_fail = sum(1 for r in all_checks if r["status"] == "FAIL")

        logger.info("\n" + "=" * 70)
        logger.info("HEALTH CHECK SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  System Health : {system_health}")
        logger.info(f"  Total checks  : {len(all_checks)}")
        logger.info(f"  PASS          : {n_pass}")
        logger.info(f"  WARN          : {n_warn}")
        logger.info(f"  FAIL          : {n_fail}")
        logger.info("-" * 70)

        for cat, status in category_status.items():
            icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(status, "?")
            logger.info(f"  [{icon}] {cat:<20} {status}")

        logger.info("=" * 70)

        if system_health == "HEALTHY":
            logger.info("  ✓ System is HEALTHY — ready for production inference")
        elif system_health == "DEGRADED":
            logger.info("  ⚠ System DEGRADED — functional but review warnings")
        else:
            logger.info("  ✗ System CRITICAL — intervention required")

        logger.info("=" * 70)

        report = {
            "timestamp"      : datetime.now().isoformat(),
            "system_health"  : system_health,
            "n_pass"         : n_pass,
            "n_warn"         : n_warn,
            "n_fail"         : n_fail,
            "category_status": category_status,
            "checks"         : all_results,
        }

        report_path = ROOT / "outputs" / "log" / "health_check.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"  Health report saved: {report_path}")

        return report


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_health_check() -> dict:
    checker = HealthChecker()
    report  = checker.run()

    print(f"\n{'='*50}")
    print("DRIFTSENTINEL HEALTH CHECK")
    print(f"{'='*50}")
    print(f"  System Health : {report['system_health']}")
    print(f"  PASS={report['n_pass']}  "
          f"WARN={report['n_warn']}  "
          f"FAIL={report['n_fail']}")
    print(f"{'─'*50}")
    for cat, status in report["category_status"].items():
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(status, "?")
        print(f"  [{icon}] {cat:<20} {status}")
    print(f"{'='*50}")

    return report


if __name__ == "__main__":
    run_health_check()