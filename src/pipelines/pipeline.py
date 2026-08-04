"""
DriftSentinel — Training Pipeline
Orchestrates the full data preparation pipeline:
    loader -> validator -> splitter -> preprocessor ->
    engineer -> selector -> consistency -> save artifacts

Outputs:
    data/train/          : train_fs.parquet, val_fs.parquet, test_fs.parquet
    data/production/     : val_fs.parquet, test_fs.parquet  (drift simulation)
    outputs/artifacts/   : all fitted artifacts (encoders, scalers, fe stats)
    outputs/log/         : per-module logs + pipeline.log
"""

import time
import json
import pickle
from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.monitoring.logger      import get_logger
from src.data.loader            import load_raw
from src.data.validator         import validate
from src.data.splitter          import split
from src.data.preprocessor      import Preprocessor
from src.features.engineer      import FeatureEngineer
from src.features.selector      import FeatureSelector
from src.features.consistency   import ConsistencyChecker

logger = get_logger("pipeline")

# ── Output directories ─────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parents[2]
TRAIN_DIR      = ROOT_DIR / "data" / "train"
PRODUCTION_DIR = ROOT_DIR / "data" / "production"
ARTIFACTS_DIR  = ROOT_DIR / "outputs" / "artifacts"
LOG_DIR        = ROOT_DIR / "outputs" / "log"

for d in [TRAIN_DIR, PRODUCTION_DIR, ARTIFACTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def run_pipeline() -> dict:
    """
    Execute full DriftSentinel training pipeline.

    Returns
    -------
    summary : dict with shapes, paths, timing, and consistency report
    """
    pipeline_start = time.perf_counter()
    summary         = {}

    logger.info("=" * 70)
    logger.info("DriftSentinel — Training Pipeline START")
    logger.info("=" * 70)

    # ── Stage 1: Load ──────────────────────────────────────────────────────
    logger.info("\n>>> STAGE 1: Data Loading")
    t0 = time.perf_counter()

    df, ids_df, loader_meta = load_raw()

    summary["stage1_load"] = {
        "shape"       : list(df.shape),
        "memory_mb"   : loader_meta["memory_mb"],
        "load_time_s" : loader_meta["load_time_s"],
        "null_cols"   : len(loader_meta["null_cols"]),
    }
    logger.info(f"Stage 1 complete — {time.perf_counter() - t0:.2f}s")

    # ── Stage 2: Validate ─────────────────────────────────────────────────
    logger.info("\n>>> STAGE 2: Data Validation")
    t0 = time.perf_counter()

    val_report = validate(df)

    summary["stage2_validate"] = {
        "passed" : val_report["passed"],
        "warned" : val_report["warned"],
        "failed" : val_report["failed"],
        "ready"  : val_report["ready"],
    }

    if not val_report["ready"]:
        logger.error("STAGE 2 FAILED — pipeline halted")
        raise RuntimeError(
            f"Validation failed with {val_report['failed']} critical issues"
        )
    logger.info(f"Stage 2 complete — {time.perf_counter() - t0:.2f}s")

    # ── Stage 3: Split ────────────────────────────────────────────────────
    logger.info("\n>>> STAGE 3: Patient-Level Split")
    t0 = time.perf_counter()

    train_raw, val_raw, test_raw, split_meta = split(df)

    summary["stage3_split"] = {
        "n_train_rows"     : split_meta["n_train_rows"],
        "n_val_rows"       : split_meta["n_val_rows"],
        "n_test_rows"      : split_meta["n_test_rows"],
        "n_train_patients" : split_meta["n_train_patients"],
        "n_val_patients"   : split_meta["n_val_patients"],
        "n_test_patients"  : split_meta["n_test_patients"],
        "leakage_free"     : split_meta["leakage_free"],
    }
    logger.info(
        f"Stage 3 complete — "
        f"train={split_meta['n_train_rows']:,} "
        f"val={split_meta['n_val_rows']:,} "
        f"test={split_meta['n_test_rows']:,} "
        f"— {time.perf_counter() - t0:.2f}s"
    )

    # ── Stage 4: Preprocess ───────────────────────────────────────────────
    logger.info("\n>>> STAGE 4: Preprocessing (fit on train only)")
    t0 = time.perf_counter()

    preprocessor = Preprocessor()
    train_clean  = preprocessor.fit_transform(train_raw)
    val_clean    = preprocessor.transform(val_raw,  split_name="val")
    test_clean   = preprocessor.transform(test_raw, split_name="test")

    summary["stage4_preprocess"] = {
        "train_shape" : list(train_clean.shape),
        "val_shape"   : list(val_clean.shape),
        "test_shape"  : list(test_clean.shape),
        "artifacts"   : [
            "label_encoders.pkl",
            "impute_modes.json",
            "age_ordinal_map.json",
            "preprocessor_config.json",
        ],
    }
    logger.info(f"Stage 4 complete — {time.perf_counter() - t0:.2f}s")

    # ── Stage 5: Feature Engineering ─────────────────────────────────────
    logger.info("\n>>> STAGE 5: Feature Engineering (fit on train only)")
    t0 = time.perf_counter()

    engineer = FeatureEngineer()
    train_fe = engineer.fit_transform(train_clean)
    val_fe   = engineer.transform(val_clean,  split_name="val")
    test_fe  = engineer.transform(test_clean, split_name="test")

    fe_cols = [c for c in train_fe.columns if c.startswith("FE_")]

    summary["stage5_engineer"] = {
        "train_shape"   : list(train_fe.shape),
        "val_shape"     : list(val_fe.shape),
        "test_shape"    : list(test_fe.shape),
        "fe_cols_count" : len(fe_cols),
        "fe_cols"       : fe_cols,
    }
    logger.info(
        f"Stage 5 complete — "
        f"{len(fe_cols)} FE_ features created "
        f"— {time.perf_counter() - t0:.2f}s"
    )

    # ── Stage 6: Feature Selection ────────────────────────────────────────
    logger.info("\n>>> STAGE 6: Feature Selection (fit on train only)")
    t0 = time.perf_counter()

    selector = FeatureSelector()
    selector.fit(train_fe)

    train_fs = selector.transform(train_fe, split_name="train")
    val_fs   = selector.transform(val_fe,   split_name="val")
    test_fs  = selector.transform(test_fe,  split_name="test")

    summary["stage6_select"] = {
        "train_shape"       : list(train_fs.shape),
        "val_shape"         : list(val_fs.shape),
        "test_shape"        : list(test_fs.shape),
        "n_selected"        : len(selector.selected_features),
        "selected_features" : selector.selected_features,
    }
    logger.info(
        f"Stage 6 complete — "
        f"{len(selector.selected_features)} features selected "
        f"— {time.perf_counter() - t0:.2f}s"
    )

    # ── Stage 7: Consistency Check ────────────────────────────────────────
    logger.info("\n>>> STAGE 7: Consistency Check")
    t0 = time.perf_counter()

    checker            = ConsistencyChecker()
    consistency_report = checker.run(train_fs, val_fs, test_fs)

    summary["stage7_consistency"] = {
        "passed"         : consistency_report["passed"],
        "warned"         : consistency_report["warned"],
        "failed"         : consistency_report["failed"],
        "ready"          : consistency_report["ready"],
        "psi_drifted"    : len(consistency_report.get("drifted_psi", [])),
        "ks_drifted"     : len(consistency_report.get("ks_drifted", [])),
    }

    if not consistency_report["ready"]:
        logger.warning("STAGE 7 — consistency warnings detected (expected shift in entry-cohort split)")
        logger.warning(f"  FAIL={consistency_report['failed']} items are expected entry-cohort shift signals")
        logger.warning(f"  PSI drifted={len(consistency_report.get('drifted_psi',[]))} — drift simulation confirmed")
        logger.warning("  Pipeline proceeds — drift detection modules will address these signals")
    else:
        logger.info("Stage 7 complete — all consistency checks passed")
    
    logger.info(f"Stage 7 complete — {time.perf_counter() - t0:.2f}s")

    # ── Stage 8: Save processed data ──────────────────────────────────────
    logger.info("\n>>> STAGE 8: Saving processed splits")
    t0 = time.perf_counter()

    # train/ — model training reference window
    train_path = TRAIN_DIR / "train_fs.parquet"
    val_path   = TRAIN_DIR / "val_fs.parquet"
    test_path  = TRAIN_DIR / "test_fs.parquet"

    train_fs.to_parquet(train_path, index=False)
    val_fs.to_parquet(val_path,     index=False)
    test_fs.to_parquet(test_path,   index=False)

    logger.info(f"  train_fs.parquet -> {train_path}  shape={train_fs.shape}")
    logger.info(f"  val_fs.parquet   -> {val_path}    shape={val_fs.shape}")
    logger.info(f"  test_fs.parquet  -> {test_path}   shape={test_fs.shape}")

    # production/ — drift simulation windows
    prod_val_path  = PRODUCTION_DIR / "production_val.parquet"
    prod_test_path = PRODUCTION_DIR / "production_test.parquet"

    val_fs.to_parquet(prod_val_path,   index=False)
    test_fs.to_parquet(prod_test_path, index=False)

    logger.info(f"  production_val.parquet  -> {prod_val_path}")
    logger.info(f"  production_test.parquet -> {prod_test_path}")

    # Save pipeline object references
    pipeline_obj = {
        "preprocessor" : preprocessor,
        "engineer"     : engineer,
        "selector"     : selector,
    }
    pipeline_path = ARTIFACTS_DIR / "pipeline_objects.pkl"
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline_obj, f)
    logger.info(f"  pipeline_objects.pkl -> {pipeline_path}")

    summary["stage8_save"] = {
        "train_path"      : str(train_path),
        "val_path"        : str(val_path),
        "test_path"       : str(test_path),
        "prod_val_path"   : str(prod_val_path),
        "prod_test_path"  : str(prod_test_path),
        "pipeline_pkl"    : str(pipeline_path),
    }
    logger.info(f"Stage 8 complete — {time.perf_counter() - t0:.2f}s")

    # ── Final summary ──────────────────────────────────────────────────────
    total_time = time.perf_counter() - pipeline_start
    summary["total_time_s"] = round(total_time, 2)
    
    # ── Pipeline readiness gate (Phase 1.0 — audit F2) ────────────────────
    # This was `summary["pipeline_ready"] = True`, written unconditionally over
    # a check reporting 12 FAILs. It is now an EVALUATED expression that names
    # every failure and separates distribution-shift observations (this
    # project's subject matter) from integrity violations (which still block).
    from src.features.consistency import evaluate_gate

    gate = evaluate_gate(consistency_report, drift_expected=True)
    summary["pipeline_ready"] = gate["ready"]
    summary["readiness_gate"] = gate

    logger.info("-" * 50)
    logger.info("Pipeline readiness gate")
    logger.info(f"  Rule                : {gate['rule']}")
    logger.info(f"  Expected failures   : {gate['n_expected_failures']} "
                f"{[e['check'] for e in gate['expected_failures']]}")
    logger.info(f"  Unexpected failures : {gate['n_unexpected_failures']} "
                f"{[u['check'] for u in gate['unexpected_failures']]}")
    logger.info(f"  Decision            : ready={gate['ready']}")
    logger.info(f"  Reason              : {gate['reason']}")
    if not gate["ready"]:
        logger.error("PIPELINE NOT READY — unexpected failures present")

    summary["drift_signals_detected"] = {
        "psi_critical"     : len(consistency_report.get("drifted_psi", [])),
        "ks_drifted_pairs" : len(consistency_report.get("ks_drifted",  [])),
        "consistency_fails": consistency_report["failed"],
        "note"             : ("Expected entry-cohort shift — DriftSentinel target "
                              "scenario. NOT temporal drift: see "
                              "outputs/reports/temporal_validity.json"),
    }

    from src.monitoring.artifact_io import write_artifact
    summary_path = LOG_DIR / "pipeline_summary.json"
    write_artifact(summary_path, summary, overwrite=True, preserve=True)

    logger.info("\n" + "=" * 70)
    logger.info("DriftSentinel — Training Pipeline COMPLETE")
    logger.info(f"  Total time          : {total_time:.2f}s")
    logger.info(f"  Train shape         : {train_fs.shape}")
    logger.info(f"  Val shape           : {val_fs.shape}")
    logger.info(f"  Test shape          : {test_fs.shape}")
    logger.info(f"  Selected features   : {len(selector.selected_features)}")
    logger.info(f"  PSI drifted         : {summary['drift_signals_detected']['psi_critical']}")
    logger.info(f"  KS  drifted         : {summary['drift_signals_detected']['ks_drifted_pairs']}")
    logger.info(f"  Pipeline ready      : {summary['pipeline_ready']}")
    logger.info(f"  Summary saved       : {summary_path}")
    logger.info("=" * 70)
    logger.info("Next step: 02_preprocessing_audit.ipynb")
    logger.info("=" * 70)

    return summary


if __name__ == "__main__":
    summary = run_pipeline()

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for stage, info in summary.items():
        if isinstance(info, dict):
            print(f"\n{stage}:")
            for k, v in info.items():
                if not isinstance(v, list) or len(str(v)) < 80:
                    print(f"  {k:<25} : {v}")
        else:
            print(f"{stage:<25} : {info}")