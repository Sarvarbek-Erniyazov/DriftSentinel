"""
DriftSentinel — one command that rebuilds the artifact chain from raw data.

WHY THIS EXISTS, AND WHAT IT CORRECTS
    The plan's global acceptance criterion read:

        "`python src/pipelines/pipeline.py` reproduces everything from raw data"

    That was **false**, in two separate ways, and a reviewer tests this claim
    first:

    1. Seven modules held HARDCODED ABSOLUTE PATHS to one developer's machine
       (`C:\\Users\\...`), including the raw-data directory and the artifact
       directory. On any other machine the pipeline read from and wrote to
       somewhere that does not exist; on Linux CI the same literal resolves to a
       relative folder whose name contains backslashes, so the run appears to
       succeed while the artifacts land nowhere useful. It worked on exactly one
       machine, which is why nothing caught it. Fixed in Tier 2C.6 — every path
       is now derived from the module's own location.

    2. `pipeline.py` is the DATA pipeline. It never trained a model, never fitted
       a calibrator, and never built a conformal predictor, because those are
       separate stages by design. So even with the paths fixed, the sentence was
       overstated.

    Rather than overstate it differently, this module makes the honest claim
    executable: **`python src/pipelines/reproduce.py` rebuilds the core artifact
    chain from raw data**, and the docstring below states exactly what is inside
    that chain and what is not.

WHAT IS IN THE CHAIN (rebuilt by this module, in order)
    1. pipeline.py    raw CSV -> splits, encoders, FE stats, selector, parquets
    2. trainer.py     -> lgbm_v1, logreg_v1, scaler, training_summary
    3. evaluator.py   -> evaluation_report.json, figures 20-24
    4. calibration.py -> isotonic + temperature calibrators
    5. quantifier.py  -> conformal predictor and coverage report
    6. registry.py    -> model registry, lgbm_v2, DeLong/bootstrap comparison

WHAT IS NOT IN THE CHAIN, AND WHY
    The analysis and investigation modules (Tier 0 regime studies, repeated
    evaluation over 20 seeds, drift detection, FDR correction, adversarial and
    data-quality robustness, fairness audit, triage, adaptive conformal) CONSUME
    this chain and take substantially longer than it does — the Tier 0 regime
    sweep alone refits models across 20 seeds and four regimes. They are listed
    in `ANALYSIS_MODULES` with their entry points and are run separately.

    This is a scoped claim on purpose. "One command rebuilds the core artifacts,
    and here is the named list of what runs on top of it" is checkable;
    "one command reproduces everything" was not, and was false.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("reproduce")

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# The core chain, in dependency order. Each entry: (script, what it produces).
CORE_CHAIN = [
    ("src/pipelines/pipeline.py",
     "splits, label encoders, FE stats, fitted selector, *_fs.parquet, production windows"),
    ("src/models/trainer.py",
     "lgbm_v1.pkl, logreg_v1.pkl, logreg_scaler.pkl, training_summary.json"),
    ("src/models/evaluator.py",
     "evaluation_report.json and the operating-point figures"),
    ("src/uncertainty/calibration.py",
     "isotonic and temperature calibrators, calibration report"),
    ("src/uncertainty/quantifier.py",
     "conformal predictor and coverage report"),
    ("src/models/registry.py",
     "model_registry.json, registry_history.csv, lgbm_v2.pkl, model_comparison.json"),
]

# Consume the chain above; run separately. Named so the exclusion is visible.
ANALYSIS_MODULES = {
    "src/investigation/temporal_validity.py": "Tier 0.1 — split-validity anchors (slow)",
    "src/investigation/split_regimes.py": "Tier 0.2-0.4 — regime matrix, 20 seeds x 4 regimes (very slow)",
    "src/models/repeated_eval.py": "Tier 2A.2 — headline metrics with intervals (slow)",
    "src/uncertainty/decontamination.py": "Tier 2A.4 — decontaminated threshold and conformal",
    "src/drift/data_drift.py": "feature-level drift tests",
    "src/drift/feature_drift.py": "SHAP-weighted drift impact",
    "src/drift/concept_drift.py": "CUSUM / Page-Hinkley / sliding-window AUC",
    "src/drift/multivariate_drift.py": "Tier 2B.2 — C2ST, MMD, BBSD",
    "src/drift/fdr_correction.py": "Tier 2A.3 — Benjamini-Hochberg across the test family",
    "src/drift/alerting.py": "alert engine",
    "src/uncertainty/adaptive_conformal.py": "Tier 2B.1 — ACI vs static split conformal",
    "src/uncertainty/threshold_policy.py": "Tier 1.3 — PPR, budgets, cost sweep, DCA",
    "src/uncertainty/triage_policy.py": "Tier 2B.3 — uncertainty-gated triage",
    "src/features/selection_ablation.py": "Tier 2A.5 — selection ablation with in-fold refit",
    "src/adversarial/data_quality_robustness.py": "Tier 2B.4 — EHR data-quality robustness",
    "src/adversarial/defense.py": "Tier 1.2 — defense evaluation",
    "src/monitoring/fairness_audit.py": "Tier 2C.1 — subgroup performance",
    "src/monitoring/tracking_audit.py": "Tier 2C.5 — traceability audit",
    "src/monitoring/determinism.py": "Tier 2C.4 — byte-reproducibility (runs the chain twice)",
    "src/monitoring/health_check.py": "system health check over all of the above",
}


def run_stage(script: str) -> dict:
    # Tier 3: force UTF-8 on the child's stdio. Windows defaults a piped
    # stdout to cp1252, so any module whose __main__ prints a non-ASCII
    # character (evaluator.py prints "Δ") raises UnicodeEncodeError and the
    # stage exits non-zero AFTER having written its artifact correctly. The
    # failure is invisible when run in a console and invisible on Linux CI; it
    # appears only when a Windows parent captures the pipe. Same shape as the
    # hardcoded-path defect: environment-dependent, and silent where it was
    # written. Fixed at the runner so it cannot recur module by module.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, script], cwd=ROOT, env=env,
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    secs = round(time.perf_counter() - t0, 2)
    ok = proc.returncode == 0
    logger.info(f"  {'OK  ' if ok else 'FAIL'}  {script}  ({secs}s)")
    if not ok:
        logger.error(proc.stderr[-3000:])
    return {"script": script, "returncode": proc.returncode, "seconds": secs,
            "stderr_tail": "" if ok else proc.stderr[-3000:]}


def run_reproduce(stop_on_failure: bool = True) -> dict:
    logger.info("=" * 78)
    logger.info("DriftSentinel — rebuild the core artifact chain from raw data")
    logger.info("=" * 78)

    raw = ROOT / "data" / "raw" / "diabetes_hospital" / "diabetic_data.csv"
    if not raw.exists():
        raise FileNotFoundError(
            f"raw data not found at {raw}. The chain starts from the raw CSV; "
            "it does not fall back to the committed parquet files, because "
            "starting from them would not test reproduction at all.")
    logger.info(f"Raw input: {raw.relative_to(ROOT).as_posix()}")

    results, t0 = [], time.perf_counter()
    for script, produces in CORE_CHAIN:
        logger.info(f"> {script}")
        logger.info(f"    produces: {produces}")
        r = run_stage(script)
        r["produces"] = produces
        results.append(r)
        if r["returncode"] != 0 and stop_on_failure:
            logger.error("Stopping: a later stage consumes this one's output, so "
                         "continuing would produce artifacts built on a failure.")
            break

    failed = [r for r in results if r["returncode"] != 0]
    report = {
        "phase": "2C.6",
        "title": "Core artifact chain rebuilt from raw data",
        "scope_claim": (
            "This rebuilds the CORE chain: data preparation, both models, "
            "calibration, conformal prediction and the registry. It does NOT run "
            "the analysis and investigation modules, which consume these "
            "artifacts and are listed in `analysis_modules_not_run_here`. The "
            "earlier claim that pipeline.py alone reproduced everything was "
            "false on both counts: pipeline.py is the data pipeline only, and "
            "seven modules held hardcoded absolute paths to one machine."),
        "core_chain": results,
        "n_stages": len(CORE_CHAIN),
        "n_completed": len(results) - len(failed),
        "n_failed": len(failed),
        "verdict": "PASS" if not failed and len(results) == len(CORE_CHAIN) else "FAIL",
        "total_seconds": round(time.perf_counter() - t0, 2),
        "analysis_modules_not_run_here": ANALYSIS_MODULES,
        "reproducibility": {"python": platform.python_version(),
                            "platform": platform.platform()},
    }
    out = REPORTS_DIR / "reproduce_run.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("-" * 62)
    logger.info(f"{report['verdict']} — {report['n_completed']}/{report['n_stages']} "
                f"stages in {report['total_seconds']}s")
    logger.info(f"Report: {out.name}")
    logger.info("=" * 78)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild the DriftSentinel core artifact chain from raw data")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a failing stage (artifacts built on a "
                         "failed stage are not trustworthy; for debugging only)")
    ap.add_argument("--list", action="store_true",
                    help="print the chain and the excluded analysis modules, run nothing")
    args = ap.parse_args()

    if args.list:
        print("Core chain (rebuilt by this command):")
        for s, p in CORE_CHAIN:
            print(f"  {s}\n      {p}")
        print("\nAnalysis modules (consume the chain; run separately):")
        for s, p in ANALYSIS_MODULES.items():
            print(f"  {s}\n      {p}")
        return 0

    r = run_reproduce(stop_on_failure=not args.keep_going)
    print(f"\n{r['verdict']} — {r['n_completed']}/{r['n_stages']} stages, "
          f"{r['total_seconds']}s")
    for s in r["core_chain"]:
        print(f"  {'OK  ' if s['returncode'] == 0 else 'FAIL'}  "
              f"{s['script']:<38} {s['seconds']:>7.2f}s")
    return 0 if r["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
