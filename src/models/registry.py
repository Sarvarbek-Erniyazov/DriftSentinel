"""
DriftSentinel — Model Registry
Tracks model versions, training metadata, and performance across splits.
Supports drift-triggered retraining comparison workflow.

Registry operations:
    register()  — add new model version with full metadata
    compare()   — compare two model versions side by side
    promote()   — set a model as active production model
    archive()   — mark a model as retired
    get_active()— return current production model metadata
    history()   — full version history with performance timeline
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
from src.monitoring.model_io import save_model
from src.models.metrics_guard import (
    assert_metrics_distinct, assert_split_disjoint, delong_roc_test,
    paired_bootstrap_auc, validate_metrics_schema,
)

logger = get_logger("registry")

ROOT        = Path(__file__).resolve().parents[2]
MODELS_DIR  = ROOT / "outputs" / "models"
REGISTRY_DIR= ROOT / "outputs" / "registry"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = REGISTRY_DIR / "model_registry.json"


# ══════════════════════════════════════════════════════════════════════════
# Registry core
# ══════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Persistent model version registry.
    Tracks performance metrics, drift status, and deployment history.
    """

    def __init__(self):
        self.registry: dict = self._load()

    # ──────────────────────────────────────────────────────────────────────
    def _load(self) -> dict:
        if REGISTRY_FILE.exists():
            with open(REGISTRY_FILE) as f:
                data = json.load(f)
            logger.info(f"Registry loaded: {len(data.get('models', {}))} models")
            return data
        return {
            "created_at"   : datetime.now().isoformat(),
            "active_model" : None,
            "models"       : {},
        }

    def _save(self):
        with open(REGISTRY_FILE, "w") as f:
            json.dump(self.registry, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────
    def register(
        self,
        model_name:    str,
        model_path:    str,
        train_metrics: dict,
        val_metrics:   dict   = None,
        test_metrics:  dict   = None,
        params:        dict   = None,
        feature_cols:  list   = None,
        n_features:    int    = None,
        train_rows:    int    = None,
        trigger:       str    = "manual",
        notes:         str    = "",
        train_splits:  tuple  = ("train",),
    ) -> dict:
        """
        Register a new model version.

        Parameters
        ----------
        model_name    : unique version name (e.g. 'lgbm_v1', 'lgbm_v2')
        model_path    : path to saved model pkl
        train_metrics : dict with auc, f1, precision, recall, brier
        val_metrics   : same keys, or None if the model was FITTED on val and no
                        held-out validation estimate exists
        test_metrics  : optional — set after deployment
        trigger       : 'manual' / 'drift_alert' / 'scheduled'
        train_splits  : which splits the model was FITTED on. Every metrics dict
                        for a split in here is in-sample and is refused.

        Raises
        ------
        MetricsIntegrityError
            if a metrics dict is filed under a split the model was fitted on, or
            if two split metric dicts are identical (R2). The original defect —
            `val_metrics = train_v2_metrics` — trips both.
        """
        logger.info("=" * 60)
        logger.info(f"Registering model: {model_name}")
        logger.info("=" * 60)

        # ── Integrity guards ──────────────────────────────────────────────
        train_splits = tuple(train_splits)
        supplied = {"train": train_metrics, "val": val_metrics, "test": test_metrics}
        validated: dict[str, dict] = {}
        for split, m in supplied.items():
            if m is None or m == {}:
                validated[split] = {}
                continue
            if split != "train":
                # 'train' metrics are in-sample BY DEFINITION and labelled as such;
                # any other split must be genuinely held out.
                assert_split_disjoint(train_splits, split)
            validated[split] = validate_metrics_schema(m, split)

        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        for s1, s2 in pairs:
            if validated[s1] and validated[s2]:
                assert_metrics_distinct(validated[s1], validated[s2],
                                        f"{model_name} {s1}", f"{model_name} {s2}")

        train_metrics = validated["train"]
        val_metrics = validated["val"]
        test_metrics = validated["test"]
        logger.info(f"  Integrity      : fitted on {list(train_splits)}; "
                    f"held-out metrics present for "
                    f"{[s for s in ('val', 'test') if validated[s]] or 'NONE'}")

        entry = {
            "train_splits"  : list(train_splits),
            "model_name"    : model_name,
            "model_path"    : str(model_path),
            "registered_at" : datetime.now().isoformat(),
            "trigger"       : trigger,
            "status"        : "registered",
            "notes"         : notes,
            "params"        : params or {},
            "n_features"    : n_features,
            "train_rows"    : train_rows,
            "feature_cols"  : feature_cols or [],
            "metrics"       : {
                "train" : train_metrics,
                "val"   : val_metrics,
                "test"  : test_metrics or {},
            },
            "drift_status"  : "unknown",
            "promoted_at"   : None,
            "archived_at"   : None,
        }

        self.registry["models"][model_name] = entry
        self._save()

        logger.info(f"  Trigger        : {trigger}")
        logger.info(f"  Train AUC      : {train_metrics.get('auc', 'N/A')}  (in-sample)")
        logger.info(f"  Val   AUC      : {val_metrics.get('auc', 'UNAVAILABLE — fitted on val')}")
        logger.info(f"  N features     : {n_features}")
        logger.info(f"  Train rows     : {train_rows}")
        logger.info(f"  Status         : registered")
        logger.info(f"  Saved to       : {REGISTRY_FILE}")

        return entry

    # ──────────────────────────────────────────────────────────────────────
    def update_test_metrics(
        self,
        model_name:   str,
        test_metrics: dict,
        drift_status: str = "unknown",
    ):
        """Update test metrics after production evaluation."""
        if model_name not in self.registry["models"]:
            raise KeyError(f"Model {model_name} not found in registry")

        self.registry["models"][model_name]["metrics"]["test"] = test_metrics
        self.registry["models"][model_name]["drift_status"]    = drift_status
        self._save()

        logger.info(f"Updated {model_name}: test AUC={test_metrics.get('auc')} "
                    f"drift_status={drift_status}")

    # ──────────────────────────────────────────────────────────────────────
    def promote(self, model_name: str):
        """Set model as active production model."""
        if model_name not in self.registry["models"]:
            raise KeyError(f"Model {model_name} not found")

        prev_active = self.registry.get("active_model")
        if prev_active and prev_active in self.registry["models"]:
            self.registry["models"][prev_active]["status"] = "superseded"

        self.registry["models"][model_name]["status"]      = "active"
        self.registry["models"][model_name]["promoted_at"] = datetime.now().isoformat()
        self.registry["active_model"]                      = model_name
        self._save()

        logger.info(f"Promoted: {model_name} → active production model")
        if prev_active:
            logger.info(f"Previous: {prev_active} → superseded")

    # ──────────────────────────────────────────────────────────────────────
    def archive(self, model_name: str, reason: str = ""):
        """Mark model as archived."""
        if model_name not in self.registry["models"]:
            raise KeyError(f"Model {model_name} not found")

        self.registry["models"][model_name]["status"]      = "archived"
        self.registry["models"][model_name]["archived_at"] = datetime.now().isoformat()
        self.registry["models"][model_name]["notes"]      += f" | Archived: {reason}"
        self._save()

        logger.info(f"Archived: {model_name}  reason={reason}")

    # ──────────────────────────────────────────────────────────────────────
    def get_active(self) -> dict | None:
        """Return active model metadata."""
        active = self.registry.get("active_model")
        if not active:
            return None
        return self.registry["models"].get(active)

    # ──────────────────────────────────────────────────────────────────────
    def compare(
        self,
        model_a: str,
        model_b: str,
        auc_test: dict = None,
    ) -> dict:
        """
        Side-by-side comparison of two model versions on COMPARABLE splits only.

        A split is comparable only if it is held out for BOTH models. lgbm_v2 was
        fitted on train+val, so 'val' is in-sample for it and is excluded — the
        original code compared v1's held-out val AUC (0.6865) against v2's
        in-sample val AUC (0.7987), declared v2 the winner by +0.1122, and used
        that to drive PROMOTE.

        `auc_test` carries the DeLong / bootstrap result for the one comparable
        split, so the verdict rests on a significance test rather than on the
        sign of a difference.
        """
        logger.info("=" * 60)
        logger.info(f"Model Comparison: {model_a}  vs  {model_b}")
        logger.info("=" * 60)

        if model_a not in self.registry["models"]:
            raise KeyError(f"{model_a} not in registry")
        if model_b not in self.registry["models"]:
            raise KeyError(f"{model_b} not in registry")

        a = self.registry["models"][model_a]
        b = self.registry["models"][model_b]

        metrics   = ["auc", "f1", "precision", "recall", "brier"]
        fitted_a  = set(a.get("train_splits", ["train"]))
        fitted_b  = set(b.get("train_splits", ["train"]))
        splits    = [s for s in ("val", "test") if s not in fitted_a | fitted_b]
        excluded  = [s for s in ("train", "val", "test") if s not in splits]
        result    = {"comparable_splits": splits, "excluded_splits": {}}

        for s in excluded:
            why = ("in-sample for both models" if s == "train" else
                   f"fitted on by {model_a if s in fitted_a else ''}"
                   f"{' and ' if s in fitted_a and s in fitted_b else ''}"
                   f"{model_b if s in fitted_b else ''}".strip())
            result["excluded_splits"][s] = why
            logger.warning(f"  EXCLUDED split {s!r}: {why} — not a valid comparison")

        if not splits:
            raise ValueError(
                f"no comparable split: {model_a} fitted on {sorted(fitted_a)}, "
                f"{model_b} fitted on {sorted(fitted_b)}. A fourth held-out split "
                f"is required to compare these models honestly.")

        logger.info(f"{'Metric':<15} {'Split':<8} {model_a:>12} {model_b:>12} {'Delta':>10} {'Winner':>8}")
        logger.info("-" * 65)

        for split in splits:
            for metric in metrics:
                val_a = a["metrics"].get(split, {}).get(metric)
                val_b = b["metrics"].get(split, {}).get(metric)

                if val_a is None or val_b is None:
                    continue

                delta = val_b - val_a
                # For brier: lower is better
                if metric == "brier":
                    winner = model_b if delta < 0 else model_a
                else:
                    winner = model_b if delta > 0 else model_a

                sign = "↑" if delta > 0 else "↓"
                logger.info(
                    f"  {metric:<13} {split:<8} "
                    f"{val_a:>12.4f} "
                    f"{val_b:>12.4f} "
                    f"{delta:>+9.4f}{sign} "
                    f"{winner:>8}"
                )
                result[f"{split}_{metric}"] = {
                    model_a: val_a,
                    model_b: val_b,
                    "delta" : round(delta, 4),
                    "winner": winner,
                }

        # ── Verdict: significance on the comparable split, not the sign ────
        test_auc_a = a["metrics"].get("test", {}).get("auc")
        test_auc_b = b["metrics"].get("test", {}).get("auc")
        delta = (test_auc_b - test_auc_a) if (test_auc_a and test_auc_b) else None

        significant = None
        if auc_test:
            significant = bool(auc_test["delong"]["significant_at_0.05"]
                               and auc_test["bootstrap"]["excludes_zero"])

        if significant is None:
            verdict = "INSUFFICIENT_EVIDENCE"
        elif significant and delta and delta > 0:
            verdict = "PROMOTE"
        elif significant and delta and delta < 0:
            verdict = "ROLLBACK"
        else:
            verdict = "NO_SIGNIFICANT_DIFFERENCE"

        logger.info("-" * 65)
        logger.info(f"  Comparable splits     : {splits}")
        logger.info(f"  Test AUC              : {test_auc_a} -> {test_auc_b} "
                    f"({delta:+.4f})" if delta is not None else "  Test AUC : N/A")
        if auc_test:
            d, bs = auc_test["delong"], auc_test["bootstrap"]
            logger.info(f"  DeLong                : z={d['z']:+.3f} p={d['p_value']:.4f} "
                        f"CI95 delta {d['ci95_delta'][0]:+.4f}..{d['ci95_delta'][1]:+.4f}")
            logger.info(f"  Paired bootstrap      : delta {bs['delta_auc_2_minus_1']:+.4f} "
                        f"CI95 {bs['ci95'][0]:+.4f}..{bs['ci95'][1]:+.4f} "
                        f"({bs['resampling_unit']} resampling)")
        logger.info(f"  VERDICT               : {verdict}")
        if verdict == "NO_SIGNIFICANT_DIFFERENCE":
            logger.info("  → The difference is within noise. Retraining did not "
                        "demonstrably improve the model on held-out data.")

        result["verdict"] = verdict
        result["test_auc_delta"] = round(delta, 4) if delta is not None else None
        result["val_auc_delta"] = None
        result["val_comparison_available"] = "val" in splits
        result["significance"] = auc_test
        return result

    # ──────────────────────────────────────────────────────────────────────
    def history(self) -> pd.DataFrame:
        """Return full version history as DataFrame."""
        rows = []
        for name, m in self.registry["models"].items():
            rows.append({
                "model_name"    : name,
                "status"        : m["status"],
                "trigger"       : m["trigger"],
                "registered_at" : m["registered_at"],
                "train_auc"     : m["metrics"].get("train", {}).get("auc"),
                "val_auc"       : m["metrics"].get("val",   {}).get("auc"),
                "test_auc"      : m["metrics"].get("test",  {}).get("auc"),
                "train_f1"      : m["metrics"].get("train", {}).get("f1"),
                "val_f1"        : m["metrics"].get("val",   {}).get("f1"),
                "test_f1"       : m["metrics"].get("test",  {}).get("f1"),
                "drift_status"  : m["drift_status"],
                "n_features"    : m["n_features"],
                "train_rows"    : m["train_rows"],
                "notes"         : m["notes"],
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("registered_at").reset_index(drop=True)
        return df

    # ──────────────────────────────────────────────────────────────────────
    def print_summary(self):
        """Print registry summary to logger."""
        logger.info("=" * 60)
        logger.info("Model Registry Summary")
        logger.info("=" * 60)
        logger.info(f"  Active model : {self.registry.get('active_model', 'None')}")
        logger.info(f"  Total models : {len(self.registry['models'])}")
        logger.info("-" * 60)
        for name, m in self.registry["models"].items():
            val_auc  = m["metrics"].get("val",  {}).get("auc",  "N/A")
            test_auc = m["metrics"].get("test", {}).get("auc",  "N/A")
            logger.info(
                f"  {name:<15} status={m['status']:<12} "
                f"val_auc={val_auc}  test_auc={test_auc}  "
                f"drift={m['drift_status']}"
            )
        logger.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
# Entry point — register v1, retrain v2, compare
# ══════════════════════════════════════════════════════════════════════════

def _recover_test_patient_ids(test_df, y_test):
    """
    Recover `patient_nbr` for the test rows so the bootstrap can cluster on it.

    The feature-selected parquet does not carry patient ids, but the splitter
    sorts each split by encounter_id, so the raw test-patient rows in that order
    should align 1:1. That is an ASSUMPTION, so it is verified against the label
    sequence before use: if the recovered labels do not match the parquet's
    target exactly, the ids are discarded and the bootstrap falls back to
    row-level resampling with an explicit caveat rather than silently claiming
    cluster-robustness it does not have.
    """
    try:
        import pandas as _pd
        with open(ROOT / "outputs" / "artifacts" / "split_index.json") as f:
            idx = json.load(f)
        raw = _pd.read_csv(ROOT / "data" / "raw" / "diabetes_hospital" / "diabetic_data.csv",
                           usecols=["encounter_id", "patient_nbr", "readmitted"],
                           na_values=["?"], keep_default_na=False)
        sub = (raw[raw["patient_nbr"].isin(set(idx["test_patient_ids"]))]
               .sort_values("encounter_id"))
        if len(sub) != len(y_test):
            return None, f"FAILED: row count {len(sub)} != {len(y_test)}"
        # Derive the label from the SAME map the pipeline uses. Hardcoding
        # `!= "NO"` here was the merged-target definition; after the Tier 2A.1
        # switch it no longer matched the parquet and the verification correctly
        # refused, falling back to row-level resampling. Reading the map keeps
        # this honest across any future target change.
        from src.data.preprocessor import TARGET_BINARY_MAP
        recovered = sub["readmitted"].map(TARGET_BINARY_MAP).to_numpy().astype(int)
        if not np.array_equal(recovered, np.asarray(y_test).astype(int)):
            return None, "FAILED: recovered label sequence does not match the parquet target"
        return sub["patient_nbr"].to_numpy(), (
            f"OK: verified against the label sequence "
            f"({sub['patient_nbr'].nunique():,} patients / {len(sub):,} rows)")
    except Exception as e:                                   # pragma: no cover
        return None, f"FAILED: {type(e).__name__}: {e}"


def run_registry() -> dict:
    """
    Full registry workflow:
        1. Register lgbm_v1 with known metrics
        2. Simulate drift-triggered retraining → lgbm_v2
        3. Compare v1 vs v2
        4. Promote winner
    """
    logger.info("=" * 70)
    logger.info("DriftSentinel — Model Registry Run")
    logger.info("=" * 70)

    registry = ModelRegistry()

    # ── Step 1: Register lgbm_v1 ──────────────────────────────────────────
    logger.info("Step 1: Registering lgbm_v1 (initial deployment)")

    eval_path = ROOT / "outputs" / "models" / "evaluation_report.json"
    with open(eval_path) as f:
        eval_report = json.load(f)

    lgbm_eval   = eval_report["lgbm"]
    feat_path   = ROOT / "outputs" / "artifacts" / "selected_features.json"
    with open(feat_path) as f:
        feat_cols = json.load(f)

    # Tier 2C.5 correction. `params` and the note previously HARDCODED
    # `n_estimators: 173` — the tree count from before the Tier 2A.1 target
    # switch. The deployed model actually stopped at a different iteration, so
    # the registry's record of the active production model was wrong about the
    # model it describes. A registry whose provenance is typed rather than read
    # is a registry that can disagree with its own artifact, which is the one
    # thing a registry exists to prevent. Read from the training run instead.
    with open(ROOT / "outputs" / "models" / "training_summary.json") as f:
        train_summary = json.load(f)["lgbm"]
    n_trees = train_summary["best_iteration"]
    lgbm_params = {k: train_summary["params"][k]
                   for k in ("learning_rate", "num_leaves", "min_child_samples",
                             "feature_fraction", "bagging_fraction", "reg_alpha",
                             "reg_lambda", "random_state")}
    lgbm_params["n_estimators"] = n_trees
    lgbm_params["n_estimators_source"] = "best_iteration from early stopping"

    registry.register(
        model_name    = "lgbm_v1",
        model_path    = str(MODELS_DIR / "lgbm_v1.pkl"),
        train_metrics = lgbm_eval["metrics_by_split"]["train"],
        val_metrics   = lgbm_eval["metrics_by_split"]["val"],
        test_metrics  = lgbm_eval["metrics_by_split"]["test"],
        params        = lgbm_params,
        feature_cols  = feat_cols,
        n_features    = len(feat_cols),
        train_rows    = lgbm_eval["metrics_by_split"]["train"]["n_samples"],
        trigger       = "manual",
        train_splits  = ("train",),
        notes         = ("Initial model. Trained on the train split = the "
                         "EARLIEST-ENTERING 60% of patients by first encounter_id "
                         f"(entry cohort). Early stopping at {n_trees} trees."),
    )

    registry.update_test_metrics(
        "lgbm_v1",
        test_metrics  = lgbm_eval["metrics_by_split"]["test"],
        drift_status  = "CRITICAL",
    )

    registry.promote("lgbm_v1")

    # ── Step 2: Drift-triggered retraining → lgbm_v2 ──────────────────────
    logger.info("\nStep 2: Drift alert triggered — retraining lgbm_v2")
    logger.info("  Strategy: retrain on train + val combined (more recent data)")

    import pandas as pd
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
    from sklearn.metrics import precision_score, recall_score

    train = pd.read_parquet(ROOT / "data" / "train" / "train_fs.parquet")
    val   = pd.read_parquet(ROOT / "data" / "train" / "val_fs.parquet")
    test  = pd.read_parquet(ROOT / "data" / "train" / "test_fs.parquet")

    target_cols = {"readmitted_binary", "readmitted_multi"}
    feat_cols   = [c for c in train.columns if c not in target_cols]

    # Combine train + val for lgbm_v2
    train_v2    = pd.concat([train, val], ignore_index=True)
    threshold   = lgbm_eval["threshold"]

    logger.info(f"  lgbm_v2 train rows: {len(train_v2):,}  "
                f"(train={len(train):,} + val={len(val):,})")

    X_v2   = train_v2[feat_cols]
    y_v2   = train_v2["readmitted_binary"].values
    X_test = test[feat_cols]
    y_test = test["readmitted_binary"].values

    lgbm_params = {
        "objective"        : "binary",
        "metric"           : "auc",
        "learning_rate"    : 0.05,
        "num_leaves"       : 63,
        "min_child_samples": 50,
        "feature_fraction" : 0.8,
        "bagging_fraction" : 0.8,
        "bagging_freq"     : 5,
        "reg_alpha"        : 0.1,
        "reg_lambda"       : 0.1,
        "n_estimators"     : 300,
        "random_state"     : 42,
        "n_jobs"           : -1,
        "verbose"          : -1,
    }

    model_v2 = lgb.LGBMClassifier(**lgbm_params)
    model_v2.fit(X_v2, y_v2)

    def _eval(y_true, y_proba, thr):
        y_pred = (y_proba >= thr).astype(int)
        return {
            "auc"      : round(float(roc_auc_score(y_true, y_proba)), 4),
            "f1"       : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall"   : round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "brier"    : round(float(brier_score_loss(y_true, y_proba)), 4),
        }

    # v2 metrics
    train_v2_proba = model_v2.predict_proba(X_v2)[:, 1]
    test_v2_proba  = model_v2.predict_proba(X_test)[:, 1]

    train_v2_metrics = _eval(y_v2,   train_v2_proba, threshold)
    test_v2_metrics  = _eval(y_test, test_v2_proba,  threshold)

    logger.info(f"  lgbm_v2 train AUC : {train_v2_metrics['auc']}")
    logger.info(f"  lgbm_v2 test  AUC : {test_v2_metrics['auc']}")

    # Save v2
    # Tier 2C.7: provenance sidecar, same reasoning as trainer.py --
    # serializer="pickle" keeps the format the eighteen consumers expect.
    v2_path = MODELS_DIR / "lgbm_v2.pkl"
    save_model(model_v2, v2_path, serializer="pickle",
               extra={"model_name": "lgbm_v2",
                      "trigger": "drift_alert",
                      "train_splits": ["train", "val"],
                      "n_estimators": lgbm_params["n_estimators"],
                      "early_stopping": False,
                      "train_rows": int(len(train_v2))})
    logger.info(f"  lgbm_v2 saved: {v2_path} (+ provenance sidecar)")

    registry.register(
        model_name    = "lgbm_v2",
        model_path    = str(v2_path),
        train_metrics = train_v2_metrics,
        # val_metrics is DELIBERATELY absent. lgbm_v2 was fitted on train+val, so
        # any "val" number is training performance. The original code passed
        # `val_metrics = train_v2_metrics` here, which produced the reported
        # "val AUC 0.7987" and the +0.1122 win over v1's honest held-out 0.6865.
        val_metrics   = None,
        test_metrics  = test_v2_metrics,
        params        = lgbm_params,
        feature_cols  = feat_cols,
        n_features    = len(feat_cols),
        train_rows    = len(train_v2),
        trigger       = "drift_alert",
        train_splits  = ("train", "val"),
        notes         = (
            "Retrained after drift alert on train+val combined. "
            "NO held-out validation estimate exists for this model. "
            f"UNCONTROLLED CONFOUND: v1 used early stopping ({n_trees} trees); "
            f"v2 uses a fixed {lgbm_params['n_estimators']} with no early "
            "stopping, because val had been consumed. The two models differ in "
            "training data AND capacity AND stopping rule, so the test-set "
            "difference is not attributable to the extra data alone."
        ),
    )

    # ── Step 3: Compare v1 vs v2 on the one comparable split ──────────────
    logger.info("\nStep 3: Comparing lgbm_v1 vs lgbm_v2 (test split only)")

    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model_v1 = pickle.load(f)
    test_v1_proba = model_v1.predict_proba(X_test)[:, 1]

    groups, cluster_note = _recover_test_patient_ids(test, y_test)
    auc_test = {
        "delong": delong_roc_test(y_test, test_v1_proba, test_v2_proba),
        "bootstrap": paired_bootstrap_auc(y_test, test_v1_proba, test_v2_proba,
                                          groups=groups, n_boot=2000, seed=42),
        "patient_id_recovery": cluster_note,
    }
    comparison = registry.compare("lgbm_v1", "lgbm_v2", auc_test=auc_test)

    # ── Step 4: Promote winner ─────────────────────────────────────────────
    logger.info(f"\nStep 4: Verdict = {comparison['verdict']}")

    if comparison["verdict"] == "PROMOTE":
        registry.archive("lgbm_v1", reason="Superseded by lgbm_v2 after drift retraining")
        registry.promote("lgbm_v2")
        logger.info("lgbm_v2 promoted to active production model")
    elif comparison["verdict"] == "ROLLBACK":
        logger.warning("lgbm_v2 is significantly WORSE — keeping lgbm_v1")
    elif comparison["verdict"] == "NO_SIGNIFICANT_DIFFERENCE":
        logger.warning("lgbm_v2 is not significantly different from lgbm_v1 on the "
                       "only comparable split. NOT promoting: a promotion needs "
                       "evidence, and 'the number went up' is not evidence.")
    else:
        logger.warning(f"verdict={comparison['verdict']} — not promoting")

    # ── Step 5: History ────────────────────────────────────────────────────
    logger.info("\nStep 5: Registry History")
    history = registry.history()
    logger.info("\n" + history.to_string(index=False))

    comparison_path = REGISTRY_DIR / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info(f"Comparison + significance saved: {comparison_path}")

    history_path = REGISTRY_DIR / "registry_history.csv"
    history.to_csv(history_path, index=False)
    logger.info(f"\nHistory saved: {history_path}")

    registry.print_summary()

    return {
        "v1_test_auc"   : lgbm_eval["metrics_by_split"]["test"]["auc"],
        "v2_test_auc"   : test_v2_metrics["auc"],
        "improvement"   : test_v2_metrics["auc"] - lgbm_eval["metrics_by_split"]["test"]["auc"],
        "verdict"       : comparison["verdict"],
        "active_model"  : registry.registry.get("active_model"),
    }


if __name__ == "__main__":
    result = run_registry()
    print(f"\n{'='*50}")
    print("REGISTRY RESULT")
    print(f"{'='*50}")
    print(f"  lgbm_v1 test AUC : {result['v1_test_auc']}")
    print(f"  lgbm_v2 test AUC : {result['v2_test_auc']}")
    print(f"  Improvement      : {result['improvement']:+.4f}")
    print(f"  Verdict          : {result['verdict']}")
    print(f"  Active model     : {result['active_model']}")