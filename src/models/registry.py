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
        val_metrics:   dict,
        test_metrics:  dict   = None,
        params:        dict   = None,
        feature_cols:  list   = None,
        n_features:    int    = None,
        train_rows:    int    = None,
        trigger:       str    = "manual",
        notes:         str    = "",
    ) -> dict:
        """
        Register a new model version.

        Parameters
        ----------
        model_name    : unique version name (e.g. 'lgbm_v1', 'lgbm_v2')
        model_path    : path to saved model pkl
        train_metrics : dict with auc, f1, precision, recall, brier
        val_metrics   : dict with same keys
        test_metrics  : optional — set after deployment
        trigger       : 'manual' / 'drift_alert' / 'scheduled'
        """
        logger.info("=" * 60)
        logger.info(f"Registering model: {model_name}")
        logger.info("=" * 60)

        entry = {
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
        logger.info(f"  Train AUC      : {train_metrics.get('auc', 'N/A')}")
        logger.info(f"  Val   AUC      : {val_metrics.get('auc', 'N/A')}")
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
    ) -> dict:
        """
        Side-by-side comparison of two model versions.
        Primary use: compare v1 (drifted) vs v2 (retrained).
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
        splits    = ["train", "val", "test"]
        result    = {}

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

        # Overall verdict
        val_auc_a = a["metrics"].get("val", {}).get("auc", 0)
        val_auc_b = b["metrics"].get("val", {}).get("auc", 0)
        test_auc_a = a["metrics"].get("test", {}).get("auc", 0)
        test_auc_b = b["metrics"].get("test", {}).get("auc", 0)

        improved_val  = val_auc_b  > val_auc_a
        improved_test = test_auc_b > test_auc_a

        verdict = (
            "PROMOTE"  if improved_val and improved_test else
            "REVIEW"   if improved_val or improved_test  else
            "ROLLBACK"
        )

        logger.info("-" * 65)
        logger.info(f"  Val  AUC improvement  : {val_auc_b - val_auc_a:+.4f}")
        logger.info(f"  Test AUC improvement  : {test_auc_b - test_auc_a:+.4f}")
        logger.info(f"  VERDICT               : {verdict}")
        logger.info(
            f"  {'→ Promote ' + model_b + ' to production' if verdict == 'PROMOTE' else '→ Review before promoting' if verdict == 'REVIEW' else '→ Keep ' + model_a}"
        )

        result["verdict"]       = verdict
        result["val_auc_delta"] = round(val_auc_b - val_auc_a,  4)
        result["test_auc_delta"]= round(test_auc_b - test_auc_a, 4)

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

    registry.register(
        model_name    = "lgbm_v1",
        model_path    = str(MODELS_DIR / "lgbm_v1.pkl"),
        train_metrics = lgbm_eval["metrics_by_split"]["train"],
        val_metrics   = lgbm_eval["metrics_by_split"]["val"],
        test_metrics  = lgbm_eval["metrics_by_split"]["test"],
        params        = {"n_estimators": 173, "learning_rate": 0.05,
                         "num_leaves": 63},
        feature_cols  = feat_cols,
        n_features    = len(feat_cols),
        train_rows    = 63492,
        trigger       = "manual",
        notes         = "Initial model. Trained on train split (2008-2009 patients).",
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
    v2_path = MODELS_DIR / "lgbm_v2.pkl"
    with open(v2_path, "wb") as f:
        pickle.dump(model_v2, f)
    logger.info(f"  lgbm_v2 saved: {v2_path}")

    registry.register(
        model_name    = "lgbm_v2",
        model_path    = str(v2_path),
        train_metrics = train_v2_metrics,
        val_metrics   = train_v2_metrics,
        test_metrics  = test_v2_metrics,
        params        = lgbm_params,
        feature_cols  = feat_cols,
        n_features    = len(feat_cols),
        train_rows    = len(train_v2),
        trigger       = "drift_alert",
        notes         = (
            "Retrained after CRITICAL drift alert. "
            "Train data: train+val combined (more recent patients). "
            f"AUC improvement on test: "
            f"{test_v2_metrics['auc'] - lgbm_eval['metrics_by_split']['test']['auc']:+.4f}"
        ),
    )

    # ── Step 3: Compare v1 vs v2 ───────────────────────────────────────────
    logger.info("\nStep 3: Comparing lgbm_v1 vs lgbm_v2")
    comparison = registry.compare("lgbm_v1", "lgbm_v2")

    # ── Step 4: Promote winner ─────────────────────────────────────────────
    logger.info(f"\nStep 4: Verdict = {comparison['verdict']}")

    if comparison["verdict"] == "PROMOTE":
        registry.archive("lgbm_v1", reason="Superseded by lgbm_v2 after drift retraining")
        registry.promote("lgbm_v2")
        logger.info("lgbm_v2 promoted to active production model")
    elif comparison["verdict"] == "REVIEW":
        logger.warning("lgbm_v2 needs manual review before promotion")
    else:
        logger.warning("lgbm_v2 did not improve — keeping lgbm_v1")

    # ── Step 5: History ────────────────────────────────────────────────────
    logger.info("\nStep 5: Registry History")
    history = registry.history()
    logger.info("\n" + history.to_string(index=False))

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