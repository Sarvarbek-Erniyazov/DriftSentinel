"""
DriftSentinel — Model Trainer
Trains baseline models on processed train split.
Purpose: establish reference model for drift detection demonstration.
Models: LightGBM (primary), Logistic Regression (baseline).
Fitted on train_fs.parquet only — val/test never seen during training.
All artifacts saved for drift monitoring pipeline.
"""

import pandas as pd
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("trainer")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
TRAIN_DIR   = ROOT / "data" / "train"
MODELS_DIR  = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
TARGET        = "readmitted_binary"
RANDOM_SEED   = 42
CV_FOLDS      = 5

LGBM_PARAMS = {
    "objective"       : "binary",
    "metric"          : "auc",
    "learning_rate"   : 0.05,
    "num_leaves"      : 63,
    "max_depth"       : -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq"    : 5,
    "reg_alpha"       : 0.1,
    "reg_lambda"      : 0.1,
    "n_estimators"    : 500,
    "random_state"    : RANDOM_SEED,
    "n_jobs"          : -1,
    "verbose"         : -1,
}

LOGREG_PARAMS = {
    "C"           : 0.1,
    "max_iter"    : 1000,
    "random_state": RANDOM_SEED,
    "n_jobs"      : -1,
    "solver"      : "lbfgs",
}


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f"Loading splits from {TRAIN_DIR}")
    train = pd.read_parquet(TRAIN_DIR / "train_fs.parquet")
    val   = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test  = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")
    logger.info(f"Train: {train.shape}  Val: {val.shape}  Test: {test.shape}")
    return train, val, test


def _get_features(df: pd.DataFrame) -> list[str]:
    exclude = {"readmitted_binary", "readmitted_multi"}
    return [c for c in df.columns if c not in exclude]


def _cross_validate(model, X: np.ndarray, y: np.ndarray, name: str) -> dict:
    logger.info(f"Cross-validation: {name} ({CV_FOLDS}-fold stratified)")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    f1_scores  = cross_val_score(model, X, y, cv=cv, scoring="f1",      n_jobs=-1)

    results = {
        "cv_auc_mean" : round(float(auc_scores.mean()), 4),
        "cv_auc_std"  : round(float(auc_scores.std()),  4),
        "cv_f1_mean"  : round(float(f1_scores.mean()),  4),
        "cv_f1_std"   : round(float(f1_scores.std()),   4),
        "cv_auc_folds": [round(float(s), 4) for s in auc_scores],
    }

    logger.info(f"  AUC: {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
    logger.info(f"  F1 : {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")
    return results


def train_lgbm(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    feat_cols: list[str],
) -> dict:
    """
    Train LightGBM with early stopping on val split.
    Val split used ONLY for early stopping — not for selection.
    """
    logger.info("=" * 60)
    logger.info("Training LightGBM")
    logger.info("=" * 60)

    X_tr  = train[feat_cols].values
    y_tr  = train[TARGET].values
    X_val = val[feat_cols].values
    y_val = val[TARGET].values

    logger.info(f"  Train: {X_tr.shape}  pos_rate={y_tr.mean():.3f}")
    logger.info(f"  Val  : {X_val.shape}  pos_rate={y_val.mean():.3f}")
    logger.info(f"  Params: {LGBM_PARAMS}")

    # Cross-validation on train
    lgbm_cv = lgb.LGBMClassifier(**LGBM_PARAMS)
    cv_results = _cross_validate(lgbm_cv, X_tr, y_tr, "LightGBM")

    # Final fit with early stopping
    t0 = time.perf_counter()
    model = lgb.LGBMClassifier(**{**LGBM_PARAMS, "n_estimators": 1000})
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
    )
    elapsed = time.perf_counter() - t0

    best_iter = model.best_iteration_
    logger.info(f"  Best iteration    : {best_iter}")
    logger.info(f"  Training time     : {elapsed:.2f}s")

    # Val predictions
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred  = (val_proba >= 0.5).astype(int)

    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    val_metrics = {
        "val_auc"      : round(roc_auc_score(y_val, val_proba), 4),
        "val_f1"       : round(f1_score(y_val, val_pred),       4),
        "val_precision": round(precision_score(y_val, val_pred), 4),
        "val_recall"   : round(recall_score(y_val, val_pred),    4),
    }
    logger.info(f"  Val AUC       : {val_metrics['val_auc']}")
    logger.info(f"  Val F1        : {val_metrics['val_f1']}")
    logger.info(f"  Val Precision : {val_metrics['val_precision']}")
    logger.info(f"  Val Recall    : {val_metrics['val_recall']}")

    # Feature importance (gain)
    importance = pd.DataFrame({
        "feature"    : feat_cols,
        "importance" : model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("  Top 10 features (gain):")
    for _, row in importance.head(10).iterrows():
        logger.info(f"    {row['feature']:<45} {row['importance']:.0f}")

    # Save
    model_path = MODELS_DIR / "lgbm_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"  Saved: {model_path}")

    imp_path = MODELS_DIR / "lgbm_feature_importance.csv"
    importance.to_csv(imp_path, index=False)
    logger.info(f"  Saved: {imp_path}")

    return {
        "model_name"   : "lgbm_v1",
        "model_path"   : str(model_path),
        "best_iteration": best_iter,
        "train_time_s" : round(elapsed, 2),
        "params"       : LGBM_PARAMS,
        "cv"           : cv_results,
        "val_metrics"  : val_metrics,
        "n_features"   : len(feat_cols),
        "feature_cols" : feat_cols,
    }, model


def train_logreg(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    feat_cols: list[str],
) -> dict:
    """
    Train Logistic Regression baseline.
    StandardScaler fitted on train only.
    Probability calibration via isotonic regression.
    """
    logger.info("=" * 60)
    logger.info("Training Logistic Regression (baseline)")
    logger.info("=" * 60)

    X_tr  = train[feat_cols].values
    y_tr  = train[TARGET].values
    X_val = val[feat_cols].values
    y_val = val[TARGET].values

    # Scale — fit on train only
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)

    logger.info(f"  Train: {X_tr_sc.shape}  pos_rate={y_tr.mean():.3f}")
    logger.info(f"  Scaler fitted on train — mean range: [{X_tr_sc.mean(axis=0).min():.3f}, {X_tr_sc.mean(axis=0).max():.3f}]")
    logger.info(f"  Params: {LOGREG_PARAMS}")

    # Cross-validation
    lr_cv = LogisticRegression(**LOGREG_PARAMS)
    cv_results = _cross_validate(lr_cv, X_tr_sc, y_tr, "LogisticRegression")

    # Final fit with probability calibration
    t0 = time.perf_counter()
    base_lr  = LogisticRegression(**LOGREG_PARAMS)
    model    = CalibratedClassifierCV(base_lr, method="isotonic", cv=3)
    model.fit(X_tr_sc, y_tr)
    elapsed  = time.perf_counter() - t0

    logger.info(f"  Training time: {elapsed:.2f}s")

    val_proba = model.predict_proba(X_val_sc)[:, 1]
    val_pred  = (val_proba >= 0.5).astype(int)

    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    val_metrics = {
        "val_auc"      : round(roc_auc_score(y_val, val_proba), 4),
        "val_f1"       : round(f1_score(y_val, val_pred),       4),
        "val_precision": round(precision_score(y_val, val_pred), 4),
        "val_recall"   : round(recall_score(y_val, val_pred),    4),
    }
    logger.info(f"  Val AUC       : {val_metrics['val_auc']}")
    logger.info(f"  Val F1        : {val_metrics['val_f1']}")
    logger.info(f"  Val Precision : {val_metrics['val_precision']}")
    logger.info(f"  Val Recall    : {val_metrics['val_recall']}")

    # Coefficient importance
    base_model = base_lr
    base_model.fit(X_tr_sc, y_tr)
    coef_df = pd.DataFrame({
        "feature"    : feat_cols,
        "coefficient": np.abs(base_model.coef_[0]),
    }).sort_values("coefficient", ascending=False)

    logger.info("  Top 10 features (|coefficient|):")
    for _, row in coef_df.head(10).iterrows():
        logger.info(f"    {row['feature']:<45} {row['coefficient']:.4f}")

    # Save model + scaler together
    model_path = MODELS_DIR / "logreg_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    logger.info(f"  Saved: {model_path}")

    scaler_path = MODELS_DIR / "logreg_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    coef_path = MODELS_DIR / "logreg_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    logger.info(f"  Saved: {coef_path}")

    return {
        "model_name"  : "logreg_v1",
        "model_path"  : str(model_path),
        "train_time_s": round(elapsed, 2),
        "params"      : LOGREG_PARAMS,
        "cv"          : cv_results,
        "val_metrics" : val_metrics,
        "n_features"  : len(feat_cols),
        "feature_cols": feat_cols,
    }, model, scaler


def run_training() -> dict:
    """
    Full training run: LightGBM + Logistic Regression.
    Returns training summary dict.
    """
    logger.info("=" * 70)
    logger.info("DriftSentinel — Model Trainer")
    logger.info("=" * 70)

    train, val, test = _load_splits()
    feat_cols        = _get_features(train)

    logger.info(f"Features      : {len(feat_cols)}")
    logger.info(f"Target        : {TARGET}")
    logger.info(f"Train pos rate: {train[TARGET].mean():.4f}")
    logger.info(f"Val   pos rate: {val[TARGET].mean():.4f}")
    logger.info(f"Test  pos rate: {test[TARGET].mean():.4f}")

    # ── Train LightGBM ─────────────────────────────────────────────────────
    lgbm_meta, lgbm_model = train_lgbm(train, val, feat_cols)

    # ── Train Logistic Regression ──────────────────────────────────────────
    lr_meta, lr_model, lr_scaler = train_logreg(train, val, feat_cols)

    # ── Model comparison ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Model Comparison (Val Split)")
    logger.info("=" * 60)
    logger.info(f"{'Model':<20} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    logger.info("-" * 60)
    for meta in [lgbm_meta, lr_meta]:
        vm = meta["val_metrics"]
        logger.info(
            f"{meta['model_name']:<20} "
            f"{vm['val_auc']:>8.4f} "
            f"{vm['val_f1']:>8.4f} "
            f"{vm['val_precision']:>8.4f} "
            f"{vm['val_recall']:>8.4f}"
        )

    # ── Drift baseline: reference predictions on val ───────────────────────
    logger.info("=" * 60)
    logger.info("Saving reference predictions (drift monitoring baseline)")
    logger.info("=" * 60)

    X_val    = val[feat_cols].values
    X_val_sc = lr_scaler.transform(X_val)

    val_ref = val[[TARGET]].copy()
    val_ref["lgbm_proba"]   = lgbm_model.predict_proba(X_val)[:, 1]
    val_ref["logreg_proba"] = lr_model.predict_proba(X_val_sc)[:, 1]
    val_ref["lgbm_pred"]    = (val_ref["lgbm_proba"] >= 0.5).astype(int)
    val_ref["logreg_pred"]  = (val_ref["logreg_proba"] >= 0.5).astype(int)

    ref_path = MODELS_DIR / "val_reference_predictions.parquet"
    val_ref.to_parquet(ref_path, index=False)
    logger.info(f"  val_reference_predictions.parquet saved -> {ref_path}")
    logger.info(f"  LGBM   mean proba on val: {val_ref['lgbm_proba'].mean():.4f}")
    logger.info(f"  LogReg mean proba on val: {val_ref['logreg_proba'].mean():.4f}")

    # ── Save training summary ──────────────────────────────────────────────
    summary = {
        "lgbm"  : lgbm_meta,
        "logreg": lr_meta,
        "primary_model"    : "lgbm_v1",
        "baseline_model"   : "logreg_v1",
        "target"           : TARGET,
        "n_features"       : len(feat_cols),
        "feature_cols"     : feat_cols,
        "reference_preds"  : str(ref_path),
    }

    summary_path = MODELS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  training_summary.json saved -> {summary_path}")

    logger.info("=" * 70)
    logger.info("Trainer complete")
    logger.info(f"  LGBM   val AUC : {lgbm_meta['val_metrics']['val_auc']}")
    logger.info(f"  LogReg val AUC : {lr_meta['val_metrics']['val_auc']}")
    logger.info(f"  Primary model  : {summary['primary_model']}")
    logger.info("  Next: evaluator.py -> drift detection pipeline")
    logger.info("=" * 70)

    return summary


if __name__ == "__main__":
    summary = run_training()
    print(f"\nLGBM   val AUC : {summary['lgbm']['val_metrics']['val_auc']}")
    print(f"LogReg val AUC : {summary['logreg']['val_metrics']['val_auc']}")
    print(f"Models saved to: {MODELS_DIR}")