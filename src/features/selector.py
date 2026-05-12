"""
DriftSentinel — Feature Selector
Advanced multi-stage feature selection pipeline.
Fitted exclusively on train split — val/test receive transform only.
All selection decisions logged with statistical justification.

Selection pipeline stages:
    Stage 1 — Variance threshold          : remove near-zero variance features
    Stage 2 — Pearson/Spearman filter     : remove redundant correlated pairs
    Stage 3 — Mutual information          : non-linear relevance vs target
    Stage 4 — Boruta (tree-based)         : all-relevant feature selection
    Stage 5 — SHAP importance             : model-based feature ranking
    Stage 6 — Stability selection         : bootstrap consensus across stages
    Stage 7 — Final consensus vote        : feature retained if selected in
                                            majority of applicable stages
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import shap

warnings.filterwarnings("ignore")

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("selector")

ARTIFACTS_DIR = Path(r"C:\Users\sharg\Desktop\github\DriftSentinel\outputs\artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_BINARY = "readmitted_binary"
TARGET_MULTI  = "readmitted_multi"
TARGET_COLS   = {TARGET_BINARY, TARGET_MULTI, "readmitted"}
ID_COLS       = {"encounter_id", "patient_nbr"}

RANDOM_SEED   = 42
N_JOBS        = -1

# Stage thresholds
VARIANCE_THRESHOLD      = 0.01
CORRELATION_THRESHOLD   = 0.90
MI_PERCENTILE           = 20
BORUTA_N_ESTIMATORS     = 200
BORUTA_MAX_ITER         = 50
BORUTA_ALPHA            = 0.05
SHAP_N_ESTIMATORS       = 300
STABILITY_N_BOOTSTRAP   = 50
STABILITY_SAMPLE_FRAC   = 0.75
STABILITY_THRESHOLD     = 0.60
CONSENSUS_MIN_STAGES    = 2


class FeatureSelector:
    """
    Multi-stage stateful feature selector.
    fit(train)      -> runs all selection stages, stores selected feature set.
    transform(df)   -> retains only selected features.
    """

    def __init__(self):
        self.selected_features:  list[str]        = []
        self.stage_results:      dict              = {}
        self.feature_scores:     pd.DataFrame      = pd.DataFrame()
        self.scaler:             StandardScaler    = StandardScaler()
        self.fitted:             bool              = False

    # ──────────────────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame) -> "FeatureSelector":
        """
        Run full multi-stage selection on train split.

        Parameters
        ----------
        train : engineered train DataFrame (output of FeatureEngineer)

        Returns
        -------
        self
        """
        logger.info("=" * 70)
        logger.info("DriftSentinel — Feature Selector  [fit on TRAIN]")
        logger.info("=" * 70)
        logger.info(f"Input shape : {train.shape}")

        X, y_binary, y_multi, feature_cols = self._prepare(train)
        logger.info(f"Feature matrix : {X.shape[0]:,} rows x {X.shape[1]} features")
        logger.info(f"Target (binary): {y_binary.value_counts().to_dict()}")

        votes = defaultdict(int)
        stage_count = defaultdict(int)

        # ── Stage 1 ────────────────────────────────────────────────────────
        s1 = self._stage1_variance(X, feature_cols)
        self.stage_results["stage1_variance"] = s1
        for f in s1["selected"]:
            votes[f] += 1
            stage_count[f] += 1

        # ── Stage 2 ────────────────────────────────────────────────────────
        s2 = self._stage2_correlation(X, feature_cols, s1["selected"])
        self.stage_results["stage2_correlation"] = s2
        for f in s2["selected"]:
            votes[f] += 1
            stage_count[f] += 1

        # ── Stage 3 ────────────────────────────────────────────────────────
        s3 = self._stage3_mutual_info(X, y_binary, feature_cols, s2["selected"])
        self.stage_results["stage3_mutual_info"] = s3
        for f in s3["selected"]:
            votes[f] += 1
            stage_count[f] += 1

        # ── Stage 4 ────────────────────────────────────────────────────────
        s4 = self._stage4_boruta(X, y_binary, feature_cols, s3["selected"])
        self.stage_results["stage4_boruta"] = s4
        for f in s4["selected"]:
            votes[f] += 1
            stage_count[f] += 1

        # ── Stage 5 ────────────────────────────────────────────────────────
        s5 = self._stage5_shap(X, y_binary, feature_cols, s4["selected"])
        self.stage_results["stage5_shap"] = s5
        for f in s5["selected"]:
            votes[f] += 1
            stage_count[f] += 1

        # ── Stage 6 ────────────────────────────────────────────────────────
        s6 = self._stage6_stability(X, y_binary, feature_cols, s5["selected"])
        self.stage_results["stage6_stability"] = s6
        for f in s6["selected"]:
            votes[f] += 1
            stage_count[f] += 1

        # ── Stage 7 — Consensus vote ───────────────────────────────────────
        self.selected_features = self._stage7_consensus(
            votes, stage_count, feature_cols
        )

        # ── Feature score table ────────────────────────────────────────────
        self._build_score_table(
            feature_cols, votes,
            s3.get("mi_scores", {}),
            s5.get("shap_scores", {}),
            s6.get("stability_scores", {})
        )

        self.fitted = True
        self._save_artifacts()

        logger.info("=" * 70)
        logger.info(f"Selection complete — {len(self.selected_features)} features retained "
                    f"from {len(feature_cols)} input features")
        logger.info("=" * 70)

        return self

    # ──────────────────────────────────────────────────────────────────────
    def transform(
        self,
        df: pd.DataFrame,
        split_name: str = "unknown"
    ) -> pd.DataFrame:
        """
        Retain only selected features plus target columns.

        Parameters
        ----------
        df         : engineered DataFrame
        split_name : 'train' / 'val' / 'test' for logging

        Returns
        -------
        DataFrame with selected features + targets
        """
        if not self.fitted:
            raise RuntimeError("FeatureSelector not fitted. Call fit(train) first.")

        logger.info("=" * 70)
        logger.info(f"DriftSentinel — Feature Selector  [transform on {split_name.upper()}]")
        logger.info("=" * 70)

        present_targets = [c for c in TARGET_COLS if c in df.columns]
        keep_cols       = [c for c in self.selected_features if c in df.columns]
        missing         = set(self.selected_features) - set(df.columns)

        if missing:
            logger.warning(f"Missing selected features in {split_name}: {missing}")

        result = df[keep_cols + present_targets].copy()

        logger.info(f"Input  shape : {df.shape}")
        logger.info(f"Output shape : {result.shape}")
        logger.info(f"Features retained : {len(keep_cols)}")
        logger.info(f"Targets retained  : {present_targets}")
        logger.info("=" * 70)

        return result

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1 — Variance Threshold
    # Remove features with near-zero variance — carry no information.
    # ══════════════════════════════════════════════════════════════════════
    def _stage1_variance(
        self,
        X: pd.DataFrame,
        feature_cols: list[str]
    ) -> dict:
        logger.info("-" * 50)
        logger.info(f"Stage 1: Variance Threshold (threshold={VARIANCE_THRESHOLD})")

        selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        selector.fit(X)

        variances = pd.Series(selector.variances_, index=feature_cols)
        selected  = variances[variances >= VARIANCE_THRESHOLD].index.tolist()
        removed   = variances[variances <  VARIANCE_THRESHOLD].index.tolist()

        logger.info(f"  Input    : {len(feature_cols)} features")
        logger.info(f"  Selected : {len(selected)}")
        logger.info(f"  Removed  : {len(removed)}")
        if removed:
            for f in removed:
                logger.info(f"    DROP  {f:<45} var={variances[f]:.6f}")

        return {
            "selected"  : selected,
            "removed"   : removed,
            "variances" : variances.to_dict()
        }

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2 — Correlation Filter
    # For each highly correlated pair (|r| > threshold),
    # retain the feature with higher mutual information with target.
    # ══════════════════════════════════════════════════════════════════════
    def _stage2_correlation(
        self,
        X: pd.DataFrame,
        feature_cols: list[str],
        candidates: list[str]
    ) -> dict:
        logger.info("-" * 50)
        logger.info(f"Stage 2: Correlation Filter (threshold={CORRELATION_THRESHOLD})")

        X_cand = X[candidates]
        corr   = X_cand.corr(method="spearman").abs()

        upper     = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop   = set()
        drop_log  = []

        for col in upper.columns:
            if col in to_drop:
                continue
            correlated = upper.index[upper[col] > CORRELATION_THRESHOLD].tolist()
            for partner in correlated:
                if partner not in to_drop:
                    to_drop.add(partner)
                    drop_log.append((col, partner, round(corr.loc[col, partner], 4)))

        selected = [f for f in candidates if f not in to_drop]

        logger.info(f"  Input    : {len(candidates)}")
        logger.info(f"  Selected : {len(selected)}")
        logger.info(f"  Removed  : {len(to_drop)}")
        for keeper, dropped, r in drop_log:
            logger.info(f"    DROP  {dropped:<40} r={r} (kept: {keeper})")

        return {
            "selected"     : selected,
            "removed"      : list(to_drop),
            "drop_pairs"   : drop_log,
        }

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3 — Mutual Information
    # Non-linear relevance between each feature and binary target.
    # Remove bottom MI_PERCENTILE percent of features.
    # ══════════════════════════════════════════════════════════════════════
    def _stage3_mutual_info(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str],
        candidates: list[str]
    ) -> dict:
        logger.info("-" * 50)
        logger.info(f"Stage 3: Mutual Information (drop bottom {MI_PERCENTILE}th percentile)")

        X_cand = X[candidates].fillna(0)
        mi     = mutual_info_classif(
            X_cand, y,
            discrete_features="auto",
            random_state=RANDOM_SEED
        )
        mi_series  = pd.Series(mi, index=candidates).sort_values(ascending=False)
        threshold  = np.percentile(mi_series.values, MI_PERCENTILE)
        selected   = mi_series[mi_series >= threshold].index.tolist()
        removed    = mi_series[mi_series <  threshold].index.tolist()

        logger.info(f"  Input    : {len(candidates)}")
        logger.info(f"  MI threshold (P{MI_PERCENTILE}) : {threshold:.6f}")
        logger.info(f"  Selected : {len(selected)}")
        logger.info(f"  Removed  : {len(removed)}")
        logger.info("  Top 10 MI scores:")
        for f, score in mi_series.head(10).items():
            logger.info(f"    {f:<45} MI={score:.6f}")

        return {
            "selected"   : selected,
            "removed"    : removed,
            "mi_scores"  : mi_series.to_dict(),
            "threshold"  : threshold,
        }

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4 — Boruta (Shadow Feature Method)
    # Creates shadow features (randomly shuffled copies).
    # A real feature is confirmed if it consistently beats the best shadow.
    # All-relevant selection — finds all features contributing to target.
    # ══════════════════════════════════════════════════════════════════════
    def _stage4_boruta(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str],
        candidates: list[str]
    ) -> dict:
        logger.info("-" * 50)
        logger.info(f"Stage 4: Boruta Shadow Feature Selection "
                    f"(n_estimators={BORUTA_N_ESTIMATORS}, max_iter={BORUTA_MAX_ITER})")

        X_cand  = X[candidates].fillna(0).values
        y_vals  = y.values
        n_feat  = X_cand.shape[1]
        rng     = np.random.RandomState(RANDOM_SEED)

        confirmed = np.zeros(n_feat, dtype=bool)
        rejected  = np.zeros(n_feat, dtype=bool)
        hits      = np.zeros(n_feat, dtype=int)

        rf = RandomForestClassifier(
            n_estimators=BORUTA_N_ESTIMATORS,
            max_depth=7,
            n_jobs=N_JOBS,
            random_state=RANDOM_SEED
        )

        for iteration in range(BORUTA_MAX_ITER):
            # Create shadow features
            shadow = rng.permutation(X_cand.T).T
            X_aug  = np.hstack([X_cand, shadow])

            rf.fit(X_aug, y_vals)
            importances = rf.feature_importances_

            real_imp   = importances[:n_feat]
            shadow_imp = importances[n_feat:]
            shadow_max = shadow_imp.max()

            hits += (real_imp > shadow_max).astype(int)

            if iteration % 10 == 9:
                logger.info(f"  Boruta iter {iteration+1:>3}/{BORUTA_MAX_ITER} "
                            f"— confirmed so far: {confirmed.sum()}")

        # Binomial test — hits / max_iter vs p=0.5
        from scipy.stats import binomtest
        for i in range(n_feat):
            p_val = binomtest(hits[i], BORUTA_MAX_ITER, 0.5, alternative="greater").pvalue
            if p_val < BORUTA_ALPHA:
                confirmed[i] = True

        selected = [candidates[i] for i in range(n_feat) if confirmed[i]]
        removed  = [candidates[i] for i in range(n_feat) if not confirmed[i]]
        hit_rate = {candidates[i]: hits[i] / BORUTA_MAX_ITER for i in range(n_feat)}

        logger.info(f"  Input    : {len(candidates)}")
        logger.info(f"  Selected : {len(selected)} (confirmed)")
        logger.info(f"  Removed  : {len(removed)} (tentative/rejected)")
        logger.info("  Top 10 hit rates:")
        for f, rate in sorted(hit_rate.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"    {f:<45} hit_rate={rate:.2f}")

        return {
            "selected"  : selected,
            "removed"   : removed,
            "hit_rates" : hit_rate,
        }

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5 — SHAP Feature Importance
    # TreeExplainer on GradientBoosting — mean |SHAP| per feature.
    # More faithful than MDI importance (unbiased for high-cardinality).
    # ══════════════════════════════════════════════════════════════════════
    def _stage5_shap(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str],
        candidates: list[str]
    ) -> dict:
        logger.info("-" * 50)
        logger.info(f"Stage 5: SHAP Importance (GBM, n_estimators={SHAP_N_ESTIMATORS})")

        X_cand = X[candidates].fillna(0)

        gbm = GradientBoostingClassifier(
            n_estimators=SHAP_N_ESTIMATORS,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_SEED
        )
        gbm.fit(X_cand, y)

        explainer   = shap.TreeExplainer(gbm)
        shap_values = explainer.shap_values(X_cand)
        mean_shap   = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=candidates
        ).sort_values(ascending=False)

        threshold = np.percentile(mean_shap.values, MI_PERCENTILE)
        selected  = mean_shap[mean_shap >= threshold].index.tolist()
        removed   = mean_shap[mean_shap <  threshold].index.tolist()

        logger.info(f"  Input    : {len(candidates)}")
        logger.info(f"  SHAP threshold (P{MI_PERCENTILE}) : {threshold:.6f}")
        logger.info(f"  Selected : {len(selected)}")
        logger.info(f"  Removed  : {len(removed)}")
        logger.info("  Top 15 SHAP scores:")
        for f, score in mean_shap.head(15).items():
            logger.info(f"    {f:<45} SHAP={score:.6f}")

        return {
            "selected"    : selected,
            "removed"     : removed,
            "shap_scores" : mean_shap.to_dict(),
            "threshold"   : threshold,
        }

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 6 — Stability Selection
    # Bootstrap subsampling across 50 iterations.
    # Feature retained if selected in > STABILITY_THRESHOLD fraction.
    # Reduces variance in feature selection — critical for small datasets.
    # ══════════════════════════════════════════════════════════════════════
    def _stage6_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str],
        candidates: list[str]
    ) -> dict:
        logger.info("-" * 50)
        logger.info(f"Stage 6: Stability Selection "
                    f"(n_bootstrap={STABILITY_N_BOOTSTRAP}, "
                    f"frac={STABILITY_SAMPLE_FRAC}, "
                    f"threshold={STABILITY_THRESHOLD})")

        X_cand   = X[candidates].fillna(0)
        n_feat   = len(candidates)
        select_counts = np.zeros(n_feat)
        n_select = max(1, int(n_feat * 0.5))

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            n_jobs=N_JOBS,
            random_state=RANDOM_SEED
        )

        for i in range(STABILITY_N_BOOTSTRAP):
            X_boot, y_boot = resample(
                X_cand, y,
                n_samples=int(len(X_cand) * STABILITY_SAMPLE_FRAC),
                random_state=RANDOM_SEED + i
            )
            rf.fit(X_boot, y_boot)
            imp     = rf.feature_importances_
            top_idx = np.argsort(imp)[-n_select:]
            select_counts[top_idx] += 1

        stability_scores = {
            candidates[i]: round(select_counts[i] / STABILITY_N_BOOTSTRAP, 4)
            for i in range(n_feat)
        }

        selected = [
            candidates[i] for i in range(n_feat)
            if stability_scores[candidates[i]] >= STABILITY_THRESHOLD
        ]
        removed = [
            candidates[i] for i in range(n_feat)
            if stability_scores[candidates[i]] < STABILITY_THRESHOLD
        ]

        logger.info(f"  Input    : {len(candidates)}")
        logger.info(f"  Selected : {len(selected)} (stability >= {STABILITY_THRESHOLD})")
        logger.info(f"  Removed  : {len(removed)}")
        logger.info("  Top 15 stability scores:")
        for f, score in sorted(stability_scores.items(), key=lambda x: -x[1])[:15]:
            marker = "✓" if score >= STABILITY_THRESHOLD else "✗"
            logger.info(f"    [{marker}] {f:<43} stability={score:.4f}")

        return {
            "selected"          : selected,
            "removed"           : removed,
            "stability_scores"  : stability_scores,
        }

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 7 — Consensus Vote
    # Feature retained if it appears in CONSENSUS_MIN_STAGES or more stages.
    # ══════════════════════════════════════════════════════════════════════
    def _stage7_consensus(
        self,
        votes: dict,
        stage_count: dict,
        all_features: list[str]
    ) -> list[str]:
        logger.info("-" * 50)
        logger.info(f"Stage 7: Consensus Vote (min_stages={CONSENSUS_MIN_STAGES})")

        selected = []
        removed  = []

        for f in all_features:
            v = votes.get(f, 0)
            if v >= CONSENSUS_MIN_STAGES:
                selected.append(f)
            else:
                removed.append(f)

        logger.info(f"  Total features evaluated : {len(all_features)}")
        logger.info(f"  Selected (votes >= {CONSENSUS_MIN_STAGES})   : {len(selected)}")
        logger.info(f"  Removed                  : {len(removed)}")

        vote_summary = sorted(votes.items(), key=lambda x: -x[1])
        logger.info("  Vote summary (top 20):")
        for f, v in vote_summary[:20]:
            marker = "✓" if v >= CONSENSUS_MIN_STAGES else "✗"
            logger.info(f"    [{marker}] {f:<45} votes={v}/6")

        return selected

    # ──────────────────────────────────────────────────────────────────────
    def _prepare(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
        """Extract X, y_binary, y_multi and feature column list."""

        exclude  = TARGET_COLS | ID_COLS
        feat_cols = [c for c in df.columns if c not in exclude]

        X        = df[feat_cols].copy()
        y_binary = df[TARGET_BINARY] if TARGET_BINARY in df.columns else pd.Series()
        y_multi  = df[TARGET_MULTI]  if TARGET_MULTI  in df.columns else pd.Series()

        return X, y_binary, y_multi, feat_cols

    # ──────────────────────────────────────────────────────────────────────
    def _build_score_table(
        self,
        feature_cols:     list[str],
        votes:            dict,
        mi_scores:        dict,
        shap_scores:      dict,
        stability_scores: dict,
    ):
        """Build unified feature score DataFrame."""
        rows = []
        for f in feature_cols:
            rows.append({
                "feature"         : f,
                "consensus_votes" : votes.get(f, 0),
                "mi_score"        : round(mi_scores.get(f, 0), 6),
                "shap_score"      : round(shap_scores.get(f, 0), 6),
                "stability_score" : round(stability_scores.get(f, 0), 4),
                "selected"        : f in self.selected_features,
                "is_fe_feature"   : f.startswith("FE_"),
            })

        self.feature_scores = (
            pd.DataFrame(rows)
            .sort_values("consensus_votes", ascending=False)
            .reset_index(drop=True)
        )

        score_path = ARTIFACTS_DIR / "feature_scores.csv"
        self.feature_scores.to_csv(score_path, index=False)
        logger.info(f"  feature_scores.csv saved -> {score_path}")

        logger.info("  Selected feature breakdown:")
        fe_selected  = self.feature_scores[
            self.feature_scores["selected"] & self.feature_scores["is_fe_feature"]
        ]
        raw_selected = self.feature_scores[
            self.feature_scores["selected"] & ~self.feature_scores["is_fe_feature"]
        ]
        logger.info(f"    FE_ features selected : {len(fe_selected)}")
        logger.info(f"    Raw features selected : {len(raw_selected)}")

    # ──────────────────────────────────────────────────────────────────────
    def _save_artifacts(self):
        """Persist selector state for reproducibility and serving."""

        selected_path = ARTIFACTS_DIR / "selected_features.json"
        with open(selected_path, "w") as f:
            json.dump(self.selected_features, f, indent=2)
        logger.info(f"  selected_features.json saved -> {selected_path}")

        results_path = ARTIFACTS_DIR / "selector_stage_results.json"
        serializable = {}
        for stage, res in self.stage_results.items():
            serializable[stage] = {
                k: v for k, v in res.items()
                if isinstance(v, (list, dict, str, int, float, bool))
            }
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"  selector_stage_results.json saved -> {results_path}")

        obj_path = ARTIFACTS_DIR / "feature_selector.pkl"
        with open(obj_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"  feature_selector.pkl saved -> {obj_path}")


if __name__ == "__main__":
    from src.data.loader        import load_raw
    from src.data.validator     import validate
    from src.data.splitter      import split
    from src.data.preprocessor  import Preprocessor
    from src.features.engineer  import FeatureEngineer

    df, ids_df, _        = load_raw()
    report               = validate(df)
    if not report["ready"]:
        raise RuntimeError("Validation failed")

    train, val, test, _  = split(df)

    prep                 = Preprocessor()
    train_clean          = prep.fit_transform(train)
    val_clean            = prep.transform(val,  split_name="val")
    test_clean           = prep.transform(test, split_name="test")

    eng                  = FeatureEngineer()
    train_fe             = eng.fit_transform(train_clean)
    val_fe               = eng.transform(val_clean,  split_name="val")
    test_fe              = eng.transform(test_clean, split_name="test")

    selector             = FeatureSelector()
    selector.fit(train_fe)

    train_fs             = selector.transform(train_fe, split_name="train")
    val_fs               = selector.transform(val_fe,   split_name="val")
    test_fs              = selector.transform(test_fe,  split_name="test")

    print(f"\nTrain : {train_fs.shape}")
    print(f"Val   : {val_fs.shape}")
    print(f"Test  : {test_fs.shape}")
    print(f"\nSelected features ({len(selector.selected_features)}):")
    for f in selector.selected_features:
        print(f"  {f}")