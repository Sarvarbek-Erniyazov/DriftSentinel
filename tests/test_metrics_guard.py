"""
Tests for the metrics integrity guards.

The first test in this file is a regression test for the exact defect the audit
found: `val_metrics = train_v2_metrics` in registry.py. Everything else exists so
that guard cannot rot.

Run: python -m pytest tests/ -q
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.models.metrics_guard import (
    MetricsIntegrityError, assert_metrics_distinct, assert_split_disjoint,
    delong_roc_test, paired_bootstrap_auc, validate_metrics_schema,
)


def _m(auc=0.7, f1=0.6, precision=0.5, recall=0.8, brier=0.2):
    return {"auc": auc, "f1": f1, "precision": precision, "recall": recall, "brier": brier}


# ── the original bug ──────────────────────────────────────────────────────

def test_regression_v2_val_metrics_copied_from_train_is_rejected():
    """lgbm_v2 was fitted on train+val; filing its train metrics as val metrics
    must raise on BOTH guards independently."""
    train_v2 = _m(auc=0.7987, f1=0.7345)
    with pytest.raises(MetricsIntegrityError, match="identical"):
        assert_metrics_distinct(train_v2, dict(train_v2), "lgbm_v2 train", "lgbm_v2 val")
    with pytest.raises(MetricsIntegrityError, match="in-sample"):
        assert_split_disjoint({"train", "val"}, "val")


def test_split_disjoint_allows_genuine_holdout():
    assert_split_disjoint({"train", "val"}, "test")
    assert_split_disjoint({"train"}, "val")


def test_metrics_distinct_allows_different_values():
    assert_metrics_distinct(_m(auc=0.6865), _m(auc=0.6560), "v1 val", "v1 test")


def test_metrics_distinct_tolerance_is_tight():
    a = _m(auc=0.70)
    b = _m(auc=0.70 + 1e-9)
    assert_metrics_distinct(a, b, "a", "b")   # distinguishable -> no raise


# ── schema ────────────────────────────────────────────────────────────────

def test_schema_stamps_split_and_rejects_mismatch():
    out = validate_metrics_schema(_m(), "val")
    assert out["split"] == "val"
    with pytest.raises(MetricsIntegrityError, match="declare split"):
        validate_metrics_schema({**_m(), "split": "train"}, "val")


def test_schema_rejects_missing_keys_and_bad_split():
    with pytest.raises(MetricsIntegrityError, match="missing required keys"):
        validate_metrics_schema({"auc": 0.7}, "val")
    with pytest.raises(MetricsIntegrityError, match="unknown split"):
        validate_metrics_schema(_m(), "production")


# ── DeLong ────────────────────────────────────────────────────────────────

def test_delong_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 800)
    p1 = rng.random(800) * 0.5 + y * 0.3
    p2 = rng.random(800) * 0.5 + y * 0.1
    r = delong_roc_test(y, p1, p2)
    assert r["auc_1"] == pytest.approx(roc_auc_score(y, p1), abs=1e-9)
    assert r["auc_2"] == pytest.approx(roc_auc_score(y, p2), abs=1e-9)


def test_delong_identical_predictors_give_zero_delta_and_p_one():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = rng.random(500) * 0.4 + y * 0.3
    r = delong_roc_test(y, p, p)
    assert r["delta_auc_1_minus_2"] == pytest.approx(0.0, abs=1e-12)
    assert r["p_value"] == pytest.approx(1.0)
    assert not r["significant_at_0.05"]


def test_delong_detects_a_large_real_difference():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    strong = rng.random(2000) * 0.3 + y * 0.8
    weak = rng.random(2000)
    r = delong_roc_test(y, strong, weak)
    assert r["delta_auc_1_minus_2"] > 0.2
    assert r["significant_at_0.05"]


def test_delong_handles_ties_via_midranks():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0] * 20)
    p1 = np.where(y == 1, 0.6, 0.4)          # heavily tied
    p2 = np.full(len(y), 0.5)                 # fully tied -> AUC 0.5
    r = delong_roc_test(y, p1, p2)
    assert r["auc_2"] == pytest.approx(0.5)
    assert r["auc_1"] > r["auc_2"]


def test_delong_rejects_single_class_and_length_mismatch():
    with pytest.raises(ValueError, match="both classes"):
        delong_roc_test(np.ones(10), np.random.rand(10), np.random.rand(10))
    with pytest.raises(ValueError, match="equal length"):
        delong_roc_test(np.array([0, 1, 0]), np.random.rand(3), np.random.rand(2))


# ── bootstrap ─────────────────────────────────────────────────────────────

def test_paired_bootstrap_labels_clustering_honestly():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    p1 = rng.random(400) * 0.5 + y * 0.2
    p2 = rng.random(400) * 0.5 + y * 0.25
    groups = rng.integers(0, 120, 400)

    row = paired_bootstrap_auc(y, p1, p2, groups=None, n_boot=200)
    clu = paired_bootstrap_auc(y, p1, p2, groups=groups, n_boot=200)

    assert row["cluster_robust"] is False and row["caveat"] is not None
    assert clu["cluster_robust"] is True and clu["caveat"] is None
    assert clu["resampling_unit"].startswith("patient")


def test_paired_bootstrap_ci_contains_zero_for_identical_models():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 500)
    p = rng.random(500) * 0.4 + y * 0.3
    r = paired_bootstrap_auc(y, p, p, n_boot=200)
    assert r["ci95"][0] <= 0 <= r["ci95"][1]
    assert not r["excludes_zero"]
