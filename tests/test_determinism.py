"""
Tier 2C.4 guards for the determinism comparator itself.

These tests do NOT run the pipeline — that is the CI `determinism` job, which
takes minutes. What they guard is the comparator's ability to FAIL. The whole
value of the determinism check rests on it being able to distinguish
"reproducible" from "not looking", so each classification it can emit is
exercised here against a constructed case.

The case that matters most is `test_byte_only_difference_is_a_failure`: identical
values, different bytes. That is the TARGET_COLS defect class, and it is the one
a metric-comparison check would have waved through.
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from src.monitoring.determinism import (
    EXCLUDED_PATTERNS, compare_snapshots, decide_verdict, falsification_arm,
    hash_file, snapshot,
)


@pytest.fixture
def runs(tmp_path):
    """Two empty sandbox roots with the compared directory layout in place."""
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        (root / "data" / "train").mkdir(parents=True)
        (root / "data" / "production").mkdir(parents=True)
        (root / "outputs" / "artifacts").mkdir(parents=True)
    return a, b


def _frame():
    rng = np.random.default_rng(7)
    return pd.DataFrame({"readmitted_binary": rng.integers(0, 2, 32),
                         "readmitted_multi": rng.integers(0, 3, 32),
                         "feature_x": rng.normal(size=32)})


def _diff(a, b):
    return compare_snapshots(snapshot(a), snapshot(b), a, b)


# ── hashing ──────────────────────────────────────────────────────────────────

def test_hash_moves_when_one_byte_moves(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    before = hash_file(p)
    p.write_bytes(b"abd")
    assert hash_file(p) != before


# ── the defect class this check exists for ───────────────────────────────────

def test_byte_only_difference_is_a_failure(runs):
    """
    Same values, different column order. Every metric would be identical; the
    bytes are not. This must be reported, not tolerated.
    """
    a, b = runs
    df = _frame()
    df.to_parquet(a / "data" / "train" / "train_fs.parquet", index=False)
    df[["feature_x", "readmitted_multi", "readmitted_binary"]].to_parquet(
        b / "data" / "train" / "train_fs.parquet", index=False)

    diffs = _diff(a, b)
    assert [d["kind"] for d in diffs] == ["BYTE_DIFFERENCE_ONLY"]
    assert diffs[0]["sha256_a"] != diffs[0]["sha256_b"]


def test_falsification_arm_reports_that_it_can_detect_that_case():
    arm = falsification_arm()
    assert arm["bytes_differ_as_constructed"] is True
    assert arm["harness_can_detect_byte_only_difference"] is True
    assert arm["classifications_returned"] == ["BYTE_DIFFERENCE_ONLY"]


# ── the other classifications ────────────────────────────────────────────────

def test_identical_runs_produce_no_differences(runs):
    a, b = runs
    df = _frame()
    for root in (a, b):
        df.to_parquet(root / "data" / "train" / "train_fs.parquet", index=False)
        (root / "outputs" / "artifacts" / "selected_features.json").write_text(
            json.dumps({"features": ["a", "b"]}))
    assert _diff(a, b) == []


def test_changed_values_are_classified_as_a_value_difference(runs):
    a, b = runs
    df = _frame()
    df.to_parquet(a / "data" / "train" / "train_fs.parquet", index=False)
    df.assign(feature_x=df["feature_x"] + 1.0).to_parquet(
        b / "data" / "train" / "train_fs.parquet", index=False)
    assert [d["kind"] for d in _diff(a, b)] == ["VALUE_DIFFERENCE"]


def test_pickles_are_opaque_and_a_byte_difference_is_not_excused(runs):
    """
    A pickle cannot be compared by value, so a byte difference in one cannot be
    shown to be benign. It must fail rather than be waved through as 'probably
    just serialisation'.
    """
    a, b = runs
    for root, payload in ((a, {"cols": ["x", "y"]}), (b, {"cols": ["y", "x"]})):
        with open(root / "outputs" / "artifacts" / "pipeline_objects.pkl", "wb") as f:
            pickle.dump(payload, f)
    assert [d["kind"] for d in _diff(a, b)] == ["OPAQUE_BYTE_DIFFERENCE"]


def test_an_artifact_missing_from_one_run_is_reported(runs):
    a, b = runs
    _frame().to_parquet(a / "data" / "train" / "train_fs.parquet", index=False)
    diffs = _diff(a, b)
    assert [d["kind"] for d in diffs] == ["MISSING_IN_ONE_RUN"]
    assert diffs[0]["present_in_run_a"] and not diffs[0]["present_in_run_b"]


def test_json_value_differences_are_caught(runs):
    a, b = runs
    for root, sel in ((a, ["x", "y"]), (b, ["x", "z"])):
        (root / "outputs" / "artifacts" / "selected_features.json").write_text(
            json.dumps({"features": sel}))
    assert [d["kind"] for d in _diff(a, b)] == ["VALUE_DIFFERENCE"]


# ── exclusions are declared, not silent ──────────────────────────────────────

def test_excluded_paths_carry_a_stated_reason():
    assert EXCLUDED_PATTERNS, "exclusions must be enumerated, not implicit"
    for pattern, reason in EXCLUDED_PATTERNS.items():
        assert isinstance(reason, str) and len(reason) > 20, (
            f"{pattern} is excluded without a usable reason")


# ── verdict logic: a green must be earned ────────────────────────────────────

_ARM_OK = {"harness_can_detect_byte_only_difference": True}
_ARM_BROKEN = {"harness_can_detect_byte_only_difference": False}


def test_clean_run_passes():
    assert decide_verdict(_ARM_OK, True, [], 6)[0] == "PASS"


def test_differences_fail():
    assert decide_verdict(_ARM_OK, True, [{"kind": "VALUE_DIFFERENCE"}], 6)[0] == "FAIL"


def test_a_broken_comparator_cannot_report_pass():
    """No differences found by a comparator that cannot find differences."""
    assert decide_verdict(_ARM_BROKEN, True, [], 6)[0] == "UNKNOWN"


def test_identical_hash_randomisation_cannot_report_pass():
    """
    If both runs saw the same string-hash randomisation, the check would report
    'no differences' whether or not a set-ordering defect existed. That is not a
    pass — it is an absence of measurement.
    """
    assert decide_verdict(_ARM_OK, False, [], 6)[0] == "UNKNOWN"
