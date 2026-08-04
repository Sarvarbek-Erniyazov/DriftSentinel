"""
Phase 1.0 — the readiness gate (audit F2, BLOCKER).

`summary["pipeline_ready"] = True` was written unconditionally over a check
reporting 12 FAILs. The gate must now be an evaluated expression: drift-related
failures may be permitted and named; integrity failures must always block.
"""

import pytest

from src.features.consistency import (
    EXPECTED_DRIFT_CHECK_PREFIXES, classify_failure, evaluate_gate,
)


def _report(*fails):
    return {"findings": [{"level": "FAIL", "check": c, "split": s, "detail": d}
                         for c, s, d in fails]}


# ── classification ────────────────────────────────────────────────────────

@pytest.mark.parametrize("check", [
    "psi_payer_code", "psi_admission_source_id",
    "target_dist_readmitted_binary_class0",
])
def test_drift_observations_are_classified_expected(check):
    assert classify_failure(check) == "expected_drift"


@pytest.mark.parametrize("check", [
    "leakage_FE_x_vs_target", "schema_columns", "total_row_coverage",
    "null_spike_weight", "dtype_mismatch", "target_leakage_all_features",
])
def test_integrity_failures_are_classified_unexpected(check):
    assert classify_failure(check) == "unexpected"


def test_unknown_checks_default_to_unexpected():
    """Fail closed: an unrecognised check must block, not be waved through."""
    assert classify_failure("some_new_check_nobody_classified") == "unexpected"


# ── the gate ──────────────────────────────────────────────────────────────

def test_regression_drift_failures_alone_do_not_block_but_are_named():
    """The real situation: 12 FAILs, all distribution shift."""
    rep = _report(*[(f"psi_feature_{i}", "test", "PSI=0.9") for i in range(8)],
                  *[(f"target_dist_class{i}", "test", "delta=0.15") for i in range(4)])
    g = evaluate_gate(rep, drift_expected=True)
    assert g["ready"] is True
    assert g["n_expected_failures"] == 12
    assert g["n_unexpected_failures"] == 0
    assert len(g["expected_failures"]) == 12, "failures must be enumerated, not just counted"
    assert "subject matter" in g["reason"]


def test_a_single_integrity_failure_blocks_even_amid_drift():
    """The property the hardcode destroyed."""
    rep = _report(("psi_payer_code", "test", "PSI=0.93"),
                  ("leakage_FE_target_leak", "train", "corr=0.99"))
    g = evaluate_gate(rep, drift_expected=True)
    assert g["ready"] is False
    assert g["n_unexpected_failures"] == 1
    assert "leakage_FE_target_leak" in g["reason"]


def test_drift_failures_block_when_drift_is_not_expected():
    rep = _report(("psi_payer_code", "test", "PSI=0.93"))
    assert evaluate_gate(rep, drift_expected=False)["ready"] is False
    assert evaluate_gate(rep, drift_expected=True)["ready"] is True


def test_clean_report_is_ready():
    g = evaluate_gate({"findings": []}, drift_expected=True)
    assert g["ready"] is True and g["reason"] == "no failures"


def test_gate_records_its_own_rule_and_categories():
    """The gate must log its reasoning, not just its verdict."""
    g = evaluate_gate({"findings": []})
    assert "ready =" in g["rule"]
    assert list(EXPECTED_DRIFT_CHECK_PREFIXES) == g["expected_categories"]
    assert g["integrity_categories"]


def test_gate_never_returns_a_bare_true_without_evidence():
    """Every readiness verdict carries its supporting breakdown."""
    g = evaluate_gate(_report(("psi_x", "test", "PSI=0.9")), drift_expected=True)
    for key in ("reason", "expected_failures", "unexpected_failures", "rule"):
        assert key in g
