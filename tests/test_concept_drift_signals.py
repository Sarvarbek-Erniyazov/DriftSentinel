"""
Tests for the Tier 1.4/1.5 changes to the concept-drift evidence rules.

1.4 — label_drift must be scale-free with respect to prevalence. The old fixed
      absolute threshold of 0.05 meant ~11% relative under the merged target and
      ~45% relative under <30, where it stopped firing on real shifts.
1.5 — cusum_alarm and ph_alarm are retained as diagnostics but must never be
      counted as evidence, and severity boundaries must scale with the number of
      voting signals rather than staying pinned to 8.
"""

import numpy as np
import pandas as pd
import pytest

from src.drift import concept_drift as cd


@pytest.fixture(autouse=True)
def _isolate_artifacts(tmp_path, monkeypatch):
    """
    Detector runs write reports to REPORTS_DIR. Without this the test suite
    litters outputs/log/ (and, since Tier 1.7, outputs/reports/superseded/)
    with throwaway artifacts — tests must not contaminate the evidence trail.
    """
    monkeypatch.setattr(cd, "REPORTS_DIR", tmp_path)


def _frame(n, pos_rate, seed, signal=0.25):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, pos_rate, n)
    x = rng.normal(y * signal, 1.0, n)
    return pd.DataFrame({"f1": x, "readmitted_binary": y})


def _predict(X):
    x = np.asarray(X, dtype=float)[:, 0]
    return 1 / (1 + np.exp(-x))


def _detect(ref, prod, threshold=0.5):
    return cd.ConceptDriftDetector("test").detect(
        ref_df=ref, prod_df=prod, feat_cols=["f1"], predict_fn=_predict,
        ref_name="t_ref", prod_name="t_prod", threshold=threshold, n_windows=5)


# ── Tier 1.5: retired signals ─────────────────────────────────────────────

def test_config_declares_six_voting_signals_and_two_diagnostics():
    assert len(cd.VOTING_SIGNALS) == 6
    assert set(cd.DIAGNOSTIC_SIGNALS) == {"cusum_alarm", "ph_alarm"}
    assert not set(cd.VOTING_SIGNALS) & set(cd.DIAGNOSTIC_SIGNALS)


def test_sequential_signals_are_never_counted_as_evidence():
    rep = _detect(_frame(1200, 0.30, 1), _frame(1200, 0.30, 2))
    assert "cusum_alarm" not in rep["evidence"]
    assert "ph_alarm" not in rep["evidence"]
    assert set(rep["diagnostics_not_evidence"]) == {"cusum_alarm", "ph_alarm"}
    assert rep["n_voting_signals"] == 6
    assert rep["n_evidence"] <= 6


def test_retired_signals_are_still_reported_not_deleted():
    """Retained in code so the claim can be verified, not just read about."""
    rep = _detect(_frame(1200, 0.30, 3), _frame(1200, 0.30, 4))
    assert "cusum" in rep and "page_hinkley" in rep
    assert rep["retired_signals"]["signals"] == cd.DIAGNOSTIC_SIGNALS
    assert "structurally broken" in rep["retired_signals"]["reason"]


def test_severity_boundaries_scale_with_voting_count():
    rep = _detect(_frame(1200, 0.30, 5), _frame(1200, 0.30, 6))
    b = rep["severity_boundaries"]
    assert b["critical_min"] == int(np.ceil(0.625 * 6))   # 4, was 5 of 8
    assert b["moderate_min"] == int(np.ceil(0.375 * 6))   # 3, was 3 of 8
    assert b["critical_min"] <= 6 and b["moderate_min"] < b["critical_min"]


# ── Tier 1.4: label_drift is prevalence-scaled ────────────────────────────

def test_no_label_shift_does_not_fire():
    rep = _detect(_frame(4000, 0.11, 10), _frame(4000, 0.11, 11))
    assert rep["label_shift"]["label_drift"] is False


def test_relative_shift_at_low_prevalence_now_fires():
    """11.2% -> 9.0% is a 20% relative drop. The old absolute rule missed it."""
    rep = _detect(_frame(9000, 0.112, 20), _frame(9000, 0.090, 21))
    ls = rep["label_shift"]
    assert ls["legacy_absolute_rule"]["would_fire"] is False   # |delta| = 0.022
    assert abs(ls["relative_change"]) >= 0.10
    assert ls["label_drift"] is True


def test_significant_but_tiny_relative_change_does_not_fire():
    """Large n makes trivial differences significant; the effect floor blocks them."""
    rep = _detect(_frame(60000, 0.300, 30), _frame(60000, 0.312, 31))
    ls = rep["label_shift"]
    assert ls["p_value"] < 0.01            # statistically significant
    assert abs(ls["relative_change"]) < 0.10
    assert ls["label_drift"] is False      # but not a meaningful effect


def test_large_relative_change_but_insignificant_does_not_fire():
    """Tiny samples: a big relative swing that the data cannot support."""
    rep = _detect(_frame(220, 0.10, 40), _frame(220, 0.13, 41))
    ls = rep["label_shift"]
    if ls["p_value"] >= 0.01:
        assert ls["label_drift"] is False


def test_rule_is_recorded_in_the_report():
    rep = _detect(_frame(1200, 0.30, 50), _frame(1200, 0.30, 51))
    ls = rep["label_shift"]
    for k in ("relative_change", "z_stat", "p_value", "alpha",
              "min_relative_change", "rule", "legacy_absolute_rule"):
        assert k in ls
    assert "two-proportion" in ls["rule"]
