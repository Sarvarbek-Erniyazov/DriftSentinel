"""
Tests for the operating-point policy (Tier 1.3).

Regression target: the shipped "cost-optimal" threshold flagged 97% of patients
and was reported as a win. A threshold that flags almost everyone must now be
labelled DEGENERATE automatically.
"""

import numpy as np
import pytest

from src.uncertainty.threshold_policy import (
    DEGENERATE_PPR, cost_ratio_sweep, decision_curve_analysis, operating_point,
    predicted_positive_rate, threshold_cost_sensitive, threshold_under_budget,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.45, 4000)
    p = np.clip(rng.normal(0.45 + 0.12 * y, 0.15), 0.001, 0.999)
    return y, p


# ── PPR and degeneracy ────────────────────────────────────────────────────

def test_ppr_matches_definition(data):
    y, p = data
    assert predicted_positive_rate(p, 0.0) == 1.0
    assert predicted_positive_rate(p, 1.01) == 0.0
    assert predicted_positive_rate(p, 0.45) == pytest.approx((p >= 0.45).mean())


def test_regression_low_threshold_is_flagged_degenerate(data):
    """The exact failure mode: a threshold flagging nearly everyone."""
    y, p = data
    op = operating_point(y, p, 0.02)
    assert op["predicted_positive_rate"] > 0.95
    assert op["degenerate"] is True
    assert "triage" in op["degenerate_reason"]


def test_sensible_threshold_is_not_flagged(data):
    y, p = data
    op = operating_point(y, p, 0.60)
    assert op["predicted_positive_rate"] < DEGENERATE_PPR
    assert op["degenerate"] is False and op["degenerate_reason"] is None


def test_operating_point_confusion_counts_are_consistent(data):
    y, p = data
    op = operating_point(y, p, 0.5)
    assert op["tp"] + op["fp"] + op["fn"] + op["tn"] == len(y)
    assert op["alerts_per_1000_patients"] == round(op["predicted_positive_rate"] * 1000)


# ── the rate-form defect ──────────────────────────────────────────────────

def test_rate_form_inflates_the_effective_cost_ratio():
    """At low prevalence the legacy per-class-rate objective silently multiplies
    FN:FP by the inverse odds."""
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.10, 5000)            # 10% prevalence
    p = np.clip(rng.normal(0.10 + 0.15 * y, 0.12), 0.001, 0.999)

    legacy = threshold_cost_sensitive(y, p, 5.0, 1.0, legacy_rate_form=True)
    fixed = threshold_cost_sensitive(y, p, 5.0, 1.0)

    assert legacy["effective_cost_ratio"] > 30      # 5 * 0.9/0.1 = 45
    assert fixed["effective_cost_ratio"] == 5.0
    assert legacy["threshold"] < fixed["threshold"]  # legacy flags more


def test_corrected_threshold_tracks_theory_for_calibrated_scores():
    """For calibrated probabilities the optimum is c_fp/(c_fp+c_fn)."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.99, 20000)
    y = rng.binomial(1, p)                      # perfectly calibrated by construction
    res = threshold_cost_sensitive(y, p, cost_fn=4.0, cost_fp=1.0)
    assert res["theoretical_optimal_threshold_if_calibrated"] == pytest.approx(0.2)
    assert res["threshold"] == pytest.approx(0.2, abs=0.06)


# ── budget constraint ─────────────────────────────────────────────────────

def test_budget_constraint_is_respected(data):
    y, p = data
    for b in (0.10, 0.20, 0.30):
        res = threshold_under_budget(y, p, budget=b)
        assert res["feasible"] is True
        assert res["predicted_positive_rate"] <= b + 1e-9
        assert res["degenerate"] is False


def test_budget_zero_is_reported_infeasible_not_silently_wrong(data):
    y, p = data
    res = threshold_under_budget(y, p, budget=0.0)
    assert res["feasible"] in (True, False)
    if res["feasible"]:
        assert res["predicted_positive_rate"] == 0.0


# ── sweep and DCA ─────────────────────────────────────────────────────────

def test_sweep_identifies_a_feasible_region(data):
    y, p = data
    sw = cost_ratio_sweep(y, p)
    assert len(sw["sweep"]) == 8
    ppr = [r["predicted_positive_rate"] for r in sw["sweep"]]
    assert ppr == sorted(ppr)                    # higher FN cost -> flag more
    assert set(sw["feasible_ratios"]).isdisjoint(sw["degenerate_ratios"])


def test_decision_curve_matches_net_benefit_formula(data):
    y, p = data
    dca = decision_curve_analysis(y, p, thresholds=[0.3])
    r = dca["curve"][0]
    pred = (p >= 0.3).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    expected = tp / len(y) - (fp / len(y)) * (0.3 / 0.7)
    assert r["net_benefit_model"] == pytest.approx(expected, abs=1e-6)


def test_decision_curve_treat_all_equals_prevalence_at_low_threshold(data):
    y, p = data
    dca = decision_curve_analysis(y, p, thresholds=[0.02])
    r = dca["curve"][0]
    w = 0.02 / 0.98
    assert r["net_benefit_treat_all"] == pytest.approx(y.mean() - (1 - y.mean()) * w, abs=1e-6)
