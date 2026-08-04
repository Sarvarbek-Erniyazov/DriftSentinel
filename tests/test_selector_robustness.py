"""
Regression tests for the selector's empty-candidate crash (Tier 2A.5).

Stages 5 and 6 consume Boruta's confirmed set. On the full training data Boruta
confirms exactly ONE feature, so these stages have always been one feature away
from receiving an empty frame. Inside a CV fold Boruta confirmed ZERO and
`GradientBoostingClassifier.fit` raised
"ValueError: at least one array or dtype is required".

The audit called stages 5-6 no-ops. They were worse than no-ops: an unguarded
crash path that any smaller sample or weaker target can trigger.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.selector import FeatureSelector


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.integers(0, 3, size=n).astype(float),
    }), pd.Series(rng.binomial(1, 0.2, size=n))


def test_stage5_returns_empty_instead_of_raising_on_empty_candidates(frame):
    """The exact crash: Boruta confirmed nothing, stage 5 fitted on 0 columns."""
    X, y = frame
    s = FeatureSelector()
    out = s._stage5_shap(X, y, list(X.columns), [])
    assert out["selected"] == []
    assert "skipped_reason" in out


def test_stage6_returns_empty_instead_of_raising_on_empty_candidates(frame):
    X, y = frame
    s = FeatureSelector()
    out = s._stage6_stability(X, y, list(X.columns), [])
    assert out["selected"] == []
    assert "skipped_reason" in out


def test_stage5_still_works_with_a_single_candidate(frame):
    """The full-data case is ONE candidate — behaviour must be unchanged."""
    X, y = frame
    s = FeatureSelector()
    out = s._stage5_shap(X, y, list(X.columns), ["a"])
    assert "skipped_reason" not in out
    assert set(out["selected"]) <= {"a"}


def test_stage6_still_works_with_a_single_candidate(frame):
    X, y = frame
    s = FeatureSelector()
    out = s._stage6_stability(X, y, list(X.columns), ["a"])
    assert "skipped_reason" not in out
