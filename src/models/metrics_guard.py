"""
DriftSentinel — Metrics integrity guards and AUC comparison tests

WHY THIS EXISTS
    `registry.py` contained `val_metrics = train_v2_metrics`. lgbm_v2 was fitted
    on train+val, so its reported "val AUC = 0.7987" was training performance on
    data it had been fitted on. The comparison table then printed
    "auc val 0.6865 -> 0.7987 +0.1122 lgbm_v2 WINS" and that verdict drove
    PROMOTE, and was written into model_registry.json and registry_history.csv.

    Prose in a markdown file would not have caught it. These are assertions,
    because anything that must hold has to be encoded as an assertion, a test or
    a CI check — documentation cannot fail.

CONTENTS
    validate_metrics_schema   explicit split labelling (R2)
    assert_split_disjoint     a model is never evaluated on data it was fitted on
    assert_metrics_distinct   two split metric dicts are never identical
    delong_roc_test           DeLong test for two correlated ROC AUCs
    paired_bootstrap_auc      paired (optionally patient-clustered) bootstrap CI
"""

from __future__ import annotations

import numpy as np
from scipy import stats

REQUIRED_METRIC_KEYS = {"auc", "f1", "precision", "recall", "brier"}
VALID_SPLITS = ("train", "val", "test", "holdout")


class MetricsIntegrityError(AssertionError):
    """Raised when a metrics dict violates an integrity guarantee."""


def validate_metrics_schema(metrics: dict, split: str) -> dict:
    """
    Enforce that a metrics dict declares which split it came from (R2).

    Returns the dict with an explicit `split` key, raising if it disagrees with
    the split it is being registered under — the failure mode that let training
    metrics be filed as validation metrics.
    """
    if split not in VALID_SPLITS:
        raise MetricsIntegrityError(f"unknown split {split!r}; expected one of {VALID_SPLITS}")
    if not isinstance(metrics, dict) or not metrics:
        raise MetricsIntegrityError(f"{split}: metrics must be a non-empty dict")

    missing = REQUIRED_METRIC_KEYS - set(metrics)
    if missing:
        raise MetricsIntegrityError(f"{split}: metrics missing required keys {sorted(missing)}")

    declared = metrics.get("split")
    if declared is not None and declared != split:
        raise MetricsIntegrityError(
            f"metrics declare split={declared!r} but are being registered as {split!r}")

    return {**metrics, "split": split}


def assert_split_disjoint(model_train_splits, eval_split: str) -> None:
    """
    Raise if a model is evaluated on a split it was fitted on.

    lgbm_v2 was fitted on {'train','val'} and evaluated on 'val'. This is the
    single assertion that would have stopped the original bug at source.
    """
    fitted = set(model_train_splits)
    if eval_split in fitted:
        raise MetricsIntegrityError(
            f"model was fitted on {sorted(fitted)} and cannot be evaluated on "
            f"{eval_split!r}: those metrics are in-sample, not held out")


def assert_metrics_distinct(a: dict, b: dict, name_a: str, name_b: str,
                            tol: float = 1e-12) -> None:
    """
    Raise if two metric dicts for different splits are identical.

    Identical dicts for two different splits means one was copied from the other.
    Held-out metrics on genuinely different data are never bit-identical.
    """
    shared = (set(a) & set(b) & REQUIRED_METRIC_KEYS)
    if not shared:
        return
    if all(abs(float(a[k]) - float(b[k])) <= tol for k in shared):
        raise MetricsIntegrityError(
            f"{name_a} and {name_b} metrics are identical across {sorted(shared)} "
            f"— one was copied from the other, not measured")


# ══════════════════════════════════════════════════════════════════════════
# AUC comparison
# ══════════════════════════════════════════════════════════════════════════

def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks with ties averaged — the rank transform DeLong's estimator needs."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and xs[j + 1] == xs[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _structural_components(pos: np.ndarray, neg: np.ndarray):
    """DeLong V10/V01 structural components for one predictor."""
    m, n = len(pos), len(neg)
    both = np.concatenate([pos, neg])
    r_all = _midrank(both)
    r_pos = _midrank(pos)
    r_neg = _midrank(neg)
    auc = (r_all[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (r_all[:m] - r_pos) / n
    v01 = 1.0 - (r_all[m:] - r_neg) / m
    return auc, v10, v01


def delong_roc_test(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """
    DeLong test for two CORRELATED ROC curves (DeLong, DeLong & Clarke-Pearson 1988).

    Correlated is the operative word: both models score the SAME samples, so an
    unpaired comparison would overstate the variance of the difference. The audit
    asked for exactly this test on the only defensible number in the v1/v2
    comparison, test AUC 0.6560 -> 0.6648.
    """
    y = np.asarray(y_true).astype(int)
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    if not (len(y) == len(p1) == len(p2)):
        raise ValueError("y_true, p1 and p2 must have equal length")
    if len(np.unique(y)) < 2:
        raise ValueError("y_true must contain both classes")

    pos_mask = y == 1
    m, n = int(pos_mask.sum()), int((~pos_mask).sum())

    auc1, v10_1, v01_1 = _structural_components(p1[pos_mask], p1[~pos_mask])
    auc2, v10_2, v01_2 = _structural_components(p2[pos_mask], p2[~pos_mask])

    v10 = np.vstack([v10_1, v10_2])
    v01 = np.vstack([v01_1, v01_2])
    s10 = np.cov(v10, ddof=1)
    s01 = np.cov(v01, ddof=1)
    s = s10 / m + s01 / n

    delta = auc1 - auc2
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        z, p = 0.0, 1.0
    else:
        z = float(delta / np.sqrt(var))
        p = float(2 * stats.norm.sf(abs(z)))
    se = float(np.sqrt(var)) if var > 0 else float("nan")
    return {
        "auc_1": float(auc1), "auc_2": float(auc2),
        "delta_auc_1_minus_2": float(delta),
        "se_delta": se,
        "ci95_delta": [float(delta - 1.96 * se), float(delta + 1.96 * se)] if var > 0 else [np.nan, np.nan],
        "z": z, "p_value": p, "n_pos": m, "n_neg": n,
        "significant_at_0.05": bool(p < 0.05),
        "method": "DeLong, DeLong & Clarke-Pearson (1988), correlated ROC curves",
    }


def paired_bootstrap_auc(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                         groups: np.ndarray | None = None,
                         n_boot: int = 2000, seed: int = 42) -> dict:
    """
    Paired bootstrap CI for the AUC difference.

    PAIRED: each replicate resamples ONE index set and scores both models on it,
    so the two AUCs move together and the difference keeps its correlation.

    `groups` (patient ids) switches on cluster resampling. Patients contribute
    multiple encounters (46.2% multi-visit rows), so row-level resampling is
    anti-conservative (audit F28). If groups is None the interval is reported as
    row-level and labelled as such rather than silently passed off as clustered.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)

    if groups is None:
        idx_pool, clustered = None, False
    else:
        groups = np.asarray(groups)
        uniq, inv = np.unique(groups, return_inverse=True)
        idx_pool = [np.flatnonzero(inv == g) for g in range(len(uniq))]
        clustered = True

    deltas, a1s, a2s = [], [], []
    for _ in range(n_boot):
        if clustered:
            pick = rng.integers(0, len(idx_pool), size=len(idx_pool))
            idx = np.concatenate([idx_pool[i] for i in pick])
        else:
            idx = rng.integers(0, len(y), size=len(y))
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        a1 = roc_auc_score(yb, p1[idx])
        a2 = roc_auc_score(yb, p2[idx])
        a1s.append(a1); a2s.append(a2); deltas.append(a2 - a1)

    d = np.asarray(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {
        "n_boot_effective": int(len(d)),
        "resampling_unit": "patient (cluster)" if clustered else "row",
        "cluster_robust": clustered,
        "caveat": None if clustered else
                  ("row-level resampling: intervals are anti-conservative because "
                   "patients contribute multiple encounters (audit F28)"),
        "mean_auc_1": float(np.mean(a1s)), "mean_auc_2": float(np.mean(a2s)),
        "delta_auc_2_minus_1": float(np.mean(d)),
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
    }
