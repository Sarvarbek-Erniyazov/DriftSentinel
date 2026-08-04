"""
DriftSentinel — Phase 0.1: Temporal validity of `encounter_id`

WHAT
    Tests whether the ordering of `encounter_id` carries calendar-time
    information, using external clinical anchors whose timing is known
    independently of this dataset, and separately tests the mechanical
    alternative (right-censoring / observation-window truncation correlated
    with patient entry order).

WHY
    The repository claims a "patient-level temporal split" and attributes an
    observed label shift to concept drift. This dataset (Health Facts,
    1999-2008) has NO date column and the validator reports
    `encounter_id_monotonic = False`. Both claims are unverified. Every
    downstream number's INTERPRETATION depends on the answer, which is why
    this runs before any bug is fixed.

HYPOTHESES
    H0 (batch/arbitrary) : encounter_id rank is independent of calendar time.
    H1 (chronological)   : encounter_id rank is MONOTONICALLY associated with
                           calendar time.

    H1 deliberately does not claim linearity. The Health Facts panel grew over
    the decade, so encounter density per unit time is not uniform and decile k
    does not mean "year 1999+k". All anchor tests are therefore rank-based.

SEPARATE VERDICTS
    This module reports TWO verdicts, because they are different questions:
      1. ordering_verdict — is encounter_id chronological?
      2. split_validity   — is the current split temporal?
    (2) is NOT_TEMPORAL_BY_CONSTRUCTION regardless of (1): splitting on
    first-encounter-per-patient sorts patients by ENTRY COHORT, so a patient
    who entered in 1999 and kept visiting through 2008 is wholly in train,
    late encounters included.

OUTPUTS
    outputs/reports/temporal_validity.json
    outputs/reports/temporal_language_inventory.json   (Phase 0.5 input only)
    outputs/figure/32..38_*.png
    outputs/log/temporal_validity.log
"""

from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("temporal_validity")

ROOT         = Path(__file__).resolve().parents[2]
CONFIG_PATH  = ROOT / "configs" / "temporal_validity.yaml"
RAW_CSV      = ROOT / "data" / "raw" / "diabetes_hospital" / "diabetic_data.csv"
REPORTS_DIR  = ROOT / "outputs" / "reports"
FIGURE_DIR   = ROOT / "outputs" / "figure"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Figure style — explicit, not matplotlib defaults (audit F30).
plt.rcParams.update({
    "figure.dpi"      : 110,
    "savefig.dpi"     : 160,
    "font.size"       : 9,
    "axes.grid"       : True,
    "grid.alpha"      : 0.25,
    "grid.linewidth"  : 0.6,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.titlesize"  : 10,
    "axes.titleweight": "bold",
    "legend.frameon"  : False,
})

C_MAIN, C_ALT, C_WARN, C_NULL = "#2b6cb0", "#c05621", "#9b2c2c", "#718096"


# ══════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════

def _load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_raw_verbatim(path: Path = RAW_CSV) -> pd.DataFrame:
    """
    Load the raw CSV with maximal fidelity.

    WHY NOT `src.data.loader.load_raw`: that loader calls read_csv with only
    `na_values="?"`, leaving pandas' DEFAULT NA list active — and the string
    "None" is in that list. `A1Cresult == "None"` means "HbA1c was not measured
    during this encounter", which is the exact quantity anchor A3 measures, and
    the standard loader silently converts it to NaN. Here "?" alone is missing;
    every other literal is preserved.
    """
    df = pd.read_csv(path, na_values=["?"], keep_default_na=False, low_memory=False)
    logger.info(f"Raw load        : {df.shape[0]:,} rows x {df.shape[1]} cols  ({path.name})")
    assert df["encounter_id"].is_unique, "encounter_id must be unique"
    return df


# ══════════════════════════════════════════════════════════════════════════
# Statistical helpers
# ══════════════════════════════════════════════════════════════════════════

def _rank_fraction(values: pd.Series) -> np.ndarray:
    """Rank of each value in (0, 1]. Ties broken by order ('first'); ids are unique."""
    return (values.rank(method="first").to_numpy() / len(values))


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Display-only; clustering is handled by bootstrap."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (float((c - h) / d), float((c + h) / d))


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of y on x. Written out so the cluster bootstrap can reuse the sums."""
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx, sxy = (x * x).sum(), (x * y).sum()
    denom = n * sxx - sx * sx
    return float("nan") if denom == 0 else float((n * sxy - sx * sy) / denom)


def _group_sufficient_stats(x: np.ndarray, y: np.ndarray,
                            gcodes: np.ndarray, n_g: int) -> np.ndarray:
    """Per-group [n, sum x, sum y, sum x^2, sum xy] — the OLS slope is a
    function of these, so a bootstrap replicate is a gather-and-sum."""
    ones = np.ones_like(x, dtype=float)
    return np.column_stack([
        np.bincount(gcodes, weights=ones,          minlength=n_g),
        np.bincount(gcodes, weights=x,             minlength=n_g),
        np.bincount(gcodes, weights=y,             minlength=n_g),
        np.bincount(gcodes, weights=x * x,         minlength=n_g),
        np.bincount(gcodes, weights=x * y,         minlength=n_g),
    ])


def _slope_from_stats(t: np.ndarray) -> float:
    n_, sx, sy, sxx, sxy = t
    denom = n_ * sxx - sx * sx
    return np.nan if denom == 0 else (n_ * sxy - sx * sy) / denom


def _cluster_bootstrap_slopes(
    x: np.ndarray, ys: dict[str, np.ndarray], groups: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    PAIRED patient-clustered bootstrap for one or more outcomes on a common x.

    WHY CLUSTERED: 46.2% of rows come from multi-visit patients, so row-level
    intervals are anti-conservative (audit F28). Patients are resampled with
    replacement and the exact OLS slope is re-derived from per-patient
    sufficient statistics.

    WHY PAIRED: contrasts between outcomes (the horizon-contrast test) must be
    computed on the SAME resample. Differencing two independently drawn
    bootstrap distributions inflates the variance of the contrast and is not a
    paired comparison.
    """
    gcodes, _ = pd.factorize(groups)
    n_g = int(gcodes.max()) + 1
    S = {k: _group_sufficient_stats(x, v.astype(float), gcodes, n_g) for k, v in ys.items()}
    out = {k: np.empty(n_boot) for k in ys}
    for b in range(n_boot):
        idx = rng.integers(0, n_g, size=n_g)          # one shared resample
        for k, mat in S.items():
            out[k][b] = _slope_from_stats(mat[idx].sum(axis=0))
    return out


def _cluster_bootstrap_slope(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    """Single-outcome convenience wrapper returning (lo, hi, draws)."""
    draws = _cluster_bootstrap_slopes(x, {"y": y}, groups, n_boot, rng)["y"]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi), draws


def _permutation_slope_test(
    x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator,
    direction: str,
) -> dict:
    """
    Permute rank positions and recompute the slope.

    The permutation null is exactly H0 ("ordering carries no information"),
    which is why it is preferred here over a parametric test.

    CAVEAT recorded in the report: row-level permutation ignores within-patient
    clustering and is therefore anti-conservative. The clustered bootstrap CI
    reported alongside is the cluster-robust quantity; a conclusion is only
    claimed when both agree.
    """
    obs = _ols_slope(x, y)
    # Only sum(x * y_perm) changes under permutation; the rest are invariant.
    n = len(x)
    sx, sy, sxx = x.sum(), y.sum(), (x * x).sum()
    denom = n * sxx - sx * sx
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = (n * float(x @ rng.permutation(y)) - sx * sy) / denom
    if direction == "decrease":
        p = float((np.sum(null <= obs) + 1) / (n_perm + 1))
        mde = float(np.percentile(null, 5))
    else:
        p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
        mde = float(np.percentile(null, 95))
    return {
        "observed_slope"          : obs,
        "p_permutation"           : p,
        "n_permutations"          : n_perm,
        "direction_tested"        : direction,
        "minimum_detectable_slope": mde,
        "null_slope_sd"           : float(null.std(ddof=1)),
    }


def _bh_fdr(pvals: dict[str, float], q: float) -> dict[str, dict]:
    """Benjamini-Hochberg. Applied to the anchor family from the outset."""
    keys = list(pvals)
    p = np.array([pvals[k] for k in keys], dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 1.0
    for rank_i in range(m - 1, -1, -1):
        idx = order[rank_i]
        val = p[idx] * m / (rank_i + 1)
        prev = min(prev, val)
        adj[idx] = min(prev, 1.0)
    return {k: {"p_raw": float(p[i]), "p_adj": float(adj[i]), "reject": bool(adj[i] <= q)}
            for i, k in enumerate(keys)}


def _js_divergence(p: pd.Series, q: pd.Series) -> float:
    """Jensen-Shannon divergence between two categorical distributions (base 2)."""
    cats = sorted(set(p.index) | set(q.index))
    pv = np.array([p.get(c, 0.0) for c in cats], dtype=float)
    qv = np.array([q.get(c, 0.0) for c in cats], dtype=float)
    pv = pv / pv.sum() if pv.sum() else pv
    qv = qv / qv.sum() if qv.sum() else qv
    m = 0.5 * (pv + qv)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(pv, m) + 0.5 * _kl(qv, m)


def _decile_rates(rank_frac: np.ndarray, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Per-bin positive rate with Wilson interval, indexed by rank bin."""
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(rank_frac, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        m = b == i
        n, k = int(m.sum()), int(y[m].sum())
        lo, hi = _wilson(k, n)
        rows.append({"bin": i + 1, "bin_mid": (edges[i] + edges[i + 1]) / 2,
                     "n": n, "k": k, "rate": (k / n) if n else np.nan,
                     "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# Anchors
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AnchorResult:
    name: str
    role: str
    kind: str
    verification: str
    n_positive_rows: int = 0
    n_positive_patients: int = 0
    outcome: str = "NOT_RUN"          # PASS / FAIL / CONTRADICTORY / UNDERPOWERED
    p_raw: float = float("nan")
    detail: dict = field(default_factory=dict)
    note: str = ""


def _anchor_extinction(
    df: pd.DataFrame, rank: np.ndarray, name: str, cfg: dict, alpha: float,
) -> AnchorResult:
    """
    Existence anchor, EARLY direction: a drug withdrawn from market on a known
    date cannot appear after it. Test statistic is the MAXIMUM rank fraction
    among positives, evaluated at PATIENT level (each patient contributes its
    earliest positive), so clustering cannot inflate the result.

    Exact one-sided null: P(all n patient-minima <= t) = t^n.

    ASYMMETRY, stated up front: positives clustered at the bottom is strong
    evidence FOR chronological ordering; positives scattered mid-range is only
    weak evidence against it (a handful of miscoded or lagged records would look
    the same). This is reported as a one-sided test and is not read as a null.
    """
    col, neg = cfg["column"], cfg["positive_when_not"]
    mask = df[col].to_numpy() != neg
    res = AnchorResult(name=name, role=cfg["role"], kind=cfg["kind"],
                       verification=cfg["verification"])
    res.n_positive_rows = int(mask.sum())
    if res.n_positive_rows == 0:
        res.outcome, res.note = "UNDERPOWERED", "no positive rows"
        return res

    sub = pd.DataFrame({"pat": df.loc[mask, "patient_nbr"].to_numpy(), "r": rank[mask]})
    pat_min = sub.groupby("pat")["r"].min().to_numpy()
    n = len(pat_min)
    stat = float(pat_min.max())
    p = float(stat ** n)

    res.n_positive_patients = n
    res.p_raw = p
    res.detail = {
        "event_date"            : cfg["event_date"],
        "positive_rank_fractions": [round(float(v), 6) for v in np.sort(sub["r"].to_numpy())],
        "patient_min_rank_fractions": [round(float(v), 6) for v in np.sort(pat_min)],
        "statistic_max_patient_rank": stat,
        "exact_p_one_sided_early": p,
        "interpretation": ("all positives fall in the lowest "
                           f"{stat*100:.2f}% of encounter_id ranks"),
    }
    res.outcome = "PASS" if p < alpha else "UNDERPOWERED"
    res.note = ("one-sided; a non-significant result here is NOT evidence "
                "against chronological ordering")
    return res


def _anchor_introduction(
    df: pd.DataFrame, rank: np.ndarray, name: str, cfg: dict, alpha: float,
) -> AnchorResult:
    """
    Existence anchor, LATE direction: an ICD-9 code cannot be assigned before
    it exists. Statistic is the MINIMUM rank fraction among positives, at
    patient level. Exact one-sided null: P(all minima >= t) = (1 - t)^n.
    """
    prefix = cfg["diag_prefix"]
    diag = df[["diag_1", "diag_2", "diag_3"]].astype(str)
    mask = diag.apply(lambda s: s.str.startswith(prefix)).any(axis=1).to_numpy()

    res = AnchorResult(name=name, role=cfg["role"], kind=cfg["kind"],
                       verification=cfg["verification"])
    res.n_positive_rows = int(mask.sum())
    if res.n_positive_rows == 0:
        res.outcome, res.note = "UNDERPOWERED", "no positive rows"
        return res

    sub = pd.DataFrame({"pat": df.loc[mask, "patient_nbr"].to_numpy(), "r": rank[mask]})
    pat_min = sub.groupby("pat")["r"].min().to_numpy()
    n = len(pat_min)
    stat = float(pat_min.min())
    p = float((1.0 - stat) ** n)

    res.n_positive_patients = n
    res.p_raw = p
    res.detail = {
        "event_date"                : cfg["event_date"],
        "statistic_min_patient_rank": stat,
        "exact_p_one_sided_late"    : p,
        "rank_quantiles_of_positives": {
            q: round(float(np.quantile(sub["r"], v)), 4)
            for q, v in [("p00", 0), ("p01", .01), ("p05", .05), ("p25", .25),
                         ("p50", .50), ("p75", .75), ("p95", .95), ("p100", 1.0)]},
        "n_positive_in_first_decile": int((sub["r"] < 0.10).sum()),
        "n_positive_in_first_half"  : int((sub["r"] < 0.50).sum()),
        "verification_note"         : cfg.get("verification_note", ""),
    }
    res.outcome = "PASS" if p < alpha else "FAIL"
    return res


def _anchor_share_changepoint(
    df: pd.DataFrame, rank: np.ndarray, name: str, cfg: dict,
    conf: dict, rng: np.random.Generator,
) -> AnchorResult:
    """
    Differential anchor: within-class share  rosi / (rosi + pio).

    WHY A SHARE, not the rosiglitazone rate: a bare rosiglitazone decline is
    confounded by the secular TZD-class decline and by Health Facts panel
    composition (hospitals were onboarded across the decade and the hospital
    identifier is absent from the UCI release, so it cannot be conditioned on).
    Both confounds act on numerator and denominator together and largely cancel
    in the ratio; the May 2007 safety event does not.

    The prediction is a STEP in the upper rank tail, not a linear decline, so
    the primary statistic is a changepoint (max two-proportion z over candidate
    splits inside a pre-registered search window), permutation-calibrated.
    A monotone-trend test is reported alongside as a secondary view.
    """
    neg = cfg["positive_when_not"]
    num = df[cfg["numerator_column"]].to_numpy() != neg
    den_cols = cfg["denominator_columns"]
    den = np.zeros(len(df), dtype=bool)
    for c in den_cols:
        den |= (df[c].to_numpy() != neg)

    both = num & (df["pioglitazone"].to_numpy() != neg)
    keep = den & ~both                       # exclude dual-TZD rows from the share
    y = num[keep].astype(float)
    r = rank[keep]
    pats = df.loc[keep, "patient_nbr"].to_numpy()
    order = np.argsort(r)
    y_o, r_o = y[order], r[order]
    m = len(y_o)

    res = AnchorResult(name=name, role=cfg["role"], kind=cfg["kind"],
                       verification=cfg["verification"])
    res.n_positive_rows = int(num.sum())
    res.n_positive_patients = int(df.loc[num, "patient_nbr"].nunique())

    lo_q, hi_q = conf["changepoint_search_lo"], conf["changepoint_search_hi"]
    cands = np.unique(np.linspace(int(lo_q * m), int(hi_q * m),
                                  conf["changepoint_n_candidates"]).astype(int))
    cands = cands[(cands > 10) & (cands < m - 10)]

    def _max_z(vec: np.ndarray) -> tuple[float, int]:
        cs = np.concatenate([[0.0], np.cumsum(vec)])
        n1 = cands.astype(float)
        n2 = float(m) - n1
        k1 = cs[cands]
        k2 = cs[-1] - k1
        p1, p2 = k1 / n1, k2 / n2
        pp = (k1 + k2) / (n1 + n2)
        se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, (p1 - p2) / se, 0.0)
        j = int(np.nanargmax(z))
        return float(z[j]), int(cands[j])

    obs_z, obs_c = _max_z(y_o)
    null = np.empty(conf["n_permutations"])
    for i in range(conf["n_permutations"]):
        null[i], _ = _max_z(rng.permutation(y_o))
    p_cp = float((np.sum(null >= obs_z) + 1) / (conf["n_permutations"] + 1))

    share_before = float(y_o[:obs_c].mean())
    share_after  = float(y_o[obs_c:].mean())
    delta = share_before - share_after

    # Cluster bootstrap on the step size at the FIXED estimated split point.
    # (Re-selecting the split inside each replicate would conflate step
    # uncertainty with selection uncertainty; the split is reported as an
    # estimate without its own CI, and that is stated in the report.)
    split_rank = float(r_o[obs_c])
    left = r < split_rank
    gcodes, _ = pd.factorize(pats)
    n_g = int(gcodes.max()) + 1
    S = np.column_stack([
        np.bincount(gcodes, weights=left.astype(float),      minlength=n_g),
        np.bincount(gcodes, weights=(left * y),              minlength=n_g),
        np.bincount(gcodes, weights=(~left).astype(float),   minlength=n_g),
        np.bincount(gcodes, weights=((~left) * y),           minlength=n_g),
    ])
    boot = np.empty(conf["n_bootstrap"])
    for b in range(conf["n_bootstrap"]):
        nl, kl, nr, kr = S[rng.integers(0, n_g, size=n_g)].sum(axis=0)
        boot[b] = (kl / nl - kr / nr) if (nl > 0 and nr > 0) else np.nan
    ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5])

    trend = _permutation_slope_test(r, y, conf["n_permutations"], rng, "decrease")
    b_lo, b_hi, _ = _cluster_bootstrap_slope(r, y, pats, conf["n_bootstrap"], rng)

    res.p_raw = p_cp
    res.detail = {
        "event_date"              : cfg["event_date"],
        "n_tzd_rows_used"         : int(m),
        "n_dual_tzd_rows_excluded": int(both.sum()),
        "n_rosiglitazone_rows"    : int(num.sum()),
        "n_pioglitazone_rows"     : int((df["pioglitazone"].to_numpy() != neg).sum()),
        "rows_per_patient_tzd"    : round(float(m / max(1, len(np.unique(pats)))), 3),
        "changepoint": {
            "search_window_rank_fraction": [lo_q, hi_q],
            "estimated_split_rank_fraction": split_rank,
            "estimated_split_index": obs_c,
            "max_z": obs_z,
            "p_permutation": p_cp,
            "minimum_detectable_z": float(np.percentile(null, 95)),
            "share_before": share_before,
            "share_after": share_after,
            "delta_share": float(delta),
            "delta_share_ci95_cluster_bootstrap": [float(ci_lo), float(ci_hi)],
            "direction_predicted": "step DOWN in rosiglitazone share at high ranks",
        },
        "monotone_trend_secondary": {**trend,
                                     "slope_ci95_cluster_bootstrap": [b_lo, b_hi]},
    }
    if p_cp < conf["alpha"] and delta > 0 and ci_lo > 0:
        res.outcome = "PASS"
    elif p_cp < conf["alpha"] and delta < 0:
        res.outcome = "CONTRADICTORY"
    else:
        res.outcome = "FAIL"
    return res


def _anchor_monotone_rate(
    df: pd.DataFrame, rank: np.ndarray, name: str, cfg: dict,
    conf: dict, rng: np.random.Generator,
) -> AnchorResult:
    """Supporting anchor: HbA1c measurement rate should rise if ordering is chronological."""
    col, neg = cfg["column"], cfg["negative_value"]
    y = (df[col].to_numpy() != neg).astype(float)
    pats = df["patient_nbr"].to_numpy()

    res = AnchorResult(name=name, role=cfg["role"], kind=cfg["kind"],
                       verification=cfg["verification"])
    res.n_positive_rows = int(y.sum())
    res.n_positive_patients = int(df.loc[y.astype(bool), "patient_nbr"].nunique())

    trend = _permutation_slope_test(rank, y, conf["n_permutations"], rng, "increase")
    lo, hi, _ = _cluster_bootstrap_slope(rank, y, pats, conf["n_bootstrap"], rng)
    rho, p_rho = stats.spearmanr(rank, y)

    res.p_raw = trend["p_permutation"]
    res.detail = {
        **trend,
        "overall_rate"                : float(y.mean()),
        "slope_ci95_cluster_bootstrap": [lo, hi],
        "spearman_rho"                : float(rho),
        "caveat"                      : cfg["caveat"].strip(),
        "role_note": ("DEMOTED to supporting: the published trend is ambulatory, "
                      "this variable is inpatient ordering behaviour; a flat "
                      "result here is uninformative, a decline would be evidence "
                      "against H1"),
    }
    if trend["p_permutation"] < conf["alpha"] and trend["observed_slope"] > 0 and lo > 0:
        res.outcome = "PASS"
    elif trend["observed_slope"] < 0 and hi < 0:
        res.outcome = "CONTRADICTORY"
    else:
        res.outcome = "FAIL"
    return res


def _null_anchor_check(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Sanity check. These columns have a single level, so no trend can exist.
    A pipeline reporting significance here is broken, not insightful.
    """
    out = {}
    for c in cols:
        vals = df[c].value_counts().to_dict()
        out[c] = {"levels": {str(k): int(v) for k, v in vals.items()},
                  "n_levels": len(vals),
                  "testable": len(vals) > 1}
    return out


def _label_interval_coherence(
    df: pd.DataFrame, rank: np.ndarray, coherence: dict,
    n_boot: int, rng: np.random.Generator,
) -> dict:
    """
    INTERNAL anchor — no external literature required.

    `readmitted` encodes a known time interval: "<30" means the patient came
    back within 30 days, ">30" means later. The label-observability audit shows
    the label is essentially an in-extract successor indicator (only ~0.9% of
    "NO" rows have any later encounter in the extract), so for labelled rows we
    can measure the RANK DISTANCE to the patient's next encounter.

    If encounter_id ordering is chronological, that rank distance must be
    systematically SHORTER for "<30" than for ">30". Nothing about a
    load-batch ordering predicts this, and no panel-composition confound
    produces it: it is a within-patient comparison of a known interval.

    Second, stronger step: the anchor timeline gives a piecewise-linear
    rank -> calendar map. Applying it converts each rank gap into implied DAYS,
    so the 30-day boundary becomes directly checkable. Agreement calibrates the
    entire chronology hypothesis against a quantity whose semantics are exact.
    """
    d = df[["patient_nbr", "encounter_id", "readmitted"]].copy()
    d["rank"] = rank
    d = d.sort_values(["patient_nbr", "encounter_id"])
    d["next_rank"] = d.groupby("patient_nbr")["rank"].shift(-1)
    d["gap_rank"] = d["next_rank"] - d["rank"]
    g = d[d["gap_rank"].notna()]

    lt = g.loc[g["readmitted"] == "<30", "gap_rank"].to_numpy()
    gt = g.loc[g["readmitted"] == ">30", "gap_rank"].to_numpy()
    u, p_u = stats.mannwhitneyu(lt, gt, alternative="less")

    # Patient-clustered bootstrap on the median difference.
    sub = g[g["readmitted"].isin(["<30", ">30"])]
    gcodes, _ = pd.factorize(sub["patient_nbr"].to_numpy())
    n_g = int(gcodes.max()) + 1
    gaps = sub["gap_rank"].to_numpy()
    is_lt = (sub["readmitted"] == "<30").to_numpy()
    order_by_group = np.argsort(gcodes, kind="stable")
    starts = np.searchsorted(gcodes[order_by_group], np.arange(n_g))
    ends = np.searchsorted(gcodes[order_by_group], np.arange(n_g), side="right")
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_g, size=n_g)
        idx = np.concatenate([order_by_group[starts[i]:ends[i]] for i in pick])
        gg, ll = gaps[idx], is_lt[idx]
        diffs[b] = (np.median(gg[ll]) - np.median(gg[~ll])
                    if ll.any() and (~ll).any() else np.nan)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])

    # rank -> date map from the anchor points (piecewise linear, extrapolated
    # at the ends using the nearest segment's rate).
    pts = coherence["anchor_points_sorted_by_date"]
    xs = np.array([p["rank"] for p in pts])
    ys = np.array([pd.Timestamp(p["date"]).value / 86_400e9 for p in pts])  # days

    def _to_days(r: np.ndarray) -> np.ndarray:
        return np.interp(r, xs, ys, left=np.nan, right=np.nan) if len(xs) < 2 else \
            np.interp(r, xs, ys) + np.where(
                r < xs[0], (r - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0]), 0.0) + np.where(
                r > xs[-1], (r - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]), 0.0)

    implied = {}
    for lab, arr in [("<30", lt), (">30", gt)]:
        rows = g[g["readmitted"] == lab]
        days = _to_days(rows["next_rank"].to_numpy()) - _to_days(rows["rank"].to_numpy())
        implied[lab] = {
            "n": int(len(days)),
            "median_gap_rank": float(np.median(arr)),
            "median_implied_days": float(np.median(days)),
            "share_implied_le_30_days": float(np.mean(days <= 30)),
            "implied_days_quartiles": [float(np.percentile(days, q)) for q in (25, 50, 75)],
        }

    return {
        "design": ("within-patient rank distance to the next encounter, compared "
                   "between labels whose time semantics are known exactly"),
        "median_gap_rank_lt30": implied["<30"]["median_gap_rank"],
        "median_gap_rank_gt30": implied[">30"]["median_gap_rank"],
        "ratio_gt30_over_lt30": float(implied[">30"]["median_gap_rank"] /
                                      implied["<30"]["median_gap_rank"]),
        "mannwhitney_u": float(u),
        "p_one_sided_lt30_shorter": float(p_u),
        "median_difference_ci95_cluster_bootstrap": [float(lo), float(hi)],
        "implied_calendar_check": implied,
        "expected_under_H1": ("<30 gaps far shorter than >30 gaps, and <30 gaps "
                              "predominantly converting to <= 30 implied days"),
        "caveat": ("the rank->date map is piecewise linear through three anchor "
                   "points (two of which are one-sided bounds), and 28.4% of "
                   "'<30' rows have no in-extract successor at all, so the "
                   "successor observed for those rows is not the readmission "
                   "that generated the label; exact agreement is not expected "
                   "and the direction and order of magnitude are the claim"),
    }


def _timeline_coherence(anchors: dict[str, AnchorResult]) -> dict:
    """
    Joint test the individual anchors cannot do on their own.

    Each anchor independently localises ONE known calendar date to ONE rank
    position. If encounter_id ordering is chronological, those rank positions
    must fall in the same order as the calendar dates — a constraint no single
    anchor imposes and that no panel-composition confound predicts.

    With k anchors, the exact one-sided probability of the correct ordering
    arising by chance is 1/k!. This is weak alone (k=3 -> p=0.167) and is
    reported as CORROBORATING, not as a primary test.

    Also reports the implied encounter-accumulation rate between consecutive
    anchors. Health Facts onboarded hospitals across the decade, so a
    chronological ordering should imply a monotonically INCREASING rate. A
    decreasing or erratic implied rate would be evidence against H1 even with
    every individual anchor significant.
    """
    pts = []
    a1 = anchors.get("A1_troglitazone")
    if a1 and a1.n_positive_rows:
        pts.append({"anchor": "A1_troglitazone", "date": a1.detail["event_date"],
                    "rank": a1.detail["statistic_max_patient_rank"],
                    "bound": "upper (drug withdrawn -> last use at or before this date)"})
    a4 = anchors.get("A4_v85_bmi")
    if a4 and a4.n_positive_rows:
        pts.append({"anchor": "A4_v85_bmi", "date": a4.detail["event_date"],
                    "rank": a4.detail["statistic_min_patient_rank"],
                    "bound": "lower (code introduced -> first use at or after this date)"})
    a2 = anchors.get("A2_tzd_share")
    if a2 and a2.detail:
        pts.append({"anchor": "A2_tzd_share", "date": a2.detail["event_date"],
                    "rank": a2.detail["changepoint"]["estimated_split_rank_fraction"],
                    "bound": "point (prescribing changepoint)"})

    pts = sorted(pts, key=lambda d: d["date"])
    ranks = [p["rank"] for p in pts]
    k = len(ranks)
    ordered = all(ranks[i] < ranks[i + 1] for i in range(k - 1))
    p_order = 1.0 / float(math.factorial(k)) if k > 1 else float("nan")

    # Implied accumulation rate between consecutive anchors (fraction/year),
    # plus the leading segment from the documented window start.
    window_start = pd.Timestamp("1999-01-01")
    segments = []
    prev_rank, prev_date = 0.0, window_start
    for p in pts:
        d = pd.Timestamp(p["date"])
        years = (d - prev_date).days / 365.25
        if years > 0:
            segments.append({
                "from": str(prev_date.date()), "to": p["date"],
                "years": round(years, 3),
                "encounter_fraction": round(p["rank"] - prev_rank, 5),
                "fraction_per_year": round((p["rank"] - prev_rank) / years, 5),
            })
        prev_rank, prev_date = p["rank"], d
    rates = [s["fraction_per_year"] for s in segments]
    increasing = all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))

    return {
        "anchor_points_sorted_by_date": pts,
        "rank_order_matches_date_order": bool(ordered),
        "exact_p_random_ordering_one_sided": p_order,
        "implied_accumulation_segments": segments,
        "implied_rate_monotonically_increasing": bool(increasing),
        "expected_under_H1": ("increasing — the Health Facts panel grew across "
                              "1999-2008, so later calendar years should carry "
                              "more encounters"),
        "role": "CORROBORATING — 1/k! is weak on its own; reported as joint consistency",
    }


# ══════════════════════════════════════════════════════════════════════════
# Block-structure diagnostics (positive evidence for the H0 mechanism)
# ══════════════════════════════════════════════════════════════════════════

def _block_structure(df: pd.DataFrame, n_bins: int) -> dict:
    """
    If encounter_id is a load-batch surrogate key rather than a clock, the
    sequence should show batch fingerprints: large gaps, and abrupt composition
    discontinuities between adjacent rank bins rather than smooth drift.
    """
    enc = np.sort(df["encounter_id"].to_numpy())
    diffs = np.diff(enc)
    med = float(np.median(diffs))
    big = diffs > (1000 * med)
    top_idx = np.argsort(diffs)[-10:][::-1]

    rank = _rank_fraction(df["encounter_id"])
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(rank, edges[1:-1]), 0, n_bins - 1)

    js = {}
    for col in ["medical_specialty", "payer_code", "admission_source_id",
                "discharge_disposition_id"]:
        series = df[col].astype(str)
        dists = [series[b == i].value_counts(normalize=True) for i in range(n_bins)]
        js[col] = [round(_js_divergence(dists[i], dists[i + 1]), 5) for i in range(n_bins - 1)]

    first_enc = df.groupby("patient_nbr")["encounter_id"].min()
    rho_pat, p_pat = stats.spearmanr(first_enc.index.to_numpy(), first_enc.to_numpy())

    return {
        "encounter_id_gaps": {
            "median_gap": med,
            "mean_gap": float(diffs.mean()),
            "p99_gap": float(np.percentile(diffs, 99)),
            "max_gap": float(diffs.max()),
            "n_gaps_gt_1000x_median": int(big.sum()),
            "largest_gap_positions_rank_fraction":
                [round(float((i + 1) / len(enc)), 5) for i in top_idx],
            "largest_gap_sizes": [float(diffs[i]) for i in top_idx],
        },
        "adjacent_bin_js_divergence": js,
        "patient_nbr_vs_first_encounter_id": {
            "spearman_rho": float(rho_pat), "p_value": float(p_pat),
            "note": ("both are Health Facts surrogate keys; a strong positive rho "
                     "shows consistent key-assignment order, which is necessary "
                     "but NOT sufficient for calendar chronology"),
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# Censoring / truncation leg
# ══════════════════════════════════════════════════════════════════════════

def _label_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    r = df["readmitted"].to_numpy()
    return {
        "lt30"  : (r == "<30").astype(float),   # PRIMARY target (agreed switch)
        "gt30"  : (r == ">30").astype(float),
        "merged": (r != "NO").astype(float),    # the original, superseded target
    }


def _censoring_analysis(df: pd.DataFrame, conf: dict, rng: np.random.Generator) -> dict:
    """
    Tests the mechanical alternative to concept drift: observation-window
    truncation correlated with patient entry order.

    Four components:
      (1) entry-rank regressions (the audit's specified diagnostic)
      (2) horizon contrast  — the discriminating test (see docstring below)
      (3) last-observed-encounter effect
      (4) matched-lookahead, stratified
      (5) label-observability audit
    """
    n_boot = conf["n_bootstrap"]
    n_bins = conf["n_bins"]
    labels = _label_columns(df)
    enc_rank = _rank_fraction(df["encounter_id"])
    pats = df["patient_nbr"].to_numpy()

    # ── (1) patient-level entry-rank regressions ──────────────────────────
    g = df.groupby("patient_nbr")
    pat = pd.DataFrame({
        "first_enc": g["encounter_id"].min(),
        "last_enc" : g["encounter_id"].max(),
        "n_enc"    : g["encounter_id"].size(),
    })
    # Span is measured in RANK units, not raw encounter_id units. encounter_id
    # values are unevenly spaced (median gap ~1.6e3, max ~4.5e5), so a raw-id
    # span is dominated by where in id-space a patient sits rather than by how
    # long they were observed. The raw-id span is kept for transparency and
    # explicitly flagged as not interpretable.
    pat["span_encounter_id_units_UNINTERPRETABLE"] = pat["last_enc"] - pat["first_enc"]
    rank_by_pat = pd.Series(enc_rank, index=df.index).groupby(df["patient_nbr"])
    pat["first_rank"] = rank_by_pat.min()
    pat["last_rank"] = rank_by_pat.max()
    pat["span_rank"] = pat["last_rank"] - pat["first_rank"]
    lab_df = pd.DataFrame(labels, index=df.index)
    lab_df["patient_nbr"] = df["patient_nbr"].to_numpy()
    pat = pat.join(lab_df.groupby("patient_nbr")[["lt30", "gt30", "merged"]].mean()
                   .rename(columns=lambda c: f"rate_{c}"))
    first_rows = df.sort_values("encounter_id").groupby("patient_nbr").first()
    pat["first_number_inpatient"] = first_rows["number_inpatient"]
    pat = pat.sort_values("first_enc")
    pat["entry_rank"] = np.arange(1, len(pat) + 1) / len(pat)

    x = pat["entry_rank"].to_numpy()
    groups_pat = pat.index.to_numpy()
    entry_regressions = {}
    for out_col in ["n_enc", "span_rank", "rate_lt30", "rate_merged", "rate_gt30",
                    "first_number_inpatient",
                    "span_encounter_id_units_UNINTERPRETABLE"]:
        y = pat[out_col].to_numpy(dtype=float)
        slope = _ols_slope(x, y)
        lo, hi, _ = _cluster_bootstrap_slope(x, y, groups_pat, n_boot, rng)
        rho, p_rho = stats.spearmanr(x, y)
        base = float(y.mean())
        entry_regressions[out_col] = {
            "slope_per_unit_entry_rank": slope,
            "slope_ci95": [lo, hi],
            "relative_slope_pct_of_mean": float(100 * slope / base) if base else float("nan"),
            "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "mean": base,
            "decile_means": [float(v) for v in
                             pat.groupby(pd.cut(x, np.linspace(0, 1, n_bins + 1),
                                                include_lowest=True), observed=False)[out_col].mean()],
            "declines_with_entry_rank": bool(slope < 0 and hi < 0),
        }

    # ── (2a) label-observability audit — MUST run before the horizon test ─
    # `readmitted` is a Health Facts field, NOT derived by us from the encounter
    # sequence, so it may record readmissions whose encounters are absent from
    # this (diabetes-filtered) extract. The horizon-contrast test below assumes
    # censoring acts through the CALENDAR horizon; that assumption is only valid
    # if the label is sourced from follow-up broader than this extract. This
    # audit decides it, so it is computed first and used to gate the result.
    max_per_pat = df.groupby("patient_nbr")["encounter_id"].transform("max").to_numpy()
    has_successor = (df["encounter_id"].to_numpy() < max_per_pat)
    obs = {}
    for lab in ["<30", ">30", "NO"]:
        m = df["readmitted"].to_numpy() == lab
        obs[lab] = {
            "n_rows": int(m.sum()),
            "share_with_in_dataset_successor": float(has_successor[m].mean()),
        }
    # If "NO" almost never has a successor, the label is effectively an
    # in-extract successor indicator: censoring then acts through SUCCESSOR
    # OBSERVABILITY, which hits <30 and >30 alike, not through calendar horizon.
    label_is_successor_derived = obs["NO"]["share_with_in_dataset_successor"] < 0.05
    label_observability = {
        "by_label": obs,
        "overall_share_with_successor": float(has_successor.mean()),
        "label_is_successor_derived": bool(label_is_successor_derived),
        "finding": ("only "
                    f"{obs['NO']['share_with_in_dataset_successor']*100:.1f}% of 'NO' rows have any "
                    "later encounter in the extract, versus "
                    f"{obs['<30']['share_with_in_dataset_successor']*100:.1f}% of '<30' and "
                    f"{obs['>30']['share_with_in_dataset_successor']*100:.1f}% of '>30' — the label "
                    "is essentially an in-extract successor indicator"),
        "consequence": ("the operative censoring mechanism is successor "
                        "observability, not calendar horizon; the horizon-contrast "
                        "test's premise therefore does not hold and its result is "
                        "reported as UNINFORMATIVE rather than as evidence"),
    }

    # ── (2) horizon contrast ──────────────────────────────────────────────
    # WHY THIS REPLACED THE PLANNED "MATCHED-LOOKAHEAD" TEST AS THE PRIMARY
    # DISCRIMINATOR: with a single collection window, a PATIENT's remaining
    # lookahead is a deterministic decreasing function of entry position, so
    # conditioning on lookahead at patient level is degenerate — the two are
    # perfectly collinear and no comparison is possible.
    #
    # What is NOT degenerate is the LABEL HORIZON. Right-censoring bites over a
    # window equal to the label's lookahead:
    #   <30    -> 30 days  -> only encounters in the final ~month are censored
    #   >30    -> years    -> censoring bites across most of the window
    # So under censoring, the >30 gradient must be much steeper (relative to its
    # base rate) than the <30 gradient. Under a genuine secular decline in
    # readmission risk, both should fall with comparable relative shape.
    horizon = {}
    paired = _cluster_bootstrap_slopes(enc_rank, labels, pats, n_boot, rng)
    rel = {}
    for lname, y in labels.items():
        slope = _ols_slope(enc_rank, y)
        lo, hi = np.nanpercentile(paired[lname], [2.5, 97.5])
        base = float(y.mean())
        horizon[lname] = {
            "base_rate": base,
            "slope_per_unit_encounter_rank": slope,
            "slope_ci95_cluster_bootstrap": [float(lo), float(hi)],
            "relative_slope_pct_of_base": float(100 * slope / base) if base else float("nan"),
            "decile_rates": _decile_rates(enc_rank, y, n_bins).to_dict("records"),
        }
        rel[lname] = paired[lname] / base if base else paired[lname] * np.nan
    diff = rel["gt30"] - rel["lt30"]      # paired: same resample in both terms
    d_lo, d_hi = np.nanpercentile(diff, [2.5, 97.5])
    raw_verdict = ("CENSORING-CONSISTENT" if d_hi < 0 else
                   "TREND-CONSISTENT" if (d_lo < 0 < d_hi) else "UNEXPECTED_SIGN")
    horizon["contrast_gt30_minus_lt30_relative_slope"] = {
        "point": float(np.nanmean(diff)),
        "ci95": [float(d_lo), float(d_hi)],
        "censoring_predicts": "strongly negative (>30 falls much faster in relative terms)",
        "secular_trend_predicts": "approximately zero",
        "premise_valid": not label_is_successor_derived,
        "raw_verdict_if_premise_held": raw_verdict,
        "verdict": ("UNINFORMATIVE_PREMISE_INVALIDATED" if label_is_successor_derived
                    else raw_verdict),
        "why": ("this test assumed censoring acts through the calendar horizon "
                "(30 days vs years). The label-observability audit shows the "
                "label is an in-extract successor indicator, so BOTH labels are "
                "censored by the same successor-observability mechanism and the "
                "horizon length is not the operative variable. The null result "
                "is therefore NOT evidence of a secular trend — the test's "
                "premise failed. The last-encounter analysis below is the "
                "correct discriminator for this label."
                if label_is_successor_derived else "premise holds"),
    }

    # ── (3) last-observed-encounter effect ────────────────────────────────
    # Given (2a), this is the operative mechanism: a row can only carry a
    # positive label if a successor exists, so the rising share of rows that are
    # a patient's FINAL observed encounter mechanically depresses the rate.
    is_last = ~has_successor
    last_slope = _ols_slope(enc_rank, is_last.astype(float))
    l_lo, l_hi, _ = _cluster_bootstrap_slope(enc_rank, is_last.astype(float), pats, n_boot, rng)
    nonlast = ~is_last
    lt30 = labels["lt30"]
    lt30_nonlast_slope = _ols_slope(enc_rank[nonlast], lt30[nonlast])
    nl_lo, nl_hi, _ = _cluster_bootstrap_slope(enc_rank[nonlast], lt30[nonlast],
                                               pats[nonlast], n_boot, rng)
    last_encounter = {
        "share_is_last_overall": float(is_last.mean()),
        "share_is_last_slope_per_unit_rank": last_slope,
        "share_is_last_slope_ci95": [l_lo, l_hi],
        "share_is_last_decile": _decile_rates(enc_rank, is_last.astype(float), n_bins)
                                 .to_dict("records"),
        "lt30_rate_on_last_encounters": float(lt30[is_last].mean()),
        "lt30_rate_on_non_last_encounters": float(lt30[nonlast].mean()),
        "lt30_slope_all_rows": horizon["lt30"]["slope_per_unit_encounter_rank"],
        "lt30_slope_non_last_rows_only": lt30_nonlast_slope,
        "lt30_slope_non_last_ci95": [nl_lo, nl_hi],
        "interpretation_key": ("if the <30 gradient largely disappears once "
                               "final-observed encounters are removed, the "
                               "gradient is an observability artifact"),
    }

    # ── (4) matched-lookahead, stratified ─────────────────────────────────
    # Non-degenerate version: an ENCOUNTER's own lookahead does not determine
    # its patient's ENTRY rank (patients span the window), so within a narrow
    # lookahead band we can still compare early- and late-entering patients.
    # The split under audit is by patient entry rank, so this is the directly
    # relevant conditional test.
    enc_lookahead = 1.0 - enc_rank
    entry_rank_map = pat["entry_rank"]
    row_entry_rank = df["patient_nbr"].map(entry_rank_map).to_numpy(dtype=float)
    k = conf["censoring"]["lookahead_strata"]
    qs = np.quantile(enc_lookahead, np.linspace(0, 1, k + 1))
    strata = []
    for i in range(k):
        upper = (enc_lookahead <= qs[i + 1]) if i == k - 1 else (enc_lookahead < qs[i + 1])
        m = (enc_lookahead >= qs[i]) & upper
        if m.sum() < conf["censoring"]["min_stratum_rows"]:
            continue
        s = _ols_slope(row_entry_rank[m], lt30[m])
        lo, hi, _ = _cluster_bootstrap_slope(row_entry_rank[m], lt30[m], pats[m], n_boot, rng)
        strata.append({
            "stratum": i + 1,
            "lookahead_range": [float(qs[i]), float(qs[i + 1])],
            "n_rows": int(m.sum()),
            "n_patients": int(pd.unique(pats[m]).size),
            "entry_rank_iqr": [float(np.percentile(row_entry_rank[m], 25)),
                               float(np.percentile(row_entry_rank[m], 75))],
            "lt30_rate": float(lt30[m].mean()),
            "lt30_slope_on_entry_rank": s,
            "slope_ci95": [lo, hi],
            "significant": bool(hi < 0 or lo > 0),
        })
    pooled_s = _ols_slope(row_entry_rank, lt30)
    p_lo, p_hi, _ = _cluster_bootstrap_slope(row_entry_rank, lt30, pats, n_boot, rng)
    matched_lookahead = {
        "design_note": ("patient-level matched-lookahead is DEGENERATE (entry "
                        "position determines remaining lookahead exactly under a "
                        "single collection window); this stratifies on the "
                        "ENCOUNTER's own lookahead instead, within which entry "
                        "rank still varies"),
        "unconditional_lt30_slope_on_entry_rank": pooled_s,
        "unconditional_ci95": [p_lo, p_hi],
        "strata": strata,
        "n_strata_with_significant_gradient": int(sum(s["significant"] for s in strata)),
    }

    return {
        "entry_rank_regressions": entry_regressions,
        "censoring_signature_met": bool(all(
            entry_regressions[c]["declines_with_entry_rank"]
            for c in ["n_enc", "span_rank", "rate_merged", "first_number_inpatient"])),
        "horizon_contrast": horizon,
        "last_encounter_effect": last_encounter,
        "matched_lookahead_stratified": matched_lookahead,
        "label_observability": label_observability,
        "patient_frame_for_figures": pat,
    }


# ══════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════

def _fig_a1(res: AnchorResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    r = res.detail.get("positive_rank_fractions", [])
    ax.axhline(0, color="#cbd5e0", lw=1, zorder=0)
    ax.scatter(r, np.zeros(len(r)), s=110, color=C_WARN, zorder=3,
               label=f"troglitazone encounters (n={len(r)})")
    ax.axvspan(0, 0.12, color=C_MAIN, alpha=0.10,
               label="expected region if ordering is chronological")
    ax.set_xlim(-0.01, 1.01)
    ax.set_yticks([])
    ax.set_xlabel("encounter_id rank fraction")
    ax.set_title("A1 — troglitazone extinction anchor (withdrawn 21 Mar 2000)\n"
                 f"exact one-sided p = {res.detail.get('exact_p_one_sided_early', float('nan')):.2e}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_a2(df: pd.DataFrame, rank: np.ndarray, res: AnchorResult,
            n_bins: int, path: Path) -> None:
    neg = "No"
    rosi = df["rosiglitazone"].to_numpy() != neg
    pio  = df["pioglitazone"].to_numpy() != neg
    keep = (rosi | pio) & ~(rosi & pio)
    tab = _decile_rates(rank[keep], rosi[keep].astype(float), n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    ax.errorbar(tab["bin_mid"], tab["rate"],
                yerr=[tab["rate"] - tab["lo"], tab["hi"] - tab["rate"]],
                fmt="o-", color=C_MAIN, capsize=3, lw=1.4, ms=5)
    cp = res.detail["changepoint"]["estimated_split_rank_fraction"]
    ax.axvline(cp, color=C_WARN, ls="--", lw=1.4,
               label=f"estimated changepoint = {cp:.3f}")
    ax.set_xlabel("encounter_id rank fraction (decile midpoint)")
    ax.set_ylabel("rosiglitazone / (rosiglitazone + pioglitazone)")
    ax.set_title("A2 — within-TZD-class share\n"
                 f"max-z = {res.detail['changepoint']['max_z']:.2f}, "
                 f"permutation p = {res.detail['changepoint']['p_permutation']:.4f}")
    ax.legend(fontsize=8)

    ax = axes[1]
    for col, c, lab in [("rosiglitazone", C_WARN, "rosiglitazone"),
                        ("pioglitazone", C_MAIN, "pioglitazone")]:
        y = (df[col].to_numpy() != neg).astype(float)
        t = _decile_rates(rank, y, n_bins)
        ax.plot(t["bin_mid"], t["rate"], "o-", color=c, lw=1.4, ms=4, label=lab)
    ax.axvline(cp, color=C_NULL, ls="--", lw=1.0)
    ax.set_xlabel("encounter_id rank fraction (decile midpoint)")
    ax.set_ylabel("share of all encounters")
    ax.set_title("Marginal rates (confounded by panel composition —\n"
                 "shown for context, not used for inference)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_a3(df: pd.DataFrame, rank: np.ndarray, res: AnchorResult,
            n_bins: int, path: Path) -> None:
    y = (df["A1Cresult"].to_numpy() != "None").astype(float)
    tab = _decile_rates(rank, y, n_bins)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.errorbar(tab["bin_mid"], tab["rate"],
                yerr=[tab["rate"] - tab["lo"], tab["hi"] - tab["rate"]],
                fmt="o-", color=C_MAIN, capsize=3, lw=1.4, ms=5)
    s = res.detail["observed_slope"]
    ax.plot([0, 1], [tab["rate"].mean() - s / 2, tab["rate"].mean() + s / 2],
            color=C_ALT, ls="--", lw=1.2, label=f"fitted slope = {s:+.4f} / unit rank")
    ax.set_xlabel("encounter_id rank fraction (decile midpoint)")
    ax.set_ylabel("share of encounters with HbA1c measured")
    ax.set_title("A3 — HbA1c measurement rate  [SUPPORTING ANCHOR ONLY]\n"
                 f"permutation p = {res.detail['p_permutation']:.4f}   "
                 f"outcome = {res.outcome}")
    ax.legend(fontsize=8, loc="best")
    ax.text(0.5, -0.30,
            "CAVEAT: published rise in A1c testing is AMBULATORY. This variable is "
            "inpatient ordering\nbehaviour during an admission. The transfer is not "
            "established, so this anchor is confirmatory only.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=C_WARN)
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _fig_a4(df: pd.DataFrame, rank: np.ndarray, res: AnchorResult,
            n_bins: int, path: Path) -> None:
    diag = df[["diag_1", "diag_2", "diag_3"]].astype(str)
    mask = diag.apply(lambda s: s.str.startswith("V85")).any(axis=1).to_numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(rank, edges[1:-1]), 0, n_bins - 1)
    counts = [int(mask[b == i].sum()) for i in range(n_bins)]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(np.arange(1, n_bins + 1), counts, color=C_MAIN, width=0.7)
    ax.set_xlabel("encounter_id rank decile")
    ax.set_ylabel("encounters coded V85 (BMI)")
    ax.set_title("A4 — ICD-9 V85 introduction anchor (effective 1 Oct 2005)\n"
                 f"min positive rank = {res.detail.get('statistic_min_patient_rank', float('nan')):.4f}, "
                 f"exact one-sided p = {res.detail.get('exact_p_one_sided_late', float('nan')):.2e}   "
                 f"[{res.verification}, EXPLORATORY]")
    for i, c in enumerate(counts):
        ax.text(i + 1, c, str(c), ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_censoring(pat: pd.DataFrame, cens: dict, n_bins: int, path: Path) -> None:
    x = pat["entry_rank"].to_numpy()
    bins = pd.cut(x, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    panels = [("n_enc", "encounters per patient"),
              ("span_rank", "observed span (rank units)"),
              ("rate_lt30", "P(<30 readmission)   [PRIMARY TARGET]"),
              ("first_number_inpatient", "number_inpatient at first encounter")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for ax, (col, lab) in zip(axes.ravel(), panels):
        m = pat.groupby(bins, observed=False)[col].mean()
        mids = np.linspace(0, 1, n_bins + 1)[:-1] + 0.5 / n_bins
        ax.plot(mids, m.to_numpy(), "o-", color=C_MAIN, lw=1.5, ms=5)
        d = cens["entry_rank_regressions"][col]
        ax.set_title(f"{lab}\nslope {d['slope_per_unit_entry_rank']:+.4g} "
                     f"[{d['slope_ci95'][0]:+.4g}, {d['slope_ci95'][1]:+.4g}]  "
                     f"rho={d['spearman_rho']:+.3f}")
        ax.set_xlabel("patient entry rank (by first encounter_id)")
    if "rate_merged" in cens["entry_rank_regressions"]:
        m2 = pat.groupby(bins, observed=False)["rate_merged"].mean()
        mids = np.linspace(0, 1, n_bins + 1)[:-1] + 0.5 / n_bins
        axes[1][0].plot(mids, m2.to_numpy(), "s--", color=C_WARN, lw=1.3, ms=4,
                        label="merged target (<30 or >30), superseded")
        axes[1][0].legend(fontsize=7.5)
    fig.suptitle("Censoring / truncation signature by patient entry cohort", y=1.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_horizon(cens: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    ax = axes[0]
    for lname, c, lab in [("lt30", C_MAIN, "<30  (30-day horizon)"),
                          ("gt30", C_WARN, ">30  (multi-year horizon)"),
                          ("merged", C_NULL, "merged (superseded)")]:
        t = pd.DataFrame(cens["horizon_contrast"][lname]["decile_rates"])
        base = cens["horizon_contrast"][lname]["base_rate"]
        ax.plot(t["bin_mid"], t["rate"] / base, "o-", color=c, lw=1.5, ms=4, label=lab)
    ax.axhline(1.0, color="#cbd5e0", lw=1)
    ax.set_xlabel("encounter_id rank fraction")
    ax.set_ylabel("readmission rate / overall base rate")
    con = cens["horizon_contrast"]["contrast_gt30_minus_lt30_relative_slope"]
    ok = con["premise_valid"]
    ax.set_title("Horizon contrast\n"
                 f"(>30 − <30) rel. slope = {con['point']:+.3f} "
                 f"[{con['ci95'][0]:+.3f}, {con['ci95'][1]:+.3f}]\n"
                 + ("verdict: " + con["verdict"] if ok else
                    "PREMISE INVALIDATED — not evidence either way"),
                 color=("black" if ok else C_WARN))
    ax.legend(fontsize=8)

    ax = axes[1]
    t = pd.DataFrame(cens["last_encounter_effect"]["share_is_last_decile"])
    ax.plot(t["bin_mid"], t["rate"], "o-", color=C_ALT, lw=1.5, ms=4)
    le = cens["last_encounter_effect"]
    ax.set_xlabel("encounter_id rank fraction")
    ax.set_ylabel("share of rows that are the\npatient's last observed encounter")
    ax.set_title("Last-observed-encounter effect\n"
                 f"<30 slope: all rows {le['lt30_slope_all_rows']:+.4f}  vs  "
                 f"non-last only {le['lt30_slope_non_last_rows_only']:+.4f}\n"
                 "(sign reverses — the gradient is an observability artifact)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _fig_timeline(coh: dict, path: Path) -> None:
    pts = coh["anchor_points_sorted_by_date"]
    dates = [pd.Timestamp(p["date"]) for p in pts]
    ranks = [p["rank"] for p in pts]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(dates, ranks, "o-", color=C_MAIN, lw=1.6, ms=8)
    for p, d, r in zip(pts, dates, ranks):
        ax.annotate(f"{p['anchor']}\n{p['date']}  →  rank {r:.3f}",
                    (d, r), textcoords="offset points", xytext=(10, -14), fontsize=8)
    ax.set_xlim(pd.Timestamp("1999-01-01"), pd.Timestamp("2009-06-30"))
    ax.set_ylim(-0.15, 1.05)
    ax.set_ylabel("encounter_id rank fraction")
    ax.set_xlabel("known calendar date of the external event")
    ax.set_title("Anchor timeline coherence — three independent events, correct order\n"
                 f"rank order matches date order: {coh['rank_order_matches_date_order']}   "
                 f"(p = {coh['exact_p_random_ordering_one_sided']:.3f} under random ordering; "
                 "corroborating)")
    ax.text(0.02, 0.95,
            "implied encounter accumulation (fraction / year):\n" +
            "\n".join(f"  {s['from']} → {s['to']}: {s['fraction_per_year']:.3f}"
                      for s in coh["implied_accumulation_segments"]) +
            f"\nmonotonically increasing: {coh['implied_rate_monotonically_increasing']}",
            transform=ax.transAxes, va="top", fontsize=7.5, color=C_NULL)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_blocks(block: dict, n_bins: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    pos = block["encounter_id_gaps"]["largest_gap_positions_rank_fraction"]
    siz = block["encounter_id_gaps"]["largest_gap_sizes"]
    ax.stem(pos, siz, linefmt="-", markerfmt="o", basefmt=" ")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("encounter_id rank fraction")
    ax.set_ylabel("gap size (encounter_id units, log)")
    ax.set_title("Ten largest gaps in the encounter_id sequence\n"
                 f"median gap = {block['encounter_id_gaps']['median_gap']:.0f}, "
                 f"gaps > 1000x median = {block['encounter_id_gaps']['n_gaps_gt_1000x_median']}")

    ax = axes[1]
    for col, c in zip(block["adjacent_bin_js_divergence"],
                      [C_MAIN, C_WARN, C_ALT, C_NULL]):
        v = block["adjacent_bin_js_divergence"][col]
        ax.plot(np.arange(1, len(v) + 1), v, "o-", color=c, lw=1.3, ms=4, label=col)
    ax.set_xlabel("boundary between adjacent rank deciles")
    ax.set_ylabel("Jensen–Shannon divergence (bits)")
    ax.set_title("Composition change across adjacent deciles\n"
                 "smooth = drift-like; spiky = batch-like")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Phase 0.5 input — language inventory (PRODUCED, NOT APPLIED)
# ══════════════════════════════════════════════════════════════════════════

def _language_inventory(conf: dict) -> dict:
    """
    Inventory every occurrence of unearned temporal language.

    SCOPE BOUNDARY: this produces the inventory only. Applying the rename is
    Phase 0.5's job and is deliberately not done here.
    """
    lc = conf["language_inventory"]
    pats = [p.lower() for p in lc["patterns"]]
    hits, per_file, per_pattern = [], {}, {p: 0 for p in pats}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in lc["scan_extensions"]:
            continue
        rel = path.relative_to(ROOT).as_posix()
        # Prefix-anchored only — see the note in language_audit.py: the substring
        # form made the `data` skip rule also exclude `src/data/`.
        if any(rel == s or rel.startswith(s.rstrip("/") + "/") for s in lc["skip_dirs"]):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            for p in pats:
                if p in low:
                    hits.append({"file": rel, "line": i, "pattern": p,
                                 "text": line.strip()[:200]})
                    per_file[rel] = per_file.get(rel, 0) + 1
                    per_pattern[p] += 1
    return {
        "status": "INVENTORY_ONLY — no file was modified (Phase 0.5 applies it)",
        "n_occurrences": len(hits),
        "n_files": len(per_file),
        "per_pattern": per_pattern,
        "per_file": dict(sorted(per_file.items(), key=lambda kv: -kv[1])),
        "occurrences": hits,
    }


# ══════════════════════════════════════════════════════════════════════════
# Verdict
# ══════════════════════════════════════════════════════════════════════════

def _decide(anchors: dict[str, AnchorResult], fdr: dict, block: dict,
            conf: dict) -> dict:
    """
    Pre-registered decision rule (fixed in the plan before any result was seen).

      SUPPORTED     >=2 of the PRIMARY anchors {A1, A2} pass after FDR, and no
                    primary anchor is significant in the contradictory direction
      NOT_SUPPORTED no primary anchor passes AND (a primary is contradictory OR
                    block structure positively indicates batch ordering)
      INCONCLUSIVE  anything else, including "all null with an MDE larger than
                    the effect the literature predicts"
    """
    primary = [k for k, v in anchors.items() if v.role == "primary"]
    passed = [k for k in primary
              if anchors[k].outcome == "PASS" and fdr.get(k, {}).get("reject", False)]
    contra = [k for k in primary if anchors[k].outcome == "CONTRADICTORY"]

    gaps = block["encounter_id_gaps"]
    batch_like = gaps["n_gaps_gt_1000x_median"] > 0

    if len(passed) >= 2 and not contra:
        verdict = "SUPPORTED"
    elif not passed and (contra or batch_like):
        verdict = "NOT_SUPPORTED"
    elif len(passed) == 1 and not contra:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "ordering_verdict": verdict,
        "primary_anchors": primary,
        "primary_passed": passed,
        "primary_contradictory": contra,
        "supporting_outcome": {k: v.outcome for k, v in anchors.items()
                               if v.role == "supporting"},
        "exploratory_outcome": {k: v.outcome for k, v in anchors.items()
                                if v.role == "exploratory"},
        "batch_structure_indicator": batch_like,
        "decision_rule": _decide.__doc__.strip(),
        "split_validity_verdict": "NOT_TEMPORAL_BY_CONSTRUCTION",
        "split_validity_reason": (
            "The split orders PATIENTS by their first encounter_id, which sorts "
            "patients by entry cohort. Even under perfect chronological ordering "
            "of encounter_id, a patient who entered early and kept visiting is "
            "wholly in train, late encounters included — which is why the split "
            "ranges overlap almost completely. Ordering validity and split "
            "validity are separate questions and this verdict does not depend "
            "on the anchor result."),
    }


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_temporal_validity(config_path: Path = CONFIG_PATH) -> dict:
    """Run the full Phase 0.1 investigation and write all artifacts."""
    conf = _load_config(config_path)
    rng = np.random.default_rng(conf["seed"])
    n_bins = conf["n_bins"]

    logger.info("=" * 78)
    logger.info("DriftSentinel — Phase 0.1: Temporal validity of encounter_id")
    logger.info("=" * 78)
    logger.info(f"Seed {conf['seed']} | {conf['n_permutations']:,} permutations | "
                f"{conf['n_bootstrap']:,} cluster-bootstrap draws | {n_bins} bins")

    df = _load_raw_verbatim()
    rank = _rank_fraction(df["encounter_id"])
    logger.info(f"Patients        : {df['patient_nbr'].nunique():,}")
    logger.info(f"encounter_id    : {df['encounter_id'].min():,} — {df['encounter_id'].max():,}")

    # ── Anchors ───────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("EXTERNAL ANCHORS")
    ac = conf["anchors"]
    anchors: dict[str, AnchorResult] = {}

    anchors["A1_troglitazone"] = _anchor_extinction(
        df, rank, "A1_troglitazone", ac["A1_troglitazone"], conf["alpha"])
    logger.info(f"  A1 troglitazone   n={anchors['A1_troglitazone'].n_positive_rows} rows / "
                f"{anchors['A1_troglitazone'].n_positive_patients} patients  "
                f"p={anchors['A1_troglitazone'].p_raw:.3e}  -> {anchors['A1_troglitazone'].outcome}")

    anchors["A2_tzd_share"] = _anchor_share_changepoint(
        df, rank, "A2_tzd_share", ac["A2_tzd_share"], conf, rng)
    cp = anchors["A2_tzd_share"].detail["changepoint"]
    logger.info(f"  A2 TZD share      split={cp['estimated_split_rank_fraction']:.3f}  "
                f"{cp['share_before']:.3f} -> {cp['share_after']:.3f}  "
                f"delta={cp['delta_share']:+.3f} CI{cp['delta_share_ci95_cluster_bootstrap']}  "
                f"p={cp['p_permutation']:.4f}  -> {anchors['A2_tzd_share'].outcome}")

    anchors["A3_a1c_rate"] = _anchor_monotone_rate(
        df, rank, "A3_a1c_rate", ac["A3_a1c_rate"], conf, rng)
    logger.info(f"  A3 A1c rate       slope={anchors['A3_a1c_rate'].detail['observed_slope']:+.5f} "
                f"CI{anchors['A3_a1c_rate'].detail['slope_ci95_cluster_bootstrap']}  "
                f"p={anchors['A3_a1c_rate'].p_raw:.4f}  -> {anchors['A3_a1c_rate'].outcome} [supporting]")

    anchors["A4_v85_bmi"] = _anchor_introduction(
        df, rank, "A4_v85_bmi", ac["A4_v85_bmi"], conf["alpha"])
    logger.info(f"  A4 V85 BMI        n={anchors['A4_v85_bmi'].n_positive_rows} rows / "
                f"{anchors['A4_v85_bmi'].n_positive_patients} patients  "
                f"min_rank={anchors['A4_v85_bmi'].detail.get('statistic_min_patient_rank', float('nan')):.4f}  "
                f"p={anchors['A4_v85_bmi'].p_raw:.3e}  -> {anchors['A4_v85_bmi'].outcome} [exploratory]")

    minors = {}
    for col in conf["minor_extinction_columns"]:
        cfg = {**ac["A1_troglitazone"], "column": col}
        r = _anchor_extinction(df, rank, col, cfg, conf["alpha"])
        r.role, r.note = "minor", "reported, not weighted — n too small to carry inference"
        minors[col] = asdict(r)
        logger.info(f"  minor {col:<26} n={r.n_positive_rows}  p={r.p_raw:.3e}")

    null_anchor = _null_anchor_check(df, conf["null_anchor_columns"])
    for col, info in null_anchor.items():
        logger.info(f"  null anchor {col:<14} n_levels={info['n_levels']}  "
                    f"testable={info['testable']}")

    # ── FDR across the anchor family ──────────────────────────────────────
    fdr = _bh_fdr({k: v.p_raw for k, v in anchors.items()}, conf["fdr_q"])
    logger.info("-" * 60)
    logger.info(f"BH-FDR at q={conf['fdr_q']}")
    for k, v in fdr.items():
        logger.info(f"  {k:<20} p_raw={v['p_raw']:.4e}  p_adj={v['p_adj']:.4e}  "
                    f"reject={v['reject']}")

    # ── Joint timeline coherence ──────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("ANCHOR TIMELINE COHERENCE")
    coherence = _timeline_coherence(anchors)
    for p in coherence["anchor_points_sorted_by_date"]:
        logger.info(f"  {p['date']}  {p['anchor']:<18} -> rank {p['rank']:.4f}   [{p['bound']}]")
    logger.info(f"  rank order matches date order : {coherence['rank_order_matches_date_order']} "
                f"(p={coherence['exact_p_random_ordering_one_sided']:.4f}, corroborating)")
    for s in coherence["implied_accumulation_segments"]:
        logger.info(f"  {s['from']} -> {s['to']}  {s['years']:.2f}y  "
                    f"{s['encounter_fraction']:.4f} of encounters  "
                    f"({s['fraction_per_year']:.4f}/yr)")
    logger.info(f"  implied rate monotonically increasing : "
                f"{coherence['implied_rate_monotonically_increasing']}")

    logger.info("-" * 60)
    logger.info("INTERNAL LABEL-INTERVAL COHERENCE (no external literature)")
    lic = _label_interval_coherence(df, rank, coherence, conf["n_bootstrap"], rng)
    logger.info(f"  median rank gap to next encounter: <30 = {lic['median_gap_rank_lt30']:.5f}  "
                f">30 = {lic['median_gap_rank_gt30']:.5f}  "
                f"(ratio {lic['ratio_gt30_over_lt30']:.2f}x)")
    logger.info(f"  Mann-Whitney one-sided p = {lic['p_one_sided_lt30_shorter']:.3e}; "
                f"median diff CI{[round(v, 5) for v in lic['median_difference_ci95_cluster_bootstrap']]}")
    for lab, v in lic["implied_calendar_check"].items():
        logger.info(f"  implied days for '{lab}': median {v['median_implied_days']:.1f}d, "
                    f"share <= 30d = {v['share_implied_le_30_days']:.3f}  (n={v['n']:,})")

    # ── Block structure ───────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("BLOCK-STRUCTURE DIAGNOSTICS")
    block = _block_structure(df, n_bins)
    g = block["encounter_id_gaps"]
    logger.info(f"  median gap {g['median_gap']:.0f} | max gap {g['max_gap']:,.0f} | "
                f"gaps>1000x median: {g['n_gaps_gt_1000x_median']}")
    logger.info(f"  spearman(patient_nbr, first encounter_id) = "
                f"{block['patient_nbr_vs_first_encounter_id']['spearman_rho']:+.4f}")

    # ── Censoring leg ─────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("CENSORING / TRUNCATION LEG")
    cens = _censoring_analysis(df, conf, rng)
    pat = cens.pop("patient_frame_for_figures")
    for k, v in cens["entry_rank_regressions"].items():
        logger.info(f"  {k:<24} slope={v['slope_per_unit_entry_rank']:+.5g} "
                    f"CI[{v['slope_ci95'][0]:+.5g},{v['slope_ci95'][1]:+.5g}] "
                    f"rho={v['spearman_rho']:+.3f} decline={v['declines_with_entry_rank']}")
    con = cens["horizon_contrast"]["contrast_gt30_minus_lt30_relative_slope"]
    logger.info(f"  horizon contrast (>30 − <30 relative slope) = {con['point']:+.4f} "
                f"CI[{con['ci95'][0]:+.4f},{con['ci95'][1]:+.4f}] -> {con['verdict']}")
    le = cens["last_encounter_effect"]
    logger.info(f"  last-encounter share slope = {le['share_is_last_slope_per_unit_rank']:+.5f}; "
                f"<30 slope all={le['lt30_slope_all_rows']:+.5f} "
                f"non-last={le['lt30_slope_non_last_rows_only']:+.5f}")
    lo = cens["label_observability"]["by_label"]
    logger.info(f"  label observability: <30 with in-dataset successor = "
                f"{lo['<30']['share_with_in_dataset_successor']:.3f}, "
                f">30 = {lo['>30']['share_with_in_dataset_successor']:.3f}, "
                f"NO = {lo['NO']['share_with_in_dataset_successor']:.3f}")

    # ── Verdict ───────────────────────────────────────────────────────────
    verdict = _decide(anchors, fdr, block, conf)
    logger.info("-" * 60)
    logger.info(f"ORDERING VERDICT : {verdict['ordering_verdict']}")
    logger.info(f"SPLIT VALIDITY   : {verdict['split_validity_verdict']}")

    # ── Figures ───────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("FIGURES")
    figs = {
        "A1": FIGURE_DIR / "32_anchor_A1_troglitazone.png",
        "A2": FIGURE_DIR / "33_anchor_A2_tzd_share.png",
        "A3": FIGURE_DIR / "34_anchor_A3_a1c_rate.png",
        "A4": FIGURE_DIR / "35_anchor_A4_v85_bmi.png",
        "censoring": FIGURE_DIR / "36_censoring_entry_cohort.png",
        "horizon": FIGURE_DIR / "37_horizon_contrast_last_encounter.png",
        "blocks": FIGURE_DIR / "38_block_structure.png",
        "timeline": FIGURE_DIR / "39_anchor_timeline_coherence.png",
    }
    _fig_a1(anchors["A1_troglitazone"], figs["A1"])
    _fig_a2(df, rank, anchors["A2_tzd_share"], n_bins, figs["A2"])
    _fig_a3(df, rank, anchors["A3_a1c_rate"], n_bins, figs["A3"])
    _fig_a4(df, rank, anchors["A4_v85_bmi"], n_bins, figs["A4"])
    _fig_censoring(pat, cens, n_bins, figs["censoring"])
    _fig_horizon(cens, figs["horizon"])
    _fig_blocks(block, n_bins, figs["blocks"])
    _fig_timeline(coherence, figs["timeline"])
    for k, v in figs.items():
        logger.info(f"  {k:<10} -> {v.name}")

    # ── Phase 0.5 inventory (produced, not applied) ───────────────────────
    inv = _language_inventory(conf)
    inv_path = REPORTS_DIR / "temporal_language_inventory.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)
    logger.info("-" * 60)
    logger.info(f"Phase 0.5 inventory: {inv['n_occurrences']} occurrences in "
                f"{inv['n_files']} files -> {inv_path.name}  (NOT applied)")

    # ── Report ────────────────────────────────────────────────────────────
    report = {
        "phase": "0.1",
        "title": "Temporal validity of encounter_id ordering",
        "verdict": verdict,
        "dataset": {
            "rows": int(len(df)),
            "patients": int(df["patient_nbr"].nunique()),
            "encounter_id_min": int(df["encounter_id"].min()),
            "encounter_id_max": int(df["encounter_id"].max()),
            "primary_target": "readmitted == '<30' (30-day readmission)",
            "primary_target_prevalence": float((df["readmitted"] == "<30").mean()),
            "merged_target_prevalence_superseded": float((df["readmitted"] != "NO").mean()),
        },
        "anchors": {k: asdict(v) for k, v in anchors.items()},
        "minor_extinction_anchors": minors,
        "null_anchors": null_anchor,
        "multiple_testing": {"method": "Benjamini-Hochberg",
                             "q": conf["fdr_q"], "results": fdr},
        "timeline_coherence": coherence,
        "label_interval_coherence": lic,
        "block_structure": block,
        "censoring": cens,
        "figures": {k: str(v.relative_to(ROOT).as_posix()) for k, v in figs.items()},
        "phase_0_5_inventory": {"path": inv_path.relative_to(ROOT).as_posix(),
                                "n_occurrences": inv["n_occurrences"],
                                "n_files": inv["n_files"],
                                "applied": False},
        "dropped_anchors": conf["dropped_anchors"],
        "reproducibility": {
            "seed": conf["seed"],
            "n_permutations": conf["n_permutations"],
            "n_bootstrap": conf["n_bootstrap"],
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "caveats": [
            "Health Facts onboarded client hospitals across 1999-2008 and the UCI "
            "release DROPS the hospital identifier, so rank-position trends "
            "conflate calendar time with panel composition and cannot be "
            "stratified. A2's within-class share design is the mitigation; the "
            "block-structure diagnostics are the detection attempt.",
            "Permutation p-values are row-level and therefore anti-conservative "
            "under within-patient clustering; cluster-bootstrap CIs are reported "
            "alongside and no conclusion is drawn unless both agree.",
            "A1/A4 are one-sided existence tests. A non-significant A1 is not "
            "evidence against chronological ordering.",
            "A4's effective date is SECONDARY_VERIFIED only (the primary NCHS "
            "addenda PDF was not machine-readable here), so A4 is excluded from "
            "the decision rule.",
            "The changepoint location in A2 is a point estimate; the reported CI "
            "is for the step size at that fixed location and does not include "
            "changepoint-selection uncertainty.",
        ],
    }

    out_path = REPORTS_DIR / "temporal_validity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report written  : {out_path}")
    logger.info("=" * 78)
    logger.info("Phase 0.1 complete")
    logger.info("=" * 78)
    return report


if __name__ == "__main__":
    rep = run_temporal_validity()
    v = rep["verdict"]
    print(f"\nOrdering verdict : {v['ordering_verdict']}")
    print(f"Primary passed   : {v['primary_passed']}")
    print(f"Split validity   : {v['split_validity_verdict']}")
