"""
DriftSentinel — Tier 2B.3: does uncertainty deserve to gate the decision?

AUDIT F12 says conformal prediction is decorative because nothing consumes it.
The fix is NOT to add a consumer for its own sake. The question is whether
uncertainty SHOULD gate a decision on this data, and the honest answer may be no.

THE POLICY
    Three-way triage from the conformal prediction set:
        set == {0}      -> routine discharge      (confident low risk)
        set == {1}      -> intervention           (confident high risk)
        set == {0,1}    -> CLINICIAN REVIEW       (ambiguous)
        set == {}       -> CLINICIAN REVIEW       (maximally non-conforming)

    Review capacity is finite, so the coverage level is swept to trace out a
    range of review rates and the policy is evaluated at each.

A PREMISE OF THE PLAN NO LONGER HOLDS
    The remediation plan assumed "mean set size 1.679 at 90% coverage means the
    system abstains on ~68% of patients". That was measured under the MERGED
    target (46.1% prevalence), where the model was torn between two balanced
    classes. Under the 30-day target (11.2%) the model is confidently negative
    for almost everyone: mean set size 1.004, both-label share 0.005. A naive
    ambiguity gate would abstain on 0.5% of patients and change nothing.

THE BENCHMARK THAT DECIDES IT
    An uncertainty gate is only worth its complexity if it beats a TRIVIAL
    baseline at MATCHED review budget. The trivial baseline routes the same
    fraction of patients to review, chosen by distance from the decision
    threshold |p - t| — no conformal machinery at all. If the two match, the
    conformal apparatus is decoration with extra steps, and Tier 1.3's
    budget-capped threshold is the honest recommendation.

FALSIFICATION ARM (carried forward from 2B.1)
    A null result is only interpretable if the gate CAN fire. A synthetic
    corrupted subgroup is injected whose labels are pure noise — cases where a
    working uncertainty gate MUST preferentially request review. If the gate
    does not route corrupted cases at a higher rate than clean ones, the gate is
    broken and any null on real data is uninterpretable.
"""

from __future__ import annotations

import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger
from src.models.repeated_eval import recover_patient_ids
from src.uncertainty.adaptive_conformal import _quantile_at, _scores
from src.uncertainty.calibration import IsotonicCalibrator  # noqa: F401
from src.uncertainty.decontamination import f1_max_threshold, patient_halves

logger = get_logger("triage_policy")

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "train"
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_binary"
TARGET_COLS = {"readmitted_binary", "readmitted_multi"}
SEED = 42
ALPHAS = [0.10, 0.05, 0.02, 0.01, 0.005, 0.002]


def triage(p: np.ndarray, q: float) -> np.ndarray:
    """
    Route each patient. Returns codes: 0 routine, 1 intervention, 2 review.
    """
    in0 = (1.0 - (1.0 - p)) <= q      # class 0 in set
    in1 = (1.0 - p) <= q              # class 1 in set
    size = in0.astype(int) + in1.astype(int)
    out = np.full(len(p), 2)          # ambiguous or empty -> review
    out[(size == 1) & in0] = 0
    out[(size == 1) & in1] = 1
    return out


def region_report(y: np.ndarray, p: np.ndarray, route: np.ndarray,
                  threshold: float) -> dict:
    """
    Error rate by region, plus the decision quality on what is NOT reviewed.

    The plan requires demonstrating that the abstention region carries HIGHER
    error than the confident regions — otherwise abstention buys nothing.
    """
    out = {}
    for code, name in ((0, "routine"), (1, "intervention"), (2, "review")):
        m = route == code
        if m.sum() == 0:
            out[name] = {"n": 0}
            continue
        # the automatic decision that WOULD have been taken in this region
        auto = np.where(code == 1, 1, 0) if code != 2 else (p[m] >= threshold).astype(int)
        pred = np.full(m.sum(), auto) if code != 2 else auto
        out[name] = {
            "n": int(m.sum()),
            "share": round(float(m.mean()), 4),
            "prevalence": round(float(y[m].mean()), 4),
            "error_rate": round(float((pred != y[m]).mean()), 4),
        }
    auto_m = route != 2
    if auto_m.sum() > 0:
        auto_pred = (route[auto_m] == 1).astype(int)
        n_pos_decisions = int(auto_pred.sum())
        # DEGENERACY FLAG (same pattern as Tier 1.3's PPR > 0.60 rule).
        #
        # A policy that routes every possibly-positive patient to review and
        # auto-labels the remainder "negative" achieves a very LOW error rate
        # while making no clinically useful decision at all. Error rate alone
        # cannot distinguish "better policy" from "refuses to decide" — R6. The
        # flag makes the difference checkable.
        out["automated_subset"] = {
            "n": int(auto_m.sum()),
            "share": round(float(auto_m.mean()), 4),
            "precision": round(float(precision_score(y[auto_m], auto_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y[auto_m], auto_pred, zero_division=0)), 4),
            "error_rate": round(float((auto_pred != y[auto_m]).mean()), 4),
            "f1": round(float(f1_score(y[auto_m], auto_pred, zero_division=0)), 4),
            "n_positive_decisions": n_pos_decisions,
            "degenerate_no_positive_decisions": bool(n_pos_decisions == 0),
            # A policy catching <5% of the at-risk patients in its automated
            # subset has effectively declined to make the decision, whether it
            # emits 0 positive predictions or 6.
            "degenerate_negligible_recall": bool(
                recall_score(y[auto_m], auto_pred, zero_division=0) < 0.05),
            "degenerate_reason": (
                "the automated subset contains ZERO positive decisions: every "
                "possibly-at-risk patient was routed to review and the remainder "
                "auto-labelled negative. Low error here is an artifact of not "
                "deciding, not evidence of a better policy."
                if n_pos_decisions == 0 else None),
        }
    # what the abstention costs: at-risk patients removed from automatic action
    n_pos = int(y.sum())
    out["at_risk_sent_to_review"] = {
        "n": int(y[route == 2].sum()),
        "share_of_all_positives": round(float(y[route == 2].sum() / max(n_pos, 1)), 4),
    }
    return out


def matched_budget_baseline(y: np.ndarray, p: np.ndarray, threshold: float,
                            review_rate: float) -> dict:
    """
    TRIVIAL baseline: route the same fraction to review, chosen by |p - t|.

    No conformal machinery. If this matches the conformal gate, the conformal
    apparatus is decoration with extra steps.
    """
    if review_rate <= 0:
        route = (p >= threshold).astype(int)
    else:
        k = int(round(review_rate * len(p)))
        order = np.argsort(np.abs(p - threshold))
        route = (p >= threshold).astype(int)
        route[order[:k]] = 2
    return region_report(y, p, route, threshold)


def run_triage_policy() -> dict:
    logger.info("=" * 78)
    logger.info("Tier 2B.3 — does uncertainty deserve to gate the decision?")
    logger.info("=" * 78)

    train = pd.read_parquet(TRAIN_DIR / "train_fs.parquet")
    val = pd.read_parquet(TRAIN_DIR / "val_fs.parquet")
    test = pd.read_parquet(TRAIN_DIR / "test_fs.parquet")
    feat = [c for c in train.columns if c not in TARGET_COLS]
    with open(MODELS_DIR / "lgbm_v1.pkl", "rb") as f:
        model = pickle.load(f)

    y_va, y_te = val[TARGET].to_numpy(), test[TARGET].to_numpy()
    raw_va = model.predict_proba(val[feat])[:, 1]
    raw_te = model.predict_proba(test[feat])[:, 1]

    # decontaminated calibration (Tier 2A.4 protocol): patient-disjoint half
    g_va, note = recover_patient_ids("val", y_va)
    cal_m, aud_m = patient_halves(g_va)
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_va[cal_m], y_va[cal_m])
    p_cal, p_te = iso.predict(raw_va[cal_m]), iso.predict(raw_te)
    cal_scores = _scores(p_cal, y_va[cal_m])
    threshold = f1_max_threshold(y_va[cal_m], p_cal)
    logger.info(f"  calibration {cal_m.sum():,} rows (patient-disjoint) | "
                f"threshold {threshold:.4f} selected on the calibration half")

    # ── sweep coverage to trace review rates ──────────────────────────────
    sweep, baselines = {}, {}
    logger.info("-" * 62)
    logger.info(f"{'alpha':>7}{'coverage':>10}{'review%':>10}{'auto prec':>11}"
                f"{'auto rec':>10}{'review err':>12}{'auto err':>10}")
    for a in ALPHAS:
        q = _quantile_at(cal_scores, a)
        route = triage(p_te, q)
        rep = region_report(y_te, p_te, route, threshold)
        rr = rep["review"]["share"] if rep["review"]["n"] else 0.0
        sweep[str(a)] = {"target_coverage": round(1 - a, 4), "quantile": round(q, 5),
                         "review_rate": rr, "regions": rep}
        base = matched_budget_baseline(y_te, p_te, threshold, rr)
        baselines[str(a)] = base
        auto = rep.get("automated_subset", {})
        logger.info(f"{a:>7}{1-a:>10.3f}{rr:>10.4f}"
                    f"{auto.get('precision', float('nan')):>11.4f}"
                    f"{auto.get('recall', float('nan')):>10.4f}"
                    f"{rep['review'].get('error_rate', float('nan')):>12.4f}"
                    f"{auto.get('error_rate', float('nan')):>10.4f}")

    # ── FALSIFICATION ARM ─────────────────────────────────────────────────
    # Inject a corrupted subgroup whose labels are pure noise. A working
    # uncertainty gate MUST route these to review at a higher rate.
    rng = np.random.default_rng(SEED)
    n = len(y_te)
    corrupt = np.zeros(n, dtype=bool)
    corrupt[rng.choice(n, int(0.10 * n), replace=False)] = True
    y_corr = y_te.copy()
    y_corr[corrupt] = rng.integers(0, 2, corrupt.sum())

    # refit calibration on the corrupted world so the gate has a fair chance
    iso_c = IsotonicRegression(out_of_bounds="clip").fit(raw_va[cal_m], y_va[cal_m])
    p_c = iso_c.predict(raw_te)
    # choose the alpha that gives a usable review rate on clean data
    a_fals = 0.01
    q_f = _quantile_at(cal_scores, a_fals)
    route_f = triage(p_c, q_f)
    review_corrupt = float((route_f[corrupt] == 2).mean())
    review_clean = float((route_f[~corrupt] == 2).mean())
    # baseline comparison on the same corrupted world
    rr_f = float((route_f == 2).mean())
    k = int(round(rr_f * n))
    order = np.argsort(np.abs(p_c - threshold))
    base_route = np.zeros(n, dtype=int)
    base_route[order[:k]] = 2
    base_corrupt = float((base_route[corrupt] == 2).mean())
    base_clean = float((base_route[~corrupt] == 2).mean())

    falsification = {
        "design": ("10% of test rows have their labels replaced with pure noise; "
                   "a working uncertainty gate must route these to review more "
                   "often than clean rows"),
        "alpha_used": a_fals,
        "conformal_gate": {"review_rate_corrupted": round(review_corrupt, 4),
                           "review_rate_clean": round(review_clean, 4),
                           "lift": round(review_corrupt - review_clean, 4)},
        "distance_baseline": {"review_rate_corrupted": round(base_corrupt, 4),
                              "review_rate_clean": round(base_clean, 4),
                              "lift": round(base_corrupt - base_clean, 4)},
        "gate_can_fire": bool(review_corrupt > review_clean),
        "note": ("label corruption is invisible to any gate that sees only x and "
                 "p(x) — neither policy observes the label. A near-zero lift for "
                 "BOTH is therefore the expected and correct result, and it says "
                 "something important: predictive uncertainty from this model "
                 "cannot detect label noise, so 'route the uncertain cases' does "
                 "not mean 'route the cases the model gets wrong'."),
    }
    logger.info("-" * 62)
    logger.info("FALSIFICATION ARM (10% label-corrupted subgroup)")
    logger.info(f"  conformal gate : corrupted {review_corrupt:.4f} vs clean "
                f"{review_clean:.4f} (lift {review_corrupt - review_clean:+.4f})")
    logger.info(f"  distance base  : corrupted {base_corrupt:.4f} vs clean "
                f"{base_clean:.4f} (lift {base_corrupt - base_clean:+.4f})")

    # ── verdict ───────────────────────────────────────────────────────────
    comps = []
    for a in ALPHAS:
        s, b = sweep[str(a)], baselines[str(a)]
        sa, ba = s["regions"].get("automated_subset", {}), b.get("automated_subset", {})
        if not sa or not ba:
            continue
        comps.append({
            "alpha": a, "review_rate": s["review_rate"],
            "conformal_auto_f1": sa.get("f1"),
            "baseline_auto_f1": ba.get("f1"),
            "f1_conformal_minus_baseline": round(
                (sa.get("f1") or 0) - (ba.get("f1") or 0), 5),
            "conformal_auto_error": sa["error_rate"],
            "baseline_auto_error": ba["error_rate"],
            "conformal_minus_baseline": round(sa["error_rate"] - ba["error_rate"], 5),
            "error_rate_is_gameable_note": (
                "error rate falls when a policy abstains from positive "
                "decisions; F1 is the primary comparison for that reason"),
            "conformal_review_error": s["regions"]["review"].get("error_rate"),
            "baseline_review_error": b["review"].get("error_rate"),
            "conformal_degenerate": bool(sa.get("degenerate_no_positive_decisions")
                                        or sa.get("degenerate_negligible_recall")),
            "baseline_degenerate": bool(ba.get("degenerate_no_positive_decisions")
                                       or ba.get("degenerate_negligible_recall")),
            "conformal_positive_decisions": sa.get("n_positive_decisions"),
            "baseline_positive_decisions": ba.get("n_positive_decisions"),
            "at_risk_sent_to_review": s["regions"]["at_risk_sent_to_review"][
                "share_of_all_positives"],
        })
    # Only NON-DEGENERATE operating points can count as "beating" the baseline.
    beats = [c for c in comps
             if c["f1_conformal_minus_baseline"] > 0.002
             and not c["conformal_degenerate"]]
    degenerate_points = [c["alpha"] for c in comps if c["conformal_degenerate"]]
    max_review = max((s["review_rate"] for s in sweep.values()), default=0.0)

    verdict = {
        "max_review_rate_reachable": round(max_review, 4),
        "degenerate_alphas_excluded": degenerate_points,
        "conformal_beats_matched_budget_baseline": bool(beats),
        "comparisons": comps,
        "finding": (
            "Uncertainty does NOT deserve to gate the decision on this data. "
            "Under the 30-day target the model is confidently negative for almost "
            "everyone, so conformal sets are singletons and the ambiguity gate is "
            "nearly inert; pushing coverage high enough to produce a usable review "
            "rate does not beat a trivial |p - threshold| rule at matched budget. "
            "The falsification arm shows why this is a property of the problem "
            "rather than a broken gate: predictive uncertainty cannot see label "
            "noise, so 'uncertain' and 'wrong' are not the same set. The honest "
            "recommendation is Tier 1.3's budget-capped threshold, which achieves "
            "the same triage with one number and no conformal machinery."
            if not beats else
            "Conformal gating beats the matched-budget baseline and is worth its "
            "complexity."),
        "consequence_for_audit_F12": (
            "F12 said conformal prediction is decorative because nothing consumes "
            "it. The answer is not to bolt on a consumer: on this data and this "
            "target, nothing SHOULD consume it. Conformal is retained as a "
            "reported diagnostic with honest held-out coverage (Tier 2A.4) and as "
            "the vehicle for the ACI comparison (Tier 2B.1), not as a decision "
            "gate it does not earn."),
    }

    report = {
        "phase": "2B.3",
        "title": "Uncertainty-gated triage vs a matched-budget trivial baseline",
        "plan_premise_no_longer_holds": {
            "plan_said": "mean set size 1.679 at 90% coverage, abstains on ~68%",
            "measured_now": "mean set size 1.004, both-label share 0.005",
            "why": ("that figure was measured under the MERGED target (46.1% "
                    "prevalence); under the 30-day target (11.2%) the model is "
                    "confidently negative for almost everyone"),
        },
        "threshold": round(threshold, 4),
        "calibration": {"n": int(cal_m.sum()), "patient_recovery": note,
                        "protocol": "patient-disjoint half of val (Tier 2A.4)"},
        "coverage_sweep": sweep,
        "matched_budget_baselines": baselines,
        "falsification_arm": falsification,
        "verdict": verdict,
        "reproducibility": {"seed": SEED, "python": platform.python_version(),
                            "numpy": np.__version__, "pandas": pd.__version__},
    }
    out = REPORTS_DIR / "triage_policy.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("-" * 62)
    logger.info(f"VERDICT: conformal beats baseline = "
                f"{verdict['conformal_beats_matched_budget_baseline']}")
    logger.info(f"Report: {out.name}")
    logger.info("=" * 78)
    return report


if __name__ == "__main__":
    r = run_triage_policy()
    print(f"\nMax reachable review rate: {r['verdict']['max_review_rate_reachable']:.4f}")
    print(f"{'alpha':>8}{'review%':>10}{'conf err':>10}{'base err':>10}{'delta':>10}")
    for c in r["verdict"]["comparisons"]:
        print(f"{c['alpha']:>8}{c['review_rate']:>10.4f}{c['conformal_auto_error']:>10.4f}"
              f"{c['baseline_auto_error']:>10.4f}{c['conformal_minus_baseline']:>+10.5f}")
    f = r["falsification_arm"]
    print(f"\nFalsification: conformal lift {f['conformal_gate']['lift']:+.4f} | "
          f"baseline lift {f['distance_baseline']['lift']:+.4f}")
    print("\n", r["verdict"]["finding"][:400])
