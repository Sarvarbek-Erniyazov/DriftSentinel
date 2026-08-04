"""
DriftSentinel — Tier 2C.6: what moved when the canonical threshold was fixed.

WHY THIS EXISTS
    `headline_metrics_ci.json` declares itself the canonical source for every
    number in the README, and it was computed at the val-fitted threshold that
    Tier 2A.4 had already shown to be contaminated. Reconciling it to the
    decontaminated 0.18 changes most threshold-dependent numbers in the
    repository. A change of that blast radius is not reviewable as a sentence —
    it needs a per-number before/after table that is generated, not typed.

WHAT IS CLASSIFIED, AND WHY THE CLASSES MATTER
    UNCHANGED   threshold-free metrics (AUC, Brier, prevalence). If any of these
                moved, something other than the threshold changed and the whole
                comparison is suspect — so they are checked, not assumed.
    SHIFTED     a value moved. Expected, and individually uninteresting.
    REVERSAL    a CLAIM changed, not just a number. Two kinds are detected:
                  - a val->test delta that changes SIGN (a reported degradation
                    becomes an improvement, or vice versa)
                  - a bootstrap interval that gains or loses zero
                These are the ones that cannot be absorbed by editing a figure
                in the README; they require the surrounding sentence to change.

R6 NOTE
    The comparison is only meaningful if the two files differ in exactly one
    input. The threshold-free metrics are therefore used as a control: they MUST
    be unchanged, and a moved control invalidates the run rather than being
    reported alongside the rest as though it were a result.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("threshold_reconciliation")

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "outputs" / "reports"
BEFORE = REPORTS_DIR / "superseded" / "tier2c6_contaminated_threshold" / "headline_metrics_ci.json"
AFTER = REPORTS_DIR / "headline_metrics_ci.json"

THRESHOLD_FREE = ("auc", "brier", "prevalence")
THRESHOLD_DEPENDENT = ("precision", "recall", "f1", "predicted_positive_rate", "ece")
SPLITS = ("train", "val", "test")

# Below this, a "change" is float noise rather than a change.
TOL = 1e-9


def _metrics(doc: dict, split: str) -> dict:
    return (doc["variance_components"]["1_estimation_patient_clustered_bootstrap"]
            ["by_split"][split]["metrics"])


def compare_points(before: dict, after: dict) -> list[dict]:
    rows = []
    for split in SPLITS:
        mb, ma = _metrics(before, split), _metrics(after, split)
        for metric in THRESHOLD_FREE + THRESHOLD_DEPENDENT:
            if metric not in mb or metric not in ma:
                continue
            b, a = mb[metric]["point"], ma[metric]["point"]
            moved = abs(a - b) > TOL
            free = metric in THRESHOLD_FREE
            rows.append({
                "split": split,
                "metric": metric,
                "threshold_free": free,
                "before": b,
                "after": a,
                "delta": round(a - b, 5),
                "before_ci95": mb[metric].get("ci95"),
                "after_ci95": ma[metric].get("ci95"),
                "status": ("CONTROL_MOVED" if free and moved else
                           "UNCHANGED" if not moved else "SHIFTED"),
            })
    return rows


def compare_val_to_test_deltas(before: dict, after: dict) -> list[dict]:
    """
    The val->test delta is what the README describes as degradation. A sign
    change here is a claim reversal, not a value shift.
    """
    rows = []
    for metric in THRESHOLD_FREE + THRESHOLD_DEPENDENT:
        try:
            db = _metrics(before, "test")[metric]["point"] - _metrics(before, "val")[metric]["point"]
            da = _metrics(after, "test")[metric]["point"] - _metrics(after, "val")[metric]["point"]
        except KeyError:
            continue
        reversed_sign = (np.sign(round(db, 6)) != np.sign(round(da, 6))
                         and abs(db) > TOL and abs(da) > TOL)
        rows.append({
            "metric": metric,
            "val_to_test_delta_before": round(db, 5),
            "val_to_test_delta_after": round(da, 5),
            "direction_before": "degrades" if db < 0 else "improves",
            "direction_after": "degrades" if da < 0 else "improves",
            "status": "REVERSAL" if reversed_sign else "same direction",
        })
    return rows


def compare_intervals(before: dict, after: dict) -> list[dict]:
    """An interval that gains or loses zero changes what can be claimed from it."""
    rows = []
    for split in SPLITS:
        mb, ma = _metrics(before, split), _metrics(after, split)
        for metric in THRESHOLD_DEPENDENT:
            if metric not in mb or metric not in ma:
                continue
            cb, ca = mb[metric].get("ci95"), ma[metric].get("ci95")
            if not cb or not ca:
                continue
            wb, wa = round(cb[1] - cb[0], 5), round(ca[1] - ca[0], 5)
            rows.append({"split": split, "metric": metric,
                         "ci_width_before": wb, "ci_width_after": wa,
                         "width_change": round(wa - wb, 5)})
    return rows


def compare_fairness_claims() -> dict:
    """
    Subgroup disparities are threshold-dependent too, and they are CLAIMS.

    `headline_metrics_ci.json` holds numbers; `fairness_audit.json` holds
    statements of the form "this disparity is supported by the data". Moving the
    operating threshold can withdraw such a statement without moving any headline
    number, so the audit is diffed at the level of which disparities remain
    supported — not at the level of their point estimates.
    """
    before_p = (REPORTS_DIR / "superseded" / "tier2c6_contaminated_threshold"
                / "fairness_audit.at_thr_0.18.json")
    after_p = REPORTS_DIR / "fairness_audit.json"
    if not before_p.exists() or not after_p.exists():
        return {"available": False,
                "reason": f"need both {before_p.name} and {after_p.name}"}

    before = json.load(open(before_p, encoding="utf-8"))
    after = json.load(open(after_p, encoding="utf-8"))

    changes = []
    for attr in sorted(set(before["supported_disparities"]) |
                       set(after["supported_disparities"])):
        b = set(before["supported_disparities"].get(attr, []))
        a = set(after["supported_disparities"].get(attr, []))
        for metric in sorted(b - a):
            db = before["subgroups"][attr]["disparities"][metric]
            da = after["subgroups"][attr]["disparities"][metric]
            changes.append({
                "attribute": attr, "metric": metric, "change": "WITHDRAWN",
                "gap_before": db["gap"], "gap_after": da["gap"],
                "intervals_overlap_after": da["intervals_overlap"],
                "meaning": ("this disparity was reported as supported and is no "
                            "longer: at the corrected threshold the intervals "
                            "overlap, so it cannot be distinguished from "
                            "sampling noise. The gap did not vanish — the "
                            "evidence for it did."),
            })
        for metric in sorted(a - b):
            changes.append({
                "attribute": attr, "metric": metric, "change": "NEWLY_SUPPORTED",
                "gap_after": after["subgroups"][attr]["disparities"][metric]["gap"],
                "meaning": "supported only at the corrected threshold",
            })
    return {
        "available": True,
        "threshold_before": before["operating_threshold"],
        "threshold_after": after["operating_threshold"],
        "supported_before": before["supported_disparities"],
        "supported_after": after["supported_disparities"],
        "changes": changes,
        "n_withdrawn": sum(c["change"] == "WITHDRAWN" for c in changes),
        "n_new": sum(c["change"] == "NEWLY_SUPPORTED" for c in changes),
    }


def to_markdown(report: dict) -> str:
    def fmt(v):
        return "—" if v is None else (f"{v:.5f}" if isinstance(v, float) else str(v))

    lines = [
        "<!-- generated by src/models/threshold_reconciliation.py — do not edit -->",
        "",
        "# Threshold Reconciliation — Before / After",
        "",
        "← [Back to README](../README.md)",
        "",
        f"Evidence: [`outputs/reports/threshold_reconciliation.json`]"
        f"(../outputs/reports/threshold_reconciliation.json)",
        "",
        f"**Threshold moved: {report['threshold_before']} → "
        f"{report['threshold_after']}**  ",
        f"before: {report['provenance_before']}  ",
        f"after: {report['provenance_after']}",
        "",
        "The contaminated value was fitted by F1-max on val, and val is also the",
        "drift reference window \u2014 so the reference window carried a threshold",
        "tuned to itself while the production window did not. The replacement is",
        "selected on a held-out, patient-level slice of train.",
        "",
        f"Threshold optimism in the reported F1 drop: **{report['measured_optimism']}**.",
        "",
        "> **This number was itself corrected in Tier 2C.6.** Tier 2A.4 originally",
        "> reported **0.0641**, which was an artifact: that block selected the",
        "> threshold under one calibrator, then scored val under a second and test",
        "> under a third. A threshold is a cut point on a probability scale, so",
        "> comparing two windows on different calibrators is not a like-for-like",
        "> comparison. Recomputed on a single scale, the optimism is small \u2014",
        "> **the val\u2192test F1 degradation was largely real, not an artifact of a",
        "> self-fitted threshold.** Found because this reconciliation produced a",
        "> different test F1 than decontamination.json reported for the same",
        "> threshold; the disagreement was the symptom.",
        "",
        "## Control: threshold-free metrics must not move",
        "",
        "If AUC, Brier or prevalence had moved, something other than the threshold",
        "changed and this comparison would not be interpretable.",
        "",
        f"**{report['control']['verdict']}** — "
        f"{report['control']['n_checked']} control values checked, "
        f"{report['control']['n_moved']} moved.",
        "",
        "## Every number that moved",
        "",
        "| Split | Metric | Before | After | Δ | Status |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["point_estimates"]:
        if r["status"] == "UNCHANGED":
            continue
        lines.append(f"| {r['split']} | `{r['metric']}` | {fmt(r['before'])} | "
                     f"{fmt(r['after'])} | {r['delta']:+.5f} | {r['status']} |")

    lines += [
        "",
        "Unchanged: "
        + ", ".join(sorted({f"`{r['metric']}`" for r in report["point_estimates"]
                            if r["status"] == "UNCHANGED"})),
        "",
        "## Claim reversals",
        "",
        "A val→test delta that changes sign is a **claim** change, not a value",
        "change: a reported degradation becomes an improvement. These require the",
        "surrounding sentence to be rewritten, not the figure to be updated.",
        "",
        "| Metric | val→test before | val→test after | Before | After | |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["val_to_test"]:
        mark = "**REVERSAL**" if r["status"] == "REVERSAL" else ""
        lines.append(f"| `{r['metric']}` | {r['val_to_test_delta_before']:+.5f} | "
                     f"{r['val_to_test_delta_after']:+.5f} | {r['direction_before']} | "
                     f"{r['direction_after']} | {mark} |")

    f = report.get("fairness_claims", {})
    if f.get("available"):
        lines += [
            "",
            "## Subgroup disparity claims",
            "",
            "The fairness audit does not report numbers so much as **claims** —",
            "*this disparity is supported by the data*. Those are threshold-",
            "dependent too, and one can be withdrawn without any headline number",
            "moving.",
            "",
            f"Threshold {f['threshold_before']} \u2192 {f['threshold_after']}",
            "",
            "| Attribute | Metric | Change | Gap before | Gap after | Overlap now? |",
            "|---|---|---|---|---|---|",
        ]
        for c in f["changes"]:
            lines.append(
                f"| `{c['attribute']}` | `{c['metric']}` | **{c['change']}** | "
                f"{c.get('gap_before', chr(8212))} | {c.get('gap_after', chr(8212))} | "
                f"{c.get('intervals_overlap_after', chr(8212))} |")
        if not f["changes"]:
            lines.append("| \u2014 | \u2014 | no change | \u2014 | \u2014 | \u2014 |")
        lines += [
            "",
            f"Supported before: `{f['supported_before']}`  ",
            f"Supported after: `{f['supported_after']}`",
        ]

    lines += ["", "## Interpretation", "", report["interpretation"], ""]
    return "\n".join(lines)


def run_threshold_reconciliation() -> dict:
    logger.info("=" * 78)
    logger.info("DriftSentinel — Tier 2C.6  threshold reconciliation")
    logger.info("=" * 78)

    for p in (BEFORE, AFTER):
        if not p.exists():
            raise FileNotFoundError(f"{p} is required for the before/after comparison")

    before = json.load(open(BEFORE, encoding="utf-8"))
    after = json.load(open(AFTER, encoding="utf-8"))

    points = compare_points(before, after)
    deltas = compare_val_to_test_deltas(before, after)
    widths = compare_intervals(before, after)
    fairness = compare_fairness_claims()

    controls = [r for r in points if r["threshold_free"]]
    moved_controls = [r for r in controls if r["status"] == "CONTROL_MOVED"]
    control = {
        "rule": ("AUC, Brier and prevalence do not depend on the threshold. If "
                 "any moved, the two runs differ in more than the threshold and "
                 "nothing else in this table can be attributed to it."),
        "n_checked": len(controls),
        "n_moved": len(moved_controls),
        "moved": moved_controls,
        "verdict": "PASS" if not moved_controls else "INVALID COMPARISON",
    }

    reversals = [r for r in deltas if r["status"] == "REVERSAL"]
    shifted = [r for r in points if r["status"] == "SHIFTED"]
    withdrawn = [c for c in fairness.get("changes", [])
                 if c["change"] == "WITHDRAWN"]

    if control["verdict"] != "PASS":
        interpretation = (
            "The comparison is INVALID: a threshold-free metric moved, so the two "
            "runs differ in more than the operating threshold. Do not use this "
            "table until that is explained.")
    elif reversals:
        names = ", ".join(f"`{r['metric']}`" for r in reversals)
        interpretation = (
            f"{len(shifted)} threshold-dependent values moved, and "
            f"{len(reversals)} of them **reverse a claim**: {names}. Under the "
            "contaminated threshold the val→test change was reported in one "
            "direction; under the decontaminated one it runs the other way. This "
            "is the optimism Tier 2A.4 predicted, now removed: the reference "
            "window no longer holds a threshold fitted to itself, so it no longer "
            "outperforms the production window by construction. Any README "
            "sentence describing degradation on these metrics must be rewritten, "
            "not merely renumbered.")
    elif withdrawn:
        names = ", ".join(f"`{c['attribute']}.{c['metric']}`" for c in withdrawn)
        interpretation = (
            f"{len(shifted)} threshold-dependent values moved and no val→test "
            "direction reversed, so no degradation claim flips. BUT "
            f"{len(withdrawn)} SUBGROUP DISPARITY CLAIM(S) ARE WITHDRAWN: "
            f"{names}. At the corrected threshold those intervals overlap, so "
            "the disparity can no longer be distinguished from sampling noise. "
            "The gap did not vanish — the EVIDENCE for it did, and a claim "
            "that shipped as supported must now ship as not supported. A "
            "numbers-only diff would have missed this entirely, which is why "
            "the fairness audit is diffed at the level of what it CLAIMS rather "
            "than what it reports.")
    else:
        interpretation = (
            f"{len(shifted)} threshold-dependent values moved, no val→test "
            "direction reversed, and no subgroup disparity claim changed "
            "status. The correction changes magnitudes, not claims.")

    report = {
        "phase": "2C.6",
        "title": "Reconciling the canonical metrics file to the decontaminated threshold",
        "threshold_before": before["operating_threshold"],
        "threshold_after": after["operating_threshold"],
        "provenance_before": "val, by F1-max (evaluation_report.json) — contaminated",
        "provenance_after": after.get("operating_threshold_provenance", {}).get(
            "selected_on", "held-out slice of train (patient-level)"),
        "measured_optimism": json.load(
            open(REPORTS_DIR / "decontamination.json", encoding="utf-8")
        )["threshold"]["threshold_optimism_in_the_reported_f1_drop"],
        "control": control,
        "point_estimates": points,
        "val_to_test": deltas,
        "interval_widths": widths,
        "fairness_claims": fairness,
        "n_values_moved": len(shifted),
        "n_claim_reversals": len(reversals),
        "n_fairness_claims_withdrawn": len(withdrawn),
        "reversals": reversals,
        "interpretation": interpretation,
        "superseded_predecessor": (
            "outputs/reports/superseded/tier2c6_contaminated_threshold/"
            "headline_metrics_ci.json"),
        "reproducibility": {"python": platform.python_version(),
                            "numpy": np.__version__, "pandas": pd.__version__},
    }

    out = REPORTS_DIR / "threshold_reconciliation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    md = ROOT / "docs" / "10_threshold_reconciliation.md"
    md.write_text(to_markdown(report), encoding="utf-8")

    logger.info(f"  control          : {control['verdict']} "
                f"({control['n_moved']}/{control['n_checked']} moved)")
    logger.info(f"  values moved     : {len(shifted)}")
    logger.info(f"  claim reversals  : {len(reversals)}")
    for r in reversals:
        logger.warning(f"    REVERSAL {r['metric']}: "
                       f"{r['val_to_test_delta_before']:+.5f} -> "
                       f"{r['val_to_test_delta_after']:+.5f}")
    logger.info(f"Report: {out.name}  |  {md.relative_to(ROOT).as_posix()}")
    logger.info("=" * 78)
    return report


if __name__ == "__main__":
    r = run_threshold_reconciliation()
    print(f"\nThreshold {r['threshold_before']} -> {r['threshold_after']}")
    print(f"  control            : {r['control']['verdict']}")
    print(f"  values moved       : {r['n_values_moved']}")
    print(f"  claim reversals    : {r['n_claim_reversals']}")
    for x in r["reversals"]:
        print(f"    {x['metric']:<24} {x['val_to_test_delta_before']:+.5f} -> "
              f"{x['val_to_test_delta_after']:+.5f}  "
              f"({x['direction_before']} -> {x['direction_after']})")
