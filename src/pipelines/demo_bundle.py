"""
DriftSentinel — keep the deployed console's numbers tied to the evidence.

THE PROBLEM THIS SOLVES
    `app/demo_data/evidence.json` is a precomputed bundle so the Streamlit
    console can cold-start in seconds without loading a model. That is the right
    deployment choice, and it creates a specific hazard: **the console becomes a
    second copy of the results, and copies go stale.** It did. The bundle was
    built before the Tier 2C.6 threshold reconciliation, so the live console was
    serving subgroup numbers measured at a threshold the repository had since
    withdrawn — the same defect class as hand-typed prose in a generated report,
    relocated to the demo layer.

    R4 says every number traces to a named artifact. A number on a public
    dashboard is not exempt because it is convenient.

WHAT THIS DOES
    Declares, per bundle section, WHICH artifact it comes from and WHERE inside
    it. Then either verifies the bundle against those sources or rewrites it from
    them. `verify` is the CI-shaped operation: it fails when the console and the
    evidence disagree.

COVERAGE IS DECLARED, NOT IMPLIED
    Sections with a declared source are checked. Sections without one are listed
    by name in the report as UNMAPPED. A sync tool that silently ignores what it
    cannot map reads as full coverage when it is partial, so the unmapped list is
    part of the output rather than an omission from it.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("demo_bundle")

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "outputs" / "reports"
BUNDLE = ROOT / "app" / "demo_data" / "evidence.json"

# ── verbatim copies ──────────────────────────────────────────────────────────
# bundle key -> (artifact relative to outputs/reports, path inside it)
SOURCES: dict[str, tuple[str, list[str]]] = {
    "fairness": ("fairness_audit.json", ["subgroups"]),
    "verdict": ("temporal_validity.json", ["verdict"]),
}

# ── projections ──────────────────────────────────────────────────────────────
# Several sections are deliberately TRIMMED views: the console needs four fields
# out of a forty-field anchor record, and shipping the whole artifact would push
# the bundle from 44 KB to several hundred for no benefit. Those cannot be
# checked by equality — a trimmed copy always differs — so each is checked
# FIELD BY FIELD against the value it was projected from.
#
# This distinction matters: treating a projection as a stale copy would produce a
# permanent false alarm, and permanent false alarms are how checks get ignored.
PROJECTIONS: dict[str, dict] = {
    "anchors": {
        "artifact": "temporal_validity.json",
        "root": ["anchors"],
        "per_item": {                       # bundle field -> path inside each anchor
            "outcome": ["outcome"],
            "p_raw": ["p_raw"],
            "role": ["role"],
            "event_date": ["detail", "event_date"],
        },
    },
    "multivariate": {
        "artifact": "multivariate_drift.json",
        "root": ["regimes"],
        "per_item": {
            "c2st_auc": ["classifier_2st", "held_out_auc"],
            "c2st_p": ["classifier_2st", "p_permutation"],
            "mmd_p": ["mmd", "p_permutation"],
            "bbsd": ["bbsd", "detects_drift"],
        },
    },
    "label_interval": {
        "artifact": "temporal_validity.json",
        "root": ["label_interval_coherence"],
        "flat": {                           # bundle field -> path inside root
            "median_gap_rank_lt30": ["median_gap_rank_lt30"],
            "median_gap_rank_gt30": ["median_gap_rank_gt30"],
            "ratio_gt30_over_lt30": ["ratio_gt30_over_lt30"],
            "implied_calendar_check": ["implied_calendar_check"],
        },
    },
}

# Sections that are assembled or reshaped rather than copied. Named here so the
# report can distinguish "not checked" from "checked and fine".
KNOWN_UNMAPPED = {
    "regimes": "reshaped from regime_matrix.json for the console's panel layout",
    "live": "assembled from the concept-drift and alert artifacts",
    "diagnosticity": "reshaped from regime_matrix.json",
    "power": "reshaped from regime_synthetic.json power curves",
    "threshold_policy": "reshaped from threshold_policy_lgbm_v1.json",
    "conformal": "assembled from decontamination.json and triage_policy.json",
    "timeline": "flattened from temporal_validity.json timeline_coherence",
    "censoring": "reshaped from temporal_validity.json censoring",
    "ablation": "reshaped from selection_ablation.json",
    "robustness": "reshaped from data_quality_robustness.json",
}


def _dig(doc: dict, path: list[str]):
    cur = doc
    for k in path:
        if k not in cur:
            raise KeyError(f"missing {k!r} on path {'/'.join(path)}")
        cur = cur[k]
    return cur


def _load_source(artifact: str, path: list[str]):
    p = REPORTS / artifact
    if not p.exists():
        raise FileNotFoundError(f"{p} is required to check the demo bundle")
    return _dig(json.load(open(p, encoding="utf-8")), path)


def check(fix: bool = False) -> dict:
    logger.info("=" * 78)
    logger.info(f"DriftSentinel — demo bundle {'sync' if fix else 'verify'}")
    logger.info("=" * 78)

    if not BUNDLE.exists():
        raise FileNotFoundError(f"{BUNDLE} not found")
    bundle = json.load(open(BUNDLE, encoding="utf-8"))

    checked, stale, missing = [], [], []
    for key, (artifact, path) in SOURCES.items():
        want = _load_source(artifact, path)
        have = bundle.get(key)
        if have is None:
            missing.append({"section": key, "source": artifact})
            if fix:
                bundle[key] = want
            continue
        if json.dumps(have, sort_keys=True) == json.dumps(want, sort_keys=True):
            checked.append({"section": key, "source": artifact, "status": "IN_SYNC"})
        else:
            stale.append({"section": key, "source": artifact,
                          "source_path": "/".join(path)})
            if fix:
                bundle[key] = want

    # Projections: checked field by field, because a trimmed view never equals
    # its source and comparing them by equality would alarm forever.
    drifted_fields = []
    for key, spec in PROJECTIONS.items():
        root = _load_source(spec["artifact"], spec["root"])
        have = bundle.get(key)
        if have is None:
            missing.append({"section": key, "source": spec["artifact"]})
            continue
        n_ok = 0
        pairs = ([(None, spec["flat"])] if "flat" in spec
                 else [(i, spec["per_item"]) for i in have])
        for item, fields in pairs:
            src_item = root if item is None else root.get(item)
            bnd_item = have if item is None else have[item]
            if src_item is None:
                drifted_fields.append({"section": key, "field": f"{item}",
                                       "reason": "absent from the source artifact"})
                continue
            for field, path in fields.items():
                if field not in bnd_item:
                    continue
                try:
                    want = _dig(src_item, path)
                except KeyError:
                    # A field that is absent at the source AND null in the bundle
                    # is consistent, not stale: not every item carries every
                    # field. A3 is a monotone-trend anchor with no event date, so
                    # `event_date: null` is the correct projection of "there
                    # isn't one". Flagging that forever would train the reader to
                    # ignore this check. A NON-null bundle value with no source
                    # is still a defect — that is a number with no provenance.
                    if bnd_item[field] is not None:
                        drifted_fields.append({
                            "section": key,
                            "field": f"{item}.{field}" if item else field,
                            "bundle_value": bnd_item[field],
                            "reason": (f"bundle holds a value but source path "
                                       f"{'/'.join(path)} does not exist — a "
                                       f"number with no provenance")})
                    else:
                        n_ok += 1
                    continue
                got = bnd_item[field]
                if json.dumps(got, sort_keys=True) != json.dumps(want, sort_keys=True):
                    drifted_fields.append({
                        "section": key,
                        "field": f"{item}.{field}" if item else field,
                        "bundle_value": got, "source_value": want,
                        "source": f"{spec['artifact']}:{'/'.join(spec['root'] + path)}"})
                    if fix:
                        bnd_item[field] = want
                else:
                    n_ok += 1
        checked.append({"section": key, "source": spec["artifact"],
                        "status": "PROJECTION", "n_fields_matched": n_ok})

    unmapped = sorted(set(bundle) - set(SOURCES) - set(PROJECTIONS))
    undeclared = [k for k in unmapped if k not in KNOWN_UNMAPPED]
    stale = stale + drifted_fields

    if fix and (stale or missing):
        with open(BUNDLE, "w", encoding="utf-8") as f:
            json.dump(bundle, f, separators=(",", ":"))
        logger.info(f"  rewrote {len(stale) + len(missing)} section(s) from source")

    verdict = ("SYNCED" if fix and (stale or missing) else
               "IN_SYNC" if not stale and not missing else "STALE")

    report = {
        "phase": "3.0",
        "title": "The deployed console's numbers, checked against the evidence",
        "bundle": BUNDLE.relative_to(ROOT).as_posix(),
        "bundle_bytes": BUNDLE.stat().st_size,
        "verdict": verdict,
        "n_sections_mapped": len(SOURCES) + len(PROJECTIONS),
        "n_verbatim_copies": len(SOURCES),
        "n_projections": len(PROJECTIONS),
        "in_sync": checked,
        "stale": stale,
        "missing_from_bundle": missing,
        "unmapped_sections": {k: KNOWN_UNMAPPED.get(k, "NOT DECLARED") for k in unmapped},
        "undeclared_sections": undeclared,
        "coverage_note": (
            f"{len(SOURCES)} bundle sections are verbatim copies of a named "
            f"artifact and are checked by equality; {len(PROJECTIONS)} are "
            "deliberately TRIMMED views and are checked field by field against "
            "the value each was projected from. The remaining "
            f"{len(bundle) - len(SOURCES) - len(PROJECTIONS)} are reshaped for the "
            "console's layout and are listed by name rather than silently skipped "
            "— a sync tool that ignores what it cannot map reads as full coverage "
            "when it is partial."),
        "reproducibility": {"python": platform.python_version()},
    }
    out = REPORTS / "demo_bundle_sync.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    for c in checked:
        logger.info(f"  IN_SYNC  {c['section']:<16} <- {c['source']}")
    for s in stale:
        where = s.get("source") or s.get("field") or s.get("source_path") or "?"
        logger.warning(f"  STALE    {s['section']:<16} <- {where}")
    for m in missing:
        logger.warning(f"  MISSING  {m['section']:<16} <- {m['source']}")
    if undeclared:
        logger.warning(f"  undeclared sections: {undeclared}")
    logger.info(f"VERDICT: {verdict}  |  Report: {out.name}")
    logger.info("=" * 78)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify or sync the Streamlit console's evidence bundle")
    ap.add_argument("--fix", action="store_true",
                    help="rewrite stale sections from their source artifacts")
    args = ap.parse_args()
    r = check(fix=args.fix)
    print(f"\nDemo bundle: {r['verdict']}  ({r['bundle_bytes'] / 1024:.0f} KB)")
    for s in r["stale"]:
        where = s.get("source") or s.get("field") or "?"
        print(f"  STALE   {s['section']} <- {where}")
    for k, v in r["unmapped_sections"].items():
        print(f"  unmapped {k:<18} {v}")
    return 0 if r["verdict"] in ("IN_SYNC", "SYNCED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
