"""
DriftSentinel — Phase 0.5 planning artifact: surgical language audit

WHAT
    Classifies every occurrence of temporal/mechanism language in the source,
    docs and config into an explicit proposed action, and writes a reviewable
    diff plan.

WHY SURGICAL RATHER THAN BLANKET
    The remediation plan assumed Phase 0.1 would return NOT SUPPORTED and that
    every use of "temporal" would therefore be unearned. It returned SUPPORTED:
    `encounter_id` ordering IS chronological, verified against troglitazone
    withdrawal (2000-03-21), ICD-9 V85 introduction (2005-10-01) and the
    rosiglitazone safety changepoint (2007-05-21).

    So the word is now EARNED where it describes encounter ordering, and still
    WRONG where it describes the split — which sorts patients by entry cohort,
    not encounters by time. A blanket rename would delete a true statement; a
    blanket keep would preserve a false one.

    Separately, unearned MECHANISM claims ("billing codes changed", "patient
    demographics shifted", "the world keeps changing") are rewritten regardless
    of the ordering verdict: Phase 0.1 evidenced chronology, not causes.

THIS MODULE WRITES NO SOURCE FILE. It emits a plan for approval.

OUTPUTS
    outputs/reports/language_audit_plan.json
    outputs/reports/language_audit_plan.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("language_audit")

ROOT        = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "temporal_validity.yaml"
REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Files that ARE the Phase 0.1 investigation — their use of the vocabulary is
# the evidence itself and is left alone.
INVESTIGATION_PATHS = ("src/investigation/", "configs/temporal_validity.yaml",
                       "configs/split_regimes.yaml")

# Occurrences describing the SPLIT as temporal — the claim Phase 0.1 refuted.
SPLIT_CLAIM = re.compile(
    r"(temporal\s+split|patient-level\s+temporal|temporal\s+proxy|"
    r"split.{0,40}temporal|temporal.{0,40}split|temporal\s+structure|"
    r"temporal\s+order|chronological(ly)?\s+order)", re.I)

# Unearned mechanism claims — rewritten regardless of the ordering verdict.
MECHANISM_CLAIM = re.compile(
    r"(billing\s+code|demographics\s+shift|newer\s+patients|"
    r"world\s+keeps\s+changing|drift\s+began)", re.I)

REPLACEMENTS = {
    "temporal split": "entry-cohort split",
    "patient-level temporal split": "patient-level entry-cohort split",
    "temporal proxy": "entry-cohort ordering",
    "temporal structure": "entry-cohort structure",
    "billing code": "[REMOVE — mechanism claim without mechanism evidence]",
    "demographics shift": "[REMOVE — mechanism claim without mechanism evidence]",
    "newer patients": "later-entering patients",
    "world keeps changing": "[REMOVE — mechanism claim without mechanism evidence]",
    "drift began": "[REMOVE unless tied to a Phase 0.2-0.4 regime result]",
}

ACTION_NOTES = {
    "KEEP": ("refers to encounter_id chronology, which Phase 0.1 evidenced, or "
             "is part of the investigation itself"),
    "REWRITE_SPLIT": ("describes the SPLIT as temporal; the split sorts patients "
                      "by entry cohort, so this is the claim Phase 0.1 refuted"),
    "REWRITE_MECHANISM": ("asserts a CAUSE for the observed shift; Phase 0.1 "
                          "evidenced chronology, not mechanism"),
    "MANUAL_REVIEW": "ambiguous from the line alone — decide in context",
}


def _classify(rel: str, line: str) -> str:
    if rel.replace("\\", "/").startswith(INVESTIGATION_PATHS):
        return "KEEP"
    if MECHANISM_CLAIM.search(line):
        return "REWRITE_MECHANISM"
    if SPLIT_CLAIM.search(line):
        return "REWRITE_SPLIT"
    low = line.lower()
    if "encounter_id" in low and "temporal" in low:
        return "KEEP"
    return "MANUAL_REVIEW"


def _suggest(line: str) -> str | None:
    for k, v in REPLACEMENTS.items():
        if k in line.lower():
            return v
    return None


def run_language_audit(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        lc = yaml.safe_load(f)["language_inventory"]

    pats = [p.lower() for p in lc["patterns"]]
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in lc["scan_extensions"]:
            continue
        rel = path.relative_to(ROOT).as_posix()
        # Prefix-anchored ONLY. The earlier substring form (`f"/{s}/" in f"/{rel}"`)
        # made the top-level `data` skip rule also swallow `src/data/`, which
        # excluded splitter.py — the single most important file for this rename —
        # from the inventory entirely. Skip rules are repo-root paths, not
        # path fragments.
        if any(rel == s or rel.startswith(s.rstrip("/") + "/") for s in lc["skip_dirs"]):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            matched = [p for p in pats if p in low]
            if not matched:
                continue
            action = _classify(rel, line)
            hits.append({
                "file": rel, "line": i, "patterns": matched,
                "action": action, "reason": ACTION_NOTES[action],
                "suggested_replacement": _suggest(line) if action.startswith("REWRITE") else None,
                "text": line.strip()[:220],
            })

    by_action: dict[str, int] = {}
    by_file: dict[str, dict[str, int]] = {}
    for h in hits:
        by_action[h["action"]] = by_action.get(h["action"], 0) + 1
        by_file.setdefault(h["file"], {})
        by_file[h["file"]][h["action"]] = by_file[h["file"]].get(h["action"], 0) + 1

    plan = {
        "phase": "0.5 (PLAN ONLY — NOT APPLIED)",
        "policy": "surgical",
        "policy_rationale": __doc__.split("WHY SURGICAL RATHER THAN BLANKET")[1]
                                   .split("THIS MODULE WRITES NO SOURCE FILE")[0].strip(),
        "phase_0_1_verdict": {"ordering": "SUPPORTED",
                              "split_validity": "NOT_TEMPORAL_BY_CONSTRUCTION"},
        "totals": {"occurrences": len(hits), "files": len(by_file),
                   "by_action": by_action},
        "by_file": by_file,
        "occurrences": hits,
        "files_modified": [],
        "status": "AWAITING APPROVAL — no file has been changed",
    }

    with open(REPORTS_DIR / "language_audit_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    lines = ["# Phase 0.5 — Language correction plan (NOT APPLIED)", "",
             f"Policy: **surgical**. Phase 0.1 verdict: ordering **SUPPORTED**, "
             f"split validity **NOT_TEMPORAL_BY_CONSTRUCTION**.", "",
             f"- occurrences: **{len(hits)}** across **{len(by_file)}** files",
             *[f"- `{a}`: {n}  — {ACTION_NOTES[a]}" for a, n in sorted(by_action.items())],
             "", "## Proposed changes", ""]
    for action in ["REWRITE_SPLIT", "REWRITE_MECHANISM", "MANUAL_REVIEW", "KEEP"]:
        sel = [h for h in hits if h["action"] == action]
        if not sel:
            continue
        lines += [f"### {action} ({len(sel)})", ""]
        if action == "KEEP":
            files = sorted({h["file"] for h in sel})
            lines += [f"- {len(sel)} occurrences in: " + ", ".join(f"`{f}`" for f in files), ""]
            continue
        for h in sel:
            rep = f" → `{h['suggested_replacement']}`" if h["suggested_replacement"] else ""
            lines.append(f"- `{h['file']}:{h['line']}`{rep}")
            lines.append(f"  - `{h['text']}`")
        lines.append("")
    with open(REPORTS_DIR / "language_audit_plan.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Language audit plan: {len(hits)} occurrences in {len(by_file)} files")
    for a, n in sorted(by_action.items()):
        logger.info(f"  {a:<20} {n}")
    logger.info("NO FILE MODIFIED — plan awaiting approval")
    return plan


if __name__ == "__main__":
    p = run_language_audit()
    print(json.dumps(p["totals"], indent=2))
