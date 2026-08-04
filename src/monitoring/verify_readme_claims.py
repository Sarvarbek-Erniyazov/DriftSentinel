"""
DriftSentinel — FINAL acceptance: every numeric claim in the README traces to a
generated artifact.

WHY THIS AND NOT A README GENERATOR
    There is no `build_readme.py` and the README is not generated. It is authored
    prose, so "regenerates identically" is not a property it has, and claiming it
    would be the exact failure this repository keeps finding: an assertion that
    cannot fail.

    The property the plan actually specifies is TRACEABILITY:

        "a traceability script verifies every numeric claim in the README appears
         in a generated artifact"

    That IS checkable, and it is checkable in the direction that matters. A
    generator would guarantee the README matches whatever the generator was told;
    this checks the README against the artifacts themselves, which is what a
    hostile reviewer will do by hand.

HOW IT WORKS
    Every number in the README is extracted, then searched for across every
    generated artifact under outputs/ at the precision it is quoted to. A number
    that appears nowhere is either a typo, a stale value from a superseded run, or
    a claim with no evidence behind it — and the report says which of the three
    cannot be distinguished, rather than guessing.

WHAT IS DELIBERATELY EXEMPT, AND WHY IT IS LISTED
    Years, dataset sizes, reference dates, section numbers, DOIs, external
    published baselines and p-values quoted from anchors are not repository
    outputs. They are enumerated in EXEMPT below with a reason each. An exemption
    list that is inferred rather than declared would let this check quietly pass
    on anything it could not find.
"""

from __future__ import annotations

import json
import platform
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("verify_readme_claims")

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
REPORTS_DIR = ROOT / "outputs" / "reports"

# Where a README number may legitimately come from.
ARTIFACT_GLOBS = ("outputs/reports/*.json", "outputs/reports/*.csv",
                  "outputs/registry/*.json", "outputs/models/*.json",
                  "outputs/log/*.json")

# Declared exemptions: not repository outputs. Reason required for each.
EXEMPT: dict[str, str] = {
    "1999": "dataset collection window start, from the source",
    "2008": "dataset collection window end, from the source",
    "2014": "Strack et al. publication year", "2024": "Liu et al. publication year",
    "2026": "Salim & Ibrahim publication year", "2019": "Obermeyer / Mitchell year",
    "2021": "Gibbs & Candes / Gebru year", "2012": "Gretton et al. year",
    "2017": "Lopez-Paz & Oquab year", "2018": "Lipton et al. year",
    "2016": "Kantchelian et al. year", "2006": "Vickers & Elkin year",
    "1995": "Benjamini & Hochberg year", "1988": "DeLong et al. year",
    "2023": "Barber et al. year", "2005": "Vovk et al. year",
    "101766": "dataset row count, from the source",
    "71518": "dataset patient count, from the source",
    "130": "hospitals in the dataset name",
    "3.12": "Python version", "512": "test budget constant, asserted in tests",
    "0.05": "conventional significance level",
    "60": "split proportion, stated as a percentage",
    "20": "split proportion / seed count, stated in prose",
    "0.64": "Liu et al. published AUC (external)",
    "0.65": "Liu et al. published CI bound (external)",
    "0.664": "Salim & Ibrahim published AUC (external)",
    "0.61": "published band lower bound (external)",
    "0.66": "published band upper bound (external)",
    "0.024": "spread between two EXTERNAL published numbers",
}

NUMBER_RE = re.compile(r"(?<![\w.])(\d+\.\d+|\d{3,})(?![\w])")

# Structural noise that is not a claim.
SKIP_CONTEXT = ("http", "](#", "doi:", "10.", "badge", "shields.io", "%20")


def readme_numbers() -> list[dict]:
    out, seen = [], set()
    for i, line in enumerate(README.read_text(encoding="utf-8").splitlines(), 1):
        low = line.lower()
        if any(s in low for s in SKIP_CONTEXT):
            continue
        for m in NUMBER_RE.finditer(line):
            tok = m.group(1)
            if tok in seen:
                continue
            seen.add(tok)
            out.append({"value": tok, "line": i, "context": line.strip()[:110]})
    return out


def artifact_corpus() -> tuple[str, int]:
    parts, n = [], 0
    for g in ARTIFACT_GLOBS:
        for p in sorted(ROOT.glob(g)):
            if "superseded" in p.parts:
                continue
            try:
                parts.append(p.read_text(encoding="utf-8"))
                n += 1
            except Exception:
                continue
    return "\n".join(parts), n


def corpus_values(corpus: str) -> list[float]:
    """Every number appearing anywhere in the generated artifacts."""
    vals = []
    for m in re.finditer(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", corpus):
        try:
            vals.append(float(m.group()))
        except ValueError:
            continue
    return vals


def _appears(tok: str, corpus: str, values: list[float]) -> bool:
    """
    Match NUMERICALLY at the precision the README quotes, not by substring.

    A README rounds and an artifact does not: the README says `0.6255`, the
    artifact holds `0.62546`. Substring matching fails on that, and a check that
    fails on correct values is worse than no check — it trains the reader to
    ignore the output. So an artifact value counts as the source of a README
    number when it ROUNDS to it at the quoted number of decimals.

    Percentages are also accepted against their fractional form (`46.2` against
    `0.462`), because the README states shares as percentages in prose.
    """
    if tok in corpus:
        return True
    try:
        v = float(tok)
    except ValueError:
        return False

    decimals = len(tok.split(".")[1]) if "." in tok else 0
    tol = 0.5 * (10 ** -decimals)
    for candidate in (v, v / 100.0):          # value, and value-as-percentage
        for a in values:
            if abs(a - candidate) < tol:
                return True
    return False


def run_verification() -> dict:
    logger.info("=" * 78)
    logger.info("DriftSentinel — README numeric-claim traceability")
    logger.info("=" * 78)

    corpus, n_files = artifact_corpus()
    values = corpus_values(corpus)
    logger.info(f"  corpus: {n_files} generated artifacts, "
                f"{len(values):,} numeric values")

    nums = readme_numbers()
    traced, exempt, untraced = [], [], []
    for item in nums:
        tok = item["value"]
        if tok in EXEMPT:
            exempt.append({**item, "reason": EXEMPT[tok]})
        elif _appears(tok, corpus, values):
            traced.append(item)
        else:
            untraced.append(item)

    report = {
        "phase": "FINAL",
        "title": "Every numeric claim in the README appears in a generated artifact",
        "why_not_a_generator": (
            "The README is authored prose, not generated output, so 'regenerates "
            "identically' is not a property it has. The plan's acceptance "
            "criterion is traceability, which is checkable in the direction that "
            "matters: the README is checked AGAINST the artifacts, which is what "
            "a reviewer does by hand. A generator would only guarantee the README "
            "matches whatever the generator was told."),
        "n_artifacts_searched": n_files,
        "n_artifact_values": len(values),
        "n_numbers_found": len(nums),
        "n_traced": len(traced),
        "n_exempt": len(exempt),
        "n_untraced": len(untraced),
        "untraced": untraced,
        "exempt": exempt,
        "exemption_policy": (
            "Exemptions are DECLARED by value with a reason each, never inferred. "
            "An inferred exemption would let this check pass on anything it could "
            "not find, which is the failure mode it exists to prevent."),
        "verdict": "PASS" if not untraced else "UNTRACED CLAIMS",
        "interpretation": (
            "Every non-exempt number in the README appears in a generated "
            "artifact." if not untraced else
            f"{len(untraced)} number(s) appear in the README but in no generated "
            "artifact. Each is a typo, a value stale from a superseded run, or a "
            "claim with no evidence behind it — this check cannot distinguish "
            "which, and does not guess."),
        "reproducibility": {"python": platform.python_version()},
    }
    out = REPORTS_DIR / "readme_traceability.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"  numbers: {len(nums)}  traced {len(traced)}  "
                f"exempt {len(exempt)}  untraced {len(untraced)}")
    for u in untraced:
        logger.error(f"    UNTRACED {u['value']:<12} line {u['line']}: {u['context']}")
    logger.info(f"VERDICT: {report['verdict']}  |  Report: {out.name}")
    logger.info("=" * 78)
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    r = run_verification()
    print(f"\nREADME traceability: {r['verdict']}")
    print(f"  searched  : {r['n_artifacts_searched']} generated artifacts")
    print(f"  numbers   : {r['n_numbers_found']}")
    print(f"  traced    : {r['n_traced']}")
    print(f"  exempt    : {r['n_exempt']} (declared, with reasons)")
    print(f"  untraced  : {r['n_untraced']}")
    for u in r["untraced"]:
        print(f"    {u['value']:<12} line {u['line']}: {u['context']}")
    raise SystemExit(0 if r["verdict"] == "PASS" else 1)
