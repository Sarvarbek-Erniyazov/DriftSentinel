"""
DriftSentinel — Tier 2C.4: determinism as a CI check.

WHY A HASH CHECK AND NOT A METRIC CHECK
    Determinism had been verified three times by hand, and by hand it kept
    passing. The defect it needed to catch would have passed too, because the
    defect was VALUE-IDENTICAL AND BYTE-DIFFERENT:

        `TARGET_COLS` is a `set`. Python randomises string hashing per process,
        so iterating it produced a different COLUMN ORDER on every run. Every
        value in every column was identical. Every metric was identical. Only
        the serialised bytes moved — and the adversarial modules index
        positionally (`X[:, i]`), so an artifact fitted in one run and applied to
        another run's data would have read the wrong column, silently.

    No seed could have caught it: the defect was in serialisation, not in
    modelling. A check that compares metrics could not have caught it either.
    So this check compares **artifact hashes**, and treats a byte difference with
    identical values as a FAILURE rather than a curiosity — that case is the
    whole reason the check exists.

WHY TWO EXPLICIT, DIFFERENT HASH SEEDS
    Left to chance, `PYTHONHASHSEED` might coincide, or two random seeds might
    happen to order a 2-3 element set identically — which for a small set is
    roughly a coin flip. A run pair that cannot distinguish the two orderings is
    a check that passes either way. The two runs are therefore given DIFFERENT
    explicit seeds, and the difference is verified by probing the interpreter in
    each environment before the pipeline runs (R6: assert the mechanism, not a
    side effect of it).

FALSIFICATION ARM
    A determinism check that has never been shown to fail is unfalsifiable. Before
    reporting any verdict, the comparator is handed a synthetic pair that is
    value-identical and byte-different — the exact defect class above — and must
    flag it. If it does not, the verdict is UNKNOWN. It is never PASS.

WHAT IS COMPARED, AND WHAT IS NOT
    Compared: every artifact the pipeline writes that is supposed to be
    reproducible. Excluded: logs and the run summary, which carry wall-clock
    timings and timestamps and are expected to differ. Exclusions are ENUMERATED
    in the report with a reason each, never applied silently — a check that
    quietly drops the files it cannot handle reads as broader coverage than it has.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.monitoring.logger import get_logger

logger = get_logger("determinism")

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Inputs a sandboxed run needs. Everything else is regenerated.
SANDBOX_INPUTS = ("src", "configs", "data/raw")
SANDBOX_MKDIRS = ("data/train", "data/production", "outputs/artifacts",
                  "outputs/log", "outputs/figure", "outputs/models",
                  "outputs/reports", "outputs/alerts", "outputs/registry")

# Artifacts that must be byte-identical across runs.
COMPARED_GLOBS = (
    "data/train/*.parquet",
    "data/production/*.parquet",
    "outputs/artifacts/*",
    "outputs/models/*",          # only populated by the `trainer` stage
)

# Excluded, each with the reason it cannot be byte-stable. Enumerated rather
# than silently skipped.
EXCLUDED_PATTERNS = {
    "outputs/log/*.log": "structured logs carry wall-clock timestamps on every line",
    "outputs/log/pipeline_summary.json": "records per-stage wall-clock timings",
    "outputs/models/training_summary.json":
        ("records `train_time_s` wall-clock timing alongside its params and CV "
         "results. Excluded on the same grounds as pipeline_summary.json rather "
         "than by stripping the timing key — normalising a field away to make a "
         "check pass is how a check stops being one. The substantive artifacts "
         "it summarises (lgbm_v1.pkl, logreg_v1.pkl, logreg_scaler.pkl) ARE "
         "compared and must be byte-identical."),
    "outputs/models/*.meta.json":
        ("provenance sidecars record `created_at` and the git commit, so they "
         "are wall-clock-dependent by design -- the same grounds as "
         "pipeline_summary.json. The MODEL BINARIES they describe ARE compared "
         "and must be byte-identical, and each sidecar carries that binary's "
         "SHA-256, so a sidecar drifting from its artifact would surface as a "
         "difference in the artifact itself rather than being hidden here."),
    "outputs/artifacts/*.tmp": "scratch files, not artifacts",
}

# Stages runnable in a sandbox, in dependency order.
STAGES = {
    "pipeline": ["src/pipelines/pipeline.py"],
    "trainer": ["src/pipelines/pipeline.py", "src/models/trainer.py"],
}

# The string set whose iteration order caused the original defect. Probing its
# hash proves the two run environments really do randomise differently.
PROBE_STRINGS = ("readmitted_binary", "readmitted_multi", "readmitted")


# ── hashing and snapshots ────────────────────────────────────────────────────

def hash_file(path: Path) -> str:
    """SHA-256 of a file's raw bytes. Bytes, not parsed content — on purpose."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _excluded(rel: str) -> str | None:
    for pattern, reason in EXCLUDED_PATTERNS.items():
        if fnmatch.fnmatch(rel, pattern):
            return reason
    return None


def snapshot(root: Path, globs=COMPARED_GLOBS) -> dict:
    """Map every compared artifact under `root` to its size and content hash."""
    out = {}
    for g in globs:
        for p in sorted(root.glob(g)):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if _excluded(rel):
                continue
            out[rel] = {"sha256": hash_file(p), "bytes": p.stat().st_size}
    return out


# ── semantic comparison: is it the values, or only the bytes? ────────────────

def _load_comparable(path: Path):
    """
    Parse an artifact into something comparable by value, or return None if the
    format is opaque (pickles). None is not treated as 'equal' anywhere.
    """
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return None


def _values_equal(a, b) -> bool | None:
    """True / False, or None when the comparison cannot be made."""
    if a is None or b is None:
        return None
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        if set(a.columns) != set(b.columns) or len(a) != len(b):
            return False
        # Column ORDER is deliberately normalised away here: the point is to
        # separate "the values moved" from "only the layout moved".
        left = a.reindex(sorted(a.columns), axis=1).reset_index(drop=True)
        right = b.reindex(sorted(b.columns), axis=1).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(left, right, check_like=False,
                                          check_dtype=True)
            return True
        except AssertionError:
            return False
    return a == b


def compare_snapshots(snap_a: dict, snap_b: dict,
                      root_a: Path, root_b: Path) -> list[dict]:
    """
    Classify every disagreement between two runs.

    BYTE_DIFFERENCE_ONLY is the classification that matters: identical values,
    different bytes. That is the TARGET_COLS defect class, and it is reported as
    a failure, not as an acceptable difference.
    """
    diffs = []
    for rel in sorted(set(snap_a) | set(snap_b)):
        if rel not in snap_a or rel not in snap_b:
            diffs.append({"artifact": rel, "kind": "MISSING_IN_ONE_RUN",
                          "present_in_run_a": rel in snap_a,
                          "present_in_run_b": rel in snap_b})
            continue
        if snap_a[rel]["sha256"] == snap_b[rel]["sha256"]:
            continue

        eq = _values_equal(_load_comparable(root_a / rel),
                           _load_comparable(root_b / rel))
        if eq is True:
            kind, note = "BYTE_DIFFERENCE_ONLY", (
                "values identical, bytes differ — the TARGET_COLS defect class: "
                "no seed and no metric comparison would catch this")
        elif eq is False:
            kind, note = "VALUE_DIFFERENCE", "the artifact's contents changed between runs"
        else:
            kind, note = "OPAQUE_BYTE_DIFFERENCE", (
                "binary artifact whose contents cannot be compared by value; "
                "a byte difference here is treated as a failure because it "
                "cannot be shown to be benign")
        diffs.append({"artifact": rel, "kind": kind, "note": note,
                      "sha256_a": snap_a[rel]["sha256"],
                      "sha256_b": snap_b[rel]["sha256"],
                      "bytes_a": snap_a[rel]["bytes"],
                      "bytes_b": snap_b[rel]["bytes"]})
    return diffs


# ── falsification arm ────────────────────────────────────────────────────────

def falsification_arm() -> dict:
    """
    Hand the comparator the exact defect class it exists to catch.

    Two parquet files holding the same values with different column order:
    value-identical, byte-different. If the comparator does not flag this, no
    PASS verdict from it is meaningful.
    """
    with tempfile.TemporaryDirectory(prefix="ds_det_fals_") as td:
        a, b = Path(td) / "a", Path(td) / "b"
        (a / "data" / "train").mkdir(parents=True)
        (b / "data" / "train").mkdir(parents=True)
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"readmitted_binary": rng.integers(0, 2, 64),
                           "readmitted_multi": rng.integers(0, 3, 64),
                           "feature_x": rng.normal(size=64)})
        df.to_parquet(a / "data" / "train" / "train_fs.parquet", index=False)
        df[["feature_x", "readmitted_multi", "readmitted_binary"]].to_parquet(
            b / "data" / "train" / "train_fs.parquet", index=False)

        sa, sb = snapshot(a), snapshot(b)
        bytes_differ = sa["data/train/train_fs.parquet"]["sha256"] != \
            sb["data/train/train_fs.parquet"]["sha256"]
        diffs = compare_snapshots(sa, sb, a, b)
        kinds = [d["kind"] for d in diffs]

    detected = bytes_differ and kinds == ["BYTE_DIFFERENCE_ONLY"]
    return {
        "design": ("the same values written with two different column orders — "
                   "value-identical, byte-different, which is what the "
                   "TARGET_COLS hash-randomisation defect produced"),
        "bytes_differ_as_constructed": bytes_differ,
        "classifications_returned": kinds,
        "harness_can_detect_byte_only_difference": detected,
        "why_it_matters": ("a determinism check that has never been shown to "
                           "fail cannot distinguish 'reproducible' from 'not "
                           "looking'. If this arm fails the verdict is UNKNOWN, "
                           "never PASS."),
    }


# ── sandboxed runs ───────────────────────────────────────────────────────────

def _probe_hash_randomisation(env: dict) -> int:
    """
    Ask a subprocess in `env` for the hash of a fixed string tuple.

    This asserts the MECHANISM: if the two run environments return the same
    value, they are not exercising different hash randomisation and a
    set-iteration-order defect could not manifest in either. Verifying that the
    env var was *set* would not establish this — `PYTHONHASHSEED=0` disables
    randomisation entirely, so the value must be observed, not assumed.
    """
    code = f"import sys; print(hash({PROBE_STRINGS!r}))"
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def _make_sandbox(dest: Path) -> None:
    for rel in SANDBOX_INPUTS:
        src = ROOT / rel
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for rel in SANDBOX_MKDIRS:
        (dest / rel).mkdir(parents=True, exist_ok=True)


def _run_stage(sandbox: Path, script: str, env: dict) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, script], cwd=sandbox, env=env,
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return {"script": script, "returncode": proc.returncode,
            "seconds": round(time.perf_counter() - t0, 2),
            "stderr_tail": proc.stderr[-2000:] if proc.returncode else ""}


def run_once(sandbox: Path, stage: str, hashseed: str) -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    env["PYTHONPATH"] = str(sandbox)
    env["MPLBACKEND"] = "agg"
    # See reproduce.py: a Windows parent capturing the pipe gives the
    # child a cp1252 stdout, and any non-ASCII print kills the stage.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    probe = _probe_hash_randomisation(env)

    _make_sandbox(sandbox)
    runs = []
    for script in STAGES[stage]:
        r = _run_stage(sandbox, script, env)
        runs.append(r)
        if r["returncode"] != 0:
            raise RuntimeError(
                f"stage {script} failed under PYTHONHASHSEED={hashseed} "
                f"(rc={r['returncode']}):\n{r['stderr_tail']}")
    return {"hashseed": hashseed, "probe_hash": probe, "stages": runs}


# ── verdict ──────────────────────────────────────────────────────────────────

def decide_verdict(fals: dict, randomisation_differs: bool,
                   diffs: list[dict], n_compared: int) -> tuple[str, str]:
    """
    Decide PASS / FAIL / UNKNOWN.

    A green result requires BOTH that nothing differed AND that the check was in
    a position to notice if it had. The two UNKNOWN branches exist because in
    each of them the check would report "no differences" whether or not a defect
    was present — an assertion that cannot fail is not a check (R6). UNKNOWN
    exits non-zero, exactly like FAIL: an unfalsifiable green is worse than a red.
    """
    if not fals["harness_can_detect_byte_only_difference"]:
        return "UNKNOWN", ("the comparator failed its own falsification arm, so "
                           "it cannot distinguish reproducible from unchecked")
    if not randomisation_differs:
        return "UNKNOWN", ("both runs saw the same string-hash randomisation, so "
                           "a set-iteration-order defect could not have manifested "
                           "in either — this check would pass whether or not one "
                           "existed")
    if diffs:
        return "FAIL", f"{len(diffs)} artifact(s) differ between runs"
    return "PASS", (f"{n_compared} artifacts byte-identical across two runs under "
                    f"different string-hash randomisation")


# ── entry point ──────────────────────────────────────────────────────────────

def run_determinism_check(stage: str = "pipeline",
                          seeds: tuple[str, str] = ("0", "12345"),
                          keep: bool = False) -> dict:
    logger.info("=" * 78)
    logger.info("DriftSentinel — Tier 2C.4  determinism check")
    logger.info("=" * 78)

    fals = falsification_arm()
    logger.info(f"Falsification arm: comparator detects a byte-only difference "
                f"= {fals['harness_can_detect_byte_only_difference']}")

    workdir = Path(tempfile.mkdtemp(prefix="ds_determinism_"))
    try:
        logger.info(f"Sandbox: {workdir}")
        meta, snaps, roots = [], [], []
        for tag, seed in zip("ab", seeds):
            box = workdir / f"run_{tag}"
            logger.info(f"  run {tag}: stage={stage} PYTHONHASHSEED={seed}")
            info = run_once(box, stage, seed)
            logger.info(f"    probe hash {info['probe_hash']}  "
                        f"({sum(s['seconds'] for s in info['stages']):.1f}s)")
            meta.append(info)
            roots.append(box)
            snaps.append(snapshot(box))

        seeds_differ = meta[0]["hashseed"] != meta[1]["hashseed"]
        randomisation_differs = meta[0]["probe_hash"] != meta[1]["probe_hash"]
        diffs = compare_snapshots(snaps[0], snaps[1], roots[0], roots[1])
        n_compared = len(set(snaps[0]) | set(snaps[1]))
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)

    verdict, reason = decide_verdict(fals, randomisation_differs, diffs, n_compared)

    # Coverage, stated rather than implied. A check reports on what the stage
    # actually produced; anything in the repository that this stage does NOT
    # regenerate is outside its reach, and saying "6 artifacts byte-identical"
    # without saying which 6 reads as broader coverage than it is.
    in_repo = set(snapshot(ROOT))
    covered = set(snaps[0]) | set(snaps[1])
    coverage = {
        "artifacts_produced_by_this_stage": sorted(covered),
        "artifacts_present_in_repo_but_not_produced_by_this_stage":
            sorted(in_repo - covered),
        "note": ("the second list is NOT checked for determinism by this run. "
                 "Those files exist in the repository because the modules that "
                 "write them were run standalone; `pipelines/pipeline.py` does "
                 "not write them, so a clean clone running only the pipeline "
                 "would not have them at all. Reported here because a coverage "
                 "gap that is not stated is indistinguishable from coverage."),
    }

    report = {
        "phase": "2C.4",
        "title": "Determinism verified by artifact hash, not by metric equality",
        "stage": stage,
        "scripts": STAGES[stage],
        "verdict": verdict,
        "reason": reason,
        "n_artifacts_compared": n_compared,
        "coverage": coverage,
        "differences": diffs,
        "seeds_requested": list(seeds),
        "seeds_differ": seeds_differ,
        "hash_randomisation_differs_between_runs": randomisation_differs,
        "probe": {"strings": list(PROBE_STRINGS),
                  "hash_run_a": meta[0]["probe_hash"],
                  "hash_run_b": meta[1]["probe_hash"],
                  "why": ("PYTHONHASHSEED is observed through its effect, not "
                          "read back from the environment: setting it to 0 "
                          "disables randomisation, so the env var alone does "
                          "not establish that the two runs differ")},
        "runs": meta,
        "falsification_arm": fals,
        "compared_globs": list(COMPARED_GLOBS),
        "excluded": EXCLUDED_PATTERNS,
        "defect_class_targeted": (
            "value-identical, byte-different — `TARGET_COLS` is a set and Python "
            "randomises string hashing per process, so output column order "
            "changed on every run. Identical values, identical metrics, "
            "different bytes. Dangerous because the adversarial modules index "
            "positionally."),
        "reproducibility": {"python": platform.python_version(),
                            "numpy": np.__version__, "pandas": pd.__version__,
                            "platform": platform.platform()},
    }

    out = REPORTS_DIR / "determinism.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("-" * 62)
    logger.info(f"VERDICT: {verdict} — {reason}")
    for d in diffs:
        logger.error(f"  {d['kind']:<24} {d['artifact']}")
    logger.info(f"Report: {out.name}")
    logger.info("=" * 78)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="DriftSentinel determinism check")
    ap.add_argument("--stage", default="pipeline", choices=sorted(STAGES))
    ap.add_argument("--seeds", nargs=2, default=["0", "12345"],
                    metavar=("SEED_A", "SEED_B"))
    ap.add_argument("--keep", action="store_true",
                    help="keep the sandboxes for inspection")
    args = ap.parse_args()

    r = run_determinism_check(args.stage, tuple(args.seeds), args.keep)
    print(f"\nDeterminism: {r['verdict']} — {r['reason']}")
    print(f"  artifacts compared : {r['n_artifacts_compared']}")
    print(f"  hash randomisation : run_a={r['probe']['hash_run_a']} "
          f"run_b={r['probe']['hash_run_b']} "
          f"(differ={r['hash_randomisation_differs_between_runs']})")
    print(f"  falsification arm  : "
          f"{r['falsification_arm']['harness_can_detect_byte_only_difference']}")
    for d in r["differences"]:
        print(f"  {d['kind']:<24} {d['artifact']}")
    return 0 if r["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
