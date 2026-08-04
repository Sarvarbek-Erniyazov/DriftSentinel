"""
DriftSentinel — model serialization with version metadata (Tier 1.7, P1)

THREE DISTINCT PICKLE SYMPTOMS, all observed during this remediation:

  1. VERSION SKEW. `outputs/models/lgbm_v1.pkl` was pickled under
     scikit-learn 1.7.2 and raises InconsistentVersionWarning under 1.9.0 on
     every load. A warning is not a guard: the load succeeds and the caller
     proceeds with an object sklearn itself says may be invalid.

  2. `__main__`-BOUND CLASS RESOLUTION. The isotonic calibrator was pickled
     from a script running as `__main__`, so unpickling resolves
     `IsotonicCalibrator` against whichever module is currently `__main__`. A
     function-local import fails; only a module-level import in the entry-point
     module works. This cost a debugging cycle in Tier 1.3 and forced an
     otherwise unnecessary module-level import into threshold_policy.py.

  3. NO PROVENANCE. Nothing recorded which code or library versions produced an
     artifact, so a stale model was indistinguishable from a current one.

FIX
    save_model() writes the estimator with joblib plus a `.meta.json` sidecar
    recording library versions, the git commit, and a content hash.
    load_model() compares the recorded versions against the runtime and FAILS
    LOUDLY by default rather than warning.

The existing v1/v2 pickles cannot be retro-fitted with provenance they never
had; `describe_legacy_artifact()` reports what can still be recovered from them.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[2]


class ArtifactVersionError(RuntimeError):
    """Raised when an artifact was produced by a different library version."""


def _versions() -> dict:
    v = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import lightgbm
        v["lightgbm"] = lightgbm.__version__
    except Exception:
        pass
    return v


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def meta_path_for(model_path) -> Path:
    return Path(model_path).with_suffix(Path(model_path).suffix + ".meta.json")


def save_model(model, path, *, extra: dict | None = None,
               serializer: str = "joblib") -> Path:
    """
    Serialize a model and write a provenance sidecar beside it.

    `serializer="joblib"` (default) is the scikit-learn project's own
    recommendation and handles large numpy arrays better.

    `serializer="pickle"` exists for a specific, stated reason. Tier 2C.7 wired
    this function into `trainer.py` and `registry.py` to close the traceability
    gap "0 of 4 models carry provenance". Eighteen modules read those artifacts
    with `pickle.load`, and a joblib file is NOT readable that way — joblib wraps
    numpy arrays in objects only its own unpickler resolves, so switching the
    format would have silently handed every consumer a wrapper instead of an
    array. So the FORMAT is unchanged and the PROVENANCE is added; those are two
    different gaps and only one is being closed here.

    The other one stays open and named: migrating off pickle is
    `docs/TIER_1_7_SCOPE.md` P1, and it needs all eighteen call sites moved to
    `load_model()` in one change, not a format swap underneath them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if serializer == "joblib":
        joblib.dump(model, path)
    elif serializer == "pickle":
        import pickle
        with open(path, "wb") as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"unknown serializer {serializer!r}")

    meta = {
        "artifact": path.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "versions": _versions(),
        "git_commit": _git_commit(),
        "sha256": _sha256(path),
        "serializer": serializer,
        **(extra or {}),
    }
    with open(meta_path_for(path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def load_model(path, *, strict: bool = True, check: tuple = ("scikit-learn",)):
    """
    Load a model and verify its provenance.

    strict=True (default) RAISES on a version mismatch for any library in
    `check`. The shipped behaviour was an InconsistentVersionWarning that
    nothing acted on — a warning that is always ignored is not a safeguard.

    Artifacts without a sidecar (everything produced before Tier 1.7) load with
    an explicit warning saying provenance is unavailable, rather than silently.
    """
    path = Path(path)
    mp = meta_path_for(path)

    if not mp.exists():
        warnings.warn(
            f"{path.name}: no provenance sidecar — this artifact predates "
            f"Tier 1.7 and its producing versions are unknown. Re-save it with "
            f"save_model() to gain version checking.", RuntimeWarning)
        return joblib.load(path)

    meta = json.loads(mp.read_text(encoding="utf-8"))

    digest = _sha256(path)
    if meta.get("sha256") and digest != meta["sha256"]:
        msg = (f"{path.name}: content hash does not match its sidecar "
               f"(recorded {meta['sha256'][:12]}, actual {digest[:12]}) — the "
               f"artifact changed after it was recorded")
        if strict:
            raise ArtifactVersionError(msg)
        warnings.warn(msg, RuntimeWarning)

    now = _versions()
    drift = {k: (meta["versions"].get(k), now.get(k)) for k in check
             if meta["versions"].get(k) != now.get(k)}
    if drift:
        msg = (f"{path.name}: produced under {drift} (recorded, runtime). "
               f"Predictions may differ from those the artifact was validated "
               f"with.")
        if strict:
            raise ArtifactVersionError(msg)
        warnings.warn(msg, RuntimeWarning)

    return joblib.load(path)


def describe_legacy_artifact(path) -> dict:
    """
    Report what can still be recovered from a pre-Tier-1.7 pickle.

    Used to document, rather than paper over, the artifacts that shipped
    without provenance.
    """
    path = Path(path)
    info = {"artifact": path.name, "exists": path.exists(),
            "has_sidecar": meta_path_for(path).exists(),
            "provenance": "UNKNOWN — produced before Tier 1.7"}
    if path.exists():
        info["sha256"] = _sha256(path)
        info["size_bytes"] = path.stat().st_size
    return info
