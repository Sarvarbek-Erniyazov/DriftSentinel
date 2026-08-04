"""
DriftSentinel — artifact write guard (Tier 1.7, P1)

WHY
    Detectors wrote reports to FIXED paths and silently clobbered whatever was
    there. During this remediation that behaviour:

      * would have destroyed the original entry-cohort alert artifacts during
        the Tier 0 regime sweep (worked around by monkey-patching ALERTS_DIR to
        a scratch directory in src/investigation/split_regimes.py)
      * did overwrite `outputs/log/concept_drift_val_test.json` during Tier 1.5
        verification; the original was recovered from git
      * makes the CLAUDE.md preservation rule ("never delete evidence of the
        original framing") unenforceable, because the code deletes it silently

    The audit trail is part of the contribution. Overwriting it must be a
    deliberate act, not the default.

BEHAVIOUR
    write_artifact() refuses to overwrite an existing file unless told to. When
    it does overwrite, it first copies the existing file into
    outputs/reports/superseded/ so the prior evidence survives.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUPERSEDED_DIR = ROOT / "outputs" / "reports" / "superseded"


class ArtifactOverwriteError(RuntimeError):
    """Raised when a write would destroy an existing artifact without consent."""


def preserve_existing(path: Path, subdir: str = "auto") -> Path | None:
    """Copy an existing artifact into superseded/ before it is replaced."""
    path = Path(path)
    if not path.exists():
        return None
    dest_dir = SUPERSEDED_DIR / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = dest_dir / f"{path.stem}.{stamp}{path.suffix}"
    shutil.copy2(path, dest)
    return dest


def write_artifact(path, payload, *, overwrite: bool = False,
                   preserve: bool = True, indent: int = 2,
                   default=str) -> Path:
    """
    Write a JSON artifact, refusing to silently destroy an existing one.

    Parameters
    ----------
    overwrite : must be True to replace an existing file. The default is False
                so that clobbering is always a deliberate choice.
    preserve  : when overwriting, copy the existing file to superseded/ first.

    Raises
    ------
    ArtifactOverwriteError
        if the target exists and overwrite is False.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if not overwrite:
            raise ArtifactOverwriteError(
                f"{path} already exists. Pass overwrite=True to replace it "
                f"(the previous version will be copied to "
                f"{SUPERSEDED_DIR.relative_to(ROOT)}/ first), or write to a "
                f"run-scoped path. Refusing to destroy an existing artifact.")
        if preserve:
            preserve_existing(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, default=default)
    return path


def write_dataframe(path, df, *, overwrite: bool = False,
                    preserve: bool = True, **to_csv_kwargs) -> Path:
    """CSV equivalent of write_artifact, with the same refusal semantics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise ArtifactOverwriteError(
                f"{path} already exists. Pass overwrite=True to replace it.")
        if preserve:
            preserve_existing(path)
    to_csv_kwargs.setdefault("index", False)
    df.to_csv(path, **to_csv_kwargs)
    return path
