"""
Tier 1.7 P1 guards: artifact overwrite protection and model provenance.
"""

import json

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.monitoring.artifact_io import (
    ArtifactOverwriteError, preserve_existing, write_artifact, write_dataframe,
)
from src.monitoring import model_io
from src.monitoring.model_io import (
    ArtifactVersionError, describe_legacy_artifact, load_model, meta_path_for,
    save_model,
)


# ── overwrite guard ───────────────────────────────────────────────────────

def test_refuses_to_overwrite_by_default(tmp_path):
    p = tmp_path / "report.json"
    write_artifact(p, {"v": 1})
    with pytest.raises(ArtifactOverwriteError, match="already exists"):
        write_artifact(p, {"v": 2})
    assert json.loads(p.read_text())["v"] == 1, "original was modified despite refusal"


def test_overwrite_preserves_the_previous_version(tmp_path, monkeypatch):
    monkeypatch.setattr("src.monitoring.artifact_io.SUPERSEDED_DIR", tmp_path / "sup")
    p = tmp_path / "report.json"
    write_artifact(p, {"v": 1})
    write_artifact(p, {"v": 2}, overwrite=True)
    assert json.loads(p.read_text())["v"] == 2
    kept = list((tmp_path / "sup").rglob("report.*.json"))
    assert len(kept) == 1
    assert json.loads(kept[0].read_text())["v"] == 1, "prior evidence not preserved"


def test_dataframe_guard_matches_json_guard(tmp_path):
    p = tmp_path / "t.csv"
    df = pd.DataFrame({"a": [1, 2]})
    write_dataframe(p, df)
    with pytest.raises(ArtifactOverwriteError):
        write_dataframe(p, df)


def test_first_write_needs_no_permission(tmp_path):
    p = tmp_path / "new.json"
    write_artifact(p, {"ok": True})
    assert p.exists()


def test_preserve_existing_returns_none_when_absent(tmp_path):
    assert preserve_existing(tmp_path / "nope.json") is None


# ── model provenance ──────────────────────────────────────────────────────

def _fitted():
    m = LogisticRegression()
    m.fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    return m


def test_save_writes_a_provenance_sidecar(tmp_path):
    p = tmp_path / "m.joblib"
    save_model(_fitted(), p)
    meta = json.loads(meta_path_for(p).read_text())
    assert meta["versions"]["scikit-learn"]
    assert meta["sha256"] and meta["class"].endswith("LogisticRegression")


def test_roundtrip_loads_cleanly(tmp_path):
    p = tmp_path / "m.joblib"
    save_model(_fitted(), p)
    assert load_model(p).predict([[3.0]])[0] in (0, 1)


def test_version_mismatch_raises_instead_of_warning(tmp_path):
    """The shipped behaviour was a warning nothing acted on."""
    p = tmp_path / "m.joblib"
    save_model(_fitted(), p)
    mp = meta_path_for(p)
    meta = json.loads(mp.read_text())
    meta["versions"]["scikit-learn"] = "0.0.1-not-a-real-version"
    mp.write_text(json.dumps(meta))
    with pytest.raises(ArtifactVersionError, match="scikit-learn"):
        load_model(p)
    with pytest.warns(RuntimeWarning):
        load_model(p, strict=False)          # opt-out still possible, but explicit


def test_tampered_artifact_is_detected(tmp_path):
    p = tmp_path / "m.joblib"
    save_model(_fitted(), p)
    p.write_bytes(p.read_bytes() + b"\x00")
    with pytest.raises(ArtifactVersionError, match="content hash"):
        load_model(p)


def test_missing_sidecar_warns_rather_than_loading_silently(tmp_path):
    p = tmp_path / "legacy.joblib"
    save_model(_fitted(), p)
    meta_path_for(p).unlink()
    with pytest.warns(RuntimeWarning, match="no provenance sidecar"):
        load_model(p)


def test_legacy_artifacts_without_a_sidecar_are_describable(tmp_path):
    """
    An artifact with no provenance must report UNKNOWN rather than load silently.

    This used to be asserted against the SHIPPED `lgbm_v1.pkl`, encoding the gap
    "0 of 4 models carry provenance" as an expectation. Tier 2C.7 closed that gap
    by routing `trainer.py` and `registry.py` through `save_model`, so the test
    started failing — correctly. It is rewritten against a constructed
    sidecar-less artifact, which is what it was actually testing; the shipped
    models are now covered by the assertion below.
    """
    p = tmp_path / "legacy.pkl"
    p.write_bytes(b"not-really-a-model")
    info = describe_legacy_artifact(p)
    assert info["provenance"].startswith("UNKNOWN")
    assert info["has_sidecar"] is False


def test_every_shipped_model_carries_provenance():
    """
    Tier 2C.7. The traceability audit measured 0 of 4 models with a sidecar while
    `save_model` existed, was tested, and had zero callers — a capability nothing
    calls protects nothing.

    Checked by the presence of the sidecar FILE and by its recorded hash matching
    the artifact's actual bytes. Checking that `save_model` exists would pass
    while the property was false, which is the whole of R6.
    """
    import hashlib
    import json as _json

    models_dir = model_io.ROOT / "outputs" / "models"
    shipped = sorted(models_dir.glob("*.pkl"))
    if not shipped:
        pytest.skip("no models present; run src/pipelines/reproduce.py first")

    for m in shipped:
        meta_path = model_io.meta_path_for(m)
        assert meta_path.exists(), f"{m.name} has no provenance sidecar"
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["versions"]["scikit-learn"], f"{m.name}: no versions recorded"
        actual = hashlib.sha256(m.read_bytes()).hexdigest()
        assert meta["sha256"] == actual, (
            f"{m.name}: sidecar hash does not match the artifact it describes")
