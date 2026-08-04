"""
Tier 3 guard: the deployed console must run on the SLIM runtime.

WHY THIS TEST EXISTS
    Streamlit Community Cloud installs the repository-root `requirements.txt`
    and nothing else. That file is deliberately small — streamlit, pandas,
    numpy, pyarrow, altair — because `app.py` loads no model and reads a 30 KB
    precomputed bundle. If anything in the app's import graph reaches for
    scikit-learn, lightgbm, shap, scipy or seaborn, the deployment breaks at
    build time on the cloud and works perfectly on the developer's machine,
    where all of them happen to be installed.

    That is precisely the failure mode this repository has already been bitten
    by: seven hardcoded absolute paths that worked on exactly one machine. The
    check therefore does not ask "does `import app` succeed here" — it succeeds
    here either way. It **makes the heavy packages unimportable first**, so the
    assertion is false when the property is false (CLAUDE.md R6).
"""

import builtins
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Present in the dev environment, absent on Streamlit Community Cloud.
HEAVY = ("sklearn", "lightgbm", "shap", "scipy", "seaborn", "matplotlib")

# Declared in requirements.txt. Anything else the app imports is a deploy risk.
RUNTIME_OK = {"streamlit", "pandas", "numpy", "pyarrow", "altair"}


@pytest.fixture
def block_heavy(monkeypatch):
    """Make the research stack unimportable, as it is on the deploy target."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        top = name.split(".")[0]
        if top in HEAVY:
            raise ModuleNotFoundError(
                f"No module named {top!r} (blocked: not in requirements.txt)")
        return real_import(name, *args, **kwargs)

    for mod in list(sys.modules):
        if mod.split(".")[0] in HEAVY:
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.delitem(sys.modules, "app", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.syspath_prepend(str(ROOT))


def test_the_guard_itself_blocks(block_heavy):
    """If this passes trivially the whole test file proves nothing."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sklearn")


def test_console_imports_without_the_research_stack(block_heavy):
    """
    The console must import with sklearn / lightgbm / shap / scipy unavailable.

    A failure here means the app would build fine locally and fail on Streamlit
    Community Cloud.
    """
    app = importlib.import_module("app")
    assert app is not None


def test_requirements_txt_is_the_slim_runtime():
    """
    The ROOT requirements.txt is what the deploy target installs. If the full
    research environment ends up back in it, the cold-start budget is gone and
    the build may time out — so this is a structural assertion, not a style one.
    """
    lines = [ln.strip() for ln in
             (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    # Comments explain WHY the heavy packages are excluded, so they name them.
    # Scanning the whole file would flag its own rationale.
    requirement_lines = [ln for ln in lines if ln and not ln.startswith("#")]
    pkgs = {ln.split("==")[0].split("#")[0].strip().lower()
            for ln in requirement_lines}
    assert pkgs == RUNTIME_OK, f"root requirements.txt drifted: {pkgs}"
    joined = " ".join(requirement_lines).lower()
    for heavy in ("scikit-learn", "lightgbm", "shap", "seaborn", "pytest"):
        assert heavy not in joined, (
            f"{heavy} is in the console runtime; it belongs in requirements-dev.txt")


def test_dev_requirements_include_the_runtime():
    """One install must cover both, or the two files drift apart."""
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "-r requirements.txt" in dev
    for needed in ("lightgbm", "scikit-learn", "pytest", "shap"):
        assert needed in dev.lower(), f"{needed} missing from the dev environment"


def test_demo_bundle_is_small_enough_to_cold_start():
    """
    The console's whole data payload. If this grows into megabytes the cold-start
    claim in the deployment doc stops being true, and nobody would notice from
    the app itself.
    """
    demo = ROOT / "app" / "demo_data"
    total = sum(p.stat().st_size for p in demo.glob("*") if p.is_file())
    assert total < 512 * 1024, f"demo payload is {total / 1024:.0f} KB (budget 512 KB)"


def test_no_model_artifact_is_loaded_by_the_console():
    """
    The console is fed precomputed evidence on purpose. Loading a pickle would
    reintroduce the research stack through the back door AND the version-skew
    fragility documented in docs/TIER_1_7_SCOPE.md.
    """
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    for forbidden in ("pickle.load", "joblib.load", "outputs/models", "lgbm_v1"):
        assert forbidden not in src, f"app.py references {forbidden!r}"


# ── FINAL: the README traceability check must be able to fail ────────────────

def test_readme_traceability_matcher_rejects_an_invented_number():
    """
    The traceability check reports PASS. That is only meaningful if it CAN
    report otherwise — a matcher generous enough to find anything would pass on
    a fabricated claim, which is precisely what it exists to catch.
    """
    from src.monitoring.verify_readme_claims import _appears, corpus_values

    corpus = '{"auc": 0.62546, "f1": 0.19645}'
    values = corpus_values(corpus)

    # Present at the precision the README would quote it to.
    assert _appears("0.6255", corpus, values), "rounding match failed"
    assert _appears("0.62546", corpus, values), "exact match failed"
    # Absent. If this returns True the whole check is decorative.
    assert not _appears("0.9999", corpus, values), "matcher accepts a fabricated number"
    assert not _appears("0.6355", corpus, values), "matcher is too loose"


def test_readme_traceability_exemptions_are_declared_with_reasons():
    """An inferred exemption would let the check pass on anything it can't find."""
    from src.monitoring.verify_readme_claims import EXEMPT

    assert EXEMPT, "exemptions must be declared, not implicit"
    for value, reason in EXEMPT.items():
        assert isinstance(reason, str) and len(reason) > 8, (
            f"{value} is exempt without a usable reason")
