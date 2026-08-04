"""
Headless-plotting guarantees.

WHY THIS TEST EXISTS
    The "all figures regenerate" criterion was previously verified by checking
    that files appeared in outputs/figure/. That observable does not distinguish
    headless from GUI behaviour — figures land on disk either way, while
    `plt.show()` also opens a window that must be closed by hand, and would
    block or fail differently in CI.

    This is the same failure mode as the health_check silent-default bug: the
    thing that was checked was not the property that was claimed. The test below
    asserts the property directly — the BACKEND, and the absence of blocking
    calls — rather than a side effect that is consistent with both outcomes.
"""

import ast
import importlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

PLOTTING_MODULES = [
    "src.models.evaluator",
    "src.uncertainty.threshold",
    "src.uncertainty.quantifier",
    "src.uncertainty.calibration",
    "src.adversarial.defense",
    "src.adversarial.robustness",
    "src.uncertainty.threshold_policy",
    "src.investigation.temporal_validity",
    "src.investigation.split_regimes",
]


def _src_files():
    return sorted((ROOT / "src").rglob("*.py"))


def test_backend_is_agg_after_importing_every_plotting_module():
    """The direct property: no interactive backend is ever selected."""
    for m in PLOTTING_MODULES:
        importlib.import_module(m)
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg", (
        f"backend is {matplotlib.get_backend()!r}; an interactive backend opens "
        f"GUI windows and can block or fail in CI")


def test_no_module_contains_a_bare_plt_show():
    """`plt.show()` is blocking under an interactive backend and pointless under Agg."""
    offenders = []
    for f in _src_files():
        src = f.read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "plt.show()" in stripped or ".show()" in stripped and "plt" in stripped:
                offenders.append(f"{f.relative_to(ROOT)}:{i}")
    assert not offenders, f"bare show() calls found: {offenders}"


def test_every_plotting_module_sets_agg_before_importing_pyplot():
    """
    `matplotlib.use()` must run BEFORE pyplot is first imported in the process.
    A central config module would only work if import order were guaranteed —
    it is not, since any module may be the entry point via `python -m`.
    """
    missing = []
    for m in PLOTTING_MODULES:
        f = ROOT / (m.replace(".", "/") + ".py")
        src = f.read_text(encoding="utf-8")
        if "pyplot" not in src:
            continue
        use_at = src.find('matplotlib.use("Agg")')
        pyplot_at = src.find("import matplotlib.pyplot")
        if use_at == -1 or (pyplot_at != -1 and use_at > pyplot_at):
            missing.append(m)
    assert not missing, (
        f"these modules import pyplot without setting Agg first: {missing}")


def test_savefig_calls_are_followed_by_a_close():
    """
    Every savefig must release its figure. 27 savefig calls against 13 closes
    meant figures accumulated in memory across long runs.
    """
    offenders = []
    for f in _src_files():
        lines = f.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if "savefig(" not in line or line.strip().startswith("#"):
                continue
            window = "\n".join(lines[i + 1:i + 4])
            if "plt.close" not in window and "fig.clf" not in window:
                offenders.append(f"{f.relative_to(ROOT)}:{i + 1}")
    assert not offenders, f"savefig without a following close: {offenders}"


def test_plotting_modules_parse_and_expose_no_interactive_calls():
    """Guard against pyplot.ion() / pyplot.pause() sneaking in."""
    banned = {"ion", "pause", "ginput", "waitforbuttonpress"}
    offenders = []
    for f in _src_files():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in banned
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "plt"):
                offenders.append(f"{f.relative_to(ROOT)}:{node.lineno} plt.{node.func.attr}")
    assert not offenders, f"interactive pyplot calls found: {offenders}"
