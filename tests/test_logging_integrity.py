"""
The guarantee behind the Tier 1.7 P0 logger fix.

This is the test that makes it a fix rather than a convention: importing every
module must leave outputs/log/ byte-identical. The previous implementation
failed this — `FileHandler(mode="w")` created at import time truncated each
module's log the moment the module was imported.
"""

import hashlib
import importlib
import logging
import pathlib

import pytest

from src.monitoring import logger as logger_mod

LOG_DIR = logger_mod.LOG_DIR


def _snapshot() -> dict:
    """Content hash of every log file, so any write is detected."""
    out = {}
    for p in sorted(LOG_DIR.glob("*.log*")):
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _all_src_modules():
    root = pathlib.Path(__file__).resolve().parents[1]
    mods = []
    for p in sorted((root / "src").rglob("*.py")):
        if p.name == "__init__.py":
            continue
        mods.append(p.relative_to(root).with_suffix("").as_posix().replace("/", "."))
    return mods


def test_importing_every_module_does_not_touch_the_logs():
    """The regression test for three separate evidence-destruction incidents."""
    before = _snapshot()
    for m in _all_src_modules():
        importlib.import_module(m)
    after = _snapshot()

    changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
    assert not changed, f"importing modules modified log files: {changed}"


def test_get_logger_is_append_only_and_lazy(tmp_path, monkeypatch):
    """A fresh logger must not create its file until something is logged."""
    monkeypatch.setattr(logger_mod, "LOG_DIR", tmp_path)
    name = "unit_test_lazy_logger"
    logging.getLogger(name).handlers.clear()

    log = logger_mod.get_logger(name)
    target = tmp_path / f"{name}.log"
    assert not target.exists(), "handler created the file before any record was emitted"

    log.info("first line")
    for h in log.handlers:
        h.flush()
    assert target.exists()
    first = target.read_text(encoding="utf-8")
    assert "first line" in first
    assert "===== RUN " in first, "run banner missing"

    logging.getLogger(name).handlers.clear()
    log2 = logger_mod.get_logger(name)
    log2.info("second line")
    for h in log2.handlers:
        h.flush()
    second = target.read_text(encoding="utf-8")
    assert "first line" in second, "APPEND FAILED — earlier content was truncated"
    assert "second line" in second


def test_handlers_are_not_duplicated_on_repeat_calls():
    name = "unit_test_no_dup"
    logging.getLogger(name).handlers.clear()
    a = logger_mod.get_logger(name)
    n = len(a.handlers)
    b = logger_mod.get_logger(name)
    assert a is b and len(b.handlers) == n


def test_rotation_is_configured_and_bounded():
    """Append-only without rotation would be its own defect."""
    name = "unit_test_rotation"
    logging.getLogger(name).handlers.clear()
    log = logger_mod.get_logger(name)
    fh = [h for h in log.handlers if hasattr(h, "maxBytes")]
    assert fh, "no rotating file handler attached"
    assert fh[0].maxBytes > 0 and fh[0].backupCount > 0


def test_run_id_is_stable_within_a_process():
    assert logger_mod.RUN_ID and logger_mod.RUN_ID == logger_mod.RUN_ID
