"""
DriftSentinel — Centralized Logger
All modules import get_logger() from here.
Each module writes to its own log file under outputs/log/.

TIER 1.7 (P0) — IMPORTING A MODULE NO LONGER DESTROYS ITS LOG.

    The previous implementation used `FileHandler(path, mode="w")`, and
    `get_logger()` is called at MODULE IMPORT TIME in every module. Opening a
    FileHandler in "w" mode truncates immediately, so simply importing a module
    erased its log. Reading the codebase destroyed the codebase's audit trail —
    the same audit trail the adversarial review called a genuine strength
    ("Every module writes structured, readable logs. This is what made the audit
    possible.").

    It caused three incidents during this remediation:
      1. a verification import loop truncated 9 original pipeline logs (867 lines)
      2. a pytest run truncated concept_drift.log
      3. the workaround for (2) — `git checkout -- outputs/log/` — reverted a
         REGENERATED REPORT along with the logs, so a before/after comparison
         silently compared the original file against itself and appeared to show
         no change

    Incident 3 is why this is P0 rather than housekeeping: the defect did not
    merely destroy evidence, it produced a wrong analytical result that looked
    right.

THE FIX
    delay=True    the file is opened on first EMIT, not on handler creation, so
                  an import that logs nothing touches nothing
    mode="a"      never truncate
    rotation      bounded growth, with history retained
    run banner    each process writes one banner on its first record, so runs
                  stay separable inside an appended file

GUARANTEE
    tests/test_logging_integrity.py asserts that importing every module leaves
    outputs/log/ byte-identical. A convention would rot; the test does not.
"""

import logging
import os
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


# Tier 2C.6 reproducibility: this was a HARDCODED ABSOLUTE PATH to one
# developer's machine, so `pipeline.py` did not reproduce anything from raw
# data on a clean clone -- it read from and wrote to a directory that exists
# nowhere else. On Linux CI the same literal resolves to a RELATIVE folder
# whose name contains backslashes, so artifacts land somewhere harmless-
# looking and the run still 'succeeds'. It worked on exactly one machine,
# which is why nothing caught it. Now derived from this file's location.
ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "outputs" / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_BYTES    = 5 * 1024 * 1024      # 5 MB per file before rotation
BACKUP_COUNT = 3                    # keep 3 rotated generations

# One run id per PROCESS, so every line written by a single execution can be
# tied together and separated from earlier runs in the same file.
RUN_ID = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


class _RunBannerHandler(RotatingFileHandler):
    """
    RotatingFileHandler that writes a run banner before its first record.

    The banner is emitted lazily, inside the first emit(), so a module that is
    imported but never used still writes nothing at all.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._banner_written = False

    def emit(self, record):
        if not self._banner_written:
            self._banner_written = True
            banner = logging.LogRecord(
                name=record.name, level=logging.INFO, pathname=__file__, lineno=0,
                msg=(f"===== RUN {RUN_ID} | pid={os.getpid()} | "
                     f"started {datetime.now():%Y-%m-%d %H:%M:%S} ====="),
                args=(), exc_info=None,
            )
            super().emit(banner)
        super().emit(record)


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a named logger that writes to:
      - outputs/log/<name>.log  (append, rotating, opened on first write)
      - stdout (console)

    Parameters
    ----------
    name : module name, e.g. 'loader', 'validator', 'splitter', 'preprocessor'

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = _RunBannerHandler(
        LOG_DIR / f"{name}.log",
        mode="a",           # NEVER truncate — see the module docstring
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",   # logs contain arrows, check marks and em dashes
        delay=True,         # open on first emit, so import alone touches nothing
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
