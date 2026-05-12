"""
DriftSentinel — Centralized Logger
All modules import get_logger() from here.
Each module writes to its own log file under outputs/log/.
"""

import logging
import os
from pathlib import Path


LOG_DIR = Path(r"C:\Users\sharg\Desktop\github\DriftSentinel\outputs\log")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a named logger that writes to:
      - outputs/log/<name>.log  (file, overwrite each run)
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

    file_handler = logging.FileHandler(
        LOG_DIR / f"{name}.log",
        mode="w",
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger