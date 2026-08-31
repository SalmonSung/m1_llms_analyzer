"""Single logging entrypoint for the package.

Rationale: services must never configure the root logger themselves (that would
fight with Colab's own handlers and duplicate every line). `get_logger` attaches
exactly one handler to the package logger and is idempotent, so re-running a
notebook cell does not stack handlers.
"""

from __future__ import annotations

import logging
import os
import sys

_PACKAGE_LOGGER = "m1_analyzer"
_CONFIGURED = False


def configure_logging(level: int | str | None = None) -> None:
    """Attach one stdout handler to the package logger. Safe to call repeatedly."""
    global _CONFIGURED
    logger = logging.getLogger(_PACKAGE_LOGGER)
    if level is None:
        level = os.environ.get("M1_LOG_LEVEL", "INFO")
    logger.setLevel(level)
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        # Colab's root logger would otherwise print every record a second time.
        logger.propagate = False
        _CONFIGURED = True


def get_logger(name: str = _PACKAGE_LOGGER) -> logging.Logger:
    configure_logging()
    if name == _PACKAGE_LOGGER:
        return logging.getLogger(name)
    return logging.getLogger(f"{_PACKAGE_LOGGER}.{name}")
