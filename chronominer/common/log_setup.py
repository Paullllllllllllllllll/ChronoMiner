"""Logging setup for front ends.

Library modules call ``logging.getLogger(__name__)`` and never configure
handlers. A front end calls :func:`configure_logging` once at startup; it
places the handlers on one top-level logger (the root by default), and child
loggers reach them through propagation, so each record is emitted once.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

__all__ = ["FILE_FORMAT", "SIMPLE_FORMAT", "configure_logging"]

SIMPLE_FORMAT = "[%(levelname)s] %(message)s"
FILE_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

_HANDLER_MARKER = "_log_setup_handler"


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Stream handler that writes to whatever ``sys.stderr`` is at emit time."""

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self) -> TextIO:
        return sys.stderr


def configure_logging(
    *,
    logger_name: str = "",
    level: int = logging.WARNING,
    log_file: Path | None = None,
    file_level: int = logging.INFO,
) -> None:
    """Install a stderr handler, and optionally a file handler, on one logger.

    ``logger_name`` selects the logger; a dotted name is reduced to its
    top-level part, and ``""`` is the root logger. Calling again replaces the
    handlers an earlier call installed. The file handler appends UTF-8 text
    to ``log_file`` and creates its folder.
    """
    target = logging.getLogger(logger_name.split(".", 1)[0])
    for existing in list(target.handlers):
        if getattr(existing, _HANDLER_MARKER, False):
            target.removeHandler(existing)
            existing.close()

    console = _StderrHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(SIMPLE_FORMAT))
    handlers: list[logging.Handler] = [console]
    effective = level

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8", delay=True)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        handlers.append(file_handler)
        effective = min(level, file_level)

    for handler in handlers:
        setattr(handler, _HANDLER_MARKER, True)
        target.addHandler(handler)
    target.setLevel(effective)
