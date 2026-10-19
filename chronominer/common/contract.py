"""Agent contract: exit codes and the one-line JSON run summary.

Exit codes: 0 when every item completed, 1 when an item failed or no item
could be processed, 2 for usage and configuration errors, 130 on interrupt.
With ``--json`` a front end writes exactly one JSON summary line to stdout on
every exit path; every summary carries a ``dry_run`` flag. A run summary
reports the run's tokens under :data:`USAGE_KEY` (see :func:`usage_field`).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from enum import IntEnum
from typing import Any, TextIO

from .usage import Usage

USAGE_KEY = "usage"
TOTAL_ROLE = "total"


class ExitCode(IntEnum):
    """Process exit codes shared by the command-line front ends."""

    OK = 0
    FAILURE = 1
    USAGE = 2
    INTERRUPTED = 130


def exit_code_for(*, failed: int) -> ExitCode:
    """Return the exit code of a finished run: 1 when any item failed, else 0."""
    return ExitCode.FAILURE if failed > 0 else ExitCode.OK


def usage_field(by_role: Mapping[str, Usage] | None = None) -> dict[str, Any]:
    """Return the token counts of a run per role plus their sum under ``total``.

    Each value maps the :class:`~.usage.Usage` field names to counts; with
    no usage only the zero ``total`` remains.
    """
    roles = dict(by_role or {})
    if TOTAL_ROLE in roles:
        raise ValueError(f"{TOTAL_ROLE!r} is reserved for the sum of the roles")
    total = Usage()
    field: dict[str, Any] = {}
    for role, usage in roles.items():
        field[role] = usage.as_dict()
        total = total + usage
    field[TOTAL_ROLE] = total.as_dict()
    return field


def write_json_line(stream: TextIO, obj: Mapping[str, Any]) -> None:
    """Write *obj* as one JSON line to *stream* and flush.

    Non-ASCII text is written as is; when the stream's encoding cannot
    represent it (a legacy console code page), the line falls back to
    ASCII ``\\uXXXX`` escapes, which decode to the same object.
    """
    try:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        stream.write(line)
    except UnicodeEncodeError:
        stream.write(json.dumps(obj, ensure_ascii=True) + "\n")
    stream.flush()


class SummaryEmitter:
    """Writes the JSON run summary at most once.

    Pass ``stream=None`` when no summary was requested; :meth:`emit` then
    writes nothing. Later calls after the first emission are ignored, so an
    interrupt handler can emit a fallback summary unconditionally.
    """

    def __init__(self, stream: TextIO | None, *, dry_run: bool = False) -> None:
        self._stream = stream
        self._dry_run = dry_run
        self._emitted = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """Whether a summary line was requested."""
        return self._stream is not None

    @property
    def emitted(self) -> bool:
        """Whether the summary line has been written."""
        return self._emitted

    def emit(self, fields: Mapping[str, Any]) -> bool:
        """Write ``{"dry_run": ..., **fields}``; return True if written now."""
        with self._lock:
            if self._stream is None or self._emitted:
                return False
            self._emitted = True
            summary: dict[str, Any] = {"dry_run": self._dry_run}
            summary.update(fields)
            write_json_line(self._stream, summary)
            return True
