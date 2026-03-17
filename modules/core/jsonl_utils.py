"""Shared JSONL read/write utilities for ChronoMiner processing pipelines.

Provides a consistent interface for line-delimited JSON I/O used by both the
extraction pipeline (temp JSONL for chunk results) and the line-range
readjuster (temp JSONL for per-range boundary decisions).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Set

from modules.core.path_utils import ensure_path_safe

logger = logging.getLogger(__name__)


class JsonlWriter:
    """Context manager for writing JSONL records with auto-flush.

    Usage::

        with JsonlWriter(path, mode="w") as writer:
            writer.write_record({"custom_id": "doc-chunk-1", ...})
    """

    def __init__(self, path: Path, *, mode: str = "w") -> None:
        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")
        self._path = ensure_path_safe(path)
        self._mode = mode
        self._handle = None

    def __enter__(self) -> "JsonlWriter":
        self._handle = self._path.open(self._mode, encoding="utf-8")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write_record(self, record: Dict[str, Any]) -> None:
        """Serialize *record* as a single JSON line, write, and flush."""
        if self._handle is None:
            raise RuntimeError("JsonlWriter is not open; use as context manager")
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()


def read_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield parsed dicts from a JSONL file.

    Skips empty lines and lines that fail JSON parsing (logged as warnings).
    Yields nothing if the file does not exist.
    """
    safe_path = ensure_path_safe(path)
    if not safe_path.exists():
        return

    with safe_path.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSON at %s:%d: %s", path, line_num, exc
                )


_DEFAULT_ID_PATTERN = re.compile(r"-(chunk|range)-(\d+)$")


def extract_completed_ids(
    path: Path,
    *,
    id_pattern: re.Pattern[str] = _DEFAULT_ID_PATTERN,
) -> Set[int]:
    """Read a JSONL file and return 1-based indices from ``custom_id`` fields.

    The *id_pattern* must contain a capturing group whose last group is the
    numeric index (e.g. ``r"-(chunk|range)-(\\d+)$"``).  Records without a
    ``custom_id`` or whose ``custom_id`` does not match are silently skipped.
    """
    completed: Set[int] = set()
    for record in read_jsonl_records(path):
        custom_id = record.get("custom_id", "")
        if not custom_id:
            continue
        match = id_pattern.search(str(custom_id))
        if match:
            try:
                completed.add(int(match.group(match.lastindex)))
            except (ValueError, IndexError):
                pass
    return completed
