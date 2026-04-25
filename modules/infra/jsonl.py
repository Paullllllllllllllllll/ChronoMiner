"""Shared JSONL read/write utilities for ChronoMiner processing pipelines.

Provides a consistent interface for line-delimited JSON I/O used by both the
extraction pipeline (temp JSONL for chunk results) and the line-range
readjuster (temp JSONL for per-range boundary decisions).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
from pathlib import Path
from collections.abc import Iterator
from typing import IO, Any

from modules.infra.paths import ensure_path_safe

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
        self._handle: IO[Any] | None = None

    def __enter__(self) -> "JsonlWriter":
        self._handle = self._path.open(self._mode, encoding="utf-8")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write_record(self, record: dict[str, Any]) -> None:
        """Serialize *record* as a single JSON line, write, and flush."""
        if self._handle is None:
            raise RuntimeError("JsonlWriter is not open; use as context manager")
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()


def read_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
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
) -> set[int]:
    """Read a JSONL file and return 1-based indices from ``custom_id`` fields.

    The *id_pattern* must contain a capturing group whose last group is the
    numeric index (e.g. ``r"-(chunk|range)-(\\d+)$"``). Records without a
    ``custom_id`` or whose ``custom_id`` does not match are silently skipped.
    """
    completed: set[int] = set()
    for record in read_jsonl_records(path):
        custom_id = record.get("custom_id", "")
        if not custom_id:
            continue
        match = id_pattern.search(str(custom_id))
        if match:
            try:
                completed.add(int(match.group(match.lastindex or 0)))
            except (ValueError, IndexError):
                pass
    return completed


# ---------------------------------------------------------------------------
# JSONL header utilities for staleness detection and resume validation
# ---------------------------------------------------------------------------

_JSONL_HEADER_VERSION = 1


def compute_ranges_fingerprint(line_ranges_file: Path) -> str:
    """Return the SHA-256 hex digest of a ``_line_ranges.txt`` file's bytes.

    The fingerprint captures range count, boundaries, and formatting so that
    any change to the input ranges (e.g. regeneration with a different
    ``tokens_per_chunk``) produces a different fingerprint.
    """
    return hashlib.sha256(
        ensure_path_safe(line_ranges_file).read_bytes()
    ).hexdigest()


def build_jsonl_header(
    *,
    ranges_fingerprint: str,
    total_ranges: int,
    boundary_type: str,
    model_name: str,
    context_window: int,
    matching_config: dict[str, Any] | None = None,
    retry_config: dict[str, Any] | None = None,
    prompt_hash: str | None = None,
    context_path: str | None = None,
) -> dict[str, Any]:
    """Construct a JSONL header record for a line-range adjustment run.

    The header is written as the first line of the temp JSONL and stores all
    settings needed to validate whether the JSONL can be used for resume. It
    uses ``"jsonl_header"`` as its top-level key (instead of ``"custom_id"``)
    so existing functions like ``extract_completed_ids()`` and
    ``_rebuild_ranges_from_jsonl()`` naturally skip it.
    """
    return {
        "jsonl_header": {
            "version": _JSONL_HEADER_VERSION,
            "ranges_fingerprint": ranges_fingerprint,
            "total_ranges": total_ranges,
            "boundary_type": boundary_type,
            "model_name": model_name,
            "context_window": context_window,
            "matching_config": matching_config,
            "retry_config": retry_config,
            "prompt_hash": prompt_hash,
            "context_path": context_path,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    }


def read_jsonl_header(path: Path) -> dict[str, Any] | None:
    """Read the header record from the first line of a JSONL file.

    Returns the ``jsonl_header`` dict if the first line contains a valid
    header record, or ``None`` otherwise (missing file, empty file, non-JSON
    first line, or first line without ``"jsonl_header"`` key).
    """
    safe_path = ensure_path_safe(path)
    if not safe_path.exists():
        return None
    try:
        with safe_path.open("r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
        if not first_line:
            return None
        record = json.loads(first_line)
        return record.get("jsonl_header")
    except (json.JSONDecodeError, OSError):
        return None


def validate_jsonl_header(
    header: dict[str, Any],
    *,
    ranges_fingerprint: str,
    boundary_type: str,
    model_name: str,
    context_window: int,
    matching_config: dict[str, Any] | None = None,
    retry_config: dict[str, Any] | None = None,
    prompt_hash: str | None = None,
) -> bool:
    """Check whether a JSONL header matches the current run settings.

    Returns ``True`` only when all provided fields match. ``prompt_hash`` is
    compared only when both the header and the caller supply a non-None value.
    """
    if header.get("ranges_fingerprint") != ranges_fingerprint:
        return False
    if header.get("boundary_type") != boundary_type:
        return False
    if header.get("model_name") != model_name:
        return False
    if header.get("context_window") != context_window:
        return False
    if matching_config is not None and header.get("matching_config") != matching_config:
        return False
    if retry_config is not None and header.get("retry_config") != retry_config:
        return False
    if (
        prompt_hash is not None
        and header.get("prompt_hash") is not None
        and header.get("prompt_hash") != prompt_hash
    ):
        return False
    return True


def finalize_jsonl_header(
    path: Path,
    *,
    stats: dict[str, int],
    source_file: str | None = None,
) -> None:
    """Update the JSONL header with completion stats after a successful run.

    Reads the entire file, merges completion fields into the header record,
    and rewrites the file. This replaces the former ``.adjusted_meta``
    sidecar with a single authoritative metadata source.
    """
    safe_path = ensure_path_safe(path)
    if not safe_path.exists():
        logger.warning("Cannot finalize header: JSONL file not found: %s", path)
        return

    lines = safe_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        logger.warning("Cannot finalize header: JSONL file is empty: %s", path)
        return

    try:
        first_record = json.loads(lines[0].strip())
    except json.JSONDecodeError:
        logger.warning(
            "Cannot finalize header: first line is not valid JSON: %s", path
        )
        return

    header = first_record.get("jsonl_header")
    if header is None:
        logger.warning(
            "Cannot finalize header: no jsonl_header key in first line: %s", path
        )
        return

    header["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if source_file is not None:
        header["source_file"] = source_file
    for key, value in stats.items():
        header[key] = value

    lines[0] = json.dumps(first_record, ensure_ascii=False) + "\n"
    safe_path.write_text("".join(lines), encoding="utf-8")
    logger.info("Finalized JSONL header with completion stats: %s", path)


def is_jsonl_adjustment_complete(
    line_ranges_file: Path,
    *,
    boundary_type: str,
    context_window: int,
    model_name: str,
    matching_config: dict[str, Any] | None = None,
    retry_config: dict[str, Any] | None = None,
    ranges_fingerprint: str | None = None,
    prompt_hash: str | None = None,
) -> bool:
    """Check whether a completed adjustment JSONL exists with matching settings.

    Replaces the former ``.adjusted_meta``-based ``is_adjustment_current()``
    check. Returns ``True`` only when the JSONL header matches all config
    settings AND has a ``completed_at`` timestamp (indicating the run finished
    successfully).

    The ``ranges_fingerprint`` is intentionally NOT compared here because a
    successful adjustment rewrites the ``_line_ranges.txt`` file, changing
    its fingerprint. The ``completed_at`` flag provides the completion
    guarantee; config fields ensure the settings match.
    """
    stem = line_ranges_file.stem
    jsonl_path = line_ranges_file.parent / f"{stem}_adjust_temp.jsonl"

    header = read_jsonl_header(jsonl_path)
    if header is None:
        return False

    if header.get("completed_at") is None:
        return False

    if header.get("boundary_type") != boundary_type:
        return False
    if header.get("model_name") != model_name:
        return False
    if header.get("context_window") != context_window:
        return False
    if matching_config is not None and header.get("matching_config") != matching_config:
        return False
    if retry_config is not None and header.get("retry_config") != retry_config:
        return False

    return True


def compute_stats_from_jsonl(path: Path) -> dict[str, int]:
    """Compute adjustment statistics from a completed JSONL.

    Reads all range records (skipping the header) and returns counts for
    adjusted, deleted, and kept-original ranges plus total LLM calls.
    """
    adjusted = 0
    deleted = 0
    kept_original = 0
    already_on_target = 0
    total_llm_calls = 0
    range_count = 0

    for record in read_jsonl_records(path):
        if "jsonl_header" in record:
            continue
        body = record.get("response", {}).get("body", {})
        if not body:
            continue
        range_count += 1
        total_llm_calls += body.get("total_llm_calls", 0)

        if body.get("should_delete", False):
            deleted += 1
        else:
            original = body.get("original_range")
            adj = body.get("adjusted_range")
            if (
                original is not None
                and adj is not None
                and list(original) != list(adj)
            ):
                adjusted += 1
            else:
                kept_original += 1
                decision = body.get("decision", {})
                if decision.get("boundary_already_on_target", False):
                    already_on_target += 1

    return {
        "total_ranges": range_count,
        "ranges_adjusted": adjusted,
        "ranges_deleted": deleted,
        "ranges_kept_original": kept_original,
        "ranges_already_on_target": already_on_target,
        "total_llm_calls": total_llm_calls,
    }
