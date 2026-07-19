# modules/extract/resume.py

"""
Resume and completeness detection utilities for ChronoMiner extraction.

Provides file-level and chunk-level resume capabilities:
- Detect whether an extraction output is complete, partial, or not started.
- Read/write processing metadata for settings-aware resume.

Line-range adjustment resume is handled by JSONL header validation in
``modules.infra.jsonl``.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from modules.infra.paths import ensure_path_safe

logger = logging.getLogger(__name__)


class FileStatus(Enum):
    """Processing status for a file."""

    NOT_STARTED = "not_started"
    PARTIAL = "partial"
    COMPLETE = "complete"


# ---------------------------------------------------------------------------
# Extraction resume helpers
# ---------------------------------------------------------------------------

METADATA_KEY = "_chronominer_metadata"

# Backwards-compatibility alias: existing tests and callers import
# ``_METADATA_KEY``. New code should use the public ``METADATA_KEY``.
_METADATA_KEY = METADATA_KEY

# Text chunking behaviour version stamped into output metadata. Bumped to 2 when
# the newline-preserving chunk join landed (lines rstripped + joined with "\n"),
# so downstream consumers can tell which chunking behaviour produced a file.
CHUNKING_TEXT_VERSION = 2

# Format-version marker written as the first line of every synchronous temp
# JSONL. On resume, a temp file without the current marker is refused rather
# than merged (its custom_ids may be slice-relative under the pre-2 format,
# which would corrupt resume/merge). No migration is attempted.
TEMP_JSONL_VERSION = 2
TEMP_VERSION_KEY = "_chronominer_temp_version"

# Absolute 1-based unit index embedded in a custom_id: "-chunk-N" (text
# runs) or "-page-N" (visual runs).
_CUSTOM_ID_INDEX_RE = re.compile(r"-(?:chunk|page)-(\d+)$")


def build_temp_header() -> dict[str, Any]:
    """Return the header record written as the first line of a sync temp JSONL."""
    return {TEMP_VERSION_KEY: TEMP_JSONL_VERSION}


def temp_jsonl_version(temp_jsonl_path: Path) -> int | None:
    """Return the format version of a temp JSONL, or ``None`` if unversioned.

    Reads only the first non-empty line; an unversioned (legacy) file returns
    ``None``.
    """
    if not temp_jsonl_path.exists():
        return None
    try:
        with temp_jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    return None
                if isinstance(record, dict) and TEMP_VERSION_KEY in record:
                    value = record[TEMP_VERSION_KEY]
                    return value if isinstance(value, int) else None
                return None
    except OSError:
        return None
    return None


def is_resumable_temp_jsonl(temp_jsonl_path: Path) -> bool:
    """Whether a temp JSONL may be safely resumed under the current format.

    A missing file is resumable (nothing to conflict with); an existing file is
    resumable only if it carries the current ``TEMP_JSONL_VERSION`` marker.
    """
    if not temp_jsonl_path.exists():
        return True
    return temp_jsonl_version(temp_jsonl_path) == TEMP_JSONL_VERSION


def build_extraction_metadata(
    *,
    schema_name: str,
    model_name: str,
    chunking_method: str,
    total_chunks: int,
    timestamp: str | None = None,
    chunk_slice_info: dict[str, Any] | None = None,
    partial: bool = False,
    failed_chunks: list[int] | None = None,
    image_provenance: dict[str, Any] | None = None,
    chunking_text_version: int | None = None,
) -> dict[str, Any]:
    """Build a metadata dict to embed in extraction output JSON.

    ``image_provenance`` (visual runs only) records the source-file hash,
    rendering library versions, and effective preprocessing parameters so
    the exact images sent to the model can be re-derived and verified
    against the per-record image hashes.

    ``chunking_text_version`` (text runs only) records which text-chunking
    behaviour produced the file; see :data:`CHUNKING_TEXT_VERSION`.
    """
    meta: dict[str, Any] = {
        "schema_name": schema_name,
        "model_name": model_name,
        "chunking_method": chunking_method,
        "total_chunks": total_chunks,
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
        "version": 1,
    }
    if chunking_text_version is not None:
        meta["chunking_text_version"] = chunking_text_version
    if chunk_slice_info:
        meta["chunk_slice"] = chunk_slice_info
    if partial:
        meta["partial"] = True
    if failed_chunks:
        meta["failed_chunks"] = list(failed_chunks)
    if image_provenance:
        meta["image_provenance"] = image_provenance
    return meta


def read_extraction_metadata(output_json: Path) -> dict[str, Any] | None:
    """Read embedded metadata from an extraction output JSON, if present."""
    try:
        with output_json.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    if isinstance(data, dict):
        return data.get(_METADATA_KEY)
    return None


def detect_extraction_status(
    output_json: Path,
    expected_chunks: int,
) -> tuple[FileStatus, set[int]]:
    """Determine if an extraction output is complete, partial, or missing.

    Args:
        output_json: Path to the ``_output.json`` file.
        expected_chunks: Number of chunks expected for this file.

    Returns:
        A tuple of ``(status, completed_chunk_indices)`` where indices are
        1-based chunk numbers that have already been processed.
    """
    if not output_json.exists():
        return FileStatus.NOT_STARTED, set()

    try:
        with output_json.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not parse %s; treating as NOT_STARTED", output_json)
        return FileStatus.NOT_STARTED, set()

    # The output is a JSON object with a "records" list and metadata,
    # or a bare list (legacy). Since v1.20.0 batch finalization writes the
    # same "records" shape, so batch outputs are covered here too.
    records: list[dict[str, Any]] = []
    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data

    completed: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        # custom_id format: "{stem}-chunk-{idx}" (text) or
        # "{stem}-page-{idx}" (visual).
        match = _CUSTOM_ID_INDEX_RE.search(str(record.get("custom_id", "")))
        if match:
            completed.add(int(match.group(1)))

    if not completed:
        # File exists but has no parseable chunk records
        return FileStatus.NOT_STARTED, set()

    if len(completed) >= expected_chunks:
        return FileStatus.COMPLETE, completed

    return FileStatus.PARTIAL, completed


def completed_indices_from_outputs(*output_paths: Path) -> set[int]:
    """Collect completed 1-based indices from existing output files.

    Reads the unified shape (a ``records`` list, written by both sync and
    batch since v1.20.0) and, as a legacy-on-disk-only fallback, the
    pre-v1.20.0 batch shape (a ``responses`` list). Extracts the numeric
    suffix of each ``custom_id`` (``...-chunk-N`` or ``...-page-N``). Used
    for batch resume parity: requests already present in a prior output are
    not re-submitted.
    """
    indices: set[int] = set()
    pattern = _CUSTOM_ID_INDEX_RE
    for path in output_paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict):
            items = data.get("records") or data.get("responses") or []
        elif isinstance(data, list):
            items = data
        else:
            items = []
        for rec in items:
            if not isinstance(rec, dict):
                continue
            match = pattern.search(str(rec.get("custom_id", "")))
            if match:
                indices.add(int(match.group(1)))
    return indices


def get_output_json_path(
    file_path: Path,
    paths_config: dict[str, Any],
    schema_paths: dict[str, Any],
) -> Path:
    """Derive the output JSON path for a given input text file."""
    if paths_config.get("general", {}).get("input_paths_is_output_path"):
        return ensure_path_safe(file_path.parent / f"{file_path.stem}_output.json")
    output_dir = schema_paths.get("output", "")
    return ensure_path_safe(Path(output_dir) / f"{file_path.stem}_output.json")
