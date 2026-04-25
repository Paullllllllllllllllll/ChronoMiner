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
from datetime import datetime, timezone
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


def build_extraction_metadata(
    *,
    schema_name: str,
    model_name: str,
    chunking_method: str,
    total_chunks: int,
    timestamp: str | None = None,
    chunk_slice_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a metadata dict to embed in extraction output JSON."""
    meta: dict[str, Any] = {
        "schema_name": schema_name,
        "model_name": model_name,
        "chunking_method": chunking_method,
        "total_chunks": total_chunks,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "version": 1,
    }
    if chunk_slice_info:
        meta["chunk_slice"] = chunk_slice_info
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

    # The output is a JSON object with a "records" list and metadata, or a bare list (legacy).
    records: list[dict[str, Any]] = []
    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data

    completed: set[int] = set()
    for record in records:
        custom_id = record.get("custom_id", "")
        # custom_id format: "{stem}-chunk-{idx}"
        if "-chunk-" in str(custom_id):
            try:
                idx = int(str(custom_id).rsplit("-chunk-", 1)[1])
                completed.add(idx)
            except (ValueError, IndexError):
                pass

    if not completed:
        # File exists but has no parseable chunk records
        return FileStatus.NOT_STARTED, set()

    if len(completed) >= expected_chunks:
        return FileStatus.COMPLETE, completed

    return FileStatus.PARTIAL, completed


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


