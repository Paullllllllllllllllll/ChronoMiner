# modules/core/resume.py

"""
Resume and completeness detection utilities for ChronoMiner processing pipelines.

Provides file-level and chunk-level resume capabilities:
- Detect whether an extraction output is complete, partial, or not started.
- Detect whether a line-ranges file has already been adjusted.
- Read/write processing metadata for settings-aware resume.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from modules.core.path_utils import ensure_path_safe

logger = logging.getLogger(__name__)


class FileStatus(Enum):
    """Processing status for a file."""
    NOT_STARTED = "not_started"
    PARTIAL = "partial"
    COMPLETE = "complete"


# ---------------------------------------------------------------------------
# Extraction resume helpers
# ---------------------------------------------------------------------------

_METADATA_KEY = "_chronominer_metadata"


def build_extraction_metadata(
    *,
    schema_name: str,
    model_name: str,
    chunking_method: str,
    total_chunks: int,
    timestamp: Optional[str] = None,
    chunk_slice_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a metadata dict to embed in extraction output JSON."""
    meta: Dict[str, Any] = {
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


def read_extraction_metadata(output_json: Path) -> Optional[Dict[str, Any]]:
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
) -> Tuple[FileStatus, Set[int]]:
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
    records: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data

    completed: Set[int] = set()
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
    paths_config: Dict[str, Any],
    schema_paths: Dict[str, Any],
) -> Path:
    """Derive the output JSON path for a given input text file."""
    if paths_config.get("general", {}).get("input_paths_is_output_path"):
        return ensure_path_safe(file_path.parent / f"{file_path.stem}_output.json")
    output_dir = schema_paths.get("output", "")
    return ensure_path_safe(Path(output_dir) / f"{file_path.stem}_output.json")


# ---------------------------------------------------------------------------
# Line-range adjustment resume helpers
# ---------------------------------------------------------------------------

_ADJUSTED_SUFFIX = ".adjusted_meta"


def _adjusted_marker_path(line_ranges_file: Path) -> Path:
    """Return the sidecar marker path for an adjusted line-ranges file."""
    return line_ranges_file.with_suffix(line_ranges_file.suffix + _ADJUSTED_SUFFIX)


def write_adjustment_marker(
    line_ranges_file: Path,
    *,
    boundary_type: str,
    context_window: int,
    model_name: str,
) -> None:
    """Write a sidecar marker indicating that line ranges have been adjusted."""
    marker = _adjusted_marker_path(line_ranges_file)
    payload = {
        "adjusted_at": datetime.now(timezone.utc).isoformat(),
        "boundary_type": boundary_type,
        "context_window": context_window,
        "model_name": model_name,
        "source_file": line_ranges_file.name,
        "version": 1,
    }
    safe_marker = ensure_path_safe(marker)
    with safe_marker.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote adjustment marker: %s", marker)


def read_adjustment_marker(line_ranges_file: Path) -> Optional[Dict[str, Any]]:
    """Read the adjustment marker for a line-ranges file, if it exists."""
    marker = _adjusted_marker_path(line_ranges_file)
    if not marker.exists():
        return None
    try:
        with marker.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def is_adjustment_current(
    line_ranges_file: Path,
    *,
    boundary_type: str,
    context_window: int,
    model_name: str,
) -> bool:
    """Check whether the adjustment marker matches the current settings."""
    meta = read_adjustment_marker(line_ranges_file)
    if meta is None:
        return False
    return (
        meta.get("boundary_type") == boundary_type
        and meta.get("context_window") == context_window
        and meta.get("model_name") == model_name
    )
