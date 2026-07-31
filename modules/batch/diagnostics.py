from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_custom_id_mapping(
    temp_file: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """
    Build mapping of custom_id -> metadata and order index by scanning
    a JSONL temp file.
    Returns (custom_id_map, order_map).
    """
    custom_id_map: dict[str, dict[str, Any]] = {}
    order_map: dict[str, int] = {}

    try:
        with temp_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "batch_request" in record:
                    request = record.get("batch_request") or {}
                    cid = request.get("custom_id")
                    if cid:
                        info = (
                            request.get("image_info") or request.get("metadata") or {}
                        )
                        custom_id_map[cid] = info
                        if "order_index" in info:
                            order_map[cid] = info["order_index"]

                elif "image_metadata" in record:
                    meta = record.get("image_metadata") or {}
                    cid = meta.get("custom_id")
                    if cid:
                        custom_id_map[cid] = meta
                        if "order_index" in meta:
                            order_map[cid] = meta["order_index"]
    except Exception as exc:
        logger.error("Failed to extract custom_id mapping from %s: %s", temp_file, exc)

    return custom_id_map, order_map
