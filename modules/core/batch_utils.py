from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from modules.llm.openai_sdk_utils import sdk_to_dict

logger = logging.getLogger(__name__)


def diagnose_batch_failure(batch_id: str, client: Any) -> str:
    """Return a human-readable explanation for a batch failure or status."""
    try:
        batch_obj = client.batches.retrieve(batch_id)
        batch = sdk_to_dict(batch_obj)
        status = str(batch.get("status", "")).lower()
        if status == "failed":
            return (
                f"Batch {batch_id} failed. Review the OpenAI dashboard for detailed errors."
            )
        if status == "cancelled":
            return f"Batch {batch_id} was cancelled."
        if status == "expired":
            return f"Batch {batch_id} expired before completion."
        return f"Batch {batch_id} currently has status '{status}'."
    except Exception as exc:
        message = str(exc).lower()
        if "not found" in message:
            return (
                f"Batch {batch_id} not found. It may have been deleted or was submitted with another API key."
            )
        if "unauthorized" in message:
            return "API key unauthorized. Verify OpenAI credentials."
        if "quota" in message:
            return "API quota exceeded. Check usage limits."
        return f"Error retrieving batch {batch_id}: {exc}"


def extract_custom_id_mapping(temp_file: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """
    Build mapping of custom_id -> metadata and order index by scanning a JSONL temp file.
    Returns (custom_id_map, order_map).
    """
    custom_id_map: Dict[str, Dict[str, Any]] = {}
    order_map: Dict[str, int] = {}

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
                        info = request.get("image_info") or {}
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
