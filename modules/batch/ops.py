"""Shared batch helpers consumed by the check, cancel, and repair scripts.

Moved from ``main/check_batches.py`` to eliminate the cross-``main`` coupling
that ``main/repair_extractions.py`` previously relied on (it imported five
helpers directly from the check-batches script). Now check, cancel, and
repair all import from a single module under :mod:`modules.batch`.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from modules.batch.backends import (
    BatchHandle,
    BatchStatus,
    BatchStatusInfo,
    get_batch_backend,
)
from modules.config.loader import get_config_loader
from modules.infra.logger import setup_logger
from modules.llm.openai_sdk_utils import coerce_file_id

logger = setup_logger(__name__)


OUTPUT_FILE_KEYS = [
    "output_file_id",
    "output_file",
    "output_file_ids",
    "response_file_id",
    "response_file",
    "response_file_ids",
    "result_file_id",
    "result_file",
    "result_file_ids",
    "results_file_id",
    "results_file_ids",
]

ERROR_FILE_KEYS = [
    "error_file_id",
    "error_file",
    "error_file_ids",
    "errors_file_id",
    "errors_file_ids",
]


def _extract_chunk_index(custom_id: Any) -> int:
    """Extract numeric chunk index from a custom_id like '<stem>-chunk-<n>' or 'req-<n>'."""
    if not isinstance(custom_id, str):
        return 10**9  # push unknowns to end
    m = re.search(r"(?:-chunk-|-page-|req-)(\d+)$", custom_id)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 10**9
    return 10**9


def _order_responses(
    responses: list[Any], order_map: dict[str, int] | None = None
) -> list[Any]:
    """Return responses sorted using explicit order_map, then chunk index."""
    try:
        sortable: list[Any] = []
        nonsortable: list[Any] = []
        for item in responses:
            if isinstance(item, dict) and ("custom_id" in item):
                sortable.append(item)
            else:
                nonsortable.append(item)

        def _sort_key(entry: dict[str, Any]) -> tuple[int, int]:
            cid = entry.get("custom_id")
            order_val = 10**9
            if isinstance(cid, str) and order_map and cid in order_map:
                order_val = order_map[cid]
            chunk_val = _extract_chunk_index(cid)
            return order_val, chunk_val

        sortable.sort(key=_sort_key)
        return sortable + nonsortable
    except Exception:
        return responses


def _response_to_text(response_obj: Any) -> str:
    """Normalize a Responses API payload into a plain text string."""
    if isinstance(response_obj, str):
        return response_obj
    if not isinstance(response_obj, dict):
        return ""

    if isinstance(response_obj.get("output_text"), str):
        return response_obj["output_text"].strip()

    parts: list[str] = []
    output = response_obj.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for content_part in item.get("content", []):
                    text_val = (
                        content_part.get("text")
                        if isinstance(content_part, dict)
                        else None
                    )
                    if isinstance(text_val, str):
                        parts.append(text_val)
    return "".join(parts).strip()


def _normalize_response_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Ensure a response entry has consistent text/raw keys."""
    normalized = dict(entry)
    response_payload = normalized.get("response")
    normalized.setdefault("raw_response", response_payload)
    if isinstance(response_payload, dict):
        normalized["response"] = _response_to_text(response_payload)
        normalized.setdefault("raw_response", response_payload)
    return normalized


def _resolve_file_id_by_keys(
    batch: dict[str, Any], keys: list[str]
) -> str | None:
    for key in keys:
        if key in batch:
            file_id = coerce_file_id(batch.get(key))
            if file_id:
                return file_id
    return None


def _download_error_file(
    client: OpenAI, error_file_id: str, target_dir: Path, batch_id: str
) -> Path | None:
    """Download an error file for diagnostics if available."""
    try:
        response = client.files.content(error_file_id)
        blob = response.read()
        error_text = (
            blob.decode("utf-8")
            if isinstance(blob, (bytes, bytearray))
            else str(blob)
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        short_batch = batch_id.replace("batch_", "")[:16]
        error_path = target_dir / f"errors_{short_batch}.jsonl"
        with error_path.open("w", encoding="utf-8") as handle:
            handle.write(error_text)
        logger.info("Saved error details for batch %s to %s", batch_id, error_path)
        return error_path
    except Exception as exc:
        logger.warning(
            "Failed to download error file %s for batch %s: %s",
            error_file_id, batch_id, exc,
        )
    return None


def _recover_missing_batch_ids(
    temp_file: Path,
    identifier: str,
    persist: bool,
) -> set[str]:
    recovered: set[str] = set()
    debug_artifact = temp_file.parent / f"{identifier}_batch_submission_debug.json"
    if not debug_artifact.exists():
        return recovered

    try:
        artifact = json.loads(debug_artifact.read_text(encoding="utf-8"))
        for candidate in artifact.get("batch_ids", []) or []:
            if isinstance(candidate, str) and candidate:
                recovered.add(candidate)
    except Exception as exc:
        logger.warning(
            "Failed to read batch debug artifact %s: %s", debug_artifact, exc
        )
        return recovered

    if recovered and persist:
        try:
            timestamp = datetime.datetime.now().isoformat()
            with temp_file.open("a", encoding="utf-8") as handle:
                for batch_id in recovered:
                    record = {
                        "batch_tracking": {
                            "batch_id": batch_id,
                            "timestamp": timestamp,
                            "batch_file": str(temp_file),
                        }
                    }
                    handle.write(json.dumps(record) + "\n")
            logger.info(
                "Persisted %s recovered batch id(s) into %s",
                len(recovered), temp_file.name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to persist recovered batch ids for %s: %s",
                temp_file.name, exc,
            )

    return recovered


def is_batch_finished(batch_id: str, provider: str = "openai") -> bool:
    """Check if a batch is finished using the appropriate provider backend."""
    try:
        backend = get_batch_backend(provider)
        handle = BatchHandle(provider=provider, batch_id=batch_id)
        status_info = backend.get_status(handle)
        if status_info.status in {
            BatchStatus.COMPLETED,
            BatchStatus.EXPIRED,
            BatchStatus.CANCELLED,
            BatchStatus.FAILED,
        }:
            return True
        else:
            logger.info(
                f"Batch {batch_id} status is '{status_info.status.value}', "
                "not finished yet."
            )
            return False
    except Exception as e:
        logger.error(f"Error retrieving batch {batch_id}: {e}")
        return False


def load_config() -> tuple[
    list[tuple[str, Path, dict[str, Any]]], dict[str, Any]
]:
    config_loader = get_config_loader()
    paths_config: dict[str, Any] = config_loader.get_paths_config()
    general: dict[str, Any] = paths_config["general"]
    input_paths_is_output_path: bool = general["input_paths_is_output_path"]
    schemas_paths: dict[str, Any] = paths_config["schemas_paths"]
    repo_info_list: list[tuple[str, Path, dict[str, Any]]] = []
    for schema, schema_config in schemas_paths.items():
        folder: Path = (
            Path(schema_config["input"])
            if input_paths_is_output_path
            else Path(schema_config["output"])
        )
        repo_info_list.append((schema, folder, schema_config))
    processing_settings: dict[str, Any] = {
        "retain_temporary_jsonl": general["retain_temporary_jsonl"]
    }
    return repo_info_list, processing_settings


def process_batch_output_file(file_path: Path) -> dict[str, list[Any]]:
    responses: list[Any] = []
    tracking: list[Any] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record: dict[str, Any] = json.loads(line)
                if "response" in record:
                    responses.append(
                        _normalize_response_entry(
                            {
                                "response": record.get("response"),
                                "custom_id": record.get("custom_id"),
                                "chunk_range": record.get("chunk_range"),
                            }
                        )
                    )
                elif "batch_tracking" in record:
                    tracking.append(record["batch_tracking"])
            except Exception as e:
                logger.error(f"Error processing line in {file_path}: {e}")
    return {"responses": responses, "tracking": tracking}


def retrieve_responses_from_batch(
    tracking_record: dict[str, Any],
    temp_dir: Path,
    status_cache: dict[str, BatchStatusInfo],
) -> list[dict[str, Any]]:
    """Retrieve responses from a batch using the appropriate provider backend."""
    responses: list[dict[str, Any]] = []
    batch_id: Any = tracking_record.get("batch_id")
    # Default to openai for backward compatibility
    provider: str = tracking_record.get("provider", "openai")

    if not batch_id:
        logger.error("No batch_id found in tracking record.")
        return responses

    batch_id = str(batch_id)

    try:
        backend = get_batch_backend(provider)
        handle = BatchHandle(
            provider=provider,
            batch_id=batch_id,
            metadata=tracking_record.get("metadata", {}),
        )

        # Download results using the provider-agnostic backend
        for result_item in backend.download_results(handle):
            if result_item.success:
                response_entry = {
                    "custom_id": result_item.custom_id,
                    "response": result_item.content,
                    "raw_response": result_item.raw_response,
                }
                if result_item.parsed_output:
                    response_entry["parsed_output"] = result_item.parsed_output
                responses.append(_normalize_response_entry(response_entry))
            else:
                logger.warning(
                    f"Request {result_item.custom_id} in batch {batch_id} "
                    f"failed: {result_item.error}"
                )
                responses.append(
                    {
                        "custom_id": result_item.custom_id,
                        "response": None,
                        "error": result_item.error,
                        "error_code": result_item.error_code,
                    }
                )

    except Exception as exc:
        logger.error(
            f"Error downloading batch results for {batch_id} ({provider}): {exc}"
        )
        return responses

    return responses
