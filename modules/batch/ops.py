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

from modules.batch.backends import (
    BatchHandle,
    get_batch_backend,
)
from modules.config.loader import get_config_loader
from modules.infra.logger import setup_logger

logger = setup_logger(__name__)


def derive_submission_output_dir(temp_file: Path) -> Path:
    """Submission-local output directory for a batch temp JSONL file.

    Batch submissions write their temp JSONL either into a ``temp_jsonl/``
    subfolder of the run's output directory (the default layout produced by
    ``FileProcessor._setup_output_paths``) or directly into the output
    directory (``input_paths_is_output_path: true``). The finalized output
    belongs with the submission, NOT in the schema's configured default
    output directory: it is the parent of ``temp_jsonl/`` when the temp file
    lives inside one, else the temp file's own directory.
    """
    parent = temp_file.parent
    if parent.name == "temp_jsonl":
        return parent.parent
    return parent


def is_batch_temp_file(path: Path) -> bool:
    """Whether a ``*_temp*.jsonl`` file is a BATCH temp file.

    Batch temp files carry ``batch_request``/``batch_tracking`` records;
    synchronous-extraction temp files (records with ``custom_id`` and a
    ``_chronominer_temp_version`` header) and readjuster ``_adjust_temp.jsonl``
    files do not. Identifying by content stops check_batches/repair from
    treating sync and readjuster temp files as batch candidates.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            for i, raw in enumerate(fh):
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    return False
                if isinstance(record, dict) and (
                    "batch_tracking" in record or "batch_request" in record
                ):
                    return True
                # A sync/readjuster record proves this is not a batch file.
                if isinstance(record, dict) and (
                    "custom_id" in record
                    or "chunk_index" in record
                    or "_chronominer_temp_version" in record
                ):
                    return False
                if i >= 50:
                    break
    except OSError:
        return False
    return False


def _group_temp_files_by_base(
    temp_files: list[Path],
) -> dict[tuple[Path, str], list[Path]]:
    """Group temp files by ``(parent directory, base stem)``.

    The base stem is the file stem with any ``_part{n}`` suffix removed, so
    ``file_temp_part1.jsonl`` and ``file_temp_part2.jsonl`` share one group.
    Keying on the parent directory as well as the base stem prevents
    identically named temp files living in different subdirectories (an rglob
    over the whole schema tree can surface several) from being merged into one
    group, which would collide their ``custom_id`` values and silently drop
    one file's chunks via dedup-last-wins.
    """
    groups: dict[tuple[Path, str], list[Path]] = {}
    for temp_file in temp_files:
        stem = temp_file.stem
        base_match = re.match(r"(.+?)(?:_part\d+)?$", stem)
        base_stem = base_match.group(1) if base_match else stem
        key = (temp_file.parent, base_stem)
        groups.setdefault(key, []).append(temp_file)

    # Sort files within each group by part number.
    for key in groups:
        groups[key].sort(
            key=lambda p: (
                int(re.search(r"_part(\d+)$", p.stem).group(1))  # type: ignore[union-attr]
                if re.search(r"_part(\d+)$", p.stem)
                else 0
            )
        )

    return groups


def _extract_chunk_index(custom_id: Any) -> int:
    """Extract numeric chunk index from a custom_id like '<stem>-chunk-<n>'
    or 'req-<n>'."""
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


def _recover_missing_batch_ids(
    temp_file: Path,
    identifier: str,
    persist: bool,
) -> tuple[set[str], str | None, dict[str, dict[str, Any]]]:
    """Recover batch ids, provider, and per-batch metadata from the debug artifact.

    Returns ``(batch_ids, provider, metadata_map)``; ``provider`` is ``None``
    when the artifact does not record one (dropping it would make callers
    default recovered Anthropic/Google batches to the OpenAI backend), and
    ``metadata_map`` maps each recovered batch id to its submitted handle
    metadata (empty when the artifact predates the ``batch_metadata`` schema).
    Restoring the metadata matters for Google inline submissions: without the
    ``custom_id_map``, ``_iter_results`` falls back to positional ``req-{i+1}``
    custom_ids and a resumed sliced submission is relabeled to the wrong chunks.
    """
    recovered: set[str] = set()
    provider: str | None = None
    metadata_map: dict[str, dict[str, Any]] = {}
    debug_artifact = temp_file.parent / f"{identifier}_batch_submission_debug.json"
    if not debug_artifact.exists():
        return recovered, provider, metadata_map

    try:
        artifact = json.loads(debug_artifact.read_text(encoding="utf-8"))
        for candidate in artifact.get("batch_ids", []) or []:
            if isinstance(candidate, str) and candidate:
                recovered.add(candidate)
        artifact_provider = artifact.get("provider")
        if isinstance(artifact_provider, str) and artifact_provider:
            provider = artifact_provider
        raw_metadata = artifact.get("batch_metadata")
        if isinstance(raw_metadata, dict):
            for bid, meta in raw_metadata.items():
                if isinstance(bid, str) and isinstance(meta, dict):
                    metadata_map[bid] = meta
    except Exception as exc:
        logger.warning(
            "Failed to read batch debug artifact %s: %s", debug_artifact, exc
        )
        return recovered, provider, metadata_map

    if recovered and persist:
        try:
            timestamp = datetime.datetime.now().isoformat()
            with temp_file.open("a", encoding="utf-8") as handle:
                for batch_id in recovered:
                    record: dict[str, Any] = {
                        "batch_tracking": {
                            "batch_id": batch_id,
                            "timestamp": timestamp,
                            "batch_file": str(temp_file),
                        }
                    }
                    if provider:
                        record["batch_tracking"]["provider"] = provider
                    # Persist the metadata inside the tracking record too so a
                    # second recovery round-trips it (needed for Google inline
                    # custom_id_map correlation).
                    if batch_id in metadata_map:
                        record["batch_tracking"]["metadata"] = metadata_map[batch_id]
                    handle.write(json.dumps(record) + "\n")
            logger.info(
                "Persisted %s recovered batch id(s) into %s",
                len(recovered),
                temp_file.name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to persist recovered batch ids for %s: %s",
                temp_file.name,
                exc,
            )

    return recovered, provider, metadata_map


def load_config() -> tuple[list[tuple[str, Path, dict[str, Any]]], dict[str, Any]]:
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
                response_entry: dict[str, Any] = {
                    "custom_id": result_item.custom_id,
                    "response": result_item.content,
                    "raw_response": result_item.raw_response,
                }
                if result_item.parsed_output:
                    response_entry["parsed_output"] = result_item.parsed_output
                # Backends populate token counts on the result item; carry them
                # through so _to_unified_record can stamp response_data.usage
                # (no provider nests usage under raw["usage"]).
                if result_item.input_tokens or result_item.output_tokens:
                    response_entry["usage"] = {
                        "input_tokens": result_item.input_tokens,
                        "output_tokens": result_item.output_tokens,
                    }
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
        # Do NOT return the partial list here: finalization would treat it as
        # the complete result set, write the output, and delete the temp files
        # and remote batch outputs, making the un-retrieved chunks
        # unrecoverable. Propagate so the caller's per-group error handling
        # keeps the artifacts for a later retry.
        logger.error(
            f"Error downloading batch results for {batch_id} ({provider}): {exc}"
        )
        raise

    return responses
