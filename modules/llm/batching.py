# modules/batching.py

import logging
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from modules.config.loader import ConfigLoader

logger = logging.getLogger(__name__)


DEFAULT_BATCH_CHUNK_SIZE: int = 50
DEFAULT_MAX_BATCH_BYTES: int = 150 * 1024 * 1024  # 150 MB safety margin


def write_batch_file(request_lines: List[str], output_path: Path) -> Path:
    """
    Write a list of batch request lines to a file.

    :param request_lines: List of JSON strings representing batch requests.
    :param output_path: Path to the output file.
    :return: The output file path.
    """
    with output_path.open("w", encoding="utf-8") as f:
        for line in request_lines:
            f.write(line + "\n")
    logger.info(f"Batch file written to {output_path}")
    return output_path


def _get_transcription_config() -> Dict[str, Any]:
    try:
        loader = ConfigLoader()
        loader.load_configs()
        concurrency_cfg = loader.get_concurrency_config() or {}
        return (concurrency_cfg.get("concurrency", {}) or {}).get("transcription", {}) or {}
    except Exception:
        return {}


def get_batch_chunk_size() -> int:
    cfg = _get_transcription_config()
    try:
        value = int(cfg.get("batch_chunk_size", DEFAULT_BATCH_CHUNK_SIZE))
        return value if value > 0 else DEFAULT_BATCH_CHUNK_SIZE
    except Exception:
        return DEFAULT_BATCH_CHUNK_SIZE


def get_max_batch_bytes() -> int:
    cfg = _get_transcription_config()
    try:
        value = int(cfg.get("max_batch_bytes", DEFAULT_MAX_BATCH_BYTES))
        return value if value > 0 else DEFAULT_MAX_BATCH_BYTES
    except Exception:
        return DEFAULT_MAX_BATCH_BYTES


def build_batch_files(request_lines: List[str], base_path: Path) -> List[Path]:
    """Split request lines into size-aware batch files and return their paths."""
    if not request_lines:
        return []

    chunk_size = get_batch_chunk_size()
    max_bytes = get_max_batch_bytes()

    batches: List[List[str]] = []
    current_lines: List[str] = []
    current_bytes = 0

    for line in request_lines:
        encoded_len = len(line.encode("utf-8")) + 1  # newline
        if current_lines and (
            len(current_lines) >= chunk_size or current_bytes + encoded_len > max_bytes
        ):
            batches.append(current_lines)
            current_lines = []
            current_bytes = 0
        current_lines.append(line)
        current_bytes += encoded_len

    if current_lines:
        batches.append(current_lines)

    batch_paths: List[Path] = []
    for index, lines in enumerate(batches, start=1):
        if len(batches) == 1 and index == 1:
            target_path = base_path
        else:
            target_path = base_path.with_name(f"{base_path.stem}_part{index}{base_path.suffix}")
        write_batch_file(lines, target_path)
        batch_paths.append(target_path)

    return batch_paths


def submit_batch(batch_file_path: Path) -> Dict[str, Any]:
    """
    Submit a batch job to the OpenAI API using the given batch file.

    :param batch_file_path: Path to the batch file (in binary mode).
    :return: The response dictionary from the batch submission.
    """
    client: OpenAI = OpenAI()  # Instantiate a new client instance
    with batch_file_path.open("rb") as f:
        file_response = client.files.create(
            file=f,
            purpose="batch"
        )
    file_id: str = file_response.id
    logger.info(f"Uploaded batch file, file id: {file_id}")
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"description": "Batch text processing job (Responses API)"}
    )
    logger.info(f"Batch submitted, batch id: {batch_response.id}")
    return batch_response
