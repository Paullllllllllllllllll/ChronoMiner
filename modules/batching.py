# modules/batching.py

import logging
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

logger = logging.getLogger(__name__)


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
