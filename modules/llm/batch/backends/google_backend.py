"""Google Gemini Batch API backend implementation.

Uses Google's Gemini Batch API for async batch text extraction.
See: https://ai.google.dev/gemini-api/docs/batch-api
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from modules.llm.batch.backends.base import (
    BatchBackend,
    BatchHandle,
    BatchRequest,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
)

logger = logging.getLogger(__name__)

# Limits for Google Batch API
MAX_BATCH_REQUESTS = 50000
MAX_BATCH_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB for file input
MAX_INLINE_BYTES = 20 * 1024 * 1024  # 20 MB for inline requests


class GoogleBatchBackend(BatchBackend):
    """Google Gemini Batch API backend."""

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy initialization of Google GenAI client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client()
        return self._client

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def max_batch_size(self) -> int:
        return MAX_BATCH_REQUESTS

    @property
    def max_batch_bytes(self) -> int:
        return MAX_BATCH_BYTES

    def submit_batch(
        self,
        requests: List[BatchRequest],
        model_config: Dict[str, Any],
        *,
        system_prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        schema_name: Optional[str] = None,
    ) -> BatchHandle:
        """Submit a batch to Google's Gemini Batch API."""
        client = self._get_client()

        # Model configuration
        tm = model_config.get("transcription_model", {}) or model_config
        model_name = tm.get("name", "gemini-2.5-flash")
        # Ensure model name has proper prefix for API
        if not model_name.startswith("models/"):
            api_model_name = f"models/{model_name}"
        else:
            api_model_name = model_name

        # Build generation config
        generation_config: Dict[str, Any] = {}
        max_tokens = tm.get("max_output_tokens") or tm.get("max_tokens")
        if max_tokens:
            generation_config["max_output_tokens"] = int(max_tokens)
        temperature = tm.get("temperature")
        if temperature is not None:
            generation_config["temperature"] = float(temperature)

        # Build inline requests
        # Each request is a GenerateContentRequest
        inline_requests = []
        for req in requests:
            # Build content parts for text extraction
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"Input text:\n{req.text}"},
                    ],
                }
            ]

            # Build the request with metadata key for correlation
            request_obj = {
                "contents": contents,
                "system_instruction": {"parts": [{"text": system_prompt}]},
            }
            if generation_config:
                request_obj["generation_config"] = generation_config

            inline_requests.append({
                "key": req.custom_id,
                "request": request_obj,
            })

        # Check if we should use file-based submission (larger batches)
        total_size = sum(len(json.dumps(r)) for r in inline_requests)

        if total_size < MAX_INLINE_BYTES:
            # Use inline requests
            logger.info("Submitting inline batch with %d requests to Google...", len(inline_requests))
            
            # Convert to the format expected by the SDK
            src_requests = []
            for item in inline_requests:
                src_requests.append({
                    "contents": item["request"]["contents"],
                    "system_instruction": item["request"].get("system_instruction"),
                    "generation_config": item["request"].get("generation_config"),
                })

            batch_job = client.batches.create(
                model=api_model_name,
                src=src_requests,
                config={
                    "display_name": f"chronominer-batch-{int(time.time())}",
                },
            )
        else:
            # Use file-based submission for larger batches
            logger.info("Submitting file-based batch with %d requests to Google...", len(inline_requests))
            
            # Create JSONL file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".jsonl",
                delete=False,
                encoding="utf-8"
            ) as f:
                for item in inline_requests:
                    f.write(json.dumps(item) + "\n")
                temp_path = Path(f.name)

            try:
                # Upload file
                from google.genai import types
                uploaded_file = client.files.upload(
                    file=str(temp_path),
                    config=types.UploadFileConfig(
                        display_name=f"batch-requests-{int(time.time())}",
                        mime_type="jsonl"
                    )
                )
                logger.info("Uploaded batch file: %s", uploaded_file.name)

                # Create batch job from file
                batch_job = client.batches.create(
                    model=api_model_name,
                    src=uploaded_file.name,
                    config={
                        "display_name": f"chronominer-batch-{int(time.time())}",
                    },
                )
            finally:
                try:
                    temp_path.unlink()
                except Exception:
                    pass

        batch_name = batch_job.name
        logger.info("Batch submitted; job name: %s", batch_name)

        return BatchHandle(
            provider="google",
            batch_id=batch_name,
            metadata={
                "request_count": len(requests),
                "custom_id_map": {req.custom_id: i for i, req in enumerate(requests)},
            },
        )

    def get_status(self, handle: BatchHandle) -> BatchStatusInfo:
        """Get status of a Google batch job."""
        client = self._get_client()

        try:
            batch_job = client.batches.get(name=handle.batch_id)
        except Exception as e:
            return BatchStatusInfo(
                status=BatchStatus.UNKNOWN,
                error_message=str(e),
            )

        # Map Google state to our enum
        state_name = batch_job.state.name if batch_job.state else ""
        status_map = {
            "JOB_STATE_PENDING": BatchStatus.PENDING,
            "JOB_STATE_RUNNING": BatchStatus.IN_PROGRESS,
            "JOB_STATE_SUCCEEDED": BatchStatus.COMPLETED,
            "JOB_STATE_FAILED": BatchStatus.FAILED,
            "JOB_STATE_CANCELLED": BatchStatus.CANCELLED,
            "JOB_STATE_EXPIRED": BatchStatus.EXPIRED,
        }
        status = status_map.get(state_name, BatchStatus.UNKNOWN)

        # Check for results
        dest = getattr(batch_job, "dest", None)
        results_available = False
        output_file_id = None

        if status == BatchStatus.COMPLETED and dest:
            if hasattr(dest, "file_name") and dest.file_name:
                results_available = True
                output_file_id = dest.file_name
            elif hasattr(dest, "inlined_responses") and dest.inlined_responses:
                results_available = True

        # Get error if failed
        error_message = None
        if status == BatchStatus.FAILED:
            error = getattr(batch_job, "error", None)
            if error:
                error_message = str(error)

        return BatchStatusInfo(
            status=status,
            total_requests=handle.metadata.get("request_count", 0),
            results_available=results_available,
            output_file_id=output_file_id,
            error_message=error_message,
        )

    def download_results(self, handle: BatchHandle) -> Iterator[BatchResultItem]:
        """Download and parse Google batch results."""
        client = self._get_client()

        batch_job = client.batches.get(name=handle.batch_id)
        dest = getattr(batch_job, "dest", None)

        if not dest:
            raise RuntimeError(f"Batch {handle.batch_id} has no results")

        # Check for file-based results
        if hasattr(dest, "file_name") and dest.file_name:
            # Download result file
            file_content = client.files.download(file=dest.file_name)
            text = file_content.decode("utf-8") if isinstance(file_content, bytes) else str(file_content)

            # Parse JSONL
            for line in text.strip().split("\n"):
                if not line.strip():
                    continue

                try:
                    result_obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                custom_id = result_obj.get("key", "")
                result_item = BatchResultItem(custom_id=custom_id)

                # Check for error
                if "error" in result_obj:
                    result_item.success = False
                    result_item.error = str(result_obj["error"])
                    yield result_item
                    continue

                # Extract response
                response = result_obj.get("response", {})
                result_item.success = True
                result_item.raw_response = response

                # Extract text content
                candidates = response.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    text_parts = []
                    for part in parts:
                        if "text" in part:
                            text_parts.append(part["text"])
                    result_item.content = "".join(text_parts)

                # Try to parse as JSON
                if result_item.content:
                    try:
                        parsed = json.loads(result_item.content)
                        if isinstance(parsed, dict):
                            result_item.parsed_output = parsed
                    except json.JSONDecodeError:
                        pass

                # Extract usage
                usage = response.get("usageMetadata", {})
                result_item.input_tokens = usage.get("promptTokenCount", 0)
                result_item.output_tokens = usage.get("candidatesTokenCount", 0)

                yield result_item

        # Check for inline results
        elif hasattr(dest, "inlined_responses") and dest.inlined_responses:
            for i, inline_response in enumerate(dest.inlined_responses):
                # For inline, we need to map index back to custom_id
                custom_id_map = handle.metadata.get("custom_id_map", {})
                # Reverse lookup
                custom_id = None
                for cid, idx in custom_id_map.items():
                    if idx == i:
                        custom_id = cid
                        break
                if custom_id is None:
                    custom_id = f"req-{i+1}"

                result_item = BatchResultItem(custom_id=custom_id)

                if hasattr(inline_response, "error") and inline_response.error:
                    result_item.success = False
                    result_item.error = str(inline_response.error)
                    yield result_item
                    continue

                if hasattr(inline_response, "response") and inline_response.response:
                    result_item.success = True
                    response = inline_response.response

                    # Extract text
                    try:
                        result_item.content = response.text
                    except AttributeError:
                        result_item.content = str(response)

                    # Try to parse as JSON
                    if result_item.content:
                        try:
                            parsed = json.loads(result_item.content)
                            if isinstance(parsed, dict):
                                result_item.parsed_output = parsed
                        except json.JSONDecodeError:
                            pass

                yield result_item

        else:
            raise RuntimeError(f"Batch {handle.batch_id} has no downloadable results")

    def cancel(self, handle: BatchHandle) -> bool:
        """Cancel a Google batch job."""
        client = self._get_client()
        try:
            client.batches.cancel(name=handle.batch_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel batch %s: %s", handle.batch_id, e)
            return False
