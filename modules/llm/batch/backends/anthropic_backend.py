"""Anthropic Message Batches API backend implementation.

Uses Anthropic's Message Batches API for async batch text extraction.
See: https://docs.anthropic.com/en/docs/build-with-claude/message-batches
"""

from __future__ import annotations

import json
import logging
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

# Limits for Anthropic Message Batches API
MAX_BATCH_REQUESTS = 100000
MAX_BATCH_BYTES = 256 * 1024 * 1024  # 256 MB


class AnthropicBatchBackend(BatchBackend):
    """Anthropic Message Batches API backend."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    @property
    def provider_name(self) -> str:
        return "anthropic"

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
        """Submit a batch to Anthropic's Message Batches API."""
        client = self._get_client()

        # Model configuration
        tm = model_config.get("transcription_model", {}) or model_config
        model_name = tm.get("name", "claude-sonnet-4-20250514")
        max_tokens = int(
            tm.get("max_output_tokens")
            or tm.get("max_tokens", 4096)
        )

        # Build batch requests
        batch_requests = []
        for req in requests:
            # Build message content for text extraction
            user_content = [
                {"type": "text", "text": f"Input text:\n{req.text}"},
            ]

            # Build params for Messages API
            params: Dict[str, Any] = {
                "model": model_name,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_content},
                ],
            }

            # Add temperature if supported and not a reasoning model
            temperature = tm.get("temperature")
            if temperature is not None:
                # Check if model supports temperature (non-reasoning models)
                model_lower = model_name.lower()
                is_reasoning = any(x in model_lower for x in ["opus-4", "sonnet-4-5", "haiku-4-5"])
                if not is_reasoning:
                    params["temperature"] = float(temperature)

            batch_requests.append({
                "custom_id": req.custom_id,
                "params": params,
            })

        # Submit batch
        logger.info("Submitting batch with %d requests to Anthropic...", len(batch_requests))
        batch_response = client.messages.batches.create(requests=batch_requests)
        batch_id = batch_response.id
        logger.info("Batch submitted; batch id: %s", batch_id)

        return BatchHandle(
            provider="anthropic",
            batch_id=batch_id,
            metadata={
                "request_count": len(requests),
            },
        )

    def get_status(self, handle: BatchHandle) -> BatchStatusInfo:
        """Get status of an Anthropic batch job."""
        client = self._get_client()

        try:
            batch = client.messages.batches.retrieve(handle.batch_id)
        except Exception as e:
            return BatchStatusInfo(
                status=BatchStatus.UNKNOWN,
                error_message=str(e),
            )

        # Map Anthropic status to our enum
        # Anthropic uses: in_progress, ended (then check request_counts)
        processing_status = getattr(batch, "processing_status", "")

        # Get request counts
        request_counts = getattr(batch, "request_counts", None)
        processing = succeeded = errored = canceled = expired = 0
        if request_counts:
            processing = getattr(request_counts, "processing", 0)
            succeeded = getattr(request_counts, "succeeded", 0)
            errored = getattr(request_counts, "errored", 0)
            canceled = getattr(request_counts, "canceled", 0)
            expired = getattr(request_counts, "expired", 0)

        total = processing + succeeded + errored + canceled + expired

        # Determine status
        if processing_status == "in_progress":
            status = BatchStatus.IN_PROGRESS
        elif processing_status == "ended":
            if errored == total:
                status = BatchStatus.FAILED
            elif canceled == total:
                status = BatchStatus.CANCELLED
            elif expired == total:
                status = BatchStatus.EXPIRED
            else:
                status = BatchStatus.COMPLETED
        else:
            status = BatchStatus.PENDING

        # Check if results are available
        results_url = getattr(batch, "results_url", None)
        results_available = processing_status == "ended" and results_url is not None

        return BatchStatusInfo(
            status=status,
            total_requests=total,
            completed_requests=succeeded,
            failed_requests=errored,
            pending_requests=processing,
            results_available=results_available,
            output_file_id=results_url,  # Use results_url as output reference
        )

    def download_results(self, handle: BatchHandle) -> Iterator[BatchResultItem]:
        """Download and parse Anthropic batch results."""
        client = self._get_client()

        # Stream results as JSONL
        # The SDK provides a helper for this
        try:
            for result in client.messages.batches.results(handle.batch_id):
                custom_id = result.custom_id
                result_item = BatchResultItem(custom_id=custom_id)

                result_data = result.result
                result_type = getattr(result_data, "type", "")

                if result_type == "succeeded":
                    message = getattr(result_data, "message", None)
                    if message:
                        result_item.success = True
                        result_item.raw_response = {"message": message}

                        # Extract content
                        content_list = getattr(message, "content", [])
                        text_parts = []
                        for block in content_list:
                            if getattr(block, "type", "") == "text":
                                text_parts.append(getattr(block, "text", ""))
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
                        usage = getattr(message, "usage", None)
                        if usage:
                            result_item.input_tokens = getattr(usage, "input_tokens", 0)
                            result_item.output_tokens = getattr(usage, "output_tokens", 0)
                    else:
                        result_item.success = True
                        result_item.content = ""

                elif result_type == "errored":
                    error = getattr(result_data, "error", None)
                    result_item.success = False
                    if error:
                        result_item.error = getattr(error, "message", "Unknown error")
                        result_item.error_code = getattr(error, "type", None)
                    else:
                        result_item.error = "Unknown error"

                elif result_type == "canceled":
                    result_item.success = False
                    result_item.error = "Request was canceled"
                    result_item.error_code = "canceled"

                elif result_type == "expired":
                    result_item.success = False
                    result_item.error = "Request expired"
                    result_item.error_code = "expired"

                else:
                    result_item.success = False
                    result_item.error = f"Unknown result type: {result_type}"

                yield result_item

        except Exception as e:
            logger.error("Error downloading Anthropic batch results: %s", e)
            raise

    def cancel(self, handle: BatchHandle) -> bool:
        """Cancel an Anthropic batch job."""
        client = self._get_client()
        try:
            client.messages.batches.cancel(handle.batch_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel batch %s: %s", handle.batch_id, e)
            return False
