"""Anthropic Message Batches API backend implementation.

Uses Anthropic's Message Batches API for async batch text extraction.
See: https://docs.anthropic.com/en/docs/build-with-claude/message-batches
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from modules.batch.backends.base import (
    BatchBackend,
    BatchHandle,
    BatchRequest,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
)
from modules.config.capabilities import detect_capabilities
from modules.config.loader import resolve_api_key

logger = logging.getLogger(__name__)

# Limits for Anthropic Message Batches API
MAX_BATCH_REQUESTS = 100000
MAX_BATCH_BYTES = 256 * 1024 * 1024  # 256 MB


def _message_to_dict(message: Any) -> Any:
    """Best-effort conversion of an Anthropic SDK Message to a plain dict.

    Prefers pydantic's ``model_dump``; falls back to ``sdk_to_dict`` for any
    object shape, and finally to ``str`` so serialization never raises.
    """
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            try:
                return model_dump()
            except Exception:
                pass
        except Exception:
            pass
    from modules.llm.openai_sdk_utils import sdk_to_dict

    try:
        return sdk_to_dict(message)
    except Exception:
        return str(message)


class AnthropicBatchBackend(BatchBackend):
    """Anthropic Message Batches API backend."""

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            import anthropic

            # api_key resolves via the api_keys_config.yaml mapping (override or
            # default); None falls back to the SDK's own env lookup, so behavior
            # is identical when no mapping is configured.
            self._client = anthropic.Anthropic(api_key=resolve_api_key("anthropic"))
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
        requests: list[BatchRequest],
        model_config: dict[str, Any],
        *,
        system_prompt: str,
        schema: dict[str, Any] | None = None,
        schema_name: str | None = None,
    ) -> BatchHandle:
        """Submit a batch to Anthropic's Message Batches API."""
        client = self._get_client()

        # Model configuration
        tm = model_config.get("extraction_model", {}) or model_config
        model_name = tm.get("name", "claude-sonnet-4-20250514")
        max_tokens = int(tm.get("max_output_tokens") or tm.get("max_tokens", 4096))

        # Clamp to the model's registry cap. The sync path clamps via
        # LangChainLLM._effective_max_tokens; without the same guard here the
        # configured default (e.g. 128000) exceeds Claude caps (8000-64000)
        # and every batch request 400s.
        caps = detect_capabilities(model_name, provider="anthropic")
        cap = getattr(caps, "max_output_tokens", None)
        if cap is not None and max_tokens > int(cap):
            logger.warning(
                "max_output_tokens %s exceeds the %s cap of %s; clamping to %s",
                f"{max_tokens:,}",
                model_name,
                f"{int(cap):,}",
                f"{int(cap):,}",
            )
            max_tokens = int(cap)

        # Build batch requests
        batch_requests = []
        for req in requests:
            # Route by input type: visual or text
            if req.is_visual:
                user_content = [
                    {"type": "text", "text": "Process this image:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": req.mime_type,
                            "data": req.image_base64,
                        },
                    },
                ]
            else:
                user_content = [
                    {"type": "text", "text": f"Input text:\n{req.text}"},
                ]

            # Build params for Messages API
            # Use structured system message with cache_control for prompt caching
            params: dict[str, Any] = {
                "model": model_name,
                "max_tokens": max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [
                    {"role": "user", "content": user_content},
                ],
            }

            temperature = tm.get("temperature")
            if temperature is not None and caps.supports_sampler_controls:
                params["temperature"] = float(temperature)

            batch_requests.append(
                {
                    "custom_id": req.custom_id,
                    "params": params,
                }
            )

        # Submit batch
        logger.info(
            "Submitting batch with %d requests to Anthropic...", len(batch_requests)
        )
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
            if total == 0:
                # request_counts absent/all-zero: outcome is indeterminate.
                # Without this guard `errored == total` (0 == 0) mislabels the
                # batch FAILED.
                status = BatchStatus.UNKNOWN
            elif errored == total:
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
                        # Serialize the pydantic SDK Message to a plain dict so
                        # the raw response survives json.dumps at finalization
                        # (check_batches); storing the SDK object raised
                        # TypeError and aborted the whole file group.
                        result_item.raw_response = {
                            "message": _message_to_dict(message)
                        }

                        # Extract content
                        content_list = getattr(message, "content", [])
                        text_parts = []
                        for block in content_list:
                            if getattr(block, "type", "") == "text":
                                text_parts.append(getattr(block, "text", ""))
                        result_item.content = "".join(text_parts)

                        # Try to parse as JSON (with balanced-brace recovery)
                        if result_item.content:
                            from modules.conversion.json_utils import (
                                parse_json_from_text,
                            )

                            json_str = parse_json_from_text(result_item.content)
                            if json_str is not None:
                                try:
                                    parsed = json.loads(json_str)
                                    if isinstance(parsed, dict):
                                        result_item.parsed_output = parsed
                                except json.JSONDecodeError:
                                    pass

                        # Extract usage
                        usage = getattr(message, "usage", None)
                        if usage:
                            result_item.input_tokens = getattr(usage, "input_tokens", 0)
                            result_item.output_tokens = getattr(
                                usage, "output_tokens", 0
                            )

                        # A message with no text blocks (e.g. stop_reason
                        # max_tokens after thinking-only output) must not be
                        # reported as a successful empty extraction; the
                        # chunk would be marked complete and never retried.
                        if not result_item.content:
                            stop_reason = getattr(message, "stop_reason", None)
                            result_item.success = False
                            result_item.error = (
                                "No text content in Anthropic batch response "
                                f"(stop_reason={stop_reason or 'unknown'})"
                            )
                    else:
                        result_item.success = False
                        result_item.error = (
                            "Anthropic batch result succeeded but carried "
                            "no message"
                        )

                elif result_type == "errored":
                    # result_data.error is an ErrorResponse wrapping the actual
                    # ErrorObject (message/type) one level down; unwrap it so the
                    # real message and code surface instead of "Unknown error".
                    error = getattr(result_data, "error", None)
                    err_obj = getattr(error, "error", None) if error else None
                    result_item.success = False
                    if err_obj is not None:
                        result_item.error = getattr(err_obj, "message", "Unknown error")
                        result_item.error_code = getattr(err_obj, "type", None)
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
