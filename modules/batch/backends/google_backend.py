"""Google Gemini Batch API backend implementation.

Uses Google's Gemini Batch API for async batch text extraction.
See: https://ai.google.dev/gemini-api/docs/batch-api
"""

from __future__ import annotations

import contextlib
import json
import logging
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
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

# Limits for Google Batch API
MAX_BATCH_REQUESTS = 50000
MAX_BATCH_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB for file input
MAX_INLINE_BYTES = 20 * 1024 * 1024  # 20 MB for inline requests

# JSON-Schema keywords Gemini's ``response_schema`` rejects. Stripped before
# the schema is forwarded so a standard ChronoMiner schema (which carries
# ``additionalProperties: false``) does not 400 at submit time. Schemas that
# rely on ``$ref``/``$defs`` cannot be expressed this way at all: stripping
# the reference would leave empty ``{}`` schema nodes (invalid for required
# properties), so ``response_schema`` is skipped entirely for such schemas
# and Gemini falls back to prompt-guided JSON (see ``_contains_schema_refs``).
_GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "definitions",
        "additionalProperties",
        "title",
        "examples",
        "default",
        "const",
    }
)


def _sanitize_gemini_schema(node: Any) -> Any:
    """Recursively drop schema keywords Gemini's ``response_schema`` rejects."""
    if isinstance(node, dict):
        return {
            key: _sanitize_gemini_schema(value)
            for key, value in node.items()
            if key not in _GEMINI_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(node, list):
        return [_sanitize_gemini_schema(item) for item in node]
    return node


def _contains_schema_refs(node: Any) -> bool:
    """Whether a JSON schema uses ``$ref``/``$defs``/``definitions`` anywhere."""
    if isinstance(node, dict):
        if any(k in node for k in ("$ref", "$defs", "definitions")):
            return True
        return any(_contains_schema_refs(v) for v in node.values())
    if isinstance(node, list):
        return any(_contains_schema_refs(item) for item in node)
    return False


class GoogleBatchBackend(BatchBackend):
    """Google Gemini Batch API backend."""

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy initialization of Google GenAI client."""
        if self._client is None:
            from google import genai

            # api_key resolves via the api_keys_config.yaml mapping (override or
            # default); None falls back to the SDK's own env lookup, so behavior
            # is identical when no mapping is configured.
            self._client = genai.Client(api_key=resolve_api_key("google"))
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
        requests: list[BatchRequest],
        model_config: dict[str, Any],
        *,
        system_prompt: str,
        schema: dict[str, Any] | None = None,
        schema_name: str | None = None,
    ) -> BatchHandle:
        """Submit a batch to Google's Gemini Batch API."""
        client = self._get_client()

        # Model configuration
        tm = model_config.get("extraction_model", {}) or model_config
        model_name = tm.get("name", "gemini-2.5-flash")
        # Ensure model name has proper prefix for API
        if not model_name.startswith("models/"):
            api_model_name = f"models/{model_name}"
        else:
            api_model_name = model_name

        # Build generation config
        caps = detect_capabilities(model_name, provider="google")
        generation_config: dict[str, Any] = {}
        max_tokens = tm.get("max_output_tokens") or tm.get("max_tokens")
        if max_tokens:
            generation_config["max_output_tokens"] = self._clamp_max_output_tokens(
                int(max_tokens), caps, model_name
            )
        temperature = tm.get("temperature")
        # Gate on the capability registry, matching the OpenAI and Anthropic
        # backends: reasoning models reject sampler controls.
        if temperature is not None and caps.supports_sampler_controls:
            generation_config["temperature"] = float(temperature)

        # Wire structured-output schema when the model supports it. Without
        # this, Gemini batch jobs run unconstrained and only emit JSON because
        # the schema is embedded in the system prompt. Gate on the capability
        # registry so reasoning/unsupported models are left unconstrained, and
        # sanitize the schema for Gemini's response_schema restrictions.
        # Schemas using $ref/$defs cannot be sanitized into a valid
        # response_schema (stripping references leaves empty nodes), so they
        # fall back to prompt-guided JSON instead of sending a broken schema.
        if schema and caps.supports_structured_outputs:
            if "schema" in schema and isinstance(schema["schema"], dict):
                response_schema = schema["schema"]
            else:
                response_schema = schema
            if response_schema and _contains_schema_refs(response_schema):
                logger.warning(
                    "Schema %s uses $ref/$defs, which Gemini's response_schema "
                    "cannot express; submitting without schema enforcement "
                    "(prompt-guided JSON).",
                    schema_name or "<unnamed>",
                )
            elif response_schema:
                generation_config["response_mime_type"] = "application/json"
                generation_config["response_schema"] = _sanitize_gemini_schema(
                    response_schema
                )

        # Build inline requests
        # Each request is a GenerateContentRequest
        inline_requests = []
        for req in requests:
            # Route by input type: visual or text
            if req.is_visual:
                parts = [
                    {"text": "Process this image:"},
                    {
                        "inline_data": {
                            "mime_type": req.mime_type,
                            "data": req.image_base64,
                        }
                    },
                ]
            else:
                parts = [{"text": f"Input text:\n{req.text}"}]

            contents = [
                {
                    "role": "user",
                    "parts": parts,
                }
            ]

            # Build the request with metadata key for correlation
            request_obj = {
                "contents": contents,
                "system_instruction": {"parts": [{"text": system_prompt}]},
            }
            if generation_config:
                request_obj["generation_config"] = generation_config

            inline_requests.append(
                {
                    "key": req.custom_id,
                    "request": request_obj,
                }
            )

        # Check if we should use file-based submission (larger batches)
        total_size = sum(len(json.dumps(r)) for r in inline_requests)

        # Remote input file uploaded in file-mode (None for inline). Stored on
        # the handle so download_results can delete it and avoid leaking files.
        uploaded_file_name: str | None = None

        if total_size < MAX_INLINE_BYTES:
            # Use inline requests
            logger.info(
                "Submitting inline batch with %d requests to Google...",
                len(inline_requests),
            )

            # Convert to the SDK's InlinedRequest shape. That model forbids
            # extra keys and only accepts model/contents/metadata/config, so
            # system_instruction and generation params must live inside config
            # (GenerateContentConfig owns system_instruction, max_output_tokens,
            # temperature, response_mime_type, response_schema).
            src_requests = []
            for item in inline_requests:
                req_config: dict[str, Any] = dict(generation_config or {})
                req_config["system_instruction"] = system_prompt
                src_requests.append(
                    {
                        "contents": item["request"]["contents"],
                        "config": req_config,
                        # Explicit correlation: InlinedResponse echoes this
                        # metadata back, so results are matched by key rather
                        # than purely by position. Positional matching alone
                        # mis-attributes every response after a gap when the
                        # API returns fewer or reordered responses (e.g. under
                        # JOB_STATE_PARTIALLY_SUCCEEDED).
                        "metadata": {"key": str(item["key"])},
                    }
                )

            batch_job = client.batches.create(
                model=api_model_name,
                src=src_requests,
                config={
                    "display_name": f"chronominer-batch-{int(time.time())}",
                },
            )
        else:
            # Use file-based submission for larger batches
            logger.info(
                "Submitting file-based batch with %d requests to Google...",
                len(inline_requests),
            )

            # Create JSONL file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
            ) as f:
                for item in inline_requests:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                temp_path = Path(f.name)

            try:
                # Upload file
                from google.genai import types

                uploaded_file = client.files.upload(
                    file=str(temp_path),
                    config=types.UploadFileConfig(
                        display_name=f"batch-requests-{int(time.time())}",
                        mime_type="jsonl",
                    ),
                )
                uploaded_file_name = uploaded_file.name
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
                with contextlib.suppress(Exception):
                    temp_path.unlink()

        batch_name = batch_job.name
        logger.info("Batch submitted; job name: %s", batch_name)

        return BatchHandle(
            provider="google",
            batch_id=batch_name,
            metadata={
                "request_count": len(requests),
                "custom_id_map": {req.custom_id: i for i, req in enumerate(requests)},
                "input_file_name": uploaded_file_name,
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
            "JOB_STATE_QUEUED": BatchStatus.PENDING,
            "JOB_STATE_PENDING": BatchStatus.PENDING,
            "JOB_STATE_RUNNING": BatchStatus.IN_PROGRESS,
            "JOB_STATE_CANCELLING": BatchStatus.IN_PROGRESS,
            "JOB_STATE_PAUSED": BatchStatus.IN_PROGRESS,
            "JOB_STATE_UPDATING": BatchStatus.IN_PROGRESS,
            "JOB_STATE_SUCCEEDED": BatchStatus.COMPLETED,
            # Terminal with downloadable results; per-request failures are
            # yielded as failed items by _iter_results, mirroring OpenAI's
            # completed-with-failed-requests semantics. Without this mapping
            # a partially-succeeded job read as UNKNOWN-without-error and
            # check_batches reported "still processing" forever.
            "JOB_STATE_PARTIALLY_SUCCEEDED": BatchStatus.COMPLETED,
            "JOB_STATE_FAILED": BatchStatus.FAILED,
            "JOB_STATE_CANCELLED": BatchStatus.CANCELLED,
            "JOB_STATE_EXPIRED": BatchStatus.EXPIRED,
        }
        status = status_map.get(state_name, BatchStatus.UNKNOWN)
        if status is BatchStatus.UNKNOWN and state_name:
            # An unmapped state must never wedge the batch silently: attach
            # the state so check_batches treats it as actionable.
            return BatchStatusInfo(
                status=BatchStatus.UNKNOWN,
                total_requests=handle.metadata.get("request_count", 0),
                error_message=f"Unmapped Google batch state: {state_name}",
            )

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
        """Download and parse Google batch results.

        Remote files are NOT deleted here: deletion is deferred to
        :meth:`cleanup`, invoked only after the final output JSON is durably
        written, so a failure between download and write never destroys
        paid-for results.
        """
        client = self._get_client()
        batch_job = client.batches.get(name=handle.batch_id)
        dest = getattr(batch_job, "dest", None)

        if not dest:
            raise RuntimeError(f"Batch {handle.batch_id} has no results")

        yield from self._iter_results(handle, client, dest)

    def cleanup(self, handle: BatchHandle) -> None:
        """Delete the remote input and result files for a finished batch."""
        client = self._get_client()
        output_file_name = None
        try:
            batch_job = client.batches.get(name=handle.batch_id)
            dest = getattr(batch_job, "dest", None)
            if dest is not None and getattr(dest, "file_name", None):
                output_file_name = dest.file_name
        except Exception as exc:
            logger.warning(
                "Could not resolve result file for batch %s during cleanup: %s",
                handle.batch_id,
                exc,
            )
        self._delete_remote_files(
            client,
            handle.metadata.get("input_file_name"),
            output_file_name,
        )

    @staticmethod
    def _delete_remote_files(client: Any, *file_names: str | None) -> None:
        """Best-effort deletion of remote Google files; never raises."""
        for name in file_names:
            if not name:
                continue
            try:
                client.files.delete(name=name)
                logger.debug("Deleted remote Google file: %s", name)
            except Exception as exc:
                logger.warning("Failed to delete Google file %s: %s", name, exc)

    def _iter_results(
        self, handle: BatchHandle, client: Any, dest: Any
    ) -> Iterator[BatchResultItem]:
        """Yield parsed results from a completed batch's destination."""
        # Check for file-based results
        if hasattr(dest, "file_name") and dest.file_name:
            # Download result file
            file_content = client.files.download(file=dest.file_name)
            text = (
                file_content.decode("utf-8")
                if isinstance(file_content, bytes)
                else str(file_content)
            )

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
                result_item.raw_response = response

                # Extract text content
                candidates = response.get("candidates", [])
                finish_reason = ""
                if candidates:
                    candidate = candidates[0]
                    finish_reason = candidate.get("finishReason", "")
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    text_parts = []
                    for part in parts:
                        if "text" in part:
                            text_parts.append(part["text"])
                    result_item.content = "".join(text_parts)

                # A candidate with no text parts (e.g. finishReason SAFETY or
                # MAX_TOKENS) must not be reported as a successful empty
                # extraction; doing so silently corrupts downstream aggregation.
                if result_item.content:
                    result_item.success = True
                else:
                    result_item.success = False
                    result_item.error = (
                        "No text content in Gemini response "
                        f"(finishReason={finish_reason or 'unknown'})"
                    )

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
                # Prefer the echoed request metadata (explicit correlation);
                # fall back to the positional custom_id_map for batches
                # submitted before metadata was attached.
                custom_id = None
                response_meta = getattr(inline_response, "metadata", None)
                if isinstance(response_meta, dict):
                    custom_id = response_meta.get("key")
                if not custom_id:
                    custom_id_map = handle.metadata.get("custom_id_map", {})
                    for cid, idx in custom_id_map.items():
                        if idx == i:
                            custom_id = cid
                            break
                if custom_id is None:
                    custom_id = f"req-{i + 1}"

                result_item = BatchResultItem(custom_id=str(custom_id))

                if hasattr(inline_response, "error") and inline_response.error:
                    result_item.success = False
                    result_item.error = str(inline_response.error)
                    yield result_item
                    continue

                if hasattr(inline_response, "response") and inline_response.response:
                    response = inline_response.response

                    # Extract text
                    try:
                        result_item.content = response.text
                    except AttributeError:
                        result_item.content = str(response)

                    # Preserve provider metadata and token usage, mirroring
                    # the file-based branch. Without this, every inline batch
                    # (the common, under-20MB case) reported zero tokens and
                    # an empty raw_response.
                    with contextlib.suppress(Exception):
                        dumped = response.model_dump(mode="json")
                        if isinstance(dumped, dict):
                            result_item.raw_response = dumped
                    usage_md = getattr(response, "usage_metadata", None)
                    if usage_md is not None:
                        result_item.input_tokens = int(
                            getattr(usage_md, "prompt_token_count", 0) or 0
                        )
                        result_item.output_tokens = int(
                            getattr(usage_md, "candidates_token_count", 0) or 0
                        )

                    # Try to parse as JSON
                    if result_item.content:
                        try:
                            parsed = json.loads(result_item.content)
                            if isinstance(parsed, dict):
                                result_item.parsed_output = parsed
                        except json.JSONDecodeError:
                            pass

                # Mirror the file-results guard above: a response with no
                # text content (or a result with neither error nor response)
                # must not be reported as a successful empty extraction.
                if result_item.content:
                    result_item.success = True
                else:
                    result_item.success = False
                    result_item.error = (
                        "No text content in Gemini inline batch response"
                    )

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
