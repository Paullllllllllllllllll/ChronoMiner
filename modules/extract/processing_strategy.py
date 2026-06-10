# modules/extract/processing_strategy.py

"""
Processing strategy abstraction for synchronous vs batch execution.
Separates execution logic from file processing orchestration.

Supports multiple LLM providers via LangChain:
- OpenAI (default)
- Anthropic (Claude)
- Google (Gemini)
- OpenRouter (multi-provider access)

Batch processing supports OpenAI, Anthropic, and Google providers.
OpenRouter does not support batch processing.
"""

import asyncio
import contextlib
import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from modules.batch.backends import (
    BatchHandle,
    BatchRequest,
    get_batch_backend,
    supports_batch,
)
from modules.config.capabilities import detect_capabilities
from modules.conversion.json_utils import strip_image_payloads
from modules.images.page_stream import PageError
from modules.llm.langchain_provider import ProviderConfig
from modules.llm.openai_utils import (
    open_extractor,
    process_image_chunk,
    process_text_chunk,
)

logger = logging.getLogger(__name__)

# Any standalone 5xx status code counts as a transient server error. This
# deliberately covers Cloudflare edge codes (520-526) in front of provider
# APIs, which previously slipped past an enumerated 500/502/503 check and
# failed pages without a single retry.
_SERVER_ERROR_CODE_RE = re.compile(r"\b5\d{2}\b")


def classify_transient_error(message: str) -> tuple[bool, bool, bool]:
    """Classify an API error message for retry purposes.

    :param message: The stringified exception, in any casing.
    :return: Tuple ``(is_429, is_timeout, is_server_error)``. The error is
        retryable if any element is True.
    """
    msg = message.lower()
    is_429 = "429" in msg or "rate_limit" in msg
    is_timeout = "timed out" in msg or "timeout" in msg
    is_server_error = (
        bool(_SERVER_ERROR_CODE_RE.search(msg))
        or "internalservererror" in msg
        or "upstream" in msg
        # Cloudflare-style bodies self-declare retryability.
        or "'retryable': true" in msg
        or '"retryable": true' in msg
        or ("connection" in msg and ("reset" in msg or "refused" in msg))
    )
    return is_429, is_timeout, is_server_error


class ProcessingStrategy(ABC):
    """Abstract base class for processing strategies."""

    @abstractmethod
    async def process_chunks(
        self,
        chunks: list[str],
        handler: Any,
        dev_message: str,
        model_config: dict[str, Any],
        schema: dict[str, Any],
        file_path: Path,
        temp_jsonl_path: Path,
        console_print: Any,
        completed_chunk_indices: set | None = None,
        image_chunks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Process text chunks using the strategy.

        :param chunks: List of text chunks to process
        :param handler: Schema handler instance
        :param dev_message: Developer/system message
        :param model_config: Model configuration
        :param schema: JSON schema
        :param file_path: Source file path
        :param temp_jsonl_path: Temporary JSONL file path
        :param console_print: Console print function
        :param completed_chunk_indices: 1-based indices of chunks already processed
            (for resume)
        :param image_chunks: Optional list of image chunk dicts with 'base64',
            'mime_type', 'detail' keys. When provided, process_image_chunk() is
            used instead of process_text_chunk().
        :return: List of processing results
        """
        pass


class SynchronousProcessingStrategy(ProcessingStrategy):
    """Synchronous (real-time) processing strategy."""

    def __init__(self, concurrency_config: dict[str, Any] | None = None) -> None:
        """
        Initialize synchronous processing strategy.

        :param concurrency_config: Concurrency configuration
        """
        self.concurrency_config = concurrency_config or {}

    async def process_chunks(
        self,
        chunks: list[str],
        handler: Any,
        dev_message: str,
        model_config: dict[str, Any],
        schema: dict[str, Any],
        file_path: Path,
        temp_jsonl_path: Path,
        console_print: Any,
        completed_chunk_indices: set | None = None,
        image_chunks: list[dict[str, Any]] | None = None,
        context_image_data: dict[str, Any] | None = None,
        image_source: AsyncIterator[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Process chunks synchronously with concurrent API calls.

        Two execution modes:

        - List mode (text chunks, or a pre-materialized ``image_chunks``
          list): one task per chunk, bounded by a semaphore.
        - Streaming mode (``image_source`` given): a producer task pulls
          page payloads from the async iterator into a bounded queue and
          ``concurrency_limit`` workers consume it, so only a small,
          constant number of page payloads is in memory at any time.
        """
        # Detect provider: prefer explicit config, fall back to auto-detection
        model_name = model_config["extraction_model"]["name"]
        config_provider = model_config["extraction_model"].get("provider")
        valid_providers = ("openai", "anthropic", "google", "openrouter", "custom")
        if config_provider and config_provider in valid_providers:
            provider = config_provider
        else:
            provider = ProviderConfig._detect_provider(model_name)
        api_key = ProviderConfig._get_api_key(provider)

        if not api_key:
            error_msg = (
                f"API key not found for provider {provider}. "
                "Set the appropriate environment variable."
            )
            logger.error(error_msg)
            console_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        skip_indices = completed_chunk_indices or set()
        chunks_to_process = [
            (idx, chunk)
            for idx, chunk in enumerate(chunks, 1)
            if idx not in skip_indices
        ]
        pending_count = len(chunks_to_process)
        if skip_indices:
            console_print(
                f"[INFO] Resuming: {pending_count} new chunks/pages "
                f"to process ({len(skip_indices)} already done)"
            )
        console_print(
            f"[INFO] Starting synchronous processing of {pending_count} chunks/pages..."
        )
        results: list[dict[str, Any]] = []

        # Extract concurrency settings
        extraction_cfg = (self.concurrency_config.get("concurrency", {}) or {}).get(
            "extraction", {}
        ) or {}
        retry_cfg = extraction_cfg.get("retry", {}) or {}
        total_chunks = len(chunks)
        try:
            configured_limit = int(
                extraction_cfg.get("concurrency_limit", total_chunks or 1)
            )
        except (ValueError, TypeError):
            configured_limit = total_chunks or 1
        concurrency_limit = max(1, min(configured_limit, total_chunks or 1))
        delay_between_tasks = float(
            extraction_cfg.get("delay_between_tasks", 0.0) or 0.0
        )

        try:
            retry_attempts = int(retry_cfg.get("attempts", 1))
        except (ValueError, TypeError):
            retry_attempts = 1
        retry_attempts = max(1, retry_attempts)
        wait_min_seconds = float(retry_cfg.get("wait_min_seconds", 1.0) or 1.0)
        wait_max_seconds = float(retry_cfg.get("wait_max_seconds", 60.0) or 60.0)
        jitter_max_seconds = float(retry_cfg.get("jitter_max_seconds", 0.0) or 0.0)

        # Detect prompt caching capability for Anthropic models
        caps = detect_capabilities(model_name, provider=provider)
        enable_cache = caps.supports_prompt_caching

        is_visual = image_source is not None or bool(image_chunks)
        unit_label = "page" if is_visual else "chunk"
        prompt_path = (
            Path("prompts/image_extraction_prompt.txt")
            if is_visual
            else Path("prompts/text_extraction_prompt.txt")
        )

        file_mode = "a" if skip_indices else "w"
        async with open_extractor(
            api_key=api_key,
            prompt_path=prompt_path,
            model=model_config["extraction_model"]["name"],
            provider=provider,
            model_config_override=model_config,
            concurrency_config_override=self.concurrency_config,
        ) as extractor:
            with temp_jsonl_path.open(file_mode, encoding="utf-8") as tempf:
                # Serialize writes to the shared handle: concurrent coroutines
                # otherwise interleave write+flush on one file object. Created
                # here (inside the running loop), never at module scope.
                write_lock = asyncio.Lock()

                async def call_and_record(
                    idx: int, chunk: str, img_data: dict[str, Any] | None
                ) -> dict[str, Any]:
                    """Run one unit through the retry loop and persist it."""
                    for attempt in range(retry_attempts):
                        try:
                            # Route to image or text processing
                            if img_data is not None:
                                result = await process_image_chunk(
                                    image_base64=img_data["base64"],
                                    mime_type=img_data["mime_type"],
                                    extractor=extractor,
                                    system_message=dev_message,
                                    json_schema=schema,
                                    image_detail=img_data.get("detail"),
                                    enable_cache_control=enable_cache,
                                    context_image_data=context_image_data,
                                )
                            else:
                                result = await process_text_chunk(
                                    text_chunk=chunk,
                                    extractor=extractor,
                                    system_message=dev_message,
                                    json_schema=schema,
                                    enable_cache_control=enable_cache,
                                    context_image_data=context_image_data,
                                )

                            # Drop base64 payloads from the persisted request
                            # metadata; they grew temp files to ~1 GB on large
                            # PDFs and pinned every page's image in RAM via
                            # the results list.
                            result = strip_image_payloads(result)

                            # chunk_index drives ordering in
                            # _generate_output_files; without it the final
                            # records sort by `None or 0` (all equal) and
                            # land in completion order.
                            response_obj: dict[str, Any] = {
                                "custom_id": f"{file_path.stem}-chunk-{idx}",
                                "chunk_index": idx,
                                "response": {"body": result},
                            }
                            provenance = (img_data or {}).get("image_provenance")
                            if provenance:
                                response_obj["image_provenance"] = provenance
                            async with write_lock:
                                tempf.write(json.dumps(response_obj) + "\n")
                                tempf.flush()

                            console_print(
                                f"[INFO] Processed {unit_label} {idx}/{total_chunks}"
                            )
                            return result
                        except (
                            Exception
                        ) as e:  # Broad: LangChain/API errors are diverse
                            is_429, is_timeout, is_server_error = (
                                classify_transient_error(str(e))
                            )
                            is_retryable = is_429 or is_timeout or is_server_error

                            if is_retryable and attempt < (retry_attempts - 1):
                                base_wait = min(
                                    wait_max_seconds,
                                    wait_min_seconds * (2**attempt),
                                )
                                jitter = (
                                    random.uniform(0.0, jitter_max_seconds)
                                    if jitter_max_seconds > 0
                                    else 0.0
                                )
                                wait_s = min(wait_max_seconds, base_wait + jitter)
                                if is_429:
                                    reason = "Rate-limited"
                                elif is_timeout:
                                    reason = "Timed out"
                                else:
                                    reason = "Server error"
                                logger.warning(
                                    "%s on %s %s (attempt %s/%s). "
                                    "Waiting %.1fs and retrying.",
                                    reason,
                                    unit_label,
                                    idx,
                                    attempt + 1,
                                    retry_attempts,
                                    wait_s,
                                )
                                await asyncio.sleep(wait_s)
                                continue

                            logger.error(
                                "Error processing %s %s: %s",
                                unit_label,
                                idx,
                                e,
                            )
                            console_print(
                                f"[ERROR] Failed to process {unit_label} {idx}: {e}"
                            )
                            # Carry chunk_index: failed chunks write no
                            # temp record, so the caller cannot recover the
                            # index from gather order otherwise.
                            return {"error": str(e), "chunk_index": idx}
                    # If all retries exhausted without returning, return error
                    return {
                        "error": (
                            f"Max retries ({retry_attempts}) exhausted "
                            f"for {unit_label} {idx}"
                        ),
                        "chunk_index": idx,
                    }

                if image_source is not None:
                    results = await self._consume_image_source(
                        image_source=image_source,
                        call_and_record=call_and_record,
                        console_print=console_print,
                        concurrency_limit=concurrency_limit,
                        delay_between_tasks=delay_between_tasks,
                        unit_label=unit_label,
                    )
                else:
                    semaphore = asyncio.Semaphore(concurrency_limit)

                    async def process_single_chunk(
                        idx: int, chunk: str
                    ) -> dict[str, Any]:
                        """Process a single chunk with semaphore control."""
                        if delay_between_tasks > 0:
                            await asyncio.sleep(delay_between_tasks)
                        async with semaphore:
                            img_data = (
                                image_chunks[idx - 1]
                                if is_visual and image_chunks is not None
                                else None
                            )
                            return await call_and_record(idx, chunk, img_data)

                    # Process chunks (skipping already-completed ones)
                    tasks = [
                        process_single_chunk(idx, chunk)
                        for idx, chunk in chunks_to_process
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=False)

        console_print(
            f"[SUCCESS] Completed synchronous processing of {len(results)} chunks/pages"
        )
        return results

    @staticmethod
    async def _consume_image_source(
        *,
        image_source: AsyncIterator[Any],
        call_and_record: Any,
        console_print: Any,
        concurrency_limit: int,
        delay_between_tasks: float,
        unit_label: str,
    ) -> list[dict[str, Any]]:
        """Producer-consumer execution over a streaming page source.

        A single producer task drains ``image_source`` (which renders and
        preprocesses one page at a time) into a bounded queue; exactly
        ``concurrency_limit`` workers consume it. Peak memory is the queue
        buffer of compact JPEG payloads plus one full-resolution page
        inside the producer, regardless of document length.
        """
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max(2, 2 * concurrency_limit))
        sentinel = object()
        producer_errors: list[BaseException] = []

        async def producer() -> None:
            try:
                async for item in image_source:
                    await queue.put(item)
            except Exception as e:  # Surface after workers drain the queue
                producer_errors.append(e)
            finally:
                for _ in range(concurrency_limit):
                    await queue.put(sentinel)

        async def worker(worker_id: int) -> list[dict[str, Any]]:
            # Stagger worker start-up like the per-task delay in list mode.
            if delay_between_tasks > 0 and worker_id > 0:
                await asyncio.sleep(delay_between_tasks * worker_id)
            collected: list[dict[str, Any]] = []
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, PageError):
                    console_print(
                        f"[ERROR] Failed to render {unit_label} "
                        f"{item.index}: {item.error}"
                    )
                    collected.append({"error": item.error, "chunk_index": item.index})
                    continue
                collected.append(await call_and_record(item.index, "", item.as_chunk()))
            return collected

        producer_task = asyncio.create_task(producer())
        try:
            worker_results = await asyncio.gather(
                *(worker(i) for i in range(concurrency_limit))
            )
        finally:
            # No-op when the producer already finished; on cancellation or a
            # worker crash this unblocks a producer stuck on a full queue.
            producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer_task
        if producer_errors:
            raise producer_errors[0]
        return [record for batch in worker_results for record in batch]


class BatchProcessingStrategy(ProcessingStrategy):
    """Batch (deferred) processing strategy.

    Supports multiple providers:
    - OpenAI: Uses OpenAI Batch API with /v1/responses endpoint
    - Anthropic: Uses Anthropic Message Batches API
    - Google: Uses Google Gemini Batch API
    - OpenRouter: Not supported (falls back to sync or raises error)
    """

    def __init__(self, concurrency_config: dict[str, Any] | None = None) -> None:
        """
        Initialize batch processing strategy.

        :param concurrency_config: Concurrency configuration (used for service_tier,
            etc.)
        """
        self.concurrency_config = concurrency_config or {}

    async def process_chunks(
        self,
        chunks: list[str],
        handler: Any,
        dev_message: str,
        model_config: dict[str, Any],
        schema: dict[str, Any],
        file_path: Path,
        temp_jsonl_path: Path,
        console_print: Any,
        completed_chunk_indices: set | None = None,
        image_chunks: list[dict[str, Any]] | None = None,
        context_image_data: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Prepare and submit batch processing job using provider-agnostic backend."""
        if context_image_data is not None:
            console_print(
                "[WARNING] Context image is not yet supported in batch mode. Ignoring."
            )

        # Detect provider from model config
        tm = model_config.get("extraction_model", {})
        provider = tm.get("provider")
        if not provider:
            model_name = tm.get("name", "")
            provider = ProviderConfig._detect_provider(model_name)

        # Check if provider supports batch processing
        if not supports_batch(provider):
            error_msg = (
                f"Provider '{provider}' does not support batch processing. "
                f"Use synchronous mode or switch to openai, anthropic, or google."
            )
            console_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        # Build BatchRequest objects for the provider-agnostic backend
        is_visual_batch = image_chunks is not None and len(image_chunks) > 0
        batch_requests: list[BatchRequest] = []

        if is_visual_batch:
            assert image_chunks is not None
            console_print(
                f"[INFO] Preparing visual batch processing for {len(image_chunks)} "
                f"page(s) (provider: {provider})..."
            )
            for idx, img in enumerate(image_chunks, 1):
                custom_id = f"{file_path.stem}-page-{idx}"
                batch_requests.append(
                    BatchRequest(
                        custom_id=custom_id,
                        image_base64=img["base64"],
                        mime_type=img["mime_type"],
                        image_detail=img.get("detail"),
                        order_index=idx,
                        metadata={
                            "file_path": str(file_path),
                            "page_index": idx,
                            "total_pages": len(image_chunks),
                        },
                    )
                )
        else:
            console_print(
                f"[INFO] Preparing batch processing for {len(chunks)} "
                f"chunks (provider: {provider})..."
            )
            for idx, chunk in enumerate(chunks, 1):
                custom_id = f"{file_path.stem}-chunk-{idx}"
                batch_requests.append(
                    BatchRequest(
                        custom_id=custom_id,
                        text=chunk,
                        order_index=idx,
                        metadata={
                            "file_path": str(file_path),
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                        },
                    )
                )

        # Get the appropriate batch backend
        try:
            backend = get_batch_backend(provider)
        except ValueError as e:
            console_print(f"[ERROR] {e}")
            raise

        # Extract schema name from handler if available
        schema_name = getattr(handler, "schema_name", None) or "ExtractionSchema"

        # Inject service_tier from concurrency_config into model_config for the
        # backend (CM-2)
        extraction_cfg = (self.concurrency_config.get("concurrency", {}) or {}).get(
            "extraction", {}
        ) or {}
        service_tier = extraction_cfg.get("service_tier")
        effective_model_config = model_config
        if service_tier:
            tm_copy = dict(model_config.get("extraction_model", {}))
            tm_copy["service_tier"] = service_tier
            effective_model_config = {**model_config, "extraction_model": tm_copy}

        # Submit batch using the provider-agnostic backend
        try:
            console_print(f"[INFO] Submitting batch to {provider}...")
            handle: BatchHandle = await asyncio.to_thread(
                backend.submit_batch,
                batch_requests,
                effective_model_config,
                system_prompt=dev_message,
                schema=schema,
                schema_name=schema_name,
            )

            # Write tracking record to temp JSONL file
            tracking_record = {
                "batch_tracking": {
                    "batch_id": handle.batch_id,
                    "provider": handle.provider,
                    "timestamp": int(time.time()),
                    "request_count": len(batch_requests),
                    "metadata": handle.metadata,
                }
            }

            # Also write batch request metadata for result correlation
            with temp_jsonl_path.open("w", encoding="utf-8") as tempf:
                # Write request metadata lines first
                for req in batch_requests:
                    request_meta = {
                        "batch_request": {
                            "custom_id": req.custom_id,
                            "order_index": req.order_index,
                            "metadata": req.metadata,
                        }
                    }
                    tempf.write(json.dumps(request_meta) + "\n")
                # Write tracking record at the end
                tempf.write(json.dumps(tracking_record) + "\n")

            console_print(
                f"[SUCCESS] Batch submitted successfully. Batch ID: {handle.batch_id}"
            )
            logger.info(
                "Batch submitted to %s. ID: %s, Requests: %d, File: %s",
                provider,
                handle.batch_id,
                len(batch_requests),
                temp_jsonl_path,
            )

        except (
            Exception
        ) as e:  # Broad: batch backends raise diverse provider-specific errors
            logger.error(f"Error during batch submission: {e}", exc_info=True)
            console_print(f"[ERROR] Failed to submit batch: {e}")
            raise

        console_print(
            f"[SUCCESS] Submitted batch to {provider}. "
            f"Use check_batches.py to monitor progress."
        )

        # Return empty list since results will be retrieved later
        return []


def create_processing_strategy(
    use_batch: bool, concurrency_config: dict[str, Any] | None = None
) -> ProcessingStrategy:
    """
    Factory function to create appropriate processing strategy.

    :param use_batch: Whether to use batch processing
    :param concurrency_config: Concurrency configuration
    :return: ProcessingStrategy instance
    """
    if use_batch:
        return BatchProcessingStrategy(concurrency_config)
    else:
        return SynchronousProcessingStrategy(concurrency_config)
