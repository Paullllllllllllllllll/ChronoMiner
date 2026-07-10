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
from modules.extract.resume import build_temp_header
from modules.images.page_stream import PageError
from modules.infra.chunking import TextProcessor
from modules.infra.rate_limit import await_capacity, get_shared_rate_limiter
from modules.infra.token_tracker import get_token_tracker
from modules.llm.langchain_provider import ProviderConfig
from modules.llm.openai_utils import (
    open_extractor,
    process_image_chunk,
    process_text_chunk,
)
from modules.llm.prompt_utils import PROMPTS_DIR

logger = logging.getLogger(__name__)


# Per-request serialization overhead (custom_id, method/url envelope, JSON
# punctuation, system prompt reference) added to the payload estimate when
# partitioning a batch by byte size. Deliberately generous so a split stays
# safely under the provider's hard limit.
_BATCH_REQUEST_OVERHEAD_BYTES = 2048


def _estimate_request_bytes(req: BatchRequest) -> int:
    """Approximate the submitted JSONL byte size of one batch request."""
    size = _BATCH_REQUEST_OVERHEAD_BYTES
    if req.text:
        size += len(req.text.encode("utf-8"))
    if req.image_base64:
        size += len(req.image_base64)
    return size


def _partition_batch_requests(
    requests: list[BatchRequest], max_count: int, max_bytes: int
) -> list[list[BatchRequest]]:
    """Split *requests* into parts respecting per-batch count and byte limits.

    A part is closed when adding the next request would exceed either
    ``max_count`` or ``max_bytes``. A single request larger than ``max_bytes``
    still occupies its own part (it cannot be split further here).
    """
    parts: list[list[BatchRequest]] = []
    current: list[BatchRequest] = []
    current_bytes = 0
    for req in requests:
        size = _estimate_request_bytes(req)
        if current and (len(current) >= max_count or current_bytes + size > max_bytes):
            parts.append(current)
            current = []
            current_bytes = 0
        current.append(req)
        current_bytes += size
    if current:
        parts.append(current)
    return parts or [[]]


def _budget_deferred(idx: int) -> dict[str, Any]:
    """Marker for a chunk/page skipped because the daily token budget was
    exhausted. Deferred units are never written to the temp JSONL, so the
    existing resume path re-processes them on the next pass."""
    return {"budget_deferred": True, "chunk_index": idx}


# A 5xx status code counts as a transient server error, but ONLY when it
# appears in a status/HTTP/error-code context or next to a canonical 5xx
# reason phrase. The former blanket ``\b5\d{2}\b`` false-positived on any
# stray number (e.g. "line 502 of file.py"), burning up to 25 retries on a
# non-retryable error. Covers Cloudflare edge codes (520-526) too.
_SERVER_ERROR_CODE_RE = re.compile(
    r"(?:status(?:[ _]?code)?|http|error[ _]?code|code)\s*[:=]?\s*5\d{2}\b"
    r"|\b5\d{2}\b\s*(?:internal server error|server error|bad gateway"
    r"|service unavailable|gateway timeout|origin)",
    re.IGNORECASE,
)


def classify_transient_error(
    message: str, exc: BaseException | None = None
) -> tuple[bool, bool, bool]:
    """Classify an API error for retry purposes.

    :param message: The stringified exception, in any casing.
    :param exc: The exception object, if available. A structured
        ``status_code`` attribute is consulted first and is authoritative;
        the message regex is only a fallback.
    :return: Tuple ``(is_429, is_timeout, is_server_error)``. The error is
        retryable if any element is True.
    """
    msg = message.lower()

    # Structured status code first (authoritative when present).
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = getattr(exc, "code", None)
        if not isinstance(status_code, int):
            status_code = None

    is_429 = status_code == 429 or "429" in msg or "rate_limit" in msg
    is_timeout = "timed out" in msg or "timeout" in msg
    is_server_error = (
        (status_code is not None and 500 <= status_code <= 599)
        or bool(_SERVER_ERROR_CODE_RE.search(message))
        or "internalservererror" in msg
        or "upstream" in msg
        # Cloudflare-style bodies self-declare retryability.
        or "'retryable': true" in msg
        or '"retryable": true' in msg
        or ("connection" in msg and ("reset" in msg or "refused" in msg))
        # openai SDK APIConnectionError stringifies to the bare message
        # "Connection error." (transport-level failure, e.g. a stale
        # keep-alive connection the server already closed). Always
        # transient: a retry opens a fresh connection. Frequent under
        # service_tier=flex, which closes connections after each response.
        or "connection error" in msg
    )
    return is_429, is_timeout, is_server_error


def parse_retry_after(exc: BaseException | None) -> float | None:
    """Extract a Retry-After delay (in seconds) from an exception's HTTP headers.

    Reads the ``Retry-After`` header off the exception's response (or a
    top-level ``headers`` attribute) across the openai/anthropic SDK exception
    shapes, tolerating both the integer/float seconds form and the HTTP-date
    form. Returns ``None`` when no usable value is present. Defensive: any
    parsing problem yields ``None`` rather than raising.
    """
    if exc is None:
        return None
    try:
        headers = None
        resp = getattr(exc, "response", None)
        if resp is not None:
            headers = getattr(resp, "headers", None)
        if headers is None:
            headers = getattr(exc, "headers", None)
        if headers is None:
            return None

        getter = getattr(headers, "get", None)
        if not callable(getter):
            return None
        raw = getter("retry-after")
        if raw is None:
            raw = getter("Retry-After")
        if raw is None:
            return None

        value = str(raw).strip()
        if not value:
            return None

        # Seconds form (integer or float).
        try:
            return max(0.0, float(value))
        except ValueError:
            pass

        # HTTP-date form (e.g. "Wed, 21 Oct 2026 07:28:00 GMT").
        from email.utils import parsedate_to_datetime

        try:
            target = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if target is None:
            return None
        import datetime as _dt

        now = _dt.datetime.now(target.tzinfo) if target.tzinfo else _dt.datetime.now()
        return max(0.0, (target - now).total_seconds())
    except Exception:
        return None


def commit_tokens_from_exception(exc: BaseException) -> None:
    """Best-effort: recover token usage from a failed call and commit it.

    Provider SDK exceptions often carry usage data in ``exc.body["usage"]`` or
    ``exc.response.json()["usage"]``; recovering it keeps the daily budget
    honest even for calls that ultimately errored. Tries ``total_tokens``, then
    ``prompt_tokens`` + ``completion_tokens`` (OpenAI), then ``input_tokens`` +
    ``output_tokens`` (Anthropic). Never raises.
    """
    try:
        usage: dict[str, Any] | None = None

        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            candidate = body.get("usage")
            if isinstance(candidate, dict):
                usage = candidate

        if usage is None:
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    resp_json = resp.json()
                    if isinstance(resp_json, dict) and isinstance(
                        resp_json.get("usage"), dict
                    ):
                        usage = resp_json["usage"]
                except Exception:
                    usage = None

        if not isinstance(usage, dict):
            return

        total = usage.get("total_tokens")
        if not isinstance(total, int) or total <= 0:
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            if (
                isinstance(prompt, int)
                and isinstance(completion, int)
                and (prompt + completion) > 0
            ):
                total = prompt + completion
            else:
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                if isinstance(inp, int) and isinstance(out, int) and (inp + out) > 0:
                    total = inp + out

        if isinstance(total, int) and total > 0:
            get_token_tracker().add_tokens(total)
            logger.info(
                "[TOKEN] Recovered %s tokens from a failed request.", f"{total:,}"
            )
    except Exception:
        logger.debug("Token recovery from exception failed", exc_info=True)


def _append_jsonl_line(handle: Any, line: str) -> None:
    """Write and flush one line to *handle* (run off-loop via to_thread)."""
    handle.write(line)
    handle.flush()


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
        chunk_indices: list[int] | None = None,
        chunk_ranges: list[tuple[int, int]] | None = None,
    ) -> list[dict[str, Any]]:
        """Process chunks synchronously with concurrent API calls.

        ``chunk_indices`` carries the ABSOLUTE 1-based index of each chunk in
        the full document (before any slice) so that a sliced run (e.g.
        ``--page-range 50-60``) writes ``custom_id``/``chunk_index`` values in
        document space rather than slice-relative 1..N. When ``None`` the
        indices default to ``1..len(chunks)``. ``chunk_ranges`` carries the
        aligned ``(start_line, end_line)`` of each chunk, stamped as
        ``chunk_range`` on the persisted record.

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
        # Map each chunk position to its absolute document index. A sliced run
        # supplies chunk_indices in document space; otherwise 1..N.
        if chunk_indices is None:
            abs_indices = list(range(1, len(chunks) + 1))
        else:
            abs_indices = list(chunk_indices)
        chunks_to_process = [
            (abs_indices[pos], pos, chunk)
            for pos, chunk in enumerate(chunks)
            if abs_indices[pos] not in skip_indices
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
            retry_attempts = int(retry_cfg.get("attempts", 8))
        except (ValueError, TypeError):
            retry_attempts = 8
        retry_attempts = max(1, retry_attempts)
        wait_min_seconds = float(retry_cfg.get("wait_min_seconds", 2.5) or 2.5)
        wait_max_seconds = float(retry_cfg.get("wait_max_seconds", 120.0) or 120.0)
        jitter_max_seconds = float(retry_cfg.get("jitter_max_seconds", 0.0) or 0.0)

        # Per-provider shared rate limiter: throttles synchronous calls under
        # the configured windows BEFORE each API call (permissive defaults when
        # unconfigured). Batch submission never passes through here.
        rate_limiter = get_shared_rate_limiter(provider)

        # Detect prompt caching capability for Anthropic models
        caps = detect_capabilities(model_name, provider=provider)
        enable_cache = caps.supports_prompt_caching

        is_visual = image_source is not None or bool(image_chunks)
        unit_label = "page" if is_visual else "chunk"
        prompt_path = (
            PROMPTS_DIR / "image_extraction_prompt.txt"
            if is_visual
            else PROMPTS_DIR / "text_extraction_prompt.txt"
        )

        # Append only to an existing, current-format temp file; otherwise start
        # fresh and stamp the format-version header as the first line.
        resume_append = bool(skip_indices) and temp_jsonl_path.exists()
        file_mode = "a" if resume_append else "w"
        async with open_extractor(
            api_key=api_key,
            prompt_path=prompt_path,
            model=model_config["extraction_model"]["name"],
            provider=provider,
            model_config_override=model_config,
            concurrency_config_override=self.concurrency_config,
        ) as extractor:
            with temp_jsonl_path.open(file_mode, encoding="utf-8") as tempf:
                if file_mode == "w":
                    tempf.write(json.dumps(build_temp_header()) + "\n")
                    tempf.flush()
                # Serialize writes to the shared handle: concurrent coroutines
                # otherwise interleave write+flush on one file object. Created
                # here (inside the running loop), never at module scope.
                write_lock = asyncio.Lock()

                async def call_and_record(
                    idx: int,
                    chunk: str,
                    img_data: dict[str, Any] | None,
                    chunk_range: tuple[int, int] | None = None,
                ) -> dict[str, Any]:
                    """Run one unit through the retry loop and persist it."""
                    for attempt in range(retry_attempts):
                        # Acquire rate-limit capacity off the event loop before
                        # each API call so bursts stay under the provider caps.
                        await await_capacity(rate_limiter)
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
                            if chunk_range is not None:
                                response_obj["chunk_range"] = list(chunk_range)
                            provenance = (img_data or {}).get("image_provenance")
                            if provenance:
                                response_obj["image_provenance"] = provenance
                            line = json.dumps(response_obj) + "\n"
                            # Serialize the append+flush under the lock, but run
                            # the blocking write off the event loop. The lock is
                            # held across the awaited to_thread call, so ordering
                            # and non-interleaving are preserved.
                            async with write_lock:
                                await asyncio.to_thread(_append_jsonl_line, tempf, line)

                            rate_limiter.report_success()
                            console_print(
                                f"[INFO] Processed {unit_label} {idx}/{total_chunks}"
                            )
                            return result
                        except (
                            Exception
                        ) as e:  # Broad: LangChain/API errors are diverse
                            is_429, is_timeout, is_server_error = (
                                classify_transient_error(str(e), e)
                            )
                            is_retryable = is_429 or is_timeout or is_server_error

                            # Feed the adaptive limiter (429/5xx tighten it) and
                            # recover any usage the failed call still reported so
                            # the daily budget stays honest.
                            rate_limiter.report_error(
                                is_rate_limit=is_429 or is_server_error
                            )
                            commit_tokens_from_exception(e)

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
                                # Honor a server-provided Retry-After: never wait
                                # less than it asks (still capped at wait_max).
                                retry_after = parse_retry_after(e)
                                if retry_after is not None:
                                    wait_s = min(
                                        wait_max_seconds, max(wait_s, retry_after)
                                    )
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

                # Chunk-level token-budget gate. The tracker is a no-op when
                # the daily limit is disabled (try_reserve returns 0); when
                # enabled, the first reservation that cannot fit sets
                # ``exhausted`` so no new units are admitted, in-flight units
                # drain, and the caller (_execute_extraction) waits for reset
                # and re-passes over the still-pending units.
                tracker = get_token_tracker()
                exhausted = asyncio.Event()

                if image_source is not None:
                    results = await self._consume_image_source(
                        image_source=image_source,
                        call_and_record=call_and_record,
                        console_print=console_print,
                        concurrency_limit=concurrency_limit,
                        delay_between_tasks=delay_between_tasks,
                        unit_label=unit_label,
                        tracker=tracker,
                        exhausted=exhausted,
                    )
                else:
                    semaphore = asyncio.Semaphore(concurrency_limit)

                    async def process_single_chunk(
                        idx: int, pos: int, chunk: str
                    ) -> dict[str, Any]:
                        """Process a single chunk with semaphore control and a
                        chunk-level token-budget gate.

                        ``idx`` is the absolute document index (used for the
                        record); ``pos`` is the position in the (possibly sliced)
                        chunk list (used to align image_chunks / chunk_ranges).
                        """
                        if exhausted.is_set():
                            return _budget_deferred(idx)
                        if delay_between_tasks > 0:
                            await asyncio.sleep(delay_between_tasks)
                        async with semaphore:
                            if exhausted.is_set():
                                return _budget_deferred(idx)
                            img_data = (
                                image_chunks[pos]
                                if is_visual and image_chunks is not None
                                else None
                            )
                            rng = (
                                chunk_ranges[pos]
                                if chunk_ranges is not None and pos < len(chunk_ranges)
                                else None
                            )
                            # Seed the reservation with the chunk's tiktoken
                            # input count for text; image payloads have no cheap
                            # pre-count, so they fall back to the rolling EWMA.
                            estimate = (
                                None
                                if img_data is not None
                                else TextProcessor.estimate_tokens(chunk)
                            )
                            reserved = tracker.try_reserve(estimate)
                            if reserved is None:
                                exhausted.set()
                                return _budget_deferred(idx)
                            try:
                                return await call_and_record(idx, chunk, img_data, rng)
                            finally:
                                tracker.release(reserved)

                    # Process chunks (skipping already-completed ones)
                    tasks = [
                        process_single_chunk(idx, pos, chunk)
                        for idx, pos, chunk in chunks_to_process
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
        tracker: Any,
        exhausted: asyncio.Event,
    ) -> list[dict[str, Any]]:
        """Producer-consumer execution over a streaming page source.

        A single producer task drains ``image_source`` (which renders and
        preprocesses one page at a time) into a bounded queue; exactly
        ``concurrency_limit`` workers consume it. Peak memory is the queue
        buffer of compact JPEG payloads plus one full-resolution page
        inside the producer, regardless of document length.

        Each page passes the chunk-level token-budget gate before its API
        call: when a reservation cannot fit, ``exhausted`` is set, the
        producer stops rendering new pages, and remaining queued pages are
        returned as deferred markers for the caller to re-pass after a reset.
        """
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max(2, 2 * concurrency_limit))
        sentinel = object()
        producer_errors: list[BaseException] = []

        async def producer() -> None:
            try:
                async for item in image_source:
                    if exhausted.is_set():
                        break
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
                if exhausted.is_set():
                    collected.append(_budget_deferred(item.index))
                    continue
                reserved = tracker.try_reserve()
                if reserved is None:
                    exhausted.set()
                    collected.append(_budget_deferred(item.index))
                    continue
                try:
                    collected.append(
                        await call_and_record(item.index, "", item.as_chunk())
                    )
                finally:
                    tracker.release(reserved)
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
        chunk_indices: list[int] | None = None,
        chunk_ranges: list[tuple[int, int]] | None = None,
    ) -> list[dict[str, Any]]:
        """Prepare and submit batch processing job using provider-agnostic backend.

        ``chunk_indices`` supplies absolute document indices for a sliced run so
        custom_ids match the sync path; ``chunk_ranges`` supplies aligned line
        ranges recorded on each request's metadata.
        """
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
            for pos, img in enumerate(image_chunks):
                idx = chunk_indices[pos] if chunk_indices is not None else pos + 1
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
            for pos, chunk in enumerate(chunks):
                idx = chunk_indices[pos] if chunk_indices is not None else pos + 1
                custom_id = f"{file_path.stem}-chunk-{idx}"
                meta: dict[str, Any] = {
                    "file_path": str(file_path),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
                if chunk_ranges is not None and pos < len(chunk_ranges):
                    meta["chunk_range"] = list(chunk_ranges[pos])
                batch_requests.append(
                    BatchRequest(
                        custom_id=custom_id,
                        text=chunk,
                        order_index=idx,
                        metadata=meta,
                    )
                )

        # Batch resume parity: drop requests whose absolute index is already
        # complete in a prior output, so a re-run only submits the remainder.
        skip = completed_chunk_indices or set()
        if skip:
            before = len(batch_requests)
            batch_requests = [r for r in batch_requests if r.order_index not in skip]
            skipped = before - len(batch_requests)
            if skipped:
                console_print(
                    f"[INFO] Resume: skipping {skipped} already-completed "
                    f"request(s); submitting {len(batch_requests)}."
                )
        if not batch_requests:
            console_print("[INFO] Nothing to submit: all requests already completed.")
            return []

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

        # Partition requests so no single submission exceeds the provider's
        # per-batch request-count or byte limits. Each part is submitted as its
        # own batch with its own tracking record and temp file; check_batches
        # merges the ``_part{n}`` files back into one final output.
        max_count = int(getattr(backend, "max_batch_size", 50000))
        max_bytes = int(getattr(backend, "max_batch_bytes", 150 * 1024 * 1024))
        parts = _partition_batch_requests(batch_requests, max_count, max_bytes)
        multi_part = len(parts) > 1
        if multi_part:
            console_print(
                f"[INFO] Splitting {len(batch_requests)} request(s) into "
                f"{len(parts)} part(s) to respect {provider} batch limits."
            )

        submitted_batch_ids: list[str] = []
        try:
            for part_no, part_requests in enumerate(parts, 1):
                console_print(
                    f"[INFO] Submitting batch to {provider}"
                    f"{f' (part {part_no}/{len(parts)})' if multi_part else ''}..."
                )
                handle: BatchHandle = await asyncio.to_thread(
                    backend.submit_batch,
                    part_requests,
                    effective_model_config,
                    system_prompt=dev_message,
                    schema=schema,
                    schema_name=schema_name,
                )
                submitted_batch_ids.append(handle.batch_id)

                tracking_record = {
                    "batch_tracking": {
                        "batch_id": handle.batch_id,
                        "provider": handle.provider,
                        "timestamp": int(time.time()),
                        "request_count": len(part_requests),
                        "metadata": handle.metadata,
                    }
                }

                part_temp_path = (
                    temp_jsonl_path
                    if not multi_part
                    else temp_jsonl_path.with_name(
                        f"{temp_jsonl_path.stem}_part{part_no}.jsonl"
                    )
                )
                # Write request metadata lines first, tracking record last.
                with part_temp_path.open("w", encoding="utf-8") as tempf:
                    for req in part_requests:
                        request_meta = {
                            "batch_request": {
                                "custom_id": req.custom_id,
                                "order_index": req.order_index,
                                "metadata": req.metadata,
                            }
                        }
                        tempf.write(json.dumps(request_meta) + "\n")
                    tempf.write(json.dumps(tracking_record) + "\n")

                console_print(
                    "[SUCCESS] Batch submitted successfully. "
                    f"Batch ID: {handle.batch_id}"
                )
                logger.info(
                    "Batch submitted to %s. ID: %s, Requests: %d, File: %s",
                    provider,
                    handle.batch_id,
                    len(part_requests),
                    part_temp_path,
                )

            # Write the documented batch-ID recovery artifact next to the temp
            # file(s) so check_batches/repair can recover ids if a tracking
            # record is ever lost.
            debug_path = (
                temp_jsonl_path.parent / f"{file_path.stem}_batch_submission_debug.json"
            )
            debug_path.write_text(
                json.dumps(
                    {"batch_ids": submitted_batch_ids, "provider": provider},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
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
