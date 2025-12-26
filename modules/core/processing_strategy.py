# modules/core/processing_strategy.py

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
import json
import logging
import os
import time
import random
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from modules.core.token_tracker import get_token_tracker
from modules.llm.batching import build_batch_files, submit_batch
from modules.llm.openai_utils import open_extractor, process_text_chunk
from modules.llm.langchain_provider import ProviderConfig
from modules.llm.batch import (
    BatchRequest,
    BatchHandle,
    get_batch_backend,
    supports_batch,
)

logger = logging.getLogger(__name__)


class ProcessingStrategy(ABC):
    """Abstract base class for processing strategies."""

    @abstractmethod
    async def process_chunks(
        self,
        chunks: List[str],
        handler,
        dev_message: str,
        model_config: Dict[str, Any],
        schema: Dict[str, Any],
        file_path: Path,
        temp_jsonl_path: Path,
        console_print,
    ) -> List[Dict[str, Any]]:
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
        :return: List of processing results
        """
        pass


class SynchronousProcessingStrategy(ProcessingStrategy):
    """Synchronous (real-time) processing strategy."""

    def __init__(self, concurrency_config: Optional[Dict[str, Any]] = None):
        """
        Initialize synchronous processing strategy.

        :param concurrency_config: Concurrency configuration
        """
        self.concurrency_config = concurrency_config or {}

    async def process_chunks(
        self,
        chunks: List[str],
        handler,
        dev_message: str,
        model_config: Dict[str, Any],
        schema: Dict[str, Any],
        file_path: Path,
        temp_jsonl_path: Path,
        console_print,
    ) -> List[Dict[str, Any]]:
        """Process chunks synchronously with concurrent API calls."""
        # Detect provider from model name
        model_name = model_config["transcription_model"]["name"]
        provider = ProviderConfig._detect_provider(model_name)
        api_key = ProviderConfig._get_api_key(provider)
        
        if not api_key:
            error_msg = f"API key not found for provider {provider}. Set the appropriate environment variable."
            logger.error(error_msg)
            console_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        console_print(f"[INFO] Starting synchronous processing of {len(chunks)} chunks...")
        results: List[Dict[str, Any]] = []

        # Extract concurrency settings
        extraction_cfg = (
            (self.concurrency_config.get("concurrency", {}) or {}).get("extraction", {}) or {}
        )
        retry_cfg = (extraction_cfg.get("retry", {}) or {})
        total_chunks = len(chunks)
        try:
            configured_limit = int(extraction_cfg.get("concurrency_limit", total_chunks or 1))
        except Exception:
            configured_limit = total_chunks or 1
        concurrency_limit = max(1, min(configured_limit, total_chunks or 1))
        delay_between_tasks = float(extraction_cfg.get("delay_between_tasks", 0.0) or 0.0)

        try:
            retry_attempts = int(retry_cfg.get("attempts", 1))
        except Exception:
            retry_attempts = 1
        retry_attempts = max(1, retry_attempts)
        wait_min_seconds = float(retry_cfg.get("wait_min_seconds", 1.0) or 1.0)
        wait_max_seconds = float(retry_cfg.get("wait_max_seconds", 60.0) or 60.0)
        jitter_max_seconds = float(retry_cfg.get("jitter_max_seconds", 0.0) or 0.0)

        if provider == "anthropic":
            concurrency_limit = 1

        # Token tracking
        token_tracker = get_token_tracker()

        async with open_extractor(
            api_key=api_key,
            prompt_path=Path("prompts/structured_output_prompt.txt"),
            model=model_config["transcription_model"]["name"]
        ) as extractor:
            with temp_jsonl_path.open("w", encoding="utf-8") as tempf:
                semaphore = asyncio.Semaphore(concurrency_limit)

                async def process_single_chunk(idx: int, chunk: str):
                    """Process a single chunk with semaphore control."""
                    async with semaphore:
                        if delay_between_tasks > 0:
                            await asyncio.sleep(delay_between_tasks)
                        
                        for attempt in range(retry_attempts):
                            try:
                                result = await process_text_chunk(
                                    text_chunk=chunk,
                                    extractor=extractor,
                                    system_message=dev_message,
                                    json_schema=schema
                                )

                                # Track tokens if enabled
                                if token_tracker.enabled:
                                    usage = result.get("usage", {})
                                    input_tokens = usage.get("input_tokens", 0)
                                    output_tokens = usage.get("output_tokens", 0)
                                    token_tracker.add_tokens(input_tokens + output_tokens)

                                # Write to temp file
                                request_obj = handler.prepare_payload(
                                    chunk, dev_message, model_config, schema
                                )
                                request_obj["custom_id"] = f"{file_path.stem}-chunk-{idx}"

                                response_obj = {
                                    "custom_id": request_obj["custom_id"],
                                    "response": {
                                        "body": result
                                    }
                                }
                                tempf.write(json.dumps(response_obj) + "\n")
                                tempf.flush()

                                console_print(f"[INFO] Processed chunk {idx}/{total_chunks}")
                                return result
                            except Exception as e:
                                msg = str(e)
                                is_429 = "429" in msg or "rate_limit" in msg.lower()
                                if provider == "anthropic" and is_429 and attempt < (retry_attempts - 1):
                                    base_wait = min(wait_max_seconds, wait_min_seconds * (2 ** attempt))
                                    jitter = random.uniform(0.0, jitter_max_seconds) if jitter_max_seconds > 0 else 0.0
                                    wait_s = min(wait_max_seconds, base_wait + jitter)
                                    logger.warning(
                                        "Rate-limited on chunk %s (attempt %s/%s). Waiting %.1fs and retrying.",
                                        idx,
                                        attempt + 1,
                                        retry_attempts,
                                        wait_s,
                                    )
                                    await asyncio.sleep(wait_s)
                                    continue

                                logger.error(f"Error processing chunk {idx}: {e}", exc_info=e)
                                console_print(f"[ERROR] Failed to process chunk {idx}: {e}")
                                return {"error": str(e)}

                # Process all chunks concurrently
                tasks = [
                    process_single_chunk(idx, chunk)
                    for idx, chunk in enumerate(chunks, 1)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=False)

        console_print(f"[SUCCESS] Completed synchronous processing of {len(chunks)} chunks")
        return results


class BatchProcessingStrategy(ProcessingStrategy):
    """Batch (deferred) processing strategy.
    
    Supports multiple providers:
    - OpenAI: Uses OpenAI Batch API with /v1/responses endpoint
    - Anthropic: Uses Anthropic Message Batches API
    - Google: Uses Google Gemini Batch API
    - OpenRouter: Not supported (falls back to sync or raises error)
    """

    async def process_chunks(
        self,
        chunks: List[str],
        handler,
        dev_message: str,
        model_config: Dict[str, Any],
        schema: Dict[str, Any],
        file_path: Path,
        temp_jsonl_path: Path,
        console_print,
    ) -> List[Dict[str, Any]]:
        """Prepare and submit batch processing job using provider-agnostic backend."""
        # Detect provider from model config
        tm = model_config.get("transcription_model", {})
        provider = tm.get("provider")
        if not provider:
            model_name = tm.get("name", "")
            provider = ProviderConfig._detect_provider(model_name)
        
        console_print(f"[INFO] Preparing batch processing for {len(chunks)} chunks (provider: {provider})...")
        
        # Check if provider supports batch processing
        if not supports_batch(provider):
            error_msg = (
                f"Provider '{provider}' does not support batch processing. "
                f"Use synchronous mode or switch to openai, anthropic, or google."
            )
            console_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)
        
        # Build BatchRequest objects for the provider-agnostic backend
        batch_requests: List[BatchRequest] = []
        for idx, chunk in enumerate(chunks, 1):
            custom_id = f"{file_path.stem}-chunk-{idx}"
            batch_requests.append(BatchRequest(
                custom_id=custom_id,
                text=chunk,
                order_index=idx,
                metadata={
                    "file_path": str(file_path),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
            ))
        
        # Get the appropriate batch backend
        try:
            backend = get_batch_backend(provider)
        except ValueError as e:
            console_print(f"[ERROR] {e}")
            raise
        
        # Extract schema name from handler if available
        schema_name = getattr(handler, "schema_name", None) or "ExtractionSchema"
        
        # Submit batch using the provider-agnostic backend
        try:
            console_print(f"[INFO] Submitting batch to {provider}...")
            handle: BatchHandle = await asyncio.to_thread(
                backend.submit_batch,
                batch_requests,
                model_config,
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
                temp_jsonl_path
            )
            
        except Exception as e:
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
    use_batch: bool,
    concurrency_config: Optional[Dict[str, Any]] = None
) -> ProcessingStrategy:
    """
    Factory function to create appropriate processing strategy.

    :param use_batch: Whether to use batch processing
    :param concurrency_config: Concurrency configuration
    :return: ProcessingStrategy instance
    """
    if use_batch:
        return BatchProcessingStrategy()
    else:
        return SynchronousProcessingStrategy(concurrency_config)
