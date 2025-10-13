# modules/core/processing_strategy.py

"""
Processing strategy abstraction for synchronous vs batch execution.
Separates execution logic from file processing orchestration.
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

from modules.core.token_tracker import get_token_tracker
from modules.llm.batching import build_batch_files, submit_batch
from modules.llm.openai_utils import open_extractor, process_text_chunk

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
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = "OPENAI_API_KEY is not set in environment variables."
            logger.error(error_msg)
            console_print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        console_print(f"[INFO] Starting synchronous processing of {len(chunks)} chunks...")
        results: List[Dict[str, Any]] = []

        # Extract concurrency settings
        transcription_cfg = (
            (self.concurrency_config.get("concurrency", {}) or {}).get("transcription", {}) or {}
        )
        total_chunks = len(chunks)
        try:
            configured_limit = int(transcription_cfg.get("concurrency_limit", total_chunks or 1))
        except Exception:
            configured_limit = total_chunks or 1
        concurrency_limit = max(1, min(configured_limit, total_chunks or 1))
        delay_between_tasks = float(transcription_cfg.get("delay_between_tasks", 0.0) or 0.0)

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
    """Batch (deferred) processing strategy."""

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
        """Prepare and submit batch processing job."""
        console_print(f"[INFO] Preparing batch processing for {len(chunks)} chunks...")
        
        request_lines: List[str] = []
        try:
            for idx, chunk in enumerate(chunks, 1):
                request_obj = handler.prepare_payload(
                    chunk,
                    dev_message,
                    model_config,
                    schema,
                )
                request_obj["custom_id"] = f"{file_path.stem}-chunk-{idx}"
                request_lines.append(json.dumps(request_obj))

            batch_files = build_batch_files(request_lines, temp_jsonl_path)
            if not batch_files:
                error_msg = f"No batch files were generated for {file_path.name}."
                console_print(f"[ERROR] {error_msg}")
                raise RuntimeError(error_msg)

            logger.info(
                "Created %s batch request(s) across %s file(s) for %s",
                len(request_lines),
                len(batch_files),
                file_path.name
            )
            console_print(
                f"[INFO] Created {len(request_lines)} batch requests split into {len(batch_files)} file(s)."
            )
        except Exception as e:
            logger.error(f"Error preparing batch requests for {file_path.name}: {e}")
            console_print(f"[ERROR] Failed to prepare batch requests: {e}")
            raise

        submitted_batches: List[str] = []
        for batch_file in batch_files:
            try:
                console_print(f"[INFO] Submitting batch file {batch_file.name}...")
                batch_response = submit_batch(batch_file)
                tracking_record = {
                    "batch_tracking": {
                        "batch_id": batch_response.id,
                        "timestamp": batch_response.created_at,
                        "batch_file": str(batch_file)
                    }
                }
                with batch_file.open("a", encoding="utf-8") as tempf:
                    tempf.write(json.dumps(tracking_record) + "\n")
                submitted_batches.append(batch_response.id)
                console_print(
                    f"[SUCCESS] Batch submitted successfully. Batch ID: {batch_response.id}"
                )
                logger.info(
                    f"Batch submitted successfully. Tracking record appended to %s",
                    batch_file
                )
            except Exception as e:
                logger.error(f"Error during batch submission for file {batch_file}: {e}")
                console_print(f"[ERROR] Failed to submit batch file {batch_file.name}: {e}")
                raise

        logger.info(
            "Submitted %s batch file(s) for %s: %s",
            len(submitted_batches),
            file_path.name,
            submitted_batches
        )
        console_print(
            f"[SUCCESS] Submitted {len(submitted_batches)} batch(es). "
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
