"""Shared utilities for JSON data extraction."""

import json
import logging
from pathlib import Path
from typing import Any, List

from modules.core.path_utils import ensure_path_safe

logger = logging.getLogger(__name__)


def extract_entries_from_json(json_file: Path) -> List[Any]:
    """
    Extract entries from a JSON file, handling various response formats.

    This function supports multiple JSON structures:
    - Direct entries: {"entries": [...]}
    - Batch responses: {"responses": [...]}
    - Chat Completions API format
    - Responses API format
    - Chunk-based output format

    :param json_file: Path to the JSON file
    :return: List of entries extracted from the JSON file
    """
    try:
        safe_json_file = ensure_path_safe(json_file)
        with safe_json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {json_file}: {e}")
        return []

    entries: List[Any] = []

    if isinstance(data, dict):
        # Check if the model indicated no content of requested type
        if data.get("contains_no_content_of_requested_type", False):
            logger.info(f"Model indicated no content of requested type in {json_file.name}")
            return []
        
        # Direct entries format
        if "entries" in data:
            entries = data["entries"]
        # Batch responses format
        elif "responses" in data:
            for resp in data.get("responses", []):
                if resp is None:
                    continue  # Skip None responses

                try:
                    if isinstance(resp, str):
                        # Try to parse response string as JSON
                        content_json = json.loads(resp)
                        if isinstance(content_json, dict):
                            # Check if this response has no content of requested type
                            if content_json.get("contains_no_content_of_requested_type", False):
                                logger.debug("Response indicated no content of requested type")
                                continue
                            
                            if "entries" in content_json:
                                # Filter out None entries
                                valid_entries = [
                                    entry for entry in content_json.get("entries", [])
                                    if entry is not None
                                ]
                                entries.extend(valid_entries)
                    elif isinstance(resp, dict):
                        # Handle batch response structure (Chat Completions and Responses API)
                        # First check for raw_response (normalized format from check_batches.py)
                        body = resp.get("raw_response") if "raw_response" in resp else resp.get("body", {})
                        if not isinstance(body, dict):
                            body = {}
                        content = ""

                        # Extract content from different API response formats
                        if "choices" in body:
                            # Chat Completions API format
                            choices = body.get("choices", [])
                            if choices and len(choices) > 0:
                                message = choices[0].get("message", {})
                                content = message.get("content", "")
                        elif "output" in body:
                            # Responses API format
                            output = body.get("output")
                            if isinstance(output, list):
                                for item in output:
                                    if isinstance(item, dict) and item.get("type") == "message":
                                        for content_part in item.get("content", []):
                                            if isinstance(content_part, dict):
                                                text_val = content_part.get("text")
                                                if isinstance(text_val, str):
                                                    content += text_val

                        # Parse the extracted content
                        if content:
                            try:
                                content_json = json.loads(content)
                                if isinstance(content_json, dict):
                                    # Check if this response has no content of requested type
                                    if content_json.get("contains_no_content_of_requested_type", False):
                                        logger.debug("Response indicated no content of requested type")
                                        continue
                                    
                                    if "entries" in content_json:
                                        valid_entries = [
                                            entry for entry in content_json.get("entries", [])
                                            if entry is not None
                                        ]
                                        entries.extend(valid_entries)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse content as JSON: {content[:100]}...")
                except Exception as e:
                    logger.warning(f"Error processing response: {e}")
                    continue

    elif isinstance(data, list):
        # Chunk-based output format
        for chunk in data:
            if isinstance(chunk, dict) and "response" in chunk:
                chunk_data = chunk["response"]
                if isinstance(chunk_data, dict):
                    # Check if chunk response has no content of requested type
                    if chunk_data.get("contains_no_content_of_requested_type", False):
                        logger.debug("Chunk response indicated no content of requested type")
                        continue
                    
                    if "entries" in chunk_data:
                        entries.extend(chunk_data["entries"])
                elif isinstance(chunk_data, list):
                    entries.extend(chunk_data)

        # Direct list of entries
        if not entries:
            entries = data

    # Filter out any None entries from the final list
    entries = [entry for entry in entries if entry is not None]

    return entries
