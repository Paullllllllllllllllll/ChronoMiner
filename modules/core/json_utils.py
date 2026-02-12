"""Shared utilities for JSON data extraction."""

import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from modules.core.path_utils import ensure_path_safe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers — keep parsing logic DRY across response formats
# ---------------------------------------------------------------------------

def _extract_text_from_api_body(body: Any) -> str:
    """
    Extract the model's text content from a raw API response body.

    Supports:
    - Chat Completions API: ``body.choices[0].message.content``
    - Responses API: ``body.output[*].content[*].text``
    - ``output_text`` shorthand (Responses API normalised form)

    :param body: Parsed API response body (dict expected)
    :return: Extracted text, or empty string if nothing found
    """
    if not isinstance(body, dict):
        return ""

    # Responses API normalised shorthand
    if isinstance(body.get("output_text"), str):
        return body["output_text"]

    # Chat Completions API
    if "choices" in body:
        choices = body.get("choices", [])
        if choices:
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            return message.get("content", "")

    # Responses API (nested output → message → content → text)
    output = body.get("output")
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for content_part in item.get("content", []):
                    if isinstance(content_part, dict):
                        text_val = content_part.get("text")
                        if isinstance(text_val, str):
                            parts.append(text_val)
        return "".join(parts)

    return ""


def _parse_entries_from_text(text: str) -> Optional[List[Any]]:
    """
    Parse a JSON string and return the ``entries`` list it contains.

    Returns ``None`` when the text cannot be parsed, has no ``entries``
    key, or signals ``contains_no_content_of_requested_type``.

    :param text: Raw JSON string produced by the model
    :return: List of entry dicts, or *None*
    """
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse content as JSON: {text[:100]}...")
        return None

    if not isinstance(parsed, dict):
        return None
    if parsed.get("contains_no_content_of_requested_type", False):
        logger.debug("Response indicated no content of requested type")
        return None
    if "entries" not in parsed:
        return None
    return [e for e in parsed["entries"] if e is not None]


def _extract_entries_from_record(record: Any) -> List[Any]:
    """
    Extract entries from a single response record.

    A record is a dict that typically carries a ``response`` field whose
    value is either:
    - a JSON string (already-serialised model output),
    - an API response body dict (Chat Completions / Responses API), or
    - a dict that directly contains ``entries``.

    :param record: Single record / chunk dict
    :return: List of extracted entries (may be empty)
    """
    if not isinstance(record, dict):
        return []

    response = record.get("response")
    if response is None:
        return []

    # Case 1: response is already a JSON string
    if isinstance(response, str):
        return _parse_entries_from_text(response) or []

    if not isinstance(response, dict):
        return []

    # Case 2: response dict directly contains entries
    if response.get("contains_no_content_of_requested_type", False):
        logger.debug("Record response indicated no content of requested type")
        return []
    if "entries" in response:
        return [e for e in response["entries"] if e is not None]

    # Case 3: response is a raw API body — extract text, then parse
    text = _extract_text_from_api_body(response)
    if text:
        return _parse_entries_from_text(text) or []

    # Case 4: response is itself a list of entries
    if isinstance(response, list):
        return response

    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entries_from_json(json_file: Path) -> List[Any]:
    """
    Extract entries from a JSON file, handling various response formats.

    This function supports multiple JSON structures:
    - Direct entries: ``{"entries": [...]}``
    - Records format: ``{"records": [...]}`` (from process_text_files.py)
    - Batch responses: ``{"responses": [...]}``
    - Chat Completions API format
    - Responses API format
    - Chunk-based output format (top-level list)

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

        # Records format (from process_text_files.py / _generate_output_files)
        elif "records" in data:
            for record in data["records"]:
                entries.extend(_extract_entries_from_record(record))

        # Batch responses format (from check_batches.py)
        elif "responses" in data:
            for resp in data.get("responses", []):
                if resp is None:
                    continue

                try:
                    if isinstance(resp, str):
                        parsed = _parse_entries_from_text(resp)
                        if parsed is not None:
                            entries.extend(parsed)
                    elif isinstance(resp, dict):
                        # check_batches.py normalises responses with raw_response / body
                        body = resp.get("raw_response") if "raw_response" in resp else resp.get("body", {})
                        if not isinstance(body, dict):
                            body = {}
                        content = _extract_text_from_api_body(body)
                        if content:
                            parsed = _parse_entries_from_text(content)
                            if parsed is not None:
                                entries.extend(parsed)
                except Exception as e:
                    logger.warning(f"Error processing response: {e}")
                    continue

    elif isinstance(data, list):
        # Chunk-based output format (top-level list of records)
        for chunk in data:
            entries.extend(_extract_entries_from_record(chunk))

        # Direct list of entries (fallback)
        if not entries:
            entries = data

    # Filter out any None entries from the final list
    entries = [entry for entry in entries if entry is not None]

    return entries
