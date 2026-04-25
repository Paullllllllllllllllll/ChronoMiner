"""ChronoMiner output conversion package.

Given intermediate JSON output files from the extraction pipeline, produce
CSV, DOCX, or TXT renderings via schema-specific converters. Also owns the
canonical JSON-response parsing logic used by the LLM layer's response
handlers.
"""

from modules.conversion.base import BaseConverter, resolve_field
from modules.conversion.csv_converter import CSVConverter
from modules.conversion.document_converter import DocumentConverter
from modules.conversion.json_utils import (
    extract_entries_from_json,
    parse_json_from_text,
    parse_llm_response_text,
)

__all__ = [
    "BaseConverter",
    "CSVConverter",
    "DocumentConverter",
    "extract_entries_from_json",
    "parse_json_from_text",
    "parse_llm_response_text",
    "resolve_field",
]
