# modules/conversion/base.py

"""
Base converter class for JSON data transformation.

Provides shared functionality for DocumentConverter (DOCX/TXT) and
CSVConverter, eliminating code duplication and ensuring consistent
entry extraction and filtering behavior.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from modules.conversion.json_utils import extract_entries_from_json

logger = logging.getLogger(__name__)


def resolve_field(entry: dict, key: str, default: Any = "") -> Any:
    """
    Resolve a possibly-dotted key from *entry*.

    Supports one level of nesting, e.g. ``"address.street"`` looks up
    ``entry["address"]["street"]``.

    :param entry: Source dictionary
    :param key: Flat or dotted key
    :param default: Value returned when the key is absent or explicitly null
    :return: Resolved value or *default*
    """
    if "." in key:
        outer, inner = key.split(".", 1)
        sub = entry.get(outer)
        if isinstance(sub, dict):
            value = sub.get(inner, default)
            return default if value is None else value
        return default
    value = entry.get(key, default)
    return default if value is None else value


class BaseConverter(ABC):
    """
    Abstract base class for data format converters.

    Provides shared functionality:
    - Schema name normalization
    - Entry extraction from JSON files
    - Entry filtering (removes None values)
    - Safe string conversion
    - Converter registry pattern
    """

    def __init__(self, schema_name: str) -> None:
        """
        Initialize the converter with a schema name.

        :param schema_name: Name of the schema (case-insensitive)
        """
        self.schema_name: str = schema_name.lower()

    def get_entries(self, json_file: Path) -> list[Any]:
        """
        Extract and filter entries from a JSON file.

        Uses extract_entries_from_json utility and filters out None values.

        :param json_file: Path to the JSON file
        :return: List of non-None entries
        """
        entries = extract_entries_from_json(json_file)
        return [entry for entry in entries if entry is not None]

    @staticmethod
    def safe_str(value: Any) -> str:
        """
        Safely convert a value to string, handling None values.

        :param value: Any value that might be None
        :return: String representation or empty string if None
        """
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def join_list(values: Any, separator: str = ", ") -> str:
        """
        Join list values into a string, filtering None and empty values.

        :param values: List of values or non-list value
        :param separator: Separator string (default: ", ")
        :return: Joined string or empty string if not a list
        """
        if isinstance(values, list):
            items = [str(v) for v in values if v not in (None, "")]
            return separator.join(items)
        return ""

    @staticmethod
    def _normalize_entries(entries: list[Any]) -> list[Any]:
        """Filter out null elements."""
        return [e for e in entries if e is not None]

    @staticmethod
    def _extract_period(entry: dict[str, Any], key: str = "timeframe") -> tuple:
        """Return (start_year, end_year, notation) from a nested timeframe dict."""
        period = entry.get(key, {})
        if not isinstance(period, dict):
            return None, None, None
        return (
            period.get("start_year"),
            period.get("end_year"),
            period.get("notation"),
        )

    @staticmethod
    def _format_period(entry: dict[str, Any], key: str = "timeframe") -> str:
        """Format a timeframe dict as 'start - end (notation)' string."""
        period = entry.get(key, {})
        if not isinstance(period, dict) or not period:
            return ""
        period_str = (
            f"{period.get('start_year') or 'Unknown'} - "
            f"{period.get('end_year') or 'Unknown'}"
        )
        if period.get("notation"):
            period_str += f" ({period['notation']})"
        return period_str

    @staticmethod
    def _format_links(links: Any) -> str:
        """Format an association list as semicolon-separated descriptor strings.

        Each item is rendered as ``entity_type: label - relationship`` where
        *label* prefers ``entity_label_modern`` and falls back to
        ``entity_label_original`` (schema v3.0 association shape). Missing
        parts are omitted rather than rendered as empty separators; an
        all-null association contributes nothing.
        """
        if not isinstance(links, list):
            return ""
        formatted: list[str] = []
        for link in links:
            if not isinstance(link, dict):
                continue
            label = link.get("entity_label_modern") or link.get("entity_label_original")
            etype = link.get("entity_type")
            rel = link.get("relationship")
            head = f"{etype}: {label}" if etype and label else (etype or label or "")
            text = f"{head} - {rel}" if head and rel else (head or rel or "")
            if text:
                formatted.append(str(text))
        return "; ".join(formatted)

    @staticmethod
    def _format_officials(entry: dict) -> str:
        """Format officials list as 'position: signature' strings.

        A null position or signature is omitted together with its separator,
        so an official with only one of the two renders as that value alone.
        """
        officials = entry.get("officials", [])
        if not officials:
            return ""
        formatted: list[str] = []
        for official in officials:
            position = official.get("position") or ""
            signature = official.get("signature") or ""
            text = (
                f"{position}: {signature}"
                if position and signature
                else (position or signature)
            )
            if text:
                formatted.append(str(text))
        return "; ".join(formatted)

    def get_converter(self, converters: dict[str, Callable]) -> Callable | None:
        """
        Get the appropriate converter function for the current schema.

        :param converters: Dictionary mapping schema names to converter functions
        :return: Converter function or None if not found
        """
        return converters.get(self.schema_name.lower())

    @abstractmethod
    def convert(self, json_file: Path, output_file: Path) -> None:
        """
        Convert JSON data to the target format.

        :param json_file: Input JSON file path
        :param output_file: Output file path
        """
        pass
