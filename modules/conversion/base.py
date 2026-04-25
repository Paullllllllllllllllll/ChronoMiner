# modules/conversion/base.py

"""
Base converter class for JSON data transformation.

Provides shared functionality for DocumentConverter (DOCX/TXT) and
CSVConverter, eliminating code duplication and ensuring consistent
entry extraction and filtering behavior.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Callable
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
    :param default: Value returned when the key is absent
    :return: Resolved value or *default*
    """
    if "." in key:
        outer, inner = key.split(".", 1)
        sub = entry.get(outer)
        if isinstance(sub, dict):
            return sub.get(inner, default)
        return default
    return entry.get(key, default)


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
        if entries is None:
            return []
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
    def format_name_variants(variants: Any) -> str:
        """
        Format name variants for display.
        
        :param variants: List of variant dictionaries with 'original' and 'modern_english' keys
        :return: Formatted string with variants
        """
        if not isinstance(variants, list):
            return ""
        formatted = []
        for variant in variants:
            if isinstance(variant, dict):
                original = variant.get("original") or ""
                modern = variant.get("modern_english")
                if modern and modern != original:
                    formatted.append(f"{original} ({modern})")
                else:
                    formatted.append(original)
        return "; ".join([f for f in formatted if f])
    
    @staticmethod
    def format_associations(assocs: Any, as_list: bool = False) -> Any:
        """
        Format associations for display.
        
        :param assocs: List of association dictionaries
        :param as_list: If True, return list of strings; if False, return joined string
        :return: Formatted associations as string or list
        """
        if not isinstance(assocs, list):
            return [] if as_list else ""
        formatted: list[str] = []
        for assoc in assocs:
            if not isinstance(assoc, dict):
                continue
            target_type = assoc.get("target_type")
            label = assoc.get("target_label_modern_english") or assoc.get("target_label_original")
            relationship = assoc.get("relationship")
            parts = [part for part in [target_type, label] if part]
            base = " - ".join(parts) if parts else ""
            if relationship:
                base = f"{base} ({relationship})" if base else relationship
            if base:
                formatted.append(base)
        return formatted if as_list else "; ".join(formatted)
    
    @staticmethod
    def _normalize_entries(entries: list[Any]) -> list[Any]:
        """Guard against a None list and filter out null elements."""
        if entries is None:
            return []
        return [e for e in entries if e is not None]

    @staticmethod
    def _extract_period(
        entry: dict[str, Any], key: str = "period"
    ) -> tuple:
        """Return (start_year, end_year, notation) from a nested period dict."""
        period = entry.get(key, {})
        if not isinstance(period, dict):
            return None, None, None
        return (
            period.get("start_year"),
            period.get("end_year"),
            period.get("notation"),
        )

    @staticmethod
    def _format_period(entry: dict[str, Any], key: str = "period") -> str:
        """Format a period dict as 'start - end (notation)' string."""
        period = entry.get(key, {})
        if not isinstance(period, dict) or not period:
            return ""
        period_str = (
            f"{period.get('start_year', 'Unknown')} - "
            f"{period.get('end_year', 'Unknown')}"
        )
        if period.get("notation"):
            period_str += f" ({period['notation']})"
        return period_str

    @staticmethod
    def _format_links(links: Any) -> str:
        """Format a links list as semicolon-separated descriptor strings."""
        if not isinstance(links, list):
            return ""
        return "; ".join(
            f"{l.get('entity_type', '')}: {l.get('entity_label', '')} - {l.get('relationship', '')}"
            for l in links
            if isinstance(l, dict)
        )

    @staticmethod
    def _format_officials(entry: dict) -> str:
        """Format officials list as 'position: signature' strings."""
        officials = entry.get("officials", [])
        if not officials:
            return ""
        return "; ".join(
            f"{o.get('position', '')}: {o.get('signature', '')}"
            for o in officials
        )

    @staticmethod
    def _extract_first_measurement(entry: dict[str, Any], key: str) -> str:
        """Extract first value/unit pair from a measurement list field."""
        items = entry.get(key, [])
        if items and isinstance(items, list) and len(items) > 0:
            item = items[0]
            if isinstance(item, dict):
                val = item.get("value_modern_english")
                unit = item.get("unit_modern_english")
                if val and unit:
                    return f"{val} {unit}"
        return ""

    def get_converter(
        self,
        converters: dict[str, Callable]
    ) -> Callable | None:
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
