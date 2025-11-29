# modules/core/converter_base.py

"""
Base converter class for JSON data transformation.

Provides shared functionality for DocumentConverter (DOCX/TXT) and
CSVConverter, eliminating code duplication and ensuring consistent
entry extraction and filtering behavior.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from modules.core.json_utils import extract_entries_from_json

logger = logging.getLogger(__name__)


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
    
    def get_entries(self, json_file: Path) -> List[Any]:
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
    
    def get_converter(
        self,
        converters: Dict[str, Callable]
    ) -> Optional[Callable]:
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
