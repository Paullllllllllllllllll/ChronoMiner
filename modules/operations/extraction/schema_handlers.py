# modules/operations/extraction/schema_handlers.py

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from modules.core.data_processing import CSVConverter
from modules.core.text_processing import DocumentConverter
from modules.operations.extraction.payload_builder import PayloadBuilder
from modules.operations.extraction.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class BaseSchemaHandler:
    """Base handler for schema-based extraction with modular components."""

    def __init__(self, schema_name: str):
        self.schema_name = schema_name
        self.payload_builder = PayloadBuilder(schema_name)
        self.response_parser = ResponseParser(schema_name)

    def prepare_payload(
        self,
        text_chunk: str,
        dev_message: str,
        model_config: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare API request payload using PayloadBuilder."""
        return self.payload_builder.build_payload(
            text_chunk, dev_message, model_config, schema
        )

    def process_response(self, response_str: str) -> Dict[str, Any]:
        """Parse response string using ResponseParser."""
        return self.response_parser.parse_response(response_str)

    def convert_to_csv(self, json_file: Path, output_csv: Path) -> None:
        """Convert JSON to CSV format."""
        csv_converter = CSVConverter(self.schema_name)
        csv_converter.convert_to_csv(json_file, output_csv)

    def convert_to_docx(self, json_file: Path, output_docx: Path) -> None:
        """Convert JSON to DOCX format."""
        doc_converter = DocumentConverter(self.schema_name)
        doc_converter.convert_to_docx(json_file, output_docx)

    def convert_to_txt(self, json_file: Path, output_txt: Path) -> None:
        """Convert JSON to TXT format."""
        doc_converter = DocumentConverter(self.schema_name)
        doc_converter.convert_to_txt(json_file, output_txt)


# Registry for schema handlers
schema_handlers_registry = {}


def register_schema_handler(schema_name: str, handler_class) -> None:
    """Register a schema handler class for a given schema name."""
    schema_handlers_registry[schema_name] = handler_class(schema_name)


def get_schema_handler(schema_name: str) -> BaseSchemaHandler:
    """Get the handler for a schema, defaulting to BaseSchemaHandler if not registered."""
    return schema_handlers_registry.get(schema_name, BaseSchemaHandler(schema_name))


# Register existing schema handlers with the default implementation.
for schema in [
    "BibliographicEntries",
    "StructuredSummaries",
    "HistoricalAddressBookEntries",
    "BrazilianMilitaryRecords",
    "CulinaryPersonsEntries",
    "CulinaryPlacesEntries",
    "CulinaryWorksEntries",
    "HistoricalRecipesEntries",
    "MilitaryRecordEntries",
]:
    register_schema_handler(schema, BaseSchemaHandler)
