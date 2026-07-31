# modules/extract/schema_handlers.py

import logging
from pathlib import Path

from modules.conversion.csv_converter import CSVConverter
from modules.conversion.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


class BaseSchemaHandler:
    """Base handler routing per-schema output conversion (CSV/DOCX/TXT)."""

    def __init__(self, schema_name: str) -> None:
        self.schema_name = schema_name

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
schema_handlers_registry: dict[str, BaseSchemaHandler] = {}


def register_schema_handler(schema_name: str, handler_class: type) -> None:
    """Register a schema handler class for a given schema name."""
    schema_handlers_registry[schema_name] = handler_class(schema_name)


def get_schema_handler(schema_name: str) -> BaseSchemaHandler:
    """Get the handler for a schema, defaulting to BaseSchemaHandler if not
    registered."""
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
    "CulinaryEntitiesEntries",
    "HistoricalRecipesEntriesProductionV3",
    "MichelinGuidesLight",
    "CookbookMetadataEntries",
    "HistoricalPriceEntries",
    "InequalityBenchmarks",
]:
    register_schema_handler(schema, BaseSchemaHandler)
