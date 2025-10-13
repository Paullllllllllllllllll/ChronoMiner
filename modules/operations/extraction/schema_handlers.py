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

	def prepare_payload(self, text_chunk: str, dev_message: str,
	                    model_config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
		"""Prepare API request payload using PayloadBuilder."""
		return self.payload_builder.build_payload(
			text_chunk, dev_message, model_config, schema
		)

	def process_response(self, response_str: str) -> Dict[str, Any]:
		"""Parse response string using ResponseParser."""
		return self.response_parser.parse_response(response_str)

	def convert_to_csv(self, json_file: Path, output_csv: Path) -> None:
		csv_converter = CSVConverter(self.schema_name)
		csv_converter.convert_to_csv(json_file, output_csv)

	def convert_to_docx(self, json_file: Path, output_docx: Path) -> None:
		doc_converter = DocumentConverter(self.schema_name)
		doc_converter.convert_to_docx(json_file, output_docx)

	def convert_to_txt(self, json_file: Path, output_txt: Path) -> None:
		doc_converter = DocumentConverter(self.schema_name)
		doc_converter.convert_to_txt(json_file, output_txt)

	def convert_to_csv_safely(self, json_file: Path, output_csv: Path) -> None:
		"""
		Safely convert JSON to CSV with error handling.

		:param json_file: Input JSON file path
		:param output_csv: Output CSV file path
		"""
		try:
			csv_converter = CSVConverter(self.schema_name)
			csv_converter.convert_to_csv(json_file, output_csv)
			print(f"CSV file generated at {output_csv}")
		except Exception as e:
			print(f"Error converting to CSV: {e}")
			logger.error(f"Error converting {json_file} to CSV: {e}")

	def convert_to_docx_safely(self, json_file: Path,
	                           output_docx: Path) -> None:
		"""
		Safely convert JSON to DOCX with error handling.

		:param json_file: Input JSON file path
		:param output_docx: Output DOCX file path
		"""
		try:
			doc_converter = DocumentConverter(self.schema_name)
			doc_converter.convert_to_docx(json_file, output_docx)
			print(f"DOCX file generated at {output_docx}")
		except Exception as e:
			print(f"Error converting to DOCX: {e}")
			logger.error(f"Error converting {json_file} to DOCX: {e}")

	def convert_to_txt_safely(self, json_file: Path, output_txt: Path) -> None:
		"""
		Safely convert JSON to TXT with error handling.

		:param json_file: Input JSON file path
		:param output_txt: Output TXT file path
		"""
		try:
			doc_converter = DocumentConverter(self.schema_name)
			doc_converter.convert_to_txt(json_file, output_txt)
			print(f"TXT file generated at {output_txt}")
		except Exception as e:
			print(f"Error converting to TXT: {e}")
			logger.error(f"Error converting {json_file} to TXT: {e}")


# Registry for schema handlers
schema_handlers_registry = {}


def register_schema_handler(schema_name: str, handler_class):
	schema_handlers_registry[schema_name] = handler_class(schema_name)


def get_schema_handler(schema_name: str):
	return schema_handlers_registry.get(schema_name,
	                                    BaseSchemaHandler(schema_name))


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
	"MilitaryRecordEntries"
]:
	register_schema_handler(schema, BaseSchemaHandler)
