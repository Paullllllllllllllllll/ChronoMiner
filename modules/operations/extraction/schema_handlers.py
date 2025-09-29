# modules/schema_handlers.py

import json
import logging
from pathlib import Path
from typing import Optional
from modules.core.data_processing import CSVConverter
from modules.core.text_processing import DocumentConverter
from modules.llm.structured_outputs import build_structured_text_format

logger = logging.getLogger(__name__)


class BaseSchemaHandler:
	def __init__(self, schema_name: str):
		self.schema_name = schema_name

	def get_json_schema_payload(self, dev_message: str, model_config: dict,
	                            schema: dict) -> dict:
		return {
			"name": self.schema_name,
			"schema": schema,
			"strict": True
		}

	def prepare_payload(self, text_chunk: str, dev_message: str,
	                    model_config: dict, schema: dict) -> dict:
		user_message = f"Input text:\n{text_chunk}"

		json_schema_payload = self.get_json_schema_payload(dev_message,
		                                                   model_config, schema)
		# Build typed input and structured outputs for Responses API
		model_cfg = model_config.get("transcription_model", {})
		fmt = build_structured_text_format(json_schema_payload, self.schema_name, True)
		body = {
			"model": model_cfg.get("name"),
			"max_output_tokens": model_cfg.get("max_output_tokens", 4096),
			"input": [
				{
					"role": "system",
					"content": [{"type": "input_text", "text": dev_message}]
				},
				{
					"role": "user",
					"content": [{"type": "input_text", "text": user_message}]
				}
			]
		}
		if fmt is not None:
			body.setdefault("text", {})["format"] = fmt
		# Optional classic sampler controls (safe for non-reasoning families)
		for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
			if k in model_cfg and model_cfg[k] is not None:
				body[k] = model_cfg[k]

		request_obj = {
			"custom_id": None,
			"method": "POST",
			"url": "/v1/responses",
			"body": body,
		}
		return request_obj

	def process_response(self, response_str: str) -> dict:
		try:
			return json.loads(response_str)
		except Exception as e:
			return {"error": str(e)}

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
for schema in ["BibliographicEntries", "StructuredSummaries",
               "HistoricalAddressBookEntries", "BrazilianMilitaryRecords"]:
	register_schema_handler(schema, BaseSchemaHandler)
