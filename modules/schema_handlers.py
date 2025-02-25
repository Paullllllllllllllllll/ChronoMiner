# modules/schema_handlers.py

import json
from pathlib import Path
from modules.data_processing import CSVConverter
from modules.text_processing import DocumentConverter


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
		instruction = "Please extract data from the following text according to the provided instructions.\n\n"
		text_chunk = instruction + text_chunk

		json_schema_payload = self.get_json_schema_payload(dev_message,
		                                                   model_config, schema)
		request_obj = {
			"custom_id": None,
			"method": "POST",
			"url": "/v1/chat/completions",
			"body": {
				"model": model_config["extraction_model"]["name"],
				"messages": [
					{"role": "system", "content": dev_message},
					{"role": "user", "content": text_chunk}
				],
				"max_completion_tokens": model_config["extraction_model"][
					"max_completion_tokens"],
				"reasoning_effort": model_config["extraction_model"][
					"reasoning_effort"],
				"response_format": {
					"type": "json_schema",
					"json_schema": json_schema_payload
				}
			}
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


# Registry for schema handlers
schema_handlers_registry = {}


def register_schema_handler(schema_name: str, handler_class):
	schema_handlers_registry[schema_name] = handler_class(schema_name)


def get_schema_handler(schema_name: str):
	return schema_handlers_registry.get(schema_name,
	                                    BaseSchemaHandler(schema_name))


# Register existing schema handlers with the default implementation.
for schema in ["BibliographicEntries", "Recipes", "StructuredSummaries",
               "HistoricalAddressBookEntries", "BrazilianMilitaryRecords"]:
	register_schema_handler(schema, BaseSchemaHandler)
