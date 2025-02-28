# modules/file_processor.py

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from modules.text_utils import TextProcessor, perform_chunking
from modules.openai_utils import open_extractor, process_text_chunk
from modules.schema_handlers import get_schema_handler
import logging

logger = logging.getLogger(__name__)


class FileProcessor:
	"""
	Handles all file processing operations, including:
	- Text normalization
	- Chunking
	- API request construction
	- Response handling
	- Output file generation
	"""

	def __init__(self, paths_config: Dict[str, Any],
	             model_config: Dict[str, Any],
	             chunking_config: Dict[str, Any]):
		"""
		Initialize with configuration
		"""
		self.paths_config = paths_config
		self.model_config = model_config
		self.chunking_config = chunking_config
		self.text_processor = TextProcessor()

	async def process_file(self, file_path: Path,
	                       use_batch: bool,
	                       selected_schema: Dict[str, Any],
	                       dev_message: str,
	                       schema_paths: Dict[str, Any],
	                       global_chunking_method: Optional[str] = None,
	                       context_settings: Dict[str, Any] = None,
	                       context_manager: Optional[Any] = None,
	                       ui=None) -> None:
		"""
		Process a single text file:
		- Read and normalize text
		- Determine chunking strategy and split text
		- Construct API requests based on schema
		- Process responses either synchronously or via batch
		- Write final output and convert to additional formats

		:param file_path: Path to the file to process
		:param use_batch: Whether to use batch processing
		:param selected_schema: The selected schema dictionary
		:param dev_message: Developer message for the schema
		:param schema_paths: Schema-specific paths
		:param global_chunking_method: Global chunking method if specified
		:param context_settings: Additional context settings
		:param context_manager: Context manager instance
		:param ui: UserInterface instance for user feedback
		"""
		# Use provided UI or print directly
		console_print = ui.console_print if ui else print
		ask_file_chunking_method = ui.ask_file_chunking_method if ui else self._default_ask_file_chunking_method

		console_print(f"\n[INFO] Processing file: {file_path.name}")
		logger.info(f"Starting processing for file: {file_path}")

		# -- Read and Normalize Text --
		try:
			encoding: str = TextProcessor.detect_encoding(file_path)
			with file_path.open("r", encoding=encoding) as f:
				lines: List[str] = f.readlines()
			normalized_lines: List[str] = [TextProcessor.normalize_text(line)
			                               for line in lines]
			console_print(
				f"[INFO] Successfully read and normalized {len(lines)} lines from {file_path.name}")
		except Exception as e:
			logger.error(f"Error reading file {file_path}: {e}")
			console_print(f"[ERROR] Failed to read file {file_path.name}: {e}")
			return

		# -- Determine Chunking Strategy --
		if global_chunking_method == "per-file":
			global_chunking_method = None  # Reset to force per-file selection

		if global_chunking_method is not None:
			chosen_method = global_chunking_method
			console_print(
				f"[INFO] Using global chunking method '{chosen_method}' for file {file_path.name}")
		else:
			chosen_method = ask_file_chunking_method(file_path.name)

		if chosen_method == "auto":
			chunk_choice: str = "auto"
			line_ranges_file: Optional[Path] = None
			console_print(
				f"[INFO] Using automatic chunking for {file_path.name}")
		elif chosen_method == "auto-adjust":
			chunk_choice = "auto-adjust"
			line_ranges_file = None
			console_print(
				f"[INFO] Using auto-adjusted chunking for {file_path.name}")
		elif chosen_method == "line_ranges.txt":
			chunk_choice = "line_ranges.txt"
			line_ranges_file = file_path.with_name(
				f"{file_path.stem}_line_ranges.txt")
			if not line_ranges_file.exists():
				console_print(
					f"[WARN] Line ranges file {line_ranges_file.name} not found. Defaulting to automatic chunking.")
				chunk_choice = "auto"
				line_ranges_file = None
			else:
				console_print(
					f"[INFO] Using line ranges from {line_ranges_file.name}")
		else:
			console_print(
				"[WARN] Invalid chunking selection, defaulting to automatic chunking.")
			chunk_choice = "auto"
			line_ranges_file = None

		# -- Perform Text Chunking --
		try:
			openai_config_task: Dict[str, Any] = {
				"model_name": self.model_config["extraction_model"]["name"],
				"default_tokens_per_chunk": self.chunking_config["chunking"][
					"default_tokens_per_chunk"]
			}
			text_processor_obj: TextProcessor = TextProcessor()
			chunks, ranges = perform_chunking(
				normalized_lines, text_processor_obj,
				openai_config_task, chunk_choice, 1, line_ranges_file
			)
			logger.info(
				f"Total chunks generated from {file_path.name}: {len(chunks)}")
			console_print(
				f"[INFO] Generated {len(chunks)} text chunks from {file_path.name}")
		except Exception as e:
			logger.error(f"Error chunking text from {file_path.name}: {e}")
			console_print(
				f"[ERROR] Failed to chunk text from {file_path.name}: {e}")
			return

		# -- Get Additional Context if Enabled --
		additional_context = None
		if context_settings and context_settings.get("use_additional_context",
		                                             False):
			if context_settings.get("use_default_context",
			                        False) and context_manager:
				# Use context from the schema-specific file
				try:
					additional_context = context_manager.get_additional_context(
						selected_schema["name"])
					if additional_context:
						console_print(
							f"[INFO] Using default additional context for schema: {selected_schema['name']}")
						logger.info(
							f"Using default additional context for schema: {selected_schema['name']}")
					else:
						console_print(
							f"[INFO] No default additional context found for schema: {selected_schema['name']}")
						logger.info(
							f"No default additional context found for schema: {selected_schema['name']}")
				except Exception as e:
					logger.error(
						f"Error loading default context for {selected_schema['name']}: {e}")
					console_print(
						f"[ERROR] Failed to load default context: {e}")
			else:
				# Use file-specific context if available
				try:
					additional_context = self.load_file_specific_context(
						file_path)
					if additional_context:
						console_print(
							f"[INFO] Using file-specific context for: {file_path.name}")
						logger.info(
							f"Using file-specific context for: {file_path.name}")
					else:
						console_print(
							f"[INFO] No file-specific context found for: {file_path.name}")
						logger.info(
							f"No file-specific context found for: {file_path.name}")
				except Exception as e:
					logger.error(
						f"Error loading file-specific context for {file_path.name}: {e}")
					console_print(
						f"[ERROR] Failed to load file-specific context: {e}")

		# -- Determine Working Folders and Output Paths --
		try:
			if self.paths_config["general"]["input_paths_is_output_path"]:
				working_folder: Path = file_path.parent
				output_json_path: Path = working_folder / f"{file_path.stem}_output.json"
				temp_jsonl_path: Path = working_folder / f"{file_path.stem}_temp.jsonl"
				working_folder.mkdir(parents=True, exist_ok=True)
			else:
				working_folder = Path(schema_paths["output"])
				temp_folder: Path = working_folder / "temp_jsonl"
				working_folder.mkdir(parents=True, exist_ok=True)
				temp_folder.mkdir(parents=True, exist_ok=True)
				output_json_path = working_folder / f"{file_path.stem}_output.json"
				temp_jsonl_path = temp_folder / f"{file_path.stem}_temp.jsonl"

			console_print(f"[INFO] Output will be saved to: {output_json_path}")
		except Exception as e:
			logger.error(
				f"Error setting up output paths for {file_path.name}: {e}")
			console_print(f"[ERROR] Failed to set up output paths: {e}")
			return

		results: List[Dict[str, Any]] = []
		try:
			handler = get_schema_handler(selected_schema["name"])
		except Exception as e:
			logger.error(
				f"Error getting schema handler for {selected_schema['name']}: {e}")
			console_print(f"[ERROR] Failed to get schema handler: {e}")
			return

		# -- Process API Requests --
		if use_batch:
			console_print(
				f"[INFO] Preparing batch processing for {len(chunks)} chunks...")
			# Batch processing: Prepare batch requests and submit
			try:
				from modules.batching import submit_batch
				batch_requests: List[Dict[str, Any]] = []
				for idx, chunk in enumerate(chunks, 1):
					request_obj = handler.prepare_payload(
						chunk, dev_message, self.model_config,
						selected_schema["schema"],
						additional_context=additional_context
					)
					request_obj["custom_id"] = f"{file_path.stem}-chunk-{idx}"
					batch_requests.append(request_obj)

				with temp_jsonl_path.open("w", encoding="utf-8") as tempf:
					for req in batch_requests:
						tempf.write(json.dumps(req) + "\n")

				logger.info(
					f"Wrote {len(batch_requests)} batch request(s) to {temp_jsonl_path}")
				console_print(
					f"[INFO] Created {len(batch_requests)} batch requests for {file_path.name}")
			except Exception as e:
				logger.error(
					f"Error preparing batch requests for {file_path.name}: {e}")
				console_print(f"[ERROR] Failed to prepare batch requests: {e}")
				return

			try:
				console_print(
					f"[INFO] Submitting batch job for {file_path.name}...")
				batch_response: Any = submit_batch(temp_jsonl_path)
				tracking_record: Dict[str, Any] = {
					"batch_tracking": {
						"batch_id": batch_response.id,
						"timestamp": batch_response.created_at,
						"batch_file": str(temp_jsonl_path)
					}
				}
				with temp_jsonl_path.open("a", encoding="utf-8") as tempf:
					tempf.write(json.dumps(tracking_record) + "\n")
				console_print(
					f"[SUCCESS] Batch submitted successfully. Batch ID: {batch_response.id}")
				logger.info(
					f"Batch submitted successfully. Tracking record appended to {temp_jsonl_path}")
			except Exception as e:
				logger.error(
					f"Error during batch submission for {file_path.name}: {e}")
				console_print(f"[ERROR] Failed to submit batch: {e}")
				return
		else:
			# Synchronous processing: Process each chunk using async API calls
			api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
			if not api_key:
				logger.error(
					"OPENAI_API_KEY is not set in environment variables.")
				console_print(
					"[ERROR] OPENAI_API_KEY is not set in environment variables.")
				return

			console_print(
				f"[INFO] Starting synchronous processing of {len(chunks)} chunks...")
			try:
				async with open_extractor(
						api_key=api_key,
						prompt_path=Path(
							"prompts/structured_output_prompt.txt"),
						model=self.model_config["extraction_model"]["name"]
				) as extractor:
					with temp_jsonl_path.open("w", encoding="utf-8") as tempf:
						for idx, chunk in enumerate(chunks, 1):
							try:
								console_print(
									f"[INFO] Processing chunk {idx}/{len(chunks)}...")
								final_payload = handler.get_json_schema_payload(
									dev_message, self.model_config,
									selected_schema["schema"]
								)

								# Prepend additional context if available
								text_to_process = chunk
								if additional_context:
									text_to_process = f"{additional_context}\n\n{chunk}"

								response: str = await process_text_chunk(
									text_chunk=text_to_process,
									extractor=extractor,
									system_message=dev_message,
									json_schema=final_payload
								)
								result_record: Dict[str, Any] = {
									"custom_id": f"{file_path.stem}-chunk-{idx}",
									"response": response,
									"chunk_range": ranges[idx - 1]
								}
								tempf.write(json.dumps(result_record) + "\n")
								results.append(result_record)
								console_print(
									f"[SUCCESS] Processed chunk {idx}/{len(chunks)}")
								logger.info(
									f"Processed chunk {idx} for file {file_path.name} with range {ranges[idx - 1]}")
							except Exception as e:
								logger.error(
									f"Error processing chunk {idx} of {file_path.name}: {e}")
								console_print(
									f"[ERROR] Failed to process chunk {idx}: {e}")
			except Exception as e:
				logger.error(
					f"Error during synchronous processing for {file_path.name}: {e}")
				console_print(f"[ERROR] Error during processing: {e}")
				return

			# Write final JSON output and optionally convert to additional formats
			try:
				with output_json_path.open("w", encoding="utf-8") as outf:
					json.dump(results, outf, indent=2)
				console_print(
					f"[SUCCESS] Final structured JSON output saved to {output_json_path}")
				logger.info(
					f"Structured JSON output saved to {output_json_path}")

				# Generate additional output formats if configured
				if schema_paths.get("csv_output", False):
					output_csv_path: Path = output_json_path.with_suffix(".csv")
					handler.convert_to_csv(output_json_path, output_csv_path)
					console_print(
						f"[INFO] CSV output saved to {output_csv_path}")

				if schema_paths.get("docx_output", False):
					output_docx_path: Path = output_json_path.with_suffix(
						".docx")
					handler.convert_to_docx(output_json_path, output_docx_path)
					console_print(
						f"[INFO] DOCX output saved to {output_docx_path}")

				if schema_paths.get("txt_output", False):
					output_txt_path: Path = output_json_path.with_suffix(".txt")
					handler.convert_to_txt(output_json_path, output_txt_path)
					console_print(
						f"[INFO] TXT output saved to {output_txt_path}")
			except Exception as e:
				logger.error(
					f"Error writing output files for {file_path.name}: {e}")
				console_print(f"[ERROR] Failed to write output files: {e}")

		# -- Cleanup Temporary Files if Not Needed --
		if use_batch:
			logger.info(
				f"Batch processing enabled. Keeping temporary JSONL for {file_path.name}")
			console_print(
				f"[INFO] Preserving {temp_jsonl_path.name} for batch tracking (required for retrieval)")
		else:
			keep_temp: bool = self.paths_config["general"].get(
				"retain_temporary_jsonl", True)
			if not keep_temp:
				try:
					temp_jsonl_path.unlink()
					logger.info(f"Deleted temporary file: {temp_jsonl_path}")
					console_print(
						f"[CLEANUP] Deleted temporary file: {temp_jsonl_path.name}")
				except Exception as e:
					logger.error(
						f"Error deleting temporary file {temp_jsonl_path}: {e}")
					console_print(
						f"[ERROR] Could not delete temporary file {temp_jsonl_path.name}: {e}")

		console_print(
			f"[SUCCESS] Completed processing of file: {file_path.name}")

	def load_file_specific_context(self, file_path: Path) -> Optional[str]:
		"""
		Load file-specific context from a corresponding _context.txt file.

		:param file_path: Path to the original text file
		:return: Context string if available, None otherwise
		"""
		context_file_path = file_path.with_name(f"{file_path.stem}_context.txt")
		if context_file_path.exists():
			try:
				with context_file_path.open("r", encoding="utf-8") as f:
					context = f.read().strip()
				logger.info(
					f"Loaded file-specific context from {context_file_path}")
				return context
			except Exception as e:
				logger.error(
					f"Error reading context file {context_file_path}: {e}")
		else:
			logger.info(
				f"No file-specific context file found at {context_file_path}")
		return None

	def _default_ask_file_chunking_method(self, file_name: str) -> str:
		"""Default implementation if UI not provided"""
		print(f"\nSelect chunking method for file '{file_name}':")
		print(
			"  1. Automatic chunking - Split text based on token limits with no intervention")
		print(
			"  2. Interactive chunking - View default chunks and manually adjust boundaries")
		print(
			"  3. Predefined chunks - Use saved boundaries from {file}_line_ranges.txt file")

		choice = input("Enter option (1-3): ").strip()
		if choice == "1":
			return "auto"
		elif choice == "2":
			return "auto-adjust"
		elif choice == "3":
			return "line_ranges.txt"
		else:
			print("Invalid selection, defaulting to automatic chunking.")
			return "auto"
