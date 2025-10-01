# modules/operations/extraction/file_processor.py

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from modules.core.prompt_context import load_basic_context, resolve_additional_context
from modules.core.text_utils import TextProcessor, perform_chunking
from modules.llm.openai_utils import open_extractor, process_text_chunk
from modules.operations.extraction.schema_handlers import get_schema_handler
from modules.llm.batching import build_batch_files, submit_batch
from modules.llm.prompt_utils import render_prompt_with_schema
from modules.operations.line_ranges.readjuster import LineRangeReadjuster
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
	             chunking_config: Dict[str, Any],
	             concurrency_config: Optional[Dict[str, Any]] = None):
		"""
		Initialize with configuration
		"""
		self.paths_config = paths_config
		self.model_config = model_config
		self.chunking_config = chunking_config
		self.concurrency_config = concurrency_config or {}
		self.text_processor = TextProcessor()
		self.basic_context = load_basic_context()
		self._line_range_readjuster: Optional[LineRangeReadjuster] = None

	def _get_line_range_readjuster(self) -> LineRangeReadjuster:
		chunk_settings = self.chunking_config.get("chunking", {}) if isinstance(self.chunking_config, dict) else {}
		context_window = int(chunk_settings.get("line_range_context_window", 6) or 6)
		prompt_path_value = chunk_settings.get("line_range_prompt_path")
		prompt_path = Path(prompt_path_value) if prompt_path_value else None
		if self._line_range_readjuster is None:
			self._line_range_readjuster = LineRangeReadjuster(
				self.model_config,
				context_window=context_window,
				prompt_path=prompt_path,
			)
		return self._line_range_readjuster

	async def process_file(self, file_path: Path,
	                       use_batch: bool,
	                       selected_schema: Dict[str, Any],
	                       prompt_template: str,
	                       schema_name: str,
	                       inject_schema: bool,
	                       schema_paths: Dict[str, Any],
	                       global_chunking_method: Optional[str] = None,
	                       context_settings: Optional[Dict[str, Any]] = None,
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
		:param prompt_template: Base system prompt template text
		:param schema_name: Name of the selected schema
		:param inject_schema: Whether to inject the JSON schema into the system prompt
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
			logger.error("Error reading file %s: %s", file_path, e)
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

		if chunk_choice == "line_ranges.txt" and line_ranges_file is not None:
			try:
				readjuster = self._get_line_range_readjuster()
				await readjuster.ensure_adjusted_line_ranges(
					text_file=file_path,
					line_ranges_file=line_ranges_file,
					boundary_type=schema_name,
					basic_context=self.basic_context,
					context_settings=context_settings,
					context_manager=context_manager,
				)
				logger.info(
					"Refined line ranges for %s using semantic boundary detection",
					file_path.name,
				)
				console_print(
					f"[INFO] Refined line ranges for {file_path.name} using semantic boundaries."
				)
			except Exception as exc:
				logger.warning(
					"Failed to refine line ranges for %s: %s",
					file_path.name,
					exc,
					exc_info=exc,
				)
				console_print(
					f"[WARN] Could not refine line ranges for {file_path.name}: {exc}"
				)

		# -- Perform Text Chunking --
		try:
			openai_config_task: Dict[str, Any] = {
				"model_name": self.model_config["transcription_model"]["name"],
				"default_tokens_per_chunk": self.chunking_config["chunking"][
					"default_tokens_per_chunk"]
			}
			text_processor_obj: TextProcessor = TextProcessor()
			chunks, ranges = perform_chunking(
				normalized_lines, text_processor_obj,
				openai_config_task, chunk_choice, 1, line_ranges_file
			)
			logger.info(
				"Total chunks generated from %s: %s",
				file_path.name,
				len(chunks)
			)
			console_print(
				f"[INFO] Generated {len(chunks)} text chunks from {file_path.name}")
		except Exception as e:
			logger.error("Error chunking text from %s: %s", file_path.name, e)
			console_print(
				f"[ERROR] Failed to chunk text from {file_path.name}: {e}")
			return

		# -- Get Additional Context if Enabled --
		context_settings = context_settings or {}
		additional_context: Optional[str] = resolve_additional_context(
			schema_name,
			context_settings=context_settings,
			context_manager=context_manager,
			text_file=file_path,
		)

		if context_settings.get("use_additional_context", False):
			if context_settings.get("use_default_context", False):
				if additional_context:
					console_print(
						f"[INFO] Using default additional context for schema: {schema_name}")
					logger.info(
						"Using default additional context for schema: %s",
						schema_name
					)
				else:
					console_print(
						f"[INFO] No default additional context found for schema: {schema_name}")
					logger.info(
						"No default additional context found for schema: %s",
						schema_name
					)
			else:
				if additional_context:
					console_print(
						f"[INFO] Using file-specific context for: {file_path.name}")
					logger.info(
						"Using file-specific context for: %s",
						file_path.name
					)
				else:
					logger.info(
						"No file-specific context found for: %s",
						file_path.name
					)

		# -- Render System Prompt --
		schema_definition = selected_schema.get("schema", {})
		effective_dev_message = render_prompt_with_schema(
			prompt_template,
			schema_definition,
			schema_name=schema_name,
			inject_schema=inject_schema,
			additional_context=additional_context,
			basic_context=self.basic_context,
		)

		results: List[Dict[str, Any]] = []
		try:
			handler = get_schema_handler(schema_name)
		except Exception as e:
			logger.error(
				"Error getting schema handler for %s: %s",
				schema_name,
				e
			)
			console_print(f"[ERROR] Failed to get schema handler: {e}")
			return

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
				"Error setting up output paths for %s: %s",
				file_path.name,
				e
			)
			console_print(f"[ERROR] Failed to set up output paths: {e}")
			return

		# -- Process API Requests --
		if use_batch:
			console_print(
				f"[INFO] Preparing batch processing for {len(chunks)} chunks...")
			# Batch processing: Prepare batch requests and submit
			request_lines: List[str] = []
			try:
				for idx, chunk in enumerate(chunks, 1):
					request_obj = handler.prepare_payload(
						chunk,
						effective_dev_message,
						self.model_config,
						selected_schema["schema"],
					)
					request_obj["custom_id"] = f"{file_path.stem}-chunk-{idx}"
					request_lines.append(json.dumps(request_obj))

				batch_files = build_batch_files(request_lines, temp_jsonl_path)
				if not batch_files:
					console_print(
						f"[ERROR] No batch files were generated for {file_path.name}.")
					return

				logger.info(
					"Created %s batch request(s) across %s file(s) for %s",
					len(request_lines),
					len(batch_files),
					file_path.name
				)
				console_print(
					f"[INFO] Created {len(request_lines)} batch requests split into {len(batch_files)} file(s).")
			except Exception as e:
				logger.error(
					"Error preparing batch requests for %s: %s",
					file_path.name,
					e
				)
				console_print(f"[ERROR] Failed to prepare batch requests: {e}")
				return

			submitted_batches: List[str] = []
			for batch_file in batch_files:
				try:
					console_print(
						f"[INFO] Submitting batch file {batch_file.name}...")
					batch_response: Any = submit_batch(batch_file)
					tracking_record: Dict[str, Any] = {
						"batch_tracking": {
							"batch_id": batch_response.id,
							"timestamp": batch_response.created_at,
							"batch_file": str(batch_file)
						}
					}
					with batch_file.open("a", encoding="utf-8") as tempf:
						tempf.write(json.dumps(tracking_record) + "\n")
					submitted_batches.append(batch_response.id)
					console_print(
						f"[SUCCESS] Batch submitted successfully. Batch ID: {batch_response.id}")
					logger.info(
						f"Batch submitted successfully. Tracking record appended to %s",
						batch_file
					)
				except Exception as e:
					logger.error(
						"Error during batch submission for file %s: %s",
						batch_file,
						e
					)
					console_print(f"[ERROR] Failed to submit batch file {batch_file.name}: {e}")
					return

			logger.info(
				"Submitted %s batch file(s) for %s: %s",
				len(submitted_batches),
				file_path.name,
				submitted_batches
			)
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
						model=self.model_config["transcription_model"]["name"]
				) as extractor:
					with temp_jsonl_path.open("w", encoding="utf-8") as tempf:
						total_chunks: int = len(chunks)
						transcription_cfg: Dict[str, Any] = (
							(self.concurrency_config.get("concurrency", {}) or {}).get("transcription", {}) or {}
						)
						try:
							configured_limit = int(transcription_cfg.get("concurrency_limit", total_chunks or 1))
						except Exception:
							configured_limit = total_chunks or 1
						concurrency_limit = max(1, min(configured_limit, total_chunks or 1))
						delay_between_tasks = float(transcription_cfg.get("delay_between_tasks", 0.0) or 0.0)

						semaphore = asyncio.Semaphore(concurrency_limit)
						write_lock = asyncio.Lock()
						results_map: Dict[int, Dict[str, Any]] = {}

						async def handle_chunk(idx: int, chunk_text: str, chunk_range: Any) -> None:
							async with semaphore:
								console_print(
									f"[INFO] Processing chunk {idx}/{total_chunks}...")
								if delay_between_tasks > 0:
									await asyncio.sleep(delay_between_tasks)

								try:
									final_payload = handler.get_json_schema_payload(
										effective_dev_message, self.model_config,
										selected_schema["schema"]
									)

									text_to_process = f"Input text:\n{chunk_text}"

									response_payload: Dict[str, Any] = await process_text_chunk(
										text_chunk=text_to_process,
										extractor=extractor,
										system_message=effective_dev_message,
										json_schema=final_payload
									)
									output_text: Any = response_payload.get("output_text")
									response_data: Dict[str, Any] = response_payload.get("response_data", {})
									request_metadata: Dict[str, Any] = response_payload.get("request_metadata", {})

									# Parse the output_text JSON string into an actual object
									try:
										parsed_response = json.loads(output_text) if isinstance(output_text, str) else output_text
									except json.JSONDecodeError:
										logger.warning(f"Failed to parse output_text as JSON for chunk {idx}")
												
									temp_record: Dict[str, Any] = {
										"custom_id": f"{file_path.stem}-chunk-{idx}",
										"chunk_index": idx,
										"chunk_range": chunk_range,
										"response": output_text,
										"response_data": response_data,
										"request_metadata": request_metadata,
										"status": "success",
									}

									async with write_lock:
										tempf.write(json.dumps(temp_record) + "\n")
										tempf.flush()

									results_map[idx] = {
										"custom_id": f"{file_path.stem}-chunk-{idx}",
										"chunk_index": idx,
										"chunk_range": chunk_range,
										"response": parsed_response,
									}
								except Exception as exc:
									logger.error(
										"Error processing chunk %s of %s: %s",
										idx,
										file_path.name,
										exc
									)
									error_record: Dict[str, Any] = {
										"custom_id": f"{file_path.stem}-chunk-{idx}",
										"chunk_index": idx,
										"chunk_range": chunk_range,
										"response": None,
										"error": str(exc),
									}
									async with write_lock:
										tempf.write(json.dumps(error_record) + "\n")
										tempf.flush()
									results_map[idx] = {
										"custom_id": f"{file_path.stem}-chunk-{idx}",
										"chunk_index": idx,
										"chunk_range": chunk_range,
										"response": None,
									}

						tasks = [
							asyncio.create_task(handle_chunk(idx, chunk, ranges[idx - 1]))
							for idx, chunk in enumerate(chunks, 1)
						]
						if tasks:
							await asyncio.gather(*tasks)

						# Write final JSON output by reading from temp JSONL file
						try:
							console_print(f"[INFO] Constructing final output from temporary file...")
							results = []
							
							# Read from the temp JSONL file
							if temp_jsonl_path.exists():
								with temp_jsonl_path.open("r", encoding="utf-8") as tempf:
									for line in tempf:
										line = line.strip()
										if not line:
											continue
										try:
											record = json.loads(line)
											# Extract only the essential fields for the final output
											if "custom_id" in record:
												# Parse the response field if it's a string
												response_field = record.get("response")
												if isinstance(response_field, str):
													try:
														response_field = json.loads(response_field)
													except json.JSONDecodeError:
														logger.warning(f"Failed to parse response field for {record.get('custom_id')}")
												
												output_record = {
													"custom_id": record.get("custom_id"),
													"chunk_index": record.get("chunk_index"),
													"chunk_range": record.get("chunk_range"),
													"response": response_field,
												}
												results.append(output_record)
										except json.JSONDecodeError as e:
											logger.warning(f"Failed to parse line in temp file: {e}")
											continue
							
							# Sort results by chunk_index
							results.sort(key=lambda x: x.get("chunk_index", 0))
							
							# Write the final output JSON
							with output_json_path.open("w", encoding="utf-8") as outf:
								json.dump(results, outf, indent=2)
							console_print(
								f"[SUCCESS] Final structured JSON output saved to {output_json_path}")
							logger.info(
								f"Structured JSON output saved to %s",
								output_json_path
							)
						except Exception as e:
							logger.error(
								"Error writing final output for %s: %s",
								file_path.name,
								e
							)
							console_print(f"[ERROR] Failed to write final output: {e}")
							return

			except Exception as e:
				logger.error(
					"Error during synchronous processing for %s: %s",
					file_path.name,
					e
				)
				console_print(f"[ERROR] Error during processing: {e}")
				return

			# Generate additional output formats if configured
			try:
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
					"Error writing output files for %s: %s",
					file_path.name,
					e
				)
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
