# modules/user_interface.py

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class UserInterface:
	"""
	Class for handling all user interaction aspects of ChronoMiner.
	Provides a consistent interface for prompts, selections, and feedback.
	"""

	def __init__(self, logger: Optional[logging.Logger] = None) -> None:
		"""
		Initialize the user interface.

		:param logger: Optional logger instance for logging user interactions
		"""
		self.logger = logger

	def console_print(self, message: str) -> None:
		"""Simple wrapper for printing to console with optional logging."""
		print(message)
		if self.logger:
			self.logger.info(message)

	def display_banner(self) -> None:
		"""Display a welcome banner with information about the application."""
		self.console_print("\n" + "=" * 80)
		self.console_print("  ChronoMiner - Structured Data Extraction Tool")
		self.console_print("=" * 80)
		self.console_print(
			"  This tool helps you extract structured data from text documents")
		self.console_print("  using various schemas and processing strategies.")
		self.console_print("=" * 80 + "\n")

	def select_option(self, prompt: str, options: List[Tuple[str, str]],
	                  allow_quit: bool = True) -> str:
		"""
		Display a prompt with detailed options and return the user's choice.

		:param prompt: The prompt to display to the user
		:param options: List of (value, description) tuples
		:param allow_quit: Whether to allow quitting
		:return: The selected option value
		"""
		self.console_print(f"\n{prompt}")
		self.console_print("-" * 80)

		for idx, (value, description) in enumerate(options, 1):
			self.console_print(f"  {idx}. {description}")

		if allow_quit:
			self.console_print(f"\n  (Type 'q' to exit the application)")

		while True:
			choice = input("\nEnter your choice: ").strip()

			if allow_quit and choice.lower() in ['q', 'quit', 'exit']:
				self.console_print("[INFO] Exiting application.")
				sys.exit(0)

			if choice.isdigit() and 1 <= int(choice) <= len(options):
				return options[int(choice) - 1][0]

			self.console_print("[ERROR] Invalid selection. Please try again.")

	def select_schema(self, schema_manager) -> Tuple[Dict[str, Any], str]:
		"""
		Present available schemas and guide the user through selection.

		:param schema_manager: SchemaManager instance
		:return: Tuple of (schema_dict, schema_name)
		"""
		available_schemas = schema_manager.get_available_schemas()

		if not available_schemas:
			self.console_print(
				"[ERROR] No schemas available. Please add schemas to the 'schemas' folder.")
			sys.exit(0)

		self.console_print("\n" + "=" * 80)
		self.console_print("  SCHEMA SELECTION")
		self.console_print("=" * 80)

		schema_options_with_paths = schema_manager.list_schema_options()
		if schema_options_with_paths:
			schema_options = [
				(name, f"{name} ({path.name})") for name, path in schema_options_with_paths
			]
		else:
			schema_options = [(name, name) for name in available_schemas.keys()]

		selected_schema_name = self.select_option(
			"Select a schema to use for extraction:",
			schema_options
		)

		return available_schemas[selected_schema_name], selected_schema_name

	def ask_global_chunking_mode(self, default_method: str) -> Optional[str]:
		"""
		Present enhanced options for global chunking strategy.

		:param default_method: Default chunking method from config
		:return: Selected chunking method or None
		"""
		self.console_print("\n" + "=" * 80)
		self.console_print("  CHUNKING STRATEGY")
		self.console_print("=" * 80)

		chunking_options = [
			("auto",
			 "Automatic - System determines optimal chunk boundaries based on token limits"),
			("auto-adjust",
			 "Auto-adjust - Like automatic, but with adjustments for better context retention"),
			("line_ranges.txt",
			 "Manual - Use predefined line ranges from corresponding _line_ranges.txt files"),
			("per-file",
			 "Per-file - Choose chunking method individually for each file")
		]

		self.console_print(f"\nDefault method (from config): {default_method}")
		return self.select_option(
			"Select how you would like to chunk the text for processing:",
			chunking_options
		)

	def ask_batch_processing(self) -> bool:
		"""
		Present enhanced options for batch processing selection.

		:return: True if batch processing selected, False otherwise
		"""
		self.console_print("\n" + "=" * 80)
		self.console_print("  PROCESSING MODE")
		self.console_print("=" * 80)

		batch_options = [
			("sync",
			 "Synchronous processing - Process data in real-time with immediate results"),
			("batch",
			 "Batch processing - Submit all chunks as a batch job (50% cost reduction, results within 24h)")
		]

		mode = self.select_option(
			"Select how you would like to process the data:",
			batch_options
		)

		return mode == "batch"

	def ask_additional_context_mode(self) -> Dict[str, Any]:
		"""
		Present enhanced options for additional context handling.

		:return: Dictionary with context settings
		"""
		self.console_print("\n" + "=" * 80)
		self.console_print("  ADDITIONAL CONTEXT")
		self.console_print("=" * 80)

		use_context_options = [
			("yes",
			 "Yes - Provide additional context to improve extraction accuracy"),
			("no", "No - Process text without additional context")
		]

		use_context = self.select_option(
			"Would you like to provide additional context for extraction?",
			use_context_options
		)

		context_settings = {"use_additional_context": use_context == "yes"}

		if context_settings["use_additional_context"]:
			context_source_options = [
				("default",
				 "Use default context files specific to the selected schema"),
				("file",
				 "Use file-specific context files (e.g., filename_context.txt)")
			]

			context_source = self.select_option(
				"Which source of context would you like to use?",
				context_source_options
			)

			context_settings[
				"use_default_context"] = context_source == "default"

		return context_settings

	def select_input_source(self, raw_text_dir: Path) -> List[Path]:
		"""
		Guide user through selecting input source (single file or folder).

		:param raw_text_dir: Base directory for input files
		:return: List of selected file paths
		"""
		self.console_print("\n" + "=" * 80)
		self.console_print("  INPUT SELECTION")
		self.console_print("=" * 80)

		mode_options = [
			("single", "Process a single file"),
			("folder", "Process all files in a folder")
		]

		mode = self.select_option(
			"Select how you would like to specify input:",
			mode_options
		)

		files = []

		if mode == "single":
			self.console_print(
				"\nEnter the filename to process (with or without .txt)."
			)
			self.console_print(
				"  • Enter the base text filename; the matching line range file will be used automatically."
			)
			self.console_print(
				"  • Or enter the line range filename ending in '_line_ranges' to work with it directly."
			)
			file_input = input("> ").strip()
			normalized_input = (
				file_input if file_input.lower().endswith(".txt") else f"{file_input}.txt"
			)

			wants_line_range = normalized_input.lower().endswith(
				("_line_ranges.txt", "_line_range.txt")
			)
			excluded_suffixes = ["_context.txt"]
			if not wants_line_range:
				excluded_suffixes.extend(["_line_ranges.txt", "_line_range.txt"])

			file_candidates = [
				f
				for f in raw_text_dir.rglob(normalized_input)
				if not any(f.name.endswith(suffix) for suffix in excluded_suffixes)
			]

			if not file_candidates:
				self.console_print(
					f"[ERROR] File {normalized_input} does not exist in {raw_text_dir}"
				)
				sys.exit(0)
			elif len(file_candidates) == 1:
				files.append(file_candidates[0])
				self.console_print(f"[INFO] Selected file: {files[0].name}")
			else:
				self.console_print("\nMultiple matching files found:")
				self.console_print("-" * 80)

				for idx, f in enumerate(file_candidates, 1):
					self.console_print(f"  {idx}. {f}")

				while True:
					selected_index = input("\nSelect file by number: ").strip()

					if selected_index.lower() in ['q', 'quit', 'exit']:
						self.console_print("[INFO] Exiting application.")
						sys.exit(0)

					try:
						idx = int(selected_index) - 1
						if 0 <= idx < len(file_candidates):
							files.append(file_candidates[idx])
							self.console_print(
								f"[INFO] Selected file: {files[0].name}")
							break
						else:
							self.console_print(
								"[ERROR] Invalid selection. Please try again.")
					except ValueError:
						self.console_print(
							"[ERROR] Invalid input. Please enter a number.")

		elif mode == "folder":
			# Get all .txt files, filtering out auxiliary files
			files = [f for f in raw_text_dir.rglob("*.txt")
			         if not (f.name.endswith("_line_ranges.txt") or
			                 f.name.endswith("_context.txt"))]

			if not files:
				self.console_print(
					f"[ERROR] No .txt files found in {raw_text_dir}")
				sys.exit(0)

			self.console_print(
				f"[INFO] Found {len(files)} text files to process.")

		return files

	def display_processing_summary(self, files: List[Path],
	                               selected_schema_name: str,
	                               global_chunking_method: Optional[str],
	                               use_batch: bool,
	                               context_settings: Dict[str, Any]) -> bool:
		"""
		Display a summary of the selected processing options and ask for confirmation.

		:param files: List of selected file paths
		:param selected_schema_name: Name of the selected schema
		:param global_chunking_method: Selected chunking method
		:param use_batch: Whether batch processing is enabled
		:param context_settings: Context settings dictionary
		:return: True if user confirms, False otherwise
		"""
		self.console_print("\n" + "=" * 80)
		self.console_print("  PROCESSING SUMMARY")
		self.console_print("=" * 80)

		file_type = "file" if len(files) == 1 else "files"
		self.console_print(
			f"\nReady to process {len(files)} {file_type} with the following settings:")
		self.console_print(f"  - Schema: {selected_schema_name}")

		chunking_display = {
			"auto": "Automatic chunking",
			"auto-adjust": "Auto-adjusted chunking",
			"line_ranges.txt": "Manual chunking (using line_ranges.txt files)",
			"per-file": "Per-file chunking selection"
		}

		self.console_print(
			f"  - Chunking strategy: {chunking_display.get(global_chunking_method, 'Per-file selection')}")

		processing_mode = "Batch (asynchronous)" if use_batch else "Synchronous (real-time)"
		self.console_print(f"  - Processing mode: {processing_mode}")

		if context_settings.get("use_additional_context", False):
			context_source = "Default schema-specific" if context_settings.get(
				"use_default_context", False) else "File-specific"
			self.console_print(f"  - Additional context: {context_source}")
		else:
			self.console_print("  - Additional context: None")

		# First few items to show
		self.console_print("\nSelected files (first 5 shown):")
		for i, item in enumerate(files[:5]):
			self.console_print(f"  {i + 1}. {item.name}")

		if len(files) > 5:
			self.console_print(f"  ... and {len(files) - 5} more")

		confirmation = input(
			"\nProceed with processing? (y/n): ").strip().lower()
		return confirmation == "y"

	def ask_file_chunking_method(self, file_name: str) -> str:
		"""
		Prompt the user to select a chunking method for the given file.

		:param file_name: Name of the file to prompt for
		:return: Selected chunking method
		"""
		self.console_print(f"\nSelect chunking method for file '{file_name}':")
		self.console_print(
			"  1. Automatic chunking - Split text based on token limits with no intervention")
		self.console_print(
			"  2. Interactive chunking - View default chunks and manually adjust boundaries")
		self.console_print(
			"  3. Predefined chunks - Use saved boundaries from {file}_line_ranges.txt file")

		choice = input("Enter option (1-3): ").strip()
		if choice == "1":
			return "auto"
		elif choice == "2":
			return "auto-adjust"
		elif choice == "3":
			return "line_ranges.txt"
		else:
			self.console_print(
				"Invalid selection, defaulting to automatic chunking.")
			return "auto"

	def display_batch_summary(self, batches: List[Any]) -> None:
		"""
		Display a summary of batch job statuses.

		:param batches: List of batch objects from OpenAI API
		"""
		# Count batches by status
		status_counts = {}
		in_progress_batches = []

		for batch in batches:
			if isinstance(batch, dict):
				status = str(batch.get("status", "unknown")).lower()
				batch_id = batch.get("id", "unknown")
				created_time = batch.get("created_at") or batch.get("created") or "Unknown"
			else:
				status = str(getattr(batch, "status", "unknown")).lower()
				batch_id = getattr(batch, "id", "unknown")
				created_time = getattr(batch, "created_at", getattr(batch, "created", "Unknown"))
			status_counts[status] = status_counts.get(status, 0) + 1

			# Keep track of non-terminal batches for detailed display
			if status not in {"completed", "expired", "cancelled", "failed"}:
				in_progress_batches.append((batch_id, status, created_time))

		# Display summary
		self.console_print("\n" + "=" * 80)
		self.console_print("  BATCH JOBS SUMMARY")
		self.console_print("=" * 80)
		self.console_print(f"Total batches found: {len(batches)}")

		# Display counts by status
		for status, count in sorted(status_counts.items()):
			self.console_print(f"  - {status.capitalize()}: {count}")

		# Display in-progress batches if any
		if in_progress_batches:
			self.console_print("\nBatches still in progress:")
			for batch_id, status, created_time in in_progress_batches:
				self.console_print(
					f"  - Batch ID: {batch_id} | Status: {status} | Created: {created_time}")

	def display_batch_processing_progress(
		self,
		temp_file: Path,
		batch_ids: List[str],
		completed_batches: int,
		missing_batches: int,
		failed_batches: List[Tuple[Dict[str, Any], str]],
	) -> None:
		"""Print a concise progress summary for a temp batch file."""
		total_batches = len(batch_ids)
		self.console_print(
			f"[INFO] {temp_file.name}: {completed_batches}/{total_batches} batches completed"
		)
		if missing_batches:
			self.console_print(
				f"[WARN] {missing_batches} batch id(s) missing from OpenAI responses."
			)
		if failed_batches:
			self.console_print(
				"[WARN] Failed or terminal batches detected:"
			)
			for track, status in failed_batches:
				bid = track.get("batch_id", "unknown")
				self.console_print(f"  - Batch ID: {bid} | Status: {status}")

	def display_batch_operation_result(self, batch_id: str, operation: str,
	                                   success: bool,
	                                   message: str = None) -> None:
		"""
		Display the result of a batch operation (checking, cancelling, etc.).

		:param batch_id: The ID of the batch
		:param operation: The operation performed (e.g., "cancel", "process")
		:param success: Whether the operation was successful
		:param message: Optional message to display
		"""
		status = "[SUCCESS]" if success else "[ERROR]"
		result = f"{status} {operation.capitalize()}ed batch {batch_id}"

		if message:
			result += f": {message}"

		self.console_print(result)
