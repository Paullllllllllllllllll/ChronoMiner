# main/process_text_files.py

"""
Main script for processing text files with schema-based structured data extraction.

Workflow:
 1. Collect all processing options (chunking, batching, additional context)
 2. Load configuration and prompt the user to select a schema.
 3. Determine input source (single file or folder) and gather files.
 4. For each file, process with the FileProcessor class for extraction.
"""

import asyncio
import sys
import traceback
from pathlib import Path

from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from modules.schema_manager import SchemaManager
from modules.context_manager import ContextManager
from modules.user_interface import UserInterface
from modules.file_processor import FileProcessor
from modules.config_manager import ConfigManager

# Initialize logger
logger = setup_logger(__name__)


async def main() -> None:
	"""Main entry point"""
	try:
		# Initialize components
		ui = UserInterface(logger)
		ui.display_banner()

		# Load configuration
		ui.console_print("[INFO] Loading configuration...")
		config_loader = ConfigLoader()
		config_loader.load_configs()

		config_manager = ConfigManager(config_loader)
		paths_config = config_loader.get_paths_config()

		# Validate paths
		try:
			config_manager.validate_paths(paths_config)
		except Exception as e:
			ui.console_print(f"[ERROR] Path validation failed: {e}")
			sys.exit(1)

		# Load other configs
		model_config = config_loader.get_model_config()
		chunking_and_context_config = config_loader.get_chunking_and_context_config()
		chunking_config = {
			'chunking': chunking_and_context_config.get('chunking', {})}

		# Initialize file processor
		file_processor = FileProcessor(
			paths_config=paths_config,
			model_config=model_config,
			chunking_config=chunking_config
		)

		# Schema selection
		schema_manager = SchemaManager()
		schema_manager.load_schemas()
		selected_schema, selected_schema_name = ui.select_schema(schema_manager)

		# Load developer message
		dev_message = config_manager.load_developer_message(
			selected_schema_name)

		# Get user preferences
		global_chunking_method = ui.ask_global_chunking_mode(
			chunking_config["chunking"]["chunking_method"])
		use_batch = ui.ask_batch_processing()
		context_settings = ui.ask_additional_context_mode()

		# Initialize context manager if needed
		context_manager = None
		if context_settings.get("use_additional_context",
		                        False) and context_settings.get(
				"use_default_context", False):
			context_manager = ContextManager()
			context_manager.load_additional_context()

		# Select input files
		schemas_paths = config_loader.get_schemas_paths()
		if selected_schema_name in schemas_paths:
			raw_text_dir = Path(
				schemas_paths[selected_schema_name].get("input"))
		else:
			raw_text_dir = Path(
				paths_config.get("input_paths", {}).get("raw_text_dir", ""))

		files = ui.select_input_source(raw_text_dir)

		# Confirm processing
		proceed = ui.display_processing_summary(
			files,
			selected_schema_name,
			global_chunking_method,
			use_batch,
			context_settings
		)

		if not proceed:
			ui.console_print("[INFO] Processing cancelled by user.")
			return

		# Process files
		ui.console_print("\n" + "=" * 80)
		ui.console_print("  STARTING PROCESSING")
		ui.console_print("=" * 80)

		tasks = []
		for file_path in files:
			tasks.append(file_processor.process_file(
				file_path=file_path,
				use_batch=use_batch,
				selected_schema=selected_schema,
				dev_message=dev_message,
				schema_paths=schemas_paths.get(selected_schema_name, {}),
				global_chunking_method=global_chunking_method,
				context_settings=context_settings,
				context_manager=context_manager,
				ui=ui
			))
		await asyncio.gather(*tasks)

		# Final summary
		ui.console_print("\n" + "=" * 80)
		ui.console_print("  PROCESSING COMPLETE")
		ui.console_print("=" * 80)

		if use_batch:
			ui.console_print(
				"\n[INFO] Batch processing jobs have been submitted.")
			ui.console_print(
				"[INFO] To check the status of your batches, run: python main/check_batches.py")
		else:
			ui.console_print("\n[INFO] All selected items have been processed.")

		ui.console_print("\n[INFO] Thank you for using ChronoMiner!")

	except KeyboardInterrupt:
		ui.console_print("\n[INFO] Processing interrupted by user.")
		sys.exit(0)
	except Exception as e:
		logger.exception(f"Unexpected error: {e}")
		ui.console_print(f"\n[ERROR] An unexpected error occurred: {e}")
		ui.console_print(f"[INFO] Check the logs for more details.")
		ui.console_print(f"Traceback: {traceback.format_exc()}")
		sys.exit(1)


if __name__ == "__main__":
	asyncio.run(main())
