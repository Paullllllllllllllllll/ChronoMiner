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

from modules.core.logger import setup_logger
from modules.ui.core import UserInterface
from modules.operations.extraction.file_processor import FileProcessor
from modules.config.manager import ConfigManager
from modules.llm.prompt_utils import load_prompt_template
from modules.core.workflow_utils import (
    load_core_resources,
    load_schema_manager,
    prepare_context_manager,
)

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
		(
			config_loader,
			paths_config,
			model_config,
			chunking_and_context_config,
			schemas_paths,
		) = load_core_resources()

		config_manager = ConfigManager(config_loader)

		# Validate paths
		try:
			config_manager.validate_paths(paths_config)
		except Exception as e:
			ui.console_print(f"[ERROR] Path validation failed: {e}")
			sys.exit(1)

		# Load other configs
		chunking_config = {
			'chunking': (chunking_and_context_config or {}).get('chunking', {})}

		# Initialize file processor
		file_processor = FileProcessor(
			paths_config=paths_config,
			model_config=model_config,
			chunking_config=chunking_config
		)

		# Schema selection
		try:
			schema_manager = load_schema_manager()
		except RuntimeError as exc:
			ui.console_print(f"[ERROR] {exc}.")
			sys.exit(1)
		selected_schema, selected_schema_name = ui.select_schema(schema_manager)

		# Load unified prompt template
		prompt_path = Path("prompts/structured_output_prompt.txt")
		try:
			prompt_template = load_prompt_template(prompt_path)
		except FileNotFoundError as exc:
			ui.console_print(f"[ERROR] {exc}")
			logger.error("Prompt template missing", exc_info=exc)
			sys.exit(1)

		# Get user preferences
		global_chunking_method = ui.ask_global_chunking_mode()
		use_batch = ui.ask_batch_processing()
		context_settings = ui.ask_additional_context_mode()

		# Initialize context manager when default context is requested
		context_manager = prepare_context_manager(context_settings)

		# Select input files
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
		inject_schema = model_config.get("transcription_model", {}).get(
			"inject_schema_into_prompt", True
		)

		for file_path in files:
			tasks.append(file_processor.process_file(
				file_path=file_path,
				use_batch=use_batch,
				selected_schema=selected_schema,
				prompt_template=prompt_template,
				schema_name=selected_schema_name,
				inject_schema=inject_schema,
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
