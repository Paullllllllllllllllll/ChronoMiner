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
from typing import Dict, List, Optional, Tuple

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
from modules.core.prompt_context import load_basic_context
from modules.operations.line_ranges.readjuster import LineRangeReadjuster

# Initialize logger
logger = setup_logger(__name__)


def _resolve_line_ranges_file(text_file: Path) -> Optional[Path]:
    """Detect the line range file associated with text_file."""
    candidates = [
        text_file.with_name(f"{text_file.stem}_line_ranges.txt"),
        text_file.with_name(f"{text_file.stem}_line_range.txt"),
        text_file.with_name("line_ranges.txt"),
        text_file.with_name("line_range.txt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _prompt_int(ui: UserInterface, message: str, default: int) -> int:
    """Prompt user for an integer value."""
    ui.console_print(f"\n{message} (press Enter to keep {default}): ")
    try:
        response = input().strip()
    except EOFError:
        response = ""
    if not response:
        return default
    try:
        value = int(response)
        return max(1, value)
    except ValueError:
        ui.console_print("[WARN] Invalid number provided; using default.")
        return default


def _prompt_yes_no(ui: UserInterface, message: str, default: bool) -> bool:
    """Prompt user for a yes/no response."""
    hint = "Y/n" if default else "y/N"
    ui.console_print(f"\n{message} ({hint}): ")
    try:
        response = input().strip().lower()
    except EOFError:
        response = ""
    if not response:
        return default
    if response in {"y", "yes"}:
        return True
    if response in {"n", "no"}:
        return False
    ui.console_print(f"[WARN] Unrecognized response '{response}'; using default.")
    return default


async def _adjust_line_ranges_workflow(
    *,
    files: List[Path],
    selected_schema_name: str,
    model_config: Dict,
    chunking_config: Dict,
    matching_config: Dict,
    retry_config: Dict,
    basic_context: Optional[str],
    context_settings: Dict,
    context_manager,
    ui: UserInterface,
) -> None:
    """Execute the line range adjustment workflow for selected files."""
    ui.console_print("\n" + "=" * 80)
    ui.console_print("  LINE RANGE ADJUSTMENT WORKFLOW")
    ui.console_print("=" * 80)

    # Ask if user wants to use schema name as boundary type
    use_same_schema = _prompt_yes_no(
        ui,
        f"Use schema '{selected_schema_name}' as the semantic boundary type?",
        default=True,
    )

    if use_same_schema:
        boundary_type = selected_schema_name
    else:
        ui.console_print("\nEnter the semantic boundary type name: ")
        try:
            boundary_type = input().strip()
            if not boundary_type:
                ui.console_print("[WARN] No boundary type provided; using schema name.")
                boundary_type = selected_schema_name
        except EOFError:
            boundary_type = selected_schema_name

    # Get context window size
    default_context_window = int(
        chunking_config.get("chunking", {}).get("line_range_context_window", 6) or 6
    )
    context_window = _prompt_int(
        ui,
        "Enter context window size (lines to inspect around boundaries)",
        default_context_window,
    )

    # Ask about dry run
    dry_run = _prompt_yes_no(
        ui,
        "Perform a dry run (preview adjustments without modifying files)?",
        default=False,
    )

    # Get prompt path override if configured
    prompt_override = chunking_config.get("chunking", {}).get("line_range_prompt_path")
    prompt_path: Optional[Path] = Path(prompt_override).resolve() if prompt_override else None

    # Display summary
    ui.console_print("\n" + "-" * 80)
    ui.console_print(f"Selected files: {len(files)}")
    ui.console_print(f"Boundary type: {boundary_type}")
    ui.console_print(f"Context window: {context_window}")
    ui.console_print(f"Dry run: {'yes' if dry_run else 'no'}")
    if prompt_path:
        ui.console_print(f"Prompt override: {prompt_path}")

    if context_settings.get("use_additional_context", False):
        context_source = (
            "Default boundary-type-specific"
            if context_settings.get("use_default_context", False)
            else "File-specific"
        )
    else:
        context_source = "None"
    ui.console_print(f"Additional context: {context_source}")
    ui.console_print("-" * 80)

    # Initialize readjuster
    readjuster = LineRangeReadjuster(
        model_config,
        context_window=context_window,
        prompt_path=prompt_path,
        matching_config=matching_config,
        retry_config=retry_config,
    )

    # Process each file
    successes = 0
    skipped = 0
    failures = 0

    for text_file in files:
        line_ranges_file = _resolve_line_ranges_file(text_file)
        if not line_ranges_file:
            ui.console_print(
                f"[WARN] Skipping {text_file.name}: no associated line range file found."
            )
            skipped += 1
            continue

        ui.console_print(
            f"[INFO] Adjusting line ranges for {text_file.name} (context window: {context_window}, boundary: {boundary_type})..."
        )
        try:
            await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=line_ranges_file,
                dry_run=dry_run,
                boundary_type=boundary_type,
                basic_context=basic_context,
                context_settings=context_settings,
                context_manager=context_manager,
            )
            status = "previewed" if dry_run else "updated"
            ui.console_print(
                f"[SUCCESS] Line ranges for {text_file.name} {status} using {line_ranges_file.name}."
            )
            successes += 1
        except Exception as exc:
            logger.exception("Error adjusting %s", text_file, exc_info=exc)
            ui.console_print(f"[ERROR] Failed to adjust {text_file.name}: {exc}")
            failures += 1

    # Display results
    ui.console_print("\n" + "=" * 80)
    ui.console_print("  ADJUSTMENT SUMMARY")
    ui.console_print("=" * 80)
    ui.console_print(f"Successful adjustments: {successes}")
    ui.console_print(f"Skipped (no line ranges): {skipped}")
    ui.console_print(f"Failures: {failures}")

    if dry_run and successes > 0:
        ui.console_print(
            "\n[INFO] Dry run enabled; no files were modified. The line ranges shown are previews only."
        )


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
            "chunking": (chunking_and_context_config or {}).get("chunking", {})
        }

        # Initialize file processor
        file_processor = FileProcessor(
            paths_config=paths_config,
            model_config=model_config,
            chunking_config=chunking_config,
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
                schemas_paths[selected_schema_name].get("input")
            )
        else:
            raw_text_dir = Path(
                paths_config.get("input_paths", {}).get("raw_text_dir", "")
            )

        files = ui.select_input_source(raw_text_dir)

        # Handle line range adjustment workflow if selected
        if global_chunking_method == "adjust-line-ranges":
            # Load basic context and additional configs needed for adjustment
            basic_context = load_basic_context()
            matching_config = (chunking_and_context_config or {}).get("matching", {})
            retry_config = (chunking_and_context_config or {}).get("retry", {})
            
            # Run the adjustment workflow
            await _adjust_line_ranges_workflow(
                files=files,
                selected_schema_name=selected_schema_name,
                model_config=model_config,
                chunking_config=chunking_and_context_config or {},
                matching_config=matching_config,
                retry_config=retry_config,
                basic_context=basic_context,
                context_settings=context_settings,
                context_manager=context_manager,
                ui=ui,
            )
            
            # After adjustment, use the adjusted line ranges for processing
            ui.console_print("\n[INFO] Line range adjustment complete. Proceeding with processing using adjusted line ranges...")
            global_chunking_method = "line_ranges.txt"

        # Confirm processing
        proceed = ui.display_processing_summary(
            files,
            selected_schema_name,
            global_chunking_method,
            use_batch,
            context_settings,
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
            tasks.append(
                file_processor.process_file(
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
                    ui=ui,
                )
            )
        await asyncio.gather(*tasks)

        # Final summary
        ui.console_print("\n" + "=" * 80)
        ui.console_print("  PROCESSING COMPLETE")
        ui.console_print("=" * 80)

        if use_batch:
            ui.console_print(
                "\n[INFO] Batch processing jobs have been submitted."
            )
            ui.console_print(
                "[INFO] To check the status of your batches, run: python main/check_batches.py"
            )
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
