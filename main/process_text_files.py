# main/process_text_files.py

"""
Main script for processing text files with schema-based structured data extraction.

Supports two execution modes:
1. Interactive Mode: User-friendly prompts and selections via UI
2. CLI Mode: Command-line arguments for automation and scripting

The mode is controlled by the 'interactive_mode' setting in config/paths_config.yaml
or by providing command-line arguments.

Workflow:
 1. Collect all processing options (chunking, batching, additional context)
 2. Load configuration and prompt the user to select a schema.
 3. Determine input source (single file or folder) and gather files.
 4. For each file, process with the FileProcessor class for extraction.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

from modules.cli.args_parser import create_process_parser, get_files_from_path, resolve_path
from modules.cli.mode_detector import should_use_interactive_mode
from modules.config.manager import ConfigManager
from modules.core.logger import setup_logger
from modules.core.prompt_context import load_basic_context
from modules.core.text_utils import TextProcessor, TokenBasedChunking
from modules.core.workflow_utils import (
    load_core_resources,
    load_schema_manager,
    prepare_context_manager,
)
from modules.llm.prompt_utils import load_prompt_template
from modules.operations.extraction.file_processor import FileProcessor
from modules.operations.line_ranges.readjuster import LineRangeReadjuster
from modules.ui.core import UserInterface

# Initialize logger
logger = setup_logger(__name__)


async def _adjust_line_ranges_workflow(
    files: List[Path],
    selected_schema_name: str,
    model_config: Dict,
    chunking_config: Dict,
    matching_config: Dict,
    retry_config: Dict,
    basic_context: Optional[str],
    context_settings: Dict,
    context_manager,
    ui: Optional[UserInterface],
) -> None:
    """
    Execute line range adjustment workflow for files.
    
    Args:
        files: List of files to adjust
        selected_schema_name: Schema name for boundary detection
        model_config: Model configuration
        chunking_config: Chunking configuration
        matching_config: Matching configuration
        retry_config: Retry configuration
        basic_context: Basic context string
        context_settings: Context settings dictionary
        context_manager: Context manager instance
        ui: Optional UserInterface instance
    """
    context_window = chunking_config.get("chunking", {}).get("line_range_context_window", 6)
    prompt_override = chunking_config.get("chunking", {}).get("line_range_prompt_path")
    prompt_path = Path(prompt_override).resolve() if prompt_override else None
    
    readjuster = LineRangeReadjuster(
        model_config,
        context_window=context_window,
        prompt_path=prompt_path,
        matching_config=matching_config,
        retry_config=retry_config,
    )
    
    for text_file in files:
        # Find associated line ranges file
        line_ranges_file = text_file.with_name(f"{text_file.stem}_line_ranges.txt")
        if not line_ranges_file.exists():
            if ui:
                ui.print_warning(f"No line ranges file for {text_file.name}, skipping adjustment")
            continue
        
        if ui:
            ui.print_info(f"Adjusting line ranges for {text_file.name}...")
        
        try:
            adjusted_path = await readjuster.adjust_line_ranges(
                text_file=text_file,
                line_ranges_file=line_ranges_file,
                boundary_type=selected_schema_name,
                basic_context=basic_context,
                context_settings=context_settings,
                context_manager=context_manager,
            )
            
            if ui:
                ui.print_success(f"Adjusted: {adjusted_path.name}")
        except Exception as e:
            logger.exception(f"Failed to adjust line ranges for {text_file}", exc_info=e)
            if ui:
                ui.print_error(f"Failed to adjust {text_file.name}: {e}")


async def _run_interactive_mode(
    config_loader,
    paths_config: Dict,
    model_config: Dict,
    chunking_and_context_config: Dict,
    schemas_paths: Dict,
) -> None:
    """Run text processing in interactive mode with back navigation support."""
    ui = UserInterface(logger)
    ui.display_banner()
    
    # Load configuration
    ui.print_info("Loading configuration...")
    logger.info("Starting ChronoMiner processing workflow (Interactive Mode).")
    
    config_manager = ConfigManager(config_loader)
    
    # Validate paths
    try:
        config_manager.validate_paths(paths_config)
    except Exception as e:
        ui.print_error(f"Path validation failed: {e}")
        logger.error(f"Path validation error: {e}")
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
        ui.print_error(str(exc))
        logger.error("Failed to load schema manager", exc_info=exc)
        sys.exit(1)
    
    # Load unified prompt template
    prompt_path = Path("prompts/structured_output_prompt.txt")
    try:
        prompt_template = load_prompt_template(prompt_path)
    except FileNotFoundError as exc:
        ui.print_error(f"Prompt template not found: {exc}")
        logger.error("Prompt template missing", exc_info=exc)
        sys.exit(1)
    
    # State machine for navigation
    # States: schema -> chunking -> batch -> context -> files -> confirm
    current_step = "schema"
    state = {}
    
    while True:
        if current_step == "schema":
            result = ui.select_schema(schema_manager, allow_back=False)
            if result is None:
                ui.print_info("Schema selection cancelled.")
                return
            state["selected_schema"], state["selected_schema_name"] = result
            current_step = "chunking"
        
        elif current_step == "chunking":
            global_chunking_method = ui.ask_global_chunking_mode(allow_back=True)
            if global_chunking_method is None:
                current_step = "schema"
                continue
            state["global_chunking_method"] = global_chunking_method
            current_step = "batch"
        
        elif current_step == "batch":
            use_batch = ui.ask_batch_processing(allow_back=True)
            if use_batch is None:
                current_step = "chunking"
                continue
            state["use_batch"] = use_batch
            current_step = "context"
        
        elif current_step == "context":
            context_settings = ui.ask_additional_context_mode(allow_back=True)
            if context_settings is None:
                current_step = "batch"
                continue
            state["context_settings"] = context_settings
            state["context_manager"] = prepare_context_manager(context_settings)
            current_step = "files"
        
        elif current_step == "files":
            # Determine input directory
            if state["selected_schema_name"] in schemas_paths:
                raw_text_dir = Path(schemas_paths[state["selected_schema_name"]].get("input"))
            else:
                raw_text_dir = Path(paths_config.get("input_paths", {}).get("raw_text_dir", ""))
            
            files = ui.select_input_source(raw_text_dir, allow_back=True)
            if files is None:
                current_step = "context"
                continue
            state["files"] = files
            current_step = "confirm"
        
        elif current_step == "confirm":
            # Handle line range adjustment workflow if selected
            if state["global_chunking_method"] == "adjust-line-ranges":
                basic_context = load_basic_context()
                matching_config = (chunking_and_context_config or {}).get("matching", {})
                retry_config = (chunking_and_context_config or {}).get("retry", {})
                
                await _adjust_line_ranges_workflow(
                    files=state["files"],
                    selected_schema_name=state["selected_schema_name"],
                    model_config=model_config,
                    chunking_config=chunking_and_context_config or {},
                    matching_config=matching_config,
                    retry_config=retry_config,
                    basic_context=basic_context,
                    context_settings=state["context_settings"],
                    context_manager=state["context_manager"],
                    ui=ui,
                )
                
                ui.print_info("Line range adjustment complete. Proceeding with processing...")
                logger.info("Line range adjustment complete. Using adjusted line ranges for processing.")
                state["global_chunking_method"] = "line_ranges.txt"
            
            # Confirm processing
            proceed = ui.display_processing_summary(
                state["files"],
                state["selected_schema_name"],
                state["global_chunking_method"],
                state["use_batch"],
                state["context_settings"],
            )
            
            if not proceed:
                ui.print_info("Processing cancelled by user.")
                logger.info("User cancelled processing.")
                return
            
            # Break out of loop to start processing
            break
    
    # Process files
    ui.print_section_header("Starting Processing")
    logger.info(f"Processing {len(state['files'])} file(s) with schema '{state['selected_schema_name']}'.")
    
    tasks = []
    inject_schema = model_config.get("transcription_model", {}).get("inject_schema_into_prompt", True)
    
    for file_path in state["files"]:
        tasks.append(
            file_processor.process_file(
                file_path=file_path,
                use_batch=state["use_batch"],
                selected_schema=state["selected_schema"],
                prompt_template=prompt_template,
                schema_name=state["selected_schema_name"],
                inject_schema=inject_schema,
                schema_paths=schemas_paths.get(state["selected_schema_name"], {}),
                global_chunking_method=state["global_chunking_method"],
                context_settings=state["context_settings"],
                context_manager=state["context_manager"],
                ui=ui,
            )
        )
    await asyncio.gather(*tasks)
    
    # Final summary
    ui.print_section_header("Processing Complete")
    
    if state["use_batch"]:
        ui.print_success("Batch processing jobs have been submitted")
        ui.print_info("To check batch status, run: python main/check_batches.py")
        logger.info("Batch jobs submitted successfully.")
    else:
        ui.print_success("All selected files have been processed")
        logger.info("Processing completed successfully.")
    
    ui.console_print(f"\n{ui.BOLD}Thank you for using ChronoMiner!{ui.RESET}\n")


async def _run_cli_mode(
    config_loader,
    paths_config: Dict,
    model_config: Dict,
    chunking_and_context_config: Dict,
    schemas_paths: Dict,
) -> None:
    """Run text processing in CLI mode."""
    parser = create_process_parser()
    args = parser.parse_args()
    
    logger.info("Starting ChronoMiner processing workflow (CLI Mode).")
    
    # Validate required arguments
    if not args.schema:
        parser.error("--schema is required in CLI mode")
    if not args.input:
        parser.error("--input is required in CLI mode")
    
    config_manager = ConfigManager(config_loader)
    
    # Validate paths
    try:
        config_manager.validate_paths(paths_config)
    except Exception as e:
        logger.error(f"Path validation error: {e}")
        print(f"[ERROR] Path validation failed: {e}")
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
    
    # Get schema
    try:
        schema_manager = load_schema_manager()
        available_schemas = schema_manager.get_available_schemas()
        if args.schema not in available_schemas:
            logger.error(f"Schema '{args.schema}' not found. Available: {list(available_schemas.keys())}")
            print(f"[ERROR] Schema '{args.schema}' not found")
            sys.exit(1)
        selected_schema = available_schemas[args.schema]
        selected_schema_name = args.schema
    except RuntimeError as exc:
        logger.error("Failed to load schema manager", exc_info=exc)
        print(f"[ERROR] Failed to load schemas: {exc}")
        sys.exit(1)
    
    # Load prompt template
    prompt_path = Path("prompts/structured_output_prompt.txt")
    try:
        prompt_template = load_prompt_template(prompt_path)
    except FileNotFoundError as exc:
        logger.error("Prompt template missing", exc_info=exc)
        print("[ERROR] Prompt template not found")
        sys.exit(1)
    
    # Process CLI arguments
    global_chunking_method = args.chunking if args.chunking else "auto"
    use_batch = args.batch if hasattr(args, 'batch') else False
    context_settings = {
        "use_additional_context": args.context if hasattr(args, 'context') else False,
        "use_default_context": args.context_source == "default" if hasattr(args, 'context_source') else True,
    }
    
    # Initialize context manager
    context_manager = prepare_context_manager(context_settings)
    
    # Resolve input path and get files
    input_path = resolve_path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        print(f"[ERROR] Input path not found: {input_path}")
        sys.exit(1)
    
    files = get_files_from_path(input_path, pattern="*.txt", exclude_patterns=["*_line_ranges.txt", "*_context.txt"])
    
    if not files:
        logger.error(f"No files found at: {input_path}")
        print(f"[ERROR] No text files found at: {input_path}")
        sys.exit(1)
    
    logger.info(f"Found {len(files)} file(s) to process")
    if not args.quiet:
        print(f"[INFO] Processing {len(files)} file(s) with schema '{selected_schema_name}'")
    
    # Handle line range adjustment if requested
    if global_chunking_method == "adjust-line-ranges":
        basic_context = load_basic_context()
        matching_config = (chunking_and_context_config or {}).get("matching", {})
        retry_config = (chunking_and_context_config or {}).get("retry", {})
        
        logger.info("Running line range adjustment workflow...")
        if not args.quiet:
            print("[INFO] Adjusting line ranges...")
        
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
            ui=None,
        )
        
        logger.info("Line range adjustment complete")
        global_chunking_method = "line_ranges.txt"
    
    # Process files
    logger.info(f"Processing {len(files)} file(s)")
    
    tasks = []
    inject_schema = model_config.get("transcription_model", {}).get("inject_schema_into_prompt", True)
    
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
                ui=None,
            )
        )
    await asyncio.gather(*tasks)
    
    # Final summary
    logger.info("Processing complete")
    if not args.quiet:
        if use_batch:
            print("[SUCCESS] Batch processing jobs submitted")
            print("[INFO] Run 'python main/check_batches.py' to check status")
        else:
            print(f"[SUCCESS] Processed {len(files)} file(s)")


async def main() -> None:
    """Main entry point."""
    try:
        # Load configuration first to determine mode
        (
            config_loader,
            paths_config,
            model_config,
            chunking_and_context_config,
            schemas_paths,
        ) = load_core_resources()
        
        if should_use_interactive_mode(config_loader):
            await _run_interactive_mode(
                config_loader,
                paths_config,
                model_config,
                chunking_and_context_config,
                schemas_paths,
            )
        else:
            await _run_cli_mode(
                config_loader,
                paths_config,
                model_config,
                chunking_and_context_config,
                schemas_paths,
            )
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("\n[INFO] Operation cancelled by user")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unexpected error in main workflow", exc_info=exc)
        print(f"[ERROR] Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
