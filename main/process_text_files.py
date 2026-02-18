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
import time
from pathlib import Path
from typing import Dict, List, Optional


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.cli.args_parser import create_process_parser, get_files_from_path, resolve_path
from modules.cli.execution_framework import AsyncDualModeScript
from modules.config.manager import ConfigManager, ConfigValidationError
from modules.core.logger import setup_logger
from modules.core.token_tracker import (
    get_token_tracker,
    check_token_limit_enabled,
    check_and_wait_for_token_limit,
)
from modules.core.workflow_utils import (
    load_core_resources,
    load_schema_manager,
    validate_schema_paths,
)
from modules.llm.prompt_utils import load_prompt_template
from modules.core.chunking_service import ChunkSlice
from modules.operations.extraction.file_processor import FileProcessorRefactored as FileProcessor
from modules.operations.line_ranges.readjuster import LineRangeReadjuster
from modules.ui.core import UserInterface

# Import line range generation functions from sibling script
from main.generate_line_ranges import generate_line_ranges_for_file, write_line_ranges_file

# Initialize logger
logger = setup_logger(__name__)


async def _adjust_line_ranges_workflow(
    files: List[Path],
    selected_schema_name: str,
    model_config: Dict,
    chunking_config: Dict,
    matching_config: Dict,
    retry_config: Dict,
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
            # Auto-generate line ranges file
            if ui:
                ui.print_info(f"Generating line ranges for {text_file.name}...")
            logger.info(f"Line ranges file not found for {text_file.name}, generating...")
            
            try:
                # Get model configuration
                model_name = model_config.get("transcription_model", {}).get("name", "o3-mini")
                tokens_per_chunk = chunking_config.get("chunking", {}).get("default_tokens_per_chunk", 7500)
                
                # Generate line ranges
                line_ranges = generate_line_ranges_for_file(
                    text_file=text_file,
                    default_tokens_per_chunk=tokens_per_chunk,
                    model_name=model_name
                )
                
                # Write line ranges file
                line_ranges_file = write_line_ranges_file(text_file, line_ranges)
                
                if ui:
                    ui.print_success(f"Generated {line_ranges_file.name}")
                logger.info(f"Successfully generated line ranges file: {line_ranges_file}")
            except Exception as e:
                logger.exception(f"Failed to generate line ranges for {text_file}", exc_info=e)
                if ui:
                    ui.print_error(f"Failed to generate line ranges for {text_file.name}: {e}")
                continue
        
        if ui:
            ui.print_info(f"Adjusting line ranges for {text_file.name}...")
        
        try:
            adjusted_ranges = await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=line_ranges_file,
                boundary_type=selected_schema_name,
            )
            
            if ui:
                ui.print_success(f"Adjusted {len(adjusted_ranges)} range(s) for {text_file.name}")
        except Exception as e:
            logger.exception(f"Failed to adjust line ranges for {text_file}", exc_info=e)
            if ui:
                ui.print_error(f"Failed to adjust {text_file.name}: {e}")


def _count_existing_outputs(
    files: List[Path],
    paths_config: Dict,
    schema_name: str,
    schemas_paths: Dict,
) -> int:
    """
    Count how many output JSON files already exist for the given input files.

    :param files: List of input file paths
    :param paths_config: Paths configuration dictionary
    :param schema_name: Name of the selected schema
    :param schemas_paths: Schema-specific paths dictionary
    :return: Number of input files that already have a corresponding output file
    """
    use_input_as_output = paths_config.get("general", {}).get("input_paths_is_output_path", False)
    count = 0
    for file_path in files:
        if use_input_as_output:
            output_path = file_path.parent / f"{file_path.stem}_output.json"
        else:
            schema_cfg = schemas_paths.get(schema_name, {})
            output_dir_str = schema_cfg.get("output", "")
            if not output_dir_str:
                continue
            output_path = Path(output_dir_str) / f"{file_path.stem}_output.json"
        if output_path.exists():
            count += 1
    return count


async def _run_interactive_mode(
    config_loader,
    paths_config: Dict,
    model_config: Dict,
    chunking_and_context_config: Dict,
    schemas_paths: Dict,
) -> None:
    """Run text processing in interactive mode with back navigation support."""
    ui = UserInterface(logger, use_colors=True)
    
    # Load configuration
    ui.print_info("Loading configuration...")
    logger.info("Starting ChronoMiner processing workflow (Interactive Mode).")
    
    config_manager = ConfigManager(config_loader)
    
    # Validate paths
    try:
        config_manager.validate_paths(paths_config)
    except ConfigValidationError as e:
        ui.print_error(f"Path validation failed: {e}")
        logger.error(f"Path validation error: {e}")
        sys.exit(1)
    except Exception as e:
        ui.print_error(f"Unexpected validation error: {e}")
        logger.error(f"Unexpected validation error: {e}")
        sys.exit(1)
    
    # Load other configs
    chunking_config = {
        "chunking": (chunking_and_context_config or {}).get("chunking", {})
    }
    concurrency_config = config_loader.get_concurrency_config()
    
    # Initialize file processor
    file_processor = FileProcessor(
        paths_config=paths_config,
        model_config=model_config,
        chunking_config=chunking_config,
        concurrency_config=concurrency_config,
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
    # States: schema -> chunking -> batch -> resume -> context -> chunk_slice -> files -> confirm
    current_step = "schema"
    state = {}
    
    while True:
        if current_step == "schema":
            result = ui.select_schema(schema_manager, allow_back=False)
            if result is None:
                ui.print_info("Schema selection cancelled.")
                return
            state["selected_schema"], state["selected_schema_name"] = result
            
            # Validate schema has paths configured
            if not validate_schema_paths(state["selected_schema_name"], schemas_paths, ui):
                logger.error(f"Exiting: No path configuration for schema '{state['selected_schema_name']}'")
                sys.exit(1)
            
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
            current_step = "resume"
        
        elif current_step == "resume":
            # CM-9: use select_option with allow_back=True so back returns to batch step
            resume_choice = ui.select_option(
                "Resume mode?",
                [
                    ("resume", "Skip / resume partial - skip fully processed files, resume partial ones"),
                    ("reprocess", "Reprocess everything - process all files from scratch"),
                ],
                allow_back=True,
            )
            if resume_choice is None:
                current_step = "batch"
                continue
            state["resume"] = (resume_choice == "resume")
            current_step = "context"

        elif current_step == "context":
            # CM-8: explicit context-selection step
            context_result = ui.ask_context_selection(allow_back=True)
            if context_result is None:
                current_step = "resume"
                continue
            state["context_override"] = context_result
            current_step = "chunk_slice"

        elif current_step == "chunk_slice":
            chunk_slice = ui.ask_chunk_slice(allow_back=True)
            if chunk_slice is None:
                current_step = "context"
                continue
            state["chunk_slice"] = chunk_slice
            current_step = "files"

        elif current_step == "files":
            # Determine input directory
            if state["selected_schema_name"] in schemas_paths:
                raw_text_dir = Path(schemas_paths[state["selected_schema_name"]].get("input"))
            else:
                raw_text_dir = Path(paths_config.get("input_paths", {}).get("raw_text_dir", ""))

            files = ui.select_input_source(raw_text_dir, allow_back=True)
            if files is None:
                current_step = "chunk_slice"
                continue
            state["files"] = files
            current_step = "confirm"
        
        elif current_step == "confirm":
            # CM-10: count existing output files before showing confirmation
            existing_output_count = _count_existing_outputs(
                state["files"],
                paths_config,
                state["selected_schema_name"],
                schemas_paths,
            )

            # Handle line range adjustment workflow if selected
            if state["global_chunking_method"] == "adjust-line-ranges":
                matching_config = (chunking_and_context_config or {}).get("matching", {})
                retry_config = (chunking_and_context_config or {}).get("retry", {})
                
                await _adjust_line_ranges_workflow(
                    files=state["files"],
                    selected_schema_name=state["selected_schema_name"],
                    model_config=model_config,
                    chunking_config=chunking_and_context_config or {},
                    matching_config=matching_config,
                    retry_config=retry_config,
                    ui=ui,
                )
                
                ui.print_info("Line range adjustment complete. Proceeding with processing...")
                logger.info("Line range adjustment complete. Using adjusted line ranges for processing.")
                state["global_chunking_method"] = "line_ranges.txt"
            
            # Derive context_mode string for the summary display
            _ctx = state.get("context_override") or {}
            _ctx_mode = _ctx.get("mode", "auto")
            if _ctx_mode == "manual" and _ctx.get("path"):
                _ctx_mode = str(_ctx["path"])

            # Confirm processing
            proceed = ui.display_processing_summary(
                state["files"],
                state["selected_schema_name"],
                state["global_chunking_method"],
                state["use_batch"],
                model_config=model_config,
                paths_config=paths_config,
                concurrency_config=concurrency_config,
                chunk_slice=state.get("chunk_slice"),
                context_mode=_ctx_mode,
                existing_output_count=existing_output_count,
            )
            
            if not proceed:
                ui.print_info("Processing cancelled by user.")
                logger.info("User cancelled processing.")
                return
            
            # Break out of loop to start processing
            break
    
    # Display initial token usage statistics if enabled
    if check_token_limit_enabled():
        token_tracker = get_token_tracker()
        stats = token_tracker.get_stats()
        logger.info(
            f"Token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%) - "
            f"{stats['tokens_remaining']:,} tokens remaining today"
        )
        ui.print_info(
            f"Daily token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%)"
        )
    
    # Track processing time
    import time
    start_time = time.time()
    processed_count = 0
    failed_count = 0
    
    # Process files
    ui.print_section_header("Starting Processing")
    logger.info(f"Processing {len(state['files'])} file(s) with schema '{state['selected_schema_name']}'.")
    
    # Process files sequentially if token limiting is enabled (for better control)
    # Otherwise process concurrently for speed
    inject_schema = model_config.get("transcription_model", {}).get("inject_schema_into_prompt", True)
    token_limit_enabled = check_token_limit_enabled()
    
    if token_limit_enabled and not state["use_batch"]:
        # Sequential processing with token limit checks
        processed_count = 0
        for index, file_path in enumerate(state["files"], start=1):
            # Check token limit before starting each file
            if not check_and_wait_for_token_limit(ui):
                logger.info(f"Processing stopped by user. Processed {processed_count}/{len(state['files'])} files.")
                ui.print_warning(f"\nProcessing stopped. Completed {processed_count}/{len(state['files'])} files.")
                break
            
            # Process this file
            await file_processor.process_file(
                file_path=file_path,
                use_batch=state["use_batch"],
                selected_schema=state["selected_schema"],
                prompt_template=prompt_template,
                schema_name=state["selected_schema_name"],
                inject_schema=inject_schema,
                schema_paths=schemas_paths.get(state["selected_schema_name"], {}),
                global_chunking_method=state["global_chunking_method"],
                ui=ui,
                resume=state.get("resume", False),
                chunk_slice=state.get("chunk_slice"),
                context_override=state.get("context_override"),
            )
            processed_count += 1
            
            # Log token usage after each file
            token_tracker = get_token_tracker()
            stats = token_tracker.get_stats()
            logger.info(
                f"Token usage after file {index}/{len(state['files'])}: "
                f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )
    else:
        # Concurrent processing (original behavior)
        tasks = []
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
                    ui=ui,
                    resume=state.get("resume", False),
                    chunk_slice=state.get("chunk_slice"),
                    context_override=state.get("context_override"),
                )
            )
        # CM-11: return_exceptions=True ensures all finally blocks complete on cancellation
        _gather_results = await asyncio.gather(*tasks, return_exceptions=True)
        for _exc in _gather_results:
            if isinstance(_exc, Exception):
                logger.error("Error in concurrent file processing: %s", _exc, exc_info=_exc)
    
    # Calculate duration and set counts for non-sequential processing
    duration_seconds = time.time() - start_time
    if not (token_limit_enabled and not state["use_batch"]):
        # For concurrent/batch mode, assume all processed successfully
        processed_count = len(state["files"])
    
    # Display completion summary
    ui.display_completion_summary(
        processed_count=processed_count,
        failed_count=failed_count,
        use_batch=state["use_batch"],
        duration_seconds=duration_seconds,
        paths_config=paths_config,
        selected_schema_name=state["selected_schema_name"],
    )
    
    # Final token usage statistics
    if check_token_limit_enabled():
        token_tracker = get_token_tracker()
        stats = token_tracker.get_stats()
        logger.info(
            f"Final token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%)"
        )
        ui.print_info(
            f"Final daily token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%)"
        )


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
    except ConfigValidationError as e:
        logger.error(f"Path validation error: {e}")
        print(f"[ERROR] Path validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}")
        print(f"[ERROR] Unexpected validation error: {e}")
        sys.exit(1)
    
    # Load other configs
    chunking_config = {
        "chunking": (chunking_and_context_config or {}).get("chunking", {})
    }
    concurrency_config = config_loader.get_concurrency_config()
    
    # Initialize file processor
    file_processor = FileProcessor(
        paths_config=paths_config,
        model_config=model_config,
        chunking_config=chunking_config,
        concurrency_config=concurrency_config,
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
    
    # Validate schema has paths configured
    if not validate_schema_paths(selected_schema_name, schemas_paths):
        logger.error(f"Exiting: No path configuration for schema '{selected_schema_name}'")
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
    use_resume = getattr(args, "resume", False) and not getattr(args, "force", False)
    
    # Build chunk slice from CLI args
    chunk_slice = None
    first_n = getattr(args, "first_n_chunks", None)
    last_n = getattr(args, "last_n_chunks", None)
    if first_n is not None:
        chunk_slice = ChunkSlice(first_n=first_n)
    elif last_n is not None:
        chunk_slice = ChunkSlice(last_n=last_n)
    
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
    
    # Display initial token usage statistics if enabled
    if check_token_limit_enabled():
        token_tracker = get_token_tracker()
        stats = token_tracker.get_stats()
        logger.info(
            f"Token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%) - "
            f"{stats['tokens_remaining']:,} tokens remaining today"
        )
        if not args.quiet:
            print(
                f"[INFO] Daily token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )
    
    # Handle line range adjustment if requested
    if global_chunking_method == "adjust-line-ranges":
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
            ui=None,
        )
        
        logger.info("Line range adjustment complete")
        global_chunking_method = "line_ranges.txt"
    
    # Process files
    logger.info(f"Processing {len(files)} file(s)")
    
    # Process files sequentially if token limiting is enabled (for better control)
    # Otherwise process concurrently for speed
    inject_schema = model_config.get("transcription_model", {}).get("inject_schema_into_prompt", True)
    token_limit_enabled = check_token_limit_enabled()
    
    # Allow overriding output directory per invocation (useful for smoke tests)
    effective_schema_paths = dict(schemas_paths.get(selected_schema_name, {}) or {})
    if getattr(args, "output", None):
        effective_schema_paths["output"] = str(resolve_path(args.output))

    if token_limit_enabled and not use_batch:
        # Sequential processing with token limit checks
        processed_count = 0
        for index, file_path in enumerate(files, start=1):
            # Check token limit before starting each file
            if not check_and_wait_for_token_limit(ui=None):
                logger.info(f"Processing stopped by user. Processed {processed_count}/{len(files)} files.")
                if not args.quiet:
                    print(f"[WARNING] Processing stopped. Completed {processed_count}/{len(files)} files.")
                break
            
            # Process this file
            await file_processor.process_file(
                file_path=file_path,
                use_batch=use_batch,
                selected_schema=selected_schema,
                prompt_template=prompt_template,
                schema_name=selected_schema_name,
                inject_schema=inject_schema,
                schema_paths=effective_schema_paths,
                global_chunking_method=global_chunking_method,
                ui=None,
                resume=use_resume,
                chunk_slice=chunk_slice,
            )
            processed_count += 1
            
            # Log token usage after each file
            token_tracker = get_token_tracker()
            stats = token_tracker.get_stats()
            logger.info(
                f"Token usage after file {index}/{len(files)}: "
                f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )
    else:
        # Concurrent processing (original behavior)
        tasks = []
        for file_path in files:
            tasks.append(
                file_processor.process_file(
                    file_path=file_path,
                    use_batch=use_batch,
                    selected_schema=selected_schema,
                    prompt_template=prompt_template,
                    schema_name=selected_schema_name,
                    inject_schema=inject_schema,
                    schema_paths=effective_schema_paths,
                    global_chunking_method=global_chunking_method,
                    ui=None,
                    resume=use_resume,
                    chunk_slice=chunk_slice,
                )
            )
        # CM-11: return_exceptions=True ensures all finally blocks complete on cancellation
        _cli_results = await asyncio.gather(*tasks, return_exceptions=True)
        for _exc in _cli_results:
            if isinstance(_exc, Exception):
                logger.error("Error in concurrent file processing: %s", _exc, exc_info=_exc)

    # Final summary
    logger.info("Processing complete")
    if not args.quiet:
        if use_batch:
            print("[SUCCESS] Batch processing jobs submitted")
            print("[INFO] Run 'python main/check_batches.py' to check status")
        else:
            print(f"[SUCCESS] Processed {len(files)} file(s)")
    
    # Final token usage statistics
    if check_token_limit_enabled():
        token_tracker = get_token_tracker()
        stats = token_tracker.get_stats()
        logger.info(
            f"Final token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%)"
        )
        if not args.quiet:
            print(
                f"[INFO] Final daily token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                f"({stats['usage_percentage']:.1f}%)"
            )


class ProcessTextFilesScript(AsyncDualModeScript):
    """Main script for processing text files with schema-based structured data extraction."""
    
    def __init__(self):
        super().__init__("process_text_files")
    
    def create_argument_parser(self):
        """Create argument parser for CLI mode."""
        return create_process_parser()
    
    async def run_interactive(self) -> None:
        """Run text processing in interactive mode."""
        await _run_interactive_mode(
            self.config_loader,
            self.paths_config,
            self.model_config,
            self.chunking_and_context_config,
            self.schemas_paths,
        )
    
    async def run_cli(self, args) -> None:
        """Run text processing in CLI mode."""
        await _run_cli_mode(
            self.config_loader,
            self.paths_config,
            self.model_config,
            self.chunking_and_context_config,
            self.schemas_paths,
        )


def main() -> None:
    """Main entry point."""
    ProcessTextFilesScript().execute()


if __name__ == "__main__":
    main()
