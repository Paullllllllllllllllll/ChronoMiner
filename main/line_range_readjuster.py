# main/line_range_readjuster.py

"""
Utility script to refine `_line_ranges.txt` files by aligning chunk boundaries
with semantic boundaries detected by the configured LLM.

The script can be used interactively or via CLI flags. It reads existing line
range files, gathers textual context around each range, and asks the LLM to
propose boundary adjustments. Adjustments are only accepted when the boundary
text can be matched verbatim within the inspected context.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from modules.core.logger import setup_logger
from modules.core.schema_manager import SchemaManager
from modules.core.context_manager import ContextManager
from modules.core.prompt_context import load_basic_context
from modules.core.token_tracker import get_token_tracker
from modules.ui.core import UserInterface
from modules.operations.line_ranges.readjuster import LineRangeReadjuster
from modules.core.workflow_utils import (
    collect_text_files,
    load_core_resources,
    load_schema_manager,
    prepare_context_manager,
)
from modules.cli.mode_detector import should_use_interactive_mode
from modules.config.loader import ConfigLoader

logger = setup_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adjust existing line range files so their boundaries align with "
            "semantic sections detected by the configured language model."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        help=(
            "Optional path to a text file or directory. When omitted, an "
            "interactive workflow is used to select files."
        ),
    )
    parser.add_argument(
        "--schema",
        type=str,
        help=(
            "Schema name to use as boundary type (e.g., BibliographicEntries). "
            "Required when using --path in non-interactive mode."
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        help="Number of surrounding lines to send to the model when searching for boundaries.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        help="Override the prompt template used when calling the model.",
    )
    parser.add_argument(
        "--use-additional-context",
        action="store_true",
        help="Include additional context (default or file-specific).",
    )
    parser.add_argument(
        "--use-default-context",
        action="store_true",
        help="When using additional context, prefer boundary-type-specific defaults from additional_context/.",
    )
    return parser.parse_args()


def _check_token_limit_enabled() -> bool:
    """Check if daily token limit is enabled in configuration."""
    config_loader = ConfigLoader()
    config_loader.load_configs()
    concurrency_config = config_loader.get_concurrency_config()
    token_limit_config = concurrency_config.get("daily_token_limit", {})
    return token_limit_config.get("enabled", False)


def _validate_schema_paths(
    schema_name: str,
    schemas_paths: Dict,
    ui: Optional[UserInterface] = None,
) -> bool:
    """
    Validate that a schema has input/output paths configured in paths_config.yaml.
    
    Args:
        schema_name: The selected schema name.
        schemas_paths: The schemas_paths dict from paths_config.yaml.
        ui: Optional UserInterface for formatted output (interactive mode).
    
    Returns:
        True if paths are configured, False otherwise.
    """
    if schema_name not in schemas_paths:
        error_msg = (
            f"Schema '{schema_name}' has no path configuration in config/paths_config.yaml. "
            f"Please add an entry under 'schemas_paths' with 'input' and 'output' paths."
        )
        logger.error(error_msg)
        if ui:
            ui.print_error(error_msg)
        else:
            print(f"[ERROR] {error_msg}")
        return False
    
    schema_config = schemas_paths[schema_name]
    input_path = schema_config.get("input")
    
    if not input_path:
        error_msg = (
            f"Schema '{schema_name}' has no 'input' path configured in config/paths_config.yaml."
        )
        logger.error(error_msg)
        if ui:
            ui.print_error(error_msg)
        else:
            print(f"[ERROR] {error_msg}")
        return False
    
    return True


def _check_and_wait_for_token_limit(ui: Optional[UserInterface] = None) -> bool:
    """
    Check if daily token limit is reached and wait until next day if needed.
    
    Args:
        ui: Optional UserInterface instance for user feedback.
    
    Returns:
        True if processing can continue, False if user cancelled wait.
    """
    token_tracker = get_token_tracker()
    
    if not token_tracker.enabled or not token_tracker.is_limit_reached():
        return True
    
    # Token limit reached - need to wait until next day
    stats = token_tracker.get_stats()
    reset_time = token_tracker.get_reset_time()
    seconds_until_reset = token_tracker.get_seconds_until_reset()
    
    logger.warning(
        f"Daily token limit reached: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
    )
    logger.info(
        f"Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"({seconds_until_reset // 3600}h {(seconds_until_reset % 3600) // 60}m) "
        "for token limit reset..."
    )
    
    if ui:
        ui.print_warning(
            f"\n⚠ Daily token limit reached: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
        )
        ui.print_info(
            f"Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} for daily reset "
            f"({seconds_until_reset // 3600}h {(seconds_until_reset % 3600) // 60}m remaining)"
        )
        ui.print_info("Press Ctrl+C to cancel and exit.")
    else:
        print(
            f"[WARNING] Daily token limit reached: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
        )
        print(
            f"[INFO] Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} for daily reset "
            f"({seconds_until_reset // 3600}h {(seconds_until_reset % 3600) // 60}m remaining)"
        )
        print("[INFO] Press Ctrl+C to cancel and exit.")
    
    try:
        # Sleep in smaller intervals to allow for interruption
        sleep_interval = 60  # Check every minute
        elapsed = 0
        
        while elapsed < seconds_until_reset:
            interval = min(sleep_interval, max(0, seconds_until_reset - elapsed))
            time.sleep(interval)
            elapsed += interval
            
            # Re-check if it's a new day
            if not token_tracker.is_limit_reached():
                logger.info("Token limit has been reset. Resuming processing.")
                if ui:
                    ui.print_success("Token limit has been reset. Resuming processing.")
                else:
                    print("[SUCCESS] Token limit has been reset. Resuming processing.")
                return True
        
        logger.info("Token limit has been reset. Resuming processing.")
        if ui:
            ui.print_success("\nToken limit has been reset. Resuming processing.")
        else:
            print("[SUCCESS] Token limit has been reset. Resuming processing.")
        return True
        
    except KeyboardInterrupt:
        logger.info("Wait cancelled by user (KeyboardInterrupt).")
        if ui:
            ui.print_warning("\nWait cancelled by user.")
        else:
            print("\n[INFO] Wait cancelled by user.")
        return False


def _resolve_line_ranges_file(text_file: Path) -> Optional[Path]:
    """Detect the line range file associated with ``text_file``."""
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


def _prompt_int(ui: Optional[UserInterface], message: str, default: int) -> int:
    if ui:
        response = ui.get_input(f"{message} (default: {default})", allow_back=False, allow_quit=False)
        if not response:
            return default
        try:
            value = int(response)
            return max(1, value)
        except ValueError:
            ui.print_warning("Invalid number provided; using default.")
            return default
    else:
        try:
            response = input(f"{message} (default: {default}): ").strip()
        except EOFError:
            response = ""
        if not response:
            return default
        try:
            value = int(response)
            return max(1, value)
        except ValueError:
            print("[WARN] Invalid number provided; using default.")
            return default


async def _adjust_files(
    *,
    text_files: Sequence[Path],
    model_config: Dict[str, any],
    context_window: int,
    prompt_path: Optional[Path],
    boundary_type: str,
    basic_context: Optional[str],
    context_settings: Optional[Dict[str, any]],
    context_manager: Optional[ContextManager],
    matching_config: Optional[Dict[str, any]],
    retry_config: Optional[Dict[str, any]],
    notifier,
    ui: Optional[UserInterface] = None,
) -> Tuple[List[Tuple[Path, Path]], List[Path], List[Tuple[Path, Exception]]]:
    readjuster = LineRangeReadjuster(
        model_config,
        context_window=context_window,
        prompt_path=prompt_path,
        matching_config=matching_config,
        retry_config=retry_config,
    )

    successes: List[Tuple[Path, Path]] = []
    skipped: List[Path] = []
    failures: List[Tuple[Path, Exception]] = []
    token_limit_enabled = _check_token_limit_enabled()

    for text_file in text_files:
        # Check token limit before processing each file
        if token_limit_enabled:
            if not _check_and_wait_for_token_limit(ui):
                logger.info(f"Processing stopped by user. Adjusted {len(successes)} file(s).")
                notifier(f"Processing stopped. Adjusted {len(successes)}/{len(text_files)} file(s).", "warning")
                break
        
        line_ranges_file = _resolve_line_ranges_file(text_file)
        if not line_ranges_file:
            notifier(f"Skipping {text_file.name}: no associated line range file found.", "warning")
            skipped.append(text_file)
            continue

        notifier(f"Adjusting line ranges for {text_file.name}...", "info")
        logger.info(f"Adjusting {text_file.name} (context window: {context_window}, boundary: {boundary_type})")
        
        try:
            await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=line_ranges_file,
                dry_run=False,
                boundary_type=boundary_type,
                basic_context=basic_context,
                context_settings=context_settings,
                context_manager=context_manager,
            )
            notifier(f"Successfully adjusted line ranges for {text_file.name}", "success")
            logger.info(f"Line ranges for {text_file.name} adjusted using {line_ranges_file.name}")
            successes.append((text_file, line_ranges_file))
            
            # Log token usage after each file if enabled
            if token_limit_enabled:
                token_tracker = get_token_tracker()
                stats = token_tracker.get_stats()
                logger.info(
                    f"Token usage after {text_file.name}: "
                    f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
                    f"({stats['usage_percentage']:.1f}%)"
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Error adjusting %s", text_file, exc_info=exc)
            notifier(f"Failed to adjust {text_file.name}: {exc}", "error")
            failures.append((text_file, exc))

    return successes, skipped, failures


async def main_async() -> None:
    """Main async entry point for line range readjustment."""
    # Load configuration first
    (
        config_loader,
        paths_config,
        model_config,
        chunking_and_context_config,
        schemas_paths,
    ) = load_core_resources()
    
    chunking_config = (chunking_and_context_config or {}).get("chunking", {})
    matching_config = (chunking_and_context_config or {}).get("matching", {})
    retry_config = (chunking_and_context_config or {}).get("retry", {})
    
    try:
        schema_manager = load_schema_manager()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}; cannot select schema for input paths.")
        sys.exit(1)
    
    default_context_window = int(chunking_config.get("line_range_context_window", 6) or 6)
    
    # Determine execution mode
    is_interactive = should_use_interactive_mode(config_loader)
    
    if is_interactive:
        # ============================================================
        # INTERACTIVE MODE
        # ============================================================
        await _run_interactive_mode(
            schema_manager=schema_manager,
            schemas_paths=schemas_paths,
            paths_config=paths_config,
            model_config=model_config,
            chunking_config=chunking_config,
            matching_config=matching_config,
            retry_config=retry_config,
            default_context_window=default_context_window,
        )
    else:
        # ============================================================
        # CLI MODE
        # ============================================================
        args = parse_arguments()
        await _run_cli_mode(
            args=args,
            schema_manager=schema_manager,
            schemas_paths=schemas_paths,
            model_config=model_config,
            chunking_config=chunking_config,
            matching_config=matching_config,
            retry_config=retry_config,
            default_context_window=default_context_window,
        )


async def _run_interactive_mode(
    schema_manager: SchemaManager,
    schemas_paths: Dict[str, any],
    paths_config: Dict[str, any],
    model_config: Dict[str, any],
    chunking_config: Dict[str, any],
    matching_config: Dict[str, any],
    retry_config: Dict[str, any],
    default_context_window: int,
) -> None:
    """Run line range readjustment in interactive mode with back navigation support."""
    ui = UserInterface(logger, use_colors=True)
    ui.display_banner()
    ui.print_section_header("Line Range Adjustment")
    
    # State machine for navigation
    # States: schema -> files -> context -> confirm
    current_step = "schema"
    state = {}
    
    while True:
        if current_step == "schema":
            result = ui.select_schema(schema_manager, allow_back=False)
            if result is None:
                ui.print_info("Schema selection cancelled.")
                return
            
            selected_schema, selected_schema_name = result
            state["selected_schema"] = selected_schema
            state["selected_schema_name"] = selected_schema_name
            state["boundary_type"] = selected_schema_name
            
            # Validate schema has paths configured
            if not _validate_schema_paths(selected_schema_name, schemas_paths, ui):
                logger.error(f"Exiting: No path configuration for schema '{selected_schema_name}'")
                sys.exit(1)
            
            # Load schema-specific basic context
            state["basic_context"] = load_basic_context(schema_name=selected_schema_name)
            logger.info(f"Loaded basic context for schema '{selected_schema_name}'")
            
            # Determine base directory (validated above, so schema_name is in schemas_paths)
            state["base_dir"] = Path(schemas_paths[selected_schema_name].get("input", ""))
            
            current_step = "files"
        
        elif current_step == "files":
            selected_files = ui.select_input_source(state["base_dir"], allow_back=True)
            if selected_files is None:
                current_step = "schema"
                continue
            if not selected_files:
                ui.print_warning("No files selected.")
                return
            state["selected_files"] = selected_files
            current_step = "context"
        
        elif current_step == "context":
            context_settings = ui.ask_additional_context_mode(allow_back=True)
            if context_settings is None:
                current_step = "files"
                continue
            state["context_settings"] = context_settings
            state["context_manager"] = prepare_context_manager(context_settings)
            current_step = "context_window"
        
        elif current_step == "context_window":
            context_window = _prompt_int(
                ui, 
                "Enter context window size (lines around boundaries)", 
                default_context_window
            )
            state["context_window"] = context_window
            
            # Get prompt override if any
            prompt_override = chunking_config.get("line_range_prompt_path")
            state["prompt_path"] = Path(prompt_override).resolve() if prompt_override else None
            
            current_step = "confirm"
        
        elif current_step == "confirm":
            # Display summary
            ui.print_section_header("Adjustment Configuration")
            ui.console_print(f"\n{ui.BOLD}Configuration:{ui.RESET}")
            ui.console_print(f"  Files to adjust: {len(state['selected_files'])}")
            ui.console_print(f"  Context window: {state['context_window']} lines")
            ui.console_print(f"  Boundary type: {state['boundary_type']}")
            if state["prompt_path"]:
                ui.console_print(f"  Prompt override: {state['prompt_path']}")
            
            if state["context_settings"].get("use_additional_context", False):
                context_source = "Default boundary-type-specific" if state["context_settings"].get("use_default_context", False) else "File-specific"
            else:
                context_source = "None"
            ui.console_print(f"  Additional context: {context_source}")
            
            if not ui.confirm("\nProceed with line range adjustment?", default=True):
                ui.print_info("Operation cancelled by user.")
                return
            
            # Break out of loop to start adjustment
            break
    
    ui.print_section_header("Adjusting Line Ranges")
    
    # Display initial token usage statistics if enabled
    if _check_token_limit_enabled():
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
    
    # UI-aware notifier
    def ui_notifier(msg: str, level: str = "info"):
        if level == "success":
            ui.print_success(msg)
        elif level == "error":
            ui.print_error(msg)
        elif level == "warning":
            ui.print_warning(msg)
        else:
            ui.print_info(msg)
    
    # Perform adjustments
    successes, skipped, failures = await _adjust_files(
        text_files=state["selected_files"],
        model_config=model_config,
        context_window=state["context_window"],
        prompt_path=state["prompt_path"],
        boundary_type=state["boundary_type"],
        basic_context=state["basic_context"],
        context_settings=state["context_settings"],
        context_manager=state["context_manager"],
        matching_config=matching_config,
        retry_config=retry_config,
        notifier=ui_notifier,
        ui=ui,
    )
    
    # Display summary
    ui.print_section_header("Adjustment Summary")
    ui.print_success(f"Successfully adjusted: {len(successes)} file(s)")
    if skipped:
        ui.print_warning(f"Skipped (no line ranges): {len(skipped)} file(s)")
    if failures:
        ui.print_error(f"Failed: {len(failures)} file(s)")
    
    if skipped:
        ui.print_subsection_header("Files with no line range file")
        for skipped_file in skipped:
            ui.console_print(f"  • {skipped_file.name}")
    if failures:
        ui.print_subsection_header("Files with errors")
        for failed_file, error in failures:
            ui.console_print(f"  • {failed_file.name}: {error}")
    
    # Final token usage statistics
    if _check_token_limit_enabled():
        token_tracker = get_token_tracker()
        stats = token_tracker.get_stats()
        logger.info(
            f"Final token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%)"
        )
        ui.print_info(
            f"\nFinal daily token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%)"
        )


async def _run_cli_mode(
    args: argparse.Namespace,
    schema_manager: SchemaManager,
    schemas_paths: Dict[str, any],
    model_config: Dict[str, any],
    chunking_config: Dict[str, any],
    matching_config: Dict[str, any],
    retry_config: Dict[str, any],
    default_context_window: int,
) -> None:
    """Run line range readjustment in CLI mode."""
    logger.info("Starting line range readjustment (CLI Mode)")
    
    # Validate and resolve input path
    if not args.path:
        print("[ERROR] --path is required in CLI mode")
        sys.exit(1)
    
    target = args.path.expanduser().resolve()
    if not target.exists():
        logger.error("Specified path does not exist: %s", target)
        print(f"[ERROR] Path does not exist: {target}")
        sys.exit(1)
    
    selected_files = collect_text_files(target)
    if not selected_files:
        print(f"[WARN] No eligible text files found under {target}.")
        sys.exit(0)
    
    # Get boundary type from --schema argument
    if not args.schema:
        print("[ERROR] Schema name is required when using --path in CLI mode.")
        sys.exit(1)
    
    available_schemas = schema_manager.get_available_schemas()
    if args.schema not in available_schemas:
        print(f"[ERROR] Schema '{args.schema}' not found.")
        print(f"[INFO] Available schemas: {list(available_schemas.keys())}")
        sys.exit(1)
    
    boundary_type = args.schema
    
    # Validate schema has paths configured
    if not _validate_schema_paths(boundary_type, schemas_paths):
        logger.error(f"Exiting: No path configuration for schema '{boundary_type}'")
        sys.exit(1)
    
    print(f"[INFO] Using boundary type: {boundary_type}")
    
    # Load schema-specific basic context
    basic_context = load_basic_context(schema_name=boundary_type)
    logger.info(f"Loaded basic context for schema '{boundary_type}'")
    
    # Setup context
    use_additional = bool(args.use_additional_context or args.use_default_context)
    context_settings = {
        "use_additional_context": use_additional,
        "use_default_context": bool(args.use_default_context),
    }
    context_manager = prepare_context_manager(context_settings)
    
    # Get configurations
    context_window = max(1, args.context_window or default_context_window)
    prompt_override = args.prompt_path or chunking_config.get("line_range_prompt_path")
    prompt_path: Optional[Path] = Path(prompt_override).resolve() if prompt_override else None
    
    # Simple notifier for CLI mode
    def cli_notifier(msg: str, level: str = "info"):
        prefixes = {"success": "[SUCCESS]", "error": "[ERROR]", "warning": "[WARN]", "info": "[INFO]"}
        print(f"{prefixes.get(level, '[INFO]')} {msg}")
    
    # Display configuration
    print("\n" + "=" * 80)
    print(f"  LINE RANGE READJUSTMENT")
    print("=" * 80)
    print(f"Selected {len(selected_files)} file(s) for adjustment.")
    print(f"Context window: {context_window}")
    print(f"Boundary type: {boundary_type}")
    if prompt_path:
        print(f"Prompt override: {prompt_path}")
    
    if context_settings.get("use_additional_context", False):
        context_source = "Default boundary-type-specific" if context_settings.get("use_default_context", False) else "File-specific"
    else:
        context_source = "None"
    print(f"Additional context: {context_source}")
    
    # Display initial token usage statistics if enabled
    if _check_token_limit_enabled():
        token_tracker = get_token_tracker()
        stats = token_tracker.get_stats()
        logger.info(
            f"Token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%) - "
            f"{stats['tokens_remaining']:,} tokens remaining today"
        )
        print(
            f"[INFO] Daily token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%)"
        )
    
    # Perform adjustments
    successes, skipped, failures = await _adjust_files(
        text_files=selected_files,
        model_config=model_config,
        context_window=context_window,
        prompt_path=prompt_path,
        boundary_type=boundary_type,
        basic_context=basic_context,
        context_settings=context_settings,
        context_manager=context_manager,
        matching_config=matching_config,
        retry_config=retry_config,
        notifier=cli_notifier,
        ui=None,
    )
    
    # Display summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"Successful adjustments: {len(successes)}")
    print(f"Skipped (missing line ranges): {len(skipped)}")
    print(f"Failures: {len(failures)}")
    
    if skipped:
        print("\nFiles with no associated line range file:")
        for skipped_file in skipped:
            print(f"  - {skipped_file}")
    if failures:
        print("\nFiles that encountered errors:")
        for failed_file, error in failures:
            print(f"  - {failed_file}: {error}")
    
    # Final token usage statistics
    if _check_token_limit_enabled():
        token_tracker = get_token_tracker()
        stats = token_tracker.get_stats()
        logger.info(
            f"Final token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%)"
        )
        print(
            f"\n[INFO] Final daily token usage: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} "
            f"({stats['usage_percentage']:.1f}%)"
        )


def main() -> None:
    """Main entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Line range readjustment cancelled by user.")
        print("\n✓ Operation cancelled by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unexpected error in line_range_readjuster", exc_info=exc)
        print(f"\n[ERROR] Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
