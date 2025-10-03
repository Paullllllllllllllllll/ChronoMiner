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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from modules.core.logger import setup_logger
from modules.core.schema_manager import SchemaManager
from modules.core.context_manager import ContextManager
from modules.core.prompt_context import load_basic_context
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

    for text_file in text_files:
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
    
    basic_context = load_basic_context()
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
            basic_context=basic_context,
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
            model_config=model_config,
            chunking_config=chunking_config,
            matching_config=matching_config,
            retry_config=retry_config,
            basic_context=basic_context,
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
    basic_context: Optional[str],
    default_context_window: int,
) -> None:
    """Run line range readjustment in interactive mode with back navigation support."""
    ui = UserInterface(logger)
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
            
            # Determine base directory
            if selected_schema_name in schemas_paths:
                state["base_dir"] = Path(schemas_paths[selected_schema_name].get("input", ""))
            else:
                state["base_dir"] = Path(paths_config.get("input_paths", {}).get("raw_text_dir", ""))
            
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
        basic_context=basic_context,
        context_settings=state["context_settings"],
        context_manager=state["context_manager"],
        matching_config=matching_config,
        retry_config=retry_config,
        notifier=ui_notifier,
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


async def _run_cli_mode(
    args: argparse.Namespace,
    schema_manager: SchemaManager,
    model_config: Dict[str, any],
    chunking_config: Dict[str, any],
    matching_config: Dict[str, any],
    retry_config: Dict[str, any],
    basic_context: Optional[str],
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
    print(f"[INFO] Using boundary type: {boundary_type}")
    
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
