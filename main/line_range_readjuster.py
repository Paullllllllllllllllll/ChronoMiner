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
        "--dry-run",
        action="store_true",
        help="Preview adjustments without rewriting the line range files.",
    )
    parser.add_argument(
        "--boundary-type",
        type=str,
        help="Name of the semantic boundary type to detect.",
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


def _prompt_boundary_type_cli(*, default: Optional[str] = None) -> str:
    """Prompt the user to enter a boundary type name."""
    prompt_suffix = f" [default: {default}]" if default else ""
    
    while True:
        try:
            selection = input(f"Enter the semantic boundary type name{prompt_suffix}: ").strip()
        except EOFError:
            selection = ""

        if not selection and default:
            return default
        if selection:
            return selection

        print("[WARN] Boundary type cannot be empty. Please try again.")


def _prompt_int(ui: Optional[UserInterface], message: str, default: int) -> int:
    if ui:
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
        if ui:
            ui.console_print("[WARN] Invalid number provided; using default.")
        return default


def _prompt_yes_no(ui: Optional[UserInterface], message: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    if ui:
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
    if ui:
        ui.console_print(f"[WARN] Unrecognized response '{response}'; using default.")
    return default


async def _adjust_files(
    *,
    text_files: Sequence[Path],
    model_config: Dict[str, any],
    context_window: int,
    prompt_path: Optional[Path],
    dry_run: bool,
    boundary_type: str,
    basic_context: Optional[str],
    context_settings: Optional[Dict[str, any]],
    context_manager: Optional[ContextManager],
    notifier,
) -> Tuple[List[Tuple[Path, Path]], List[Path], List[Tuple[Path, Exception]]]:
    readjuster = LineRangeReadjuster(
        model_config,
        context_window=context_window,
        prompt_path=prompt_path,
    )

    successes: List[Tuple[Path, Path]] = []
    skipped: List[Path] = []
    failures: List[Tuple[Path, Exception]] = []

    for text_file in text_files:
        line_ranges_file = _resolve_line_ranges_file(text_file)
        if not line_ranges_file:
            notifier(
                f"[WARN] Skipping {text_file.name}: no associated line range file found.")
            skipped.append(text_file)
            continue

        notifier(
            f"[INFO] Adjusting line ranges for {text_file.name} (context window: {context_window}, boundary: {boundary_type})...")
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
            notifier(
                f"[SUCCESS] Line ranges for {text_file.name} {status} using {line_ranges_file.name}.")
            successes.append((text_file, line_ranges_file))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Error adjusting %s", text_file, exc_info=exc)
            notifier(
                f"[ERROR] Failed to adjust {text_file.name}: {exc}")
            failures.append((text_file, exc))

    return successes, skipped, failures


async def main_async() -> None:
    args = parse_arguments()

    (
        config_loader,
        paths_config,
        model_config,
        chunking_and_context_config,
        schemas_paths,
    ) = load_core_resources()
    chunking_config = (chunking_and_context_config or {}).get("chunking", {})

    try:
        schema_manager = load_schema_manager()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}; cannot select schema for input paths.")
        sys.exit(1)

    basic_context = load_basic_context()
    context_settings: Dict[str, any] = {}
    context_manager: Optional[ContextManager] = None

    default_context_window = int(chunking_config.get("line_range_context_window", 6) or 6)
    prompt_override = args.prompt_path or chunking_config.get("line_range_prompt_path")
    prompt_path: Optional[Path] = Path(prompt_override).resolve() if prompt_override else None

    ui: Optional[UserInterface] = None
    selected_files: List[Path] = []
    boundary_type: Optional[str] = None

    if args.path:
        target = args.path.expanduser().resolve()
        if not target.exists():
            logger.error("Specified path does not exist: %s", target)
            print(f"[ERROR] Path does not exist: {target}")
            sys.exit(1)
        selected_files = collect_text_files(target)
        if not selected_files:
            print(f"[WARN] No eligible text files found under {target}.")
            sys.exit(0)

        if args.boundary_type:
            boundary_type = args.boundary_type
        else:
            boundary_type = _prompt_boundary_type_cli(default=None)

        use_additional = bool(args.use_additional_context or args.use_default_context)
        context_settings = {
            "use_additional_context": use_additional,
            "use_default_context": bool(args.use_default_context),
        }
        context_manager = prepare_context_manager(context_settings)
    else:
        ui = UserInterface(logger)
        ui.display_banner()

        # Select schema to determine input directory and boundary type
        selected_schema, selected_schema_name = ui.select_schema(schema_manager)

        if selected_schema_name in schemas_paths:
            base_dir = Path(schemas_paths[selected_schema_name].get("input", ""))
        else:
            base_dir = Path(paths_config.get("input_paths", {}).get("raw_text_dir", ""))
        selected_files = ui.select_input_source(base_dir)

        use_same_schema = _prompt_yes_no(
            ui,
            f"Use schema '{selected_schema_name}' as the semantic boundary type?",
            default=True,
        )
        if use_same_schema:
            boundary_type = selected_schema_name
        else:
            boundary_type = _prompt_boundary_type_cli()

        context_settings = ui.ask_additional_context_mode()
        context_manager = prepare_context_manager(context_settings)

    if not selected_files:
        if ui:
            ui.console_print("[WARN] No files selected; exiting.")
        return

    context_window = max(1, args.context_window or default_context_window)
    if ui and args.context_window is None:
        context_window = _prompt_int(
            ui,
            f"Enter context window size",
            default_context_window,
        )

    dry_run = bool(args.dry_run)
    if ui and not args.dry_run:
        dry_run = _prompt_yes_no(
            ui,
            "Perform a dry run (do not overwrite line range files)?",
            default=False,
        )

    notifier = ui.console_print if ui else print

    if boundary_type is None:
        print("[ERROR] Semantic boundary type selection failed; exiting.")
        sys.exit(1)

    notifier("\n" + "=" * 80)
    notifier("  LINE RANGE READJUSTMENT")
    notifier("=" * 80)
    notifier(f"Selected {len(selected_files)} file(s) for adjustment.")
    notifier(f"Context window: {context_window}")
    notifier(f"Dry run: {'yes' if dry_run else 'no'}")
    notifier(f"Boundary type: {boundary_type}")
    if prompt_path:
        notifier(f"Prompt override: {prompt_path}")

    if context_settings.get("use_additional_context", False):
        context_source = "Default boundary-type-specific" if context_settings.get("use_default_context", False) else "File-specific"
    else:
        context_source = "None"
    notifier(f"Additional context: {context_source}")

    successes, skipped, failures = await _adjust_files(
        text_files=selected_files,
        model_config=model_config,
        context_window=context_window,
        prompt_path=prompt_path,
        dry_run=dry_run,
        boundary_type=boundary_type,
        basic_context=basic_context,
        context_settings=context_settings,
        context_manager=context_manager,
        notifier=notifier,
    )

    notifier("\n" + "=" * 80)
    notifier("  SUMMARY")
    notifier("=" * 80)
    notifier(f"Successful adjustments: {len(successes)}")
    notifier(f"Skipped (missing line ranges): {len(skipped)}")
    notifier(f"Failures: {len(failures)}")

    if skipped:
        notifier("\nFiles with no associated line range file:")
        for skipped_file in skipped:
            notifier(f"  - {skipped_file}")
    if failures:
        notifier("\nFiles that encountered errors:")
        for failed_file, error in failures:
            notifier(f"  - {failed_file}: {error}")

    if dry_run and successes:
        notifier(
            "\n[INFO] Dry run enabled; no files were modified. Re-run without --dry-run to persist adjustments."
        )


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Line range readjustment cancelled by user.")
        print("\n[INFO] Operation cancelled by user.")
        sys.exit(0)
    except Exception as exc:  # pragma: no cover - top-level guard
        logger.exception("Unexpected error in line_range_readjuster", exc_info=exc)
        print(f"\n[ERROR] Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
