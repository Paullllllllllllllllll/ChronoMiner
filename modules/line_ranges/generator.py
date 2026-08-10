"""Token-based line-range generation for ChronoMiner.

Holds the full generation workflow: the core range computation and
sidecar writing (also imported by the extraction workflow for automatic
chunking), interactive input-file selection, and the per-file processing
loop with optional chunk slicing. ``main/generate_line_ranges.py`` is the
thin dual-mode entry point that wires these helpers to CLI arguments and
interactive navigation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from modules.infra.chunking import ChunkSlice, TextProcessor, TokenBasedChunking
from modules.ui import print_error, print_info, print_success, print_warning
from modules.ui.core import UserInterface

# Auxiliary sidecar files excluded from interactive discovery (mirrors
# UserInterface._AUXILIARY_SUFFIXES).
_AUXILIARY_SUFFIXES = (
    "_line_ranges.txt",
    "_line_range.txt",
    "_context.txt",
    "_output.txt",
)


def generate_line_ranges_for_file(
    text_file: Path, default_tokens_per_chunk: int, model_name: str
) -> list[tuple[int, int]]:
    """
    Generate line ranges for a text file based on token-based chunking.

    Args:
        text_file: The text file to process.
        default_tokens_per_chunk: The default token count per chunk.
        model_name: The name of the model used for token estimation.

    Returns:
        A list of tuples representing line ranges.
    """
    # Mirror FileProcessor's tolerant read: UTF-8 first, then charset
    # detection. A file that extracts fine must not crash range generation.
    try:
        with text_file.open("r", encoding="utf-8") as f:
            lines: list[str] = f.readlines()
    except UnicodeDecodeError:
        encoding = TextProcessor.detect_encoding(text_file)
        with text_file.open("r", encoding=encoding) as f:
            lines = f.readlines()

    normalized_lines: list[str] = [TextProcessor.normalize_text(line) for line in lines]
    text_processor: TextProcessor = TextProcessor()
    strategy: TokenBasedChunking = TokenBasedChunking(
        tokens_per_chunk=default_tokens_per_chunk,
        model_name=model_name,
        text_processor=text_processor,
    )
    line_ranges: list[tuple[int, int]] = strategy.get_line_ranges(normalized_lines)
    return line_ranges


def write_line_ranges_file(text_file: Path, line_ranges: list[tuple[int, int]]) -> Path:
    """
    Write the generated line ranges to a '_line_ranges.txt' file.

    Args:
        text_file: The original text file.
        line_ranges: A list of line ranges to write.

    Returns:
        Path to the created line ranges file.
    """
    line_ranges_file: Path = text_file.with_name(f"{text_file.stem}_line_ranges.txt")
    with line_ranges_file.open("w", encoding="utf-8", newline="\n") as f:
        for r in line_ranges:
            f.write(f"({r[0]}, {r[1]})\n")
    return line_ranges_file


def select_input_files(
    ui: UserInterface, raw_text_dir: Path, allow_back: bool = False
) -> list[Path] | None:
    """Prompt the user to select input files for range generation.

    Offers single-file and whole-folder selection; returns ``None`` when
    the user backs out of the selection entirely.
    """
    ui.print_section_header("Input Selection")

    mode_options = [
        ("single", "Process a single file"),
        ("folder", "Process all files in a folder"),
    ]

    mode = ui.select_option(
        "Select how you would like to specify input:",
        mode_options,
        allow_back=allow_back,
        allow_quit=True,
    )

    if mode is None:
        return None

    files: list[Path] | None = []

    if mode == "single":
        files = _select_single_file(ui, raw_text_dir, allow_back=allow_back)
        if files is None:
            return select_input_files(ui, raw_text_dir, allow_back=allow_back)
    elif mode == "folder":
        files = _select_folder_files(ui, raw_text_dir)
        if not files:
            # Nothing to process: return to input mode selection.
            return select_input_files(ui, raw_text_dir, allow_back=allow_back)

    return files


def _select_single_file(
    ui: UserInterface, raw_text_dir: Path, allow_back: bool = False
) -> list[Path] | None:
    """Select a single file for processing."""
    # Containment base: the filename is handed to rglob as a glob pattern,
    # so guard every match against escaping the configured input directory
    # (e.g. via "../" or an absolute path).
    resolved_base = raw_text_dir.resolve()

    while True:
        file_input = ui.get_input(
            "Enter the filename to process (extension optional; defaults to .txt)",
            allow_back=allow_back,
            allow_quit=True,
        )

        if not file_input:
            return None

        # Only supply the default extension when none was typed; an
        # explicit suffix (.md in particular) must be honored as given.
        if not Path(file_input).suffix:
            file_input += ".txt"

        try:
            file_candidates: list[Path] = [
                f
                for f in raw_text_dir.rglob(file_input)
                if f.resolve().is_relative_to(resolved_base)
                and not any(f.name.endswith(suffix) for suffix in _AUXILIARY_SUFFIXES)
            ]
        except (NotImplementedError, ValueError):
            # Python raises NotImplementedError for non-relative (absolute)
            # glob patterns and ValueError for malformed ones.
            file_candidates = []

        if not file_candidates:
            if Path(file_input).is_absolute():
                ui.print_info(
                    "Enter a name relative to the input directory,"
                    " not an absolute path."
                )
            ui.print_error(f"File '{file_input}' not found in {raw_text_dir}")
            ui.print_info("Please try again or press 'b' to go back.")
            continue

        if len(file_candidates) == 1:
            file_path: Path = file_candidates[0]
            ui.print_success(f"Selected: {file_path.name}")
            return [file_path]

        result = _select_from_multiple(
            ui, file_candidates, raw_text_dir, allow_back=allow_back
        )
        if result is not None:
            return result
        # User went back from the numbered list: ask for a filename again.


def _select_from_multiple(
    ui: UserInterface,
    candidates: list[Path],
    base_dir: Path,
    allow_back: bool = False,
) -> list[Path] | None:
    """Handle selection when multiple matching files are found."""
    ui.print_warning(f"Found {len(candidates)} matching files:")
    ui.console_print(ui.HORIZONTAL_LINE)

    for idx, f in enumerate(candidates, 1):
        ui.console_print(f"  {idx}. {f.relative_to(base_dir)}")

    while True:
        selected_index = ui.get_input(
            "Select file by number", allow_back=allow_back, allow_quit=True
        )

        if not selected_index:
            return None

        try:
            idx = int(selected_index) - 1
            if 0 <= idx < len(candidates):
                file_path = candidates[idx]
                ui.print_success(f"Selected: {file_path.name}")
                return [file_path]
            else:
                ui.print_error(
                    f"Please enter a number between 1 and {len(candidates)}."
                )
        except ValueError:
            ui.print_error("Invalid input. Please enter a number.")


def _select_folder_files(ui: UserInterface, raw_text_dir: Path) -> list[Path]:
    """Select all text files in a folder.

    Returns an empty list when the folder holds no eligible files, so the
    caller can re-prompt instead of aborting the run.
    """
    seen: dict[Path, None] = {}
    for pattern in ("*.txt", "*.md"):
        for f in raw_text_dir.rglob(pattern):
            if any(f.name.endswith(suffix) for suffix in _AUXILIARY_SUFFIXES):
                continue
            seen[f] = None
    files = sorted(seen)

    if not files:
        ui.print_error(f"No .txt or .md files found in {raw_text_dir}")
        ui.print_info(
            "Please check the directory or go back to select a different option."
        )
        return []

    ui.print_success(f"Found {len(files)} text files to process")
    return files


def _print_or_log(
    ui: UserInterface | None,
    logger: logging.Logger,
    message: str,
    level: str = "info",
) -> None:
    """Print via the UI when present, otherwise via console helpers; always log.

    Mirrors ``DualModeScript.print_or_log`` so the moved processing loop
    reports identically under both execution modes.
    """
    if ui:
        if level == "error":
            ui.print_error(message)
        elif level == "warning":
            ui.print_warning(message)
        elif level == "success":
            ui.print_success(message)
        else:
            ui.print_info(message)
    else:
        if level == "error":
            print_error(message)
        elif level == "warning":
            print_warning(message)
        elif level == "success":
            print_success(message)
        else:
            print_info(message)

    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)


def process_files(
    files: Sequence[Path],
    *,
    tokens_per_chunk: int,
    model_name: str,
    logger: logging.Logger,
    ui: UserInterface | None = None,
    verbose: bool = False,
    chunk_slice: ChunkSlice | None = None,
) -> tuple[int, int]:
    """
    Generate and write line ranges for each file.

    Args:
        files: Files to process.
        tokens_per_chunk: Token count per generated chunk.
        model_name: Model name used for token estimation.
        logger: Logger for progress and error reporting.
        ui: Optional interactive UI for user-facing feedback.
        verbose: Whether to show verbose output in CLI mode.
        chunk_slice: Optional slice to limit written ranges.

    Returns:
        Tuple of (success_count, fail_count).
    """
    success_count = 0
    fail_count = 0

    for file_path in files:
        try:
            if verbose or ui:
                _print_or_log(ui, logger, f"Processing {file_path.name}...")

            logger.info(f"Generating line ranges for {file_path}")

            line_ranges = generate_line_ranges_for_file(
                text_file=file_path,
                default_tokens_per_chunk=tokens_per_chunk,
                model_name=model_name,
            )

            # Apply chunk slice if requested
            if chunk_slice is not None and (
                chunk_slice.first_n is not None
                or chunk_slice.last_n is not None
                or chunk_slice.page_range is not None
            ):
                original_count = len(line_ranges)
                if chunk_slice.first_n is not None:
                    n = min(chunk_slice.first_n, len(line_ranges))
                    line_ranges = line_ranges[:n]
                elif chunk_slice.last_n is not None:
                    n = min(chunk_slice.last_n, len(line_ranges))
                    line_ranges = line_ranges[-n:]
                elif chunk_slice.page_range is not None:
                    # page_range: 1-based inclusive selection over the
                    # generated ranges, clamped to what exists.
                    start, end = chunk_slice.page_range
                    lo = max(start - 1, 0)
                    hi = min(end, len(line_ranges))
                    line_ranges = line_ranges[lo:hi] if lo < hi else []
                _print_or_log(
                    ui,
                    logger,
                    f"Chunk slice applied: writing "
                    f"{len(line_ranges)}/{original_count} ranges",
                )

            line_ranges_file = write_line_ranges_file(file_path, line_ranges)

            if ui:
                ui.print_success(f"Line ranges written to {line_ranges_file.name}")
            elif verbose:
                print(f"[SUCCESS] Created {line_ranges_file.name}")

            logger.info(f"Line ranges written to {line_ranges_file}")
            success_count += 1

        except Exception as e:
            logger.exception(f"Error processing {file_path}")
            if ui:
                ui.print_error(f"Failed to process {file_path.name}: {e}")
            else:
                print(f"[ERROR] Failed to process {file_path.name}: {e}")
            fail_count += 1

    return success_count, fail_count
