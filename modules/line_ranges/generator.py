"""Token-based line-range generation for ChronoMiner.

Extracted from ``main/generate_line_ranges.py`` so that the extraction
workflow (which also needs to generate line ranges during automatic
chunking) can import these helpers without depending on a sibling
entry-point script.
"""

from __future__ import annotations

from pathlib import Path

from modules.infra.chunking import TextProcessor, TokenBasedChunking


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
    with text_file.open("r", encoding="utf-8") as f:
        lines: list[str] = f.readlines()

    normalized_lines: list[str] = [
        TextProcessor.normalize_text(line) for line in lines
    ]
    text_processor: TextProcessor = TextProcessor()
    strategy: TokenBasedChunking = TokenBasedChunking(
        tokens_per_chunk=default_tokens_per_chunk,
        model_name=model_name,
        text_processor=text_processor,
    )
    line_ranges: list[tuple[int, int]] = strategy.get_line_ranges(normalized_lines)
    return line_ranges


def write_line_ranges_file(
    text_file: Path, line_ranges: list[tuple[int, int]]
) -> Path:
    """
    Write the generated line ranges to a '_line_ranges.txt' file.

    Args:
        text_file: The original text file.
        line_ranges: A list of line ranges to write.

    Returns:
        Path to the created line ranges file.
    """
    line_ranges_file: Path = text_file.with_name(
        f"{text_file.stem}_line_ranges.txt"
    )
    with line_ranges_file.open("w", encoding="utf-8", newline="\n") as f:
        for r in line_ranges:
            f.write(f"({r[0]}, {r[1]})\n")
    return line_ranges_file
