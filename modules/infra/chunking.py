"""Text chunking for ChronoMiner.

Merges the former ``modules.infra.chunking`` and
``modules.infra.chunking`` into one cohesive module. Public surface:

* :class:`TextProcessor` — encoding detection, normalization, token estimation
* :class:`ChunkingStrategy` (ABC), :class:`TokenBasedChunking`,
  :class:`ChunkHandler` — low-level chunking primitives
* :class:`ChunkingService` — high-level entry point accepting a strategy name
* :class:`ChunkSlice`, :func:`apply_chunk_slice` — first-n / last-n slicing
* :func:`load_line_ranges` — read a ``_line_ranges.txt`` file
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

from charset_normalizer import detect as _charset_detect
import tiktoken

from modules.infra.paths import ensure_path_safe

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_cl100k_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# TextProcessor and chunking primitives (formerly modules.infra.chunking).
# ---------------------------------------------------------------------------


class TextProcessor:
    """
    Handles tasks such as encoding detection, text normalization, and token
    estimation.
    """

    @staticmethod
    def detect_encoding(file_path: Path) -> str:
        """
        Detect the encoding of a file.

        :param file_path: Path to the file.
        :return: The detected encoding.
        """
        safe_path = ensure_path_safe(file_path)
        with safe_path.open("rb") as f:
            raw_data = f.read(100000)
        result = _charset_detect(raw_data)
        encoding = result["encoding"]
        logger.info(f"Detected file encoding: {encoding}")
        return encoding or "utf-8"

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text by stripping whitespace.
        """
        return text.strip()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate the number of tokens in the text using cl100k_base encoding.
        """
        if not text:
            return 0
        encoding = _get_cl100k_encoding()
        return len(encoding.encode(text))


class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies."""

    @abstractmethod
    def get_line_ranges(self, lines: list[str]) -> list[tuple[int, int]]:
        """Determine line ranges for chunking."""
        pass


class TokenBasedChunking(ChunkingStrategy):
    """A chunking strategy that groups lines based on token count."""

    def __init__(
        self,
        tokens_per_chunk: int,
        model_name: str,
        text_processor: TextProcessor,
    ) -> None:
        self.tokens_per_chunk = tokens_per_chunk
        self.model_name = model_name
        self.text_processor = text_processor

    def get_line_ranges(self, lines: list[str]) -> list[tuple[int, int]]:
        """Compute line ranges based on the token count of each line."""
        ranges = []
        current_tokens = 0
        start_line = 1
        end_line = 1
        tokens_per_chunk = self.tokens_per_chunk
        encode = _get_cl100k_encoding().encode

        for idx, line in enumerate(lines, 1):
            if not line:
                line_tokens = 0
            else:
                line_tokens = len(encode(line))
            if current_tokens + line_tokens > tokens_per_chunk and current_tokens > 0:
                ranges.append((start_line, end_line))
                start_line = idx
                current_tokens = line_tokens
            else:
                current_tokens += line_tokens
            end_line = idx

        if start_line <= end_line:
            ranges.append((start_line, end_line))
        logger.info(f"Created {len(ranges)} chunks based on token limits")
        return ranges


class ChunkHandler:
    """Handles splitting text into chunks based on computed line ranges."""

    def __init__(
        self,
        model_name: str,
        default_tokens_per_chunk: int,
        text_processor: TextProcessor,
    ) -> None:
        self.model_name = model_name
        self.default_tokens_per_chunk = default_tokens_per_chunk
        self.text_processor = text_processor

    def get_line_ranges(
        self, strategy: ChunkingStrategy, lines: list[str]
    ) -> list[tuple[int, int]]:
        """Get line ranges using the provided chunking strategy."""
        return strategy.get_line_ranges(lines)

    def split_text_into_chunks(
        self, all_lines: list[str], ranges: list[tuple[int, int]]
    ) -> list[str]:
        """Split the full text into chunks based on line ranges."""
        chunks: list[str] = []
        for start, end in ranges:
            chunk: str = "".join(all_lines[start - 1:end])
            chunks.append(chunk)
        return chunks

    def adjust_line_ranges(
        self,
        initial_ranges: list[tuple[int, int]],
        original_start_line: int,
        total_processed_lines: int,
    ) -> list[tuple[int, int] | None]:
        """Interactively adjust the default line ranges."""
        final_ranges: list[tuple[int, int] | None] = []
        current_start: int = original_start_line
        total_lines: int = original_start_line + total_processed_lines - 1

        i: int = 0
        while current_start <= total_lines:
            actual_start: int = current_start
            if i < len(initial_ranges):
                initial_end: int = original_start_line + initial_ranges[i][1] - 1
            else:
                initial_end = total_lines

            logger.info(f"Chunk {i + 1}: Lines {actual_start} - {initial_end}")
            while True:
                user_input: str = input(
                    f"Enter the new end line for Chunk {i + 1} "
                    f"(current end line: {initial_end}) "
                    "or press Enter to keep it: "
                ).strip()
                if user_input == "":
                    actual_end = initial_end
                    final_ranges.append((actual_start, actual_end))
                    logger.info(
                        f"Chunk {i + 1} kept as Lines "
                        f"{actual_start} - {actual_end}\n"
                    )
                    break
                else:
                    try:
                        new_end: int = int(user_input)
                        if actual_start <= new_end <= total_lines:
                            actual_end = new_end
                            final_ranges.append((actual_start, actual_end))
                            logger.info(
                                f"Chunk {i + 1} redefined to Lines "
                                f"{actual_start} - {actual_end}\n"
                            )
                            break
                        else:
                            logger.error(
                                f"Invalid end line. It must be between "
                                f"{actual_start} and {total_lines}."
                            )
                    except ValueError:
                        logger.error(
                            "Invalid input. Please enter a valid integer."
                        )
            current_start = actual_end + 1
            i += 1
        return final_ranges


def load_line_ranges(line_ranges_file: Path) -> list[tuple[int, int]]:
    """Load line ranges from a ``_line_ranges.txt`` file."""
    ranges = []
    try:
        safe_file = ensure_path_safe(line_ranges_file)
        with safe_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("(") and line.endswith(")"):
                    line = line[1:-1]
                elif not line[0].isdigit():
                    logger.warning(f"Invalid line range format: {line}")
                    continue

                parts = line.split(",")
                if len(parts) != 2:
                    logger.warning(f"Expected two numbers in line range, got: {line}")
                    continue

                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    ranges.append((start, end))
                except ValueError:
                    logger.error(f"Invalid integer values in line range: {line}")
    except Exception as e:
        logger.error(f"Error reading line ranges from {line_ranges_file}: {e}")

    logger.info(f"Loaded {len(ranges)} line ranges from {line_ranges_file}")
    return ranges


# ---------------------------------------------------------------------------
# ChunkSlice / ChunkingService (formerly modules.infra.chunking).
# ---------------------------------------------------------------------------


@dataclass
class ChunkSlice:
    """Specifies which subset of chunks/pages to process.

    Exactly one of *first_n* or *last_n* may be set (they are mutually
    exclusive). When both are ``None`` the full chunk list is used.
    """

    first_n: int | None = None
    last_n: int | None = None

    def __post_init__(self) -> None:
        if self.first_n is not None and self.last_n is not None:
            raise ValueError("first_n and last_n are mutually exclusive")
        if self.first_n is not None and self.first_n < 1:
            raise ValueError("first_n must be >= 1")
        if self.last_n is not None and self.last_n < 1:
            raise ValueError("last_n must be >= 1")


def apply_chunk_slice(
    chunks: list[str],
    ranges: list[tuple[int, int]],
    chunk_slice: ChunkSlice | None,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Return the subset of *chunks* (or pages) and *ranges* selected by
    *chunk_slice*.

    If *chunk_slice* is ``None`` or requests more chunks than available the
    full lists are returned (with a warning in the latter case).
    """
    if chunk_slice is None:
        return chunks, ranges

    total = len(chunks)

    if chunk_slice.first_n is not None:
        n = chunk_slice.first_n
        if n >= total:
            logger.warning(
                "Requested first %d chunks/pages but only %d available; "
                "processing all",
                n,
                total,
            )
            return chunks, ranges
        return chunks[:n], ranges[:n]

    if chunk_slice.last_n is not None:
        n = chunk_slice.last_n
        if n >= total:
            logger.warning(
                "Requested last %d chunks/pages but only %d available; "
                "processing all",
                n,
                total,
            )
            return chunks, ranges
        return chunks[-n:], ranges[-n:]

    return chunks, ranges


class ChunkingService:
    """
    Centralized service for text chunking operations.
    Provides unified interface for different chunking strategies.
    """

    def __init__(
        self,
        model_name: str,
        default_tokens_per_chunk: int,
        text_processor: TextProcessor | None = None,
    ):
        self.model_name = model_name
        self.default_tokens_per_chunk = default_tokens_per_chunk
        self.text_processor = text_processor or TextProcessor()
        self.chunk_handler = ChunkHandler(
            model_name=model_name,
            default_tokens_per_chunk=default_tokens_per_chunk,
            text_processor=self.text_processor,
        )

    def chunk_text(
        self,
        lines: list[str],
        strategy: str = "auto",
        line_ranges_file: Path | None = None,
        original_start_line: int = 1,
        console_print: Callable[[str], None] | None = None,
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """Chunk text using the specified strategy."""
        if strategy in {"line_ranges.txt", "line_ranges"}:
            return self._chunk_from_file(lines, line_ranges_file)
        elif strategy == "auto":
            return self._chunk_automatic(lines, original_start_line)
        elif strategy == "auto-adjust":
            return self._chunk_with_adjustment(
                lines, original_start_line, console_print=console_print
            )
        else:
            logger.warning(
                f"Unknown chunking strategy '{strategy}', defaulting to 'auto'"
            )
            return self._chunk_automatic(lines, original_start_line)

    def _chunk_from_file(
        self, lines: list[str], line_ranges_file: Path | None
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """Load line ranges from file and chunk accordingly."""
        if not line_ranges_file or not line_ranges_file.exists():
            logger.warning(
                "Line ranges file not found, falling back to automatic chunking"
            )
            return self._chunk_automatic(lines, 1)

        line_ranges = load_line_ranges(line_ranges_file)
        chunks = self.chunk_handler.split_text_into_chunks(lines, line_ranges)
        logger.info(f"Using line ranges from {line_ranges_file}")
        return chunks, line_ranges

    def _chunk_automatic(
        self, lines: list[str], original_start_line: int
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """Perform automatic token-based chunking."""
        strategy = TokenBasedChunking(
            tokens_per_chunk=self.default_tokens_per_chunk,
            model_name=self.model_name,
            text_processor=self.text_processor,
        )
        token_ranges = self.chunk_handler.get_line_ranges(strategy, lines)

        # Adjust ranges to account for original start line
        final_ranges = [
            (original_start_line + start - 1, original_start_line + end - 1)
            for (start, end) in token_ranges
        ]
        chunks = self.chunk_handler.split_text_into_chunks(lines, token_ranges)
        logger.info(f"Created {len(chunks)} automatic chunks")
        return chunks, final_ranges

    def _chunk_with_adjustment(
        self,
        lines: list[str],
        original_start_line: int,
        console_print: Callable[[str], None] | None = None,
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """Perform automatic chunking with interactive adjustment."""
        _print = console_print or logger.info
        strategy = TokenBasedChunking(
            tokens_per_chunk=self.default_tokens_per_chunk,
            model_name=self.model_name,
            text_processor=self.text_processor,
        )
        token_ranges = self.chunk_handler.get_line_ranges(strategy, lines)

        _print("\nThe following default token-based chunks were created:")
        for i, (start, end) in enumerate(token_ranges, 1):
            actual_start = original_start_line + start - 1
            actual_end = original_start_line + end - 1
            _print(f"  Chunk {i}: Lines {actual_start} - {actual_end}")

        _print(
            "\nYou can now adjust the chunk boundaries if you wish. "
            "Press Enter to keep the default for each chunk."
        )

        final_ranges = self.chunk_handler.adjust_line_ranges(
            token_ranges, original_start_line, len(lines)
        )

        adjusted_ranges = [r for r in final_ranges if r is not None]
        chunks = self.chunk_handler.split_text_into_chunks(lines, adjusted_ranges)
        logger.info(f"Created {len(chunks)} adjusted chunks")
        return chunks, adjusted_ranges

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ChunkingService":
        """Create ChunkingService from configuration dictionary."""
        model_name = config.get("model_name", "o3-mini")
        tokens_per_chunk = config.get("default_tokens_per_chunk", 7500)
        return cls(model_name=model_name, default_tokens_per_chunk=tokens_per_chunk)
