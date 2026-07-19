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
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken
from charset_normalizer import detect as _charset_detect

from modules.infra.paths import ensure_path_safe

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_cl100k_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


@functools.lru_cache(maxsize=8)
def _get_encoding_for_model(model_name: str) -> tiktoken.Encoding:
    """Return the model's tiktoken encoding, falling back to cl100k_base.

    ``encoding_for_model`` raises ``KeyError`` for models tiktoken does not know
    (every non-OpenAI model, and newer OpenAI models before the library is
    updated). The cl100k_base fallback keeps chunk token counting working for
    all providers rather than crashing on an unknown name.
    """
    if not model_name:
        return _get_cl100k_encoding()
    try:
        return tiktoken.encoding_for_model(model_name)
    except (KeyError, ValueError):
        return _get_cl100k_encoding()


@functools.lru_cache(maxsize=8)
def _special_token_pattern(encoding_name: str) -> re.Pattern[str] | None:
    """Return a compiled alternation matching any special-token string.

    Cached per encoding name. Returns ``None`` when the encoding declares no
    special tokens. Used to decide whether a text can safely be tokenized with
    ``encode_ordinary`` (faster, skips the special-token disallow check) without
    changing behavior: ``encode`` raises ``ValueError`` on a literal special
    token, whereas ``encode_ordinary`` would silently tokenize it. When no
    special token is present anywhere, both produce identical token counts.
    """
    specials = tiktoken.get_encoding(encoding_name).special_tokens_set
    if not specials:
        return None
    alternation = "|".join(re.escape(tok) for tok in sorted(specials))
    return re.compile(alternation)


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
        # Fast path: when the text contains no literal special-token string,
        # encode_ordinary yields identical counts and skips the per-call
        # disallow check. Otherwise fall back to encode so its ValueError is
        # raised exactly as before.
        pattern = _special_token_pattern(encoding.name)
        if pattern is None or pattern.search(text) is None:
            return len(encoding.encode_ordinary(text))
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
        encoding = _get_encoding_for_model(self.model_name)
        # One pre-scan over the whole text decides the tokenizer for the loop:
        # if no literal special-token string occurs anywhere, encode_ordinary
        # gives identical counts and skips the per-line disallow check (faster).
        # If one is present, use encode so its ValueError is raised at the same
        # line with the same message as before. Special tokens contain no
        # newline, so scanning the joined text is equivalent to scanning each
        # line individually.
        pattern = _special_token_pattern(encoding.name)
        if pattern is None or pattern.search("\n".join(lines)) is None:
            encode = encoding.encode_ordinary
        else:
            encode = encoding.encode

        for idx, line in enumerate(lines, 1):
            # Count the newline too: chunks are joined with "\n" downstream
            # (chunking_text_version 2), so per-line counts without it would
            # systematically undershoot the real chunk size.
            line_tokens = len(encode(line + "\n"))
            if current_tokens + line_tokens > tokens_per_chunk and current_tokens > 0:
                ranges.append((start_line, end_line))
                start_line = idx
                current_tokens = line_tokens
            else:
                current_tokens += line_tokens
            end_line = idx

        # Guard against empty input: without it the initial (1, 1) sentinel
        # would be emitted as a phantom chunk for an empty file.
        if lines and start_line <= end_line:
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
        """Split the full text into chunks based on line ranges.

        Lines are joined with ``"\n"`` rather than ``""`` so that inputs whose
        line terminators have already been stripped (the production path
        normalizes each line with ``rstrip("\n\r")`` before chunking) are not
        run together at chunk-internal line boundaries — e.g. ``"Zucker"`` +
        ``"und"`` must not become ``"Zuckerund"``.
        """
        chunks: list[str] = []
        for start, end in ranges:
            chunk: str = "\n".join(all_lines[start - 1 : end])
            chunks.append(chunk)
        return chunks

    def adjust_line_ranges(
        self,
        initial_ranges: list[tuple[int, int]],
        original_start_line: int,
        total_processed_lines: int,
        console_print: Callable[[str], None] | None = None,
    ) -> list[tuple[int, int] | None]:
        """Interactively adjust the default line ranges.

        User-facing lines are routed through ``console_print`` (the console log
        handler is WARNING-level, so a bare ``logger.info`` fallback leaves the
        interactive ``input()`` prompts with no visible context). Logging is
        kept as a secondary sink.
        """

        def _tell(message: str, *, error: bool = False) -> None:
            if console_print is not None:
                console_print(message)
            if error:
                logger.error(message)
            else:
                logger.info(message)

        final_ranges: list[tuple[int, int] | None] = []
        current_start: int = original_start_line
        total_lines: int = original_start_line + total_processed_lines - 1

        prompting: bool = True
        i: int = 0
        while current_start <= total_lines:
            actual_start: int = current_start
            if i < len(initial_ranges):
                initial_end: int = original_start_line + initial_ranges[i][1] - 1
            else:
                initial_end = total_lines

            _tell(f"Chunk {i + 1}: Lines {actual_start} - {initial_end}")
            while True:
                if prompting:
                    try:
                        user_input: str = input(
                            f"Enter the new end line for Chunk {i + 1} "
                            f"(current end line: {initial_end}) "
                            "or press Enter to keep it: "
                        ).strip()
                    except (EOFError, KeyboardInterrupt):
                        prompting = False
                        user_input = ""
                        _tell(
                            "No interactive input available; keeping remaining "
                            "chunk ranges as-is.",
                            error=True,
                        )
                else:
                    user_input = ""
                if user_input == "":
                    actual_end = initial_end
                    final_ranges.append((actual_start, actual_end))
                    _tell(
                        f"Chunk {i + 1} kept as Lines {actual_start} - {actual_end}\n"
                    )
                    break
                else:
                    try:
                        new_end: int = int(user_input)
                        if actual_start <= new_end <= total_lines:
                            actual_end = new_end
                            final_ranges.append((actual_start, actual_end))
                            _tell(
                                f"Chunk {i + 1} redefined to Lines "
                                f"{actual_start} - {actual_end}\n"
                            )
                            break
                        else:
                            _tell(
                                f"Invalid end line. It must be between "
                                f"{actual_start} and {total_lines}.",
                                error=True,
                            )
                    except ValueError:
                        _tell(
                            "Invalid input. Please enter a valid integer.",
                            error=True,
                        )
            current_start = actual_end + 1
            i += 1
        return final_ranges


def load_line_ranges(line_ranges_file: Path) -> list[tuple[int, int]]:
    """Load line ranges from a ``_line_ranges.txt`` file."""
    ranges = []
    try:
        safe_file = ensure_path_safe(line_ranges_file)
        # utf-8-sig strips a leading BOM (Windows "UTF-8 with BOM" from Notepad)
        # that would otherwise cling to the first line and make the first range
        # fail its parse checks, silently shifting every chunk index.
        with safe_file.open("r", encoding="utf-8-sig") as f:
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

    Exactly one of *first_n*, *last_n*, or *page_range* may be
    set. When all are ``None`` the full chunk list is used.
    *page_range* is a (start, end) tuple of 1-based page numbers
    (inclusive on both ends).
    """

    first_n: int | None = None
    last_n: int | None = None
    page_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        active = sum(
            x is not None for x in (self.first_n, self.last_n, self.page_range)
        )
        if active > 1:
            raise ValueError("first_n, last_n, and page_range are mutually exclusive")
        if self.first_n is not None and self.first_n < 1:
            raise ValueError("first_n must be >= 1")
        if self.last_n is not None and self.last_n < 1:
            raise ValueError("last_n must be >= 1")
        if self.page_range is not None:
            start, end = self.page_range
            if start < 1:
                raise ValueError("page_range start must be >= 1")
            if end < start:
                raise ValueError("page_range end must be >= start")


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
                "Requested first %d chunks/pages but only %d available; processing all",
                n,
                total,
            )
            return chunks, ranges
        return chunks[:n], ranges[:n]

    if chunk_slice.last_n is not None:
        n = chunk_slice.last_n
        if n >= total:
            logger.warning(
                "Requested last %d chunks/pages but only %d available; processing all",
                n,
                total,
            )
            return chunks, ranges
        return chunks[-n:], ranges[-n:]

    if chunk_slice.page_range is not None:
        start, end = chunk_slice.page_range
        # 1-based inclusive range → 0-based slice
        i = max(start - 1, 0)
        j = min(end, total)
        if i >= total:
            logger.warning(
                "Page range %d-%d is beyond available %d pages; processing none",
                start,
                end,
                total,
            )
            return [], []
        return chunks[i:j], ranges[i:j]

    return chunks, ranges


def chunk_slice_indices(total: int, chunk_slice: ChunkSlice | None) -> list[int]:
    """Return the absolute 1-based indices selected by *chunk_slice*.

    Mirrors :func:`apply_chunk_slice` exactly (including its "requested more
    than available → process all" fallback), so the returned indices stay in
    lockstep with the chunks/ranges that function keeps. Used to carry document-
    space chunk indices into a sliced run so custom_ids and resume records are
    absolute rather than slice-relative.
    """
    all_idx = list(range(1, total + 1))
    if chunk_slice is None:
        return all_idx
    if chunk_slice.first_n is not None:
        n = chunk_slice.first_n
        return all_idx if n >= total else all_idx[:n]
    if chunk_slice.last_n is not None:
        n = chunk_slice.last_n
        return all_idx if n >= total else all_idx[-n:]
    if chunk_slice.page_range is not None:
        start, end = chunk_slice.page_range
        i = max(start - 1, 0)
        j = min(end, total)
        if i >= total:
            return []
        return all_idx[i:j]
    return all_idx


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
            token_ranges,
            original_start_line,
            len(lines),
            console_print=console_print,
        )

        # adjust_line_ranges already offsets its ranges into document space
        # (seeded at original_start_line). split_text_into_chunks indexes the
        # local `lines` list with 1-based positions, so convert back to local
        # space for splitting and return the document-space ranges — mirroring
        # _chunk_automatic, which splits with local ranges and offsets only the
        # returned ranges.
        adjusted_ranges = [r for r in final_ranges if r is not None]
        offset = original_start_line - 1
        local_ranges = [
            (start - offset, end - offset) for (start, end) in adjusted_ranges
        ]
        chunks = self.chunk_handler.split_text_into_chunks(lines, local_ranges)
        logger.info(f"Created {len(chunks)} adjusted chunks")
        return chunks, adjusted_ranges

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ChunkingService:
        """Create ChunkingService from configuration dictionary."""
        model_name = config.get("model_name", "o3-mini")
        tokens_per_chunk = config.get("default_tokens_per_chunk", 7500)
        return cls(model_name=model_name, default_tokens_per_chunk=tokens_per_chunk)
