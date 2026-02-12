# modules/core/chunking_service.py

"""
Centralized chunking service for text processing.
Consolidates chunking logic from text_utils and file_processor.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from modules.core.text_utils import (
    TextProcessor,
    TokenBasedChunking,
    ChunkHandler,
    load_line_ranges
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkSlice:
    """Specifies which subset of chunks to process.

    Exactly one of *first_n* or *last_n* may be set (they are mutually
    exclusive).  When both are ``None`` the full chunk list is used.
    """
    first_n: Optional[int] = None
    last_n: Optional[int] = None

    def __post_init__(self) -> None:
        if self.first_n is not None and self.last_n is not None:
            raise ValueError("first_n and last_n are mutually exclusive")
        if self.first_n is not None and self.first_n < 1:
            raise ValueError("first_n must be >= 1")
        if self.last_n is not None and self.last_n < 1:
            raise ValueError("last_n must be >= 1")


def apply_chunk_slice(
    chunks: List[str],
    ranges: List[Tuple[int, int]],
    chunk_slice: Optional[ChunkSlice],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Return the subset of *chunks* and *ranges* selected by *chunk_slice*.

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
                "Requested first %d chunks but only %d available; processing all",
                n, total,
            )
            return chunks, ranges
        return chunks[:n], ranges[:n]

    if chunk_slice.last_n is not None:
        n = chunk_slice.last_n
        if n >= total:
            logger.warning(
                "Requested last %d chunks but only %d available; processing all",
                n, total,
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
        text_processor: Optional[TextProcessor] = None
    ):
        """
        Initialize chunking service.

        :param model_name: Model name for token counting
        :param default_tokens_per_chunk: Default chunk size in tokens
        :param text_processor: Optional TextProcessor instance
        """
        self.model_name = model_name
        self.default_tokens_per_chunk = default_tokens_per_chunk
        self.text_processor = text_processor or TextProcessor()
        self.chunk_handler = ChunkHandler(
            model_name=model_name,
            default_tokens_per_chunk=default_tokens_per_chunk,
            text_processor=self.text_processor
        )

    def chunk_text(
        self,
        lines: List[str],
        strategy: str = "auto",
        line_ranges_file: Optional[Path] = None,
        original_start_line: int = 1
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Chunk text using the specified strategy.

        :param lines: List of text lines
        :param strategy: Chunking strategy ('auto', 'auto-adjust', 'line_ranges.txt')
        :param line_ranges_file: Optional path to line ranges file
        :param original_start_line: Starting line number
        :return: Tuple of (chunks, line_ranges)
        """
        if strategy in {"line_ranges.txt", "line_ranges"}:
            return self._chunk_from_file(lines, line_ranges_file)
        elif strategy == "auto":
            return self._chunk_automatic(lines, original_start_line)
        elif strategy == "auto-adjust":
            return self._chunk_with_adjustment(lines, original_start_line)
        else:
            logger.warning(f"Unknown chunking strategy '{strategy}', defaulting to 'auto'")
            return self._chunk_automatic(lines, original_start_line)

    def _chunk_from_file(
        self,
        lines: List[str],
        line_ranges_file: Optional[Path]
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Load line ranges from file and chunk accordingly."""
        if not line_ranges_file or not line_ranges_file.exists():
            logger.warning("Line ranges file not found, falling back to automatic chunking")
            return self._chunk_automatic(lines, 1)

        line_ranges = load_line_ranges(line_ranges_file)
        chunks = self.chunk_handler.split_text_into_chunks(lines, line_ranges)
        logger.info(f"Using line ranges from {line_ranges_file}")
        return chunks, line_ranges

    def _chunk_automatic(
        self,
        lines: List[str],
        original_start_line: int
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Perform automatic token-based chunking."""
        strategy = TokenBasedChunking(
            tokens_per_chunk=self.default_tokens_per_chunk,
            model_name=self.model_name,
            text_processor=self.text_processor
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
        lines: List[str],
        original_start_line: int
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Perform automatic chunking with interactive adjustment."""
        strategy = TokenBasedChunking(
            tokens_per_chunk=self.default_tokens_per_chunk,
            model_name=self.model_name,
            text_processor=self.text_processor
        )
        token_ranges = self.chunk_handler.get_line_ranges(strategy, lines)

        print("\nThe following default token-based chunks were created:")
        for i, (start, end) in enumerate(token_ranges, 1):
            actual_start = original_start_line + start - 1
            actual_end = original_start_line + end - 1
            print(f"  Chunk {i}: Lines {actual_start} - {actual_end}")
        
        print("\nYou can now adjust the chunk boundaries if you wish. Press Enter to keep the default for each chunk.")
        
        final_ranges = self.chunk_handler.adjust_line_ranges(
            token_ranges,
            original_start_line,
            len(lines)
        )
        
        # Filter out None values and convert to proper ranges
        adjusted_ranges = [r for r in final_ranges if r is not None]
        chunks = self.chunk_handler.split_text_into_chunks(lines, adjusted_ranges)
        logger.info(f"Created {len(chunks)} adjusted chunks")
        return chunks, adjusted_ranges

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ChunkingService":
        """
        Create ChunkingService from configuration dictionary.

        :param config: Configuration dictionary with model and chunking settings
        :return: Configured ChunkingService instance
        """
        model_name = config.get("model_name", "o3-mini")
        tokens_per_chunk = config.get("default_tokens_per_chunk", 7500)
        return cls(model_name=model_name, default_tokens_per_chunk=tokens_per_chunk)
