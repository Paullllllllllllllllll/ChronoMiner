# modules/core/text_utils.py

"""Text processing utilities for chunking and encoding detection."""

import functools
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import chardet
import tiktoken

from modules.core.path_utils import ensure_path_safe

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_cl100k_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding('cl100k_base')


class TextProcessor:
    """
    Handles tasks such as encoding detection, text normalization, and token estimation.
    """
    @staticmethod
    def detect_encoding(file_path: Path) -> str:
        """
        Detect the encoding of a file.

        :param file_path: Path to the file.
        :return: The detected encoding.
        """
        safe_path = ensure_path_safe(file_path)
        with safe_path.open('rb') as f:
            raw_data = f.read(100000)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        logger.info(f"Detected file encoding: {encoding}")
        return encoding or 'utf-8'

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text by stripping whitespace.

        :param text: Input text.
        :return: Normalized text.
        """
        return text.strip()

    @staticmethod
    def estimate_tokens(text: str, model_name: str = "o3-mini") -> int:
        """
        Estimate the number of tokens in the text using cl100k_base encoding.

        :param text: Input text.
        :param model_name: Model name (currently unused, reserved for future model-specific tokenization).
        :return: Estimated token count.
        """
        if not text:
            return 0
        encoding = _get_cl100k_encoding()
        return len(encoding.encode(text))


class ChunkingStrategy(ABC):
    """
    Abstract base class for text chunking strategies.
    """
    @abstractmethod
    def get_line_ranges(self, lines: List[str]) -> List[Tuple[int, int]]:
        """
        Determine line ranges for chunking.

        :param lines: List of text lines.
        :return: List of (start, end) tuples.
        """
        pass


class TokenBasedChunking(ChunkingStrategy):
    """
    A chunking strategy that groups lines based on token count.
    """
    def __init__(self, tokens_per_chunk: int, model_name: str, text_processor: TextProcessor) -> None:
        self.tokens_per_chunk = tokens_per_chunk
        self.model_name = model_name
        self.text_processor = text_processor

    def get_line_ranges(self, lines: List[str]) -> List[Tuple[int, int]]:
        """
        Compute line ranges based on the token count of each line.

        :param lines: List of text lines.
        :return: List of (start, end) tuples representing chunks.
        """
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
    """
    Handles splitting text into chunks based on computed line ranges.
    """
    def __init__(self, model_name: str, default_tokens_per_chunk: int, text_processor: TextProcessor) -> None:
        self.model_name = model_name
        self.default_tokens_per_chunk = default_tokens_per_chunk
        self.text_processor = text_processor

    def get_line_ranges(self, strategy: ChunkingStrategy, lines: List[str]) -> List[Tuple[int, int]]:
        """
        Get line ranges using the provided chunking strategy.

        :param strategy: An instance of a chunking strategy.
        :param lines: List of text lines.
        :return: List of (start, end) tuples.
        """
        return strategy.get_line_ranges(lines)

    def split_text_into_chunks(self, all_lines: List[str], ranges: List[Tuple[int, int]]) -> List[str]:
        """
        Split the full text into chunks based on line ranges.

        :param all_lines: List of all text lines.
        :param ranges: List of (start, end) tuples.
        :return: List of text chunks.
        """
        chunks: List[str] = []
        for start, end in ranges:
            chunk: str = ''.join(all_lines[start - 1:end])
            chunks.append(chunk)
        return chunks

    def adjust_line_ranges(
        self, initial_ranges: List[Tuple[int, int]], original_start_line: int, total_processed_lines: int
    ) -> List[Optional[Tuple[int, int]]]:
        """
        Interactively adjust the default line ranges.

        :param initial_ranges: The default computed line ranges.
        :param original_start_line: The starting line number.
        :param total_processed_lines: Total number of lines.
        :return: List of adjusted (start, end) tuples.
        """
        final_ranges: List[Optional[Tuple[int, int]]] = []
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
                    f"Enter the new end line for Chunk {i + 1} (current end line: {initial_end}) or press Enter to keep it: "
                ).strip()
                if user_input == '':
                    actual_end = initial_end
                    final_ranges.append((actual_start, actual_end))
                    logger.info(f"Chunk {i + 1} kept as Lines {actual_start} - {actual_end}\n")
                    break
                else:
                    try:
                        new_end: int = int(user_input)
                        if actual_start <= new_end <= total_lines:
                            actual_end = new_end
                            final_ranges.append((actual_start, actual_end))
                            logger.info(f"Chunk {i + 1} redefined to Lines {actual_start} - {actual_end}\n")
                            break
                        else:
                            logger.error(f"Invalid end line. It must be between {actual_start} and {total_lines}.")
                    except ValueError:
                        logger.error("Invalid input. Please enter a valid integer.")
            current_start = actual_end + 1
            i += 1
        return final_ranges


def load_line_ranges(line_ranges_file: Path) -> List[Tuple[int, int]]:
    """
    Load line ranges from a '_line_ranges.txt' file.

    :param line_ranges_file: Path to the line ranges file.
    :return: List of (start, end) tuples.
    """
    ranges = []
    try:
        safe_file = ensure_path_safe(line_ranges_file)
        with safe_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Remove parentheses if present
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
