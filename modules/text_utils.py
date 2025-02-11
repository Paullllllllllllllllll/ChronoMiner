# modules/text_utils.py

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import chardet
import tiktoken
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


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
        with file_path.open('rb') as f:
            raw_data: bytes = f.read(100000)
        result: dict = chardet.detect(raw_data)
        encoding: str = result['encoding']
        logger.info(f"Detected file encoding: {encoding}")
        return encoding

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text by stripping whitespace.

        :param text: Input text.
        :return: Normalized text.
        """
        normalized: str = text.strip()
        logger.debug("Text normalized successfully.")
        return normalized

    @staticmethod
    def estimate_tokens(text: str, model_name: str = "o3-mini") -> int:
        """
        Estimate the number of tokens in the text using cl100k_base encoding.

        :param text: Input text.
        :param model_name: Model name (unused in token estimation).
        :return: Estimated token count.
        """
        encoding = tiktoken.get_encoding('cl100k_base')
        token_count: int = len(encoding.encode(text))
        logger.debug(f"Estimated {token_count} tokens for the given text using cl100k_base encoding.")
        return token_count


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
        self.tokens_per_chunk: int = tokens_per_chunk
        self.model_name: str = model_name
        self.text_processor: TextProcessor = text_processor

    def get_line_ranges(self, lines: List[str]) -> List[Tuple[int, int]]:
        """
        Compute line ranges based on the token count of each line.

        :param lines: List of text lines.
        :return: List of (start, end) tuples representing chunks.
        """
        token_ranges: List[Tuple[int, int]] = []
        current_tokens: int = 0
        start_line: int = 1
        end_line: int = 1

        for idx, line in enumerate(lines, 1):
            line_tokens: int = self.text_processor.estimate_tokens(line, self.model_name)
            if current_tokens + line_tokens > self.tokens_per_chunk and current_tokens > 0:
                token_ranges.append((start_line, end_line))
                start_line = idx
                current_tokens = line_tokens
            else:
                current_tokens += line_tokens
            end_line = idx

        if start_line <= end_line:
            token_ranges.append((start_line, end_line))
        logger.info(f"Total chunks created based on tokens: {len(token_ranges)}")
        return token_ranges


class ChunkHandler:
    """
    Handles splitting text into chunks based on computed line ranges.
    """
    def __init__(self, model_name: str, default_tokens_per_chunk: int, text_processor: TextProcessor) -> None:
        self.model_name: str = model_name
        self.default_tokens_per_chunk: int = default_tokens_per_chunk
        self.text_processor: TextProcessor = text_processor

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
    line_ranges: List[Tuple[int, int]] = []
    try:
        with line_ranges_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("(") and line.endswith(")"):
                    line = line[1:-1]
                else:
                    logger.warning(f"Line range format invalid: {line}")
                    continue
                parts = line.split(",")
                if len(parts) != 2:
                    logger.warning(f"Expected two numbers in line range, got: {line}")
                    continue
                try:
                    start: int = int(parts[0].strip())
                    end: int = int(parts[1].strip())
                    line_ranges.append((start, end))
                except ValueError:
                    logger.error(f"Invalid integer values in line range: {line}")
    except Exception as e:
        logger.error(f"Error reading line ranges from {line_ranges_file}: {e}")
    logger.info(f"Loaded {len(line_ranges)} line ranges from {line_ranges_file}")
    return line_ranges


def perform_chunking(
    selected_lines: List[str],
    text_processor: TextProcessor,
    openai_config_task: dict,
    chunk_choice: str,
    original_start_line: int = 1,
    line_ranges_file: Optional[Path] = None
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Perform text chunking using the specified strategy.

    Strategies:
      - "auto": Automatic token-based chunking.
      - "auto-adjust": Automatic token-based chunking with interactive adjustments.
      - "line_ranges.txt": Use an existing line ranges file.

    :param selected_lines: List of normalized text lines.
    :param text_processor: Instance of TextProcessor.
    :param openai_config_task: Configuration with model name and token count.
    :param chunk_choice: Chunking strategy choice.
    :param original_start_line: Starting line number.
    :param line_ranges_file: Optional path to an existing line ranges file.
    :return: Tuple of list of text chunks and the corresponding line ranges.
    """
    if chunk_choice == "line_ranges.txt":
        if line_ranges_file and line_ranges_file.exists():
            line_ranges = load_line_ranges(line_ranges_file)
            chunk_handler = ChunkHandler(
                model_name=openai_config_task["model_name"],
                default_tokens_per_chunk=openai_config_task["default_tokens_per_chunk"],
                text_processor=text_processor
            )
            chunks = chunk_handler.split_text_into_chunks(selected_lines, line_ranges)
            logger.info(f"Using user-provided line ranges from {line_ranges_file}")
            return chunks, line_ranges
        else:
            print("Line ranges file not found; defaulting to automatic chunking.")
            logger.warning("Line ranges file not found; defaulting to auto chunking.")
            chunk_choice = "auto"

    chunk_handler = ChunkHandler(
        model_name=openai_config_task["model_name"],
        default_tokens_per_chunk=openai_config_task["default_tokens_per_chunk"],
        text_processor=text_processor
    )
    strategy: TokenBasedChunking = TokenBasedChunking(
        tokens_per_chunk=openai_config_task["default_tokens_per_chunk"],
        model_name=openai_config_task["model_name"],
        text_processor=text_processor
    )
    token_ranges: List[Tuple[int, int]] = chunk_handler.get_line_ranges(strategy, selected_lines)

    if chunk_choice == 'auto':
        final_ranges: List[Tuple[int, int]] = [
            (original_start_line + start - 1, original_start_line + end - 1)
            for (start, end) in token_ranges
        ]
        chunks: List[str] = chunk_handler.split_text_into_chunks(selected_lines, token_ranges)
        return chunks, final_ranges
    else:
        print("\nThe following default token-based chunks were created:")
        for i, (start, end) in enumerate(token_ranges, 1):
            actual_start: int = original_start_line + start - 1
            actual_end: int = original_start_line + end - 1
            print(f"  Chunk {i}: Lines {actual_start} - {actual_end}")
        print("\nYou can now adjust the chunk boundaries if you wish. Press Enter to keep the default for each chunk.")
        final_ranges = chunk_handler.adjust_line_ranges(token_ranges, original_start_line, len(selected_lines))
        adjusted_ranges: List[Tuple[int, int]] = [
            (start - original_start_line + 1, end - original_start_line + 1)
            for (start, end) in final_ranges
        ]
        chunks = chunk_handler.split_text_into_chunks(selected_lines, adjusted_ranges)
        return chunks, final_ranges
