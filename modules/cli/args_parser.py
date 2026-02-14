# modules/cli/args_parser.py

"""
CLI argument parsing utilities for ChronoMiner scripts.

Provides consistent argument parsing across all main scripts when running in CLI mode.
"""

import argparse
from pathlib import Path
from typing import Optional, List


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add common arguments used across multiple scripts.
    
    :param parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument(
        "--schema",
        type=str,
        help="Schema name to use for extraction (e.g., BibliographicEntries)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input file or directory path (relative or absolute)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory path (relative or absolute). If not specified, uses config or input directory."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output"
    )


def create_process_parser() -> argparse.ArgumentParser:
    """Create argument parser for process_text_files.py"""
    parser = argparse.ArgumentParser(
        description="Process text files with structured data extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file with default settings
  python main/process_text_files.py --schema BibliographicEntries --input data/file.txt
  
  # Process a directory with specific chunking
  python main/process_text_files.py --schema BibliographicEntries --input data/ --chunking auto
  
  # Use batch processing
  python main/process_text_files.py --schema BibliographicEntries --input data/ --batch
        """
    )
    
    add_common_arguments(parser)
    
    parser.add_argument(
        "--chunking",
        type=str,
        choices=["auto", "auto-adjust", "line_ranges", "adjust-line-ranges"],
        default="auto",
        help="Chunking strategy to use (default: auto)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch processing (50%% cost reduction, results within 24h)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip fully processed files and resume partially processed ones"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of all files, ignoring existing outputs"
    )
    
    chunk_slice_group = parser.add_mutually_exclusive_group()
    chunk_slice_group.add_argument(
        "--first-n-chunks",
        type=int,
        metavar="N",
        help="Process only the first N chunks of each input file"
    )
    chunk_slice_group.add_argument(
        "--last-n-chunks",
        type=int,
        metavar="N",
        help="Process only the last N chunks of each input file"
    )
    
    return parser


def create_check_batches_parser() -> argparse.ArgumentParser:
    """Create argument parser for check_batches.py"""
    parser = argparse.ArgumentParser(
        description="Check and retrieve batch processing results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all batches across all schemas
  python main/check_batches.py
  
  # Check batches for a specific schema
  python main/check_batches.py --schema BibliographicEntries
  
  # Check batches in a specific directory
  python main/check_batches.py --input data/output/
        """
    )
    
    parser.add_argument(
        "--schema",
        type=str,
        help="Only check batches for this schema"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Directory to scan for batch files (default: uses config paths)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed processing information"
    )
    
    return parser


def create_generate_ranges_parser() -> argparse.ArgumentParser:
    """Create argument parser for generate_line_ranges.py"""
    parser = argparse.ArgumentParser(
        description="Generate line ranges for text files based on token limits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate ranges for a single file
  python main/generate_line_ranges.py --input data/file.txt
  
  # Generate ranges for all files in a directory
  python main/generate_line_ranges.py --input data/ --schema BibliographicEntries
  
  # Use custom token limit
  python main/generate_line_ranges.py --input data/ --tokens 5000

  # Generate ranges and keep only the first 5
  python main/generate_line_ranges.py --input data/ --first-n-chunks 5
        """
    )
    
    parser.add_argument(
        "--schema",
        type=str,
        help="Schema name (used to determine input directory from config if --input not provided)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory path"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        help="Tokens per chunk (default: from config)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed processing information"
    )
    
    chunk_slice_group = parser.add_mutually_exclusive_group()
    chunk_slice_group.add_argument(
        "--first-n-chunks",
        type=int,
        metavar="N",
        help="Write only the first N generated line ranges"
    )
    chunk_slice_group.add_argument(
        "--last-n-chunks",
        type=int,
        metavar="N",
        help="Write only the last N generated line ranges"
    )
    
    return parser


def create_adjust_ranges_parser() -> argparse.ArgumentParser:
    """Create argument parser for line_range_readjuster.py"""
    parser = argparse.ArgumentParser(
        description="Adjust line ranges to align with semantic boundaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Adjust ranges for a file
  python main/line_range_readjuster.py --input data/file.txt --schema BibliographicEntries
  
  # Adjust with custom context window size
  python main/line_range_readjuster.py --input data/ --schema BibliographicEntries --context-window 10
  
  # Resume: skip files whose line ranges were already adjusted
  python main/line_range_readjuster.py --input data/ --schema BibliographicEntries --resume
        """
    )
    
    parser.add_argument(
        "--schema",
        type=str,
        required=True,
        help="Schema name (used as boundary type)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory path"
    )
    parser.add_argument(
        "--context-window",
        type=int,
        help="Number of lines to inspect around boundaries (default: from config)"
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        help="Path to custom prompt template"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed processing information"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files whose line ranges were already adjusted with the same settings"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-adjustment of all files, ignoring existing adjustment markers"
    )
    
    chunk_slice_group = parser.add_mutually_exclusive_group()
    chunk_slice_group.add_argument(
        "--first-n-chunks",
        type=int,
        metavar="N",
        help="Adjust only the first N line ranges of each file"
    )
    chunk_slice_group.add_argument(
        "--last-n-chunks",
        type=int,
        metavar="N",
        help="Adjust only the last N line ranges of each file"
    )
    
    return parser


def create_cancel_batches_parser() -> argparse.ArgumentParser:
    """Create argument parser for cancel_batches.py"""
    parser = argparse.ArgumentParser(
        description="Cancel ongoing batch processing jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cancel all non-terminal batches (with confirmation)
  python main/cancel_batches.py
  
  # Cancel without confirmation (use with caution!)
  python main/cancel_batches.py --force
        """
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt (use with caution)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    
    return parser


def create_repair_parser() -> argparse.ArgumentParser:
    """Create argument parser for repair_extractions.py"""
    parser = argparse.ArgumentParser(
        description="Repair incomplete batch extractions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Repair all pending extractions
  python main/repair_extractions.py
  
  # Repair extractions for a specific schema
  python main/repair_extractions.py --schema BibliographicEntries
  
  # Repair specific files
  python main/repair_extractions.py --files file1_temp.jsonl file2_temp.jsonl
        """
    )
    
    parser.add_argument(
        "--schema",
        type=str,
        help="Only repair extractions for this schema"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific temp files to repair (filenames or paths)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed processing information"
    )
    
    return parser


def resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a path string to an absolute Path object.
    
    :param path_str: Path string (relative or absolute)
    :param base_dir: Base directory for relative paths (default: current working directory)
    :return: Resolved absolute Path
    """
    path = Path(path_str)
    
    if path.is_absolute():
        return path
    
    if base_dir:
        return (base_dir / path).resolve()
    
    return path.resolve()


def validate_input_path(path: Path, must_exist: bool = True) -> None:
    """
    Validate that an input path exists and is accessible.
    
    :param path: Path to validate
    :param must_exist: Whether the path must already exist
    :raises ValueError: If path is invalid
    """
    if must_exist and not path.exists():
        raise ValueError(f"Input path does not exist: {path}")
    
    if must_exist and not (path.is_file() or path.is_dir()):
        raise ValueError(f"Input path is neither a file nor directory: {path}")


def validate_output_path(path: Path, create_parents: bool = True) -> None:
    """
    Validate and prepare an output path.
    
    :param path: Path to validate
    :param create_parents: Whether to create parent directories
    :raises ValueError: If path is invalid
    """
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {path}")
    
    if create_parents:
        path.mkdir(parents=True, exist_ok=True)


def parse_indices(indices_str: str) -> List[int]:
    """
    Parse a comma-separated string of indices or ranges.
    
    :param indices_str: String like "0,5,12" or "1-5,10"
    :return: Sorted list of integer indices
    :raises ValueError: If string format is invalid
    """
    result: set[int] = set()
    
    for part in indices_str.split(","):
        part = part.strip()
        if not part:
            continue
        
        if "-" in part:
            # Range: "1-5"
            try:
                start, end = part.split("-", 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                result.update(range(start_idx, end_idx + 1))
            except ValueError:
                raise ValueError(f"Invalid range format: {part}")
        else:
            # Single index: "5"
            try:
                result.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid index: {part}")
    
    return sorted(result)


def get_files_from_path(path: Path, pattern: str = "*.txt", exclude_patterns: Optional[List[str]] = None) -> List[Path]:
    """
    Get list of files from a path (file or directory).
    
    :param path: Input path (file or directory)
    :param pattern: Glob pattern for files (default: *.txt)
    :param exclude_patterns: List of patterns to exclude (e.g., ["*_line_ranges.txt", "*_context.txt"])
    :return: List of file paths
    """
    if exclude_patterns is None:
        exclude_patterns = []
    
    if path.is_file():
        # Check if file matches exclude patterns
        if any(path.match(excl) for excl in exclude_patterns):
            return []
        return [path]
    
    if path.is_dir():
        files = []
        for file in path.rglob(pattern):
            if not file.is_file():
                continue

            try:
                rel_parts = file.relative_to(path).parts
            except Exception:
                rel_parts = file.parts

            if rel_parts:
                top = str(rel_parts[0]).lower()
                if top == "output" or top.endswith("_outputs"):
                    continue

            if not any(file.match(excl) for excl in exclude_patterns):
                files.append(file)
        return files
    
    return []
