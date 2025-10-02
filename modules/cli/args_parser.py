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
  
  # Use batch processing with additional context
  python main/process_text_files.py --schema BibliographicEntries --input data/ --batch --context
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
        "--context",
        action="store_true",
        help="Use additional context for extraction"
    )
    parser.add_argument(
        "--context-source",
        type=str,
        choices=["default", "file"],
        default="default",
        help="Context source: 'default' for schema-specific, 'file' for file-specific (default: default)"
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
  
  # Adjust with custom context window
  python main/line_range_readjuster.py --input data/ --schema BibliographicEntries --context-window 10
  
  # Use additional context
  python main/line_range_readjuster.py --input data/ --schema BibliographicEntries --use-context
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
        "--use-context",
        action="store_true",
        help="Use additional context for boundary detection"
    )
    parser.add_argument(
        "--default-context",
        action="store_true",
        help="Use default schema-specific context (requires --use-context)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed processing information"
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


def validate_input_path(path: Path, must_exist: bool = True) -> bool:
    """
    Validate that an input path exists and is accessible.
    
    :param path: Path to validate
    :param must_exist: Whether the path must already exist
    :return: True if valid, False otherwise
    """
    if must_exist and not path.exists():
        return False
    
    if path.exists() and not (path.is_file() or path.is_dir()):
        return False
    
    return True


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
            if file.is_file() and not any(file.match(excl) for excl in exclude_patterns):
                files.append(file)
        return files
    
    return []
