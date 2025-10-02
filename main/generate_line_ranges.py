# main/generate_line_ranges.py

"""
Script to generate line ranges for text files.

This script selects a schema, reads a text file (or multiple files), and generates line ranges based
on token-based chunking. The line ranges are written to a '_line_ranges.txt' file.

Supports two execution modes:
1. Interactive Mode: User-friendly prompts
2. CLI Mode: Command-line arguments for automation
"""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Tuple

from modules.cli.args_parser import create_generate_ranges_parser, get_files_from_path, resolve_path
from modules.cli.execution_framework import DualModeScript
from modules.core.schema_manager import SchemaManager
from modules.core.text_utils import TextProcessor, TokenBasedChunking


def generate_line_ranges_for_file(
    text_file: Path, default_tokens_per_chunk: int, model_name: str
) -> List[Tuple[int, int]]:
    """
    Generate line ranges for a text file based on token-based chunking.

    Args:
        text_file: The text file to process
        default_tokens_per_chunk: The default token count per chunk
        model_name: The name of the model used for token estimation
    
    Returns:
        A list of tuples representing line ranges
    """
    encoding: str = TextProcessor.detect_encoding(text_file)
    with text_file.open('r', encoding=encoding) as f:
        lines: List[str] = f.readlines()
    
    normalized_lines: List[str] = [TextProcessor.normalize_text(line) for line in lines]
    text_processor: TextProcessor = TextProcessor()
    strategy: TokenBasedChunking = TokenBasedChunking(
        tokens_per_chunk=default_tokens_per_chunk,
        model_name=model_name,
        text_processor=text_processor
    )
    line_ranges: List[Tuple[int, int]] = strategy.get_line_ranges(normalized_lines)
    return line_ranges


def write_line_ranges_file(text_file: Path, line_ranges: List[Tuple[int, int]]) -> Path:
    """
    Write the generated line ranges to a '_line_ranges.txt' file.

    Args:
        text_file: The original text file
        line_ranges: A list of line ranges to write
    
    Returns:
        Path to the created line ranges file
    """
    line_ranges_file: Path = text_file.with_name(f"{text_file.stem}_line_ranges.txt")
    with line_ranges_file.open("w", encoding="utf-8") as f:
        for r in line_ranges:
            f.write(f"({r[0]}, {r[1]})\n")
    return line_ranges_file


class GenerateLineRangesScript(DualModeScript):
    """Script to generate line ranges for text files based on token chunking."""
    
    def __init__(self):
        super().__init__("generate_line_ranges")
        self.model_name: Optional[str] = None
        self.tokens_per_chunk: Optional[int] = None
    
    def create_argument_parser(self) -> ArgumentParser:
        """Create argument parser for CLI mode."""
        return create_generate_ranges_parser()
    
    def _get_model_config(self) -> tuple[str, int]:
        """Get model name and tokens per chunk from configuration."""
        chunking_config = self.chunking_and_context_config.get("chunking", {})
        model_cfg = self.model_config.get("transcription_model", {})
        model_name = model_cfg.get("name", "o3-mini")
        tokens_per_chunk = chunking_config.get("default_tokens_per_chunk", 7500)
        return model_name, tokens_per_chunk
    
    def _select_schema(self) -> str:
        """Prompt user to select a schema."""
        schema_manager = SchemaManager()
        schema_manager.load_schemas()
        available_schemas = schema_manager.get_available_schemas()
        
        if not available_schemas:
            self.ui.print_error("No schemas available. Please add schemas to the 'schemas/' folder.")
            sys.exit(1)
        
        result = self.ui.select_schema(schema_manager)
        if result is None:
            self.ui.print_info("Schema selection cancelled.")
            sys.exit(0)
        
        _, selected_schema_name = result
        return selected_schema_name
    
    def _get_input_directory(self, schema_name: str) -> Path:
        """Get input directory for the selected schema."""
        if schema_name in self.schemas_paths:
            return Path(self.schemas_paths[schema_name].get("input"))
        else:
            return Path(self.paths_config.get("input_paths", {}).get("raw_text_dir", ""))
    
    def _select_files_interactive(self, raw_text_dir: Path) -> List[Path]:
        """Prompt user to select files for processing."""
        self.ui.print_section_header("Input Selection")
        
        mode_options = [
            ("single", "Process a single file"),
            ("folder", "Process all files in a folder")
        ]
        
        mode = self.ui.select_option(
            "Select how you would like to specify input:",
            mode_options,
            allow_back=False,
            allow_quit=True
        )
        
        files: List[Path] = []
        
        if mode == "single":
            files = self._select_single_file(raw_text_dir)
        elif mode == "folder":
            files = self._select_folder_files(raw_text_dir)
        
        return files
    
    def _select_single_file(self, raw_text_dir: Path) -> List[Path]:
        """Select a single file for processing."""
        file_input = self.ui.get_input(
            "Enter the filename to process (with or without .txt extension)",
            allow_back=False,
            allow_quit=True
        )
        
        if not file_input:
            self.ui.print_info("Operation cancelled.")
            sys.exit(0)
        
        if not file_input.lower().endswith(".txt"):
            file_input += ".txt"
        
        file_candidates: List[Path] = [
            f for f in raw_text_dir.rglob(file_input) 
            if not f.name.endswith("_line_ranges.txt")
        ]
        
        if not file_candidates:
            self.ui.print_error(f"File '{file_input}' not found in {raw_text_dir}")
            sys.exit(1)
        elif len(file_candidates) == 1:
            file_path: Path = file_candidates[0]
            self.ui.print_success(f"Selected: {file_path.name}")
            return [file_path]
        else:
            return self._select_from_multiple(file_candidates, raw_text_dir)
    
    def _select_from_multiple(self, candidates: List[Path], base_dir: Path) -> List[Path]:
        """Handle selection when multiple matching files are found."""
        self.ui.print_warning(f"Found {len(candidates)} matching files:")
        self.ui.console_print(self.ui.HORIZONTAL_LINE)
        
        for idx, f in enumerate(candidates, 1):
            self.ui.console_print(f"  {idx}. {f.relative_to(base_dir)}")
        
        while True:
            selected_index = self.ui.get_input(
                "Select file by number",
                allow_back=False,
                allow_quit=True
            )
            
            if not selected_index:
                sys.exit(0)
            
            try:
                idx: int = int(selected_index) - 1
                if 0 <= idx < len(candidates):
                    file_path = candidates[idx]
                    self.ui.print_success(f"Selected: {file_path.name}")
                    return [file_path]
                else:
                    self.ui.print_error(f"Please enter a number between 1 and {len(candidates)}.")
            except ValueError:
                self.ui.print_error("Invalid input. Please enter a number.")
    
    def _select_folder_files(self, raw_text_dir: Path) -> List[Path]:
        """Select all text files in a folder."""
        files = [
            f for f in raw_text_dir.rglob("*.txt") 
            if not f.name.endswith("_line_ranges.txt")
        ]
        
        if not files:
            self.ui.print_error(f"No .txt files found in {raw_text_dir}")
            sys.exit(1)
        
        self.ui.print_success(f"Found {len(files)} text files to process")
        return files
    
    def _process_files(self, files: List[Path], verbose: bool = False) -> tuple[int, int]:
        """
        Process files and generate line ranges.
        
        Args:
            files: List of files to process
            verbose: Whether to show verbose output
        
        Returns:
            Tuple of (success_count, fail_count)
        """
        success_count = 0
        fail_count = 0
        
        for file_path in files:
            try:
                if verbose or self.ui:
                    self.print_or_log(f"Processing {file_path.name}...")
                
                self.logger.info(f"Generating line ranges for {file_path}")
                
                line_ranges = generate_line_ranges_for_file(
                    text_file=file_path,
                    default_tokens_per_chunk=self.tokens_per_chunk,
                    model_name=self.model_name
                )
                
                line_ranges_file = write_line_ranges_file(file_path, line_ranges)
                
                if self.ui:
                    self.ui.print_success(f"Line ranges written to {line_ranges_file.name}")
                elif verbose:
                    print(f"[SUCCESS] Created {line_ranges_file.name}")
                
                self.logger.info(f"Line ranges written to {line_ranges_file}")
                success_count += 1
                
            except Exception as e:
                self.logger.exception(f"Error processing {file_path}", exc_info=e)
                if self.ui:
                    self.ui.print_error(f"Failed to process {file_path.name}: {e}")
                else:
                    print(f"[ERROR] Failed to process {file_path.name}: {e}")
                fail_count += 1
        
        return success_count, fail_count
    
    def run_interactive(self) -> None:
        """Run line range generation in interactive mode."""
        self.ui.print_section_header("Line Range Generation")
        self.ui.print_info("Loading configuration...")
        
        # Get model configuration
        self.model_name, self.tokens_per_chunk = self._get_model_config()
        
        # Select schema
        selected_schema_name = self._select_schema()
        raw_text_dir = self._get_input_directory(selected_schema_name)
        
        # Select files
        files = self._select_files_interactive(raw_text_dir)
        
        # Confirm processing
        if not self.ui.confirm(f"Generate line ranges for {len(files)} file(s)?", default=True):
            self.ui.print_info("Operation cancelled by user.")
            return
        
        # Process files
        self.ui.print_section_header("Generating Line Ranges")
        success_count, fail_count = self._process_files(files, verbose=False)
        
        # Final summary
        self.ui.print_section_header("Generation Complete")
        self.ui.print_success(f"Successfully generated line ranges for {success_count} file(s)")
        if fail_count > 0:
            self.ui.print_warning(f"Failed to process {fail_count} file(s)")
    
    def run_cli(self, args: Namespace) -> None:
        """Run line range generation in CLI mode."""
        self.logger.info("Starting line range generation (CLI Mode)")
        
        # Get model configuration
        self.model_name, default_tokens = self._get_model_config()
        self.tokens_per_chunk = args.tokens if args.tokens else default_tokens
        
        # Resolve input path
        input_path = resolve_path(args.input)
        if not input_path.exists():
            self.logger.error(f"Input path does not exist: {input_path}")
            print(f"[ERROR] Input path not found: {input_path}")
            sys.exit(1)
        
        # Get files
        files = get_files_from_path(
            input_path,
            pattern="*.txt",
            exclude_patterns=["*_line_ranges.txt"]
        )
        
        if not files:
            self.logger.error(f"No text files found at: {input_path}")
            print(f"[ERROR] No text files found at: {input_path}")
            sys.exit(1)
        
        self.logger.info(f"Found {len(files)} file(s) to process")
        if args.verbose:
            print(f"[INFO] Processing {len(files)} file(s) with {self.tokens_per_chunk} tokens per chunk")
        
        # Process files
        success_count, fail_count = self._process_files(files, verbose=args.verbose)
        
        # Final summary
        self.logger.info(f"Generation complete: {success_count} succeeded, {fail_count} failed")
        print(f"[SUCCESS] Generated line ranges for {success_count}/{len(files)} file(s)")
        if fail_count > 0:
            print(f"[WARNING] {fail_count} file(s) failed")


def main() -> None:
    """Main entry point."""
    script = GenerateLineRangesScript()
    script.execute()


if __name__ == "__main__":
    main()
