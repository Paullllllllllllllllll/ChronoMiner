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
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from argparse import ArgumentParser, Namespace
from typing import Any

from main.bootstrap import validate_schema_paths
from main.cli_args import (
    create_generate_ranges_parser,
    get_files_from_path,
    resolve_path,
)
from main.dual_mode import DualModeScript
from modules.config.schema_manager import SchemaManager
from modules.infra.chunking import ChunkSlice, TextProcessor, TokenBasedChunking
from modules.line_ranges.generator import (
    generate_line_ranges_for_file,
    write_line_ranges_file,
)


class GenerateLineRangesScript(DualModeScript):
    """Script to generate line ranges for text files based on token chunking."""
    
    def __init__(self) -> None:
        super().__init__("generate_line_ranges")
        self.model_name: str | None = None
        self.tokens_per_chunk: int | None = None
    
    def create_argument_parser(self) -> ArgumentParser:
        """Create argument parser for CLI mode."""
        return create_generate_ranges_parser()
    
    def _get_model_config(self) -> tuple[str, int]:
        """Get model name and tokens per chunk from configuration."""
        chunking_config = self.chunking_and_context_config.get("chunking", {})
        model_cfg = self.model_config.get("extraction_model", {})
        model_name = model_cfg.get("name", "o3-mini")
        tokens_per_chunk = chunking_config.get("default_tokens_per_chunk", 7500)
        return model_name, tokens_per_chunk
    
    def _select_schema(self) -> str:
        """Prompt user to select a schema."""
        assert self.ui is not None
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
        # Assumes validation has been done beforehand
        return Path(self.schemas_paths[schema_name].get("input", ""))
    
    def _select_files_interactive(self, raw_text_dir: Path, allow_back: bool = False) -> list[Path] | None:
        """Prompt user to select files for processing."""
        assert self.ui is not None
        self.ui.print_section_header("Input Selection")
        
        mode_options = [
            ("single", "Process a single file"),
            ("folder", "Process all files in a folder")
        ]
        
        mode = self.ui.select_option(
            "Select how you would like to specify input:",
            mode_options,
            allow_back=allow_back,
            allow_quit=True
        )
        
        if mode is None:
            return None
        
        files: list[Path] | None = []

        if mode == "single":
            files = self._select_single_file(raw_text_dir, allow_back=allow_back)
            if files is None:
                return self._select_files_interactive(raw_text_dir, allow_back=allow_back)
        elif mode == "folder":
            files = self._select_folder_files(raw_text_dir)

        return files
    
    def _select_single_file(self, raw_text_dir: Path, allow_back: bool = False) -> list[Path] | None:
        """Select a single file for processing."""
        assert self.ui is not None
        file_input = self.ui.get_input(
            "Enter the filename to process (with or without .txt extension)",
            allow_back=allow_back,
            allow_quit=True
        )
        
        if not file_input:
            return None
        
        if not file_input.lower().endswith(".txt"):
            file_input += ".txt"
        
        file_candidates: list[Path] = [
            f for f in raw_text_dir.rglob(file_input) 
            if not f.name.endswith("_line_ranges.txt") and not f.name.endswith("_context.txt")
        ]
        
        if not file_candidates:
            self.ui.print_error(f"File '{file_input}' not found in {raw_text_dir}")
            sys.exit(1)
        elif len(file_candidates) == 1:
            file_path: Path = file_candidates[0]
            self.ui.print_success(f"Selected: {file_path.name}")
            return [file_path]
        else:
            result = self._select_from_multiple(file_candidates, raw_text_dir, allow_back=allow_back)
            if result is None:
                # User went back, recursively call this method again
                return self._select_single_file(raw_text_dir, allow_back=allow_back)
            return result
    
    def _select_from_multiple(self, candidates: list[Path], base_dir: Path, allow_back: bool = False) -> list[Path] | None:
        """Handle selection when multiple matching files are found."""
        assert self.ui is not None
        self.ui.print_warning(f"Found {len(candidates)} matching files:")
        self.ui.console_print(self.ui.HORIZONTAL_LINE)
        
        for idx, f in enumerate(candidates, 1):
            self.ui.console_print(f"  {idx}. {f.relative_to(base_dir)}")
        
        while True:
            selected_index = self.ui.get_input(
                "Select file by number",
                allow_back=allow_back,
                allow_quit=True
            )
            
            if not selected_index:
                return None
            
            try:
                idx = int(selected_index) - 1
                if 0 <= idx < len(candidates):
                    file_path = candidates[idx]
                    self.ui.print_success(f"Selected: {file_path.name}")
                    return [file_path]
                else:
                    self.ui.print_error(f"Please enter a number between 1 and {len(candidates)}.")
            except ValueError:
                self.ui.print_error("Invalid input. Please enter a number.")
    
    def _select_folder_files(self, raw_text_dir: Path) -> list[Path]:
        """Select all text files in a folder."""
        assert self.ui is not None
        files = [
            f for f in raw_text_dir.rglob("*.txt") 
            if not f.name.endswith("_line_ranges.txt") and not f.name.endswith("_context.txt")
        ]
        
        if not files:
            self.ui.print_error(f"No .txt files found in {raw_text_dir}")
            sys.exit(1)
        
        self.ui.print_success(f"Found {len(files)} text files to process")
        return files
    
    def _process_files(
        self,
        files: list[Path],
        verbose: bool = False,
        chunk_slice: ChunkSlice | None = None,
    ) -> tuple[int, int]:
        """
        Process files and generate line ranges.
        
        Args:
            files: List of files to process
            verbose: Whether to show verbose output
            chunk_slice: Optional slice to limit written ranges
        
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
                
                assert self.tokens_per_chunk is not None
                assert self.model_name is not None
                line_ranges = generate_line_ranges_for_file(
                    text_file=file_path,
                    default_tokens_per_chunk=self.tokens_per_chunk,
                    model_name=self.model_name,
                )
                
                # Apply chunk slice if requested
                if chunk_slice is not None and (chunk_slice.first_n is not None or chunk_slice.last_n is not None):
                    original_count = len(line_ranges)
                    if chunk_slice.first_n is not None:
                        n = min(chunk_slice.first_n, len(line_ranges))
                        line_ranges = line_ranges[:n]
                    elif chunk_slice.last_n is not None:
                        n = min(chunk_slice.last_n, len(line_ranges))
                        line_ranges = line_ranges[-n:]
                    self.print_or_log(
                        f"Chunk slice applied: writing {len(line_ranges)}/{original_count} ranges"
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
        """Run line range generation in interactive mode with back navigation support."""
        assert self.ui is not None
        self.ui.print_section_header("Line Range Generation")
        self.ui.print_info("Loading configuration...")
        
        # Get model configuration
        self.model_name, self.tokens_per_chunk = self._get_model_config()
        
        # State machine for navigation
        # States: schema -> files -> chunk_slice -> confirm
        current_step = "schema"
        state: dict[str, Any] = {}
        
        while True:
            if current_step == "schema":
                selected_schema_name = self._select_schema()
                state["selected_schema_name"] = selected_schema_name
                
                # Validate schema has paths configured
                if not validate_schema_paths(selected_schema_name, self.schemas_paths, self.ui):
                    self.logger.error(f"Exiting: No path configuration for schema '{selected_schema_name}'")
                    sys.exit(1)
                
                state["raw_text_dir"] = self._get_input_directory(selected_schema_name)
                current_step = "files"
            
            elif current_step == "files":
                files = self._select_files_interactive(state["raw_text_dir"], allow_back=True)
                if files is None:
                    current_step = "schema"
                    continue
                state["files"] = files
                current_step = "chunk_slice"
            
            elif current_step == "chunk_slice":
                chunk_slice = self.ui.ask_chunk_slice(allow_back=True)
                if chunk_slice is None:
                    current_step = "files"
                    continue
                state["chunk_slice"] = chunk_slice
                current_step = "confirm"
            
            elif current_step == "confirm":
                if not self.ui.confirm(f"Generate line ranges for {len(state['files'])} file(s)?", default=True):
                    self.ui.print_info("Operation cancelled by user.")
                    return
                # Break out of loop to start processing
                break
        
        # Process files
        self.ui.print_section_header("Generating Line Ranges")
        success_count, fail_count = self._process_files(
            state["files"], verbose=False, chunk_slice=state.get("chunk_slice")
        )
        
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
            exclude_patterns=["*_line_ranges.txt", "*_context.txt"]
        )
        
        if not files:
            self.logger.error(f"No text files found at: {input_path}")
            print(f"[ERROR] No text files found at: {input_path}")
            sys.exit(1)
        
        self.logger.info(f"Found {len(files)} file(s) to process")
        if args.verbose:
            print(f"[INFO] Processing {len(files)} file(s) with {self.tokens_per_chunk} tokens per chunk")
        
        # Build chunk slice from CLI args
        chunk_slice = None
        first_n = getattr(args, "first_n_chunks", None)
        last_n = getattr(args, "last_n_chunks", None)
        if first_n is not None:
            chunk_slice = ChunkSlice(first_n=first_n)
        elif last_n is not None:
            chunk_slice = ChunkSlice(last_n=last_n)
        
        # Process files
        success_count, fail_count = self._process_files(
            files, verbose=args.verbose, chunk_slice=chunk_slice
        )
        
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
