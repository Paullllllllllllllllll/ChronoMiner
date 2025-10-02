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
from typing import List, Tuple

from modules.config.loader import ConfigLoader
from modules.core.text_utils import TextProcessor, TokenBasedChunking
from modules.core.schema_manager import SchemaManager
from modules.core.logger import setup_logger
from modules.ui.core import UserInterface
from modules.cli.args_parser import create_generate_ranges_parser, resolve_path, get_files_from_path
from modules.cli.mode_detector import should_use_interactive_mode

logger = setup_logger(__name__)


def select_schema_for_line_ranges(ui: UserInterface) -> str:
    """
    Display available schemas and prompt the user to select one.

    :param ui: UserInterface instance
    :return: The name of the selected schema.
    """
    schema_manager = SchemaManager()
    schema_manager.load_schemas()
    available_schemas = schema_manager.get_available_schemas()
    
    if not available_schemas:
        ui.print_error("No schemas available. Please add schemas to the 'schemas/' folder.")
        sys.exit(1)
    
    result = ui.select_schema(schema_manager)
    if result is None:
        ui.print_info("Schema selection cancelled.")
        sys.exit(0)
    
    _, selected_schema_name = result
    return selected_schema_name


def generate_line_ranges_for_file(
    text_file: Path, default_tokens_per_chunk: int, model_name: str
) -> List[Tuple[int, int]]:
    """
    Generate line ranges for a text file based on token-based chunking.

    :param text_file: The text file to process.
    :param default_tokens_per_chunk: The default token count per chunk.
    :param model_name: The name of the model used for token estimation.
    :return: A list of tuples representing line ranges.
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


def write_line_ranges_file(text_file: Path, line_ranges: List[Tuple[int, int]], ui: UserInterface) -> None:
    """
    Write the generated line ranges to a '_line_ranges.txt' file.

    :param text_file: The original text file.
    :param line_ranges: A list of line ranges to write.
    :param ui: UserInterface instance
    """
    line_ranges_file: Path = text_file.with_name(f"{text_file.stem}_line_ranges.txt")
    with line_ranges_file.open("w", encoding="utf-8") as f:
        for r in line_ranges:
            f.write(f"({r[0]}, {r[1]})\n")
    ui.print_success(f"Line ranges written to {line_ranges_file.name}")
    logger.info(f"Line ranges written to {line_ranges_file}")


def main() -> None:
    """
    Main function to generate and write line ranges for selected text files.

    Loads configuration, prompts for schema selection, and processes either a single file or a folder.
    """
    # Load config to determine mode
    config_loader = ConfigLoader()
    config_loader.load_configs()
    
    if should_use_interactive_mode(config_loader):
        # ============================================================
        # INTERACTIVE MODE
        # ============================================================
        ui = UserInterface(logger)
        ui.display_banner()
        ui.print_section_header("Line Range Generation")
        
        # Load configuration
        ui.print_info("Loading configuration...")
        paths_config = config_loader.get_paths_config()
        chunking_and_context_config = config_loader.get_chunking_and_context_config()
        chunking_config = chunking_and_context_config.get("chunking", {})
        model_cfg = config_loader.get_model_config().get("transcription_model", {})
        model_name: str = model_cfg.get("name", "o3-mini")

        # Select schema
        selected_schema_name: str = select_schema_for_line_ranges(ui)
        schemas_paths = config_loader.get_schemas_paths()
        
        if selected_schema_name in schemas_paths:
            raw_text_dir: Path = Path(schemas_paths[selected_schema_name].get("input"))
        else:
            raw_text_dir = Path(paths_config.get("input_paths", {}).get("raw_text_dir", ""))

        # Select input source
        ui.print_section_header("Input Selection")
        
        mode_options = [
            ("single", "Process a single file"),
            ("folder", "Process all files in a folder")
        ]
        
        mode = ui.select_option(
            "Select how you would like to specify input:",
            mode_options,
            allow_back=False,
            allow_quit=True
        )

        files: List[Path] = []
        
        if mode == "single":
            file_input = ui.get_input("Enter the filename to process (with or without .txt extension)", allow_back=False, allow_quit=True)
            
            if not file_input:
                ui.print_info("Operation cancelled.")
                sys.exit(0)
            
            if not file_input.lower().endswith(".txt"):
                file_input += ".txt"
            
            file_candidates: List[Path] = [f for f in raw_text_dir.rglob(file_input) if not f.name.endswith("_line_ranges.txt")]
            
            if not file_candidates:
                ui.print_error(f"File '{file_input}' not found in {raw_text_dir}")
                sys.exit(1)
            elif len(file_candidates) == 1:
                file_path: Path = file_candidates[0]
                files.append(file_path)
                ui.print_success(f"Selected: {file_path.name}")
            else:
                ui.print_warning(f"Found {len(file_candidates)} matching files:")
                ui.console_print(ui.HORIZONTAL_LINE)
                
                for idx, f in enumerate(file_candidates, 1):
                    ui.console_print(f"  {idx}. {f.relative_to(raw_text_dir)}")
                
                while True:
                    selected_index = ui.get_input("Select file by number", allow_back=False, allow_quit=True)
                    
                    if not selected_index:
                        sys.exit(0)
                    
                    try:
                        idx: int = int(selected_index) - 1
                        if 0 <= idx < len(file_candidates):
                            file_path = file_candidates[idx]
                            files.append(file_path)
                            ui.print_success(f"Selected: {file_path.name}")
                            break
                        else:
                            ui.print_error(f"Please enter a number between 1 and {len(file_candidates)}.")
                    except ValueError:
                        ui.print_error("Invalid input. Please enter a number.")
                        
        elif mode == "folder":
            files = [f for f in raw_text_dir.rglob("*.txt") if not f.name.endswith("_line_ranges.txt")]
            
            if not files:
                ui.print_error(f"No .txt files found in {raw_text_dir}")
                sys.exit(1)
            
            ui.print_success(f"Found {len(files)} text files to process")

        # Confirm processing
        if not ui.confirm(f"Generate line ranges for {len(files)} file(s)?", default=True):
            ui.print_info("Operation cancelled by user.")
            return

        # Process files
        ui.print_section_header("Generating Line Ranges")
        
        success_count = 0
        fail_count = 0
        
        for file_path in files:
            try:
                ui.print_info(f"Processing {file_path.name}...")
                logger.info(f"Generating line ranges for {file_path}")
                
                line_ranges: List[Tuple[int, int]] = generate_line_ranges_for_file(
                    text_file=file_path,
                    default_tokens_per_chunk=chunking_config["default_tokens_per_chunk"],
                    model_name=model_name
                )
                write_line_ranges_file(file_path, line_ranges, ui)
                success_count += 1
            except Exception as e:
                ui.print_error(f"Failed to process {file_path.name}: {e}")
                logger.exception(f"Error processing {file_path}", exc_info=e)
                fail_count += 1

        # Final summary
        ui.print_section_header("Generation Complete")
        ui.print_success(f"Successfully generated line ranges for {success_count} file(s)")
        if fail_count > 0:
            ui.print_warning(f"Failed to process {fail_count} file(s)")
    
    else:
        # ============================================================
        # CLI MODE
        # ============================================================
        parser = create_generate_ranges_parser()
        args = parser.parse_args()
        
        logger.info("Starting line range generation (CLI Mode)")
        
        # Load configuration
        paths_config = config_loader.get_paths_config()
        chunking_and_context_config = config_loader.get_chunking_and_context_config()
        chunking_config = chunking_and_context_config.get("chunking", {})
        model_cfg = config_loader.get_model_config().get("transcription_model", {})
        model_name: str = model_cfg.get("name", "o3-mini")
        
        # Get token limit
        tokens_per_chunk = args.tokens if args.tokens else chunking_config.get("default_tokens_per_chunk", 7500)
        
        # Resolve input path
        input_path = resolve_path(args.input)
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            print(f"[ERROR] Input path not found: {input_path}")
            sys.exit(1)
        
        # Get files
        files = get_files_from_path(input_path, pattern="*.txt", exclude_patterns=["*_line_ranges.txt"])
        
        if not files:
            logger.error(f"No text files found at: {input_path}")
            print(f"[ERROR] No text files found at: {input_path}")
            sys.exit(1)
        
        logger.info(f"Found {len(files)} file(s) to process")
        if args.verbose:
            print(f"[INFO] Processing {len(files)} file(s) with {tokens_per_chunk} tokens per chunk")
        
        # Process files
        success_count = 0
        fail_count = 0
        
        for file_path in files:
            try:
                if args.verbose:
                    print(f"[INFO] Processing {file_path.name}...")
                logger.info(f"Generating line ranges for {file_path}")
                
                line_ranges = generate_line_ranges_for_file(
                    text_file=file_path,
                    default_tokens_per_chunk=tokens_per_chunk,
                    model_name=model_name
                )
                
                # Write line ranges
                line_ranges_file = file_path.with_name(f"{file_path.stem}_line_ranges.txt")
                with line_ranges_file.open("w", encoding="utf-8") as f:
                    for r in line_ranges:
                        f.write(f"({r[0]}, {r[1]})\n")
                
                logger.info(f"Line ranges written to {line_ranges_file}")
                if args.verbose:
                    print(f"[SUCCESS] Created {line_ranges_file.name}")
                success_count += 1
            except Exception as e:
                logger.exception(f"Error processing {file_path}", exc_info=e)
                print(f"[ERROR] Failed to process {file_path.name}: {e}")
                fail_count += 1
        
        # Final summary
        logger.info(f"Generation complete: {success_count} succeeded, {fail_count} failed")
        print(f"[SUCCESS] Generated line ranges for {success_count}/{len(files)} file(s)")
        if fail_count > 0:
            print(f"[WARNING] Failed to process {fail_count} file(s)")


if __name__ == "__main__":
    main()
