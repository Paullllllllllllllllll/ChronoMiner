# main/generate_line_ranges.py

"""
Script to generate line ranges for text files.

This script selects a schema, reads a text file (or multiple files), and generates line ranges based
on token-based chunking. The line ranges are written to a '_line_ranges.txt' file.
"""

import sys
import yaml
from pathlib import Path
from typing import List, Tuple

from modules.config_loader import ConfigLoader
from modules.text_utils import TextProcessor, TokenBasedChunking
from modules.schema_manager import SchemaManager


def select_schema_for_line_ranges() -> str:
    """
    Display available schemas and prompt the user to select one.

    :return: The name of the selected schema.
    """
    schema_manager = SchemaManager()
    schema_manager.load_schemas()
    available_schemas = schema_manager.get_available_schemas()
    if not available_schemas:
        print("No schemas available. Please add schemas to the 'schemas' folder.")
        sys.exit(0)
    print("Available Schemas:")
    schema_list: List[str] = list(available_schemas.keys())
    for idx, schema_name in enumerate(schema_list, 1):
        print(f"{idx}. {schema_name}")
    selection: str = input("Select a schema by number: ").strip()
    try:
        schema_index: int = int(selection) - 1
        selected_schema_name: str = schema_list[schema_index]
        return selected_schema_name
    except (ValueError, IndexError):
        print("Invalid schema selection.")
        sys.exit(0)


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


def write_line_ranges_file(text_file: Path, line_ranges: List[Tuple[int, int]]) -> None:
    """
    Write the generated line ranges to a '_line_ranges.txt' file.

    :param text_file: The original text file.
    :param line_ranges: A list of line ranges to write.
    """
    line_ranges_file: Path = text_file.with_name(f"{text_file.stem}_line_ranges.txt")
    with line_ranges_file.open("w", encoding="utf-8") as f:
        for r in line_ranges:
            f.write(f"({r[0]}, {r[1]})\n")
    print(f"Line ranges written to {line_ranges_file}")


def main() -> None:
    """
    Main function to generate and write line ranges for selected text files.

    Loads configuration, prompts for schema selection, and processes either a single file or a folder.
    """
    config_loader = ConfigLoader()
    config_loader.load_configs()
    paths_config = config_loader.get_paths_config()

    chunking_config_path: Path = Path(__file__).resolve().parent.parent / "config" / "chunking_config.yaml"
    with chunking_config_path.open('r', encoding='utf-8') as f:
        chunking_config = yaml.safe_load(f)["chunking"]

    model_config_path: Path = Path(__file__).resolve().parent.parent / "config" / "model_config.yaml"
    with model_config_path.open('r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)["extraction_model"]
    model_name: str = model_config.get("name", "o3-mini")

    selected_schema_name: str = select_schema_for_line_ranges()
    schemas_paths = config_loader.get_schemas_paths()
    if selected_schema_name in schemas_paths:
        raw_text_dir: Path = Path(schemas_paths[selected_schema_name].get("input"))
    else:
        raw_text_dir = Path(paths_config.get("input_paths", {}).get("raw_text_dir", ""))

    mode: str = input("Enter 1 to process a single file or 2 for a folder of files (or 'q' to exit): ").strip()
    if mode.lower() in ["q", "exit"]:
        print("Exiting.")
        sys.exit(0)

    files: List[Path] = []
    if mode == "1":
        file_input: str = input("Enter the filename to process (with or without .txt extension): ").strip()
        if not file_input.lower().endswith(".txt"):
            file_input += ".txt"
        file_candidates: List[Path] = list(raw_text_dir.rglob(file_input))
        if not file_candidates:
            print(f"File {file_input} does not exist in {raw_text_dir}.")
            sys.exit(0)
        elif len(file_candidates) == 1:
            file_path: Path = file_candidates[0]
        else:
            print("Multiple files found:")
            for idx, f in enumerate(file_candidates, 1):
                print(f"{idx}. {f}")
            selected_index: str = input("Select file by number: ").strip()
            try:
                idx: int = int(selected_index) - 1
                file_path = file_candidates[idx]
            except Exception as e:
                print("Invalid selection.")
                sys.exit(0)
        files.append(file_path)
    elif mode == "2":
        files = list(raw_text_dir.rglob("*.txt"))
        if not files:
            print("No .txt files found in the specified folder.")
            sys.exit(0)
    else:
        print("Invalid selection.")
        sys.exit(0)

    for file_path in files:
        line_ranges: List[Tuple[int, int]] = generate_line_ranges_for_file(
            text_file=file_path,
            default_tokens_per_chunk=chunking_config["default_tokens_per_chunk"],
            model_name=model_name
        )
        write_line_ranges_file(file_path, line_ranges)


if __name__ == "__main__":
    main()
