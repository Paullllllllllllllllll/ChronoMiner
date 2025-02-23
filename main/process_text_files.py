"""
Main script for processing text files with schema-based structured data extraction.

This script:
 - Loads configuration and prompts the user to select a schema.
 - Uses the schema-specific input and output paths.
 - Recursively processes each .txt file from the input directory (ignoring _line_ranges.txt files used for chunking).
 - Constructs API requests using the selected schema’s JSON definition and its associated developer message.
 - Writes the final output as a JSON file.
 - If enabled in paths_config.yaml, produces additional .csv, .docx, or .txt outputs.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from modules.openai_utils import open_extractor, process_text_chunk
from modules.schema_manager import SchemaManager
from modules.batching import submit_batch
from modules.text_utils import TextProcessor, perform_chunking
from modules.schema_handlers import get_schema_handler
from modules.user_interface import (ask_global_chunking_mode,
                             ask_file_chunking_method)

logger = setup_logger(__name__)

def console_print(message: str) -> None:
    print(message)

def validate_absolute_paths(paths_config: Dict[str, Any]) -> None:
    """
    Validate that all critical file paths in the configuration are absolute.
    If a relative path is found, print an error and exit gracefully.
    """
    error_found = False
    # Validate general settings (e.g., logs_dir)
    general = paths_config.get("general", {})
    logs_dir = general.get("logs_dir")
    if logs_dir and not os.path.isabs(logs_dir):
        print(f"Error: The 'logs_dir' path '{logs_dir}' is not absolute. Please use an absolute path.")
        error_found = True

    # Validate each schema's input and output paths
    schemas_paths = paths_config.get("schemas_paths", {})
    for schema, schema_config in schemas_paths.items():
        input_path = schema_config.get("input")
        output_path = schema_config.get("output")
        if input_path and not os.path.isabs(input_path):
            print(f"Error: The input path for schema '{schema}' ('{input_path}') is not absolute. Please use an absolute path.")
            error_found = True
        if output_path and not os.path.isabs(output_path):
            print(f"Error: The output path for schema '{schema}' ('{output_path}') is not absolute. Please use an absolute path.")
            error_found = True

    if error_found:
        sys.exit(1)

def load_developer_message(schema_name: str) -> str:
    """
    Automatically load the developer message corresponding to the given schema from the developer_messages folder.

    Parameters:
    - schema_name (str): The name of the extraction schema (e.g., "BibliographicEntries").

    Returns:
    - str: The contents of the corresponding developer message file.
    """
    developer_messages_dir = Path("developer_messages")
    file_name = f"{schema_name}.txt"
    file_path = developer_messages_dir / file_name
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as f:
            return f.read()
    else:
        print(f"Error: Developer message file '{file_name}' not found in {developer_messages_dir}.")
        sys.exit(1)

async def process_file(
        file_path: Path,
        paths_config: Dict[str, Any],
        model_config: Dict[str, Any],
        chunking_config: Dict[str, Any],
        use_batch: bool,
        selected_schema: Dict[str, Any],
        dev_message: str,
        schema_paths: Dict[str, Any],
        manual_adjust: bool = True,
        global_chunking_method: Optional[str] = None
) -> None:
    console_print(f"Processing file: {file_path.name}")
    logger.info(f"Starting processing for file: {file_path}")
    encoding: str = TextProcessor.detect_encoding(file_path)
    with file_path.open("r", encoding=encoding) as f:
        lines: List[str] = f.readlines()
    normalized_lines: List[str] = [TextProcessor.normalize_text(line) for line in lines]

    # Determine the chunking strategy.
    # If global_chunking_method is provided, use it; otherwise, prompt for each file.
    if global_chunking_method is not None:
        chosen_method = global_chunking_method
        console_print(f"Using default chunking method '{chosen_method}' for file {file_path.name}.")
    else:
        chosen_method = ask_file_chunking_method(file_path.name)

    if chosen_method == "auto":
        chunk_choice: str = "auto"
        line_ranges_file: Optional[Path] = None
    elif chosen_method == "auto-adjust":
        chunk_choice = "auto-adjust"
        line_ranges_file = None
    elif chosen_method == "line_ranges.txt":
        chunk_choice = "line_ranges.txt"
        line_ranges_file = file_path.with_name(f"{file_path.stem}_line_ranges.txt")
    else:
        console_print("Invalid selection, defaulting to automatic chunking.")
        chunk_choice = "auto"
        line_ranges_file = None

    openai_config_task: Dict[str, Any] = {
        "model_name": model_config["extraction_model"]["name"],
        "default_tokens_per_chunk": chunking_config["chunking"]["default_tokens_per_chunk"]
    }
    text_processor_obj: TextProcessor = TextProcessor()
    chunks, ranges = perform_chunking(
        normalized_lines, text_processor_obj,
        openai_config_task, chunk_choice, 1, line_ranges_file
    )
    logger.info(f"Total chunks generated from {file_path.name}: {len(chunks)}")

    # Determine working folders and output paths.
    if paths_config["general"]["input_paths_is_output_path"]:
        working_folder: Path = file_path.parent
        output_json_path: Path = working_folder / f"{file_path.stem}_output.json"
        temp_jsonl_path: Path = working_folder / f"{file_path.stem}_temp.jsonl"
        working_folder.mkdir(parents=True, exist_ok=True)
    else:
        working_folder = Path(schema_paths["output"])
        temp_folder: Path = working_folder / "temp_jsonl"
        working_folder.mkdir(parents=True, exist_ok=True)
        temp_folder.mkdir(parents=True, exist_ok=True)
        output_json_path = working_folder / f"{file_path.stem}_output.json"
        temp_jsonl_path = temp_folder / f"{file_path.stem}_temp.jsonl"

    results: List[Dict[str, Any]] = []
    # Use the schema handler for dynamic payload creation.
    handler = get_schema_handler(selected_schema["name"])

    # Process using batch or synchronous API calls.
    if use_batch:
        batch_requests: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, 1):
            request_obj = handler.prepare_payload(chunk, dev_message,
                                                  model_config,
                                                  selected_schema["schema"])
            request_obj["custom_id"] = f"{file_path.stem}-chunk-{idx}"
            batch_requests.append(request_obj)
        with temp_jsonl_path.open("w", encoding="utf-8") as tempf:
            for req in batch_requests:
                tempf.write(json.dumps(req) + "\n")
        logger.info(f"Wrote {len(batch_requests)} batch request(s) to {temp_jsonl_path}")
        try:
            batch_response: Any = submit_batch(temp_jsonl_path)
            tracking_record: Dict[str, Any] = {
                "batch_tracking": {
                    "batch_id": batch_response.id,
                    "timestamp": batch_response.created_at,
                    "batch_file": str(temp_jsonl_path)
                }
            }
            with temp_jsonl_path.open("a", encoding="utf-8") as tempf:
                tempf.write(json.dumps(tracking_record) + "\n")
            console_print(f"Batch submitted successfully. Batch ID: {batch_response.id}")
            logger.info(f"Batch submitted successfully. Tracking record appended to {temp_jsonl_path}")
        except Exception as e:
            logger.error(f"Error during batch submission: {e}")
            console_print(f"Error during batch submission: {e}")
    else:
        api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY is not set in environment variables.")
            console_print("Error: OPENAI_API_KEY is not set.")
            return
        async with open_extractor(
                api_key=api_key,
                prompt_path=Path("prompts/structured_output_prompt.txt"),
                model=model_config["extraction_model"]["name"]
        ) as extractor:
            with temp_jsonl_path.open("a", encoding="utf-8") as tempf:
                for idx, chunk in enumerate(chunks, 1):
                    try:
                        final_payload = handler.get_json_schema_payload(dev_message, model_config,
                                                                          selected_schema["schema"])
                        response: str = await process_text_chunk(
                            text_chunk=chunk,
                            extractor=extractor,
                            system_message=dev_message,
                            json_schema=final_payload
                        )
                        result_record: Dict[str, Any] = {
                            "custom_id": f"{file_path.stem}-chunk-{idx}",
                            "response": response,
                            "chunk_range": ranges[idx - 1]
                        }
                        tempf.write(json.dumps(result_record) + "\n")
                        results.append(result_record)
                        console_print(f"Processed chunk {idx} of {file_path.name}.")
                        logger.info(f"Processed chunk {idx} for file {file_path.name} with range {ranges[idx - 1]}")
                    except Exception as e:
                        logger.error(f"Error processing chunk {idx} of {file_path.name}: {e}")
        with output_json_path.open("w", encoding="utf-8") as outf:
            json.dump(results, outf, indent=2)
        console_print(f"Final structured JSON output saved to {output_json_path}")
        logger.info(f"Structured JSON output saved to {output_json_path}")
        if schema_paths.get("csv_output", False):
            output_csv_path: Path = output_json_path.with_suffix(".csv")
            handler.convert_to_csv(output_json_path, output_csv_path)
        if schema_paths.get("docx_output", False):
            output_docx_path: Path = output_json_path.with_suffix(".docx")
            handler.convert_to_docx(output_json_path, output_docx_path)
        if schema_paths.get("txt_output", False):
            output_txt_path: Path = output_json_path.with_suffix(".txt")
            handler.convert_to_txt(output_json_path, output_txt_path)
    if use_batch:
        logger.info("Batch processing enabled. Keeping temporary JSONL for check_batches.py.")
    else:
        keep_temp: bool = paths_config["general"].get("retain_temporary_jsonl", True)
        if not keep_temp:
            try:
                temp_jsonl_path.unlink()
                logger.info(f"Deleted temporary file: {temp_jsonl_path}")
            except Exception as e:
                logger.error(f"Error deleting temporary file {temp_jsonl_path}: {e}")

def main() -> None:
    config_loader = ConfigLoader()
    config_loader.load_configs()
    paths_config: Dict[str, Any] = config_loader.get_paths_config()
    validate_absolute_paths(paths_config)
    model_config_path: Path = Path(__file__).resolve().parent.parent / "config" / "model_config.yaml"
    with model_config_path.open('r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    chunking_config_path: Path = Path(__file__).resolve().parent.parent / "config" / "chunking_config.yaml"
    with chunking_config_path.open('r', encoding='utf-8') as f:
        chunking_config = yaml.safe_load(f)
    # Prompt for global chunking method preference
    global_chunking_method = ask_global_chunking_mode(chunking_config["chunking"]["chunking_method"])

    # Schema selection and file retrieval logic remains unchanged.
    schema_manager = SchemaManager()
    schema_manager.load_schemas()
    available_schemas = schema_manager.get_available_schemas()
    if not available_schemas:
        console_print("No schemas available. Please add schemas to the 'schemas' folder.")
        sys.exit(0)
    print("Available Schemas:")
    schema_list: List[str] = list(available_schemas.keys())
    for idx, schema_name in enumerate(schema_list, 1):
        print(f"{idx}. {schema_name}")
    selection: str = input("Select a schema by number: ").strip()
    try:
        schema_index: int = int(selection) - 1
        selected_schema_name: str = schema_list[schema_index]
    except (ValueError, IndexError):
        console_print("Invalid schema selection.")
        sys.exit(0)
    selected_schema: Dict[str, Any] = available_schemas[selected_schema_name]

    # Automatically load the developer message corresponding to the selected schema.
    dev_message: str = load_developer_message(selected_schema_name)

    # Determine input source (single file or folder).
    mode: str = input("Enter 1 to process a single file or 2 for a folder of files (or 'q' to exit): ").strip()
    if mode.lower() in ["q", "exit"]:
        console_print("Exiting.")
        sys.exit(0)
    files: List[Path] = []
    schemas_paths = config_loader.get_schemas_paths()
    if selected_schema_name in schemas_paths:
        raw_text_dir: Path = Path(schemas_paths[selected_schema_name].get("input"))
    else:
        raw_text_dir = Path(paths_config.get("input_paths", {}).get("raw_text_dir", ""))
    if mode == "1":
        file_input: str = input("Enter the filename to process (with or without .txt extension): ").strip()
        if not file_input.lower().endswith(".txt"):
            file_input += ".txt"
        file_candidates: List[Path] = list(raw_text_dir.rglob(file_input))
        if not file_candidates:
            console_print(f"File {file_input} does not exist in {raw_text_dir}.")
            sys.exit(0)
        elif len(file_candidates) == 1:
            file_path: Path = file_candidates[0]
        else:
            console_print("Multiple files found:")
            for idx, f in enumerate(file_candidates, 1):
                print(f"{idx}. {f}")
            selected_index: str = input("Select file by number: ").strip()
            try:
                idx: int = int(selected_index) - 1
                file_path = file_candidates[idx]
            except Exception as e:
                console_print("Invalid selection.")
                sys.exit(0)
        files.append(file_path)
    elif mode == "2":
        files = list(raw_text_dir.rglob("*.txt"))
        if not files:
            console_print("No .txt files found in the specified folder.")
            sys.exit(0)
    else:
        console_print("Invalid selection.")
        sys.exit(0)

    # Prompt for batch mode.
    batch_mode_input: str = input("Use batch processing? (y/n): ").strip().lower()
    use_batch: bool = batch_mode_input in ["y", "yes"]

    # Process each file asynchronously.
    async def process_all_files() -> None:
        tasks = []
        for file in files:
            tasks.append(process_file(file, paths_config, model_config, chunking_config, use_batch,
                                      selected_schema, dev_message, schemas_paths,
                                      manual_adjust=True, global_chunking_method=global_chunking_method))
        await asyncio.gather(*tasks)

    asyncio.run(process_all_files())
    console_print("Processing complete.")

if __name__ == "__main__":
    main()
