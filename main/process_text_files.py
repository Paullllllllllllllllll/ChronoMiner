# process_text_files.py

"""
Main script for processing text files with schema-based structured data extraction.

This script:
 - Loads configuration and prompts the user to select a schema.
 - Uses the schema-specific input and output paths.
 - Recursively processes each .txt file from the input directory (ignoring _line_ranges.txt files used for chunking).
 - Constructs API requests using the selected schema’s JSON definition and developer message.
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

logger = setup_logger(__name__)

def console_print(message: str) -> None:
    print(message)

async def process_file(
    file_path: Path,
    paths_config: Dict[str, Any],
    model_config: Dict[str, Any],
    chunking_config: Dict[str, Any],
    use_batch: bool,
    selected_schema: Dict[str, Any],
    dev_message: str,
    schema_paths: Dict[str, Any],
    manual_adjust: bool = True
) -> None:
    console_print(f"Processing file: {file_path.name}")
    logger.info(f"Starting processing for file: {file_path}")
    encoding: str = TextProcessor.detect_encoding(file_path)
    with file_path.open("r", encoding=encoding) as f:
        lines: List[str] = f.readlines()
    normalized_lines: List[str] = [TextProcessor.normalize_text(line) for line in lines]

    # Determine the chunking strategy.
    if manual_adjust:
        console_print("Select chunking strategy:")
        console_print("1. Automatic token-based chunking")
        console_print("2. Automatic token-based chunking with manual re-adjustments")
        console_print("3. Use _line_ranges.txt file (if available)")
        strategy_choice: str = input("Enter 1, 2, or 3: ").strip()
        if strategy_choice == "1":
            chunk_choice: str = "auto"
            line_ranges_file: Optional[Path] = None
        elif strategy_choice == "2":
            chunk_choice = "auto-adjust"
            line_ranges_file = None
        elif strategy_choice == "3":
            chunk_choice = "line_ranges.txt"
            line_ranges_file = file_path.with_name(file_path.stem + "_line_ranges.txt")
        else:
            console_print("Invalid selection, defaulting to automatic chunking.")
            chunk_choice = "auto"
            line_ranges_file = None
    else:
        default_line_ranges_file: Path = file_path.with_name(file_path.stem + "_line_ranges.txt")
        if default_line_ranges_file.exists():
            chunk_choice = "line_ranges.txt"
            line_ranges_file = default_line_ranges_file
            console_print(f"Using existing _line_ranges.txt for file {file_path.name}.")
            logger.info(f"Using _line_ranges.txt file for {file_path}")
        else:
            chunk_choice = "auto"
            line_ranges_file = None
            console_print(f"Using automatic token-based chunking for file {file_path.name}.")
            logger.info(f"Using automatic token-based chunking for {file_path}")

    # Build configuration for OpenAI processing.
    openai_config_task: Dict[str, Any] = {
        "model_name": model_config["extraction_model"]["name"],
        "default_tokens_per_chunk": chunking_config["default_tokens_per_chunk"]
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
            request_obj = handler.prepare_payload(chunk, dev_message, model_config, selected_schema["schema"])
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
                        final_payload = handler.get_json_schema_payload(dev_message, model_config, selected_schema["schema"])
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
        else:
            logger.info("Non-batch processing, but retain_temporary_jsonl is true. Keeping temporary JSONL.")

async def main() -> None:
    config_loader = ConfigLoader()
    config_loader.load_configs()
    paths_config: Dict[str, Any] = config_loader.get_paths_config()
    model_config_path: Path = Path(__file__).resolve().parent.parent / "config" / "model_config.yaml"
    with model_config_path.open("r", encoding="utf-8") as f:
        model_config: Dict[str, Any] = yaml.safe_load(f)["extraction_model"]
    chunking_config_path: Path = Path(__file__).resolve().parent.parent / "config" / "chunking_config.yaml"
    with chunking_config_path.open("r", encoding="utf-8") as f:
        chunking_config: Dict[str, Any] = yaml.safe_load(f)["chunking"]
    schema_manager = SchemaManager()
    schema_manager.load_schemas()
    schema_manager.load_dev_messages()
    available_schemas: Dict[str, Any] = schema_manager.get_available_schemas()
    if not available_schemas:
        console_print("No schemas available. Please add schemas to the 'schemas' folder.")
        sys.exit(1)
    console_print("Available Schemas for Structured Data Extraction:")
    schema_list: List[str] = list(available_schemas.keys())
    for idx, schema_name in enumerate(schema_list, 1):
        console_print(f"{idx}. {schema_name}")
    selection: str = input("Select a schema by number: ").strip()
    try:
        schema_index: int = int(selection) - 1
        selected_schema_name: str = schema_list[schema_index]
        selected_schema: Dict[str, Any] = available_schemas[selected_schema_name]
        selected_dev_message: Optional[str] = schema_manager.get_dev_message(selected_schema_name)
    except (ValueError, IndexError):
        console_print("Invalid schema selection.")
        sys.exit(1)
    schemas_paths: Dict[str, Any] = config_loader.get_paths_config()["schemas_paths"]
    if selected_schema_name not in schemas_paths:
        console_print(f"Schema {selected_schema_name} not found in schemas_paths.")
        sys.exit(1)
    schema_paths: Dict[str, Any] = schemas_paths[selected_schema_name]
    raw_text_dir: Path = Path(schema_paths["input"])
    console_print("Text File Processing: Choose the mode:")
    console_print("1. Process a single file (with manual adjustments)")
    console_print("2. Process all .txt files in the folder automatically")
    mode: str = input("Enter 1 or 2 (or type 'q' to exit): ").strip()
    if mode.lower() in ["q", "exit"]:
        console_print("Exiting.")
        return
    files_to_process: List[Path] = []
    manual_adjust: bool = True
    if mode == "1":
        file_input: str = input("Enter the filename to process (with or without .txt extension): ").strip()
        if file_input.lower() in ["q", "exit"]:
            console_print("Exiting.")
            return
        if not file_input.lower().endswith(".txt"):
            file_input += ".txt"
        file_candidates: List[Path] = list(raw_text_dir.rglob(file_input))
        file_candidates = [f for f in file_candidates if not f.name.endswith("_line_ranges.txt")]
        if not file_candidates:
            console_print(f"File {file_input} not found in {raw_text_dir}.")
            sys.exit(1)
        elif len(file_candidates) == 1:
            file_path = file_candidates[0]
        else:
            console_print("Multiple files found:")
            for idx, candidate in enumerate(file_candidates, 1):
                console_print(f"{idx}. {candidate.relative_to(raw_text_dir)}")
            try:
                selection = int(input("Select file by number: ").strip())
                file_path = file_candidates[selection - 1]
            except Exception as e:
                console_print("Invalid selection.")
                sys.exit(1)
        files_to_process.append(file_path)
        manual_adjust = True
    elif mode == "2":
        files_to_process = [f for f in raw_text_dir.rglob("*.txt") if not f.name.endswith("_line_ranges.txt")]
        if not files_to_process:
            console_print("No .txt files found in the specified folder.")
            sys.exit(1)
        manual_adjust = False
    else:
        console_print("Invalid selection. Exiting.")
        sys.exit(1)
    for file_path in files_to_process:
        await process_file(
            file_path=file_path,
            paths_config=paths_config,
            model_config={"extraction_model": model_config},
            chunking_config=chunking_config,
            use_batch=(input("Do you want to use batch processing? (y/n): ").strip().lower() == "y"),
            selected_schema=selected_schema,
            dev_message=selected_dev_message if selected_dev_message else "",
            schema_paths=schema_paths,
            manual_adjust=manual_adjust
        )

if __name__ == "__main__":
    asyncio.run(main())
