"""
Script to check whether batch jobs have finished successfully (i.e.,
are marked as completed) and—if so—to download and process them.
Temporary .jsonl files and image folders will only be deleted if the final
output is successfully written.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

from openai import OpenAI
from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from modules.text_processing import process_batch_output

logger = setup_logger(__name__)

def console_print(message: str) -> None:
    print(message)

def load_config() -> Tuple[List[Path], Dict[str, Any]]:
    """
    Load and parse configuration YAML files. Identify directories to scan and
    retrieve general processing settings (e.g., concurrency limits, keep_raw_images flag).
    """
    config_loader = ConfigLoader()
    config_loader.load_configs()
    paths_config = config_loader.get_paths_config()
    processing_settings = paths_config.get("general", {})

    file_paths = paths_config.get("file_paths", {})
    scan_dirs: List[Path] = []
    for key, folders in file_paths.items():
        for folder_key in ["input", "output"]:
            folder_path = folders.get(folder_key)
            if folder_path:
                dir_path = Path(folder_path)
                dir_path.mkdir(parents=True, exist_ok=True)
                scan_dirs.append(dir_path.resolve())
    scan_dirs = list(set(scan_dirs))
    return scan_dirs, processing_settings

def process_all_batches(root_folder: Path, processing_settings: Dict[str, Any], client: OpenAI) -> None:
    """
    Scans the root folder for *_transcription.jsonl files, locates batch IDs
    within those files, checks if the batch is completed, and if so, downloads
    the results and writes them to a final text file.
    """
    console_print(f"\n[INFO] Scanning directory '{root_folder}' for temporary batch files...")
    temp_files = list(root_folder.rglob("*_transcription.jsonl"))
    if not temp_files:
        console_print(f"[INFO] No temporary batch files found in {root_folder}.")
        logger.info(f"No temporary batch files found in {root_folder}.")
        return

    batch_map: Dict[str, Path] = {}
    for temp_file in temp_files:
        with temp_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if "batch_tracking" in record:
                        batch_id = record["batch_tracking"].get("batch_id")
                        if batch_id:
                            batch_map[batch_id] = temp_file
                except json.JSONDecodeError:
                    continue

    console_print("[INFO] Retrieving list of submitted batches from OpenAI...")
    try:
        batches = list(client.batches.list(limit=50))
    except Exception as e:
        console_print(f"[ERROR] Failed to retrieve batches from OpenAI: {e}")
        logger.exception(f"Error retrieving batches: {e}")
        return

    if not batches:
        console_print("[INFO] No batch jobs found online.")
    else:
        console_print(f"[INFO] Found {len(batches)} batch(es) online:")
        for batch in batches:
            info = f"  Batch ID: {batch.id} | Status: {batch.status}"
            console_print(info)
            logger.info(info)

    for batch in batches:
        if batch.id in batch_map and batch.status.lower() == "completed":
            temp_file = batch_map[batch.id]
            console_print(f"\n[PROCESS] Processing temporary batch file: {temp_file.name} for Batch ID: {batch.id}")
            try:
                file_obj = client.files.content(batch.output_file_id)
                file_content = file_obj.content
            # Raw response logging
            except Exception as e:
                logger.exception(f"Error downloading batch {batch.id}: {e}")
                console_print(f"[ERROR] Failed to download output for Batch ID {batch.id}: {e}")
                continue

            transcriptions = process_batch_output(file_content)
            if not transcriptions:
                logger.info(f"No transcriptions extracted for batch {batch.id}.")
                console_print(f"[WARN] No transcriptions extracted for Batch ID {batch.id}. Skipping this batch.")
                continue

            # Build the final text file path
            identifier = temp_file.stem.replace("_transcription", "")
            final_txt_path = temp_file.parent / f"{identifier}_transcription.txt"

            processing_success = False
            try:
                with final_txt_path.open("w", encoding="utf-8") as fout:
                    for text in transcriptions:
                        fout.write(text + "\n")
                logger.info(f"Batch {batch.id} results processed and saved to {final_txt_path}")
                console_print(f"[SUCCESS] Processed Batch ID {batch.id}. Results saved to {final_txt_path.name}")
                processing_success = True
            except Exception as e:
                logger.exception(f"Error writing final output for batch {batch.id}: {e}")
                console_print(f"[ERROR] Failed to write final output for Batch ID {batch.id}: {e}")

            # Delete temporary JSONL file if we successfully wrote the final output
            if processing_success and not processing_settings.get("retain_temporary_jsonl", True):
                try:
                    temp_file.unlink()
                    logger.info(f"Deleted temporary file after processing: {temp_file}")
                    console_print(f"[CLEANUP] Deleted temporary file: {temp_file.name}")
                except Exception as e:
                    logger.exception(f"Error deleting temporary file {temp_file}: {e}")
                    console_print(f"[ERROR] Could not delete temporary file {temp_file.name}: {e}")

    console_print(f"\n[INFO] Completed processing batches in directory: {root_folder}")
    logger.info(f"Batch results processing complete for directory: {root_folder}")

def main() -> None:
    """
    Entrypoint for checking all directories for batch outputs and finalizing them.
    """
    scan_dirs, processing_settings = load_config()
    client = OpenAI()
    for directory in scan_dirs:
        process_all_batches(directory, processing_settings, client)
    console_print("\n[INFO] Batch results processing complete across all directories.")
    logger.info("Batch results processing complete across all directories.")

if __name__ == "__main__":
    main()
