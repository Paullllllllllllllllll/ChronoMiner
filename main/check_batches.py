# main/check_batches.py

"""
Script to retrieve and process batch results.

This script scans all schema-specific repositories for temporary batch results, aggregates tracking
and response records, retrieves missing responses via OpenAI's API, and writes a final JSON output.
If enabled in paths_config.yaml, additional .csv, .txt, or .docx outputs are generated.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import datetime

from openai import OpenAI
from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from modules.schema_handlers import get_schema_handler
from modules.user_interface import UserInterface

logger = setup_logger(__name__)


def is_batch_finished(batch_id: str, client: OpenAI) -> bool:
	try:
		batch_info: Any = client.batches.retrieve(batch_id)
		status: str = batch_info.status.lower()
		if status in {"completed", "expired", "cancelled", "failed"}:
			return True
		else:
			logger.info(
				f"Batch {batch_id} status is '{status}', not finished yet.")
			return False
	except Exception as e:
		logger.error(f"Error retrieving batch {batch_id}: {e}")
		return False


def load_config() -> Tuple[
	List[Tuple[str, Path, Dict[str, Any]]], Dict[str, Any]]:
	config_loader = ConfigLoader()
	config_loader.load_configs()
	paths_config: Dict[str, Any] = config_loader.get_paths_config()
	general: Dict[str, Any] = paths_config["general"]
	input_paths_is_output_path: bool = general["input_paths_is_output_path"]
	schemas_paths: Dict[str, Any] = paths_config["schemas_paths"]
	repo_info_list: List[Tuple[str, Path, Dict[str, Any]]] = []
	for schema, schema_config in schemas_paths.items():
		folder: Path = Path(
			schema_config["input"]) if input_paths_is_output_path else Path(
			schema_config["output"])
		repo_info_list.append((schema, folder, schema_config))
	processing_settings: Dict[str, Any] = {
		"retain_temporary_jsonl": general["retain_temporary_jsonl"]}
	return repo_info_list, processing_settings


def process_batch_output_file(file_path: Path) -> Dict[str, List[Any]]:
	responses: List[Any] = []
	tracking: List[Any] = []
	with file_path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				record: Dict[str, Any] = json.loads(line)
				if "response" in record:
					responses.append(record["response"])
				elif "batch_tracking" in record:
					tracking.append(record["batch_tracking"])
			except Exception as e:
				logger.error(f"Error processing line in {file_path}: {e}")
	return {"responses": responses, "tracking": tracking}


def retrieve_responses_from_batch(tracking_record: Dict[str, Any],
                                  client: OpenAI) -> List[Any]:
	responses: List[Any] = []
	batch_id: Any = tracking_record.get("batch_id")
	if not batch_id:
		logger.error("No batch_id found in tracking record.")
		return responses
	try:
		batch_info: Any = client.batches.retrieve(batch_id)
		if isinstance(batch_info, dict):
			output_file_id = batch_info.get("output_file_id") or batch_info.get(
				"result_file_id")
		else:
			output_file_id = getattr(batch_info, "output_file_id",
			                         None) or getattr(batch_info,
			                                          "result_file_id", None)
		if not output_file_id:
			logger.error(f"No output file id found for batch {batch_id}.")
			return responses
		output_file_content: str = client.files.content(output_file_id).text
		for line in output_file_content.splitlines():
			line = line.strip()
			if not line:
				continue
			try:
				out_record: Dict[str, Any] = json.loads(line)
				if "response" in out_record:
					responses.append(out_record["response"])
			except Exception as e:
				logger.error(
					f"Error processing a line from batch output file: {e}")
	except Exception as e:
		logger.error(f"Error retrieving batch {batch_id}: {e}")
	if not responses:
		logger.warning(f"No responses retrieved for batch {batch_id}.")
	return responses


def process_all_batches(
		root_folder: Path,
		processing_settings: Dict[str, Any],
		client: OpenAI,
		schema_name: str,
		schema_config: Dict[str, Any],
		ui: UserInterface
) -> None:
	temp_files: List[Path] = list(root_folder.rglob("*_temp.jsonl"))
	if not temp_files:
		ui.console_print(f"No temporary batch files found in {root_folder}.")
		logger.info(f"No temporary batch files found in {root_folder}.")
		return

	for temp_file in temp_files:
		try:
			ui.console_print(f"\nProcessing batch file: {temp_file.name}")
			logger.info(f"Processing temporary batch file: {temp_file}")

			# Load batch tracking info and responses
			results: Dict[str, Any] = process_batch_output_file(temp_file)
			responses: List[Any] = results.get("responses", [])
			tracking: List[Any] = results.get("tracking", [])

			if not tracking:
				ui.console_print(
					f"[WARNING] Tracking information missing in {temp_file.name}. Skipping final output.")
				logger.warning(
					f"Tracking information missing in {temp_file}. Skipping final output.")
				continue

			# Check batch status and retrieve completed results
			all_finished: bool = True
			completed_batches = []
			failed_batches = []

			for track in tracking:
				batch_id: Any = track.get("batch_id")
				if not batch_id:
					logger.error(
						f"Missing batch_id in tracking record in {temp_file}")
					continue

				try:
					batch_info = client.batches.retrieve(batch_id)
					status: str = batch_info.status.lower()

					if status in {"completed"}:
						completed_batches.append(track)
					elif status in {"expired", "failed", "cancelled"}:
						failed_batches.append((track, status))
						all_finished = False
					else:
						ui.console_print(
							f"[INFO] Batch {batch_id} is still in progress (status: {status}). Skipping for now.")
						logger.info(
							f"Batch {batch_id} is still in progress (status: {status}).")
						all_finished = False
				except Exception as e:
					logger.error(f"Error retrieving batch {batch_id}: {e}")
					failed_batches.append((track, f"error: {e}"))
					all_finished = False

			# Process completed batches
			if not completed_batches and not all_finished:
				ui.console_print(
					f"[INFO] No completed batches found for {temp_file.name}. Try running this script again later.")
				continue

			# Retrieve responses from completed batches
			for track in completed_batches:
				batch_responses = retrieve_responses_from_batch(track, client)
				responses.extend(batch_responses)

			if not responses:
				ui.console_print(
					f"[WARNING] No responses retrieved for {temp_file.name}. Check for errors in the batch processing.")
				continue

			# Generate final output files
			identifier: str = temp_file.stem.replace("_temp", "")
			final_json_path: Path = temp_file.parent / f"{identifier}_final_output.json"

			final_results: Dict[str, Any] = {
				"responses": responses,
				"tracking": tracking,
				"processing_metadata": {
					"fully_completed": all_finished,
					"processed_at": datetime.datetime.now().isoformat(),
					"completed_batches": len(completed_batches),
					"failed_batches": len(failed_batches)
				}
			}

			with final_json_path.open("w", encoding="utf-8") as fout:
				json.dump(final_results, fout, indent=2)

			ui.console_print(
				f"[SUCCESS] Saved final output to {final_json_path.name}")
			logger.info(f"Final batch results saved to {final_json_path}")

			# Generate additional output formats if configured
			handler = get_schema_handler(schema_name)
			if schema_config.get("csv_output", False):
				output_csv_path: Path = final_json_path.with_suffix(".csv")
				handler.convert_to_csv_safely(final_json_path, output_csv_path)
			if schema_config.get("docx_output", False):
				output_docx_path: Path = final_json_path.with_suffix(".docx")
				handler.convert_to_docx_safely(final_json_path,
				                               output_docx_path)
			if schema_config.get("txt_output", False):
				output_txt_path: Path = final_json_path.with_suffix(".txt")
				handler.convert_to_txt_safely(final_json_path, output_txt_path)

			# Clean up temporary files if configured
			if not processing_settings.get("retain_temporary_jsonl",
			                               True) and all_finished:
				try:
					temp_file.unlink()
					logger.info(f"Deleted temporary file: {temp_file}")
				except Exception as e:
					logger.error(
						f"Error deleting temporary file {temp_file}: {e}")

		except Exception as e:
			logger.error(f"Error processing batch file {temp_file}: {e}")
			ui.console_print(
				f"[ERROR] Error processing batch file {temp_file.name}: {e}")


def main() -> None:
	repo_info_list, processing_settings = load_config()
	client: OpenAI = OpenAI()
	ui = UserInterface(logger)

	ui.console_print("Retrieving list of submitted batches...")

	try:
		batches: List[Any] = list(client.batches.list(limit=50))
	except Exception as e:
		ui.console_print(f"[ERROR] Error retrieving batches: {e}")
		logger.error(f"Error retrieving batches: {e}")
		return

	if not batches:
		ui.console_print("No batch jobs found online.")
	else:
		# Display batch summary instead of listing all batches
		ui.display_batch_summary(batches)

	for schema_name, repo_dir, schema_config in repo_info_list:
		ui.console_print(
			f"\n[INFO] Processing batches for schema: {schema_name} in directory: {repo_dir}")
		logger.info(
			f"Starting batch processing for schema: {schema_name} in {repo_dir}")
		process_all_batches(repo_dir, processing_settings, client, schema_name,
		                    schema_config, ui)

	ui.console_print("[SUCCESS] Batch results processing complete.")
	logger.info("Batch results processing complete.")


if __name__ == "__main__":
	main()
