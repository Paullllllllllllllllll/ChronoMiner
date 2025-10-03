"""
Script to retrieve and process batch results.

This script scans all schema-specific repositories for temporary batch results, aggregates tracking
and response records, retrieves missing responses via OpenAI's API, and writes a final JSON output.
If enabled in paths_config.yaml, additional .csv, .txt, or .docx outputs are generated.

Supports two execution modes:
1. Interactive Mode: User-friendly prompts and UI feedback
2. CLI Mode: Command-line arguments for automation
"""
import json
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import datetime

from openai import OpenAI
from modules.config.loader import ConfigLoader
from modules.core.logger import setup_logger
from modules.operations.extraction.schema_handlers import get_schema_handler
from modules.ui.core import UserInterface
from modules.llm.openai_sdk_utils import list_all_batches, sdk_to_dict, coerce_file_id
from modules.core.batch_utils import diagnose_batch_failure, extract_custom_id_mapping
from modules.cli.args_parser import create_check_batches_parser, resolve_path
from modules.cli.execution_framework import DualModeScript

logger = setup_logger(__name__)

OUTPUT_FILE_KEYS = [
    "output_file_id",
    "output_file",
    "output_file_ids",
    "response_file_id",
    "response_file",
    "response_file_ids",
    "result_file_id",
    "result_file",
    "result_file_ids",
    "results_file_id",
    "results_file_ids",
]

ERROR_FILE_KEYS = [
    "error_file_id",
    "error_file",
    "error_file_ids",
    "errors_file_id",
    "errors_file_ids",
]


def _extract_chunk_index(custom_id: Any) -> int:
    """Extract numeric chunk index from a custom_id like '<stem>-chunk-<n>' or 'req-<n>'."""
    if not isinstance(custom_id, str):
        return 10**9  # push unknowns to end
    m = re.search(r"(?:-chunk-|req-)(\d+)$", custom_id)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 10**9
    return 10**9


def _order_responses(responses: List[Any], order_map: Optional[Dict[str, int]] = None) -> List[Any]:
    """Return responses sorted using explicit order_map, then chunk index."""
    try:
        sortable: List[Any] = []
        nonsortable: List[Any] = []
        for item in responses:
            if isinstance(item, dict) and ("custom_id" in item):
                sortable.append(item)
            else:
                nonsortable.append(item)

        def _sort_key(entry: Dict[str, Any]) -> Tuple[int, int]:
            cid = entry.get("custom_id")
            order_val = 10**9
            if isinstance(cid, str) and order_map and cid in order_map:
                order_val = order_map[cid]
            chunk_val = _extract_chunk_index(cid)
            return order_val, chunk_val

        sortable.sort(key=_sort_key)
        return sortable + nonsortable
    except Exception:
        return responses


def _response_to_text(response_obj: Any) -> str:
    """Normalize a Responses API payload into a plain text string."""
    if isinstance(response_obj, str):
        return response_obj
    if not isinstance(response_obj, dict):
        return ""

    if isinstance(response_obj.get("output_text"), str):
        return response_obj["output_text"].strip()

    parts: List[str] = []
    output = response_obj.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                for content_part in item.get("content", []):
                    text_val = content_part.get("text") if isinstance(content_part, dict) else None
                    if isinstance(text_val, str):
                        parts.append(text_val)
    return "".join(parts).strip()


def _normalize_response_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure a response entry has consistent text/raw keys."""
    normalized = dict(entry)
    response_payload = normalized.get("response")
    normalized.setdefault("raw_response", response_payload)
    if isinstance(response_payload, dict):
        normalized["response"] = _response_to_text(response_payload)
        normalized.setdefault("raw_response", response_payload)
    return normalized


def _resolve_file_id_by_keys(batch: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in batch:
            file_id = coerce_file_id(batch.get(key))
            if file_id:
                return file_id
    return None


def _download_error_file(
    client: OpenAI, error_file_id: str, target_dir: Path, batch_id: str
) -> Optional[Path]:
    """Download an error file for diagnostics if available."""
    try:
        response = client.files.content(error_file_id)
        blob = response.read()
        error_text = (
            blob.decode("utf-8")
            if isinstance(blob, (bytes, bytearray))
            else str(blob)
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        short_batch = batch_id.replace("batch_", "")[:16]
        error_path = target_dir / f"errors_{short_batch}.jsonl"
        with error_path.open("w", encoding="utf-8") as handle:
            handle.write(error_text)
        logger.info("Saved error details for batch %s to %s", batch_id, error_path)
        return error_path
    except Exception as exc:
        logger.warning("Failed to download error file %s for batch %s: %s", error_file_id, batch_id, exc)
    return None


def _recover_missing_batch_ids(
    temp_file: Path,
    identifier: str,
    persist: bool,
) -> Set[str]:
    recovered: Set[str] = set()
    debug_artifact = temp_file.parent / f"{identifier}_batch_submission_debug.json"
    if not debug_artifact.exists():
        return recovered

    try:
        artifact = json.loads(debug_artifact.read_text(encoding="utf-8"))
        for candidate in artifact.get("batch_ids", []) or []:
            if isinstance(candidate, str) and candidate:
                recovered.add(candidate)
    except Exception as exc:
        logger.warning("Failed to read batch debug artifact %s: %s", debug_artifact, exc)
        return recovered

    if recovered and persist:
        try:
            timestamp = datetime.datetime.now().isoformat()
            with temp_file.open("a", encoding="utf-8") as handle:
                for batch_id in recovered:
                    record = {
                        "batch_tracking": {
                            "batch_id": batch_id,
                            "timestamp": timestamp,
                            "batch_file": str(temp_file),
                        }
                    }
                    handle.write(json.dumps(record) + "\n")
            logger.info(
                "Persisted %s recovered batch id(s) into %s", len(recovered), temp_file.name
            )
        except Exception as exc:
            logger.warning("Failed to persist recovered batch ids for %s: %s", temp_file.name, exc)

    return recovered


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
					responses.append(
						_normalize_response_entry(
							{
								"response": record.get("response"),
								"custom_id": record.get("custom_id"),
								"chunk_range": record.get("chunk_range"),
							}
						)
					)
				elif "batch_tracking" in record:
					tracking.append(record["batch_tracking"])
			except Exception as e:
				logger.error(f"Error processing line in {file_path}: {e}")
	return {"responses": responses, "tracking": tracking}


def retrieve_responses_from_batch(
    tracking_record: Dict[str, Any],
    client: OpenAI,
    temp_dir: Path,
    batch_cache: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    responses: List[Dict[str, Any]] = []
    batch_id: Any = tracking_record.get("batch_id")
    if not batch_id:
        logger.error("No batch_id found in tracking record.")
        return responses

    batch_id = str(batch_id)
    batch: Optional[Dict[str, Any]] = batch_cache.get(batch_id)
    if not batch:
        try:
            batch_obj = client.batches.retrieve(batch_id)
            batch = sdk_to_dict(batch_obj)
            batch_cache[batch_id] = batch
        except Exception as exc:
            logger.error(f"Error retrieving batch {batch_id}: {exc}")
            return responses

    output_file_id = _resolve_file_id_by_keys(batch, OUTPUT_FILE_KEYS)
    if not output_file_id:
        error_file_id = _resolve_file_id_by_keys(batch, ERROR_FILE_KEYS)
        if error_file_id:
            _download_error_file(client, error_file_id, temp_dir, batch_id)
        logger.error(f"No output file id found for batch {batch_id}.")
        return responses

    try:
        file_stream = client.files.content(output_file_id)
        blob = file_stream.read()
        text = (
            blob.decode("utf-8")
            if isinstance(blob, (bytes, bytearray))
            else str(blob)
        )
    except Exception as exc:
        logger.error(f"Error downloading batch output for {batch_id}: {exc}")
        return responses

    for line in text.splitlines():
        payload = line.strip()
        if not payload:
            continue
        try:
            out_record: Dict[str, Any] = json.loads(payload)
        except json.JSONDecodeError:
            logger.error(
                "Error parsing a line from batch output %s for batch %s", output_file_id, batch_id
            )
            continue

        if "response" in out_record:
            responses.append(
                _normalize_response_entry(
                    {
                        "response": out_record.get("response"),
                        "custom_id": out_record.get("custom_id"),
                    }
                )
            )

    if not responses:
        logger.warning(f"No responses retrieved for batch {batch_id}.")
    return responses


def _safe_print(ui: Optional[UserInterface], message: str, level: str = "info"):
	"""Safely print message to UI or logger depending on mode."""
	if ui:
		if level == "info":
			ui.print_info(message)
		elif level == "warning":
			ui.print_warning(message)
		elif level == "error":
			ui.print_error(message)
		elif level == "success":
			ui.print_success(message)
		else:
			ui.log(message, level)
	else:
		# CLI mode - use logger and print for verbose output
		if level == "info":
			logger.info(message)
		elif level == "warning":
			logger.warning(message)
		elif level == "error":
			logger.error(message)
		else:
			logger.info(message)


def _safe_subsection(ui: Optional[UserInterface], title: str):
	"""Safely print subsection header."""
	if ui:
		ui.print_subsection_header(title)
	else:
		logger.info(f"=== {title} ===")


def process_all_batches(
		root_folder: Path,
		processing_settings: Dict[str, Any],
		client: OpenAI,
		schema_name: str,
		schema_config: Dict[str, Any],
		ui: Optional[UserInterface]
) -> None:
	temp_files: List[Path] = list(root_folder.rglob("*_temp.jsonl"))
	if not temp_files:
		_safe_print(ui, f"No temporary batch files found in {root_folder}", "info")
		logger.info(f"No temporary batch files found in {root_folder}.")
		return

	try:
		batch_listing = list_all_batches(client)
		batch_dict: Dict[str, Dict[str, Any]] = {
			b.get("id"): b for b in batch_listing if isinstance(b, dict) and b.get("id")
		}
	except Exception as exc:
		logger.warning("Unable to list all batches (falling back to on-demand retrieval): %s", exc)
		batch_dict = {}

	for temp_file in temp_files:
		try:
			_safe_subsection(ui, f"Processing: {temp_file.name}")
			logger.info(f"Processing temporary batch file: {temp_file}")

			# Load batch tracking info and responses
			results: Dict[str, Any] = process_batch_output_file(temp_file)
			responses: List[Any] = results.get("responses", [])
			tracking: List[Any] = results.get("tracking", [])
			custom_id_map, order_map = extract_custom_id_mapping(temp_file)

			if not tracking:
				_safe_print(ui, f"Tracking information missing in {temp_file.name}. Skipping final output.", "warning")
				logger.warning(f"Tracking information missing in {temp_file}. Skipping final output.")
				continue

			persist_recovered = processing_settings.get("persist_recovered_batch_ids", True)
			batch_ids: Set[str] = {
				str(track.get("batch_id"))
				for track in tracking
				if track.get("batch_id")
			}
			recovered_ids = set()
			if not batch_ids:
				recovered_ids = _recover_missing_batch_ids(temp_file, temp_file.stem.replace("_temp", ""), persist_recovered)
				for recovered in recovered_ids:
					tracking.append({"batch_id": recovered})
					batch_ids.add(recovered)

			if not batch_ids:
				_safe_print(ui, f"No batch IDs found for {temp_file.name}. Unable to finalize this file.", "warning")
				logger.warning("No batch IDs recovered for %s", temp_file)
				continue

			# Check batch status and retrieve completed results
			all_finished: bool = True
			completed_batches = []
			failed_batches = []
			missing_batches: List[str] = []
			local_batch_cache: Dict[str, Dict[str, Any]] = batch_dict.copy()

			for track in tracking:
				batch_id: Any = track.get("batch_id")
				if not batch_id:
					logger.error(f"Missing batch_id in tracking record in {temp_file}")
				batch_id = str(batch_id)
				batch: Optional[Dict[str, Any]] = local_batch_cache.get(batch_id)
				if not batch:
					try:
						batch_obj: Any = client.batches.retrieve(batch_id)
						batch = sdk_to_dict(batch_obj)
						local_batch_cache[batch_id] = batch
					except Exception as exc:
						logger.error(f"Error retrieving batch {batch_id}: {exc}")
						_safe_print(ui, f"Failed to retrieve batch {batch_id}", "error")
						failed_batches.append((track, f"error: {exc}"))
						missing_batches.append(batch_id)
						all_finished = False
						continue

				status: str = str(batch.get("status", "")).lower()
				if status == "completed":
					completed_batches.append(track)
				elif status in {"expired", "failed", "cancelled"}:
					if status == "failed":
						diagnosis: str = diagnose_batch_failure(batch_id, client)
						_safe_print(ui, f"Batch {batch_id} failed: {diagnosis}", "warning")
						logger.warning(f"Batch {batch_id} failed: {diagnosis}")
					failed_batches.append((track, status))
					all_finished = False
				else:
					_safe_print(ui, f"Batch {batch_id} status: {status} (waiting for completion)", "info")
					logger.info(f"Batch {batch_id} is {status}; not finished.")
					all_finished = False

			# Display progress
			if ui:
				ui.display_batch_processing_progress(
					temp_file, list(batch_ids), len(completed_batches), len(missing_batches), failed_batches
				)

			if not all_finished:
				_safe_print(ui, f"Not all batches are completed for {temp_file.name}. Skipping finalization.", "warning")
				logger.info(f"Skipping finalization for {temp_file} (incomplete batches).")
				continue

			# Retrieve responses from completed batches
			for track in completed_batches:
				batch_responses: List[Any] = retrieve_responses_from_batch(
					track, client, temp_file.parent, local_batch_cache
				)
				responses.extend(batch_responses)

			if not responses:
				_safe_print(ui, f"No responses retrieved for {temp_file.name}. Cannot finalize.", "warning")
				logger.warning(f"No responses retrieved for {temp_file}.")
				continue

			# Order responses
			ordered_responses: List[Any] = _order_responses(responses, order_map)

			# Write final output
			identifier: str = temp_file.stem.replace("_temp", "")
			final_json_path: Path = temp_file.parent / f"{identifier}_final_output.json"

			final_results: Dict[str, Any] = {
				"responses": ordered_responses,
				"tracking": tracking,
				"processing_metadata": {
					"fully_completed": all_finished,
					"processed_at": datetime.datetime.now(
						datetime.timezone.utc
					).isoformat(),
					"completed_batches": len(completed_batches),
					"failed_batches": len(failed_batches),
					"ordered_by_custom_id": True,
					"missing_batches": missing_batches,
					"recovered_batch_ids": sorted(recovered_ids),
				},
			}
			if custom_id_map:
				final_results["custom_id_map"] = custom_id_map

			final_json_path.write_text(
				json.dumps(final_results, indent=2), encoding="utf-8"
			)
			_safe_print(ui, f"Final output written: {final_json_path.name}", "success")
			logger.info(f"Final output written to {final_json_path}")

			# Generate additional output formats
			handler = get_schema_handler(schema_name)
			if schema_config.get("csv_output", False):
				handler.convert_to_csv_safely(
					final_json_path, final_json_path.with_suffix(".csv")
				)
				_safe_print(ui, f"CSV output generated for {identifier}", "info")
			if schema_config.get("docx_output", False):
				handler.convert_to_docx_safely(
					final_json_path, final_json_path.with_suffix(".docx")
				)
				_safe_print(ui, f"DOCX output generated for {identifier}", "info")
			if schema_config.get("txt_output", False):
				handler.convert_to_txt_safely(
					final_json_path, final_json_path.with_suffix(".txt")
				)
				_safe_print(ui, f"TXT output generated for {identifier}", "info")

			# Optionally remove temp file
			if not processing_settings.get("retain_temporary_jsonl", False):
				temp_file.unlink()
				_safe_print(ui, f"Removed temporary file: {temp_file.name}", "info")
				logger.info(f"Removed temporary file {temp_file}")

		except Exception as exc:
			logger.exception(f"Error processing {temp_file}", exc_info=exc)
			_safe_print(ui, f"Failed to process {temp_file.name}: {exc}", "error")


class CheckBatchesScript(DualModeScript):
    """Script to check and retrieve batch processing results."""
    
    def __init__(self):
        super().__init__("check_batches")
        self.client: OpenAI = OpenAI()
        self.repo_info_list: List[Tuple[str, Path, Dict[str, Any]]] = []
        self.processing_settings: Dict[str, Any] = {}
    
    def create_argument_parser(self) -> ArgumentParser:
        """Create argument parser for CLI mode."""
        return create_check_batches_parser()
    
    def _load_batch_config(self) -> None:
        """Load configuration for batch processing."""
        self.repo_info_list, self.processing_settings = load_config()
    
    def run_interactive(self) -> None:
        """Run batch checking in interactive mode."""
        self.ui.print_section_header("Batch Results Retrieval")
        
        self._load_batch_config()
        
        self.ui.print_info("Scanning for batch files across all schemas...")
        self.logger.info("Starting batch results retrieval process.")
        
        for schema_name, repo_dir, schema_config in self.repo_info_list:
            if not repo_dir.exists():
                self.ui.log(f"Repository directory does not exist: {repo_dir}", "warning")
                continue
            
            self.ui.print_subsection_header(f"Schema: {schema_name}")
            self.ui.print_info(f"Processing directory: {repo_dir}")
            self.logger.info(f"Processing schema {schema_name} in directory {repo_dir}")
            
            process_all_batches(
                root_folder=repo_dir,
                processing_settings=self.processing_settings,
                client=self.client,
                schema_name=schema_name,
                schema_config=schema_config,
                ui=self.ui,
            )
        
        self.ui.print_section_header("Batch Processing Complete")
        self.ui.print_success("All batch results have been processed")
    
    def run_cli(self, args: Namespace) -> None:
        """Run batch checking in CLI mode."""
        from argparse import Namespace
        
        self.logger.info("Starting batch results retrieval (CLI Mode)")
        
        self._load_batch_config()
        
        # Filter by schema if specified
        if args.schema:
            self.repo_info_list = [
                (name, dir, cfg) for name, dir, cfg in self.repo_info_list 
                if name == args.schema
            ]
            if not self.repo_info_list:
                self.logger.error(f"Schema '{args.schema}' not found in configuration")
                print(f"[ERROR] Schema '{args.schema}' not found")
                return
        
        # Override with input path if specified
        if args.input:
            input_path = resolve_path(args.input)
            if not input_path.exists():
                self.logger.error(f"Input path does not exist: {input_path}")
                print(f"[ERROR] Input path not found: {input_path}")
                return
            # Use single directory with first schema config as template
            if self.repo_info_list:
                schema_name, _, schema_config = self.repo_info_list[0]
                self.repo_info_list = [(schema_name, input_path, schema_config)]
            else:
                self.logger.error("No schema configuration available")
                print("[ERROR] No schema configuration found")
                return
        
        if not self.repo_info_list:
            self.logger.info("No repositories to process")
            print("[INFO] No batch files found")
            return
        
        self.logger.info(f"Scanning {len(self.repo_info_list)} schema(s) for batch files")
        if args.verbose:
            print(f"[INFO] Scanning {len(self.repo_info_list)} schema(s)")
        
        for schema_name, repo_dir, schema_config in self.repo_info_list:
            if not repo_dir.exists():
                self.logger.warning(f"Repository directory does not exist: {repo_dir}")
                continue
            
            if args.verbose:
                print(f"[INFO] Processing schema: {schema_name}")
            self.logger.info(f"Processing schema {schema_name} in directory {repo_dir}")
            
            process_all_batches(
                root_folder=repo_dir,
                processing_settings=self.processing_settings,
                client=self.client,
                schema_name=schema_name,
                schema_config=schema_config,
                ui=None,
            )
        
        self.logger.info("Batch processing complete")
        if args.verbose:
            print("[SUCCESS] Batch processing complete")


def main() -> None:
    """Main entry point."""
    script = CheckBatchesScript()
    script.execute()


if __name__ == "__main__":
    main()
