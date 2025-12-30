"""
Script to retrieve and process batch results.

This script scans all schema-specific repositories for temporary batch results, aggregates tracking
and response records, retrieves missing responses via provider-specific APIs, and writes a final JSON output.
If enabled in paths_config.yaml, additional .csv, .txt, or .docx outputs are generated.

Supports multiple providers:
- OpenAI: Uses OpenAI Batch API
- Anthropic: Uses Anthropic Message Batches API  
- Google: Uses Google Gemini Batch API

Supports two execution modes:
1. Interactive Mode: User-friendly prompts and UI feedback
2. CLI Mode: Command-line arguments for automation
"""
import json
import os
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import datetime

from modules.config.loader import ConfigLoader, get_config_loader
from modules.core.logger import setup_logger
from modules.operations.extraction.schema_handlers import get_schema_handler
from modules.ui.core import UserInterface
from modules.core.batch_utils import diagnose_batch_failure, extract_custom_id_mapping
from modules.cli.args_parser import create_check_batches_parser, resolve_path
from modules.cli.execution_framework import DualModeScript
from modules.llm.batch import (
    BatchHandle,
    BatchStatus,
    BatchStatusInfo,
    BatchResultItem,
    get_batch_backend,
    supports_batch,
)

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


def _group_temp_files_by_base(temp_files: List[Path]) -> Dict[str, List[Path]]:
    """Group temp files by their base identifier (removing _part suffixes).
    
    For example:
    - file_temp.jsonl -> base: file_temp
    - file_temp_part1.jsonl, file_temp_part2.jsonl -> base: file_temp
    """
    groups: Dict[str, List[Path]] = {}
    
    for temp_file in temp_files:
        stem = temp_file.stem
        # Remove _part{n} suffix if present
        base_match = re.match(r"(.+?)(?:_part\d+)?$", stem)
        if base_match:
            base_identifier = base_match.group(1)
        else:
            base_identifier = stem
        
        if base_identifier not in groups:
            groups[base_identifier] = []
        groups[base_identifier].append(temp_file)
    
    # Sort files within each group by part number
    for base_id in groups:
        groups[base_id].sort(key=lambda p: (
            int(re.search(r"_part(\d+)$", p.stem).group(1))
            if re.search(r"_part(\d+)$", p.stem)
            else 0
        ))
    
    return groups


def _get_output_directory(schema_config: Dict[str, Any], paths_config: Dict[str, Any]) -> Path:
    """Determine the output directory from schema configuration."""
    # Always use the output directory from schema config
    output_dir = schema_config.get("output")
    if output_dir:
        return Path(output_dir)
    
    # Fallback: if output is not specified, this shouldn't happen but handle it
    raise ValueError("Output directory not specified in schema configuration")


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


def is_batch_finished(batch_id: str, provider: str = "openai") -> bool:
    """Check if a batch is finished using the appropriate provider backend."""
    try:
        backend = get_batch_backend(provider)
        handle = BatchHandle(provider=provider, batch_id=batch_id)
        status_info = backend.get_status(handle)
        if status_info.status in {BatchStatus.COMPLETED, BatchStatus.EXPIRED, BatchStatus.CANCELLED, BatchStatus.FAILED}:
            return True
        else:
            logger.info(
                f"Batch {batch_id} status is '{status_info.status.value}', not finished yet.")
            return False
    except Exception as e:
        logger.error(f"Error retrieving batch {batch_id}: {e}")
        return False


def load_config() -> Tuple[
    List[Tuple[str, Path, Dict[str, Any]]], Dict[str, Any]]:
    config_loader = get_config_loader()
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
    temp_dir: Path,
    status_cache: Dict[str, BatchStatusInfo],
) -> List[Dict[str, Any]]:
    """Retrieve responses from a batch using the appropriate provider backend."""
    responses: List[Dict[str, Any]] = []
    batch_id: Any = tracking_record.get("batch_id")
    provider: str = tracking_record.get("provider", "openai")  # Default to openai for backward compatibility
    
    if not batch_id:
        logger.error("No batch_id found in tracking record.")
        return responses

    batch_id = str(batch_id)
    
    try:
        backend = get_batch_backend(provider)
        handle = BatchHandle(provider=provider, batch_id=batch_id, metadata=tracking_record.get("metadata", {}))
        
        # Download results using the provider-agnostic backend
        for result_item in backend.download_results(handle):
            if result_item.success:
                # Build response entry compatible with existing format
                response_entry = {
                    "custom_id": result_item.custom_id,
                    "response": result_item.content,
                    "raw_response": result_item.raw_response,
                }
                # Include parsed output if available
                if result_item.parsed_output:
                    response_entry["parsed_output"] = result_item.parsed_output
                responses.append(_normalize_response_entry(response_entry))
            else:
                logger.warning(
                    f"Request {result_item.custom_id} in batch {batch_id} failed: {result_item.error}"
                )
                # Include failed responses with error info
                responses.append({
                    "custom_id": result_item.custom_id,
                    "response": None,
                    "error": result_item.error,
                    "error_code": result_item.error_code,
                })
                
    except Exception as exc:
        logger.error(f"Error downloading batch results for {batch_id} ({provider}): {exc}")
        return responses

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
        schema_name: str,
        schema_config: Dict[str, Any],
        ui: Optional[UserInterface]
) -> None:
    """Process all batch results using provider-agnostic backends."""
    temp_files: List[Path] = list(root_folder.rglob("*_temp*.jsonl"))
    if not temp_files:
        _safe_print(ui, f"No temporary batch files found in {root_folder}", "info")
        logger.info(f"No temporary batch files found in {root_folder}.")
        return
    
    # Group temp files by base identifier (handling split files)
    file_groups = _group_temp_files_by_base(temp_files)
    
    # Determine output directory
    try:
        # Get paths_config to pass to _get_output_directory
        config_loader = get_config_loader()
        paths_config = config_loader.get_paths_config()
        output_dir = _get_output_directory(schema_config, paths_config)
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error(f"Failed to determine output directory: {exc}")
        _safe_print(ui, f"Failed to determine output directory: {exc}", "error")
        return

    # Status cache for batch status info (provider-agnostic)
    status_cache: Dict[str, BatchStatusInfo] = {}

    # Process each group of temp files (handling merged outputs for split files)
    for base_identifier, temp_file_group in file_groups.items():
        try:
            if len(temp_file_group) > 1:
                _safe_subsection(ui, f"Processing merged file: {base_identifier} ({len(temp_file_group)} parts)")
                logger.info(f"Processing {len(temp_file_group)} split files for base: {base_identifier}")
            else:
                _safe_subsection(ui, f"Processing: {temp_file_group[0].name}")
                logger.info(f"Processing temporary batch file: {temp_file_group[0]}")
            
            # Aggregate responses and tracking from all parts
            all_responses: List[Any] = []
            all_tracking: List[Any] = []
            combined_custom_id_map: Dict[str, Any] = {}
            combined_order_map: Dict[str, int] = {}

            # Load batch tracking info and responses from all parts
            for temp_file in temp_file_group:
                results: Dict[str, Any] = process_batch_output_file(temp_file)
                all_responses.extend(results.get("responses", []))
                all_tracking.extend(results.get("tracking", []))
                
                # Merge custom_id mappings
                custom_id_map, order_map = extract_custom_id_mapping(temp_file)
                if custom_id_map:
                    combined_custom_id_map.update(custom_id_map)
                if order_map:
                    # Offset order_map indices to account for multiple parts
                    offset = len(combined_order_map)
                    for cid, idx in order_map.items():
                        combined_order_map[cid] = idx + offset
            
            responses = all_responses
            tracking = all_tracking
            custom_id_map = combined_custom_id_map if combined_custom_id_map else None
            order_map = combined_order_map if combined_order_map else None

            if not tracking:
                _safe_print(ui, f"Tracking information missing for {base_identifier}. Skipping final output.", "warning")
                logger.warning(f"Tracking information missing for {base_identifier}. Skipping final output.")
                continue

            persist_recovered = processing_settings.get("persist_recovered_batch_ids", True)
            batch_ids: Set[str] = {
                str(track.get("batch_id"))
                for track in tracking
                if track.get("batch_id")
            }
            recovered_ids = set()
            if not batch_ids:
                # Try to recover from any of the temp files in the group
                for temp_file in temp_file_group:
                    temp_identifier = temp_file.stem.replace("_temp", "")
                    recovered = _recover_missing_batch_ids(temp_file, temp_identifier, persist_recovered)
                    recovered_ids.update(recovered)
                    for batch_id in recovered:
                        tracking.append({"batch_id": batch_id})
                        batch_ids.add(batch_id)

            if not batch_ids:
                _safe_print(ui, f"No batch IDs found for {base_identifier}. Unable to finalize this file.", "warning")
                logger.warning("No batch IDs recovered for %s", base_identifier)
                continue

            # Check batch status and retrieve completed results using provider-agnostic backends
            all_finished: bool = True
            completed_batches = []
            failed_batches = []
            missing_batches: List[str] = []

            for track in tracking:
                batch_id: Any = track.get("batch_id")
                provider: str = track.get("provider", "openai")  # Default to openai for backward compatibility
                
                if not batch_id:
                    logger.error(f"Missing batch_id in tracking record for {base_identifier}")
                    continue
                batch_id = str(batch_id)
                
                # Use provider-agnostic backend to check status
                try:
                    backend = get_batch_backend(provider)
                    handle = BatchHandle(provider=provider, batch_id=batch_id, metadata=track.get("metadata", {}))
                    status_info = backend.get_status(handle)
                    status_cache[batch_id] = status_info
                except Exception as exc:
                    logger.error(f"Error retrieving batch {batch_id} ({provider}): {exc}")
                    _safe_print(ui, f"Batch {batch_id} not found (may have expired or been deleted)", "error")
                    failed_batches.append((track, f"not found: {exc}"))
                    missing_batches.append(batch_id)
                    all_finished = False
                    continue

                status = status_info.status
                if status == BatchStatus.COMPLETED:
                    completed_batches.append(track)
                    _safe_print(ui, f"Batch {batch_id} ({provider}): completed ✓", "success")
                    logger.info(f"Batch {batch_id} ({provider}) completed.")
                elif status in {BatchStatus.EXPIRED, BatchStatus.FAILED, BatchStatus.CANCELLED}:
                    if status == BatchStatus.FAILED:
                        diagnosis: str = status_info.error_message or backend.diagnose_failure(handle)
                        _safe_print(ui, f"Batch {batch_id} failed: {diagnosis}", "warning")
                        logger.warning(f"Batch {batch_id} failed: {diagnosis}")
                    else:
                        _safe_print(ui, f"Batch {batch_id}: {status.value}", "warning")
                        logger.warning(f"Batch {batch_id} is {status.value}.")
                    failed_batches.append((track, status.value))
                    all_finished = False
                else:
                    _safe_print(ui, f"Batch {batch_id}: {status.value} (still processing...)", "info")
                    logger.info(f"Batch {batch_id} is {status.value}; not finished.")
                    all_finished = False

            # Display progress
            if ui:
                ui.display_batch_processing_progress(
                    temp_file_group[0], list(batch_ids), len(completed_batches), len(missing_batches), failed_batches
                )

            if not all_finished:
                in_progress_count = len([t for t in tracking if t.get("batch_id") not in missing_batches]) - len(completed_batches)
                if in_progress_count > 0:
                    _safe_print(ui, f"{in_progress_count} batch(es) still processing. Run this script again once complete.", "info")
                else:
                    _safe_print(ui, f"Cannot finalize {base_identifier} - some batches failed or expired.", "warning")
                logger.info(f"Skipping finalization for {base_identifier} (incomplete batches).")
                continue

            # Retrieve responses from completed batches using provider-agnostic backends
            for track in completed_batches:
                batch_responses: List[Any] = retrieve_responses_from_batch(
                    track, temp_file_group[0].parent, status_cache
                )
                responses.extend(batch_responses)

            if not responses:
                _safe_print(ui, f"No responses retrieved for {base_identifier}. Cannot finalize.", "warning")
                logger.warning(f"No responses retrieved for {base_identifier}.")
                continue

            # Order responses
            ordered_responses: List[Any] = _order_responses(responses, order_map)

            # Write final output to output directory (not temp file parent)
            # Remove _temp suffix from base identifier
            final_identifier = base_identifier.replace("_temp", "")
            final_json_path: Path = output_dir / f"{final_identifier}_final_output.json"

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
            _safe_print(ui, f"Final output written to: {final_json_path}", "success")
            logger.info(f"Final output written to {final_json_path}")
            
            # Log info about merged files
            if len(temp_file_group) > 1:
                _safe_print(ui, f"Merged {len(temp_file_group)} split files into single output", "info")
            
            # Summary of processed chunks
            num_responses = len(ordered_responses)
            num_batches = len(completed_batches)
            _safe_print(
                ui, 
                f"✓ Successfully processed {num_responses} chunk(s) from {num_batches} batch(es) for '{final_identifier}'",
                "success"
            )

            # Generate additional output formats
            handler = get_schema_handler(schema_name)
            if schema_config.get("csv_output", False):
                try:
                    handler.convert_to_csv(final_json_path, final_json_path.with_suffix(".csv"))
                    _safe_print(ui, f"CSV output generated for {final_identifier}", "info")
                except Exception as e:
                    logger.error(f"Error converting {final_json_path} to CSV: {e}")
                    _safe_print(ui, f"Error converting to CSV: {e}", "error")
            if schema_config.get("docx_output", False):
                try:
                    handler.convert_to_docx(final_json_path, final_json_path.with_suffix(".docx"))
                    _safe_print(ui, f"DOCX output generated for {final_identifier}", "info")
                except Exception as e:
                    logger.error(f"Error converting {final_json_path} to DOCX: {e}")
                    _safe_print(ui, f"Error converting to DOCX: {e}", "error")
            if schema_config.get("txt_output", False):
                try:
                    handler.convert_to_txt(final_json_path, final_json_path.with_suffix(".txt"))
                    _safe_print(ui, f"TXT output generated for {final_identifier}", "info")
                except Exception as e:
                    logger.error(f"Error converting {final_json_path} to TXT: {e}")
                    _safe_print(ui, f"Error converting to TXT: {e}", "error")

            # Optionally remove temp files (all parts in the group)
            if not processing_settings.get("retain_temporary_jsonl", False):
                for temp_file in temp_file_group:
                    temp_file.unlink()
                    _safe_print(ui, f"Removed temporary file: {temp_file.name}", "info")
                    logger.info(f"Removed temporary file {temp_file}")

        except Exception as exc:
            logger.exception(f"Error processing {base_identifier}", exc_info=exc)
            _safe_print(ui, f"Failed to process {base_identifier}: {exc}", "error")


class CheckBatchesScript(DualModeScript):
    """Script to check and retrieve batch processing results.
    
    Supports multiple providers:
    - OpenAI: Uses OpenAI Batch API
    - Anthropic: Uses Anthropic Message Batches API
    - Google: Uses Google Gemini Batch API
    
    Provider detection is automatic based on tracking records in temp files.
    """
    
    def __init__(self):
        super().__init__("check_batches")
        # No longer require OPENAI_API_KEY at init - provider backends handle their own keys
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
                schema_name=schema_name,
                schema_config=schema_config,
                ui=self.ui,
            )
        
        self.ui.print_section_header("Batch Processing Complete")
        self.ui.print_success("All batch results have been processed")
    
    def run_cli(self, args: Namespace) -> None:
        """Run batch checking in CLI mode."""
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
