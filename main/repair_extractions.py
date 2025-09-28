# main/repair_extractions.py
"""Interactive helper for repairing incomplete batch extractions."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from modules.logger import setup_logger
from modules.user_interface import UserInterface
from modules.schema_handlers import get_schema_handler
from main.check_batches import (
    load_config,
    process_batch_output_file,
    retrieve_responses_from_batch,
    _order_responses,
    _recover_missing_batch_ids,
)
from modules.batch_utils import extract_custom_id_mapping, diagnose_batch_failure
from modules.openai_sdk_utils import list_all_batches, sdk_to_dict

logger = setup_logger(__name__)


def _discover_candidate_temp_files(repo_info_list: List[Tuple[str, Path, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for schema_name, repo_dir, schema_config in repo_info_list:
        if not repo_dir.exists():
            continue
        for temp_file in repo_dir.rglob("*_temp.jsonl"):
            try:
                result = process_batch_output_file(temp_file)
                responses = result.get("responses", [])
                tracking = result.get("tracking", [])
                identifier = temp_file.stem.replace("_temp", "")
                final_json = temp_file.parent / f"{identifier}_final_output.json"
                candidates.append(
                    {
                        "schema_name": schema_name,
                        "schema_config": schema_config,
                        "temp_file": temp_file,
                        "final_json": final_json,
                        "responses_count": len(responses),
                        "tracking_count": len(tracking),
                        "has_final": final_json.exists(),
                        "tracking": tracking,
                        "responses": responses,
                    }
                )
            except Exception as exc:
                logger.warning("Failed to inspect %s: %s", temp_file, exc)
    return candidates


def _repair_temp_file(
    candidate: Dict[str, Any],
    processing_settings: Dict[str, Any],
    client: OpenAI,
    ui: UserInterface,
) -> None:
    temp_file: Path = candidate["temp_file"]
    schema_name: str = candidate["schema_name"]
    schema_config: Dict[str, Any] = candidate["schema_config"]
    responses: List[Any] = list(candidate.get("responses", []))
    tracking: List[Any] = list(candidate.get("tracking", []))

    ui.console_print(f"\n[INFO] Repairing {temp_file.name} for schema '{schema_name}'...")

    if not tracking:
        ui.console_print("[WARN] No tracking entries found; cannot repair this file.")
        return

    custom_id_map, order_map = extract_custom_id_mapping(temp_file)
    persist_recovered = processing_settings.get("persist_recovered_batch_ids", True)

    batch_ids = {str(track.get("batch_id")) for track in tracking if track.get("batch_id")}
    recovered_ids = set()
    if not batch_ids:
        recovered_ids = _recover_missing_batch_ids(temp_file, temp_file.stem.replace("_temp", ""), persist_recovered)
        for bid in recovered_ids:
            tracking.append({"batch_id": bid})
            batch_ids.add(bid)

    if not batch_ids:
        ui.console_print("[WARN] Unable to identify any batch IDs for this temp file.")
        return

    try:
        batch_listing = list_all_batches(client)
        batch_dict = {b.get("id"): b for b in batch_listing if isinstance(b, dict) and b.get("id")}
    except Exception as exc:
        logger.warning("Unable to list all batches: %s", exc)
        batch_dict = {}

    completed_batches: List[Dict[str, Any]] = []
    failed_batches: List[Tuple[Dict[str, Any], str]] = []
    missing_batches: List[str] = []
    local_batch_cache = batch_dict.copy()

    for track in tracking:
        batch_id = track.get("batch_id")
        if not batch_id:
            continue
        batch_id = str(batch_id)
        batch = local_batch_cache.get(batch_id)
        if not batch:
            try:
                batch_obj = client.batches.retrieve(batch_id)
                batch = sdk_to_dict(batch_obj)
                local_batch_cache[batch_id] = batch
            except Exception as exc:
                failed_batches.append((track, f"error: {exc}"))
                missing_batches.append(batch_id)
                continue

        status = str(batch.get("status", "")).lower()
        if status == "completed":
            completed_batches.append(track)
        elif status in {"expired", "failed", "cancelled"}:
            if status == "failed":
                diagnosis = diagnose_batch_failure(batch_id, client)
                ui.console_print(f"[WARN] Batch {batch_id} failed: {diagnosis}")
            failed_batches.append((track, status))
        else:
            ui.console_print(f"[INFO] Batch {batch_id} is {status}; waiting for completion.")

    ui.display_batch_processing_progress(temp_file, list(batch_ids), len(completed_batches), len(missing_batches), failed_batches)

    if not completed_batches:
        ui.console_print("[INFO] No completed batches ready for repair.")
        return

    for track in completed_batches:
        batch_responses = retrieve_responses_from_batch(track, client, temp_file.parent, local_batch_cache)
        responses.extend(batch_responses)

    if not responses:
        ui.console_print("[WARN] No responses retrieved; nothing to repair.")
        return

    identifier = temp_file.stem.replace("_temp", "")
    final_json_path = temp_file.parent / f"{identifier}_final_output.json"
    ordered_responses = _order_responses(responses, order_map)

    final_results: Dict[str, Any] = {
        "responses": ordered_responses,
        "tracking": tracking,
        "processing_metadata": {
            "fully_completed": True,
            "processed_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "completed_batches": len(completed_batches),
            "failed_batches": len(failed_batches),
            "ordered_by_custom_id": True,
            "missing_batches": missing_batches,
            "recovered_batch_ids": sorted(recovered_ids),
        },
    }
    if custom_id_map:
        final_results["custom_id_map"] = custom_id_map

    final_json_path.write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    ui.console_print(f"[SUCCESS] Final output regenerated at {final_json_path.name}")
    logger.info("Repaired extraction written to %s", final_json_path)

    handler = get_schema_handler(schema_name)
    if schema_config.get("csv_output", False):
        handler.convert_to_csv_safely(final_json_path, final_json_path.with_suffix(".csv"))
    if schema_config.get("docx_output", False):
        handler.convert_to_docx_safely(final_json_path, final_json_path.with_suffix(".docx"))
    if schema_config.get("txt_output", False):
        handler.convert_to_txt_safely(final_json_path, final_json_path.with_suffix(".txt"))


def main() -> None:
    ui = UserInterface(logger)
    ui.display_banner()

    repo_info_list, processing_settings = load_config()
    candidates = _discover_candidate_temp_files(repo_info_list)

    if not candidates:
        ui.console_print("[INFO] No temporary batch files found. Nothing to repair.")
        return

    ui.console_print("\nAvailable batch temp files:")
    ui.console_print("-" * 80)
    for idx, candidate in enumerate(candidates, 1):
        temp_file = candidate["temp_file"]
        final_exists = candidate["has_final"]
        responses_count = candidate["responses_count"]
        tracking_count = candidate["tracking_count"]
        status = "FINAL EXISTS" if final_exists else "PENDING"
        ui.console_print(
            f"  {idx}. {temp_file} | schema={candidate['schema_name']} | responses={responses_count}/{tracking_count} | {status}"
        )

    selection = input("\nEnter the numbers of the files to repair (comma-separated, or 'q' to exit): ").strip()
    if selection.lower() in {"q", "quit", "exit"}:
        ui.console_print("[INFO] Repair cancelled by user.")
        return

    try:
        indices = sorted({int(part.strip()) - 1 for part in selection.split(",") if part.strip()})
    except ValueError:
        ui.console_print("[ERROR] Invalid selection; please provide comma-separated numbers.")
        sys.exit(1)

    if not indices:
        ui.console_print("[WARN] No valid selections provided. Exiting.")
        return

    client = OpenAI()
    for index in indices:
        if 0 <= index < len(candidates):
            _repair_temp_file(candidates[index], processing_settings, client, ui)
        else:
            ui.console_print(f"[WARN] Selection {index + 1} is out of range; skipping.")

    ui.console_print("\n[INFO] Repair session complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Repair interrupted by user.")
        sys.exit(0)
