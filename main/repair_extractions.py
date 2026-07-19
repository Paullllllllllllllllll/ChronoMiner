# main/repair_extractions.py
"""Interactive helper for repairing incomplete batch extractions.

Supports two execution modes:
1. Interactive Mode: User-friendly selection and confirmation
2. CLI Mode: Command-line arguments for automation
"""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from main.cli_args import create_repair_parser
from main.dual_mode import DualModeScript
from modules.batch import BatchHandle, BatchStatus, get_batch_backend
from modules.batch.diagnostics import extract_custom_id_mapping
from modules.batch.ops import (
    _recover_missing_batch_ids,
    derive_submission_output_dir,
    is_batch_temp_file,
    load_config,
    process_batch_output_file,
    retrieve_responses_from_batch,
)
from modules.extract.batch_output import (
    build_unified_batch_output,
    merge_existing_batch_output,
)
from modules.extract.schema_handlers import get_schema_handler
from modules.infra.jsonl import atomic_write_json
from modules.infra.logger import setup_logger
from modules.ui.core import UserInterface

logger = setup_logger(__name__)


def _discover_candidate_temp_files(
    repo_info_list: list[tuple[str, Path, dict[str, Any]]], ui: UserInterface
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    ui.print_info("Scanning for temporary batch files...")

    for schema_name, repo_dir, schema_config in repo_info_list:
        if not repo_dir.exists():
            ui.log(f"Repository directory does not exist: {repo_dir}", "warning")
            continue

        for temp_file in repo_dir.rglob("*_temp.jsonl"):
            if not is_batch_temp_file(temp_file):
                continue
            try:
                result = process_batch_output_file(temp_file)
                responses = result.get("responses", [])
                tracking = result.get("tracking", [])
                identifier = temp_file.stem.removesuffix("_temp")
                # CM-6: the final output lives with the submission (the
                # parent of temp_jsonl/), not next to the temp file itself.
                final_json = (
                    derive_submission_output_dir(temp_file)
                    / f"{identifier}_output.json"
                )
                # Legacy name written by pre-v1.20.0 batch finalization;
                # still counts as "final exists" for display purposes.
                legacy_final = temp_file.parent / f"{identifier}_final_output.json"
                candidates.append(
                    {
                        "schema_name": schema_name,
                        "schema_config": schema_config,
                        "temp_file": temp_file,
                        "final_json": final_json,
                        "responses_count": len(responses),
                        "tracking_count": len(tracking),
                        "has_final": final_json.exists() or legacy_final.exists(),
                        "tracking": tracking,
                        "responses": responses,
                    }
                )
                ui.log(f"Found candidate: {temp_file}", "debug")
            except Exception as exc:
                logger.warning("Failed to inspect %s: %s", temp_file, exc)
                ui.log(f"Failed to inspect {temp_file}: {exc}", "warning")

    return candidates


def _repair_temp_file(
    candidate: dict[str, Any],
    processing_settings: dict[str, Any],
    ui: UserInterface,
) -> None:
    temp_file: Path = candidate["temp_file"]
    schema_name: str = candidate["schema_name"]
    schema_config: dict[str, Any] = candidate["schema_config"]
    responses: list[Any] = list(candidate.get("responses", []))
    tracking: list[Any] = list(candidate.get("tracking", []))

    ui.print_subsection_header(f"Repairing: {temp_file.name}")
    logger.info(f"Repairing {temp_file.name} for schema '{schema_name}'")

    if not tracking:
        ui.print_warning("No tracking entries found; cannot repair this file.")
        return

    custom_id_map, order_map = extract_custom_id_mapping(temp_file)
    persist_recovered = processing_settings.get("persist_recovered_batch_ids", True)

    batch_ids = {
        str(track.get("batch_id")) for track in tracking if track.get("batch_id")
    }
    recovered_ids: set[str] = set()
    if not batch_ids:
        recovered_ids, recovered_provider = _recover_missing_batch_ids(
            temp_file, temp_file.stem.removesuffix("_temp"), persist_recovered
        )
        for bid in recovered_ids:
            track_record: dict[str, Any] = {"batch_id": bid}
            if recovered_provider:
                track_record["provider"] = recovered_provider
            tracking.append(track_record)
            batch_ids.add(bid)

    if not batch_ids:
        ui.print_warning("Unable to identify any batch IDs for this temp file.")
        return

    completed_batches: list[dict[str, Any]] = []
    failed_batches: list[tuple[dict[str, Any], str]] = []
    missing_batches: list[str] = []

    # Route status checks through the provider-agnostic backend, like
    # check_batches, so repair is not OpenAI-only.
    for track in tracking:
        batch_id = track.get("batch_id")
        if not batch_id:
            continue
        batch_id = str(batch_id)
        provider = track.get("provider", "openai")
        try:
            backend = get_batch_backend(provider)
            handle = BatchHandle(
                provider=provider,
                batch_id=batch_id,
                metadata=track.get("metadata", {}),
            )
            info = backend.get_status(handle)
        except Exception as exc:
            failed_batches.append((track, f"error: {exc}"))
            missing_batches.append(batch_id)
            continue

        status = info.status
        if status == BatchStatus.COMPLETED:
            completed_batches.append(track)
        elif status in {
            BatchStatus.EXPIRED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
        }:
            if status == BatchStatus.FAILED:
                diagnosis = info.error_message or backend.diagnose_failure(handle)
                ui.print_warning(f"Batch {batch_id} failed: {diagnosis}")
            failed_batches.append((track, status.value))
        else:
            ui.print_info(
                f"Batch {batch_id} is {status.value}; waiting for completion."
            )

    ui.display_batch_processing_progress(
        temp_file,
        list(batch_ids),
        len(completed_batches),
        len(missing_batches),
        failed_batches,
    )

    if not completed_batches:
        ui.print_info("No completed batches ready for repair.")
        return

    status_cache: dict[str, Any] = {}
    for track in completed_batches:
        batch_responses = retrieve_responses_from_batch(
            track,
            temp_file.parent,
            status_cache,
        )
        responses.extend(batch_responses)

    if not responses:
        ui.print_warning("No responses retrieved; nothing to repair.")
        return

    identifier = temp_file.stem.removesuffix("_temp")
    # CM-6: write the repaired output to the submission-local directory (the
    # parent of temp_jsonl/), matching check_batches finalization.
    output_dir = derive_submission_output_dir(temp_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_json_path = output_dir / f"{identifier}_output.json"

    # Honest completeness: only fully_completed when nothing failed or is
    # missing (previously stamped True unconditionally).
    fully_completed = not failed_batches and not missing_batches
    final_results: dict[str, Any] = build_unified_batch_output(
        responses,
        tracking,
        schema_name=schema_name,
        order_map=order_map,
        custom_id_map=custom_id_map,
        fully_completed=fully_completed,
        completed_batches=len(completed_batches),
        failed_batches=len(failed_batches),
        missing_batches=missing_batches,
        recovered_batch_ids=sorted(recovered_ids),
    )

    # Merge with any existing output so repair tops up prior records instead of
    # overwriting {identifier}_output.json with only this run's retrievable
    # subset.
    final_results = merge_existing_batch_output(final_results, final_json_path)

    # Atomic write: a crash mid-write must not destroy the merged output from
    # earlier runs.
    atomic_write_json(final_json_path, final_results)
    ui.print_success(f"Final output regenerated: {final_json_path.name}")
    logger.info("Repaired extraction written to %s", final_json_path)

    handler = get_schema_handler(schema_name)
    if schema_config.get("csv_output", False):
        try:
            handler.convert_to_csv(final_json_path, final_json_path.with_suffix(".csv"))
            ui.log("CSV output generated", "info")
        except Exception as e:
            logger.error(f"Error converting {final_json_path} to CSV: {e}")
            ui.log(f"Error converting to CSV: {e}", "error")
    if schema_config.get("docx_output", False):
        try:
            handler.convert_to_docx(
                final_json_path, final_json_path.with_suffix(".docx")
            )
            ui.log("DOCX output generated", "info")
        except Exception as e:
            logger.error(f"Error converting {final_json_path} to DOCX: {e}")
            ui.log(f"Error converting to DOCX: {e}", "error")
    if schema_config.get("txt_output", False):
        try:
            handler.convert_to_txt(final_json_path, final_json_path.with_suffix(".txt"))
            ui.log("TXT output generated", "info")
        except Exception as e:
            logger.error(f"Error converting {final_json_path} to TXT: {e}")
            ui.log(f"Error converting to TXT: {e}", "error")


class RepairExtractionsScript(DualModeScript):
    """Script to repair incomplete batch extractions."""

    def __init__(self) -> None:
        super().__init__("repair_extractions")
        # No provider key is required at construction: each batch's status and
        # results are fetched through the provider-agnostic backend, which
        # resolves its own key lazily only when actually used.
        self.repo_info_list: list[tuple[str, Path, dict[str, Any]]] = []
        self.processing_settings: dict[str, Any] = {}

    def create_argument_parser(self) -> ArgumentParser:
        """Create argument parser for CLI mode."""
        return create_repair_parser()

    def _load_repair_config(self) -> None:
        """Load configuration for repair operations."""
        self.repo_info_list, self.processing_settings = load_config()

    def run_interactive(self) -> None:
        """Run extraction repair in interactive mode."""
        assert self.ui is not None
        self.ui.print_section_header("Batch Extraction Repair")

        self._load_repair_config()
        candidates = _discover_candidate_temp_files(self.repo_info_list, self.ui)

        if not candidates:
            self.ui.print_info("No temporary batch files found. Nothing to repair.")
            return

        self.ui.print_subsection_header("Available Batch Files")
        self.ui.console_print(self.ui.HORIZONTAL_LINE)

        for idx, candidate in enumerate(candidates, 1):
            temp_file = candidate["temp_file"]
            final_exists = candidate["has_final"]
            responses_count = candidate["responses_count"]
            tracking_count = candidate["tracking_count"]
            status = "[OK] FINAL EXISTS" if final_exists else "[PENDING]"
            self.ui.console_print(f"  {idx}. {temp_file.name}")
            self.ui.console_print(
                f"      Schema: {candidate['schema_name']} | "
                f"Responses: {responses_count}/{tracking_count} | {status}"
            )

        selection = self.ui.get_input(
            "\nEnter the numbers of files to repair (comma-separated, e.g., 1,3,5)",
            allow_back=False,
            allow_quit=True,
        )

        if not selection:
            self.ui.print_info("Repair cancelled by user.")
            return

        try:
            indices = sorted(
                {int(part.strip()) - 1 for part in selection.split(",") if part.strip()}
            )
        except ValueError:
            self.ui.print_error(
                "Invalid selection. Please provide comma-separated numbers."
            )
            sys.exit(1)

        if not indices:
            self.ui.print_warning("No valid selections provided.")
            return

        # Confirm repair
        if not self.ui.confirm(f"Repair {len(indices)} file(s)?", default=True):
            self.ui.print_info("Repair cancelled by user.")
            return

        self.ui.print_section_header("Repairing Files")

        success_count = 0

        for index in indices:
            if 0 <= index < len(candidates):
                try:
                    _repair_temp_file(
                        candidates[index],
                        self.processing_settings,
                        self.ui,
                    )
                    success_count += 1
                except Exception as e:
                    self.ui.print_error(f"Failed to repair file {index + 1}: {e}")
                    self.logger.exception(f"Error repairing file at index {index}")
            else:
                self.ui.print_warning(
                    f"Selection {index + 1} is out of range; skipping."
                )

        self.ui.print_section_header("Repair Complete")
        self.ui.print_success(f"Successfully repaired {success_count} file(s)")

    def run_cli(self, args: Namespace) -> None:
        """Run extraction repair in CLI mode."""
        self.logger.info("Starting extraction repair (CLI Mode)")

        self._load_repair_config()

        # Create a simple UI-less notifier for CLI
        def cli_print(msg: str, level: str = "info") -> None:
            prefixes = {
                "success": "[SUCCESS]",
                "error": "[ERROR]",
                "warning": "[WARN]",
                "info": "[INFO]",
            }
            print(f"{prefixes.get(level, '[INFO]')} {msg}")

        # Mock UI for repair function
        class MockUI:
            def print_subsection_header(self, title: str) -> None:
                if args.verbose:
                    print(f"\n--- {title} ---")

            def print_warning(self, msg: str) -> None:
                cli_print(msg, "warning")

            def print_info(self, msg: str) -> None:
                if args.verbose:
                    cli_print(msg, "info")

            def print_success(self, msg: str) -> None:
                cli_print(msg, "success")

            def print_error(self, msg: str) -> None:
                cli_print(msg, "error")

            def log(self, msg: str, level: str) -> None:
                log_method = getattr(logger, level.lower(), logger.info)
                log_method(msg)

            def display_batch_processing_progress(self, *args: Any) -> None:
                pass

        mock_ui = MockUI()

        # Discover candidates
        candidates = _discover_candidate_temp_files(self.repo_info_list, mock_ui)  # type: ignore[arg-type]

        if not candidates:
            self.logger.info("No temporary batch files found")
            print("[INFO] No temporary batch files found. Nothing to repair.")
            return

        # Filter by schema if specified
        if args.schema:
            candidates = [c for c in candidates if c["schema_name"] == args.schema]
            if not candidates:
                self.logger.error(f"No temp files found for schema '{args.schema}'")
                print(f"[ERROR] No temp files found for schema '{args.schema}'")
                return

        # Filter by specific files if specified
        if args.files:
            file_names = set(args.files)
            candidates = [
                c
                for c in candidates
                if c["temp_file"].name in file_names
                or str(c["temp_file"]) in file_names
            ]
            if not candidates:
                self.logger.error("None of the specified files were found")
                print("[ERROR] Specified files not found")
                return

        self.logger.info(f"Found {len(candidates)} file(s) to repair")
        if args.verbose:
            print(f"[INFO] Found {len(candidates)} file(s) to repair:")
            for candidate in candidates:
                status = "FINAL EXISTS" if candidate["has_final"] else "PENDING"
                print(
                    f"  - {candidate['temp_file'].name} "
                    f"({candidate['schema_name']}) - {status}"
                )

        # Confirm unless --force
        if not args.force:
            print(f"\n[WARNING] About to repair {len(candidates)} file(s)")
            try:
                confirm = input("Proceed? (y/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n[INFO] Repair aborted")
                return

            if confirm not in ["y", "yes"]:
                self.logger.info("Repair aborted by user")
                print("[INFO] Repair aborted")
                return

        self.logger.info(f"Repairing {len(candidates)} file(s)")

        success_count = 0
        fail_count = 0

        for candidate in candidates:
            try:
                if args.verbose:
                    print(f"[INFO] Repairing {candidate['temp_file'].name}...")
                _repair_temp_file(
                    candidate,
                    self.processing_settings,
                    mock_ui,  # type: ignore[arg-type]
                )
                success_count += 1
            except Exception as e:
                self.logger.exception(f"Error repairing {candidate['temp_file'].name}")
                print(f"[ERROR] Failed to repair {candidate['temp_file'].name}: {e}")
                fail_count += 1

        # Final summary
        self.logger.info(
            f"Repair complete: {success_count} succeeded, {fail_count} failed"
        )
        print(f"[SUCCESS] Repaired {success_count}/{len(candidates)} file(s)")
        if fail_count > 0:
            print(f"[WARNING] Failed to repair {fail_count} file(s)")
            sys.exit(1)


def main() -> None:
    """Main entry point."""
    try:
        script = RepairExtractionsScript()
        script.execute()
    except KeyboardInterrupt:
        print("\n[INFO] Repair session cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
