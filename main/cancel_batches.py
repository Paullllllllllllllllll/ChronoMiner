# main/cancel_batches.py

"""
Script to cancel all ongoing batches.

This script scans for batch tracking records in temp JSONL files and cancels any batch 
whose status is not terminal. Supports multiple providers:
- OpenAI: Uses OpenAI Batch API
- Anthropic: Uses Anthropic Message Batches API
- Google: Uses Google Gemini Batch API

Terminal statuses are assumed to be: completed, expired, cancelled, or failed.

Supports two execution modes:
1. Interactive Mode: User prompts with confirmation
2. CLI Mode: Command-line arguments with optional --force flag
"""

import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from modules.cli.args_parser import create_cancel_batches_parser
from modules.cli.execution_framework import DualModeScript
from modules.config.loader import get_config_loader
from modules.ui.core import UserInterface
from modules.llm.batch import (
    BatchHandle,
    BatchStatus,
    get_batch_backend,
    supports_batch,
)

# Define terminal statuses where cancellation is not applicable
TERMINAL_STATUSES: Set[BatchStatus] = {
    BatchStatus.COMPLETED, 
    BatchStatus.EXPIRED, 
    BatchStatus.CANCELLED, 
    BatchStatus.FAILED
}


def _scan_for_batch_tracking(root_folders: List[Path]) -> List[Dict[str, Any]]:
    """
    Scan temp JSONL files for batch tracking records.
    
    Returns list of tracking records with batch_id and provider.
    """
    tracking_records: List[Dict[str, Any]] = []
    
    for root_folder in root_folders:
        if not root_folder.exists():
            continue
        
        temp_files = list(root_folder.rglob("*_temp*.jsonl"))
        for temp_file in temp_files:
            try:
                with temp_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            if "batch_tracking" in record:
                                tracking = record["batch_tracking"]
                                tracking["source_file"] = str(temp_file)
                                tracking_records.append(tracking)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
    
    return tracking_records


class CancelBatchesScript(DualModeScript):
    """Script to cancel all ongoing batches.
    
    Supports multiple providers:
    - OpenAI: Uses OpenAI Batch API
    - Anthropic: Uses Anthropic Message Batches API
    - Google: Uses Google Gemini Batch API
    
    Provider detection is automatic based on tracking records in temp files.
    """
    
    def __init__(self):
        super().__init__("cancel_batches")
        # No longer require any API key at init - backends handle their own keys
        self.root_folders: List[Path] = []
    
    def create_argument_parser(self) -> ArgumentParser:
        """Create argument parser for CLI mode."""
        return create_cancel_batches_parser()
    
    def _load_root_folders(self) -> None:
        """Load root folders from paths_config."""
        try:
            config_loader = get_config_loader()
            paths_config = config_loader.get_paths_config()
            general = paths_config.get("general", {})
            input_paths_is_output_path = general.get("input_paths_is_output_path", False)
            schemas_paths = paths_config.get("schemas_paths", {})
            
            for schema_config in schemas_paths.values():
                folder = Path(
                    schema_config["input"] if input_paths_is_output_path 
                    else schema_config["output"]
                )
                if folder not in self.root_folders:
                    self.root_folders.append(folder)
        except Exception as e:
            self.logger.warning(f"Failed to load paths config: {e}")
    
    def _get_cancellable_batches(self) -> List[Tuple[Dict[str, Any], BatchStatus]]:
        """
        Get list of batches that can be cancelled with their current status.
        
        Returns list of (tracking_record, status) tuples.
        """
        tracking_records = _scan_for_batch_tracking(self.root_folders)
        cancellable: List[Tuple[Dict[str, Any], BatchStatus]] = []
        
        for tracking in tracking_records:
            batch_id = tracking.get("batch_id")
            provider = tracking.get("provider", "openai")  # Default to openai for backward compatibility
            
            if not batch_id:
                continue
            
            try:
                backend = get_batch_backend(provider)
                handle = BatchHandle(provider=provider, batch_id=batch_id, metadata=tracking.get("metadata", {}))
                status_info = backend.get_status(handle)
                
                if status_info.status not in TERMINAL_STATUSES:
                    cancellable.append((tracking, status_info.status))
            except Exception as e:
                self.logger.warning(f"Failed to get status for batch {batch_id} ({provider}): {e}")
        
        return cancellable
    
    def run_interactive(self) -> None:
        """Run batch cancellation in interactive mode."""
        self.ui.print_section_header("Batch Cancellation")
        
        self._load_root_folders()
        
        self.ui.print_info("Scanning for batch tracking records...")
        self.logger.info("Starting batch cancellation process.")
        
        cancellable_batches = self._get_cancellable_batches()
        
        if not cancellable_batches:
            self.ui.print_info("No batches require cancellation. All batches are in terminal states or no batches found.")
            self.logger.info("No batches require cancellation.")
            return
        
        self.ui.print_subsection_header("Cancellation Process")
        
        # Group by provider
        by_provider: Dict[str, List[Tuple[Dict[str, Any], BatchStatus]]] = {}
        for tracking, status in cancellable_batches:
            provider = tracking.get("provider", "openai")
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append((tracking, status))
        
        # Display summary
        for provider, batches in by_provider.items():
            self.ui.print_info(f"{provider}: {len(batches)} batch(es)")
        
        self.ui.print_warning(f"Found {len(cancellable_batches)} batch(es) that can be cancelled")
        
        # Ask for confirmation
        if not self.ui.confirm(
            f"Do you want to cancel {len(cancellable_batches)} batch(es)?", 
            default=False
        ):
            self.ui.print_info("Cancellation aborted by user.")
            self.logger.info("User aborted batch cancellation.")
            return
        
        self.ui.print_info(f"Processing cancellations for {len(cancellable_batches)} batch(es)...")
        
        # Cancel batches
        cancelled_count, failed_count = self._cancel_batches(cancellable_batches, self.ui)
        
        # Display summary
        self.ui.print_section_header("Cancellation Complete")
        self.ui.print_success(f"Successfully cancelled {cancelled_count} batch(es)")
        if failed_count > 0:
            self.ui.print_warning(f"Failed to cancel {failed_count} batch(es)")
        
        self.logger.info(f"Batch cancellation complete: {cancelled_count} cancelled, {failed_count} failed.")
    
    def run_cli(self, args: Namespace) -> None:
        """Run batch cancellation in CLI mode."""
        self.logger.info("Starting batch cancellation process (CLI mode).")
        
        self._load_root_folders()
        
        print("[INFO] Scanning for batch tracking records...")
        cancellable_batches = self._get_cancellable_batches()
        
        if not cancellable_batches:
            print("[INFO] No batches require cancellation. All batches are in terminal states or no batches found.")
            self.logger.info("No batches require cancellation.")
            return
        
        # Group by provider for display
        by_provider: Dict[str, int] = {}
        for tracking, status in cancellable_batches:
            provider = tracking.get("provider", "openai")
            by_provider[provider] = by_provider.get(provider, 0) + 1
        
        for provider, count in by_provider.items():
            print(f"[INFO] {provider}: {count} batch(es)")
        
        print(f"[INFO] Found {len(cancellable_batches)} batch(es) that can be cancelled")
        
        # Check for force flag
        if not args.force:
            print("[ERROR] Use --force flag to confirm batch cancellation")
            self.logger.info("Cancellation aborted: --force flag not provided")
            return
        
        print(f"[INFO] Processing cancellations for {len(cancellable_batches)} batch(es)...")
        
        # Cancel batches
        cancelled_count, failed_count = self._cancel_batches(cancellable_batches)
        
        # Display summary
        print(f"[SUCCESS] Successfully cancelled {cancelled_count} batch(es)")
        if failed_count > 0:
            print(f"[WARNING] Failed to cancel {failed_count} batch(es)")
        
        self.logger.info(f"Batch cancellation complete: {cancelled_count} cancelled, {failed_count} failed.")
    
    def _cancel_batches(
        self, 
        batches: List[Tuple[Dict[str, Any], BatchStatus]], 
        ui: UserInterface = None
    ) -> Tuple[int, int]:
        """
        Cancel a list of batches using provider-agnostic backends.
        
        Args:
            batches: List of (tracking_record, status) tuples
            ui: Optional UserInterface for progress display
        
        Returns:
            Tuple of (cancelled_count, failed_count)
        """
        cancelled_count = 0
        failed_count = 0
        
        for tracking, status in batches:
            batch_id = tracking.get("batch_id", "")
            provider = tracking.get("provider", "openai")
            
            msg = f"Cancelling batch {batch_id} ({provider}, status: '{status.value}')..."
            if ui:
                ui.print_info(msg)
            else:
                print(f"[INFO] {msg}")
            
            self.logger.info(f"Attempting to cancel batch {batch_id} ({provider}) with status '{status.value}'.")
            
            try:
                backend = get_batch_backend(provider)
                handle = BatchHandle(provider=provider, batch_id=batch_id, metadata=tracking.get("metadata", {}))
                success = backend.cancel(handle)
                
                if success:
                    self.logger.info(f"Batch {batch_id} cancelled successfully.")
                    if ui:
                        ui.display_batch_operation_result(batch_id, "cancel", True)
                    else:
                        print(f"[SUCCESS] Batch {batch_id} cancelled")
                    cancelled_count += 1
                else:
                    self.logger.warning(f"Batch {batch_id} cancellation returned False.")
                    if ui:
                        ui.display_batch_operation_result(batch_id, "cancel", False, "Cancellation returned False")
                    else:
                        print(f"[WARNING] Batch {batch_id} cancellation may not have succeeded")
                    failed_count += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to cancel batch {batch_id}: {e}")
                
                if ui:
                    ui.display_batch_operation_result(batch_id, "cancel", False, str(e))
                else:
                    print(f"[ERROR] Failed to cancel batch {batch_id}: {e}")
                
                failed_count += 1
        
        return cancelled_count, failed_count


def main() -> None:
    """Main entry point."""
    script = CancelBatchesScript()
    script.execute()


if __name__ == "__main__":
    main()
