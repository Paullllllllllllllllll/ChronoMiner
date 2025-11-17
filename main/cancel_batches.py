# main/cancel_batches.py

"""
Script to cancel all ongoing batches.

This script retrieves all batches using the OpenAI API and cancels any batch whose status is not terminal.
Terminal statuses are assumed to be: completed, expired, cancelled, or failed.

Supports two execution modes:
1. Interactive Mode: User prompts with confirmation
2. CLI Mode: Command-line arguments with optional --force flag
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, List, Set

from openai import OpenAI

from modules.cli.args_parser import create_cancel_batches_parser
from modules.cli.execution_framework import DualModeScript
from modules.ui.core import UserInterface

# Define terminal statuses where cancellation is not applicable
TERMINAL_STATUSES: Set[str] = {"completed", "expired", "cancelled", "failed"}


class CancelBatchesScript(DualModeScript):
    """Script to cancel all ongoing OpenAI batches."""
    
    def __init__(self):
        super().__init__("cancel_batches")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set or is empty")
        self.client: OpenAI = OpenAI(api_key=api_key)
    
    def create_argument_parser(self) -> ArgumentParser:
        """Create argument parser for CLI mode."""
        return create_cancel_batches_parser()
    
    def run_interactive(self) -> None:
        """Run batch cancellation in interactive mode."""
        self.ui.print_section_header("Batch Cancellation")
        
        self.ui.print_info("Retrieving list of batches from OpenAI...")
        self.logger.info("Starting batch cancellation process.")
        
        try:
            batches: List[Any] = list(self.client.batches.list(limit=100))
        except Exception as e:
            self.logger.error(f"Error listing batches: {e}")
            self.ui.print_error(f"Failed to retrieve batches: {e}")
            return
        
        if not batches:
            self.ui.print_info("No batches found.")
            self.logger.info("No batches found to cancel.")
            return
        
        # Display batch summary
        self.ui.display_batch_summary(batches)
        
        # Count batches that need cancellation
        cancellable_batches = [
            batch for batch in batches 
            if batch.status.lower() not in TERMINAL_STATUSES
        ]
        
        if not cancellable_batches:
            self.ui.print_info("No batches require cancellation. All batches are in terminal states.")
            self.logger.info("No batches require cancellation.")
            return
        
        self.ui.print_subsection_header("Cancellation Process")
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
        
        try:
            batches: List[Any] = list(self.client.batches.list(limit=100))
        except Exception as e:
            self.logger.error(f"Error listing batches: {e}")
            print(f"[ERROR] Failed to retrieve batches: {e}")
            return
        
        if not batches:
            print("[INFO] No batches found.")
            self.logger.info("No batches found to cancel.")
            return
        
        # Count cancellable batches
        cancellable_batches = [
            batch for batch in batches 
            if batch.status.lower() not in TERMINAL_STATUSES
        ]
        
        if not cancellable_batches:
            print("[INFO] No batches require cancellation. All batches are in terminal states.")
            self.logger.info("No batches require cancellation.")
            return
        
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
        batches: List[Any], 
        ui: UserInterface = None
    ) -> tuple[int, int]:
        """
        Cancel a list of batches.
        
        Args:
            batches: List of batch objects to cancel
            ui: Optional UserInterface for progress display
        
        Returns:
            Tuple of (cancelled_count, failed_count)
        """
        cancelled_count = 0
        failed_count = 0
        
        for batch in batches:
            batch_id: str = batch.id
            status: str = batch.status.lower()
            
            msg = f"Cancelling batch {batch_id} (status: '{status}')..."
            if ui:
                ui.print_info(msg)
            else:
                print(f"[INFO] {msg}")
            
            self.logger.info(f"Attempting to cancel batch {batch_id} with status '{status}'.")
            
            try:
                self.client.batches.cancel(batch_id)
                self.logger.info(f"Batch {batch_id} cancelled successfully.")
                
                if ui:
                    ui.display_batch_operation_result(batch_id, "cancel", True)
                else:
                    print(f"[SUCCESS] Batch {batch_id} cancelled")
                
                cancelled_count += 1
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
