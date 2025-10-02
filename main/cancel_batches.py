# main/cancel_batches.py

"""
Script to cancel all ongoing batches.

This script retrieves all batches using the OpenAI API and cancels any batch whose status is not terminal.
Terminal statuses are assumed to be: completed, expired, cancelled, or failed.

Supports two execution modes:
1. Interactive Mode: User prompts with confirmation
2. CLI Mode: Command-line arguments with optional --force flag
"""

from typing import Any, Set, List

from openai import OpenAI
from modules.core.logger import setup_logger
from modules.ui.core import UserInterface
from modules.config.loader import ConfigLoader
from modules.cli.args_parser import create_cancel_batches_parser
from modules.cli.mode_detector import should_use_interactive_mode

logger = setup_logger(__name__)

# Define terminal statuses where cancellation is not applicable.
TERMINAL_STATUSES: Set[str] = {"completed", "expired", "cancelled", "failed"}


def main() -> None:
	"""
    Retrieve all batches via the OpenAI API and cancel those that are not in a terminal state.
    """
	# Load config to determine mode
	config_loader = ConfigLoader()
	config_loader.load_configs()
	
	client: OpenAI = OpenAI()
	
	if should_use_interactive_mode(config_loader):
		# ============================================================
		# INTERACTIVE MODE
		# ============================================================
		ui = UserInterface(logger)
		ui.display_banner()
		ui.print_section_header("Batch Cancellation")
		
		ui.print_info("Retrieving list of batches from OpenAI...")
		logger.info("Starting batch cancellation process.")

		try:
			batches: List[Any] = list(client.batches.list(limit=100))
		except Exception as e:
			logger.error(f"Error listing batches: {e}")
			ui.print_error(f"Failed to retrieve batches: {e}")
			return

		if not batches:
			ui.print_info("No batches found.")
			logger.info("No batches found to cancel.")
			return

		# Display batch summary
		ui.display_batch_summary(batches)

		# Count batches that need cancellation
		cancellable_batches = [batch for batch in batches if
		                       batch.status.lower() not in TERMINAL_STATUSES]

		if not cancellable_batches:
			ui.print_info("No batches require cancellation. All batches are in terminal states.")
			logger.info("No batches require cancellation.")
			return

		ui.print_subsection_header("Cancellation Process")
		ui.print_warning(f"Found {len(cancellable_batches)} batch(es) that can be cancelled")
		
		# Ask for confirmation
		if not ui.confirm(f"Do you want to cancel {len(cancellable_batches)} batch(es)?", default=False):
			ui.print_info("Cancellation aborted by user.")
			logger.info("User aborted batch cancellation.")
			return

		ui.print_info(f"Processing cancellations for {len(cancellable_batches)} batch(es)...")

		cancelled_count = 0
		failed_count = 0

		for batch in cancellable_batches:
			batch_id: str = batch.id
			status: str = batch.status.lower()

			ui.print_info(f"Cancelling batch {batch_id} (status: '{status}')...")
			logger.info(f"Attempting to cancel batch {batch_id} with status '{status}'.")

			try:
				client.batches.cancel(batch_id)
				logger.info(f"Batch {batch_id} cancelled successfully.")
				ui.display_batch_operation_result(batch_id, "cancel", True)
				cancelled_count += 1
			except Exception as e:
				logger.error(f"Error cancelling batch {batch_id}: {e}")
				ui.display_batch_operation_result(batch_id, "cancel", False, str(e))
				failed_count += 1

		# Final summary
		ui.print_section_header("Cancellation Summary")
		ui.print_success(f"Successfully cancelled: {cancelled_count} batch(es)")
		if failed_count > 0:
			ui.print_warning(f"Failed to cancel: {failed_count} batch(es)")
		logger.info(f"Batch cancellation complete: {cancelled_count} successful, {failed_count} failed.")
	
	else:
		# ============================================================
		# CLI MODE
		# ============================================================
		parser = create_cancel_batches_parser()
		args = parser.parse_args()
		
		logger.info("Starting batch cancellation (CLI Mode)")
		
		try:
			batches: List[Any] = list(client.batches.list(limit=100))
		except Exception as e:
			logger.error(f"Error listing batches: {e}")
			print(f"[ERROR] Failed to retrieve batches: {e}")
			return

		if not batches:
			logger.info("No batches found to cancel")
			print("[INFO] No batches found")
			return

		# Count batches that need cancellation
		cancellable_batches = [batch for batch in batches if
		                       batch.status.lower() not in TERMINAL_STATUSES]

		if not cancellable_batches:
			logger.info("No batches require cancellation")
			print("[INFO] No batches require cancellation. All batches are in terminal states.")
			return

		logger.info(f"Found {len(cancellable_batches)} cancellable batch(es)")
		if args.verbose:
			print(f"[INFO] Found {len(cancellable_batches)} batch(es) that can be cancelled")
			for batch in cancellable_batches:
				print(f"  - {batch.id} (status: {batch.status})")
		
		# Ask for confirmation unless --force is used
		if not args.force:
			print(f"\n[WARNING] About to cancel {len(cancellable_batches)} batch(es)")
			confirm = input("Proceed? (y/N): ").strip().lower()
			if confirm not in ['y', 'yes']:
				logger.info("Cancellation aborted by user")
				print("[INFO] Cancellation aborted")
				return
		
		logger.info(f"Processing cancellations for {len(cancellable_batches)} batch(es)")
		
		cancelled_count = 0
		failed_count = 0

		for batch in cancellable_batches:
			batch_id: str = batch.id
			status: str = batch.status.lower()

			if args.verbose:
				print(f"[INFO] Cancelling {batch_id} (status: {status})...")
			logger.info(f"Attempting to cancel batch {batch_id}")

			try:
				client.batches.cancel(batch_id)
				logger.info(f"Batch {batch_id} cancelled successfully")
				if args.verbose:
					print(f"[SUCCESS] Cancelled {batch_id}")
				cancelled_count += 1
			except Exception as e:
				logger.error(f"Error cancelling batch {batch_id}: {e}")
				print(f"[ERROR] Failed to cancel {batch_id}: {e}")
				failed_count += 1

		# Final summary
		logger.info(f"Cancellation complete: {cancelled_count} successful, {failed_count} failed")
		print(f"[SUCCESS] Cancelled {cancelled_count}/{len(cancellable_batches)} batch(es)")
		if failed_count > 0:
			print(f"[WARNING] Failed to cancel {failed_count} batch(es)")


if __name__ == "__main__":
	main()
