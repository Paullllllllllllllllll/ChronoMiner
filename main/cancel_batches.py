# main/cancel_batches.py

"""
Script to cancel all ongoing batches.

This script retrieves all batches using the OpenAI API and cancels any batch whose status is not terminal.
Terminal statuses are assumed to be: completed, expired, cancelled, or failed.
"""

from typing import Any, Set, List

from openai import OpenAI
from modules.logger import setup_logger
from modules.user_interface import UserInterface

logger = setup_logger(__name__)

# Define terminal statuses where cancellation is not applicable.
TERMINAL_STATUSES: Set[str] = {"completed", "expired", "cancelled", "failed"}


def main() -> None:
	"""
    Retrieve all batches via the OpenAI API and cancel those that are not in a terminal state.
    """
	ui = UserInterface(logger)
	client: OpenAI = OpenAI()

	ui.console_print("Retrieving list of batches...")

	try:
		batches: List[Any] = list(client.batches.list(limit=100))
	except Exception as e:
		logger.error(f"Error listing batches: {e}")
		ui.console_print(f"[ERROR] Error listing batches: {e}")
		return

	if not batches:
		ui.console_print("No batches found.")
		return

	# Display batch summary
	ui.display_batch_summary(batches)

	# Count batches that need cancellation
	cancellable_batches = [batch for batch in batches if
	                       batch.status.lower() not in TERMINAL_STATUSES]

	if not cancellable_batches:
		ui.console_print("\nNo batches require cancellation.")
		return

	ui.console_print(
		f"\nProcessing cancellations for {len(cancellable_batches)} batch(es)...")

	for batch in cancellable_batches:
		batch_id: str = batch.id
		status: str = batch.status.lower()

		ui.console_print(f"Cancelling batch {batch_id} (status: '{status}')...")

		try:
			client.batches.cancel(batch_id)
			logger.info(f"Batch {batch_id} cancelled.")
			ui.display_batch_operation_result(batch_id, "cancel", True)
		except Exception as e:
			logger.error(f"Error cancelling batch {batch_id}: {e}")
			ui.display_batch_operation_result(batch_id, "cancel", False, str(e))


if __name__ == "__main__":
	main()
