# main/cancel_batches.py

"""
Script to cancel all ongoing batches.

This script retrieves all batches using the OpenAI API and cancels any batch whose status is not terminal.
Terminal statuses are assumed to be: completed, expired, cancelled, or failed.
"""

from typing import Any, Set

from openai import OpenAI
from modules.logger import setup_logger

logger = setup_logger(__name__)

# Define terminal statuses where cancellation is not applicable.
TERMINAL_STATUSES: Set[str] = {"completed", "expired", "cancelled", "failed"}


def main() -> None:
    """
    Retrieve all batches via the OpenAI API and cancel those that are not in a terminal state.
    """
    client: OpenAI = OpenAI()
    print("Retrieving list of batches...")
    try:
        batches: list[Any] = list(client.batches.list(limit=100))
    except Exception as e:
        logger.error(f"Error listing batches: {e}")
        print(f"Error listing batches: {e}")
        return

    if not batches:
        print("No batches found.")
        return

    print(f"Found {len(batches)} batches. Processing cancellations...")
    for batch in batches:
        batch_id: str = batch.id
        status: str = batch.status.lower()
        if status in TERMINAL_STATUSES:
            logger.info(f"Skipping batch {batch_id} with terminal status '{status}'.")
            print(f"Skipping batch {batch_id} with terminal status '{status}'.")
            continue

        print(f"Cancelling batch {batch_id} (status: '{status}')...")
        try:
            client.batches.cancel(batch_id)
            logger.info(f"Batch {batch_id} cancelled.")
            print(f"Batch {batch_id} cancelled.")
        except Exception as e:
            logger.error(f"Error cancelling batch {batch_id}: {e}")
            print(f"Error cancelling batch {batch_id}: {e}")


if __name__ == "__main__":
    main()
