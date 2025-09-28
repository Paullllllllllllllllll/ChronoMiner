# modules/concurrency.py

import asyncio
import logging
from typing import Any, Callable, List, Tuple

logger = logging.getLogger(__name__)


async def run_concurrent_tasks(
    corofunc: Callable[..., Any],
    args_list: List[Tuple[Any, ...]],
    concurrency_limit: int = 20,
    delay: float = 0
) -> List[Any]:
    """
    Run multiple asynchronous tasks concurrently with a limit and optional delay.

    :param corofunc: The coroutine function to be executed.
    :param args_list: A list of argument tuples for the coroutine.
    :param concurrency_limit: Maximum number of tasks to run concurrently.
    :param delay: Delay in seconds before each task.
    :return: A list of results from the executed tasks.
    """
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def worker(args: Tuple[Any, ...]) -> Any:
        async with semaphore:
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                return await corofunc(*args)
            except Exception as e:
                logger.error(f"Task failed with arguments {args}: {e}")
                return None

    tasks = [asyncio.create_task(worker(args)) for args in args_list]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results
