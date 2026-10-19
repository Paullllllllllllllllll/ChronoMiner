"""Bounded worker pools that run the calls of an item on the event loop.

:func:`run_concurrent_transcription_tasks` runs a coroutine over argument
tuples; :func:`run_streaming_transcription_tasks` consumes an async producer
through a bounded queue. Both report each result to a callback as soon as it
is ready.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

__all__ = [
    "run_concurrent_transcription_tasks",
    "run_streaming_transcription_tasks",
]

logger = logging.getLogger(__name__)

_SENTINEL: Any = object()


async def run_concurrent_transcription_tasks(
    corofunc: Callable[..., Awaitable[Any]],
    args_list: list[tuple[Any, ...]],
    concurrency_limit: int = 20,
    delay: float = 0,
    on_result: Callable[[Any], Awaitable[None]] | None = None,
) -> list[Any]:
    """Run *corofunc* over argument tuples with at most *concurrency_limit* tasks.

    *delay* is waited before each call; *on_result* receives each result as
    soon as it is ready.

    Returns:
        One result per input tuple in the original order (``None`` for a
        failed task or a failed callback).
    """
    n = len(args_list)
    results: list[Any] = [None] * n
    if n == 0:
        return results

    # A fixed set of workers pulls indices, so a long document never creates
    # one task per chunk.
    index_queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(n):
        index_queue.put_nowait(i)
    workers_n = max(1, min(int(concurrency_limit), n))

    async def worker() -> None:
        while True:
            try:
                i = index_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            args = args_list[i]
            try:
                if delay > 0:
                    await asyncio.sleep(delay)
                result = await corofunc(*args)
                results[i] = result
                if on_result is not None:
                    try:
                        await on_result(result)
                    except Exception as cb_exc:  # noqa: BLE001 - isolate callback
                        # A unit the callback could not store counts as failed,
                        # so resume runs it again.
                        logger.error(
                            f"on_result callback failed for args {args}: {cb_exc}"
                        )
                        results[i] = None
            except Exception as e:  # noqa: BLE001 - one failed task must not stop the pool
                logger.error(f"Extraction task failed with arguments {args}: {e}")
                results[i] = None

    worker_tasks = [asyncio.create_task(worker()) for _ in range(workers_n)]
    try:
        await asyncio.gather(*worker_tasks)
    except BaseException:
        for task in worker_tasks:
            task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        raise
    return results


async def run_streaming_transcription_tasks(
    producer: AsyncIterator[Any],
    handler: Callable[[Any], Awaitable[Any]],
    concurrency_limit: int = 20,
    delay: float = 0,
    on_result: Callable[[Any], Awaitable[None]] | None = None,
) -> list[Any]:
    """Consume an async producer with a bounded queue and worker pool.

    A single producer task fills a queue bounded to ``2 * concurrency_limit``
    items; ``concurrency_limit`` workers run *handler* on each item. A
    producer exception is re-raised after the workers have processed what
    was produced before it, so partial progress reaches *on_result*.

    Returns:
        The handler results in completion order (``None`` for a failed item
        or a failed callback).
    """
    workers_n = max(1, int(concurrency_limit))
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=2 * workers_n)
    producer_error: list[BaseException] = []
    results: list[Any] = []

    async def produce() -> None:
        try:
            async for item in producer:
                await queue.put(item)
        except BaseException as e:  # noqa: BLE001 - re-raised after the workers drain
            producer_error.append(e)
        finally:
            for _ in range(workers_n):
                await queue.put(_SENTINEL)

    async def work() -> None:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                return
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                result = await handler(item)
            except Exception as e:  # noqa: BLE001 - one failed item must not stop the pool
                logger.error(f"Streaming extraction task failed: {e}")
                results.append(None)
                continue
            results.append(result)
            # Other workers append while the callback awaits, so fix the slot now.
            slot = len(results) - 1
            if on_result is not None:
                try:
                    await on_result(result)
                except Exception as cb_exc:  # noqa: BLE001 - isolate callback
                    logger.error(f"on_result callback failed: {cb_exc}")
                    results[slot] = None

    producer_task = asyncio.create_task(produce())
    worker_tasks = [asyncio.create_task(work()) for _ in range(workers_n)]
    all_tasks = [producer_task, *worker_tasks]
    try:
        await asyncio.gather(*all_tasks)
    except BaseException:
        for task in all_tasks:
            task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        raise

    if producer_error:
        raise producer_error[0]
    return results
