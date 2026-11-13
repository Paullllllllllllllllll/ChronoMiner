"""One request limit for all items of a run, and how many items are open at once.

A tool processes its items one after another, each with a pool of
``--concurrency`` workers. An item with fewer units than workers leaves the
rest idle. :class:`RunScheduler` shares the limit across items: every unit
of every open item takes a :meth:`~RunScheduler.slot` for its requests, and
:meth:`~RunScheduler.run` opens a further item whenever the open ones cannot
fill the limit, up to *window* items.

An open item says how many units it still has to process through its
:class:`ItemTicket`. Until it has said so it counts as able to fill the
limit alone, so the next item waits while an item is being planned. With a
window of 1 the items run one after another, as without a scheduler.

A run stop (:class:`~.retry.StopSignal`) and an error of an item open no
further item; the open ones run to their end before :meth:`~RunScheduler.run`
returns or raises. Cancelling the run cancels the open items and waits for
them, so that their leases are released before the caller closes its stores.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from .retry import StopSignal

__all__ = ["ItemTicket", "RunScheduler"]


class ItemTicket:
    """What an open item tells the scheduler about the work it has left."""

    def __init__(self, changed: Callable[[], None]) -> None:
        self._changed = changed
        self._left: int | None = None

    @property
    def left(self) -> int | None:
        """The units still to process, or None before :meth:`expect`."""
        return self._left

    def expect(self, units: int) -> None:
        """Say that the item has *units* units still to process."""
        self._left = max(0, units)
        self._changed()

    def done(self, units: int = 1) -> None:
        """Say that *units* of the expected units are processed or given up."""
        if self._left is not None:
            self._left = max(0, self._left - units)
            self._changed()


class RunScheduler:
    """The request slots of a run and the opening of its items.

    *limit* is the number of units that may hold a slot at once, over all
    open items; *window* the number of items open at once. *stop* is the
    run's stop signal: once it is set, no further item is opened.
    """

    def __init__(
        self, limit: int, *, window: int = 1, stop: StopSignal | None = None
    ) -> None:
        self.limit = max(1, limit)
        self.window = max(1, window)
        self._stop = stop
        self._slots = asyncio.Semaphore(self.limit)
        self._change: asyncio.Future[None] | None = None

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one of the run's request slots for a unit's calls."""
        async with self._slots:
            yield

    def _changed(self) -> None:
        if self._change is not None and not self._change.done():
            self._change.set_result(None)

    def _may_open(self, tickets: Sequence[ItemTicket]) -> bool:
        if self._stop is not None and self._stop.reason is not None:
            return False
        if not tickets:
            return True
        if len(tickets) >= self.window:
            return False
        demand = sum(
            self.limit if ticket.left is None else ticket.left for ticket in tickets
        )
        return demand < self.limit

    async def run[T, R](
        self,
        items: Sequence[T],
        process: Callable[[T, ItemTicket], Awaitable[R]],
    ) -> list[R]:
        """Process *items*, opened in the order given; return their results.

        The results are those of the items opened, in that order. After a
        run stop they cover the first items only: the ones behind were not
        started. *process* handles the failures of its own item and returns
        them in its result.

        Raises:
            BaseException: The first error a *process* call raised, once the
                items open at that moment have ended; no result is returned.
        """
        results: dict[int, R] = {}
        open_items: dict[asyncio.Task[R], tuple[int, ItemTicket]] = {}
        error: BaseException | None = None
        position = 0
        loop = asyncio.get_running_loop()
        try:
            while True:
                while (
                    position < len(items)
                    and error is None
                    and self._may_open([ticket for _, ticket in open_items.values()])
                ):
                    ticket = ItemTicket(self._changed)
                    task = asyncio.ensure_future(process(items[position], ticket))
                    open_items[task] = (position, ticket)
                    position += 1
                if not open_items:
                    break
                self._change = loop.create_future()
                waiting: set[asyncio.Future[Any]] = {*open_items, self._change}
                await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                for task in [task for task in open_items if task.done()]:
                    index, _ = open_items.pop(task)
                    failure: BaseException | None
                    if task.cancelled():
                        failure = asyncio.CancelledError()
                    else:
                        failure = task.exception()
                    if failure is None:
                        results[index] = task.result()
                    elif error is None:
                        error = failure
        finally:
            self._change = None
            for task in open_items:
                task.cancel()
            await asyncio.gather(*open_items, return_exceptions=True)
        if error is not None:
            raise error
        return [results[index] for index in sorted(results)]
