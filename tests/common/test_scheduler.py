"""The run scheduler: shared request slots and the opening of items."""

from __future__ import annotations

import asyncio

import pytest

from chronominer.common.retry import StopSignal
from chronominer.common.scheduler import ItemTicket, RunScheduler


class _Probe:
    """Items of some units each, processed under the scheduler's slots.

    *open_peak* and *slot_peak* are the most items and the most units in
    work at once; *opened* lists the items in the order they were opened.
    """

    def __init__(self, scheduler: RunScheduler) -> None:
        self.scheduler = scheduler
        self.opened: list[str] = []
        self.open_now = self.open_peak = 0
        self.slot_now = self.slot_peak = 0

    async def unit(self, ticket: ItemTicket) -> None:
        async with self.scheduler.slot():
            self.slot_now += 1
            self.slot_peak = max(self.slot_peak, self.slot_now)
            await asyncio.sleep(0.01)
            self.slot_now -= 1
        ticket.done()

    async def process(self, item: tuple[str, int], ticket: ItemTicket) -> str:
        name, units = item
        self.opened.append(name)
        self.open_now += 1
        self.open_peak = max(self.open_peak, self.open_now)
        try:
            await asyncio.sleep(0)  # planning
            ticket.expect(units)
            await asyncio.gather(*(self.unit(ticket) for _ in range(units)))
        finally:
            self.open_now -= 1
        return name


async def test_a_window_of_one_runs_the_items_one_after_another() -> None:
    probe = _Probe(RunScheduler(4))

    results = await probe.scheduler.run([("a", 1), ("b", 1), ("c", 1)], probe.process)

    assert results == ["a", "b", "c"]
    assert probe.open_peak == 1


async def test_small_items_are_opened_until_they_fill_the_limit() -> None:
    probe = _Probe(RunScheduler(4, window=8))
    items = [(name, 1) for name in "abcdefgh"]

    results = await probe.scheduler.run(items, probe.process)

    assert results == list("abcdefgh")
    assert probe.opened == list("abcdefgh")
    assert 1 < probe.open_peak <= 4
    assert probe.slot_peak <= 4


async def test_an_item_that_fills_the_limit_keeps_the_next_one_closed() -> None:
    probe = _Probe(RunScheduler(4, window=8))

    results = await probe.scheduler.run([("big", 10), ("small", 1)], probe.process)

    assert results == ["big", "small"]
    # The units end four at a time: the second item opens with two left.
    assert probe.opened == ["big", "small"]
    assert probe.open_peak == 2
    assert probe.slot_peak == 4


async def test_the_window_bounds_the_open_items() -> None:
    probe = _Probe(RunScheduler(16, window=3))

    await probe.scheduler.run([(str(n), 1) for n in range(9)], probe.process)

    assert probe.open_peak == 3


async def test_an_item_without_a_count_holds_the_next_one_back() -> None:
    scheduler = RunScheduler(4, window=4)
    order: list[str] = []

    async def process(item: str, ticket: ItemTicket) -> str:
        order.append(f"open {item}")
        await asyncio.sleep(0.01)
        order.append(f"end {item}")
        return item

    assert await scheduler.run(["a", "b"], process) == ["a", "b"]
    assert order == ["open a", "end a", "open b", "end b"]


async def test_a_run_stop_opens_no_further_item() -> None:
    stop = StopSignal()
    scheduler = RunScheduler(2, window=2, stop=stop)

    async def process(item: str, ticket: ItemTicket) -> str:
        ticket.expect(1)
        await asyncio.sleep(0.01)
        if item == "b":
            stop.set("no further calls")
        ticket.done()
        return item

    assert await scheduler.run(["a", "b", "c", "d"], process) == ["a", "b"]


async def test_an_error_ends_the_run_after_the_open_items() -> None:
    scheduler = RunScheduler(2, window=2)
    ended: list[str] = []

    async def process(item: str, ticket: ItemTicket) -> str:
        ticket.expect(1)
        await asyncio.sleep(0.05 if item == "a" else 0.15)
        if item == "a":
            raise RuntimeError("broken item")
        ended.append(item)
        return item

    with pytest.raises(RuntimeError, match="broken item"):
        await scheduler.run(["a", "b", "c"], process)

    assert ended == ["b"]


async def test_a_cancelled_item_ends_the_run_as_cancelled() -> None:
    scheduler = RunScheduler(2, window=2)
    ended: list[str] = []

    async def process(item: str, ticket: ItemTicket) -> str:
        ticket.expect(1)
        await asyncio.sleep(0.05 if item == "a" else 0.15)
        if item == "a":
            raise asyncio.CancelledError
        ended.append(item)
        return item

    with pytest.raises(asyncio.CancelledError):
        await scheduler.run(["a", "b", "c"], process)

    assert ended == ["b"]


async def test_cancelling_the_run_cancels_and_awaits_the_open_items() -> None:
    scheduler = RunScheduler(2, window=2)
    released: list[str] = []

    async def process(item: str, ticket: ItemTicket) -> str:
        ticket.expect(1)
        try:
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0)
            released.append(item)
        return item

    run = asyncio.ensure_future(scheduler.run(["a", "b", "c"], process))
    await asyncio.sleep(0.01)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert sorted(released) == ["a", "b"]
