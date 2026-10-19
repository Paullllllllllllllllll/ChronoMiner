"""Tests for events.py: the null sink drops every event."""

from __future__ import annotations

from chronominer.events import (
    ChunkDone,
    ItemFinished,
    ItemStarted,
    NullEvents,
    Waiting,
)


def test_null_events_drop_everything() -> None:
    events = NullEvents()

    events.item_started(ItemStarted("ledger.pdf", "pdf", 1, 1))
    events.chunk_done(ChunkDone("ledger.pdf", 0, "ok", 1, 2))
    events.waiting(Waiting("batch pending", seconds=30.0))
    events.warning("chunk 2 failed")
    events.item_finished(ItemFinished("ledger.pdf", 1, 1, False, ("ledger.json",)))
