"""Run events: how the core reports progress to a front end.

The core never prints. It calls a :class:`RunEvents` implementation instead:
the CLI writes the events to stderr, a test records them, and
:class:`NullEvents` drops them. Events are called on the event-loop thread.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "ChunkDone",
    "ItemFinished",
    "ItemStarted",
    "NullEvents",
    "RunEvents",
    "Waiting",
]


@dataclass(frozen=True)
class ItemStarted:
    """An item is about to be processed; *index* counts from 1 to *total*.

    *kind* is the item's input type: text, image or pdf.
    """

    name: str
    kind: str
    index: int
    total: int


@dataclass(frozen=True)
class ChunkDone:
    """One pending chunk of an item finished.

    *chunk* is its 0-based index: a line range of a text, a page of a PDF,
    or the one chunk of an image. *status* is ``ok``, ``failed`` or
    ``deferred`` (left for a later run or a batch). *done* of *total* counts
    the chunks pending in this run; *already_complete* chunks came from an
    earlier run.
    """

    item: str
    chunk: int
    status: str
    done: int
    total: int
    already_complete: int = 0


@dataclass(frozen=True)
class Waiting:
    """The run waits; *cancel*, when given, ends the wait early."""

    reason: str
    seconds: float | None = None
    cancel: Callable[[], None] | None = None


@dataclass(frozen=True)
class ItemFinished:
    """An item finished; *report* holds its chunk statistics, if any.

    *error* says why an item that did not succeed failed. *submitted* means
    its chunks went to a batch, which ``batch check`` collects. *partial*
    marks a successful ``--pages`` item whose document is still short of
    complete.
    """

    name: str
    index: int
    total: int
    success: bool
    outputs: tuple[str, ...] = ()
    report: Mapping[str, Any] | None = None
    error: str | None = None
    submitted: bool = False
    partial: bool = False


class RunEvents(Protocol):
    """The events a run reports."""

    def item_started(self, event: ItemStarted) -> None:
        """Report that an item is about to be processed."""
        ...

    def chunk_done(self, event: ChunkDone) -> None:
        """Report that one pending chunk of an item finished."""
        ...

    def waiting(self, event: Waiting) -> None:
        """Report that the run waits."""
        ...

    def warning(self, message: str) -> None:
        """Report a warning that does not stop the run."""
        ...

    def item_finished(self, event: ItemFinished) -> None:
        """Report that an item finished."""
        ...


class NullEvents:
    """Drop every event."""

    def item_started(self, event: ItemStarted) -> None:
        """Ignore the event."""

    def chunk_done(self, event: ChunkDone) -> None:
        """Ignore the event."""

    def waiting(self, event: Waiting) -> None:
        """Ignore the event."""

    def warning(self, message: str) -> None:
        """Ignore the warning."""

    def item_finished(self, event: ItemFinished) -> None:
        """Ignore the event."""
