"""A run-events sink that records every event for assertions."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from chronominer.events import ChunkDone, ItemFinished, ItemStarted, Waiting


@dataclass
class RecordingEvents:
    """Keep every event in order as ``(kind, payload)`` pairs."""

    events: list[tuple[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _record(self, kind: str, payload: Any) -> None:
        with self._lock:
            self.events.append((kind, payload))

    def item_started(self, event: ItemStarted) -> None:
        self._record("item_started", event)

    def chunk_done(self, event: ChunkDone) -> None:
        self._record("chunk_done", event)

    def waiting(self, event: Waiting) -> None:
        self._record("waiting", event)

    def warning(self, message: str) -> None:
        self._record("warning", message)

    def item_finished(self, event: ItemFinished) -> None:
        self._record("item_finished", event)

    def of(self, kind: str) -> list[Any]:
        """Return the payloads of one kind of event, in order."""
        with self._lock:
            return [payload for name, payload in self.events if name == kind]

    @property
    def kinds(self) -> list[str]:
        """The kinds of the recorded events, in order."""
        with self._lock:
            return [name for name, _payload in self.events]
