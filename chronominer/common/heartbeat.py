"""Progress file for the job launcher: units done, the item and an ETA.

The launcher names a file in ``CHRONO_HEARTBEAT_FILE`` and its ``list``
shows what the running step writes there: ``units_done`` of ``units_total``
``unit_label``, ``eta_seconds`` (null when unknown) and ``updated_at``
(ISO 8601, UTC). :meth:`Heartbeat.from_env` returns None when the variable
is unset or empty, so a run outside the launcher writes nothing.

A write goes to a temporary file in the same folder, which replaces the
heartbeat file in one step; a reader never sees half a file. Writes are
called on the event-loop thread, so none waits: an update that meets a
locked file or any other ``OSError`` is skipped and the next one writes.
Updates come at most every :data:`INTERVAL` seconds unless forced. No
failure of the heartbeat fails the run; the first is logged as a warning,
later ones at debug level.

:class:`ItemProgress` turns the events of a run over items into updates:
the current item's units, its position (``pages processed, item 3/12``)
and, while the last item runs, an ETA from :class:`RateEta`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["ENV_VAR", "INTERVAL", "Heartbeat", "ItemProgress", "RateEta"]

ENV_VAR = "CHRONO_HEARTBEAT_FILE"
INTERVAL = 15.0

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Heartbeat:
    """Write the progress file at *path*, at most every *interval* seconds."""

    def __init__(
        self,
        path: Path,
        *,
        interval: float = INTERVAL,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], str] = _utc_now,
    ) -> None:
        self.path = path
        self.interval = interval
        self._clock = clock
        self._now = now
        self._last: float | None = None
        self._warned = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Heartbeat | None:
        """Return a writer for the file in :data:`ENV_VAR`; None when unset."""
        environ = os.environ if environ is None else environ
        try:
            raw = environ.get(ENV_VAR, "").strip()
            return cls(Path(raw)) if raw else None
        except Exception as exc:  # noqa: BLE001 - a heartbeat never fails the run
            logger.warning("Heartbeat disabled: %s", exc)
            return None

    def update(
        self,
        done: int,
        total: int,
        label: str,
        eta_seconds: float | None = None,
        *,
        force: bool = False,
    ) -> bool:
        """Write the progress unless throttled; return whether it was written."""
        try:
            now = self._clock()
            if (
                not force
                and self._last is not None
                and now - self._last < self.interval
            ):
                return False
            payload = {
                "units_done": int(done),
                "units_total": int(total),
                "unit_label": label,
                "eta_seconds": None if eta_seconds is None else round(eta_seconds),
                "updated_at": self._now(),
            }
            if not self._write(json.dumps(payload, ensure_ascii=False)):
                return False
            self._last = now
            return True
        except Exception as exc:  # noqa: BLE001 - a heartbeat never fails the run
            self.failed(exc)
            return False

    def _write(self, text: str) -> bool:
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8", newline="\n")
            os.replace(temporary, self.path)
        except OSError as exc:
            self.failed(exc)
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup:
                self.failed(cleanup)
            return False
        return True

    def failed(self, exc: BaseException) -> None:
        """Log a heartbeat failure: the first as a warning, later ones at debug."""
        if self._warned:
            logger.debug("Heartbeat update skipped: %s", exc)
            return
        self._warned = True
        logger.warning("Heartbeat update skipped (%s): %s", self.path, exc)


class RateEta:
    """Seconds left from the unit rate since the first observed unit.

    :meth:`start` sets the reference point; :meth:`seconds` returns
    ``remaining * elapsed / done_since_start``, or None before the first
    unit after the start.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._start: tuple[float, int] | None = None

    def start(self, done: int = 0) -> None:
        """Count units from *done* on, from now."""
        self._start = (self._clock(), done)

    def seconds(self, done: int, total: int) -> float | None:
        """Return the seconds until *total*; None without a rate yet."""
        if self._start is None:
            self.start(done)
            return None
        began, first = self._start
        progressed = done - first
        if progressed <= 0:
            return None
        return max(total - done, 0) * (self._clock() - began) / progressed


class ItemProgress:
    """Heartbeat updates for a run over items.

    The units are the current item's, as its unit events report them; the
    label carries the item position. Only while the last item runs is the
    remaining work known, so only then is an ETA written. Item and run ends
    are forced writes. Every method swallows its own failures.
    """

    def __init__(
        self,
        heartbeat: Heartbeat,
        noun: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.heartbeat = heartbeat
        self.noun = noun
        self._eta = RateEta(clock)
        self._index = 0
        self._items = 0
        self._done = 0
        self._total = 0

    def item_started(self, index: int, total: int) -> None:
        """Begin item *index* of *total* with no units yet."""
        self._guard(self._item_started, index, total)

    def units(self, done: int, total: int, noun: str | None = None) -> None:
        """Record *done* of *total* units of the current item."""
        self._guard(self._units, done, total, noun)

    def item_finished(self) -> None:
        """Write the finished item's units."""
        self._guard(self._write, force=True, finished=True)

    def run_finished(self) -> None:
        """Write the last state once more at the end of a run that had items."""
        if self._items:
            self._guard(self._write, force=True, finished=True)

    def _item_started(self, index: int, total: int) -> None:
        self._index, self._items = index, total
        self._done = self._total = 0
        self._eta.start()
        self._write()

    def _units(self, done: int, total: int, noun: str | None) -> None:
        if noun:
            self.noun = noun
        self._done, self._total = done, total
        self._write()

    def _write(self, *, force: bool = False, finished: bool = False) -> None:
        eta: float | None = None
        if self._items and self._index == self._items:
            eta = 0.0 if finished else self._eta.seconds(self._done, self._total)
        label = f"{self.noun} processed, item {self._index}/{self._items}"
        self.heartbeat.update(self._done, self._total, label, eta, force=force)

    def _guard(self, method: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        try:
            method(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a heartbeat never fails the run
            self.heartbeat.failed(exc)
