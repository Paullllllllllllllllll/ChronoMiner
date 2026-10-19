"""The attempt recorder that appends every provider attempt to the store.

:class:`AttemptRows` implements the recorder protocol of
:mod:`chronominer.common.capture`. Each attempt becomes one
``responses`` row in its own transaction, committed before the caller sends
the next attempt: the body as it came over HTTP (``raw_kind`` ``http``), or
the SDK object's dump (``sdk``) where no hook saw it. The store stamps the
row with its run and tool; the record names the provider and model, and the
unit comes from :func:`~chronominer.common.store.unit_context`.

A pipeline installs one recorder per unit, so :meth:`AttemptRows.selected`
names the attempt the unit's content came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chronominer.common.capture import AttemptRecord
from chronominer.common.store import ResponseRow, Store, UnitKey, current_unit

__all__ = ["AttemptRows", "StoredAttempt"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredAttempt:
    """One attempt as stored: its response id, unit and record."""

    response_id: int
    unit: UnitKey | None
    record: AttemptRecord


class AttemptRows:
    """Append each attempt record to the store's ``responses`` table.

    *provider* and *model* stand in where a record leaves them empty. A
    failed append is kept in :attr:`failures` and raised to the caller.
    """

    def __init__(self, store: Store, *, provider: str = "", model: str = "") -> None:
        self.store = store
        self.provider = provider
        self.model = model
        self.stored: list[StoredAttempt] = []
        self.failures: list[BaseException] = []

    async def record(self, record: AttemptRecord) -> int | None:
        """Append *record* as one response row; return its response id."""
        unit = current_unit()
        row = ResponseRow(
            provider=record.provider or self.provider,
            model=record.model or self.model,
            unit=unit,
            role=record.role,
            attempt=record.attempt,
            request=record.request,
            raw=record.raw,
            raw_kind=record.raw_kind,
            usage=record.usage,
            error=record.error,
            error_type=record.error_type,
            http_status=record.http_status,
            latency=record.latency,
        )
        try:
            response_id = await self.store.append_response(row)
        except Exception as exc:
            logger.error(
                "Could not store attempt %s of %s in %s: %s",
                record.attempt,
                record.label or "a unit",
                self.store.path,
                exc,
            )
            self.failures.append(exc)
            raise
        self.stored.append(StoredAttempt(response_id, unit, record))
        return response_id

    def selected(self) -> StoredAttempt | None:
        """Return the last attempt without an error, else the last attempt."""
        for attempt in reversed(self.stored):
            if attempt.record.error is None:
                return attempt
        return self.stored[-1] if self.stored else None
