"""Store-backed unit pipelines: each chunk of a document is one store row.

A :class:`DocumentHandle` is what extraction needs to work one document of
the run's store: the store, the document's id and name, the run's lease on it
and the model and provider stamped on its units. It plans the expected units
as ``pending``, lists those still to do, resets units and writes one unit in
one short transaction that checks the lease.

:func:`run_unit_stream` extracts in-memory payloads (text chunks, rendered
pages, image files) as a producer yields them. For every unit it waits for
room in the store's write queue before the provider call, attributes the
call's attempts to the unit and records them through
:class:`~chronominer.output.attempts.AttemptRows`, writes the unit with its
entries, and only then reports it to *on_unit*. After all units it raises
:class:`UnitsFailedError` when a unit failed or could not be written; a lost
lease stops the document and is raised as
:class:`~chronominer.common.store.LeaseLostError`.

Unit conventions (the ``chunk`` units of :mod:`chronominer.output.store`):

- ``idx`` is the chunk's absolute 0-based position in its document, so runs
  over different ``--pages`` selections share the rows.
- ``label`` is the chunk's name: ``start-end`` of a text chunk, the page name
  of a PDF page, the file name of an image.
- ``source_index`` is ``start-end`` of a text chunk, the 0-based page index
  of a PDF page or the image's file name.
- ``status`` is ``failed`` for a chunk without answer text or whose answer
  to a request in a JSON format holds no JSON object, else ``done``;
  ``outcome`` is ``failed``, ``ok``, ``no_content`` (the answer sets the
  no-content flag) or ``unparsed`` (the answer to a plain-text request,
  kept as text, does not parse as a JSON object); a done unit keeps the
  answer verbatim as ``content``, a failed one none.
- ``provenance`` is the payload's record: ``custom_id``, ``chunk_range`` and,
  for a page, ``image_provenance``.
- ``response_id`` is the selected attempt (the last one without an error,
  else the last one), whose provider and model the unit carries; ``attempts``
  counts the provider attempts of this pass, and a unit without provider
  calls counts one more attempt.

The entries of a done unit are parsed from its answer
(:func:`~chronominer.extract.response_parsing.parse_extraction`) and replace
the unit's rows of the ``entry_rows`` table in the transaction that writes
the unit; a failed unit clears them.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chronominer.common.capture import use_recorder
from chronominer.common.store import (
    Lease,
    LeaseLostError,
    PlannedUnit,
    Store,
    UnitKey,
    UnitResult,
    unit_context,
)
from chronominer.extract.pool import run_streaming_transcription_tasks
from chronominer.extract.response_parsing import answer_outcome, parse_extraction
from chronominer.output.attempts import AttemptRows
from chronominer.output.entries import entries_writer
from chronominer.output.store import KIND_CHUNK

__all__ = [
    "UNIT_KIND",
    "DocumentHandle",
    "EmptyAnswerError",
    "EntrySchema",
    "StreamHandler",
    "UnitsFailedError",
    "run_unit_stream",
]

logger = logging.getLogger(__name__)

UNIT_KIND = KIND_CHUNK
"""The unit kind of extraction: one chunk."""

EMPTY_ANSWER = "empty model output (possible truncation or refusal)"
"""The error of a chunk whose answer has no text."""

PayloadResult = tuple[Any, str, str | None, Any, int]
"""``(payload, label, answer text, response dict, order_index)``.

The text is None when the chunk produced no answer.
"""

StreamHandler = Callable[[Any, Any], Awaitable[PayloadResult]]
"""``(payload, extractor) -> PayloadResult`` for one streamed unit."""

UnitCallback = Callable[[int, bool], None]
"""``(index, failed)`` after each unit's row is committed."""

Write = Callable[[sqlite3.Connection], None]


class EmptyAnswerError(RuntimeError):
    """The model returned no answer text for a chunk."""


class UnitsFailedError(Exception):
    """One or more units of a document failed or could not be written.

    Raised after every unit was processed; the units written stay in the
    store for a later resume or repair. *write_errors* are the errors of
    the unit writes that failed; the first is the exception's cause.
    """

    def __init__(
        self,
        source_name: str,
        failed_chunks: int,
        total_chunks: int,
        write_errors: Sequence[BaseException] = (),
    ) -> None:
        self.source_name = source_name
        self.failed_chunks = failed_chunks
        self.total_chunks = total_chunks
        self.write_errors = list(write_errors)
        message = (
            f"{failed_chunks} of {total_chunks} chunk(s) failed to extract "
            f"for '{source_name}'"
        )
        if self.write_errors:
            first = self.write_errors[0]
            message += (
                f"; {len(self.write_errors)} could not be stored: "
                f"{type(first).__name__}: {first}"
            )
        super().__init__(message)


@dataclass(frozen=True)
class DocumentHandle:
    """One document of the run's store, worked under the run's lease.

    *name* is the document's unique name, *model* and *provider* the stamps
    of its units when no attempt names others, and *kind* the unit kind.
    The run core builds the handle after acquiring the lease; the lease
    keeper renews it while the handle is in use.
    """

    store: Store
    doc_id: int
    name: str
    lease: Lease
    model: str
    provider: str
    kind: str = UNIT_KIND

    @property
    def tool(self) -> str:
        """Return the tool name the store stamps on every row."""
        return self.store.tool

    @property
    def run_id(self) -> str | None:
        """Return the id of the run the store records."""
        return self.store.run_id

    def key(self, idx: int) -> UnitKey:
        """Return the key of unit *idx*."""
        return UnitKey(self.doc_id, self.kind, idx)

    async def plan(self, units: Sequence[PlannedUnit]) -> int:
        """Create the expected units as ``pending``; return the number created."""
        return await self.store.plan_units(
            self.lease, self.kind, units, model=self.model, provider=self.provider
        )

    def todo(self, *, retry_errors: bool = False) -> list[int]:
        """Return the indices of the pending units no open batch holds.

        With *retry_errors* the failed units are included.
        """
        state = self.store.resume_state(self.doc_id)
        if state is None:
            return []
        blocked = set(state.blocked)
        keys = list(state.ready)
        if retry_errors:
            keys.extend(key for key in state.failed if key not in blocked)
        return sorted(key.idx for key in keys if key.kind == self.kind)

    async def reset(
        self,
        idxs: Sequence[int] | None = None,
        *,
        statuses: Sequence[str] | None = None,
    ) -> int:
        """Return units to ``pending``, keeping their responses; return the count.

        None selects every unit of the kind, or every status.
        """
        return await self.store.reset_units(
            self.lease, kind=self.kind, idxs=idxs, statuses=statuses
        )

    async def write(  # noqa: PLR0913 - mirrors the UnitResult fields; tests call it
        self,
        idx: int,
        *,
        status: str,
        content: str | None,
        outcome: str | None = None,
        label: str | None = None,
        source_index: int | str | None = None,
        usage: Mapping[str, Any] | None = None,
        error: str | None = None,
        error_type: str | None = None,
        provenance: Mapping[str, Any] | None = None,
        response_id: int | None = None,
        attempts: int | None = None,
        model: str | None = None,
        provider: str | None = None,
        extra: Write | None = None,
    ) -> None:
        """Commit unit *idx* in one transaction that checks the lease.

        *extra* runs first in the same transaction (the unit's entries).

        Raises:
            LeaseLostError: The lease expired or passed to another holder.
            sqlite3.Error: The write failed (a busy database only delays it).
        """
        await self.store.write_unit(
            self.lease,
            UnitResult(
                kind=self.kind,
                idx=idx,
                status=status,
                model=model or self.model,
                provider=provider or self.provider,
                content=content,
                outcome=outcome,
                label=label,
                source_index=source_index,
                usage=usage,
                error=error,
                error_type=error_type,
                provenance=provenance,
                response_id=response_id,
                attempts=attempts,
            ),
            extra=extra,
        )


def _source_index(payload: Any, label: str) -> int | str:
    page_index = getattr(payload, "page_index", None)
    if isinstance(page_index, int):
        return page_index
    source_index = getattr(payload, "source_index", None)
    if isinstance(source_index, int | str):
        return source_index
    return label


def _provenance(payload: Any) -> Mapping[str, Any] | None:
    provenance = getattr(payload, "provenance", None)
    if not callable(provenance):
        return None
    value = provenance()
    return value if isinstance(value, Mapping) and value else None


@dataclass(frozen=True)
class EntrySchema:
    """The schema the entries of a document's chunks are stored under.

    *schema* is the schema file's JSON; *name* its schema name.
    """

    schema: Mapping[str, Any] | None = None
    name: str | None = None


class _Units:
    """The per-unit steps of the pipeline, and their tally."""

    def __init__(
        self,
        handle: DocumentHandle,
        on_unit: UnitCallback | None,
        entries: EntrySchema,
    ) -> None:
        self.handle = handle
        self.on_unit = on_unit
        self.entries = entries
        self.total = 0
        self.failed = 0
        self.write_errors: list[BaseException] = []
        self.lost: LeaseLostError | None = None

    async def run(
        self,
        idx: int,
        label: str,
        call: Callable[[], Awaitable[PayloadResult]],
        *,
        source_index: int | str,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Extract one unit through *call*, write it, then report it."""
        handle = self.handle
        if self.lost is not None:
            return
        await handle.store.wait_capacity()
        if self.lost is not None:
            return
        recorder = AttemptRows(
            handle.store, provider=handle.provider, model=handle.model
        )
        result: PayloadResult | None = None
        raised: BaseException | None = None
        with unit_context(handle.key(idx)), use_recorder(recorder):
            try:
                result = await call()
            except Exception as exc:
                logger.exception("Error extracting %s: %s", label, exc)
                raised = exc
        self.total += 1
        try:
            if recorder.failures:
                raise recorder.failures[0]
            failed = await self._write(
                idx, label, result, raised, recorder, source_index, provenance
            )
        except LeaseLostError as exc:
            logger.error(
                "Stopping '%s': unit %d was not stored: %s", handle.name, idx, exc
            )
            self.lost = exc
            self.failed += 1
            return
        except Exception as exc:  # noqa: BLE001 - tallied, raised after all units
            logger.error(
                "Unit %d of '%s' could not be stored in %s: %s",
                idx,
                handle.name,
                handle.store.path,
                exc,
            )
            self.write_errors.append(exc)
            self.failed += 1
            return
        if failed:
            self.failed += 1
        if self.on_unit is not None:
            self.on_unit(idx, failed)

    def _entries(self, idx: int, text: str | None, model: str, provider: str) -> Write:
        handle = self.handle
        return entries_writer(
            handle.store,
            handle.doc_id,
            idx,
            parse_extraction(text) if text is not None else [],
            model=model,
            provider=provider,
            schema=self.entries.schema,
            schema_name=self.entries.name,
            kind=handle.kind,
        )

    async def _write(
        self,
        idx: int,
        label: str,
        result: PayloadResult | None,
        raised: BaseException | None,
        recorder: AttemptRows,
        source_index: int | str,
        provenance: Mapping[str, Any] | None,
    ) -> bool:
        """Write the unit's row and entries; return whether the unit failed."""
        text: str | None = None
        raw: Any = None
        if result is not None and len(result) >= 5:
            label = result[1] or label
            text, raw = result[2], result[3]
        failed = text is None or not text.strip()
        raw_dict = raw if isinstance(raw, Mapping) else {}
        selected = recorder.selected()
        error: str | None = None
        error_type: str | None = None
        if raised is not None:
            error = str(raised) or type(raised).__name__
            error_type = type(raised).__name__
        elif failed:
            last = recorder.stored[-1].record if recorder.stored else None
            error = (
                raw_dict.get("error") or (last.error if last else None) or EMPTY_ANSWER
            )
            error_type = (
                raw_dict.get("error_type")
                or (last.error_type if last else None)
                or EmptyAnswerError.__name__
            )
        usage = raw_dict.get("usage")
        model = (selected.record.model if selected else None) or self.handle.model
        provider = (
            selected.record.provider if selected else None
        ) or self.handle.provider
        content = None if failed else text
        await self.handle.write(
            idx,
            status="failed" if failed else "done",
            content=content,
            outcome="failed" if failed else answer_outcome(content),
            label=label,
            source_index=source_index,
            usage=usage if isinstance(usage, Mapping) else None,
            error=error,
            error_type=error_type,
            provenance=provenance,
            response_id=selected.response_id if selected else None,
            attempts=len(recorder.stored) or None,
            model=model,
            provider=provider,
            extra=self._entries(idx, content, model, provider),
        )
        return failed

    def finish(self) -> None:
        """Raise the lost lease or the item failure, if any."""
        if self.lost is not None:
            raise self.lost
        if self.failed:
            cause = self.write_errors[0] if self.write_errors else None
            raise UnitsFailedError(
                self.handle.name, self.failed, self.total, self.write_errors
            ) from cause


async def run_unit_stream(
    handle: DocumentHandle,
    payloads: AsyncIterator[Any],
    extractor: Any,
    handler: StreamHandler,
    *,
    concurrency: int = 20,
    delay: float = 0.0,
    on_unit: UnitCallback | None = None,
    entries: EntrySchema | None = None,
) -> int:
    """Extract streamed payloads into the document's units.

    A pool of *concurrency* workers runs *handler* on each payload as the
    producer yields it, each call *delay* seconds after the worker's
    previous one. A payload carries ``index`` and ``image_name`` (its
    label); the producer leaves out the units already done. *entries* types
    the entry columns. Returns the number of units processed.

    Raises:
        UnitsFailedError: A unit failed or could not be written.
        LeaseLostError: The lease was lost; the remaining units were skipped.
    """
    units = _Units(handle, on_unit, entries or EntrySchema())

    async def handle_payload(payload: Any) -> None:
        label = str(payload.image_name)
        await units.run(
            int(payload.index),
            label,
            lambda: handler(payload, extractor),
            source_index=_source_index(payload, label),
            provenance=_provenance(payload),
        )

    await run_streaming_transcription_tasks(
        payloads, handle_payload, concurrency, delay
    )
    units.finish()
    return units.total
