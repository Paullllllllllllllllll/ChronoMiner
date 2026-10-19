"""Line ranges in the store: one ``range`` unit per range of a file.

The ranges of a text file are the ``range`` units of its document, named
like the extraction document of the file (``ledger`` for ``ledger.txt``
beside its ``chronominer.sqlite``) unless the caller names it. Unit ``idx``
is the range's 0-based position in the starting ranges and
``source_index`` its starting bounds as ``start-end``.

``content`` holds the range the unit contributes to ``_line_ranges.txt``,
as JSON ``[start, end]`` (see :mod:`chronominer.output.line_ranges`), and
NULL for a deleted or merged range. Every write recomputes the contents of
all range units of the document from the starting ranges and the
decisions (:func:`~chronominer.ranges.decision.final_layout`), in the
transaction that writes the unit.

``provenance`` holds the decision on the range, ``{"decision": ...,
"settings": ...}``: the :class:`~chronominer.ranges.decision.RangeResult`
in its stored form and the adjust settings it was made with. A decided range
is ``done`` with ``outcome`` ``moved``, ``kept``, ``on_target`` or
``deleted``; a range whose model calls all failed is ``failed`` with
``outcome`` ``degraded`` and decided again by the next run; a range whose
call raised is ``failed`` without a decision. Generated ranges are ``done``
with ``outcome`` ``generated`` and no decision.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from chronominer.common.capture import AttemptRecord
from chronominer.common.sqlite import read
from chronominer.common.store import (
    Lease,
    PlannedUnit,
    RenderReport,
    Store,
    UnitResult,
    refresh_documents,
)
from chronominer.output.line_ranges import LINE_RANGES_FORMAT, range_content
from chronominer.output.render import render_ranges as render_line_ranges_file
from chronominer.output.store import KIND_RANGE, document_name, open_store
from chronominer.ranges.decision import DecidedRange, RangeResult, final_layout
from chronominer.ranges.lines import Range, ranges_path

__all__ = [
    "DOCUMENT_KIND",
    "GENERATED",
    "LINE_RANGES_FORMAT",
    "AttemptCollector",
    "DocumentBusyError",
    "RangeUnit",
    "group_by_folder",
    "layout",
    "lease_document",
    "open_folder_store",
    "plan_ranges",
    "range_units",
    "ranges_document",
    "recorded_sha256",
    "render_ranges",
    "replace_ranges",
    "stored_ranges",
    "write_failure",
    "write_result",
]

DOCUMENT_KIND: Final = "text"
GENERATED: Final = "generated"
"""The outcome of a range unit that ``ranges generate`` wrote."""


class DocumentBusyError(RuntimeError):
    """Another live run holds the lease on the document."""


def ranges_document(text_file: Path, root: Path | None = None) -> str:
    """Return the document name of *text_file*'s ranges in output root *root*.

    *root* defaults to the folder of *text_file*.
    """
    return document_name(ranges_path(text_file), root or text_file.parent)


def open_folder_store(folder: Path) -> Store:
    """Return the store of *folder*, not yet opened (``chronominer.sqlite``)."""
    return open_store(folder)


def group_by_folder(paths: Sequence[Path]) -> dict[Path, list[Path]]:
    """Group *paths*, resolved, by their folder, keeping their order."""
    grouped: dict[Path, list[Path]] = {}
    for path in paths:
        resolved = path.resolve()
        grouped.setdefault(resolved.parent, []).append(resolved)
    return grouped


def _source_index(bounds: Range) -> str:
    return f"{bounds[0]}-{bounds[1]}"


def _bounds(value: Any) -> Range:
    start, _, end = str(value).partition("-")
    return int(start), int(end)


@dataclass(frozen=True)
class RangeUnit:
    """One stored range.

    *idx* is its 0-based position, *original* its starting bounds,
    *result* the decision on it and *settings* the adjust settings of that
    decision.
    """

    idx: int
    original: Range
    status: str
    outcome: str | None = None
    result: RangeResult | None = None
    settings: Mapping[str, Any] | None = None

    @property
    def decided(self) -> bool:
        """Whether the range holds a decision that a run keeps."""
        return self.status == "done" and self.result is not None


def _provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _unit(row: Sequence[Any]) -> RangeUnit:
    idx, status, outcome, source, provenance = row
    stored = _provenance(provenance)
    body = stored.get("decision")
    settings = stored.get("settings")
    return RangeUnit(
        idx=int(idx),
        original=_bounds(source),
        status=str(status),
        outcome=None if outcome is None else str(outcome),
        result=RangeResult.from_body(int(idx) + 1, body)
        if isinstance(body, dict)
        else None,
        settings=settings if isinstance(settings, dict) else None,
    )


def range_units(connection: sqlite3.Connection, doc_id: int) -> list[RangeUnit]:
    """Return the range units of document *doc_id*, in order."""
    rows = connection.execute(
        "SELECT idx, status, outcome, source_index, provenance FROM units "
        "WHERE doc_id = ? AND kind = ? ORDER BY idx",
        (doc_id, KIND_RANGE),
    ).fetchall()
    return [_unit(row) for row in rows]


def stored_ranges(store: Store, doc_id: int) -> list[RangeUnit]:
    """Return the range units of *doc_id* through a short read."""
    return read(store.path, lambda connection: range_units(connection, doc_id))


def layout(
    units: Sequence[RangeUnit], replace: RangeUnit | None = None
) -> dict[int, Range | None]:
    """Return the range each unit contributes to the file, by unit index.

    *replace* stands in for the stored unit of the same index. Failed
    units and units without a decision keep their starting bounds.
    """
    current = [
        replace if replace is not None and unit.idx == replace.idx else unit
        for unit in units
    ]
    entries = [
        DecidedRange(unit.original, unit.result if unit.status != "pending" else None)
        for unit in current
    ]
    contents: dict[int, Range | None] = {unit.idx: None for unit in current}
    for position, bounds in final_layout(entries):
        contents[current[position].idx] = bounds
    return contents


def _content(bounds: Range | None) -> str | None:
    return None if bounds is None else range_content(*bounds)


def _write_contents(
    connection: sqlite3.Connection,
    doc_id: int,
    contents: Mapping[int, Range | None],
    *,
    skip: int | None = None,
) -> None:
    connection.executemany(
        "UPDATE units SET content = ? WHERE doc_id = ? AND kind = ? AND idx = ?",
        [
            (_content(bounds), doc_id, KIND_RANGE, idx)
            for idx, bounds in contents.items()
            if idx != skip
        ],
    )


async def lease_document(store: Store, text_file: Path, name: str) -> Lease:
    """Ensure document *name* for *text_file* and lease it.

    Raises:
        DocumentBusyError: Another live run holds the document.
    """
    doc_id = await store.ensure_document(name, input_path=text_file, kind=DOCUMENT_KIND)
    lease = await store.acquire(doc_id)
    if lease is None:
        raise DocumentBusyError(f"'{name}' is leased by another run.")
    return lease


async def plan_ranges(
    store: Store,
    lease: Lease,
    ranges: Sequence[Range],
    *,
    model: str,
    provider: str,
    generated: bool = False,
) -> None:
    """Replace the document's range units with *ranges* as starting ranges.

    The units are pending, or ``done`` with outcome ``generated`` when
    *generated*.
    """
    doc_id = lease.doc_id

    def clear(connection: sqlite3.Connection) -> None:
        connection.execute(
            "DELETE FROM units WHERE doc_id = ? AND kind = ?", (doc_id, KIND_RANGE)
        )
        refresh_documents(connection, doc_id)

    await store.write(lease, clear)
    planned = [
        PlannedUnit(idx=idx, source_index=_source_index(bounds))
        for idx, bounds in enumerate(ranges)
    ]
    await store.plan_units(lease, KIND_RANGE, planned, model=model, provider=provider)
    contents = layout(stored_ranges(store, doc_id))

    def fill(connection: sqlite3.Connection) -> None:
        _write_contents(connection, doc_id, contents)
        if generated:
            connection.execute(
                "UPDATE units SET status = 'done', outcome = ? "
                "WHERE doc_id = ? AND kind = ?",
                (GENERATED, doc_id, KIND_RANGE),
            )
            refresh_documents(connection, doc_id)

    await store.write(lease, fill)


def recorded_sha256(store: Store, doc_id: int, path: Path) -> str | None:
    """Return the hash recorded for the rendered file *path*, or None."""
    resolved = path.resolve()
    try:
        key = resolved.relative_to(store.path.resolve().parent).as_posix()
    except ValueError:
        key = str(resolved)
    for row in store.outputs(doc_id):
        if row.path == key:
            return row.sha256
    return None


async def render_ranges(
    store: Store, lease: Lease, path: Path, *, force: bool = False
) -> RenderReport:
    """Render the document's ``_line_ranges.txt`` at *path*.

    A document whose range units are all pending gets no file.
    """
    return await render_line_ranges_file(store, lease.doc_id, path, lease, force=force)


async def replace_ranges(
    store: Store,
    text_file: Path,
    ranges: Sequence[Range],
    *,
    model: str,
    name: str | None = None,
) -> RenderReport:
    """Record *ranges* as the generated ranges of *text_file* and render them.

    The rendering replaces the existing file.
    """
    lease = await lease_document(store, text_file, name or ranges_document(text_file))
    try:
        await plan_ranges(
            store, lease, ranges, model=model, provider="", generated=True
        )
        return await render_ranges(store, lease, ranges_path(text_file), force=True)
    finally:
        await store.release(lease)


@dataclass
class AttemptCollector:
    """Store each provider attempt of one range and keep its record.

    Install it with :func:`~chronominer.common.capture.use_recorder` inside
    :func:`~chronominer.common.store.unit_context` of the range's unit.
    """

    store: Store
    stored: list[tuple[int | None, AttemptRecord]] = field(default_factory=list)

    async def record(self, record: AttemptRecord) -> int | None:
        """Append *record* as a response row and return its id."""
        response_id = await self.store.recorder().record(record)
        self.stored.append((response_id, record))
        return response_id

    def selected(self) -> tuple[int | None, AttemptRecord] | None:
        """Return the last attempt without an error, else the last attempt."""
        for entry in reversed(self.stored):
            if entry[1].error is None:
                return entry
        return self.stored[-1] if self.stored else None

    def usage(self) -> dict[str, int] | None:
        """Return the summed token counts of the stored attempts."""
        totals: dict[str, int] = {}
        for _, record in self.stored:
            for key, value in (record.usage or {}).items():
                if isinstance(value, int) and not isinstance(value, bool):
                    totals[key] = totals.get(key, 0) + value
        return totals or None


def _stamp(
    collector: AttemptCollector | None, model: str, provider: str
) -> dict[str, Any]:
    selected = collector.selected() if collector is not None else None
    if collector is None or selected is None:
        return {
            "model": model,
            "provider": provider,
            "response_id": None,
            "attempts": None,
            "usage": None,
        }
    response_id, record = selected
    return {
        "model": record.model or model,
        "provider": record.provider or provider,
        "response_id": response_id,
        "attempts": len(collector.stored),
        "usage": collector.usage(),
    }


async def _write(
    store: Store, lease: Lease, unit: RangeUnit, result: UnitResult
) -> None:
    """Write *result* for *unit* and the new contents of every range unit."""
    contents = layout(stored_ranges(store, lease.doc_id), unit)
    own = contents.get(unit.idx)
    final = dataclasses.replace(result, content=_content(own))
    await store.write_unit(
        lease,
        final,
        extra=lambda connection: _write_contents(
            connection, lease.doc_id, contents, skip=unit.idx
        ),
    )


async def write_result(
    store: Store,
    lease: Lease,
    result: RangeResult,
    *,
    model: str,
    provider: str,
    collector: AttemptCollector | None = None,
    settings: Mapping[str, Any] | None = None,
    degraded: bool = False,
) -> None:
    """Commit the decision on one range.

    A *degraded* decision, made without any usable answer, is stored as
    ``failed`` so that the next run decides the range again.
    """
    status = "failed" if degraded else "done"
    unit = RangeUnit(result.index - 1, result.original, status, result=result)
    await _write(
        store,
        lease,
        unit,
        UnitResult(
            kind=KIND_RANGE,
            idx=unit.idx,
            status=status,
            outcome="degraded" if degraded else result.outcome,
            source_index=_source_index(result.original),
            provenance={"decision": result.to_body(), "settings": dict(settings or {})},
            error="every model call failed" if degraded else None,
            **_stamp(collector, model, provider),
        ),
    )


async def write_failure(
    store: Store,
    lease: Lease,
    idx: int,
    original: Range,
    error: BaseException,
    *,
    model: str,
    provider: str,
    collector: AttemptCollector | None = None,
) -> None:
    """Commit range *idx* (0-based) as failed with *error*; its bounds stay."""
    unit = RangeUnit(idx, original, "failed")
    await _write(
        store,
        lease,
        unit,
        UnitResult(
            kind=KIND_RANGE,
            idx=idx,
            status="failed",
            outcome="failed",
            source_index=_source_index(original),
            error=str(error) or type(error).__name__,
            error_type=type(error).__name__,
            **_stamp(collector, model, provider),
        ),
    )
