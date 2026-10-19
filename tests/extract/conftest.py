"""Builders of the store-backed extraction tests: stores, handles and payloads."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.sqlite import read_rows
from chronominer.common.store import PlannedUnit, Store
from chronominer.extract.units import UNIT_KIND, DocumentHandle
from chronominer.output.store import COMPONENT
from tests.llm.conftest import Sleeps

TOOL = "chronominer"
MODEL = "model-a"
PROVIDER = "openai"

_REAL_SLEEP = asyncio.sleep


@dataclass(frozen=True)
class FakePayload:
    """A streamed chunk: the attributes the unit pipeline reads."""

    index: int
    image_name: str
    page_index: int | None = None

    def provenance(self) -> dict[str, Any]:
        return {"custom_id": f"doc-page-{self.index + 1}", "chunk_range": None}


def make_store(db: Path, **kwargs: Any) -> Store:
    """Return a store at *db* with ChronoMiner's content tables, not yet open."""
    return Store(db, TOOL, components=(COMPONENT,), **kwargs)


def payloads(count: int) -> list[FakePayload]:
    """Return *count* PDF page payloads named like rendered pages."""
    return [FakePayload(i, f"page_{i + 1:04d}", page_index=i) for i in range(count)]


async def stream(items: Iterable[Any]) -> AsyncIterator[Any]:
    """Yield *items* as an async producer."""
    for item in items:
        yield item


async def open_handle(
    store: Store,
    name: str = "doc",
    pages: int = 5,
    *,
    model: str = MODEL,
    provider: str = PROVIDER,
    kind: str = UNIT_KIND,
) -> DocumentHandle:
    """Begin a run if needed, lease document *name* and plan its units."""
    if store.run_id is None:
        await store.begin_run("extract")
    doc_id = await store.ensure_document(name, input_path=f"C:/in/{name}.pdf")
    lease = await store.acquire(doc_id)
    assert lease is not None
    handle = DocumentHandle(store, doc_id, name, lease, model, provider, kind)
    await handle.plan(
        [
            PlannedUnit(i, label=f"page_{i + 1:04d}", source_index=i)
            for i in range(pages)
        ]
    )
    return handle


def units(db: Path) -> list[dict[str, Any]]:
    """Return the unit rows through a read-only connection, by index."""
    return read_rows(db, "units", order_by=["idx"])


def responses(db: Path) -> list[dict[str, Any]]:
    """Return the response rows through a read-only connection, in order."""
    return read_rows(db, "responses", order_by=["response_id"])


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> Sleeps:
    """Record every ``asyncio.sleep`` delay and only yield instead of waiting.

    Pass ``sleeps.clock`` to rate limiters so their waits end in fake time.
    """
    recorded = Sleeps()

    async def fake_sleep(delay: float, result: Any = None) -> Any:
        recorded.append(delay)
        recorded.now += max(0.0, delay)
        await _REAL_SLEEP(0)
        return result

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded
