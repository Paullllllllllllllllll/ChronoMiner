"""Batch submission: submit_batch on the store, for text and image chunks."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chronominer import ext
from chronominer.batch import submit as submit_module
from chronominer.batch.submit import (
    REQUEST_TOO_LARGE,
    BatchSubmissionError,
    SubmitOutcome,
    chunk_request,
    submit_batch,
)
from chronominer.common.image_presets import MAX_IMAGE_BYTES
from chronominer.extract.images import PageImage
from chronominer.llm.prompts import EXTRACTION_SCHEMA_NAME
from chronominer.output.store import DATABASE_NAME, open_store
from chronominer.settings import Settings
from chronominer.spec import resolve_spec
from tests.batch.conftest import (
    World,
    leased_document,
    page_images,
    run_spec,
    text_chunks,
)
from tests.conftest import DEFAULT_TEST_SCHEMA


async def submit(root: Path, payloads: list[Any]) -> SubmitOutcome:
    async with open_store(root) as store:
        await store.begin_run("extract")
        document = await leased_document(store)
        return await submit_batch(
            payloads,
            document=document,
            spec=run_spec(root),
            settings=Settings(),
        )


async def test_the_submit_attempts_name_the_model(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities: list[ext.Identity] = []
    real = ext.attempt

    def attempt(*args: Any, **kwargs: Any) -> ext.Attempt[Any]:
        opened: ext.Attempt[Any] = real(*args, **kwargs)
        identities.append(opened.identity)
        return opened

    monkeypatch.setattr(ext, "attempt", attempt)
    await submit(tmp_path, page_images(1))
    assert identities
    assert {(i.provider, i.role, i.model) for i in identities} == {
        ("openai", "extract", "gpt-5.6-luna")
    }


async def test_submit_batch_reads_the_run_and_writes_no_working_file(
    world: World, tmp_path: Path
) -> None:
    spec = resolve_spec(
        {
            "input": tmp_path,
            "schema": DEFAULT_TEST_SCHEMA,
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        }
    ).spec
    payloads = [
        PageImage(index=0, image_name="page_0001", base64="UEFHRQ=="),
        PageImage(index=1, image_name="page_0002", render_error="broken"),
    ]
    root = tmp_path / "out"
    root.mkdir()
    async with open_store(root) as store:
        await store.begin_run("extract")
        document = await leased_document(store, provider="anthropic")
        outcome = await submit_batch(
            payloads, document=document, spec=spec, settings=Settings()
        )
    assert outcome.status == "submitted"
    assert len(outcome.submission_ids) == 1
    (batch,) = world.batches.values()
    assert batch.provider == "anthropic"
    (body,) = world.bodies()
    assert body["model"] == "claude-sonnet-4-6"
    assert sorted(path.name for path in root.iterdir()) == [DATABASE_NAME]


async def test_an_oversized_page_fails_with_its_error_and_the_others_are_sent(
    world: World, tmp_path: Path
) -> None:
    spec = resolve_spec(
        {
            "input": tmp_path,
            "schema": DEFAULT_TEST_SCHEMA,
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        }
    ).spec
    oversized = "A" * (MAX_IMAGE_BYTES["anthropic"] + 4)
    payloads = [
        PageImage(index=index, image_name=f"page_{index + 1:04d}", base64=data)
        for index, data in enumerate(["UEFHRQ==", oversized, "UEFHRQ=="])
    ]
    async with open_store(tmp_path) as store:
        await store.begin_run("extract")
        document = await leased_document(store, provider="anthropic")
        outcome = await submit_batch(
            payloads, document=document, spec=spec, settings=Settings()
        )
        (submission_id,) = outcome.submission_ids
        members = store.submission_requests(submission_id)
        units = {row.idx: row for row in store.units(document.doc_id)}
    assert outcome.status == "submitted"
    assert sorted(key.idx for key in members.values()) == [0, 2]
    assert len(world.bodies()) == 2
    assert [units[idx].status for idx in range(3)] == ["pending", "failed", "pending"]
    failed = units[1]
    assert failed.error_type == REQUEST_TOO_LARGE
    assert failed.error is not None and "over the anthropic limit" in failed.error


async def test_text_chunks_are_sent_as_text_with_the_schema(
    world: World, tmp_path: Path
) -> None:
    chunks = text_chunks(2)
    outcome = await submit(tmp_path, list(chunks))

    assert outcome.status == "submitted"
    bodies = world.bodies()
    assert len(bodies) == 2
    for chunk, body in zip(chunks, bodies, strict=True):
        user = body["input"][1]["content"]
        assert user == [{"type": "input_text", "text": chunk.text}]
        assert body["text"]["format"]["name"] == EXTRACTION_SCHEMA_NAME
    async with open_store(tmp_path) as store:
        units = store.units(1)
    assert [unit.source_index for unit in units] == ["1-3", "4-6"]
    assert [unit.provenance["custom_id"] for unit in units] == [
        "doc-chunk-1",
        "doc-chunk-2",
    ]


def test_chunk_requests_follow_the_payload_kind() -> None:
    (page,) = page_images(1)
    (chunk,) = text_chunks(1)
    image = chunk_request(page, detail="high")
    text = chunk_request(chunk, detail="high")
    assert image is not None and text is not None
    assert (image.is_visual, image.image_detail) == (True, "high")
    assert (text.is_visual, text.text) == (False, chunk.text)
    assert chunk_request(page_images(1, unrendered=frozenset({0}))[0]) is None


async def test_submit_batch_refuses_providers_without_a_batch_api(
    world: World, tmp_path: Path
) -> None:
    spec = resolve_spec(
        {
            "input": tmp_path,
            "schema": DEFAULT_TEST_SCHEMA,
            "provider": "openrouter",
            "model": "openai/gpt-5.4",
        }
    ).spec
    async with open_store(tmp_path) as store:
        await store.begin_run("extract")
        document = await leased_document(store)
        with pytest.raises(BatchSubmissionError, match="does not support batch"):
            await submit_batch([], document=document, spec=spec, settings=Settings())
        assert store.submissions() == []
    assert world.calls == []


async def test_a_document_without_a_sendable_chunk_is_refused(
    world: World, tmp_path: Path
) -> None:
    with pytest.raises(BatchSubmissionError, match="No chunk"):
        await submit(tmp_path, page_images(1, unrendered=frozenset({0})))
    assert world.calls == []


class Source:
    """Rendered pages handed out one at a time; counts them and its closing."""

    def __init__(self, payloads: list[Any]) -> None:
        self.payloads = payloads
        self.taken = 0
        self.closed = False

    async def __call__(self) -> AsyncIterator[Any]:
        try:
            for payload in self.payloads:
                self.taken += 1
                yield payload
        finally:
            self.closed = True


async def _sliced(
    root: Path, source: Source
) -> tuple[SubmitOutcome, list[int], dict[str, list[int]]]:
    """Submit *source*; return the outcome, the parts and each submission's units."""
    async with open_store(root) as store:
        await store.begin_run("extract")
        document = await leased_document(store)
        outcome = await submit_batch(
            source(), document=document, spec=run_spec(root), settings=Settings()
        )
        parts = [row.part for row in store.submissions()]
        members = {
            submission_id: sorted(
                key.idx for key in store.submission_requests(submission_id).values()
            )
            for submission_id in outcome.submission_ids
        }
    return outcome, parts, members


def test_the_slice_budget_counts_the_context_image_of_every_request() -> None:
    request = chunk_request(page_images(1)[0], detail="high")
    plain: Any = SimpleNamespace(context_image=None)
    shared: Any = SimpleNamespace(context_image=("A" * 100, "image/png", None))

    own = submit_module._payload_bytes(request, plain)

    assert own > 0
    assert submit_module._payload_bytes(request, shared) == own + 100
    assert submit_module._payload_bytes(None, shared) == 0


@pytest.fixture
def two_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a slice hold the image data of two test pages."""
    monkeypatch.setattr(submit_module, "SLICE_BYTES", 2 * len("UEFHRQ=="))


@pytest.mark.usefixtures("two_pages")
async def test_pages_are_taken_and_submitted_one_slice_at_a_time(
    world: World, tmp_path: Path
) -> None:
    source = Source(page_images(6, unrendered=frozenset({5})))
    taken_at_upload: list[int] = []
    world.hooks["upload"] = lambda _id: taken_at_upload.append(source.taken)

    outcome, parts, members = await _sliced(tmp_path, source)

    assert outcome.status == "submitted"
    assert parts == [1, 2, 3]
    assert list(members.values()) == [[0, 1], [2, 3], [4, 5]]
    assert taken_at_upload == [3, 5, 6]
    assert len(world.bodies()) == 5
    assert source.closed


@pytest.mark.usefixtures("two_pages")
async def test_a_refused_part_leaves_the_later_pages_untaken(
    world: World, tmp_path: Path
) -> None:
    source = Source(page_images(7))
    creates: list[str] = []

    def create(submission_id: str) -> None:
        creates.append(submission_id)
        if len(creates) == 2:
            raise RuntimeError("too many batches")

    world.hooks["create"] = create

    outcome, parts, members = await _sliced(tmp_path, source)

    assert outcome.status == "refused"
    assert len(outcome.submission_ids) == 2
    assert parts == [1, 2]
    assert list(members.values()) == [[0, 1], [2, 3]]
    assert (source.taken, source.closed) == (5, True)
    async with open_store(tmp_path) as store:
        stored = {row.idx: row.status for row in store.units(1)}
    assert set(stored) == set(range(4))
