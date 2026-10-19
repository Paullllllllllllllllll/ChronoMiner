"""Batch submission: submit_batch on the store, for text and image chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chronominer import ext
from chronominer.batch.submit import (
    BatchSubmissionError,
    SubmitOutcome,
    chunk_request,
    submit_batch,
)
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
