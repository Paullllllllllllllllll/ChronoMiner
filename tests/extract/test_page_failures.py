"""Chunk failures: a failed chunk fails its unit and its item, and a stream error
from the page path propagates instead of counting as processed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import chronominer.extract.llm as llm_module
from chronominer.extract import Item
from chronominer.extract.chunks import TextChunk, extract_payload
from chronominer.extract.llm import Extractor
from chronominer.extract.units import UnitsFailedError, run_unit_stream
from chronominer.output.store import database_path, open_store
from tests.conftest import make_job, make_spec
from tests.extract.conftest import make_store, open_handle, payloads, stream, units


@pytest.mark.unit
class TestChunkFailuresFailTheItem:
    async def test_streaming_pipeline_raises_when_a_page_errors(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "chronominer.sqlite"
        payload = SimpleNamespace(
            image_name="page_0001",
            base64="Zm9v",
            mime_type="image/png",
            index=0,
            source_file="doc.pdf",
            page_index=0,
            render_error=None,
            provenance=lambda: {},
        )
        extractor = MagicMock()
        extractor.extract_image_from_base64 = MagicMock(
            side_effect=RuntimeError("API 404")
        )

        async with make_store(db) as store:
            handle = await open_handle(store, pages=1)
            with pytest.raises(UnitsFailedError):
                await run_unit_stream(
                    handle,
                    stream([payload]),
                    extractor,
                    extract_payload,
                    concurrency=2,
                )

        (row,) = units(db)
        assert (row["status"], row["content"]) == ("failed", None)

    async def test_a_text_chunk_error_answer_fails_its_unit(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "chronominer.sqlite"
        chunk = TextChunk(0, 1, 3, "Record 001.", "doc-chunk-1")

        class Failing:
            async def extract_text(self, text: str, *, label: str) -> Any:
                assert (text, label) == ("Record 001.", "1-3")
                return {"output_text": "", "error": "status 400"}

        async with make_store(db) as store:
            handle = await open_handle(store, pages=1)
            with pytest.raises(UnitsFailedError):
                await run_unit_stream(
                    handle, stream([chunk]), Failing(), extract_payload
                )

        (row,) = units(db)
        assert (row["status"], row["error"]) == ("failed", "status 400")


class TestUnitStatusFollowsTheResult:
    @pytest.mark.parametrize(
        ("result", "status"),
        [
            (("p", "page_0001", None, None, 0), "failed"),
            (("p", "page_0001", "", None, 0), "failed"),
            (("p", "page_0001", "  \n", None, 0), "failed"),
            (("p", "page_0001", "real answer", None, 0), "done"),
            (("a", "b"), "failed"),
            (None, "failed"),
        ],
    )
    async def test_status(self, tmp_path: Path, result: Any, status: str) -> None:
        async with open_store(tmp_path) as store:
            handle = await open_handle(store, pages=1)

            async def handler(_payload: Any, _extractor: Any) -> Any:
                return result

            try:
                await run_unit_stream(handle, stream(payloads(1)), None, handler)
            except UnitsFailedError:
                assert status == "failed"
        (unit,) = units(database_path(tmp_path))
        assert unit["status"] == status


class _GuardError(RuntimeError):
    """A RuntimeError subclass, as the render failure-rate guard raises."""


@pytest.mark.unit
class TestPageStreamErrors:
    async def test_runtime_error_from_the_page_stream_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image = tmp_path / "page1.png"
        image.write_bytes(b"\x89PNG")
        spec = make_spec(image)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        extractor = Extractor(
            make_job(spec, output_dir=tmp_path / "out", input_root=tmp_path)
        )
        extractor._client = MagicMock()

        def failing_stream(_pages: Any, _indices: Any) -> Any:
            async def gen() -> Any:
                # A RuntimeError subclass must not be swallowed as a skip.
                raise _GuardError("render failure rate exceeded")
                yield

            return gen()

        monkeypatch.setattr(llm_module, "stream_pages", failing_stream)

        async with open_store(tmp_path) as store:
            handle = await open_handle(store, "page1", pages=0)
            item = Item(image, "image", document=handle)
            with pytest.raises(_GuardError):
                await extractor.extract(item)
