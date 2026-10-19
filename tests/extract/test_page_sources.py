"""Page sources: ChronoMiner's page names, page chunks and file payloads."""

from __future__ import annotations

import base64
import hashlib
import random
import tracemalloc
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import fitz
import pytest
from PIL import Image

from chronominer.extract.chunks import extract_payload
from chronominer.extract.images import (
    FolderPages,
    PageImage,
    encode_image_to_base64,
    load_image,
    stream_pages,
)
from chronominer.extract.pdf import PdfPages, PDFProcessor, render_pdf_page

IMG_CFG: dict[str, Any] = {
    "target_dpi": 72,
    "grayscale_conversion": True,
    "handle_transparency": True,
    "jpeg_quality": 90,
    "llm_detail": "high",
    "high_target_box": [64, 128],
    "resize_profile": "auto",
}


def make_pdf(path: Path, pages: int = 3) -> Path:
    with fitz.open() as doc:
        for i in range(pages):
            page = doc.new_page(width=200, height=100)
            page.insert_text((20, 50), f"Page {i + 1}")
        doc.save(str(path))
    return path


def make_image(path: Path, size: tuple[int, int] = (60, 40)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (20, 20, 20)).save(path)
    return path


async def collect(source: Any, indices: list[int] | None = None) -> list[PageImage]:
    return [page async for page in stream_pages(source, indices)]


class TestPdfStream:
    async def test_yields_named_pages_with_provenance(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf")
        pages = await collect(PdfPages(pdf, IMG_CFG, "openai"))

        assert [p.index for p in pages] == [0, 1, 2]
        assert [p.image_name for p in pages] == ["page_0001", "page_0002", "page_0003"]
        for page in pages:
            assert page.render_error is None
            assert page.source_file == str(pdf)
            assert page.page_index == page.index
            data = base64.b64decode(page.base64)
            assert page.mime_type == "image/jpeg"
            record = page.image_provenance
            assert record["byte_size"] == len(data)
            assert record["image_sha256"] == hashlib.sha256(data).hexdigest()
            # The direct strategy renders at a whole DPI, never above the target.
            assert isinstance(record["effective_dpi"], int)
            assert 0 < record["effective_dpi"] <= 72
            assert record["render_dpi"] == record["effective_dpi"]

    async def test_streams_only_the_selected_pages(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=5)
        pages = await collect(PdfPages(pdf, IMG_CFG, "openai"), [1, 3, 99])
        assert [p.index for p in pages] == [1, 3]

    async def test_memory_stays_bounded(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=40)
        tracemalloc.start()
        try:
            count = 0
            async for _page in stream_pages(
                PdfPages(pdf, {**IMG_CFG, "target_dpi": 150}, "openai")
            ):
                count += 1
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert count == 40
        assert peak < 80 * 1024 * 1024

    async def test_the_source_is_closed_afterwards(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=1)
        source = PdfPages(pdf, IMG_CFG, "openai")
        await collect(source)
        assert len(source) == 0

    def test_supersample_renders_at_the_target(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=1)
        cfg = {**IMG_CFG, "render_strategy": "supersample"}
        page = render_pdf_page(pdf, 0, cfg, "openai")
        assert page.image_provenance["effective_dpi"] == 72

    def test_direct_renders_below_the_target_for_a_box_fit(
        self, tmp_path: Path
    ) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=1)
        page = render_pdf_page(pdf, 0, IMG_CFG, "openai")
        # 200 pt at 72 DPI fit into a 64 px box: round(72 * 64 / 200).
        assert page.image_provenance["effective_dpi"] == 23

    def test_the_memory_guard_lowers_the_whole_dpi(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=1)
        cfg = {
            **IMG_CFG,
            "target_dpi": 600,
            "resize_profile": "none",
            "max_pixels_per_page": 100_000,
        }
        page = render_pdf_page(pdf, 0, cfg, "openai")
        dpi = page.image_provenance["effective_dpi"]
        # 200 x 100 pt at 600 DPI is 1.39 MP; the guard floors 160.99 to 160.
        assert dpi == 160
        assert page.image_provenance["downscale_reason"] == "memory_guard"

    def test_a_single_page_keeps_its_name(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=2)
        page = render_pdf_page(pdf, 1, IMG_CFG, "openai")
        assert (page.index, page.page_index) == (1, 1)
        assert page.image_name == "page_0002"
        with pytest.raises(IndexError, match="out of range"):
            render_pdf_page(pdf, 2, IMG_CFG, "openai")


class TestImageFiles:
    async def test_an_image_file_is_one_page_named_by_its_file(
        self, tmp_path: Path
    ) -> None:
        path = make_image(tmp_path / "scans" / "scan_00.png")
        pages = await collect(
            FolderPages(path.parent, IMG_CFG, "openai", image_paths=[path])
        )

        (page,) = pages
        assert (page.index, page.image_name, page.page_index) == (
            0,
            "scan_00.png",
            None,
        )
        assert page.source_file == str(path)
        assert page.image_provenance["effective_dpi"] is None
        assert page.image_provenance["dpi_source"] == "file"

    def test_a_single_image_takes_the_given_index(self, tmp_path: Path) -> None:
        path = make_image(tmp_path / "scan.png", (30, 30))
        page = load_image(path, 4, IMG_CFG, "openai")
        assert page.image_name == "scan.png"
        assert page.index == 4

    def test_png_is_encoded_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.png"
        Image.new("RGB", (1, 1), (255, 0, 0)).save(path, format="PNG")
        data, mime_type = encode_image_to_base64(path)
        assert mime_type == "image/png"
        assert base64.b64decode(data) == path.read_bytes()

    def test_mime_type_follows_the_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.jpg"
        Image.new("RGB", (2, 2), (0, 255, 0)).save(path, format="JPEG")
        assert encode_image_to_base64(path)[1] == "image/jpeg"

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "file.xyz"
        path.write_bytes(b"nothing")
        with pytest.raises(ValueError, match="Unsupported image format"):
            encode_image_to_base64(path)


class TestPageChunks:
    def test_the_provenance_is_the_chunk_record(self, tmp_path: Path) -> None:
        path = make_image(tmp_path / "page.png")
        page = replace(load_image(path, 0, IMG_CFG, "openai"), custom_id="page-page-1")

        record = page.provenance()

        assert record["custom_id"] == "page-page-1"
        assert record["chunk_range"] is None
        assert record["image_provenance"]["width"] == 64
        assert page.chunk_range is None
        assert page.text is None


class TestRenderErrors:
    async def test_an_oversized_page_becomes_an_error_page(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "page_2.png"
        noise = random.Random(7).randbytes(300 * 300 * 3)
        Image.frombytes("RGB", (300, 300), noise).save(path)
        cfg = {
            "payload_format": "jpeg",
            "jpeg_quality": 95,
            "grayscale_conversion": False,
            "max_image_bytes": 20000,
            "resize_profile": "none",
        }

        (failed,) = await collect(
            FolderPages(tmp_path, cfg, "custom", image_paths=[path])
        )

        assert failed.render_error is not None
        assert failed.image_name == "page_2.png"
        assert failed.source_file == str(path)
        assert (
            "max_image_bytes" in failed.provenance()["image_provenance"]["render_error"]
        )
        extractor = MagicMock()
        extractor.extract_image_from_base64 = AsyncMock()
        result = await extract_payload(failed, extractor)
        assert result == (failed, "page_2.png", None, None, 0)
        extractor.extract_image_from_base64.assert_not_called()

    async def test_too_many_failures_fail_the_item(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=3)
        cfg = {**IMG_CFG, "max_image_bytes": 10}

        with pytest.raises(RuntimeError, match="3/3 pages"):
            await collect(PdfPages(pdf, cfg, "openai"))


class TestPDFProcessor:
    def test_enter_opens_and_exit_closes(self, tmp_path: Path) -> None:
        pdf = make_pdf(tmp_path / "doc.pdf", pages=2)
        proc = PDFProcessor(pdf)
        with proc:
            assert proc.doc is not None
            assert proc.doc.page_count == 2
        assert proc.doc is None

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        proc = PDFProcessor(make_pdf(tmp_path / "doc.pdf"))
        proc.open_pdf()
        proc.close_pdf()
        proc.close_pdf()
        assert proc.doc is None

    def test_page_count_closes_the_document(self, tmp_path: Path) -> None:
        proc = PDFProcessor(make_pdf(tmp_path / "doc.pdf", pages=4))
        assert proc.page_count() == 4
        assert proc.doc is None

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="no such file|not found|Failed"):
            PDFProcessor(tmp_path / "missing.pdf").open_pdf()
