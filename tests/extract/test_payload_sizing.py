"""The page sizes ChronoMiner sends, pinned to the values of its earlier stack.

The sizes and render resolutions below are those the earlier ChronoMiner
image pipeline produced for the same synthetic inputs. Callers that pack
pages to these sizes (10.24 MP under the 10k-patch original cap) depend on
them, so a change here changes what the tool sends.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

import fitz
import pytest
from PIL import Image

from chronominer.common.images import resampling_filter
from chronominer.extract.images import (
    IMAGE_PROCESSING,
    FolderPages,
    PageImage,
    resolve_image_settings,
    stream_pages,
)
from chronominer.extract.pdf import PdfPages
from chronominer.spec import ImageSpec

OLD_SECTIONS: dict[str, dict[str, Any]] = {
    "api_image_processing": {
        "target_dpi": "native",
        "native_fallback_dpi": 300,
        "payload_format": "jpeg",
        "max_image_bytes": 50_000_000,
        "grayscale_conversion": False,
        "handle_transparency": True,
        "jpeg_quality": 95,
        "llm_detail": "original",
        "resize_profile": "high",
        "low_max_side_px": 512,
        "high_target_box": [768, 1536],
    },
    "anthropic_image_processing": {
        "target_dpi": "native",
        "native_fallback_dpi": 300,
        "payload_format": "jpeg",
        "max_image_bytes": 10_000_000,
        "grayscale_conversion": False,
        "handle_transparency": True,
        "jpeg_quality": 95,
        "resize_profile": "auto",
        "low_max_side_px": 512,
        "high_target_box": [768, 1536],
    },
    "google_image_processing": {
        "target_dpi": 300,
        "native_fallback_dpi": 300,
        "payload_format": "jpeg",
        "max_image_bytes": 20_000_000,
        "grayscale_conversion": False,
        "handle_transparency": True,
        "jpeg_quality": 95,
        "media_resolution": "high",
        "resize_profile": "high",
        "low_max_side_px": 512,
        "high_target_box": [768, 1536],
    },
    "custom_image_processing": {
        "target_dpi": 150,
        "native_fallback_dpi": 300,
        "payload_format": "jpeg",
        "max_image_bytes": 0,
        "grayscale_conversion": False,
        "handle_transparency": True,
        "jpeg_quality": 95,
        "llm_detail": "high",
        "resize_profile": "low",
        "low_max_side_px": 768,
        "high_target_box": [512, 1024],
    },
}


def test_the_sections_keep_the_old_sizes_in_colour_at_jpeg_95() -> None:
    assert {key: dict(value) for key, value in IMAGE_PROCESSING.items()} == (
        OLD_SECTIONS
    )


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A page photo, a 35 MP scan and a three-page PDF."""
    root = tmp_path_factory.mktemp("sizing")
    page = root / "page.png"
    Image.new("RGB", (1200, 1700), (240, 236, 225)).save(page)
    large = root / "large.png"
    Image.new("L", (5000, 7000), 245).save(large)
    scan = io.BytesIO()
    Image.new("L", (1654, 2339), 200).save(scan, format="PNG")
    pdf = root / "book.pdf"
    with fitz.open() as doc:
        first = doc.new_page(width=595, height=842)
        first.insert_text((72, 100), "Price list of the town granary, 1701")
        second = doc.new_page(width=595, height=842)
        second.insert_image(second.rect, stream=scan.getvalue())
        third = doc.new_page(width=842, height=595)
        third.insert_image(fitz.Rect(40, 40, 400, 555), stream=scan.getvalue())
        doc.save(str(pdf))
    return {"page": page, "large": large, "pdf": pdf}


def _pages(path: Path, provider: str, model: str, **images: Any) -> list[PageImage]:
    cfg, model_type, section = resolve_image_settings(
        provider, model, ImageSpec(**images)
    )
    source: PdfPages | FolderPages
    if path.suffix == ".pdf":
        source = PdfPages(path, cfg, model_type, section=section)
    else:
        source = FolderPages(
            path.parent, cfg, model_type, image_paths=[path], section=section
        )

    async def collect() -> list[PageImage]:
        return [page async for page in stream_pages(source)]

    return asyncio.run(collect())


def _sizes(pages: list[PageImage]) -> list[tuple[int, int]]:
    return [
        (page.image_provenance["width"], page.image_provenance["height"])
        for page in pages
    ]


@pytest.mark.parametrize(
    ("provider", "model", "detail", "size"),
    [
        ("openai", "gpt-5.6-luna", "high", (768, 1536)),
        ("openai", "gpt-5.6-luna", "low", (361, 511)),
        ("openai", "gpt-5.6-luna", "original", (1200, 1700)),
        ("anthropic", "claude-sonnet-4-5", None, (924, 1310)),
        ("anthropic", "claude-opus-4-7", None, (1200, 1700)),
        ("google", "gemini-2.5-flash", None, (768, 1536)),
        ("openrouter", "anthropic/claude-sonnet-4-5", None, (924, 1310)),
        ("custom", "org/fake-model", None, (512, 1024)),
    ],
)
def test_an_image_page_keeps_its_size(
    inputs: dict[str, Path],
    provider: str,
    model: str,
    detail: str | None,
    size: tuple[int, int],
) -> None:
    pages = _pages(inputs["page"], provider, model, detail=detail)

    assert _sizes(pages) == [size]


@pytest.mark.parametrize(
    ("provider", "model", "detail", "size"),
    [
        ("openai", "gpt-5.6-luna", "original", (4672, 6542)),
        ("openai", "gpt-5.4", "original", (2688, 3764)),
        ("openai", "gpt-5.6-luna", "low", (365, 512)),
        ("anthropic", "claude-sonnet-4-5", None, (924, 1294)),
        ("anthropic", "claude-opus-4-7", None, (1624, 2274)),
        ("custom", "org/fake-model", None, (512, 1024)),
    ],
)
def test_a_large_scan_is_capped_as_before(
    inputs: dict[str, Path],
    provider: str,
    model: str,
    detail: str | None,
    size: tuple[int, int],
) -> None:
    pages = _pages(inputs["large"], provider, model, detail=detail)

    assert _sizes(pages) == [size]
    width, height = size
    if model == "gpt-5.4":
        assert width * height <= 10_240_000


@pytest.mark.parametrize(
    ("provider", "model", "options", "sizes", "dpis"),
    [
        (
            "openai",
            "gpt-5.6-luna",
            {"detail": "original"},
            [(2480, 3509), (1654, 2341), (3509, 2480)],
            None,
        ),
        (
            "google",
            "gemini-2.5-flash",
            {},
            [(768, 1536), (768, 1536), (768, 1536)],
            [93, 93, 66],
        ),
        (
            "custom",
            "org/fake-model",
            {},
            [(512, 1024), (512, 1024), (512, 1024)],
            [62, 62, 44],
        ),
        (
            "anthropic",
            "claude-opus-4-7",
            {},
            [(1622, 2295), (1622, 2296), (2295, 1622)],
            None,
        ),
        (
            "openai",
            "gpt-5.6-luna",
            {"detail": "original", "dpi": 600},
            [(4116, 5824), (4116, 5824), (5824, 4116)],
            [498, 498, 498],
        ),
    ],
)
def test_pdf_pages_keep_their_sizes_and_whole_dpis(
    inputs: dict[str, Path],
    provider: str,
    model: str,
    options: dict[str, Any],
    sizes: list[tuple[int, int]],
    dpis: list[int] | None,
) -> None:
    pages = _pages(inputs["pdf"], provider, model, **options)

    assert _sizes(pages) == sizes
    if dpis is not None:
        assert [page.image_provenance["effective_dpi"] for page in pages] == dpis


def test_every_page_source_downscales_with_lanczos(inputs: dict[str, Path]) -> None:
    cfg, model_type, section = resolve_image_settings("openai", "gpt-5.6-luna")
    image = FolderPages(
        inputs["page"].parent, cfg, model_type, image_paths=[inputs["page"]]
    )
    with PdfPages(inputs["pdf"], cfg, model_type, section=section) as pdf:
        filters = [resampling_filter(image.img_cfg), resampling_filter(pdf.img_cfg)]

    assert filters == [Image.Resampling.LANCZOS] * 2
