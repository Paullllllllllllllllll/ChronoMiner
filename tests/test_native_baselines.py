"""Numeric payload hashes captured before native image support."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import fitz
import pytest
from PIL import Image

from modules.images.page_stream import PagePayload, stream_page_payloads


def numeric_config(provider: str, strategy: str) -> dict[str, Any]:
    """Pin every previous example value rather than inherit shipped defaults."""
    section = {
        "target_dpi": 150 if provider == "custom" else 300,
        "grayscale_conversion": provider != "openai",
        "handle_transparency": True,
        "jpeg_quality": 85 if provider == "custom" else 95,
        "resize_profile": {
            "openai": "original",
            "anthropic": "auto",
            "google": "auto",
            "custom": "low",
        }[provider],
        "low_max_side_px": 768 if provider == "custom" else 512,
    }
    if provider == "openai":
        section.update(
            llm_detail="original",
            high_target_box=[768, 1536],
            original_max_side_px=6000,
            original_max_pixels=10240000,
        )
    elif provider == "anthropic":
        section["high_max_side_px"] = 2576
    elif provider == "google":
        section.update(media_resolution="high", high_target_box=[768, 1536])
    else:
        section["high_target_box"] = [512, 1024]
    name = "api" if provider == "openai" else provider
    return {
        "target_dpi": 300,
        "render_strategy": strategy,
        "max_pixels_per_page": 24000000,
        f"{name}_image_processing": section,
    }


def fixture_files(folder: Path) -> tuple[Path, Path]:
    image = Image.new("RGB", (240, 360), "white")
    image.putdata(
        [
            ((x * 7) % 256, (y * 3) % 256, (x + y) % 256)
            for y in range(360)
            for x in range(240)
        ]
    )
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    image_path = folder / "fixture.png"
    image_path.write_bytes(raw.getvalue())
    pdf_path = folder / "fixture.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=144, height=216)
        page.insert_image(page.rect, stream=raw.getvalue())
        page.insert_text((10, 20), "Synthetic fixture")
        doc.save(pdf_path)
    return pdf_path, image_path


BASELINES: dict[tuple[str, str], tuple[str, str]] = {
    ("openai", "direct"): (
        "5ab321b0e187b580a035aa76deeb61ac91c601eb9b4e9fe096fc3b6f824b8814",
        "76c06906d4ba36ae71e90f2821db24efb919a34a3aec352ef6862cd6d6c15ecb",
    ),
    ("openai", "supersample"): (
        "5ab321b0e187b580a035aa76deeb61ac91c601eb9b4e9fe096fc3b6f824b8814",
        "76c06906d4ba36ae71e90f2821db24efb919a34a3aec352ef6862cd6d6c15ecb",
    ),
    ("anthropic", "direct"): (
        "182b2eb330f4c9a0dcf2f0e0d525e3fe39da29f846c9b1dc227eb520bdafcf10",
        "7c10f2c7865a28a4cc414e6f26a2893216bfc9798facd5f4855b0b05b909bea9",
    ),
    ("anthropic", "supersample"): (
        "182b2eb330f4c9a0dcf2f0e0d525e3fe39da29f846c9b1dc227eb520bdafcf10",
        "7c10f2c7865a28a4cc414e6f26a2893216bfc9798facd5f4855b0b05b909bea9",
    ),
    ("google", "direct"): (
        "5244d12d1d6878c3bd0fc2932871c1a681fa9bb685b56092c12668ec5281ec12",
        "74f2315bd209c24e3c085a19699abc5d7c4dcef8a078b2374579316e4c096a18",
    ),
    ("google", "supersample"): (
        "5244d12d1d6878c3bd0fc2932871c1a681fa9bb685b56092c12668ec5281ec12",
        "74f2315bd209c24e3c085a19699abc5d7c4dcef8a078b2374579316e4c096a18",
    ),
    ("custom", "direct"): (
        "3ea55592a579dc2c70c97e18e98e0a9662745f51bd2d33132e97cea7ea37e620",
        "3e39af16d84afd41cdc96436ea931029559f43dcf154b97ca723b0ac5b828d66",
    ),
    ("custom", "supersample"): (
        "3ea55592a579dc2c70c97e18e98e0a9662745f51bd2d33132e97cea7ea37e620",
        "3e39af16d84afd41cdc96436ea931029559f43dcf154b97ca723b0ac5b828d66",
    ),
}


async def payload_hashes(
    pdf: Path, image: Path, provider: str, strategy: str
) -> tuple[str, str]:
    config = numeric_config(provider, strategy)
    model = "claude-opus-5" if provider == "anthropic" else "gpt-6-astra"
    hashes = []
    for source in (pdf, image):
        payloads = [
            payload
            async for payload in stream_page_payloads(
                source, [1], config, provider, model, None
            )
        ]
        assert len(payloads) == 1
        payload = payloads[0]
        assert isinstance(payload, PagePayload)
        hashes.append(payload.sha256)
    return hashes[0], hashes[1]


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "custom"])
@pytest.mark.parametrize("strategy", ["direct", "supersample"])
@pytest.mark.asyncio
async def test_numeric_payload_baseline(
    tmp_path: Path, provider: str, strategy: str
) -> None:
    pdf, image = fixture_files(tmp_path)
    assert (
        await payload_hashes(pdf, image, provider, strategy)
        == BASELINES[provider, strategy]
    )
