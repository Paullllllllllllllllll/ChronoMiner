"""Page sources downscale with Lanczos without changing the settings fingerprint."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import fitz
import pytest
from PIL import Image, ImageDraw

from chronominer.common.images import FolderPageSource, resampling_filter
from chronominer.common.native import image_settings_fingerprint
from chronominer.extract.images import (
    PAYLOAD_SHA256_KEY,
    FolderPages,
    resolve_image_settings,
)
from chronominer.extract.pdf import PdfPages
from chronominer.output.store import image_settings_record

MODELS = [
    ("anthropic", "claude-sonnet-4.5"),
    ("google", "gemini-2.5-flash"),
    ("openai", "gpt-4o"),
]


@pytest.fixture(scope="module", params=["png", "jpg"])
def page_file(request: pytest.FixtureRequest, tmp_path_factory: Any) -> Path:
    """A large, text-like scan that every model in MODELS downscales."""
    image = Image.new("L", (2400, 3200), 245)
    draw = ImageDraw.Draw(image)
    rng = random.Random(7)
    for top in range(120, 3080, 48):
        left = 150
        while left < 2200:
            width = rng.randint(30, 160)
            draw.rectangle((left, top, left + width, top + 22), fill=rng.randint(0, 60))
            left += width + rng.randint(15, 30)
    path = tmp_path_factory.mktemp(request.param) / f"scan_001.{request.param}"
    image.save(path)
    return path


def _vendored_payload(path: Path, cfg: dict[str, Any], model_type: str) -> str:
    source = FolderPageSource(
        path.parent,
        cfg,
        model_type,
        image_paths=[path],
        payload_sha256_key=PAYLOAD_SHA256_KEY,
    )
    return source.build_payload(0).base64


@pytest.mark.parametrize(("provider", "model"), MODELS)
def test_image_pages_downscale_with_lanczos(
    page_file: Path, provider: str, model: str
) -> None:
    img_cfg, model_type, section = resolve_image_settings(provider, model)
    pages = FolderPages(
        page_file.parent, img_cfg, model_type, image_paths=[page_file], section=section
    )
    payload = pages.build_payload(0).base64
    lanczos = {**img_cfg, "resampling_algorithm": "lanczos"}
    assert payload == _vendored_payload(page_file, lanczos, model_type)
    assert payload != _vendored_payload(page_file, img_cfg, model_type)


def _provenance(img_cfg: dict[str, Any], model_type: str) -> dict[str, Any]:
    record = image_settings_record(img_cfg, model_type, 0)
    return {"record": record, "fingerprint": image_settings_fingerprint(record)}


@pytest.mark.parametrize(("provider", "model"), MODELS)
def test_settings_and_fingerprint_stay_free_of_the_filter(
    tmp_path: Path, page_file: Path, provider: str, model: str
) -> None:
    img_cfg, model_type, section = resolve_image_settings(provider, model)
    assert "resampling_algorithm" not in img_cfg
    snapshot = dict(img_cfg)
    before = _provenance(img_cfg, model_type)

    FolderPages(page_file.parent, img_cfg, model_type, image_paths=[page_file])
    pdf = tmp_path / "doc.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.save(str(pdf))
    with PdfPages(pdf, img_cfg, model_type, section=section):
        pass

    assert img_cfg == snapshot
    assert _provenance(img_cfg, model_type) == before
    assert "resampling_algorithm" not in before["record"]


def test_pdf_pages_carry_the_lanczos_filter(tmp_path: Path) -> None:
    img_cfg, model_type, section = resolve_image_settings("openai", "gpt-4o")
    pdf = tmp_path / "doc.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.save(str(pdf))
    with PdfPages(pdf, img_cfg, model_type, section=section) as pages:
        assert pages.img_cfg["resampling_algorithm"] == "lanczos"
        assert resampling_filter(pages.img_cfg) is Image.Resampling.LANCZOS
