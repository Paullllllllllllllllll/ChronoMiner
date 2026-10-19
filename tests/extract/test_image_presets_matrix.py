"""Characterization of the image presets and the resolved image settings.

The cases record the recommended image values the wizard shows per model
family, the request table (:data:`IMAGE_PROCESSING`), and the image settings
that :func:`resolve_image_settings` returns for providers, models and run
image options, with the stored image-settings record of each model's
defaults. Key order is recorded too, since the record is stored as JSON and
compared on resume. ``golden/image_presets.json`` holds the values of
v1.0.0 before the presets moved to ``common/image_presets.py``; the move must
leave every case identical.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest

from chronominer.common.testing import WriteGuard
from chronominer.extract.images import (
    IMAGE_PROCESSING,
    RESAMPLING_ALGORITHM,
    resolve_image_settings,
    with_resampling,
)
from chronominer.output.store import image_settings_record
from chronominer.spec import ImageSpec, recommended_images
from tests.matrix_golden import check_matrix

FAMILIES = ("openai", "anthropic", "google", "custom", "unknown")
MODELS: tuple[tuple[str | None, str | None], ...] = (
    ("openai", "gpt-6-luna"),
    ("openai", "gpt-5.6-luna"),
    ("openai", "gpt-5.3"),
    ("openai", "gpt-4o"),
    ("anthropic", "claude-haiku-5-5"),
    ("anthropic", "claude-sonnet-4-0"),
    ("google", "gemini-3.1-pro"),
    ("google", "gemini-2.5-flash"),
    ("openrouter", "openai/gpt-5.6-luna"),
    ("openrouter", "anthropic/claude-sonnet-4.5"),
    ("openrouter", "google/gemini-3.1-pro"),
    ("openrouter", "meta/llama-4"),
    ("custom", "org/fake-model"),
    (None, "gpt-6-luna"),
)
IMAGES: Mapping[str, ImageSpec | None] = {
    "none": None,
    "unset": ImageSpec(),
    "detail-low": ImageSpec(detail="low"),
    "detail-high": ImageSpec(detail="high"),
    "detail-auto": ImageSpec(detail="auto"),
    "detail-original": ImageSpec(detail="original"),
    "dpi-200": ImageSpec(dpi=200),
    "dpi-native": ImageSpec(dpi="native"),
    "png": ImageSpec(format="png"),
    "quality-80": ImageSpec(jpeg_quality=80),
    "gray-on": ImageSpec(grayscale=True),
    "gray-off": ImageSpec(grayscale=False),
    "all-set": ImageSpec(300, "high", "jpeg", 90, False),
}
ORIGINAL_SIDES = (None, 2000)


def _outcome(build: Callable[..., Any], *args: Any) -> Any:
    """Return what *build* returns for *args*, or the error it raises."""
    try:
        return build(*args)
    except Exception as exc:  # noqa: BLE001 - the error is the recorded outcome
        return {"<error>": f"{type(exc).__name__}: {exc}"}


def _resolved(
    provider: str | None,
    model: str | None,
    images: ImageSpec | None,
    side: int | None,
    record: bool,
) -> dict[str, Any]:
    settings, model_type, section = resolve_image_settings(
        provider, model, images, side
    )
    resolved: dict[str, Any] = {
        "settings": settings,
        "model_type": model_type,
        "section": section,
        "rendered_with": with_resampling(settings)["resampling_algorithm"],
    }
    if record:
        resolved["record"] = image_settings_record(settings, model_type)
    return resolved


def _cases() -> Iterator[tuple[str, Any]]:
    for family in FAMILIES:
        yield f"recommended:{family}", recommended_images(family).__dict__
    yield "request-table", IMAGE_PROCESSING
    yield "resampling", RESAMPLING_ALGORITHM
    for provider, model in MODELS:
        for name, images in IMAGES.items():
            for side in ORIGINAL_SIDES:
                key = f"{provider}:{model}:{name}:side-{side}"
                record = name == "unset" and side is None
                yield key, _outcome(_resolved, provider, model, images, side, record)


@pytest.mark.unit
def test_image_presets_match_the_golden(write_guard: WriteGuard) -> None:
    check_matrix(__file__, "image_presets", dict(_cases()), write_guard.consume)
