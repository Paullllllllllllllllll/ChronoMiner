"""The effective spec of a run with the model and image values it resolves.

A run's effective spec lists every field with its value and source. A field
left unset (the output limit, the response format, the image options) is
resolved when the run builds its requests; :func:`resolved_view` writes the
resolved value into such a field with the source ``resolved``, and removes
the service tier for providers other than OpenAI, which never receive one.
The dry run prints this view; the store keeps the unresolved spec, from
which a later command may rebuild the run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["RESOLVED", "image_values", "resolved_view"]

RESOLVED = "resolved"
"""The source of a value the run resolved instead of the user setting it."""

_IMAGE_KEYS: Mapping[str, str] = {
    "dpi": "target_dpi",
    "detail": "resolved_detail",
    "format": "payload_format",
    "jpeg_quality": "jpeg_quality",
    "grayscale": "grayscale_conversion",
}
"""Each image option and the key of the resolved image settings it shows."""


def image_values(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return the image options as resolved image *settings* hold them."""
    return {
        option: settings[key] for option, key in _IMAGE_KEYS.items() if key in settings
    }


def resolved_view(
    view: Mapping[str, Mapping[str, Any]],
    *,
    model: str,
    provider: str,
    limit: int | None = None,
    response_format: str | None = None,
    images: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return *view* with the unset fields of one model filled in.

    *model* is the dotted prefix of the model's fields (``model``,
    ``transcription_model``); *limit* and *response_format* are the output
    limit and response format its requests use, *images* the resolved image
    settings of its pages (:func:`image_values`). Fields the user set keep
    their value and source.
    """
    out = {path: dict(entry) for path, entry in view.items()}
    if provider != "openai":
        out.pop(f"{model}.service_tier", None)
    resolved: dict[str, Any] = {
        f"{model}.max_output_tokens": limit,
        f"{model}.response_format": response_format,
    }
    if images is not None:
        resolved.update(
            {
                f"images.{option}": value
                for option, value in image_values(images).items()
            }
        )
    for path, value in resolved.items():
        entry = out.get(path)
        if value is None or entry is None or entry.get("value") is not None:
            continue
        out[path] = {"value": value, "source": RESOLVED}
    return out
