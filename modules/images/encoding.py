"""Image encoding helpers for LLM vision inputs.

Pure helpers: file path -> base64 + MIME type, and base64 -> data URL.
No image transformation lives here; see :mod:`modules.images.llm_preprocess`
for provider-specific preprocessing.
"""

from __future__ import annotations

import base64
from pathlib import Path

from modules.config.constants import SUPPORTED_IMAGE_FORMATS


def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
    """Encode an image file to base64.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (base64_data, mime_type).

    Raises:
        ValueError: If the image format is not supported.
    """
    ext = image_path.suffix.lower()
    mime_type = SUPPORTED_IMAGE_FORMATS.get(ext)
    if not mime_type:
        raise ValueError(f"Unsupported image format: {ext}")

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return data, mime_type


def create_data_url(base64_data: str, mime_type: str) -> str:
    """Create a data URL from base64 data.

    Args:
        base64_data: Base64-encoded image data.
        mime_type: MIME type of the image.

    Returns:
        Data URL string.
    """
    return f"data:{mime_type};base64,{base64_data}"
