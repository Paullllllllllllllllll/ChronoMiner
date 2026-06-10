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


def encode_bytes_to_base64(data: bytes, mime_type: str = "image/jpeg") -> str:
    """Encode raw image bytes to base64.

    Counterpart of :func:`encode_image_to_base64` for in-memory pipelines
    that never touch disk (see ``modules.images.page_stream``).

    Args:
        data: Raw image bytes (e.g., an encoded JPEG).
        mime_type: MIME type of the image data.

    Returns:
        Base64-encoded string.

    Raises:
        ValueError: If the MIME type is not a supported image type.
    """
    if mime_type not in SUPPORTED_IMAGE_FORMATS.values():
        raise ValueError(f"Unsupported image MIME type: {mime_type}")
    return base64.b64encode(data).decode("utf-8")


def create_data_url(base64_data: str, mime_type: str) -> str:
    """Create a data URL from base64 data.

    Args:
        base64_data: Base64-encoded image data.
        mime_type: MIME type of the image.

    Returns:
        Data URL string.
    """
    return f"data:{mime_type};base64,{base64_data}"
