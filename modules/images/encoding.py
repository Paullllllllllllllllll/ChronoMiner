"""Image encoding helpers for LLM vision inputs.

Pure helpers: image bytes -> base64, and base64 -> data URL.
No image transformation lives here; see :mod:`modules.images.llm_preprocess`
for provider-specific preprocessing.
"""

from __future__ import annotations

import base64

from modules.config.constants import SUPPORTED_IMAGE_FORMATS


def encode_bytes_to_base64(data: bytes, mime_type: str = "image/jpeg") -> str:
    """Encode raw image bytes to base64.

    Serves the in-memory pipelines that never touch disk (see
    ``modules.images.page_stream``).

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
