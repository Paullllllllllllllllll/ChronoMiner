"""Image file types the tool reads: extensions and their MIME types."""

from __future__ import annotations

__all__ = ["SUPPORTED_IMAGE_EXTENSIONS", "SUPPORTED_IMAGE_FORMATS"]

SUPPORTED_IMAGE_FORMATS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".jp2": "image/jp2",
    ".j2k": "image/jp2",
    ".pbm": "image/x-portable-bitmap",
    ".pgm": "image/x-portable-graymap",
    ".ppm": "image/x-portable-pixmap",
}
"""The MIME type of every supported image extension."""

SUPPORTED_IMAGE_EXTENSIONS = frozenset(SUPPORTED_IMAGE_FORMATS)
