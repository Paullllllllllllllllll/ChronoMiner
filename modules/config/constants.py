"""Centralized constants used across the application.

Defines supported image formats, PDF extensions, and other shared constants.
"""

from __future__ import annotations

# Supported image extensions and their MIME types for data URLs
SUPPORTED_IMAGE_FORMATS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Convenience set of supported extensions
SUPPORTED_IMAGE_EXTENSIONS = set(SUPPORTED_IMAGE_FORMATS.keys())

# Supported PDF extensions
SUPPORTED_PDF_EXTENSIONS = {".pdf"}

# All visual input extensions (images + PDFs)
SUPPORTED_VISUAL_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS
