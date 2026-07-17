"""Vision-model input preparation for ChronoMiner.

Image encoding, provider-specific preprocessing (resize, transparency,
grayscale), content-block assembly for multimodal LLM messages, and PDF
page rendering. All inputs destined for vision-capable LLMs pass through
this package.
"""

from modules.images.encoding import (
    create_data_url,
    encode_bytes_to_base64,
    encode_image_to_base64,
)
from modules.images.llm_preprocess import (
    ImageProcessor,
    detect_model_type,
    get_image_config_section_name,
)
from modules.images.message_builder import build_image_content_block
from modules.images.page_stream import (
    PageError,
    PagePayload,
    build_image_provenance,
    resolve_image_section,
    resolve_target_dpi,
    stream_page_payloads,
)
from modules.images.pdf_utils import PDFProcessor

__all__ = [
    "ImageProcessor",
    "PDFProcessor",
    "PageError",
    "PagePayload",
    "build_image_content_block",
    "build_image_provenance",
    "create_data_url",
    "detect_model_type",
    "encode_bytes_to_base64",
    "encode_image_to_base64",
    "get_image_config_section_name",
    "resolve_image_section",
    "resolve_target_dpi",
    "stream_page_payloads",
]
