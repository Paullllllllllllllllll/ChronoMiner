"""Vision-model input preparation for ChronoMiner.

Image encoding, provider-specific preprocessing (resize, transparency,
grayscale), content-block assembly for multimodal LLM messages, and PDF
page rendering. All inputs destined for vision-capable LLMs pass through
this package.
"""

from modules.images.encoding import create_data_url, encode_image_to_base64
from modules.images.llm_preprocess import (
    ImageProcessor,
    detect_model_type,
    get_image_config_section_name,
)
from modules.images.message_builder import build_image_content_block
from modules.images.pdf_utils import PDFProcessor

__all__ = [
    "ImageProcessor",
    "PDFProcessor",
    "build_image_content_block",
    "create_data_url",
    "detect_model_type",
    "encode_image_to_base64",
    "get_image_config_section_name",
]
