"""Image preprocessing utilities for LLM vision inputs.

Provides image preprocessing (resize, grayscale, transparency handling) for
sending images to vision-capable LLMs. Ported from ChronoTranscriber with
Tesseract/multiprocessing paths removed.

Dependencies: Pillow only.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageOps

from modules.config.constants import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger(__name__)


def detect_model_type(provider: str, model_name: Optional[str] = None) -> str:
    """Detect the underlying model type from provider and model name.

    Allows correct preprocessing even when using models via OpenRouter.

    Args:
        provider: The LLM provider name (e.g., 'openai', 'anthropic', 'google', 'openrouter')
        model_name: The model name (e.g., 'gpt-5-mini', 'claude-sonnet-4-5', 'gemini-2.5-flash')

    Returns:
        Model type: 'google', 'anthropic', or 'openai'
    """
    provider = provider.lower()
    model_name = model_name.lower() if model_name else ""

    if provider == "google":
        return "google"
    if provider == "anthropic":
        return "anthropic"
    if provider == "openai":
        return "openai"

    # For OpenRouter or unknown providers, detect from model name
    if model_name:
        if "gemini" in model_name or "google/" in model_name:
            return "google"
        if "claude" in model_name or "anthropic/" in model_name:
            return "anthropic"
        if any(x in model_name for x in ["gpt-", "o1", "o3", "o4", "openai/"]):
            return "openai"

    return "openai"


def get_image_config_section_name(model_type: str) -> str:
    """Get the image processing config section name for a model type.

    Args:
        model_type: The model type ('google', 'anthropic', or 'openai')

    Returns:
        Config section name (e.g., 'google_image_processing', 'api_image_processing')
    """
    if model_type == "google":
        return "google_image_processing"
    elif model_type == "anthropic":
        return "anthropic_image_processing"
    else:
        return "api_image_processing"


def encode_image_to_base64(image_path: Path) -> Tuple[str, str]:
    """Encode an image file to base64.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (base64_data, mime_type)

    Raises:
        ValueError: If the image format is not supported
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
        base64_data: Base64-encoded image data
        mime_type: MIME type of the image

    Returns:
        Data URL string
    """
    return f"data:{mime_type};base64,{base64_data}"


class ImageProcessor:
    """Preprocesses images for LLM vision APIs.

    Handles transparency flattening, grayscale conversion, and provider-specific
    resizing before images are sent to vision-capable LLMs.
    """

    def __init__(
        self,
        image_path: Path,
        provider: str = "openai",
        model_name: str = "",
        image_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ImageProcessor with provider-specific config.

        Args:
            image_path: Path to the image file
            provider: Provider name (openai, google, anthropic, openrouter)
            model_name: Model name for detecting underlying model type
            image_config: Full image processing config dict. If None, loaded from ConfigLoader.
        """
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        self.image_path = image_path
        self.provider = provider.lower()
        self.model_name = model_name.lower() if model_name else ""

        self.model_type = detect_model_type(self.provider, self.model_name)

        if image_config is None:
            from modules.config.loader import get_config_loader
            image_config = get_config_loader().get_image_processing_config()

        section_name = get_image_config_section_name(self.model_type)
        self.img_cfg = image_config.get(section_name, {})

    def handle_transparency(self, image: Image.Image) -> Image.Image:
        """Flatten transparency by pasting the image onto a white background."""
        if self.img_cfg.get("handle_transparency", True):
            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                return background
        return image

    def convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        """Convert the image to grayscale if enabled in config."""
        if self.img_cfg.get("grayscale_conversion", True):
            if image.mode == "L":
                return image
            return ImageOps.grayscale(image)
        return image

    @staticmethod
    def resize_for_detail(
        image: Image.Image, detail: str, img_cfg: dict, model_type: str = "openai"
    ) -> Image.Image:
        """Resize strategy based on desired LLM detail and model type.

        - low: downscale longest side to low_max_side_px.
        - high/auto: fit/pad into high_target_box (OpenAI/Google) or
                     cap longest side to high_max_side_px (Anthropic).
        - original: cap to original_max_side_px and max_pixels (GPT-5.4+).

        Args:
            image: PIL Image to resize
            detail: Detail level ('low', 'high', 'auto', 'original')
            img_cfg: Config dict for the provider
            model_type: 'openai', 'google', or 'anthropic'
        """
        resize_profile = (img_cfg.get("resize_profile", "auto") or "auto").lower()
        if resize_profile == "none":
            return image
        detail_norm = (detail or "high").lower()
        if detail_norm not in ("low", "high", "auto", "medium", "ultra_high", "original"):
            detail_norm = "high"

        # Low detail: cap longest side (same for all providers)
        if detail_norm == "low":
            max_side = int(img_cfg.get("low_max_side_px", 512))
            w, h = image.size
            longest = max(w, h)
            if longest <= max_side:
                return image
            scale = max_side / float(longest)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            return image.resize(new_size, Image.Resampling.LANCZOS)

        # Original detail (GPT-5.4+): cap to max side / max pixels, no padding
        if detail_norm == "original":
            max_side = int(img_cfg.get("original_max_side_px", 6000))
            max_pixels = int(img_cfg.get("original_max_pixels", 10240000))
            w, h = image.size
            longest = max(w, h)
            if longest > max_side:
                scale = max_side / float(longest)
                w = max(1, int(w * scale))
                h = max(1, int(h * scale))
                image = image.resize((w, h), Image.Resampling.LANCZOS)
            if w * h > max_pixels:
                scale = (max_pixels / float(w * h)) ** 0.5
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            return image

        # High/auto/medium/ultra_high: provider-specific strategy
        if model_type == "anthropic":
            max_side = int(img_cfg.get("high_max_side_px", 1568))
            w, h = image.size
            longest = max(w, h)
            if longest <= max_side:
                return image
            scale = max_side / float(longest)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            # OpenAI/Google: fit into box and pad with white
            box = img_cfg.get("high_target_box", [768, 1536])
            try:
                target_width = int(box[0])
                target_height = int(box[1])
            except Exception:
                target_width, target_height = 768, 1536
            orig_width, orig_height = image.size
            scale_w = target_width / orig_width
            scale_h = target_height / orig_height
            scale = min(scale_w, scale_h)
            new_width = max(1, int(orig_width * scale))
            new_height = max(1, int(orig_height * scale))
            resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            if resized_img.mode not in ("RGB", "L"):
                resized_img = resized_img.convert("RGB")
            if resized_img.mode == "L":
                final_img = Image.new("L", (target_width, target_height), 255)
            else:
                final_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            final_img.paste(resized_img, (paste_x, paste_y))
            return final_img

    def process_image(self, output_path: Path) -> Path:
        """Process the image and save it to the given output path as JPEG.

        Returns:
            Path to the saved processed image (with .jpg extension).
        """
        try:
            with Image.open(self.image_path) as _raw_img:
                img: Image.Image = _raw_img
                if self.model_type == "google":
                    detail = self.img_cfg.get("media_resolution", "high") or "high"
                elif self.model_type == "anthropic":
                    detail = self.img_cfg.get("resize_profile", "auto") or "auto"
                else:
                    detail = self.img_cfg.get("llm_detail", "high") or "high"

                img = self.handle_transparency(img)
                img = self.convert_to_grayscale(img)
                img = ImageProcessor.resize_for_detail(img, detail, self.img_cfg, self.model_type)

                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                jpg_output_path = output_path.with_suffix(".jpg")
                jpeg_quality = int(self.img_cfg.get("jpeg_quality", 95))
                img.save(jpg_output_path, format="JPEG", quality=jpeg_quality)
                logger.debug(
                    "Saved processed image %s size=%s quality=%d detail=%s",
                    jpg_output_path.name, img.size, jpeg_quality, detail,
                )
            return jpg_output_path
        except Exception as e:
            logger.error("Error processing image %s: %s", self.image_path.name, e)
            raise
