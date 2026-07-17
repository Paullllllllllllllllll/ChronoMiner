"""Provider-specific image preprocessing for LLM vision inputs.

Handles transparency flattening, grayscale conversion, and provider-specific
resizing before images are sent to vision-capable LLMs. Config-driven via
the project's ``image_processing`` YAML section.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from modules.config.constants import SUPPORTED_IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def detect_model_type(provider: str, model_name: str | None = None) -> str:
    """Detect the underlying model type from provider and model name.

    Allows correct preprocessing even when using models via OpenRouter.

    Args:
        provider: The LLM provider name (e.g., 'openai', 'anthropic', 'google',
            'openrouter').
        model_name: The model name (e.g., 'gpt-5-mini', 'claude-sonnet-4-5',
            'gemini-2.5-flash').

    Returns:
        Model type: 'google', 'anthropic', 'openai', or 'custom'.
    """
    provider = provider.lower()
    model_name = model_name.lower() if model_name else ""

    if provider == "custom":
        return "custom"
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
    """Get the image processing config section name for a model type."""
    _SECTION_MAP = {
        "custom": "custom_image_processing",
        "google": "google_image_processing",
        "anthropic": "anthropic_image_processing",
    }
    return _SECTION_MAP.get(model_type, "api_image_processing")


class ImageProcessor:
    """Preprocesses images for LLM vision APIs.

    Handles transparency flattening, grayscale conversion, and
    provider-specific resizing before images are sent to vision-capable LLMs.
    """

    def __init__(
        self,
        image_path: Path | None = None,
        provider: str = "openai",
        model_name: str = "",
        image_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ImageProcessor with provider-specific config.

        ``image_path`` may be omitted for in-memory use via
        :meth:`process_pil`; it is required only by :meth:`process_image`.
        """
        if (
            image_path is not None
            and image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
        ):
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
        if self.img_cfg.get("handle_transparency", True) and (
            image.mode in ("RGBA", "LA")
            or (image.mode == "P" and "transparency" in image.info)
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
    def _normalize_detail(detail: str) -> str:
        """Normalize a detail string to the set understood by resizing."""
        detail_norm = (detail or "high").lower()
        if detail_norm not in (
            "low",
            "high",
            "auto",
            "medium",
            "ultra_high",
            "original",
        ):
            detail_norm = "high"
        return detail_norm

    @staticmethod
    def compute_resize_scale(
        width_px: float,
        height_px: float,
        detail: str,
        img_cfg: dict,
        model_type: str = "openai",
    ) -> float:
        """Return the downscale ratio ``resize_for_detail`` would apply.

        Given the pixel dimensions of a render, report the linear scale factor
        (``<= 1.0``; the resize pipeline never upscales content it is fed) that
        the active resize profile would apply to that render. This is the sole
        knowledge the "direct" render strategy needs to rasterize a page at the
        final target size instead of rendering high and downscaling afterwards.

        For the white-padded box-fit profiles (OpenAI/Google high) the ratio is
        the fit-into-box scale clamped to ``1.0``: a page smaller than the box
        is left for :meth:`resize_for_detail` to upscale and pad, since the
        render itself must never exceed ``target_dpi``.
        """
        resize_profile = (img_cfg.get("resize_profile", "auto") or "auto").lower()
        if resize_profile == "none":
            return 1.0
        if width_px <= 0 or height_px <= 0:
            return 1.0
        detail_norm = ImageProcessor._normalize_detail(detail)
        longest = max(width_px, height_px)

        if detail_norm == "low":
            max_side = int(img_cfg.get("low_max_side_px", 512))
            return min(1.0, max_side / longest)

        if detail_norm == "original":
            max_side = int(img_cfg.get("original_max_side_px", 6000))
            max_pixels = int(img_cfg.get("original_max_pixels", 10240000))
            ratio = 1.0
            if longest > max_side:
                ratio = min(ratio, max_side / longest)
            pixels = width_px * height_px
            if pixels > max_pixels:
                ratio = min(ratio, (max_pixels / pixels) ** 0.5)
            return ratio

        if model_type == "anthropic":
            max_side = int(img_cfg.get("high_max_side_px", 1568))
            return min(1.0, max_side / longest)

        # OpenAI/Google high: fit into box (clamped; render never upscales)
        box = img_cfg.get("high_target_box", [768, 1536])
        try:
            box_w = int(box[0])
            box_h = int(box[1])
        except Exception:
            box_w, box_h = 768, 1536
        scale = min(box_w / width_px, box_h / height_px)
        return min(1.0, scale)

    @staticmethod
    def compute_direct_render_dpi(
        page_width_pt: float,
        page_height_pt: float,
        target_dpi: int,
        detail: str,
        img_cfg: dict,
        model_type: str = "openai",
    ) -> int:
        """Derive the DPI at which to rasterize a page under "direct" strategy.

        Rendering directly at this DPI produces a page whose pixel dimensions
        already match what the resize profile would downscale a full
        ``target_dpi`` render to, avoiding the wasted work of rendering high
        and shrinking. The DPI never exceeds ``target_dpi`` (no render
        upscaling; matches the no-upscale semantics of the resize pipeline), so
        when the resize profile would not shrink the page the render is left at
        ``target_dpi`` and the payload is byte-identical to the supersample
        path.
        """
        zoom = target_dpi / 72.0
        width_px = page_width_pt * zoom
        height_px = page_height_pt * zoom
        ratio = ImageProcessor.compute_resize_scale(
            width_px, height_px, detail, img_cfg, model_type
        )
        return max(1, round(target_dpi * ratio))

    @staticmethod
    def _cap_longest_side(image: Image.Image, max_side: int) -> Image.Image:
        """Downscale so the longest side is at most ``max_side`` (no upscaling)."""
        w, h = image.size
        longest = max(w, h)
        if longest <= max_side:
            return image
        scale = max_side / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    @staticmethod
    def resize_for_detail(
        image: Image.Image,
        detail: str,
        img_cfg: dict,
        model_type: str = "openai",
    ) -> Image.Image:
        """Resize strategy based on desired LLM detail and model type.

        - low: downscale longest side to low_max_side_px.
        - high/auto: fit/pad into high_target_box (OpenAI/Google) or
                     cap longest side to high_max_side_px (Anthropic).
        - original: cap to original_max_side_px and max_pixels (GPT-5.4+).
        """
        resize_profile = (img_cfg.get("resize_profile", "auto") or "auto").lower()
        if resize_profile == "none":
            return image
        detail_norm = ImageProcessor._normalize_detail(detail)

        # Low detail: cap longest side (same for all providers)
        if detail_norm == "low":
            max_side = int(img_cfg.get("low_max_side_px", 512))
            return ImageProcessor._cap_longest_side(image, max_side)

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
            return ImageProcessor._cap_longest_side(image, max_side)
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
            resized_img = image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
            if resized_img.mode not in ("RGB", "L"):
                resized_img = resized_img.convert("RGB")
            if resized_img.mode == "L":
                final_img = Image.new("L", (target_width, target_height), 255)
            else:
                final_img = Image.new(
                    "RGB", (target_width, target_height), (255, 255, 255)
                )
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            final_img.paste(resized_img, (paste_x, paste_y))
            return final_img

    def _effective_detail(self) -> str:
        """Resolve the configured detail level for the current model type."""
        if self.model_type == "google":
            return self.img_cfg.get("media_resolution", "high") or "high"
        if self.model_type == "anthropic":
            return self.img_cfg.get("resize_profile", "auto") or "auto"
        return self.img_cfg.get("llm_detail", "high") or "high"

    def _apply_transforms(self, img: Image.Image) -> Image.Image:
        """Run the full transform chain on an open PIL image."""
        detail = self._effective_detail()
        img = self.handle_transparency(img)
        img = self.convert_to_grayscale(img)
        img = ImageProcessor.resize_for_detail(
            img, detail, self.img_cfg, self.model_type
        )
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img

    def process_pil(self, img: Image.Image) -> bytes:
        """Process an in-memory PIL image and return encoded JPEG bytes.

        Applies the same transform chain as :meth:`process_image`
        (transparency flattening, optional grayscale, provider-specific
        resize, mode coercion) without any disk round-trip. Used by the
        streaming page pipeline (``modules.images.page_stream``).

        Args:
            img: Source PIL image.

        Returns:
            JPEG-encoded bytes of the processed image.
        """
        import io

        processed = self._apply_transforms(img)
        jpeg_quality = int(self.img_cfg.get("jpeg_quality", 95))
        buffer = io.BytesIO()
        processed.save(buffer, format="JPEG", quality=jpeg_quality)
        return buffer.getvalue()

    def process_image(self, output_path: Path) -> Path:
        """Process the image and save it to the given output path as JPEG.

        Returns:
            Path to the saved processed image (with .jpg extension).
        """
        if self.image_path is None:
            raise ValueError("process_image() requires an image_path")
        try:
            with Image.open(self.image_path) as _raw_img:
                img = self._apply_transforms(_raw_img)

                jpg_output_path = output_path.with_suffix(".jpg")
                jpeg_quality = int(self.img_cfg.get("jpeg_quality", 95))
                img.save(jpg_output_path, format="JPEG", quality=jpeg_quality)
                logger.debug(
                    "Saved processed image %s size=%s quality=%d",
                    jpg_output_path.name,
                    img.size,
                    jpeg_quality,
                )
            return jpg_output_path
        except Exception as e:
            logger.error("Error processing image %s: %s", self.image_path.name, e)
            raise
