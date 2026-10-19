"""Page image payloads for vision requests.

Page sources render PDF pages or load folder images lazily, preprocess them in
memory (16-bit reduction, transparency, grayscale, resizing to the model cap
or the resize profile) and encode them as base64 payloads with provenance.

Callers resolve the image settings (``img_cfg``) and the model type
(``openai``, ``google``, ``anthropic`` or ``custom``). Keys read from
``img_cfg``: ``target_dpi``, ``native_fallback_dpi``, ``render_strategy``,
``max_pixels_per_page``, ``payload_format``, ``max_image_bytes``,
``jpeg_quality``, ``grayscale_conversion``, ``handle_transparency``,
``resize_profile``, ``low_max_side_px``, ``high_target_box``,
``resampling_algorithm``, the detail keys (``resolved_detail``, else
``media_resolution``, ``resize_profile`` or ``llm_detail`` by model type),
``model_name``, ``request_detail``, ``cap_policy`` and the cap flags read by
``native.model_image_cap``. The image-settings fingerprint covers the whole
dict.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import math
import re
import threading
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import fitz
import PIL
from PIL import Image, ImageOps

from .native import (
    TargetDpi,
    format_downscale_log,
    guarded_payload,
    image_settings_fingerprint,
    model_image_cap,
    native_page_dpi,
    resolve_target_size,
)

logger = logging.getLogger(__name__)

POINTS_PER_INCH = 72.0
DEFAULT_LOW_MAX_SIDE_PX = 512
DEFAULT_HIGH_TARGET_BOX = (768, 1536)
DEFAULT_JPEG_QUALITY = 95
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
)
PDF_PAGE_NAME = "page_{number:04d}.jpg"
FAILURE_RATE_THRESHOLD = 0.5

DETAIL_LEVELS = ("low", "high", "auto", "medium", "ultra_high", "original")

_WHITE = (255, 255, 255)
_HASH_CHUNK_SIZE = 1024 * 1024

# Pillow's grayscale conversion clips these modes instead of scaling them.
_SIXTEEN_BIT_MODES = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N"})

_RESAMPLING_FILTERS = {
    "bilinear": Image.Resampling.BILINEAR,
    "lanczos": Image.Resampling.LANCZOS,
}

# Gemini per-part media resolution: configured token to google-genai enum name.
GOOGLE_MEDIA_RESOLUTIONS: Mapping[str, str] = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
    "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH",
    "unspecified": "MEDIA_RESOLUTION_UNSPECIFIED",
    "media_resolution_low": "MEDIA_RESOLUTION_LOW",
    "media_resolution_medium": "MEDIA_RESOLUTION_MEDIUM",
    "media_resolution_high": "MEDIA_RESOLUTION_HIGH",
    "media_resolution_ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH",
    "media_resolution_unspecified": "MEDIA_RESOLUTION_UNSPECIFIED",
}


@dataclass(frozen=True)
class PagePayload:
    """One page, preprocessed and base64-encoded, ready for a vision request.

    ``index`` is the absolute 0-based page or source index. ``number`` is the
    1-based page number taken from the image file name; for PDF pages it is
    ``index + 1``.
    """

    base64: str
    image_name: str
    number: int
    index: int
    provenance: dict[str, Any]
    source_file: str
    page_index: int | None = None
    mime_type: str = "image/jpeg"


@dataclass(frozen=True)
class PageError:
    """A page that could not be rendered, preprocessed or encoded."""

    index: int
    image_name: str
    error: str


@dataclass(frozen=True)
class _PageIdentity:
    index: int
    image_name: str
    number: int
    source_file: str
    page_index: int | None


def google_media_resolution(detail: str | None) -> str | None:
    """Return the google-genai media-resolution name for *detail*, or None."""
    if not detail:
        return None
    return GOOGLE_MEDIA_RESOLUTIONS.get(detail.strip().lower())


def natural_sort_key(name: str) -> tuple[str | int, ...]:
    """Sort key comparing digit runs numerically (page_2 before page_10)."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    )


def list_folder_images(
    folder: Path, extensions: Iterable[str] = IMAGE_EXTENSIONS
) -> list[Path]:
    """Return the images in *folder* in natural filename order."""
    suffixes = frozenset(extensions)
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes),
        key=lambda p: natural_sort_key(p.name),
    )


def sequence_number(image_path: Path) -> int:
    """Page number in an image filename: the trailing ``_N``, else the last
    number in the stem, else 0."""
    last = image_path.stem.split("_")[-1]
    if last.isdecimal():
        return int(last)
    numbers = re.findall(r"\d+", image_path.stem)
    return int(numbers[-1]) if numbers else 0


def sha256_of_file(path: Path) -> str:
    """Hash a file in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def resampling_filter(img_cfg: Mapping[str, Any] | None = None) -> Image.Resampling:
    """Return the configured downscaling filter; bilinear by default."""
    name = str((img_cfg or {}).get("resampling_algorithm", "bilinear")).lower()
    return _RESAMPLING_FILTERS.get(name, Image.Resampling.BILINEAR)


def resolve_detail(img_cfg: Mapping[str, Any], model_type: str) -> str:
    """Return the effective local detail of *img_cfg* for *model_type*."""
    if "resolved_detail" in img_cfg:
        return str(img_cfg["resolved_detail"])
    if model_type == "google":
        return str(img_cfg.get("media_resolution", "high") or "high")
    if model_type == "anthropic":
        return str(img_cfg.get("resize_profile", "auto") or "auto")
    return str(img_cfg.get("llm_detail", "high") or "high")


def _normalized_detail(detail: str | None) -> str:
    value = (detail or "high").lower()
    return value if value in DETAIL_LEVELS else "high"


def _target_box(img_cfg: Mapping[str, Any]) -> tuple[int, int]:
    box = img_cfg.get("high_target_box", DEFAULT_HIGH_TARGET_BOX)
    try:
        return int(box[0]), int(box[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return DEFAULT_HIGH_TARGET_BOX


def _resize_max_side(
    image: Image.Image, max_side: int, img_cfg: Mapping[str, Any]
) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, resampling_filter(img_cfg))


def _resize_box_fit(image: Image.Image, img_cfg: Mapping[str, Any]) -> Image.Image:
    """Fit *image* into the target box and pad it with white."""
    box_width, box_height = _target_box(img_cfg)
    scale = min(box_width / image.width, box_height / image.height)
    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
    resized = image.resize((width, height), resampling_filter(img_cfg))
    if image.mode == "L":
        canvas = Image.new("L", (box_width, box_height), 255)
    else:
        canvas = Image.new("RGB", (box_width, box_height), _WHITE)
    canvas.paste(resized, ((box_width - width) // 2, (box_height - height) // 2))
    return canvas


def resize_for_detail(
    image: Image.Image,
    detail: str | None,
    img_cfg: Mapping[str, Any],
    model_type: str = "openai",
) -> Image.Image:
    """Resize *image* to the model cap or, without a cap, the resize profile.

    Without a cap, ``low`` caps the longest side at ``low_max_side_px``,
    ``original`` keeps the rendered size, and every other detail fits the
    image into ``high_target_box`` with padding.
    """
    detail = _normalized_detail(detail)
    model_name = img_cfg.get("model_name", "")
    if model_image_cap(model_type, model_name, detail, dict(img_cfg)):
        target = img_cfg.get("_target_size")
        if target is None:
            target, _ = resolve_target_size(
                *image.size, model_type, model_name, detail, dict(img_cfg)
            )
        # Real pixmap rounding can land just beyond a patch boundary.
        size = min(image.width, target[0]), min(image.height, target[1])
        if size == image.size:
            return image
        return image.resize(size, resampling_filter(img_cfg))
    if (img_cfg.get("resize_profile", "auto") or "auto").lower() == "none":
        return image
    if detail == "low":
        max_side = int(img_cfg.get("low_max_side_px", DEFAULT_LOW_MAX_SIDE_PX))
        return _resize_max_side(image, max_side, img_cfg)
    if detail == "original":
        return image
    return _resize_box_fit(image, img_cfg)


def content_scale_factor(
    src_size: tuple[float, float],
    img_cfg: Mapping[str, Any],
    model_type: str = "openai",
) -> float:
    """Scale ``resize_for_detail`` would apply to a source of *src_size* pixels.

    Box padding is ignored. ``original`` without a model cap returns 1.0.
    The box-fit profile can return more than 1.0; callers that must not
    upscale clamp the result.
    """
    width, height = src_size
    if width <= 0 or height <= 0:
        return 1.0
    detail = _normalized_detail(resolve_detail(img_cfg, model_type))
    model_name = img_cfg.get("model_name", "")
    if model_image_cap(model_type, model_name, detail, dict(img_cfg)):
        int_width = math.ceil(width - 1e-7)
        int_height = math.ceil(height - 1e-7)
        size = img_cfg.get("_target_size")
        if size is None:
            size, _ = resolve_target_size(
                int_width, int_height, model_type, model_name, detail, dict(img_cfg)
            )
        return float(min(size[0] / int_width, size[1] / int_height))
    if (img_cfg.get("resize_profile", "auto") or "auto").lower() == "none":
        return 1.0
    if detail == "low":
        max_side = int(img_cfg.get("low_max_side_px", DEFAULT_LOW_MAX_SIDE_PX))
        longest = max(width, height)
        return 1.0 if longest <= max_side else max_side / float(longest)
    if detail == "original":
        return 1.0
    box_width, box_height = _target_box(img_cfg)
    return min(box_width / width, box_height / height)


def preprocess_image(
    image: Image.Image, img_cfg: Mapping[str, Any], model_type: str = "openai"
) -> Image.Image:
    """Reduce 16-bit depth, flatten transparency, convert to grayscale, resize."""
    if image.mode in _SIXTEEN_BIT_MODES:
        # point() accepts only the endian-neutral "I" mode.
        if image.mode != "I":
            image = image.convert("I")
        image = image.point(lambda v: v * (1 / 256)).convert("L")
    if img_cfg.get("handle_transparency", True) and (
        image.mode in ("RGBA", "LA")
        or (image.mode == "P" and "transparency" in image.info)
    ):
        if image.mode == "P":
            image = image.convert("RGBA")
        background = Image.new("RGB", image.size, _WHITE)
        mask = image.split()[-1] if image.mode in ("RGBA", "LA") else None
        background.paste(image, mask=mask)
        image = background
    if img_cfg.get("grayscale_conversion", True) and image.mode != "L":
        image = ImageOps.grayscale(image)
    detail = resolve_detail(img_cfg, model_type)
    return resize_for_detail(image, detail, img_cfg, model_type)


class PageSource:
    """Settings, payload encoding and provenance shared by the page sources.

    *section* names the settings section recorded in the provenance. With
    *payload_sha256_key*, each page's provenance also records the SHA-256 of
    the encoded bytes under that key.
    """

    _native_profile_warned: ClassVar[bool] = False

    def __init__(
        self,
        source_path: Path,
        img_cfg: dict[str, Any],
        model_type: str,
        *,
        section: str = "",
        payload_sha256_key: str | None = None,
    ) -> None:
        self.source_path = source_path
        self.img_cfg = img_cfg
        self.model_type = model_type
        self.section = section
        self.payload_sha256_key = payload_sha256_key
        self.jpeg_quality = int(img_cfg.get("jpeg_quality", DEFAULT_JPEG_QUALITY))
        self.render_strategy = str(img_cfg.get("render_strategy", "direct"))
        self.target_dpi: TargetDpi = img_cfg.get("target_dpi", 300)
        self.max_pixels = int(img_cfg.get("max_pixels_per_page", 0))
        if (
            self.target_dpi == "native"
            and img_cfg.get("cap_policy", "profile-v1") == "profile-v1"
            and not PageSource._native_profile_warned
        ):
            logger.warning(
                "Native resolution is bounded by a resize profile; "
                "the profile discards native resolution."
            )
            PageSource._native_profile_warned = True

    def __len__(self) -> int:
        raise NotImplementedError

    def image_name(self, index: int) -> str:
        """Stable name of the page at *index* (0-based)."""
        raise NotImplementedError

    def build_payload(self, index: int) -> PagePayload:
        """Render or load, preprocess and encode the page at *index*."""
        raise NotImplementedError

    def settings_provenance(self) -> dict[str, Any]:
        """Source, library versions, settings and their fingerprint."""
        return {
            "source_file": str(self.source_path),
            "pymupdf_version": getattr(fitz, "pymupdf_version", None)
            or getattr(fitz, "VersionBind", "unknown"),
            "pillow_version": PIL.__version__,
            "image_config_section": self.section,
            "image_config": dict(self.img_cfg),
            "model_type": self.model_type,
            "model_name": self.img_cfg.get("model_name", ""),
            "detail": self.img_cfg.get("request_detail"),
            "cap_policy": self.img_cfg.get("cap_policy", "profile-v1"),
            "image_settings_fingerprint": image_settings_fingerprint(self.img_cfg),
        }

    def _encode(
        self,
        image: Image.Image,
        page: _PageIdentity,
        provenance: dict[str, Any],
        img_cfg: dict[str, Any],
        *,
        effective_dpi: float | None = None,
    ) -> PagePayload:
        """Preprocess and encode *image* and record what is sent."""
        detail = resolve_detail(img_cfg, self.model_type)
        model_name = img_cfg.get("model_name", "")
        if "downscale_reason" not in provenance:
            size, provenance["downscale_reason"] = resolve_target_size(
                image.width, image.height, self.model_type, model_name, detail, img_cfg
            )
            img_cfg = {**img_cfg, "_target_size": size}
        processed = preprocess_image(image, img_cfg, self.model_type)
        data, mime, fallback = guarded_payload(
            processed,
            img_cfg.get("payload_format", "jpeg"),
            self.jpeg_quality,
            int(img_cfg.get("max_image_bytes", 0)),
            log=logger,
        )
        padded = (
            model_image_cap(self.model_type, model_name, detail, img_cfg) is None
            and self.model_type != "anthropic"
            and detail not in ("low", "original")
            and img_cfg.get("resize_profile") != "none"
        )
        source_dpi = provenance["source_dpi_x"]
        provenance.update(
            width=processed.width,
            height=processed.height,
            byte_size=len(data),
            sent_dpi=(
                None
                if padded or source_dpi is None
                else max(source_dpi, provenance["source_dpi_y"])
                * processed.width
                / provenance["source_width"]
            ),
            cap_policy=img_cfg.get("cap_policy", "profile-v1"),
            payload_format=mime.split("/")[1],
            mime_type=mime,
            format_fallback=fallback,
        )
        if self.payload_sha256_key:
            provenance[self.payload_sha256_key] = hashlib.sha256(data).hexdigest()
        if effective_dpi is not None:
            provenance["effective_dpi"] = effective_dpi
        if provenance["downscale_reason"] != "none":
            logger.info(format_downscale_log(page.index + 1, provenance, model_name))
        return PagePayload(
            base64=base64.b64encode(data).decode("utf-8"),
            image_name=page.image_name,
            number=page.number,
            index=page.index,
            provenance=provenance,
            source_file=page.source_file,
            page_index=page.page_index,
            mime_type=mime,
        )

    def close(self) -> None:
        """Release the resources held by the source."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PdfPageSource(PageSource):
    """Render PDF pages lazily from one open document.

    PyMuPDF documents are not thread-safe, so all document access runs under
    a lock; preprocessing and encoding run outside it. Page names follow
    ``name_template`` with the 1-based page ``number``.
    """

    name_template: ClassVar[str] = PDF_PAGE_NAME

    def __init__(
        self,
        pdf_path: Path,
        img_cfg: dict[str, Any],
        model_type: str,
        *,
        section: str = "",
        payload_sha256_key: str | None = None,
    ) -> None:
        super().__init__(
            pdf_path,
            img_cfg,
            model_type,
            section=section,
            payload_sha256_key=payload_sha256_key,
        )
        self._render_lock = threading.Lock()
        self._doc: fitz.Document | None = fitz.open(pdf_path)
        # An encrypted PDF reports a page count but fails every render.
        if self._doc.needs_pass:
            self._doc.close()
            self._doc = None
            raise ValueError(f"Cannot open password-protected PDF: {pdf_path.name}")
        logger.debug(
            f"Opened {pdf_path.name}: {len(self)} page(s), "
            f"model_type={self.model_type}, dpi={self.target_dpi}"
        )

    def __len__(self) -> int:
        return len(self._doc) if self._doc is not None else 0

    @classmethod
    def image_name(cls, index: int) -> str:
        """Stable name of the page at *index* (0-based)."""
        return cls.name_template.format(number=index + 1)

    def _render_zoom(self, page: Any, img_cfg: dict[str, Any]) -> float:
        """Zoom for a numeric target DPI under the render strategy.

        ``supersample`` renders at ``target_dpi``; ``direct`` renders close to
        the final content size, never above ``target_dpi``. The memory guard
        caps the integer pixmap MuPDF allocates.
        """
        if self.target_dpi == "native":
            raise ValueError("Native zoom is resolved per page inside the render lock")
        base_zoom = self.target_dpi / POINTS_PER_INCH
        rect = page.rect
        pt_width, pt_height = float(rect.width), float(rect.height)
        zoom = base_zoom
        if self.render_strategy == "direct" and pt_width > 0 and pt_height > 0:
            scale = content_scale_factor(
                (pt_width * base_zoom, pt_height * base_zoom), img_cfg, self.model_type
            )
            zoom *= min(scale, 1.0)
        pixels = (pt_width * zoom) * (pt_height * zoom)
        if self.max_pixels > 0 and pixels > self.max_pixels:
            zoom *= math.sqrt(self.max_pixels / pixels)
            while True:
                box = (rect * fitz.Matrix(zoom, zoom)).irect
                area = box.width * box.height
                if area <= self.max_pixels:
                    break
                zoom *= math.sqrt(self.max_pixels / area) * 0.9999
        return float(zoom)

    def build_payload(self, index: int) -> PagePayload:
        """Render, preprocess and encode page *index* (0-based)."""
        if self._doc is None:
            raise RuntimeError("PDF page source is closed")
        cfg = self.img_cfg
        grayscale = bool(cfg.get("grayscale_conversion", True))
        with self._render_lock:
            if self._doc is None:
                raise RuntimeError("PDF page source is closed")
            page = self._doc[index]
            density = (
                native_page_dpi(page, int(cfg.get("native_fallback_dpi", 300)))
                if self.target_dpi == "native"
                else None
            )
            source_dpi = density.dpi if density else float(self.target_dpi)
            source_width = math.ceil(page.rect.width * source_dpi / 72 - 1e-7)
            source_height = math.ceil(page.rect.height * source_dpi / 72 - 1e-7)
            size, reason = resolve_target_size(
                source_width,
                source_height,
                self.model_type,
                cfg.get("model_name", ""),
                resolve_detail(cfg, self.model_type),
                cfg,
            )
            img_cfg = {**cfg, "_target_size": size}
            if density:
                zoom = (
                    source_dpi
                    / 72
                    * min(size[0] / source_width, size[1] / source_height)
                )
            else:
                zoom = self._render_zoom(page, img_cfg)
                if self.max_pixels > 0 and self.max_pixels < size[0] * size[1]:
                    reason = "memory_guard"
            provenance = {
                "source_dpi_x": density.dpi_x if density else source_dpi,
                "source_dpi_y": density.dpi_y if density else source_dpi,
                "dpi_source": density.source if density else "numeric",
                "source_width": source_width,
                "source_height": source_height,
                "render_dpi": zoom * 72,
                "downscale_reason": reason,
            }
            matrix = fitz.Matrix(zoom, zoom)
            if grayscale:
                pix = page.get_pixmap(
                    matrix=matrix, alpha=False, colorspace=fitz.csGRAY
                )
                image = Image.frombytes("L", (pix.width, pix.height), pix.samples_mv)
            else:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples_mv)
            del pix, page
        return self._encode(
            image,
            _PageIdentity(
                index, self.image_name(index), index + 1, str(self.source_path), index
            ),
            provenance,
            img_cfg,
            effective_dpi=round(zoom * POINTS_PER_INCH, 2),
        )

    def file_provenance(self) -> dict[str, Any]:
        """Settings provenance plus the file hash, size and target DPI."""
        provenance = self.settings_provenance()
        try:
            provenance["source_sha256"] = sha256_of_file(self.source_path)
        except OSError as exc:
            logger.warning(f"Could not hash {self.source_path.name}: {exc}")
            provenance["source_sha256"] = None
        # Size only: mtime changes on copies of an identical file.
        try:
            provenance["size"] = self.source_path.stat().st_size
        except OSError:
            provenance["size"] = None
        provenance["target_dpi"] = self.target_dpi
        return provenance

    def close(self) -> None:
        # The lock keeps the document open until an in-flight render ends.
        with self._render_lock:
            if self._doc is not None:
                self._doc.close()
                self._doc = None


class FolderPageSource(PageSource):
    """Load images from a folder lazily, in natural filename order."""

    def __init__(
        self,
        folder_path: Path,
        img_cfg: dict[str, Any],
        model_type: str,
        *,
        image_paths: list[Path] | None = None,
        section: str = "",
        payload_sha256_key: str | None = None,
    ) -> None:
        super().__init__(
            folder_path,
            img_cfg,
            model_type,
            section=section,
            payload_sha256_key=payload_sha256_key,
        )
        self.image_paths = (
            list_folder_images(folder_path) if image_paths is None else image_paths
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def image_name(self, index: int) -> str:
        """Filename of the image at *index* (0-based)."""
        return self.image_paths[index].name

    def build_payload(self, index: int) -> PagePayload:
        """Load, preprocess and encode the image at *index* (0-based)."""
        image_path = self.image_paths[index]
        source_bytes = image_path.read_bytes()
        with Image.open(io.BytesIO(source_bytes)) as image_file:
            image_file.load()
            # exif_transpose returns an independent copy, or None.
            transposed = ImageOps.exif_transpose(image_file)
            oriented = transposed if transposed is not None else image_file
            provenance = {
                "source_dpi_x": None,
                "source_dpi_y": None,
                "dpi_source": "file",
                "source_width": oriented.width,
                "source_height": oriented.height,
                "render_dpi": None,
                "file_dpi_metadata": image_file.info.get("dpi"),
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
            # Encode inside the block: preprocessing can return its input.
            return self._encode(
                oriented,
                _PageIdentity(
                    index,
                    image_path.name,
                    sequence_number(image_path),
                    str(image_path),
                    None,
                ),
                provenance,
                self.img_cfg,
            )

    def file_provenance(self) -> dict[str, Any]:
        """Settings provenance plus the identity of the image set.

        The image count, a hash of the sorted names (which fixes the page
        order) and the combined byte size, None when a file cannot be read.
        """
        provenance = self.settings_provenance()
        provenance["image_count"] = len(self.image_paths)
        provenance["image_names_sha256"] = hashlib.sha256(
            "\n".join(sorted(p.name for p in self.image_paths)).encode("utf-8")
        ).hexdigest()
        total: int | None = 0
        for path in self.image_paths:
            try:
                total = (total or 0) + path.stat().st_size
            except OSError:
                total = None
                break
        provenance["total_image_bytes"] = total
        return provenance


def check_failure_rate(
    source_name: str,
    total: int,
    failed: int,
    threshold: float = FAILURE_RATE_THRESHOLD,
) -> None:
    """Raise when at least two pages and *threshold* of *total* pages failed."""
    if total > 1 and failed >= 2 and failed / total >= threshold:
        raise RuntimeError(
            f"Page preprocessing failed for {failed}/{total} pages in "
            f"'{source_name}'. Check the source for corruption or unsupported "
            "formats."
        )
    if failed:
        logger.warning(
            f"{failed} of {total} page(s) in '{source_name}' failed to render "
            "or preprocess."
        )


async def stream_payloads(
    source: PageSource,
    indices: Iterable[int] | None = None,
    *,
    threshold: float = FAILURE_RATE_THRESHOLD,
) -> AsyncIterator[PagePayload | PageError]:
    """Yield the payloads of *indices* (0-based; all pages by default) in order.

    Pages build one at a time in a worker thread. A failed page yields a
    ``PageError``; after the last page, ``check_failure_rate`` runs.
    """
    total = len(source)
    wanted = range(total) if indices is None else indices
    needed = [i for i in wanted if 0 <= i < total]
    failed = 0
    for index in needed:
        result: PagePayload | PageError
        try:
            result = await asyncio.to_thread(source.build_payload, index)
        except Exception as exc:  # noqa: BLE001 - a failed page is reported
            failed += 1
            logger.error(f"Error preparing page {index + 1} of {source.source_path}")
            result = PageError(index, source.image_name(index), str(exc))
        yield result
    check_failure_rate(source.source_path.name, len(needed), failed, threshold)


__all__ = [
    "DEFAULT_HIGH_TARGET_BOX",
    "DEFAULT_JPEG_QUALITY",
    "DEFAULT_LOW_MAX_SIDE_PX",
    "DETAIL_LEVELS",
    "FAILURE_RATE_THRESHOLD",
    "GOOGLE_MEDIA_RESOLUTIONS",
    "IMAGE_EXTENSIONS",
    "PDF_PAGE_NAME",
    "POINTS_PER_INCH",
    "FolderPageSource",
    "PageError",
    "PagePayload",
    "PageSource",
    "PdfPageSource",
    "check_failure_rate",
    "content_scale_factor",
    "google_media_resolution",
    "list_folder_images",
    "natural_sort_key",
    "preprocess_image",
    "resampling_filter",
    "resize_for_detail",
    "resolve_detail",
    "sequence_number",
    "sha256_of_file",
    "stream_payloads",
]
