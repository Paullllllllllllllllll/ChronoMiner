"""Portable native image geometry, sizing, encoding and provenance helpers."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image

logger = logging.getLogger(__name__)
TargetDpi = int | Literal["native"]


@dataclass(frozen=True)
class NativeDpi:
    dpi_x: float
    dpi_y: float
    source: str

    @property
    def dpi(self) -> float:
        return max(self.dpi_x, self.dpi_y)


def _clipped_area(points: list[tuple[float, float]], rect: Any) -> float:
    """Clip a placement quadrilateral against an unrotated page rectangle."""
    for axis, bound, sign in (
        (0, rect.x0, 1),
        (0, rect.x1, -1),
        (1, rect.y0, 1),
        (1, rect.y1, -1),
    ):
        output = []
        for start, end in zip(points, points[1:] + points[:1], strict=True):
            inside_start = sign * (start[axis] - bound) >= 0
            inside_end = sign * (end[axis] - bound) >= 0
            if inside_start != inside_end:
                ratio = (bound - start[axis]) / (end[axis] - start[axis])
                output.append(
                    tuple(start[i] + ratio * (end[i] - start[i]) for i in (0, 1))
                )
            if inside_end:
                output.append(end)
        points = output
    return (
        abs(
            sum(
                a[0] * b[1] - b[0] * a[1]
                for a, b in zip(points, points[1:] + points[:1], strict=True)
            )
        )
        / 2
    )


def native_page_dpi(page: Any, fallback_dpi: int) -> NativeDpi:
    """Find the densest scan covering at least half the visible page."""
    rect = page.rect * page.derotation_matrix
    candidates = []
    for info in page.get_image_info(xrefs=True):
        a, b, c, d, e, f = info["transform"]
        x_length, y_length = math.hypot(a, b), math.hypot(c, d)
        if not x_length or not y_length:
            continue
        points = [(e, f), (a + e, b + f), (a + c + e, b + d + f), (c + e, d + f)]
        if _clipped_area(points, rect) + 1e-6 < rect.get_area() / 2:
            continue
        candidates.append(
            NativeDpi(
                72 * info["width"] / x_length,
                72 * info["height"] / y_length,
                "image",
            )
        )
    result = max(candidates, key=lambda value: value.dpi, default=None)
    if result is None or result.dpi < 72:
        return NativeDpi(float(fallback_dpi), float(fallback_dpi), "fallback")
    return result


@dataclass(frozen=True)
class ImageCap:
    max_side: int
    patches: int
    patch_px: int
    policy: str


def model_image_cap(
    model_type: str,
    model_name: str,
    detail: str,
    flags: dict[str, Any] | None = None,
) -> ImageCap | None:
    """Resolve a cap from caller-supplied registry flags, never model guesses."""
    flags = flags or {}
    if model_type in ("openai", "openrouter") and detail == "original":
        if flags.get("image_original_patch_cap_30k") and model_type == "openai":
            return ImageCap(65535, 30000, 32, "openai-original-30k-v1")
        return ImageCap(6000, 10000, 32, "openai-original-10k-v1")
    if model_type == "anthropic" and detail != "low":
        if flags.get("image_high_res_tier"):
            return ImageCap(2576, 4784, 28, "anthropic-high-v1")
        return ImageCap(1568, 1568, 28, "anthropic-standard-v1")
    return None


def resized_size(
    width: int,
    height: int,
    max_side: int,
    patch_budget: int,
    patch_px: int = 28,
    max_pixels: int = 0,
) -> tuple[int, int]:
    """Largest aspect-preserving integer size within edge and patch limits."""
    longest = max(width, height)

    def size(edge: int) -> tuple[int, int]:
        return max(1, width * edge // longest), max(1, height * edge // longest)

    low, high = 1, min(longest, max_side)
    while low < high:
        middle = (low + high + 1) // 2
        w, h = size(middle)
        fits = math.ceil(w / patch_px) * math.ceil(h / patch_px) <= patch_budget
        if fits and (not max_pixels or w * h <= max_pixels):
            low = middle
        else:
            high = middle - 1
    return size(low)


def resolve_target_size(
    src_w: int,
    src_h: int,
    model_type: str,
    model_name: str,
    detail: str,
    img_cfg: dict[str, Any],
) -> tuple[tuple[int, int], str]:
    """Resolve content dimensions; box padding is applied by the caller."""
    cap = model_image_cap(model_type, model_name, detail, img_cfg)
    if cap:
        side_key = (
            "high_max_side_px"
            if model_type == "anthropic"
            else ("original_max_side_px")
        )
        side = min(cap.max_side, int(img_cfg.get(side_key, cap.max_side)))
        pixels = (
            int(img_cfg.get("original_max_pixels", 0))
            if (model_type != "anthropic")
            else 0
        )
        size = resized_size(src_w, src_h, side, cap.patches, cap.patch_px, pixels)
        reason = "model_cap"
    elif img_cfg.get("resize_profile") == "none":
        return (src_w, src_h), "none"
    else:
        if detail == "low":
            scale = min(
                1.0, int(img_cfg.get("low_max_side_px", 512)) / max(src_w, src_h)
            )
        else:
            box = img_cfg.get("high_target_box", [768, 1536])
            scale = min(1.0, box[0] / src_w, box[1] / src_h)
        size = max(1, int(src_w * scale)), max(1, int(src_h * scale))
        reason = "profile"
    return size, reason if size != (src_w, src_h) else "none"


def validate_image_settings(cfg: dict[str, Any], section: str) -> None:
    """Reject invalid new settings with a provider-qualified diagnostic."""
    for key, default, minimum in (
        ("target_dpi", 300, 1),
        ("native_fallback_dpi", 300, 1),
        ("max_image_bytes", 0, 0),
    ):
        value = cfg.get(key, default)
        if (
            key == "target_dpi"
            and value == "native"
            and section != ("tesseract_image_processing")
        ):
            continue
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {key} in {section}: {value!r}")
    if cfg.get("payload_format", "jpeg") not in ("jpeg", "png"):
        raise ValueError(f"Invalid payload_format in {section}")


def encode_payload(img: Image.Image, fmt: str, jpeg_quality: int) -> tuple[bytes, str]:
    """Encode JPEG or lossless L/RGB PNG."""
    if fmt not in ("jpeg", "png"):
        raise ValueError(f"Invalid payload_format: {fmt!r}")
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    buffer = io.BytesIO()
    if fmt == "png":
        img.save(buffer, format="PNG", compress_level=6)
    else:
        img.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue(), f"image/{fmt}"


def guarded_payload(
    img: Image.Image,
    fmt: str,
    jpeg_quality: int,
    max_image_bytes: int,
    *,
    log: logging.Logger | None = None,
) -> tuple[bytes, str, bool]:
    """Enforce a base64-byte limit, falling back from PNG to JPEG once."""
    data, mime = encode_payload(img, fmt, jpeg_quality)
    fallback = False
    if max_image_bytes and 4 * ((len(data) + 2) // 3) > max_image_bytes:
        if fmt == "png":
            (log or logger).warning("PNG exceeds max_image_bytes; falling back to JPEG")
            data, mime = encode_payload(img, "jpeg", jpeg_quality)
            fallback = True
        if 4 * ((len(data) + 2) // 3) > max_image_bytes:
            raise ValueError("Payload exceeds max_image_bytes after encoding")
    return data, mime, fallback


def image_settings_fingerprint(settings: dict[str, Any]) -> str:
    canonical = json.dumps(
        settings,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def format_downscale_log(page: int, provenance: dict[str, Any], model: str) -> str:
    """One compact per-page explanation, including the active model policy."""
    source_pixels = provenance["source_width"] * provenance["source_height"]
    sent_pixels = provenance["width"] * provenance["height"]
    dpi = provenance["source_dpi_x"]
    sent_dpi = provenance["sent_dpi"]
    source_density = f"{dpi:.1f}" if dpi is not None else "unknown"
    sent_density = f"{sent_dpi:.1f}" if sent_dpi is not None else "unknown"
    policy = provenance.get("cap_policy", "profile")
    return (
        f"page {page}: {provenance['dpi_source']} {source_density} dpi "
        f"{provenance['source_width']}x{provenance['source_height']} "
        f"({source_pixels / 1e6:.1f} MP) -> sent {sent_density} dpi "
        f"{provenance['width']}x{provenance['height']} "
        f"({sent_pixels / 1e6:.1f} MP), "
        f"reason {provenance['downscale_reason']} ({model}, {policy})"
    )
