"""Provider-specific image content block construction for multimodal LLM messages.

Encapsulates the different formats required by OpenAI, Anthropic, and Google
for embedding images in LangChain ``HumanMessage`` content lists. Same-package
import of :func:`create_data_url` eliminates the former llm -> processing
dependency inversion.
"""

from __future__ import annotations

from typing import Any

from modules.images.encoding import create_data_url

# Gemini per-part media resolution. The configured ``media_resolution`` value
# (google_image_processing section) is a short token; google-genai's
# ``PartMediaResolutionLevel`` enum only accepts the fully qualified names, and
# an unrecognized value is passed through with a warning as an invalid enum
# member. Anything outside this map keeps the legacy block (no media
# resolution) rather than emitting a value the API would reject.
_GOOGLE_MEDIA_RESOLUTIONS: dict[str, str] = {
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


def _normalize_google_media_resolution(detail: str | None) -> str | None:
    """Map a configured media-resolution token to the google-genai enum name."""
    if not detail:
        return None
    return _GOOGLE_MEDIA_RESOLUTIONS.get(detail.strip().lower())


def build_image_content_block(
    image_base64: str,
    mime_type: str,
    provider: str,
    detail: str | None = None,
    supports_image_detail: bool = False,
    supports_original_detail: bool = False,
) -> dict[str, Any]:
    """Build a provider-specific image content block for LangChain HumanMessage.

    Args:
        image_base64: Base64-encoded image data.
        mime_type: MIME type of the image (e.g., 'image/jpeg').
        provider: LLM provider ('openai', 'anthropic', 'google', 'openrouter').
        detail: Image detail level. For OpenAI/OpenRouter one of
            'low', 'high', 'auto', 'original'; for Google the configured
            ``media_resolution`` ('low', 'medium', 'high', 'ultra_high').
        supports_image_detail: Whether the model supports the detail parameter.
        supports_original_detail: Whether the request is routed to an API that
            accepts ``detail: "original"`` (OpenAI Responses only). When False,
            a configured 'original' is downgraded to 'high', which is the
            highest fidelity the Chat Completions image param allows.

    Returns:
        Dict suitable for inclusion in a LangChain HumanMessage content list.
    """
    provider_lower = provider.lower()

    if provider_lower == "anthropic":
        # Anthropic uses direct base64 embedding (no data URL)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": image_base64,
            },
        }

    elif provider_lower == "google":
        media_resolution = _normalize_google_media_resolution(detail)
        if media_resolution is not None:
            # LangChain v1 data block: the only shape on which
            # langchain-google-genai forwards a per-part media resolution
            # (honored on Gemini 3+, ignored with a warning on 2.5 and older).
            # The block itself is accepted by every Gemini generation.
            return {
                "type": "image",
                "base64": image_base64,
                "mime_type": mime_type,
                "media_resolution": media_resolution,
            }
        # No usable media resolution configured: legacy data-URL block.
        data_url = create_data_url(image_base64, mime_type)
        return {
            "type": "image_url",
            "image_url": data_url,
        }

    else:
        # OpenAI / OpenRouter: image_url with optional detail
        data_url = create_data_url(image_base64, mime_type)
        image_url_obj: dict[str, Any] = {"url": data_url}
        if detail and supports_image_detail:
            effective_detail = detail
            if effective_detail.strip().lower() == "original" and not (
                supports_original_detail
            ):
                # Chat Completions rejects "original" (auto/low/high only).
                effective_detail = "high"
            image_url_obj["detail"] = effective_detail
        return {
            "type": "image_url",
            "image_url": image_url_obj,
        }
