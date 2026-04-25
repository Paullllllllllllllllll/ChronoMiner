"""Provider-specific image content block construction for multimodal LLM messages.

Encapsulates the different formats required by OpenAI, Anthropic, and Google
for embedding images in LangChain ``HumanMessage`` content lists. Same-package
import of :func:`create_data_url` eliminates the former llm -> processing
dependency inversion.
"""

from __future__ import annotations

from typing import Any

from modules.images.encoding import create_data_url


def build_image_content_block(
    image_base64: str,
    mime_type: str,
    provider: str,
    detail: str | None = None,
    supports_image_detail: bool = False,
) -> dict[str, Any]:
    """Build a provider-specific image content block for LangChain HumanMessage.

    Args:
        image_base64: Base64-encoded image data.
        mime_type: MIME type of the image (e.g., 'image/jpeg').
        provider: LLM provider ('openai', 'anthropic', 'google', 'openrouter').
        detail: Image detail level ('low', 'high', 'auto', 'original').
        supports_image_detail: Whether the model supports the detail parameter.

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
        # Google uses data URL in image_url (no detail parameter)
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
            image_url_obj["detail"] = detail
        return {
            "type": "image_url",
            "image_url": image_url_obj,
        }
