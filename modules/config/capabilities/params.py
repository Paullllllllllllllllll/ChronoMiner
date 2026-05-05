"""LangChain disabled_params computation from capability flags.

LangChain does NOT automatically filter unsupported parameters for different
models (e.g., it will pass temperature to o1/o3 reasoning models that reject
it). This module converts a :class:`Capabilities` instance into the
``disabled_params`` dict that LangChain's chat models accept to suppress
those parameters before the API call is issued.
"""

from __future__ import annotations

from typing import Any

from modules.config.capabilities.detection import detect_capabilities
from modules.config.capabilities.registry import Capabilities, ProviderType


def disabled_params_for_capabilities(
    caps: Capabilities,
) -> dict[str, Any] | None:
    """Return the ``disabled_params`` dict for the given capabilities, or None.

    For reasoning models that do not accept sampler controls, disables
    ``temperature``, ``top_p``, ``presence_penalty``, ``frequency_penalty``.
    For models without structured-output support, disables ``response_format``.
    Returns ``None`` when no parameters need to be suppressed.
    """
    disabled: dict[str, Any] = {}

    if not caps.supports_sampler_controls:
        disabled["temperature"] = None
        disabled["top_p"] = None
        disabled["presence_penalty"] = None
        disabled["frequency_penalty"] = None

    if not caps.supports_structured_outputs:
        disabled["response_format"] = None

    return disabled or None


def disabled_params_for_model(
    model_name: str,
    provider: ProviderType | None = None,
) -> dict[str, Any] | None:
    """Resolve capabilities for *model_name* and return its disabled_params dict.

    Convenience wrapper that combines :func:`detect_capabilities` and
    :func:`disabled_params_for_capabilities`.
    """
    caps = detect_capabilities(model_name, provider=provider)
    return disabled_params_for_capabilities(caps)
