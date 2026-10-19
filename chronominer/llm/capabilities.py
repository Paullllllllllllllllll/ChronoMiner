"""ChronoMiner's view of the shared model capability table.

The family rules and their sources are in
:mod:`chronominer.common.capability_table`; this tool adds no extension. Ids
are matched after a ``provider:`` or ``models/`` prefix is removed, and
``vendor/model`` ids resolve through the OpenRouter rules. Unmatched ids get
the conservative profile: no image input and no structured output.
"""

from __future__ import annotations

from chronominer.common import capabilities as engine
from chronominer.common.capabilities import Capabilities, CapabilityError
from chronominer.common.capability_table import ORIGINAL_MAX_SIDE, TABLE

__all__ = [
    "ORIGINAL_MAX_SIDE",
    "TABLE",
    "Capabilities",
    "CapabilityError",
    "custom_capabilities",
    "detect_capabilities",
    "detect_provider",
    "ensure_image_support",
]


def detect_provider(model: str) -> str:
    """Return the provider a model id names, or ``unknown``."""
    return engine.detect_provider(model, TABLE)


def detect_capabilities(model: str, provider: str | None = None) -> Capabilities:
    """Return the capabilities of *model*; *provider* overrides detection."""
    return engine.resolve_capabilities(model, TABLE, provider)


def custom_capabilities(
    model: str, *, supports_vision: bool, supports_schema: bool
) -> Capabilities:
    """Return the capabilities a named custom endpoint declares for *model*."""
    return Capabilities(
        provider_name=engine.CUSTOM,
        model_name=model,
        family=engine.CUSTOM,
        supports_vision=supports_vision,
        supports_structured_output=supports_schema,
        supports_json_mode=supports_schema,
    )


def ensure_image_support(model_name: str, capabilities: Capabilities) -> None:
    """Raise ``CapabilityError`` unless the model accepts image input."""
    if not capabilities.supports_vision:
        raise CapabilityError(
            f"Model '{model_name}' does not support image inputs. Choose an "
            "image-capable model (e.g., gpt-5.6-luna, claude-sonnet-5, "
            "gemini-3.5-flash)."
        )
