"""Provider names and the provider a model name implies.

A ``provider:`` prefix naming a supported provider selects it. Any other name
goes through the provider detection of the shared capability table, so
routing and capability resolution agree: Google's ``models/`` prefix and the
``gemini`` and ``gemma`` names select google, a ``vendor/model`` id selects
openrouter, and the Anthropic and OpenAI names select theirs. A name the
table assigns no provider implies none and runs on openai.
:func:`api_model_id` turns a dotted Anthropic alias into its API id.
"""

from __future__ import annotations

from chronominer.common.capabilities import detect_provider
from chronominer.common.capability_table import TABLE

__all__ = [
    "DEFAULT_PROVIDER",
    "PROVIDERS",
    "api_model_id",
    "implied_provider",
    "infer_provider",
    "split_provider_prefix",
]

PROVIDERS = ("openai", "anthropic", "google", "openrouter", "custom")
DEFAULT_PROVIDER = "openai"


def split_provider_prefix(model: str) -> tuple[str | None, str]:
    """Return the provider a ``provider:`` prefix names and the bare model name.

    Only a prefix naming a supported provider counts, so model ids that
    contain a colon (OpenAI fine-tunes, ``ft:gpt-4o:org:id``) stay whole.
    """
    prefix, separator, rest = model.partition(":")
    provider = prefix.strip().lower()
    if separator and provider in PROVIDERS:
        return provider, rest.strip()
    return None, model


def implied_provider(model: str) -> str | None:
    """Return the provider *model* implies, or None when the name implies none."""
    prefixed, name = split_provider_prefix(model)
    if prefixed is not None:
        return prefixed
    detected = detect_provider(name, TABLE)
    return detected if detected in PROVIDERS else None


def infer_provider(model: str) -> str:
    """Return the provider of *model*: the implied one, else openai."""
    return implied_provider(model) or DEFAULT_PROVIDER


def api_model_id(provider: str, model: str) -> str:
    """Return the id under which *provider*'s API knows *model*.

    Anthropic ids spell versions with dashes: the dotted alias
    ``claude-haiku-5.5``, which the capability table accepts, is sent as
    ``claude-haiku-5-5``. Other names keep their spelling.
    """
    if provider == "anthropic" and model.lower().startswith("claude"):
        return model.replace(".", "-")
    return model
