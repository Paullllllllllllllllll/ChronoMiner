"""Client options, request plan and response format of each provider's model.

:func:`request_plan` turns a :class:`~chronominer.spec.ModelSpec`, the
model's capabilities and the settings into the shared
:class:`~chronominer.common.provider_requests.RequestPlan`, whose
request parameters (:func:`request_kwargs`) every batch body carries and
every synchronous chat model takes as constructor kwargs
(:func:`constructor_kwargs`), together with what the provider's LangChain
class needs beyond them, the API key excepted: the model, timeouts
(per-phase httpx timeouts for OpenAI-compatible clients, a number for
Anthropic and Google), OpenAI's Responses API and ``disabled_params``,
OpenRouter's base URL and headers, a custom endpoint's ``base_url``.

An unset output limit takes the model family's limit on OpenAI, Anthropic and
Google, the documented limit of an OpenRouter model the capability facts
list, and :data:`ROUTED_LIMIT` on other OpenRouter models and custom
endpoints; Anthropic, Google and documented OpenRouter models clamp a set
limit to the family's. :func:`response_format_for` decides between schema,
json and text.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from chronominer.common.capabilities import UNKNOWN, Capabilities
from chronominer.common.http_timeouts import build_httpx_timeout
from chronominer.common.provider_requests import (
    RequestPlan,
    chat_model_fields,
    request_params,
)
from chronominer.common.request_options import (
    ROUTED_LIMIT,
    OutputLimits,
    RequestRules,
    disabled_params,
)
from chronominer.common.structured import RESPONSE_FORMATS, ResponseFormat
from chronominer.llm.capabilities import (
    TABLE,
    custom_capabilities,
    detect_capabilities,
)
from chronominer.settings import Endpoint, Settings
from chronominer.spec import ModelSpec

__all__ = [
    "OPENROUTER_BASE_URL",
    "OPENROUTER_HEADERS",
    "ROUTED_LIMIT",
    "RULES",
    "constructor_kwargs",
    "disabled_params",
    "endpoint_of",
    "model_capabilities",
    "output_limit",
    "request_kwargs",
    "request_plan",
    "response_format_for",
]

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS: Mapping[str, str] = MappingProxyType(
    {
        "HTTP-Referer": "https://github.com/ChronoMiner",
        "X-Title": "ChronoMiner",
    }
)

_CONSERVATIVE_LIMIT = int(
    TABLE.unknown.get(
        "max_output_tokens",
        Capabilities(provider_name=UNKNOWN, model_name="").max_output_tokens,
    )
)

RULES = RequestRules(
    limits=OutputLimits(
        conservative=_CONSERVATIVE_LIMIT,
        clamped=frozenset({"anthropic", "google"}),
        routed=ROUTED_LIMIT,
    ),
    reasoning_only_service_tier=True,
)
"""This tool's request rules: Anthropic and Google clamp a set output limit,
OpenRouter and custom endpoints take :data:`ROUTED_LIMIT` when none is set
and none is documented, and only OpenAI reasoning models take the service
tier."""


def endpoint_of(model: ModelSpec, settings: Settings) -> Endpoint:
    """Return the named custom endpoint of *model*.

    Raises:
        ValueError: The model names no endpoint.
        SettingsError: The settings define no endpoint of that name.
    """
    if model.endpoint is None:
        raise ValueError("provider custom requires a named endpoint")
    return settings.endpoint(model.endpoint)


def model_capabilities(model: ModelSpec, settings: Settings) -> Capabilities:
    """Return the capabilities of *model*; a custom endpoint declares its own."""
    if model.provider == "custom":
        endpoint = endpoint_of(model, settings)
        return custom_capabilities(
            model.model,
            supports_vision=endpoint.supports_vision,
            supports_schema=endpoint.supports_schema,
        )
    return detect_capabilities(model.model, model.provider)


def response_format_for(model: ModelSpec, caps: Capabilities) -> ResponseFormat:
    """Return the response format of the calls of *model*.

    Unset takes schema when the model supports structured output, else json.
    Schema on a model without structured output falls back to json with a
    warning; a custom endpoint keeps the requested format.

    Raises:
        ValueError: The requested format is not a response format.
    """
    requested = model.response_format
    supported = caps.supports_structured_output
    if requested is None:
        return "schema" if supported else "json"
    if requested not in RESPONSE_FORMATS:
        raise ValueError(f"Unknown response format: {requested!r}")
    if requested == "schema" and not supported and model.provider != "custom":
        logger.warning(
            "Model %s has no structured output; using prompted JSON", model.model
        )
        return "json"
    return requested


def output_limit(model: ModelSpec, caps: Capabilities) -> int | None:
    """Return the output limit sent to the model, or None to send none.

    See :class:`~chronominer.common.request_options.OutputLimits` and
    :data:`RULES`.
    """
    return RULES.limits.resolve(model.provider, model.max_output_tokens, caps)


def request_plan(
    model: ModelSpec, settings: Settings, caps: Capabilities | None = None
) -> RequestPlan:
    """Return the request plan of *model*.

    *caps* defaults to :func:`model_capabilities`.

    Raises:
        ValueError: A custom model names no endpoint.
        SettingsError: The named endpoint is not defined.
    """
    caps = caps or model_capabilities(model, settings)
    return RequestPlan.resolve(
        model.provider,
        model.model,
        caps,
        RULES,
        requested_limit=model.max_output_tokens,
        effort=model.reasoning_effort,
        temperature=model.temperature,
        top_p=model.top_p,
        verbosity=model.verbosity,
        service_tier=model.service_tier,
        caching=settings.prompt_caching,
    )


def request_kwargs(
    model: ModelSpec, settings: Settings, caps: Capabilities | None = None
) -> dict[str, Any]:
    """Return the request parameters every call of *model* sends.

    Raises:
        ValueError: The provider is unknown, or a custom model names no
            endpoint.
        SettingsError: The named endpoint is not defined.
    """
    return request_params(request_plan(model, settings, caps))


def _httpx_timeout(settings: Settings) -> Any:
    timeouts = settings.timeouts
    return build_httpx_timeout(
        timeouts.request,
        connect=timeouts.connect,
        write=timeouts.write,
        pool=timeouts.pool,
    )


def _openai(model: ModelSpec, caps: Capabilities, settings: Settings) -> dict[str, Any]:
    return {
        "model": model.model,
        "timeout": _httpx_timeout(settings),
        "max_retries": 0,
        "disabled_params": disabled_params(caps),
        "use_responses_api": True,
    }


def _anthropic(
    model: ModelSpec, caps: Capabilities, settings: Settings
) -> dict[str, Any]:
    return {
        "model": model.model,
        "timeout": float(settings.timeouts.request),
        "max_retries": 0,
    }


def _google(model: ModelSpec, caps: Capabilities, settings: Settings) -> dict[str, Any]:
    return {
        "model": model.model,
        "timeout": float(settings.timeouts.request),
        "max_retries": 0,
        "top_k": None,
    }


def _openrouter(
    model: ModelSpec, caps: Capabilities, settings: Settings
) -> dict[str, Any]:
    return {
        "model": model.model,
        "base_url": OPENROUTER_BASE_URL,
        "timeout": _httpx_timeout(settings),
        "max_retries": 0,
        "disabled_params": disabled_params(caps),
        "default_headers": dict(OPENROUTER_HEADERS),
    }


def _custom(model: ModelSpec, caps: Capabilities, settings: Settings) -> dict[str, Any]:
    return {
        "model": model.model,
        "base_url": endpoint_of(model, settings).base_url,
        "timeout": _httpx_timeout(settings),
        "max_retries": 0,
    }


_BUILDERS = MappingProxyType(
    {
        "openai": _openai,
        "anthropic": _anthropic,
        "google": _google,
        "openrouter": _openrouter,
        "custom": _custom,
    }
)


def constructor_kwargs(
    model: ModelSpec, settings: Settings, caps: Capabilities | None = None
) -> dict[str, Any]:
    """Return the constructor kwargs of *model*'s chat model, without its key.

    *caps* defaults to :func:`model_capabilities`. SDK retries are off; the
    request parameters are those of :func:`request_kwargs`, under the
    class's field names (:func:`~chronominer.common.provider_requests.
    chat_model_fields`).

    Raises:
        ValueError: The provider is unknown, or a custom model names no
            endpoint.
        SettingsError: The named endpoint is not defined.
    """
    try:
        builder = _BUILDERS[model.provider]
    except KeyError:
        raise ValueError(f"Unsupported provider: {model.provider}") from None
    caps = caps or model_capabilities(model, settings)
    fields = chat_model_fields(request_plan(model, settings, caps))
    return {**builder(model, caps, settings), **fields}
