"""One request builder per provider, for synchronous calls and batch bodies.

A :class:`RequestPlan` holds what a tool decided for one model: provider,
capabilities, the resolved output limit, the reasoning effort, samplers,
verbosity, service tier and prompt caching. :func:`request_params` turns it
into the request parameters in the provider's own names, which a batch body
carries as they are and a synchronous call sends either as LangChain invoke
kwargs or, renamed to the LangChain class's fields by
:func:`chat_model_fields`, as constructor kwargs:

- OpenAI (Responses API): ``max_output_tokens``, ``service_tier`` (``flex``
  only for the models listed for it), ``reasoning.effort`` (a level the model
  documents) or the samplers it accepts, ``text.verbosity`` and the
  prompt-cache parameter the model takes.
- Anthropic (Messages API): ``max_tokens``; adaptive-thinking models get
  ``thinking: {type: adaptive}`` and the other reasoning models a thinking
  budget below the limit; ``output_config.effort`` for the families that take
  it; effort ``none`` sends neither thinking nor effort; samplers only
  without thinking.
- Google (generation config): ``max_output_tokens``, the samplers the model
  accepts and ``thinking_config`` with a ``thinking_level`` (Gemini 3 and
  later) or a ``thinking_budget`` (Gemini 2).
- OpenRouter (Chat Completions): ``max_tokens``, samplers and the unified
  ``reasoning`` object in ``extra_body``, its effort clamped to a level the
  model and OpenRouter take.
- Custom endpoints: ``max_tokens`` and the temperature and top_p where set.

:func:`message_content` and :func:`system_content` build the user and system
content of a synchronous call; :func:`batch_body` builds the provider-native
batch body from the same parameters, the same content and the schema the
provider enforces (OpenAI ``text.format``, Anthropic
``output_config.format``, Google ``response_json_schema``). A Gemini image
carries the media resolution its detail maps to, per part, in both paths.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .capabilities import Capabilities
from .images import google_media_resolution
from .request_options import (
    SAMPLERS,
    RequestRules,
    anthropic_effort,
    anthropic_thinking,
    batch_service_tier,
    clamp_effort,
    google_thinking_budget,
    google_thinking_level,
    openai_prompt_cache,
    openai_service_tier,
    openrouter_reasoning,
)
from .settings_schema import PromptCaching
from .structured import anthropic_output_schema, build_text_format, schema_body

__all__ = [
    "BATCH_PROVIDERS",
    "Content",
    "Part",
    "RequestPlan",
    "batch_body",
    "cache_control",
    "chat_model_fields",
    "image_detail",
    "media_resolution",
    "message_content",
    "request_params",
    "system_blocks",
    "system_content",
]

logger = logging.getLogger(__name__)

Content = list[str | dict[Any, Any]]
"""Message content as LangChain messages take it: text and content blocks."""

BATCH_PROVIDERS = frozenset({"openai", "anthropic", "google"})
"""The providers :func:`batch_body` builds bodies for."""

_DETAILS: Mapping[str, frozenset[str]] = {
    "openai": frozenset({"low", "high", "auto", "original"}),
    "openrouter": frozenset({"low", "high", "auto"}),
}


@dataclass(frozen=True)
class RequestPlan:
    """What a tool decided for the requests of one model.

    *limit* is the output limit sent (None: none); *conservative* the limit
    an Anthropic request sends when none is known. *caching* None sends no
    prompt-cache parameter or breakpoint. With *reasoning_only_service_tier*
    only OpenAI reasoning models take the service tier.
    """

    provider: str
    model: str
    caps: Capabilities
    limit: int | None
    effort: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    verbosity: str | None = None
    service_tier: str | None = None
    caching: PromptCaching | None = None
    conservative: int = 4096
    reasoning_only_service_tier: bool = False

    @classmethod
    def resolve(
        cls,
        provider: str,
        model: str,
        caps: Capabilities,
        rules: RequestRules,
        *,
        requested_limit: int | None = None,
        **values: Any,
    ) -> RequestPlan:
        """Return the plan of *model*, its output limit decided by *rules*."""
        return cls(
            provider=provider,
            model=model,
            caps=caps,
            limit=rules.limits.resolve(provider, requested_limit, caps),
            conservative=rules.limits.conservative,
            reasoning_only_service_tier=rules.reasoning_only_service_tier,
            **values,
        )

    @property
    def caching_on(self) -> bool:
        """Whether prompt caching is on."""
        return self.caching is not None and self.caching.enabled


@dataclass(frozen=True)
class Part:
    """One part of a user message: a text, or a base64 image with its detail.

    *detail* is the image detail of the page settings: the OpenAI or
    OpenRouter ``detail``, or the Gemini media resolution it maps to.
    """

    text: str | None = None
    data: str | None = None
    mime: str | None = None
    detail: str | None = None

    @property
    def is_image(self) -> bool:
        """Whether the part is an image."""
        return self.data is not None


def _samplers(plan: RequestPlan) -> dict[str, Any]:
    """Return the samplers a model accepts: temperature and top_p where they
    are set, the penalties (as 0) where the model takes them."""
    caps = plan.caps
    values: dict[str, Any] = {}
    if caps.supports_temperature and plan.temperature is not None:
        values["temperature"] = float(plan.temperature)
    if caps.supports_top_p and plan.top_p is not None:
        values["top_p"] = float(plan.top_p)
    if caps.supports_frequency_penalty:
        values["frequency_penalty"] = 0.0
    if caps.supports_presence_penalty:
        values["presence_penalty"] = 0.0
    return values


def _openai(plan: RequestPlan) -> dict[str, Any]:
    caps = plan.caps
    params: dict[str, Any] = {}
    if plan.limit is not None:
        params["max_output_tokens"] = plan.limit
    tier = openai_service_tier(plan.service_tier, caps)
    if tier and (caps.is_reasoning_model or not plan.reasoning_only_service_tier):
        params["service_tier"] = tier
    if caps.is_reasoning_model:
        if caps.supports_reasoning_effort and plan.effort:
            params["reasoning"] = {"effort": clamp_effort(plan.effort, caps)}
    else:
        params.update(_samplers(plan))
    if caps.supports_text_verbosity and plan.verbosity:
        params["text"] = {"verbosity": plan.verbosity}
    if plan.caching_on and plan.caching is not None:
        params.update(openai_prompt_cache(caps, plan.caching.openai_retention))
    return params


def _anthropic(plan: RequestPlan) -> dict[str, Any]:
    caps = plan.caps
    limit = plan.limit if plan.limit is not None else plan.conservative
    params: dict[str, Any] = {"max_tokens": limit}
    thinking = anthropic_thinking(plan.effort, caps, limit, model=plan.model)
    if thinking is not None:
        params["thinking"] = thinking
    else:
        params.update(_samplers(plan))
    effort = anthropic_effort(caps.family, plan.effort)
    if effort is not None and caps.supports_reasoning_effort:
        params["output_config"] = {"effort": effort}
    return params


def _google(plan: RequestPlan) -> dict[str, Any]:
    caps = plan.caps
    params: dict[str, Any] = {}
    if plan.limit is not None:
        params["max_output_tokens"] = plan.limit
    params.update(_samplers(plan))
    if plan.effort:
        level = google_thinking_level(plan.effort, caps)
        if level is not None:
            params["thinking_config"] = {"thinking_level": level}
        else:
            budget = google_thinking_budget(plan.effort, caps, plan.limit)
            if budget is not None:
                params["thinking_config"] = {"thinking_budget": budget}
    return params


def _openrouter(plan: RequestPlan) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if plan.limit is not None:
        params["max_tokens"] = plan.limit
    params.update(_samplers(plan))
    reasoning = openrouter_reasoning(
        plan.model, plan.effort or "", plan.caps, plan.limit
    )
    if reasoning is not None:
        params["extra_body"] = {"reasoning": reasoning}
    return params


def _custom(plan: RequestPlan) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if plan.limit is not None:
        params["max_tokens"] = plan.limit
    if plan.temperature is not None:
        params["temperature"] = float(plan.temperature)
    if plan.top_p is not None:
        params["top_p"] = float(plan.top_p)
    return params


_PARAMS = {
    "openai": _openai,
    "anthropic": _anthropic,
    "google": _google,
    "openrouter": _openrouter,
    "custom": _custom,
}


def request_params(plan: RequestPlan) -> dict[str, Any]:
    """Return the request parameters of *plan* in the provider's names.

    Raises:
        ValueError: The provider is unknown.
    """
    try:
        build = _PARAMS[plan.provider]
    except KeyError:
        raise ValueError(f"Unsupported provider: {plan.provider}") from None
    return build(plan)


def chat_model_fields(plan: RequestPlan) -> dict[str, Any]:
    """Return :func:`request_params` as fields of the provider's LangChain class.

    The values are those of :func:`request_params`; only the names follow
    the LangChain classes: OpenAI's output limit is ``max_completion_tokens``
    (reasoning models, which also get their samplers unset) or
    ``max_tokens``, the verbosity ``verbosity`` and
    ``prompt_cache_retention`` a ``model_kwargs`` entry; Anthropic's
    ``output_config.effort`` is ``effort``; Google's output limit is
    ``max_tokens`` and its ``thinking_config`` the flat ``thinking_level`` or
    ``thinking_budget``. A tool that builds its chat model per call passes
    :func:`request_params` as invoke kwargs instead.

    Raises:
        ValueError: The provider is unknown.
    """
    params = request_params(plan)
    fields: dict[str, Any] = {}
    if plan.provider == "openai":
        limit = params.pop("max_output_tokens", None)
        if plan.caps.is_reasoning_model:
            if limit is not None:
                fields["max_completion_tokens"] = limit
            fields.update(dict.fromkeys(SAMPLERS))
        elif limit is not None:
            fields["max_tokens"] = limit
        text = params.pop("text", None)
        if isinstance(text, Mapping) and "verbosity" in text:
            fields["verbosity"] = text["verbosity"]
        retention = params.pop("prompt_cache_retention", None)
        if retention is not None:
            fields["model_kwargs"] = {"prompt_cache_retention": retention}
    elif plan.provider == "anthropic":
        output_config = params.pop("output_config", None)
        if isinstance(output_config, Mapping) and "effort" in output_config:
            fields["effort"] = output_config["effort"]
    elif plan.provider == "google":
        if "max_output_tokens" in params:
            fields["max_tokens"] = params.pop("max_output_tokens")
        thinking = params.pop("thinking_config", None)
        if isinstance(thinking, Mapping):
            fields.update(thinking)
    fields.update(params)
    return fields


def image_detail(plan: RequestPlan, detail: str | None) -> str | None:
    """Return the ``detail`` of an OpenAI or OpenRouter image, or None for none.

    Only models with an image detail take one; ``original`` becomes
    ``high`` where the model or OpenRouter does not take it.
    """
    allowed = _DETAILS.get(plan.provider)
    if allowed is None or not detail or not plan.caps.supports_image_detail:
        return None
    chosen = detail.strip().lower()
    if chosen == "original" and (
        "original" not in allowed or not plan.caps.supports_original_image_detail
    ):
        return "high"
    return chosen if chosen in allowed else None


def media_resolution(plan: RequestPlan, detail: str | None) -> str | None:
    """Return the per-part Gemini media resolution of an image, or None.

    The level follows the image's detail (``high`` becomes
    ``MEDIA_RESOLUTION_HIGH``); Gemini 2 and earlier, and models without
    the control, take none.
    """
    caps = plan.caps
    if plan.provider != "google" or not caps.supports_media_resolution:
        return None
    if caps.family.startswith(("gemini-1", "gemini-2")):
        return None
    return google_media_resolution(detail)


def _data_url(part: Part) -> str:
    return f"data:{part.mime};base64,{part.data}"


def message_content(plan: RequestPlan, parts: Sequence[Part]) -> Content:
    """Return the LangChain content blocks of a synchronous user message."""
    blocks: Content = []
    for part in parts:
        if not part.is_image:
            blocks.append({"type": "text", "text": part.text or ""})
        elif plan.provider == "anthropic":
            source = {"type": "base64", "media_type": part.mime, "data": part.data}
            blocks.append({"type": "image", "source": source})
        elif plan.provider == "google":
            block: dict[str, Any] = {
                "type": "image",
                "base64": part.data,
                "mime_type": part.mime,
            }
            level = media_resolution(plan, part.detail)
            if level is not None:
                block["media_resolution"] = level
            blocks.append(block)
        else:
            url: dict[str, Any] = {"url": _data_url(part)}
            detail = image_detail(plan, part.detail)
            if detail is not None:
                url["detail"] = detail
            blocks.append({"type": "image_url", "image_url": url})
    return blocks


def cache_control(
    provider: str, model: str, caching: PromptCaching | None
) -> dict[str, Any] | None:
    """Return the ``cache_control`` of a system prompt, or None for none.

    With prompt caching on, an Anthropic system prompt carries a breakpoint
    with the configured TTL, and so does a Claude model's through OpenRouter
    when ``openrouter_cache_control`` is set.
    """
    if caching is None or not caching.enabled:
        return None
    if provider == "anthropic":
        control: dict[str, Any] = {"type": "ephemeral"}
        if caching.anthropic_ttl == "1h":
            control["ttl"] = "1h"
        return control
    name = model.lower()
    if (
        provider == "openrouter"
        and caching.openrouter_cache_control
        and ("claude" in name or "anthropic/" in name)
    ):
        return {"type": "ephemeral"}
    return None


def system_blocks(prompt: str, control: Mapping[str, Any] | None) -> str | Content:
    """Return *prompt* as one text block carrying *control*, else as text."""
    if control is None:
        return prompt
    return [{"type": "text", "text": prompt, "cache_control": dict(control)}]


def system_content(plan: RequestPlan, prompt: str) -> str | Content:
    """Return the system prompt as sent: one text block with the cache
    breakpoint where there is one (:func:`cache_control`), else the text."""
    return system_blocks(prompt, cache_control(plan.provider, plan.model, plan.caching))


def _openai_body(
    plan: RequestPlan, system: str, parts: Sequence[Part], schema: Any
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for part in parts:
        if not part.is_image:
            content.append({"type": "input_text", "text": part.text or ""})
            continue
        image: dict[str, Any] = {"type": "input_image", "image_url": _data_url(part)}
        detail = image_detail(plan, part.detail)
        if detail is not None:
            image["detail"] = detail
        content.append(image)
    params = request_params(plan)
    tier = batch_service_tier(params.pop("service_tier", None))
    text = dict(params.pop("text", None) or {})
    if schema is not None:
        text["format"] = build_text_format(schema)
    body: dict[str, Any] = {
        "model": plan.model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": content},
        ],
        **params,
    }
    if tier is not None:
        body["service_tier"] = tier
    if text:
        body["text"] = text
    return body


def _anthropic_body(
    plan: RequestPlan, system: str, parts: Sequence[Part], schema: Any
) -> dict[str, Any]:
    params = request_params(plan)
    if schema is not None:
        output_config = dict(params.get("output_config") or {})
        output_config["format"] = {
            "type": "json_schema",
            "schema": anthropic_output_schema(schema),
        }
        params["output_config"] = output_config
    content = message_content(plan, parts)
    return {
        "model": plan.model,
        "system": system_content(plan, system),
        "messages": [{"role": "user", "content": content}],
        **params,
    }


def _google_body(
    plan: RequestPlan, system: str, parts: Sequence[Part], schema: Any
) -> dict[str, Any]:
    contents: list[dict[str, Any]] = []
    for part in parts:
        if not part.is_image:
            contents.append({"text": part.text or ""})
            continue
        image: dict[str, Any] = {
            "inline_data": {"mime_type": part.mime, "data": part.data}
        }
        level = media_resolution(plan, part.detail)
        if level is not None:
            image["media_resolution"] = {"level": level}
        contents.append(image)
    config = request_params(plan)
    config["system_instruction"] = system
    body = schema_body(schema)
    if body is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = body
    return {
        "model": plan.model,
        "contents": [{"role": "user", "parts": contents}],
        "config": config,
    }


_BODIES = {
    "openai": _openai_body,
    "anthropic": _anthropic_body,
    "google": _google_body,
}


def batch_body(
    plan: RequestPlan,
    system: str,
    parts: Sequence[Part],
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the provider-native batch body of one request.

    *schema* is the schema spec the provider enforces (``{name, strict,
    schema}``, for Anthropic as the synchronous request sends it), None
    under prompted JSON and plain text. The body carries the parameters of
    :func:`request_params`; an OpenAI ``flex`` tier, which batches do not
    take, becomes ``auto``.

    Raises:
        ValueError: The provider has no batch API.
    """
    try:
        build = _BODIES[plan.provider]
    except KeyError:
        raise ValueError(f"Provider '{plan.provider}' has no batch API") from None
    return build(plan, system, parts, schema)
