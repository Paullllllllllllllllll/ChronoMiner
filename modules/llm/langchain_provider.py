# modules/llm/langchain_provider.py

"""
LangChain-based provider abstraction for multi-provider LLM support.

Supports:
- OpenAI (GPT-4o, GPT-4.1, o1, o3, GPT-5)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, etc.)
- Google Gemini (Gemini 2.0 Flash, Gemini 1.5 Pro, etc.)
- OpenRouter (unified access to multiple providers)
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from modules.config.capabilities import detect_provider as _canonical_detect_provider
from modules.config.loader import get_config_loader, resolve_api_key
from modules.infra.logger import setup_logger
from modules.infra.token_tracker import get_token_tracker

logger = setup_logger(__name__)

ProviderType = Literal["openai", "anthropic", "google", "openrouter", "custom"]


async def _aclose_maybe(obj: Any) -> None:
    """Call ``obj.close()`` if present, awaiting an async closer. Never raises.

    Used to tear down provider SDK clients (which own httpx connections) when a
    LangChain chat model is disposed. A missing or non-callable ``close`` is a
    no-op; any exception is logged at debug level and swallowed.
    """
    if obj is None:
        return
    close = getattr(obj, "close", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # Best-effort cleanup; never propagate.
        logger.debug("Error closing client %r: %s", type(obj).__name__, exc)


def _normalize_schema_for_anthropic(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_normalize_schema_for_anthropic(item) for item in schema]

    if not isinstance(schema, dict):
        return schema

    type_val = schema.get("type")
    if isinstance(type_val, list):
        types = [t for t in type_val if t != "null"]
        has_null = any(t == "null" for t in type_val)
        any_of: list[dict[str, Any]] = []

        for t in types:
            variant: dict[str, Any] = {}
            for key, value in schema.items():
                if key == "type":
                    continue
                variant[key] = _normalize_schema_for_anthropic(value)
            variant["type"] = t
            any_of.append(variant)

        if has_null:
            any_of.append({"type": "null"})

        normalized: dict[str, Any] = {}
        for meta_key in ("title", "description", "default", "examples"):
            if meta_key in schema:
                normalized[meta_key] = _normalize_schema_for_anthropic(schema[meta_key])
        normalized["anyOf"] = any_of
        return normalized

    normalized = {}
    for key, value in schema.items():
        normalized[key] = _normalize_schema_for_anthropic(value)
    return normalized


def _load_concurrency_config() -> dict[str, Any]:
    """Load concurrency config for retry and rate limiting settings."""
    try:
        return get_config_loader().get_concurrency_config() or {}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def _compute_reasoning_budget(max_tokens: int, effort: str) -> int:
    """
    Compute reasoning token budget based on effort level.

    Used by all providers (Anthropic, Google, OpenRouter) that support
    reasoning/thinking modes. Maps effort levels to a token budget.

    Args:
        max_tokens: The max_output_tokens from config
        effort: Reasoning effort level (low, medium, high)

    Returns:
        Token budget for reasoning, or 0 to skip
    """
    effort_lower = effort.lower().strip()

    # Effort to budget ratio mapping
    # low: minimal reasoning overhead
    # medium: balanced
    # high: maximum reasoning depth
    ratios = {
        "low": 0.1,  # 10% of output budget for reasoning
        "medium": 0.25,  # 25% of output budget for reasoning
        "high": 0.5,  # 50% of output budget for reasoning
        "none": 0.0,  # Disable reasoning
    }

    ratio = ratios.get(effort_lower, 0.25)  # Default to medium
    if ratio <= 0:
        return 0

    # Compute budget with reasonable bounds
    budget = int(max_tokens * ratio)
    # Minimum 1024 tokens for meaningful reasoning, max 32768
    return max(1024, min(budget, 32768))


def _build_reasoning_payload(
    model_name: str,
    reasoning_config: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any] | None:
    """
    Build provider-compatible reasoning payload based on model type.

    Routes reasoning configuration to the appropriate format for the
    target provider (OpenRouter, Anthropic, Google, DeepSeek).

    Args:
        model_name: The model identifier
        reasoning_config: Reasoning config from model_config.yaml
        max_tokens: Max output tokens for budget computation

    Returns:
        Reasoning payload dict, or None if not applicable
    """
    if not reasoning_config:
        return None

    m = model_name.lower().strip()
    reasoning_payload: dict[str, Any] = {}

    effort = reasoning_config.get("effort")
    if effort:
        reasoning_payload["effort"] = str(effort)

    # Handle explicit max_tokens override
    max_reasoning_tokens = reasoning_config.get("max_tokens")
    if max_reasoning_tokens is not None:
        with contextlib.suppress(ValueError, TypeError):
            reasoning_payload["max_tokens"] = int(max_reasoning_tokens)

    # Handle exclude flag
    exclude = reasoning_config.get("exclude")
    if exclude is not None:
        reasoning_payload["exclude"] = bool(exclude)

    # Handle enabled flag — auto-enable when effort is set, except "none"
    enabled = reasoning_config.get("enabled")
    if enabled is not None:
        reasoning_payload["enabled"] = bool(enabled)
    elif "effort" in reasoning_payload:
        effort_val = str(reasoning_payload["effort"]).lower().strip()
        reasoning_payload["enabled"] = effort_val != "none"

    if not reasoning_payload:
        return None

    # Provider-specific translations

    # For Anthropic and Gemini thinking models, map effort -> max_tokens budget
    if (
        "anthropic/" in m or "claude" in m or "gemini" in m
    ) and "max_tokens" not in reasoning_payload:
        eff = reasoning_payload.get("effort", "medium")
        budget = _compute_reasoning_budget(max_tokens=max_tokens, effort=str(eff))
        if budget > 0:
            # Remove effort and use max_tokens instead for these providers
            reasoning_payload.pop("effort", None)
            reasoning_payload["max_tokens"] = budget

    # For DeepSeek models, map effort to enabled flag
    if "deepseek/" in m or "deepseek" in m:
        eff = str(reasoning_payload.get("effort", "medium")).lower().strip()
        reasoning_payload.pop("effort", None)
        if "enabled" not in reasoning_payload:
            reasoning_payload["enabled"] = eff != "none"

    return reasoning_payload if reasoning_payload else None


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    provider: ProviderType
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout: float = 600.0
    max_retries: int = 5
    reasoning_effort: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        model_config: dict[str, Any],
        provider_override: ProviderType | None = None,
        concurrency_config: dict[str, Any] | None = None,
    ) -> ProviderConfig:
        """Create ProviderConfig from model_config dictionary."""
        tm = model_config.get("extraction_model", {})
        model_name = tm.get("name", "")

        # Check for explicit provider field in config, then override, then auto-detect
        config_provider = tm.get("provider")
        if provider_override:
            provider = provider_override
        elif config_provider:
            # Normalize provider name
            provider = config_provider.lower().strip()
            if provider not in (
                "openai",
                "anthropic",
                "google",
                "openrouter",
                "custom",
            ):
                logger.warning(
                    f"Unknown provider '{config_provider}', "
                    "auto-detecting from model name"
                )
                provider = cls._detect_provider(model_name)
        else:
            provider = cls._detect_provider(model_name)

        # Get API key based on provider
        api_key = cls._get_api_key(provider)

        # Get base URL for OpenRouter or custom endpoints
        base_url = None
        if provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        elif provider == "custom":
            custom_cfg = tm.get("custom_endpoint", {})
            base_url = custom_cfg.get("base_url")
            if not base_url:
                raise ValueError(
                    "provider: custom requires custom_endpoint.base_url "
                    "in extraction_model config"
                )

        # Retry authority lives in SynchronousProcessingStrategy's outer loop
        # (429-aware exponential backoff + jitter). LangChain's internal retry
        # is disabled by passing max_retries=0 to avoid double-layered retries.
        if concurrency_config is None:
            concurrency_config = _load_concurrency_config()
        extraction_cfg = (concurrency_config.get("concurrency", {}) or {}).get(
            "extraction", {}
        ) or {}
        max_retries = 0

        # Get timeout from config
        timeouts_cfg = extraction_cfg.get("timeouts", {}) or {}
        timeout = float(timeouts_cfg.get("total", 600.0))

        # Get service_tier from concurrency config
        service_tier = extraction_cfg.get("service_tier")

        # Build extra_params including reasoning config
        extra_params = {
            "presence_penalty": float(tm.get("presence_penalty", 0.0)),
            "frequency_penalty": float(tm.get("frequency_penalty", 0.0)),
        }

        # Add reasoning config if present
        reasoning_cfg = tm.get("reasoning")
        if reasoning_cfg:
            extra_params["reasoning_config"] = reasoning_cfg
            # Extract effort for direct use
            effort = reasoning_cfg.get("effort", "medium")
            extra_params["reasoning_effort"] = effort

        # Add text verbosity config if present (GPT-5 family)
        text_cfg = tm.get("text")
        if text_cfg:
            extra_params["text_config"] = text_cfg

        # Add service_tier from concurrency config
        if service_tier:
            extra_params["service_tier"] = service_tier

        return cls(
            provider=provider,
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=float(tm.get("temperature", 0.0)),
            max_tokens=int(tm.get("max_output_tokens", 4096)),
            top_p=float(tm.get("top_p", 1.0)),
            timeout=timeout,
            max_retries=max_retries,
            extra_params=extra_params,
        )

    @staticmethod
    def _detect_provider(model_name: str) -> ProviderType:
        """
        Detect provider from model name.

        Delegates to the canonical detect_provider() in model_capabilities.py,
        but defaults to "openai" for backward compatibility when provider is unknown.
        """
        provider = _canonical_detect_provider(model_name)
        # Default to "openai" for backward compatibility (canonical returns "unknown")
        if provider == "unknown":
            return "openai"
        return provider

    @staticmethod
    def _get_api_key(provider: ProviderType) -> str | None:
        """Get API key for the specified provider.

        Standard providers resolve through the api_keys_config.yaml mapping
        (override or default env var name). The custom-endpoint branch keeps
        reading its dedicated ``api_key_env_var`` from model config.
        """
        if provider == "custom":
            model_cfg = get_config_loader().get_model_config()
            tm = model_cfg.get("extraction_model", {})
            custom_cfg = tm.get("custom_endpoint", {})
            env_var = custom_cfg.get("api_key_env_var")
            if env_var:
                key = os.getenv(env_var, "").strip()
                if key:
                    return key
            logger.warning(
                f"Custom endpoint API key not found. Set env var: {env_var!r}"
            )
            return None
        return resolve_api_key(provider)


class LangChainLLM:
    """
    Unified LangChain LLM wrapper supporting multiple providers.

    Provides a consistent interface for text processing with structured outputs
    regardless of the underlying provider.

    Capability Guarding:
    ====================
    LangChain does NOT automatically filter unsupported parameters for different
    models (e.g., temperature for o1/o3 reasoning models). We use LangChain's
    `disabled_params` feature combined with our model_capabilities detection
    to automatically filter out unsupported parameters.

    This replaces manual parameter filtering with LangChain's built-in mechanism.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._chat_model = None
        self._initialized = False
        self._capabilities = None

    def _ensure_initialized(self) -> None:
        """Lazily initialize the chat model."""
        if self._initialized:
            return

        self._chat_model = self._create_chat_model()
        self._initialized = True

    def _get_capabilities(self) -> Any:
        """Get model capabilities using our detection logic."""
        if self._capabilities is None:
            from modules.config.capabilities import detect_capabilities

            self._capabilities = detect_capabilities(
                self.config.model, provider=self.config.provider
            )
        return self._capabilities

    def _get_disabled_params(self) -> dict[str, Any] | None:
        """
        Get parameters to disable based on model capabilities.

        LangChain's disabled_params feature allows us to specify which parameters
        should NOT be sent to the API. This is used for capability guarding.

        For reasoning models (o1, o3, gpt-5):
        - Disable temperature, top_p, presence_penalty, frequency_penalty

        For non-structured-output models:
        - Disable response_format
        """
        caps = self._get_capabilities()
        disabled: dict[str, Any] = {}

        # Reasoning models don't support sampler controls
        if not caps.supports_sampler_controls:
            disabled["temperature"] = None
            disabled["top_p"] = None
            disabled["presence_penalty"] = None
            disabled["frequency_penalty"] = None
            logger.debug(
                f"Model {self.config.model}: Disabled sampler controls "
                "(reasoning model)"
            )

        # Some models don't support structured outputs via response_format
        if not caps.supports_structured_outputs:
            disabled["response_format"] = None
            logger.debug(
                f"Model {self.config.model}: Disabled response_format (not supported)"
            )

        return disabled if disabled else None

    def _effective_max_tokens(self) -> int:
        """Requested max output tokens, clamped to the model's known ceiling.

        The capability registry records the provider-enforced per-model output
        cap (``max_output_tokens``). Sending a larger value makes the provider
        reject the request with a 400 (e.g. 128,000 to claude-haiku-4-5,
        whose cap is 64,000), so the requested value is clamped here at
        request-build time with a logged warning. Models with an unknown cap
        (``None``) are never clamped.
        """
        requested = int(self.config.max_tokens)
        cap = getattr(self._get_capabilities(), "max_output_tokens", None)
        if cap is not None and requested > int(cap):
            logger.warning(
                "max_output_tokens %s exceeds the %s cap of %s; clamping to %s",
                f"{requested:,}",
                self.config.model,
                f"{int(cap):,}",
                f"{int(cap):,}",
            )
            return int(cap)
        return requested

    def _create_chat_model(self) -> Any:
        """
        Create the appropriate LangChain chat model based on provider.

        Uses disabled_params for capability guarding - LangChain will automatically
        filter out parameters that are incompatible with the model.
        """
        provider = self.config.provider
        caps = self._get_capabilities()

        # Get disabled params for capability guarding
        disabled_params = self._get_disabled_params()

        # Requested output budget, clamped to the model's known hard cap (CM-2)
        max_tokens = self._effective_max_tokens()

        # Only include sampler controls if supported by model
        common_params = {}
        if caps.supports_sampler_controls:
            common_params["temperature"] = self.config.temperature
            common_params["max_tokens"] = max_tokens
            common_params["top_p"] = self.config.top_p
            if self.config.extra_params.get("presence_penalty"):
                common_params["presence_penalty"] = self.config.extra_params[
                    "presence_penalty"
                ]
            if self.config.extra_params.get("frequency_penalty"):
                common_params["frequency_penalty"] = self.config.extra_params[
                    "frequency_penalty"
                ]
        else:
            # For reasoning models, only set max_tokens (required for output budget)
            common_params["max_tokens"] = max_tokens

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            params: dict[str, Any] = {
                "model": self.config.model,
                "api_key": self.config.api_key,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                **common_params,
            }

            # Add disabled_params for capability guarding
            # LangChain will filter these out before sending to API
            if disabled_params:
                params["disabled_params"] = disabled_params

            # Add reasoning effort for reasoning models
            if caps.supports_reasoning_effort:
                params["reasoning_effort"] = self.config.extra_params.get(
                    "reasoning_effort", "medium"
                )

            # Add service_tier if configured (CM-1)
            service_tier = self.config.extra_params.get("service_tier")
            if service_tier:
                params["service_tier"] = service_tier

            # Add text verbosity for GPT-5 family models (CM-5)
            text_config = self.config.extra_params.get("text_config", {})
            if text_config and str(caps.family).startswith("gpt-5"):
                verbosity = text_config.get("verbosity")
                if verbosity:
                    params.setdefault("model_kwargs", {})["text"] = {
                        "verbosity": verbosity
                    }

            return ChatOpenAI(**params)

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            anthropic_params = dict(common_params)
            if "temperature" in anthropic_params and "top_p" in anthropic_params:
                if anthropic_params.get("temperature") is not None:
                    anthropic_params.pop("top_p", None)
                else:
                    anthropic_params.pop("temperature", None)

            # Add extended thinking for Anthropic when reasoning is configured (CM-4)
            reasoning_config = self.config.extra_params.get("reasoning_config", {})
            if (
                reasoning_config
                and reasoning_config.get("effort")
                and reasoning_config.get("effort") != "none"
            ):
                effort = reasoning_config.get("effort", "medium")
                budget = _compute_reasoning_budget(
                    max_tokens=max_tokens, effort=str(effort)
                )
                if budget > 0:
                    anthropic_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget,
                    }
                    anthropic_params["temperature"] = 1.0
                    anthropic_params.pop("top_p", None)
                    # Both betas are required: interleaved-thinking for
                    # extended thinking, and structured-outputs for native
                    # JSON-schema constrained decoding (used by
                    # with_structured_output(method="json_schema")).
                    # LangChain only auto-adds the structured-outputs beta
                    # when the betas list is empty, so we include it here.
                    anthropic_params["betas"] = [
                        "interleaved-thinking-2025-05-14",
                        "structured-outputs-2025-11-13",
                    ]
                    logger.info(
                        f"Anthropic extended thinking enabled: "
                        f"budget_tokens={budget}, effort={effort}"
                    )

            return ChatAnthropic(
                model=self.config.model,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                **anthropic_params,
            )

        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            google_params = dict(common_params)

            # Add thinking controls for Google when reasoning is configured.
            # langchain-google-genai exposes ``thinking_level`` (Gemini 3.x)
            # and ``thinking_budget`` (Gemini 2.5 / Gemma) as constructor
            # fields; the legacy ``thinking_config`` dict is NOT a supported
            # parameter and is silently shunted into model_kwargs, making the
            # reasoning control ineffective (CM-1, live bug).
            reasoning_config = self.config.extra_params.get("reasoning_config", {})
            if (
                reasoning_config
                and reasoning_config.get("effort")
                and reasoning_config.get("effort") != "none"
            ):
                effort = str(reasoning_config.get("effort", "medium"))
                model_lower = self.config.model.lower().strip()
                if "gemini-3" in model_lower:
                    # Gemini 3.x controls reasoning via a discrete level.
                    level_map = {
                        "minimal": "minimal",
                        "low": "low",
                        "medium": "medium",
                        "high": "high",
                        "xhigh": "high",
                    }
                    level = level_map.get(effort.lower().strip(), "medium")
                    google_params["thinking_level"] = level
                    logger.info(
                        f"Google thinking enabled: thinking_level={level}, "
                        f"effort={effort}"
                    )
                else:
                    budget = _compute_reasoning_budget(
                        max_tokens=max_tokens, effort=effort
                    )
                    if budget > 0:
                        google_params["thinking_budget"] = budget
                        google_params["include_thoughts"] = True
                        logger.info(
                            f"Google thinking enabled: thinking_budget={budget}, "
                            f"effort={effort}"
                        )

            return ChatGoogleGenerativeAI(
                model=self.config.model,
                google_api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                **google_params,
            )

        elif provider == "openrouter":
            from langchain_openai import ChatOpenAI

            # OpenRouter uses OpenAI-compatible API
            model_name = self.config.model
            if model_name.startswith("openrouter/"):
                model_name = model_name[len("openrouter/") :]

            params = {
                "model": model_name,
                "api_key": self.config.api_key,
                "base_url": self.config.base_url,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                **common_params,
            }

            # Add disabled_params for capability guarding
            if disabled_params:
                params["disabled_params"] = disabled_params

            # Build extra_body for OpenRouter (reasoning + provider routing)
            extra_body: dict[str, Any] = {}

            if caps.supports_reasoning_effort:
                reasoning_config = self.config.extra_params.get("reasoning_config", {})
                reasoning_payload = _build_reasoning_payload(
                    model_name=model_name,
                    reasoning_config=reasoning_config,
                    max_tokens=self.config.max_tokens,
                )
                if reasoning_payload:
                    extra_body["reasoning"] = reasoning_payload
                    logger.info(
                        f"Using OpenRouter reasoning={reasoning_payload} "
                        f"for model {model_name}"
                    )

            # Force vision-capable provider for models that need it
            if "gemma" in model_name.lower():
                extra_body["provider"] = {
                    "order": ["Novita"],
                }

            if extra_body:
                params["extra_body"] = extra_body

            # OpenRouter-specific headers
            params["default_headers"] = {
                "HTTP-Referer": "https://github.com/ChronoMiner",
                "X-Title": "ChronoMiner",
            }

            return ChatOpenAI(**params)

        elif provider == "custom":
            from langchain_openai import ChatOpenAI

            params = {
                "model": self.config.model,
                "api_key": self.config.api_key,
                "base_url": self.config.base_url,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                **common_params,
            }
            if disabled_params:
                params["disabled_params"] = disabled_params

            logger.info(
                f"Creating custom endpoint model: {self.config.model} "
                f"at {self.config.base_url}"
            )
            return ChatOpenAI(**params)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @property
    def supports_structured_outputs(self) -> bool:
        """
        Check if the provider/model supports structured outputs.

        Now delegates to model_capabilities.py for centralized capability detection.
        LangChain does NOT automatically detect this - we handle it via our
        capability system and use disabled_params if needed.
        """
        caps = self._get_capabilities()
        return bool(caps.supports_structured_outputs)

    @staticmethod
    def _to_lc_messages(messages: list[dict[str, Any]]) -> list[Any]:
        """Convert role/content message dicts into LangChain message objects.

        Text-only content is flattened to a string; multimodal or
        cache-annotated content keeps its list form, with ``input_text`` blocks
        rewritten to ``text`` for Anthropic compatibility.
        """
        from langchain_core.messages import (
            AIMessage,
            BaseMessage,
            HumanMessage,
            SystemMessage,
        )

        lc_messages: list[BaseMessage] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle content that might be a list (for multimodal or cache-annotated)
            if isinstance(content, list):
                has_image = any(
                    isinstance(item, dict)
                    and item.get("type") in ("image_url", "image")
                    for item in content
                )
                has_cache_control = any(
                    isinstance(item, dict) and "cache_control" in item
                    for item in content
                )
                if has_image or has_cache_control:
                    # Preserve list structure for multimodal or cache-annotated blocks.
                    # Convert input_text type to text type for Anthropic compatibility.
                    if has_cache_control:
                        content = [
                            {
                                **{k: v for k, v in item.items() if k != "type"},
                                "type": "text",
                            }
                            if isinstance(item, dict)
                            and item.get("type") == "input_text"
                            else item
                            for item in content
                        ]
                else:
                    # Text-only without cache_control: extract text content
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "input_text":
                            text_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = "\n".join(text_parts)

            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        return lc_messages

    @staticmethod
    def _extract_usage(raw_response: Any) -> dict[str, int]:
        """Parse provider usage metadata into normalized token counts.

        Scans both ``usage_metadata`` and ``response_metadata.token_usage`` on
        the raw response, tolerating dict- and attribute-style containers and
        the differing OpenAI/Anthropic field names, and logs any prompt-cache
        hit or miss. Returns zeros when no usage information is present.
        """
        usage_candidates: list[Any] = []
        usage_metadata = getattr(raw_response, "usage_metadata", None)
        if usage_metadata:
            usage_candidates.append(usage_metadata)

        response_metadata = getattr(raw_response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            token_usage = response_metadata.get("token_usage")
            if token_usage:
                usage_candidates.append(token_usage)
        elif response_metadata is not None:
            token_usage = getattr(response_metadata, "token_usage", None)
            if token_usage:
                usage_candidates.append(token_usage)

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        for usage in usage_candidates:
            if isinstance(usage, dict):
                input_tokens = input_tokens or int(
                    usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                )
                output_tokens = output_tokens or int(
                    usage.get("output_tokens") or usage.get("completion_tokens") or 0
                )
                total_tokens = total_tokens or int(usage.get("total_tokens") or 0)
            else:
                input_tokens = input_tokens or int(
                    getattr(usage, "input_tokens", 0)
                    or getattr(usage, "prompt_tokens", 0)
                    or 0
                )
                output_tokens = output_tokens or int(
                    getattr(usage, "output_tokens", 0)
                    or getattr(usage, "completion_tokens", 0)
                    or 0
                )
                total_tokens = total_tokens or int(
                    getattr(usage, "total_tokens", 0) or 0
                )

        if total_tokens <= 0 and (input_tokens > 0 or output_tokens > 0):
            total_tokens = input_tokens + output_tokens

        # Extract cache-specific token counts (Anthropic prompt caching)
        cache_creation_tokens = 0
        cache_read_tokens = 0
        for usage in usage_candidates:
            if isinstance(usage, dict):
                cache_creation_tokens = cache_creation_tokens or int(
                    usage.get("cache_creation_input_tokens", 0) or 0
                )
                cache_read_tokens = cache_read_tokens or int(
                    usage.get("cache_read_input_tokens", 0) or 0
                )
            else:
                cache_creation_tokens = cache_creation_tokens or int(
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                )
                cache_read_tokens = cache_read_tokens or int(
                    getattr(usage, "cache_read_input_tokens", 0) or 0
                )

        if cache_creation_tokens > 0 or cache_read_tokens > 0:
            if cache_read_tokens > 0:
                logger.info(
                    "[CACHE] Hit: %s tokens read from cache, %s written",
                    f"{cache_read_tokens:,}",
                    f"{cache_creation_tokens:,}",
                )
            else:
                logger.info(
                    "[CACHE] Miss: %s tokens written to cache",
                    f"{cache_creation_tokens:,}",
                )

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
        }

    @staticmethod
    def _committed_total_tokens(usage_counts: dict[str, int]) -> int:
        """Daily-budget total that counts prompt-cache tokens at full weight.

        Anthropic's raw usage reports ``cache_creation_input_tokens`` and
        ``cache_read_input_tokens`` SEPARATELY from ``input_tokens`` (which
        excludes them), so ``total_tokens`` (input + output) is short by the
        cache total and we add it back at full weight.

        OpenAI instead folds cached tokens into ``prompt_tokens`` /
        ``total_tokens`` and exposes them under
        ``prompt_tokens_details.cached_tokens`` — a field ``_extract_usage``
        deliberately never sums — so ``cache_creation_tokens`` /
        ``cache_read_tokens`` stay 0 there and no double counting occurs. The
        same holds for langchain-anthropic's ``usage_metadata`` path, which
        already folds cache tokens into ``input_tokens`` while leaving the
        separate cache fields unset.
        """
        total = int(usage_counts.get("total_tokens", 0) or 0)
        cache_creation = int(usage_counts.get("cache_creation_tokens", 0) or 0)
        cache_read = int(usage_counts.get("cache_read_tokens", 0) or 0)
        return total + cache_creation + cache_read

    async def ainvoke_with_structured_output(
        self,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Invoke the model asynchronously with optional structured output.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            json_schema: Optional JSON schema for structured output

        Returns:
            Dict with 'output_text', 'response_data', and 'request_metadata'
        """
        self._ensure_initialized()

        lc_messages = self._to_lc_messages(messages)

        chat_model = self._chat_model
        response_data: dict[str, Any] = {}

        if chat_model is None:
            raise RuntimeError("Chat model not initialized")

        try:
            # Use structured output if schema provided and supported
            if json_schema and self.supports_structured_outputs:
                try:
                    response = await self._invoke_with_schema(lc_messages, json_schema)
                except Exception as e:
                    provider = getattr(self.config, "provider", None)
                    err_msg = str(e)
                    anthropic_schema_limit = provider == "anthropic" and (
                        "Tool schema contains too many conditional branches" in err_msg
                        or "reduce the use of anyOf constructs" in err_msg
                        or "anyOf constructs (limit: 8)" in err_msg
                        or "too many parameters with union types" in err_msg
                        or "parameters with unions" in err_msg
                    )
                    if anthropic_schema_limit:
                        logger.warning(
                            "Anthropic rejected the structured-output schema as "
                            "too complex; falling back to non-strict "
                            "JSON-instructed extraction (prompt-embedded schema "
                            "+ JSON parsing): %s",
                            err_msg,
                        )
                        response_data["structured_output_fallback"] = True
                        response = await chat_model.ainvoke(lc_messages)
                    else:
                        raise
            else:
                response = await chat_model.ainvoke(lc_messages)

            raw_response = response
            parsed_response = None
            parsing_error = None
            if isinstance(response, dict) and "raw" in response:
                raw_response = response.get("raw") or response
                parsed_response = response.get("parsed")
                parsing_error = response.get("parsing_error")

            if parsing_error is not None:
                response_data["parsing_error"] = str(parsing_error)

            if parsed_response is not None and parsing_error is None:
                if isinstance(parsed_response, str):
                    output_text = parsed_response
                else:
                    output_text = json.dumps(parsed_response, ensure_ascii=False)
            else:
                content = (
                    raw_response.content
                    if hasattr(raw_response, "content")
                    else str(raw_response)
                )
                if isinstance(content, list):
                    content = next(
                        (
                            item.get("text", "")
                            for item in content
                            if isinstance(item, dict) and item.get("type") == "text"
                        ),
                        "",
                    ) or str(content)
                output_text = content

            usage_counts = self._extract_usage(raw_response)
            input_tokens = usage_counts["input_tokens"]
            output_tokens = usage_counts["output_tokens"]
            total_tokens = usage_counts["total_tokens"]
            cache_creation_tokens = usage_counts["cache_creation_tokens"]
            cache_read_tokens = usage_counts["cache_read_tokens"]

            if total_tokens > 0 or input_tokens > 0 or output_tokens > 0:
                response_data["usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
                if cache_creation_tokens > 0 or cache_read_tokens > 0:
                    response_data["usage"]["cache_creation_input_tokens"] = (
                        cache_creation_tokens
                    )
                    response_data["usage"]["cache_read_input_tokens"] = (
                        cache_read_tokens
                    )

                # Report to token tracker. The committed number counts cache
                # creation + read tokens at full weight (see
                # _committed_total_tokens) so the daily budget reflects true
                # consumption on cache-enabled providers.
                try:
                    committed_total = self._committed_total_tokens(usage_counts)
                    if committed_total > 0:
                        token_tracker = get_token_tracker()
                        token_tracker.add_tokens(committed_total)
                        logger.debug(
                            f"[TOKEN] API call consumed {committed_total:,} tokens "
                            f"(daily total: {token_tracker.get_tokens_used_today():,})"
                        )
                except Exception as e:
                    logger.warning(f"Error reporting token usage: {e}")

            # Store response metadata
            response_data["model"] = self.config.model
            response_data["provider"] = self.config.provider

            return {
                "output_text": output_text,
                "response_data": response_data,
                "request_metadata": {
                    "messages": messages,
                    "json_schema": json_schema,
                    "provider": self.config.provider,
                    "model": self.config.model,
                },
            }

        except Exception as e:
            logger.debug("Error invoking LLM: %s", e, exc_info=True)
            raise

    async def _invoke_with_schema(
        self,
        messages: list[Any],
        json_schema: dict[str, Any],
    ) -> Any:
        """Invoke model with structured output schema."""
        if self._chat_model is None:
            raise RuntimeError("Chat model not initialized")
        provider = self.config.provider

        # Extract schema definition
        schema_def = json_schema.get("schema", json_schema)
        schema_name = json_schema.get("name", "Response")

        if provider == "openai":
            # Use OpenAI's native structured output
            return await self._invoke_openai_structured(messages, json_schema)

        elif provider == "anthropic":
            # Anthropic: Use with_structured_output
            return await self._invoke_anthropic_structured(
                messages, schema_def, schema_name
            )

        elif provider == "google":
            # Google: Use with_structured_output
            return await self._invoke_google_structured(messages, schema_def)

        elif provider == "openrouter":
            # Models routed through OpenRouter that lack native json_schema
            # and tool-calling support fall back to plain invocation —
            # the schema is already in the prompt, so the model outputs JSON
            # which ChronoMiner parses from the text response.
            m = (self.config.model or "").lower()
            if "gemma" in m or "llama" in m or "mistral" in m:
                return await self._chat_model.ainvoke(messages)
            # OpenAI/Claude/Gemini via OpenRouter: native json_schema
            return await self._invoke_openai_structured(messages, json_schema)

        elif provider == "custom":
            caps = self._get_capabilities()
            if caps.supports_function_calling or caps.supports_structured_outputs:
                return await self._invoke_openai_structured(messages, json_schema)
            return await self._chat_model.ainvoke(messages)

        # Fallback: regular invoke
        return await self._chat_model.ainvoke(messages)

    async def _invoke_openai_structured(
        self,
        messages: list[Any],
        json_schema: dict[str, Any],
    ) -> Any:
        """Invoke OpenAI with native structured output."""
        if self._chat_model is None:
            raise RuntimeError("Chat model not initialized")
        # Build the response_format for OpenAI
        schema_def = json_schema.get("schema", json_schema)
        schema_name = json_schema.get("name", "Response")
        strict = json_schema.get("strict", True)

        # Use bind with response_format
        structured_model = self._chat_model.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema_def,
                    "strict": strict,
                },
            }
        )

        return await structured_model.ainvoke(messages)

    async def _invoke_anthropic_structured(
        self,
        messages: list[Any],
        schema_def: dict[str, Any],
        schema_name: str,
    ) -> Any:
        """Invoke Anthropic with structured output.

        Uses ``method="json_schema"`` (Anthropic native constrained decoding)
        when extended thinking is active, because forced tool calling
        (``tool_choice: any/tool``) is incompatible with the thinking API.
        Falls back to default function-calling when thinking is off.
        """
        if self._chat_model is None:
            raise RuntimeError("Chat model not initialized")

        schema_def = _normalize_schema_for_anthropic(schema_def)
        if isinstance(schema_def, dict):
            schema_def.setdefault("title", schema_name or "Response")
            schema_def.setdefault(
                "description",
                "Return a JSON object that conforms to this schema.",
            )

        # Detect whether extended thinking is enabled on the chat model.
        thinking_cfg = getattr(self._chat_model, "thinking", None)
        thinking_enabled = (
            isinstance(thinking_cfg, dict) and thinking_cfg.get("type") == "enabled"
        )

        if thinking_enabled:
            # Native constrained-decoding path -- avoids forced tool calling,
            # which Anthropic prohibits when thinking is on.
            structured_model = self._chat_model.with_structured_output(
                schema_def,
                method="json_schema",
                include_raw=True,
            )
        else:
            # Standard function-calling path (default behaviour).
            structured_model = self._chat_model.with_structured_output(
                schema_def,
                include_raw=True,
            )

        result = await structured_model.ainvoke(messages)
        return result

    async def _invoke_google_structured(
        self,
        messages: list[Any],
        schema_def: dict[str, Any],
    ) -> Any:
        """Invoke Google Gemini with structured output."""
        if self._chat_model is None:
            raise RuntimeError("Chat model not initialized")
        # Use include_raw=True to get AIMessage with usage_metadata
        structured_model = self._chat_model.with_structured_output(
            schema_def,
            method="json_schema",
            include_raw=True,
        )
        result = await structured_model.ainvoke(messages)
        return result

    async def aclose(self) -> None:
        """Close the underlying provider HTTP client(s).

        LangChain chat models wrap provider SDK clients that in turn hold an
        httpx client; without an explicit close those connections leak. This
        closes every known client attribute defensively (ChatOpenAI exposes
        ``root_client`` / ``root_async_client``; ChatAnthropic exposes
        ``_client`` / ``_async_client``), awaiting async closers. Providers
        that expose none (e.g. Google) are simply skipped. Never raises: any
        close failure is logged at debug level and swallowed.
        """
        chat_model = self._chat_model
        self._chat_model = None
        self._initialized = False
        if chat_model is None:
            return
        for attr in (
            "root_async_client",
            "root_client",
            "async_client",
            "client",
            "_async_client",
            "_client",
        ):
            await _aclose_maybe(getattr(chat_model, attr, None))

    def invoke_with_structured_output(
        self,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synchronous wrapper for ainvoke_with_structured_output."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.ainvoke_with_structured_output(messages, json_schema),
                )
                return future.result()
        return asyncio.run(self.ainvoke_with_structured_output(messages, json_schema))


class LLMProvider:
    """
    Factory class for creating LLM instances.

    Handles provider detection, configuration, and instance caching.
    """

    _instances: dict[str, LangChainLLM] = {}

    @classmethod
    def get_llm(
        cls,
        model_config: dict[str, Any] | None = None,
        provider: ProviderType | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LangChainLLM:
        """
        Get or create an LLM instance.

        Args:
            model_config: Model configuration dict (from config loader)
            provider: Override provider type
            model: Override model name
            **kwargs: Additional config parameters

        Returns:
            LangChainLLM instance
        """
        # Build config from model_config or direct parameters
        if model_config:
            config = ProviderConfig.from_config(model_config, provider)
        else:
            if not model:
                raise ValueError("Either model_config or model must be provided")

            detected_provider = provider or ProviderConfig._detect_provider(model)
            api_key = ProviderConfig._get_api_key(detected_provider)

            config = ProviderConfig(
                provider=detected_provider,
                model=model,
                api_key=api_key,
                **kwargs,
            )

        cache_key = (
            f"{config.provider}:{config.model}"
            f":{config.temperature}:{config.max_tokens}"
            f":{config.reasoning_effort}"
        )

        if cache_key not in cls._instances:
            cls._instances[cache_key] = LangChainLLM(config)

        return cls._instances[cache_key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached LLM instances."""
        cls._instances.clear()


def get_default_provider() -> ProviderType:
    """Get the default provider based on available (possibly remapped) keys."""
    if resolve_api_key("openai"):
        return "openai"
    if resolve_api_key("anthropic"):
        return "anthropic"
    if resolve_api_key("google"):
        return "google"
    if resolve_api_key("openrouter"):
        return "openrouter"
    return "openai"  # Default fallback


def list_available_providers() -> list[ProviderType]:
    """List providers with configured (possibly remapped) API keys."""
    available: list[ProviderType] = []
    if resolve_api_key("openai"):
        available.append("openai")
    if resolve_api_key("anthropic"):
        available.append("anthropic")
    if resolve_api_key("google"):
        available.append("google")
    if resolve_api_key("openrouter"):
        available.append("openrouter")
    return available
