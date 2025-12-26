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
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Type, Union

from pydantic import BaseModel

from modules.config.loader import get_config_loader
from modules.core.logger import setup_logger
from modules.core.token_tracker import get_token_tracker
from modules.llm.model_capabilities import detect_provider as _canonical_detect_provider

logger = setup_logger(__name__)

ProviderType = Literal["openai", "anthropic", "google", "openrouter"]


def _normalize_schema_for_anthropic(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_normalize_schema_for_anthropic(item) for item in schema]

    if not isinstance(schema, dict):
        return schema

    type_val = schema.get("type")
    if isinstance(type_val, list):
        types = [t for t in type_val if t != "null"]
        has_null = any(t == "null" for t in type_val)
        any_of: List[Dict[str, Any]] = []

        for t in types:
            variant: Dict[str, Any] = {}
            for key, value in schema.items():
                if key == "type":
                    continue
                variant[key] = _normalize_schema_for_anthropic(value)
            variant["type"] = t
            any_of.append(variant)

        if has_null:
            any_of.append({"type": "null"})

        normalized: Dict[str, Any] = {}
        for meta_key in ("title", "description", "default", "examples"):
            if meta_key in schema:
                normalized[meta_key] = _normalize_schema_for_anthropic(schema[meta_key])
        normalized["anyOf"] = any_of
        return normalized

    normalized = {}
    for key, value in schema.items():
        normalized[key] = _normalize_schema_for_anthropic(value)
    return normalized


def _load_concurrency_config() -> Dict[str, Any]:
    """Load concurrency config for retry and rate limiting settings."""
    try:
        return get_config_loader().get_concurrency_config() or {}
    except Exception:
        return {}


def _compute_openrouter_reasoning_max_tokens(max_tokens: int, effort: str) -> int:
    """
    Compute reasoning token budget for OpenRouter models based on effort level.
    
    For Anthropic and Gemini thinking models via OpenRouter, we map effort levels
    to a reasoning.max_tokens budget. This mirrors ChronoTranscriber's approach.
    
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
        "low": 0.1,      # 10% of output budget for reasoning
        "medium": 0.25,  # 25% of output budget for reasoning  
        "high": 0.5,     # 50% of output budget for reasoning
        "none": 0.0,     # Disable reasoning
    }
    
    ratio = ratios.get(effort_lower, 0.25)  # Default to medium
    if ratio <= 0:
        return 0
    
    # Compute budget with reasonable bounds
    budget = int(max_tokens * ratio)
    # Minimum 1024 tokens for meaningful reasoning, max 32768
    return max(1024, min(budget, 32768))


def _build_openrouter_reasoning_payload(
    model_name: str,
    reasoning_config: Dict[str, Any],
    max_tokens: int,
) -> Optional[Dict[str, Any]]:
    """
    Build OpenRouter-compatible reasoning payload based on model type.
    
    OpenRouter accepts a top-level `reasoning` object and routes/translates
    it when supported by the selected model/provider.
    
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
    reasoning_payload: Dict[str, Any] = {}
    
    effort = reasoning_config.get("effort")
    if effort:
        reasoning_payload["effort"] = str(effort)
    
    # Handle explicit max_tokens override
    max_reasoning_tokens = reasoning_config.get("max_tokens")
    if max_reasoning_tokens is not None:
        try:
            reasoning_payload["max_tokens"] = int(max_reasoning_tokens)
        except (ValueError, TypeError):
            pass
    
    # Handle exclude flag
    exclude = reasoning_config.get("exclude")
    if exclude is not None:
        reasoning_payload["exclude"] = bool(exclude)
    
    # Handle enabled flag
    enabled = reasoning_config.get("enabled")
    if enabled is not None:
        reasoning_payload["enabled"] = bool(enabled)
    
    if not reasoning_payload:
        return None
    
    # Provider-specific translations
    
    # For Anthropic and Gemini thinking models, map effort -> max_tokens budget
    if ("anthropic/" in m or "claude" in m or "gemini" in m) and "max_tokens" not in reasoning_payload:
        eff = reasoning_payload.get("effort", "medium")
        budget = _compute_openrouter_reasoning_max_tokens(max_tokens=max_tokens, effort=str(eff))
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
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout: float = 600.0
    max_retries: int = 5
    requests_per_second: Optional[float] = None  # For rate limiting
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_config(
        cls,
        model_config: Dict[str, Any],
        provider_override: Optional[ProviderType] = None,
        concurrency_config: Optional[Dict[str, Any]] = None,
    ) -> "ProviderConfig":
        """Create ProviderConfig from model_config dictionary."""
        tm = model_config.get("transcription_model", {})
        model_name = tm.get("name", "")
        
        # Check for explicit provider field in config, then override, then auto-detect
        config_provider = tm.get("provider")
        if provider_override:
            provider = provider_override
        elif config_provider:
            # Normalize provider name
            provider = config_provider.lower().strip()
            if provider not in ("openai", "anthropic", "google", "openrouter"):
                logger.warning(f"Unknown provider '{config_provider}', auto-detecting from model name")
                provider = cls._detect_provider(model_name)
        else:
            provider = cls._detect_provider(model_name)
        
        # Get API key based on provider
        api_key = cls._get_api_key(provider)
        
        # Get base URL for OpenRouter
        base_url = None
        if provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        
        # Load retry settings from concurrency config
        # LangChain handles retries internally via max_retries parameter
        if concurrency_config is None:
            concurrency_config = _load_concurrency_config()
        extraction_cfg = (concurrency_config.get("concurrency", {}) or {}).get("extraction", {}) or {}
        retry_cfg = extraction_cfg.get("retry", {}) or {}
        max_retries = max(1, int(retry_cfg.get("attempts", 5)))
        
        # Get timeout from config
        timeouts_cfg = extraction_cfg.get("timeouts", {}) or {}
        timeout = float(timeouts_cfg.get("total", 600.0))
        
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
        return provider  # type: ignore[return-value]
    
    @staticmethod
    def _get_api_key(provider: ProviderType) -> Optional[str]:
        """Get API key for the specified provider."""
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_var = key_mapping.get(provider)
        if env_var:
            return os.getenv(env_var, "").strip() or None
        return None


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
    
    def __init__(self, config: ProviderConfig):
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
    
    def _get_capabilities(self):
        """Get model capabilities using our detection logic."""
        if self._capabilities is None:
            from modules.llm.model_capabilities import detect_capabilities
            self._capabilities = detect_capabilities(self.config.model)
        return self._capabilities
    
    def _get_disabled_params(self) -> Dict[str, Any]:
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
        disabled = {}
        
        # Reasoning models don't support sampler controls
        if not caps.supports_sampler_controls:
            disabled["temperature"] = None
            disabled["top_p"] = None
            disabled["presence_penalty"] = None
            disabled["frequency_penalty"] = None
            logger.debug(f"Model {self.config.model}: Disabled sampler controls (reasoning model)")
        
        # Some models don't support structured outputs via response_format
        if not caps.supports_structured_outputs:
            disabled["response_format"] = None
            logger.debug(f"Model {self.config.model}: Disabled response_format (not supported)")
        
        return disabled if disabled else None
    
    def _create_chat_model(self):
        """
        Create the appropriate LangChain chat model based on provider.
        
        Uses disabled_params for capability guarding - LangChain will automatically
        filter out parameters that are incompatible with the model.
        """
        provider = self.config.provider
        caps = self._get_capabilities()
        
        # Get disabled params for capability guarding
        disabled_params = self._get_disabled_params()
        
        # Only include sampler controls if supported by model
        common_params = {}
        if caps.supports_sampler_controls:
            common_params["temperature"] = self.config.temperature
            common_params["max_tokens"] = self.config.max_tokens
            common_params["top_p"] = self.config.top_p
            if self.config.extra_params.get("presence_penalty"):
                common_params["presence_penalty"] = self.config.extra_params["presence_penalty"]
            if self.config.extra_params.get("frequency_penalty"):
                common_params["frequency_penalty"] = self.config.extra_params["frequency_penalty"]
        else:
            # For reasoning models, only set max_tokens (required for output budget)
            common_params["max_tokens"] = self.config.max_tokens
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            
            params = {
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
                params["reasoning_effort"] = self.config.extra_params.get("reasoning_effort", "medium")
            
            return ChatOpenAI(**params)
        
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            anthropic_params = dict(common_params)
            if "temperature" in anthropic_params and "top_p" in anthropic_params:
                if anthropic_params.get("temperature") is not None:
                    anthropic_params.pop("top_p", None)
                else:
                    anthropic_params.pop("temperature", None)

            return ChatAnthropic(
                model=self.config.model,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                **anthropic_params,
            )
        
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            return ChatGoogleGenerativeAI(
                model=self.config.model,
                google_api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                **common_params,
            )
        
        elif provider == "openrouter":
            from langchain_openai import ChatOpenAI
            
            # OpenRouter uses OpenAI-compatible API
            model_name = self.config.model
            if model_name.startswith("openrouter/"):
                model_name = model_name[len("openrouter/"):]
            
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
            
            # Build OpenRouter reasoning payload if supported
            if caps.supports_reasoning_effort:
                reasoning_config = self.config.extra_params.get("reasoning_config", {})
                reasoning_payload = _build_openrouter_reasoning_payload(
                    model_name=model_name,
                    reasoning_config=reasoning_config,
                    max_tokens=self.config.max_tokens,
                )
                if reasoning_payload:
                    # OpenRouter expects reasoning in extra_body
                    params["model_kwargs"] = {"extra_body": {"reasoning": reasoning_payload}}
                    logger.info(f"Using OpenRouter reasoning={reasoning_payload} for model {model_name}")
            
            # OpenRouter-specific headers
            params["default_headers"] = {
                "HTTP-Referer": "https://github.com/ChronoMiner",
                "X-Title": "ChronoMiner",
            }
            
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
        return caps.supports_structured_outputs
    
    async def ainvoke_with_structured_output(
        self,
        messages: List[Dict[str, Any]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the model asynchronously with optional structured output.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            json_schema: Optional JSON schema for structured output
            
        Returns:
            Dict with 'output_text', 'response_data', and 'request_metadata'
        """
        self._ensure_initialized()
        
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        # Convert message dicts to LangChain message objects
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle content that might be a list (for multimodal)
            if isinstance(content, list):
                # Extract text content
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
        
        chat_model = self._chat_model
        response_data: Dict[str, Any] = {}
        
        try:
            # Use structured output if schema provided and supported
            if json_schema and self.supports_structured_outputs:
                try:
                    response = await self._invoke_with_schema(lc_messages, json_schema)
                except Exception as e:
                    provider = getattr(self.config, "provider", None)
                    msg = str(e)
                    anthropic_schema_limit = (
                        provider == "anthropic"
                        and (
                            "Tool schema contains too many conditional branches" in msg
                            or "reduce the use of anyOf constructs" in msg
                            or "anyOf constructs (limit: 8)" in msg
                        )
                    )
                    if anthropic_schema_limit:
                        logger.warning(
                            "Anthropic structured output schema too complex; falling back to plain invocation: %s",
                            msg,
                        )
                        response_data["structured_output_fallback"] = True
                        response = await chat_model.ainvoke(lc_messages)
                    else:
                        raise
            else:
                response = await chat_model.ainvoke(lc_messages)
            
            # Extract content
            output_text = response.content if hasattr(response, 'content') else str(response)
            
            # Track token usage - usage_metadata can be dict or object
            usage_metadata = getattr(response, 'usage_metadata', None)
            if usage_metadata:
                # Handle both dict and object access patterns
                if isinstance(usage_metadata, dict):
                    input_tokens = usage_metadata.get('input_tokens', 0)
                    output_tokens = usage_metadata.get('output_tokens', 0)
                    total_tokens = usage_metadata.get('total_tokens', 0)
                else:
                    input_tokens = getattr(usage_metadata, 'input_tokens', 0)
                    output_tokens = getattr(usage_metadata, 'output_tokens', 0)
                    total_tokens = getattr(usage_metadata, 'total_tokens', 0)
                
                response_data["usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
                
                # Report to token tracker
                try:
                    if total_tokens:
                        token_tracker = get_token_tracker()
                        token_tracker.add_tokens(total_tokens)
                        logger.debug(
                            f"[TOKEN] API call consumed {total_tokens:,} tokens "
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
            logger.error(f"Error invoking LLM: {e}", exc_info=True)
            raise
    
    async def _invoke_with_schema(
        self,
        messages: List,
        json_schema: Dict[str, Any],
    ):
        """Invoke model with structured output schema."""
        provider = self.config.provider
        
        # Extract schema definition
        schema_def = json_schema.get("schema", json_schema)
        schema_name = json_schema.get("name", "Response")
        
        if provider == "openai":
            # Use OpenAI's native structured output
            return await self._invoke_openai_structured(messages, json_schema)
        
        elif provider == "anthropic":
            # Anthropic: Use with_structured_output
            return await self._invoke_anthropic_structured(messages, schema_def, schema_name)
        
        elif provider == "google":
            # Google: Use with_structured_output
            return await self._invoke_google_structured(messages, schema_def)
        
        elif provider == "openrouter":
            # OpenRouter: Use OpenAI-compatible structured output
            return await self._invoke_openai_structured(messages, json_schema)
        
        # Fallback: regular invoke
        return await self._chat_model.ainvoke(messages)
    
    async def _invoke_openai_structured(
        self,
        messages: List,
        json_schema: Dict[str, Any],
    ):
        """Invoke OpenAI with native structured output."""
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
                }
            }
        )
        
        return await structured_model.ainvoke(messages)
    
    async def _invoke_anthropic_structured(
        self,
        messages: List,
        schema_def: Dict[str, Any],
        schema_name: str,
    ):
        """Invoke Anthropic with structured output via tool calling."""
        # Anthropic uses tool calling for structured output
        # Use include_raw=True to get AIMessage with usage_metadata
        schema_def = _normalize_schema_for_anthropic(schema_def)
        if isinstance(schema_def, dict):
            schema_def.setdefault("title", schema_name or "Response")
            schema_def.setdefault(
                "description",
                "Return a JSON object that conforms to this schema.",
            )
        structured_model = self._chat_model.with_structured_output(
            schema_def,
            include_raw=True,
        )
        result = await structured_model.ainvoke(messages)
        # include_raw returns {"raw": AIMessage, "parsed": dict, "parsing_error": ...}
        if isinstance(result, dict) and "raw" in result:
            return result["raw"]
        return result
    
    async def _invoke_google_structured(
        self,
        messages: List,
        schema_def: Dict[str, Any],
    ):
        """Invoke Google Gemini with structured output."""
        # Use include_raw=True to get AIMessage with usage_metadata
        structured_model = self._chat_model.with_structured_output(
            schema_def,
            method="json_schema",
            include_raw=True,
        )
        result = await structured_model.ainvoke(messages)
        # include_raw returns {"raw": AIMessage, "parsed": dict, "parsing_error": ...}
        if isinstance(result, dict) and "raw" in result:
            return result["raw"]
        return result
    
    def invoke_with_structured_output(
        self,
        messages: List[Dict[str, Any]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for ainvoke_with_structured_output."""
        return asyncio.run(self.ainvoke_with_structured_output(messages, json_schema))


class LLMProvider:
    """
    Factory class for creating LLM instances.
    
    Handles provider detection, configuration, and instance caching.
    """
    
    _instances: Dict[str, LangChainLLM] = {}
    
    @classmethod
    def get_llm(
        cls,
        model_config: Optional[Dict[str, Any]] = None,
        provider: Optional[ProviderType] = None,
        model: Optional[str] = None,
        **kwargs,
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
        
        # Override model if specified
        if model:
            config.model = model
        
        # Override provider if specified
        if provider:
            config.provider = provider
            config.api_key = ProviderConfig._get_api_key(provider)
        
        # Create cache key
        cache_key = f"{config.provider}:{config.model}"
        
        # Return cached instance or create new one
        if cache_key not in cls._instances:
            cls._instances[cache_key] = LangChainLLM(config)
        
        return cls._instances[cache_key]
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached LLM instances."""
        cls._instances.clear()


def get_default_provider() -> ProviderType:
    """Get the default provider based on available API keys."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "openai"  # Default fallback


def list_available_providers() -> List[ProviderType]:
    """List providers with configured API keys."""
    available = []
    if os.getenv("OPENAI_API_KEY"):
        available.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        available.append("anthropic")
    if os.getenv("GOOGLE_API_KEY"):
        available.append("google")
    if os.getenv("OPENROUTER_API_KEY"):
        available.append("openrouter")
    return available
