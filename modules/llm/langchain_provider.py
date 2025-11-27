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

from modules.config.loader import ConfigLoader
from modules.core.logger import setup_logger
from modules.core.token_tracker import get_token_tracker

logger = setup_logger(__name__)

ProviderType = Literal["openai", "anthropic", "google", "openrouter"]


def _load_concurrency_config() -> Dict[str, Any]:
    """Load concurrency config for retry and rate limiting settings."""
    try:
        config_loader = ConfigLoader()
        config_loader.load_configs()
        return config_loader.get_concurrency_config() or {}
    except Exception:
        return {}


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
        
        # Detect provider from model name if not overridden
        provider = provider_override or cls._detect_provider(model_name)
        
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
            extra_params={
                "presence_penalty": float(tm.get("presence_penalty", 0.0)),
                "frequency_penalty": float(tm.get("frequency_penalty", 0.0)),
            },
        )
    
    @staticmethod
    def _detect_provider(model_name: str) -> ProviderType:
        """Detect provider from model name."""
        m = model_name.lower().strip()
        
        # OpenRouter models typically have openrouter/ prefix or contain /
        if m.startswith("openrouter/") or "/" in m:
            return "openrouter"
        
        # Anthropic models
        if m.startswith("claude") or "anthropic" in m:
            return "anthropic"
        
        # Google models
        if m.startswith("gemini") or "google" in m:
            return "google"
        
        # Default to OpenAI
        return "openai"
    
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
            
            return ChatAnthropic(
                model=self.config.model,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                **common_params,
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
            
            return ChatOpenAI(
                model=model_name,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                **common_params,
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _is_reasoning_model(self) -> bool:
        """
        Check if the model is a reasoning model (o1, o3, gpt-5).
        
        DEPRECATED: Use _get_capabilities().is_reasoning_model instead.
        This method is kept for backward compatibility.
        """
        caps = self._get_capabilities()
        return caps.is_reasoning_model
    
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
                response = await self._invoke_with_schema(lc_messages, json_schema)
            else:
                response = await chat_model.ainvoke(lc_messages)
            
            # Extract content
            output_text = response.content if hasattr(response, 'content') else str(response)
            
            # Track token usage
            usage_metadata = getattr(response, 'usage_metadata', None)
            if usage_metadata:
                response_data["usage"] = {
                    "input_tokens": getattr(usage_metadata, 'input_tokens', 0),
                    "output_tokens": getattr(usage_metadata, 'output_tokens', 0),
                    "total_tokens": getattr(usage_metadata, 'total_tokens', 0),
                }
                
                # Report to token tracker
                try:
                    total = response_data["usage"].get("total_tokens", 0)
                    if total:
                        token_tracker = get_token_tracker()
                        token_tracker.add_tokens(total)
                        logger.debug(
                            f"[TOKEN] API call consumed {total:,} tokens "
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
        # Create a dynamic Pydantic model from schema
        structured_model = self._chat_model.with_structured_output(
            schema_def,
            method="json_mode",
        )
        return await structured_model.ainvoke(messages)
    
    async def _invoke_google_structured(
        self,
        messages: List,
        schema_def: Dict[str, Any],
    ):
        """Invoke Google Gemini with structured output."""
        structured_model = self._chat_model.with_structured_output(
            schema_def,
            method="json_schema",
        )
        return await structured_model.ainvoke(messages)
    
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
