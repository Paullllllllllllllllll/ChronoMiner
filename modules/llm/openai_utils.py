# modules/openai_utils.py

"""
LLM utilities for text processing with structured outputs.

This module provides a unified interface for LLM interactions using LangChain
as the backend. It supports multiple providers:
- OpenAI (default)
- Anthropic (Claude)
- Google (Gemini)
- OpenRouter (multi-provider access)

LangChain handles retries, error classification, and structured outputs internally.
"""

from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from modules.config.loader import get_config_loader
from modules.core.logger import setup_logger
from modules.llm.model_capabilities import detect_capabilities
from modules.llm.langchain_provider import (
    LangChainLLM,
    ProviderConfig,
    ProviderType,
)

logger = setup_logger(__name__)


def _load_retry_config() -> int:
    """
    Load retry attempts from concurrency configuration.
    
    This value is passed to LangChain's max_retries parameter.
    Uses cached config loader for efficiency.
    """
    try:
        config = get_config_loader()
        cc = config.get_concurrency_config() or {}
        extraction_cfg = (cc.get("concurrency", {}) or {}).get("extraction", {}) or {}
        retry_cfg = (extraction_cfg.get("retry", {}) or {})
        attempts = int(retry_cfg.get("attempts", 5))
        return max(1, attempts)
    except Exception:
        return 5


# Load retry config for LangChain's max_retries
_MAX_RETRIES = _load_retry_config()


class LLMExtractor:
    """
    A unified wrapper for interacting with LLM providers via LangChain.
    
    Supports OpenAI, Anthropic, Google Gemini, and OpenRouter through a
    consistent interface for structured data extraction tasks.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        prompt_path: Optional[Path] = None,
        model: str = "",
        provider: Optional[ProviderType] = None,
    ) -> None:
        if not model:
            raise ValueError("Model must be specified.")
        
        self.model: str = model
        self.provider: ProviderType = provider or ProviderConfig._detect_provider(model)
        
        # Get API key from parameter or environment
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = ProviderConfig._get_api_key(self.provider)
        
        if not self.api_key:
            raise ValueError(f"API key not found for provider {self.provider}")
        
        # Load prompt text if path provided
        self.prompt_text: str = ""
        if prompt_path and prompt_path.exists():
            try:
                with prompt_path.open('r', encoding='utf-8') as prompt_file:
                    self.prompt_text = prompt_file.read().strip()
            except Exception as e:
                logger.error(f"Failed to read prompt: {e}")
                raise
        
        # Load configuration using cached loader
        config = get_config_loader()
        self.model_config: Dict[str, Any] = config.get_model_config()
        self.concurrency_config: Dict[str, Any] = config.get_concurrency_config()
        
        tm: Dict[str, Any] = self.model_config.get("transcription_model", {})
        
        # Model parameters
        self.max_output_tokens: int = int(tm.get("max_output_tokens", 4096))
        self.temperature: float = float(tm.get("temperature", 0.0))
        self.top_p: float = float(tm.get("top_p", 1.0))
        self.presence_penalty: float = float(tm.get("presence_penalty", 0.0))
        self.frequency_penalty: float = float(tm.get("frequency_penalty", 0.0))
        
        # Reasoning / text controls (used for reasoning models)
        self.reasoning: Dict[str, Any] = tm.get("reasoning", {"effort": "medium"})
        self.text_params: Dict[str, Any] = tm.get("text", {"verbosity": "medium"})
        
        # Capabilities gating
        self.caps = detect_capabilities(self.model)
        
        # Create LangChain LLM instance
        self._llm: Optional[LangChainLLM] = None
        self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """Initialize the LangChain LLM instance."""
        # Build provider config
        config = ProviderConfig(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature if self.caps.supports_sampler_controls else 0.0,
            max_tokens=self.max_output_tokens,
            top_p=self.top_p if self.caps.supports_sampler_controls else 1.0,
            extra_params={
                "presence_penalty": self.presence_penalty if self.caps.supports_sampler_controls else 0.0,
                "frequency_penalty": self.frequency_penalty if self.caps.supports_sampler_controls else 0.0,
            },
        )
        
        # Set base URL for OpenRouter
        if self.provider == "openrouter":
            config.base_url = "https://openrouter.ai/api/v1"
        
        self._llm = LangChainLLM(config)
    
    @property
    def llm(self) -> LangChainLLM:
        """Get the underlying LangChain LLM instance."""
        if self._llm is None:
            self._initialize_llm()
        return self._llm
    
    async def close(self) -> None:
        """Clean up resources (no-op for LangChain, kept for API compatibility)."""
        pass


@asynccontextmanager
async def open_extractor(
    api_key: str,
    prompt_path: Path,
    model: str,
    provider: Optional[ProviderType] = None,
) -> AsyncGenerator[LLMExtractor, None]:
    """
    Asynchronous context manager for LLMExtractor.

    :param api_key: API key for the provider.
    :param prompt_path: Path to the prompt file.
    :param model: Model name.
    :param provider: Optional provider type override.
    :yield: An instance of LLMExtractor.
    """
    extractor = LLMExtractor(api_key, prompt_path, model, provider)
    try:
        yield extractor
    finally:
        await extractor.close()


async def process_text_chunk(
    text_chunk: str,
    extractor: LLMExtractor,
    system_message: Optional[str] = None,
    json_schema: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Process a text chunk using LangChain with the configured provider.

    :param text_chunk: The text to process.
    :param extractor: An instance of LLMExtractor.
    :param system_message: Optional system message.
    :param json_schema: Optional JSON schema for response formatting.
    :return: Dictionary containing the model output text, raw response payload,
        and request metadata used for the call.
    :raises Exception: If the API call fails after all LangChain retries.
    """
    if system_message is None:
        system_message = ""
    
    # Build messages in LangChain format
    messages = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": text_chunk}],
        },
    ]
    
    # Build structured output schema if provided
    # NOTE: LangChain's with_structured_output() handles schema validation internally
    structured_schema = None
    if json_schema and extractor.caps.supports_structured_outputs:
        # Handle both wrapped {"name", "schema", "strict"} and bare JSON Schema formats
        if "schema" in json_schema and isinstance(json_schema.get("schema"), dict):
            structured_schema = {
                "name": json_schema.get("name", "TranscriptionSchema"),
                "schema": json_schema.get("schema", {}),
                "strict": bool(json_schema.get("strict", True)),
            }
        elif isinstance(json_schema, dict) and json_schema:
            structured_schema = {
                "name": "TranscriptionSchema",
                "schema": json_schema,
                "strict": True,
            }
    
    # LangChain handles retries internally via max_retries parameter
    # Token tracking is handled via usage_metadata on the response
    result = await extractor.llm.ainvoke_with_structured_output(
        messages=messages,
        json_schema=structured_schema,
    )
    
    return {
        "output_text": result.get("output_text", ""),
        "response_data": result.get("response_data", {}),
        "request_metadata": result.get("request_metadata", {}),
        "usage": result.get("response_data", {}).get("usage", {}),
    }


async def process_text_chunk_with_provider(
    text_chunk: str,
    system_message: str,
    json_schema: Optional[dict] = None,
    model: Optional[str] = None,
    provider: Optional[ProviderType] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process a text chunk with explicit provider selection.
    
    This is a convenience function for one-off calls without managing
    an extractor context.

    :param text_chunk: The text to process.
    :param system_message: System message for the LLM.
    :param json_schema: Optional JSON schema for response formatting.
    :param model: Model name (uses config default if not specified).
    :param provider: Provider type (auto-detected from model if not specified).
    :param model_config: Optional model configuration dict.
    :return: Dictionary containing the model output text and metadata.
    """
    # Load config if not provided (uses cached loader)
    if model_config is None:
        model_config = get_config_loader().get_model_config()
    
    # Get model from config if not specified
    if model is None:
        model = model_config.get("transcription_model", {}).get("name", "")
    
    if not model:
        raise ValueError("Model must be specified either directly or in config")
    
    # Get API key based on provider
    detected_provider = provider or ProviderConfig._detect_provider(model)
    api_key = ProviderConfig._get_api_key(detected_provider)
    
    if not api_key:
        raise ValueError(f"API key not found for provider {detected_provider}")
    
    # Create extractor and process
    async with open_extractor(
        api_key=api_key,
        prompt_path=Path("prompts/structured_output_prompt.txt"),
        model=model,
        provider=detected_provider,
    ) as extractor:
        return await process_text_chunk(
            text_chunk=text_chunk,
            extractor=extractor,
            system_message=system_message,
            json_schema=json_schema,
        )
