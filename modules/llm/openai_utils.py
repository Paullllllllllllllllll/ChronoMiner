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

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from modules.config.capabilities import detect_capabilities
from modules.config.loader import get_config_loader
from modules.infra.logger import setup_logger
from modules.llm.langchain_provider import (
    LangChainLLM,
    ProviderConfig,
    ProviderType,
)

logger = setup_logger(__name__)


class LLMExtractor:
    """
    A unified wrapper for interacting with LLM providers via LangChain.

    Supports OpenAI, Anthropic, Google Gemini, and OpenRouter through a
    consistent interface for structured data extraction tasks.
    """

    def __init__(
        self,
        api_key: str | None = None,
        prompt_path: Path | None = None,
        model: str = "",
        provider: ProviderType | None = None,
        model_config_override: dict[str, Any] | None = None,
        concurrency_config_override: dict[str, Any] | None = None,
    ) -> None:
        if not model:
            raise ValueError("Model must be specified.")

        self.model: str = model
        self.provider: ProviderType = provider or ProviderConfig._detect_provider(model)

        # Get API key from parameter or environment
        if api_key:
            self.api_key: str = api_key
        else:
            resolved_key = ProviderConfig._get_api_key(self.provider)
            if not resolved_key:
                raise ValueError(f"API key not found for provider {self.provider}")
            self.api_key = resolved_key

        # Load prompt text if path provided
        self.prompt_text: str = ""
        if prompt_path and prompt_path.exists():
            try:
                with prompt_path.open("r", encoding="utf-8") as prompt_file:
                    self.prompt_text = prompt_file.read().strip()
            except Exception as e:
                logger.error(f"Failed to read prompt: {e}")
                raise

        # Load configuration using cached loader
        config = get_config_loader()
        self.model_config: dict[str, Any] = (
            model_config_override or config.get_model_config()
        )
        self.concurrency_config: dict[str, Any] = (
            concurrency_config_override or config.get_concurrency_config()
        )

        tm: dict[str, Any] = self.model_config.get("extraction_model", {})

        # Model parameters
        self.max_output_tokens: int = int(tm.get("max_output_tokens", 4096))
        self.temperature: float = float(tm.get("temperature", 0.0))
        self.top_p: float = float(tm.get("top_p", 1.0))
        self.presence_penalty: float = float(tm.get("presence_penalty", 0.0))
        self.frequency_penalty: float = float(tm.get("frequency_penalty", 0.0))

        # Reasoning / text controls (used for reasoning models)
        self.reasoning: dict[str, Any] = tm.get("reasoning", {"effort": "medium"})
        self.text_params: dict[str, Any] = tm.get("text", {"verbosity": "medium"})

        # Capabilities gating
        self.caps = detect_capabilities(self.model, provider=self.provider)

        # Create LangChain LLM instance
        self._llm: LangChainLLM | None = None
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize the LangChain LLM instance."""
        # Load service_tier from concurrency config (CM-1)
        extraction_cfg = (self.concurrency_config.get("concurrency", {}) or {}).get(
            "extraction", {}
        ) or {}
        service_tier = extraction_cfg.get("service_tier")

        # Build extra_params including reasoning (CM-3) and service_tier (CM-1)
        extra_params: dict[str, Any] = {
            "presence_penalty": self.presence_penalty
            if self.caps.supports_sampler_controls
            else 0.0,
            "frequency_penalty": self.frequency_penalty
            if self.caps.supports_sampler_controls
            else 0.0,
            "reasoning_config": self.reasoning,
            "reasoning_effort": self.reasoning.get("effort", "medium"),
            "text_config": self.text_params,
        }
        if service_tier:
            extra_params["service_tier"] = service_tier

        # Build provider config. max_retries=0 disables LangChain's internal
        # retries so the outer loop in SynchronousProcessingStrategy is the
        # single retry authority (429-aware exponential backoff + jitter).
        config = ProviderConfig(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature
            if self.caps.supports_sampler_controls
            else 0.0,
            max_tokens=self.max_output_tokens,
            top_p=self.top_p if self.caps.supports_sampler_controls else 1.0,
            max_retries=0,
            extra_params=extra_params,
        )

        # Set base URL for OpenRouter and custom endpoints
        if self.provider == "openrouter":
            config.base_url = "https://openrouter.ai/api/v1"
        elif self.provider == "custom":
            tm = self.model_config.get("extraction_model", {})
            custom_cfg = tm.get("custom_endpoint", {})
            config.base_url = custom_cfg.get("base_url")

        self._llm = LangChainLLM(config)

    @property
    def llm(self) -> LangChainLLM:
        """Get the underlying LangChain LLM instance."""
        if self._llm is None:
            self._initialize_llm()
        assert self._llm is not None, "LLM initialization failed"
        return self._llm

    async def close(self) -> None:
        """Clean up resources (no-op for LangChain, kept for API compatibility)."""
        pass


@asynccontextmanager
async def open_extractor(
    api_key: str,
    prompt_path: Path,
    model: str,
    provider: ProviderType | None = None,
    model_config_override: dict[str, Any] | None = None,
    concurrency_config_override: dict[str, Any] | None = None,
) -> AsyncGenerator[LLMExtractor, None]:
    """
    Asynchronous context manager for LLMExtractor.

    :param api_key: API key for the provider.
    :param prompt_path: Path to the prompt file.
    :param model: Model name.
    :param provider: Optional provider type override.
    :yield: An instance of LLMExtractor.
    """
    extractor = LLMExtractor(
        api_key=api_key,
        prompt_path=prompt_path,
        model=model,
        provider=provider,
        model_config_override=model_config_override,
        concurrency_config_override=concurrency_config_override,
    )
    try:
        yield extractor
    finally:
        await extractor.close()


def _build_messages(
    system_message: str | None,
    user_blocks: list[dict[str, Any]],
    *,
    extractor: LLMExtractor,
    enable_cache_control: bool,
    context_image_data: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Assemble the system and user messages in the provider-agnostic format.

    Builds the system block (with optional Anthropic ``cache_control``),
    injects an optional context image ahead of the caller's ``user_blocks``
    (OpenAI and OpenRouter only), and wraps both into the role envelope.

    :param system_message: Optional system prompt; treated as empty when None.
    :param user_blocks: The caller's trailing user-content blocks (text chunk,
        or instruction plus image), appended after any context image.
    :param extractor: The active extractor, used for provider gating.
    :param enable_cache_control: Whether to mark the system block ephemeral.
    :param context_image_data: Optional context image dict with 'base64',
        'mime_type', and 'detail' keys.
    :return: The two-element system/user message list.
    """
    system_content: list[dict[str, Any]] = [
        {"type": "input_text", "text": system_message or ""}
    ]
    if enable_cache_control:
        system_content[-1]["cache_control"] = {"type": "ephemeral"}

    user_content: list[dict[str, Any]] = []
    if context_image_data is not None:
        if extractor.provider.lower() in ("openai", "openrouter"):
            from modules.images.message_builder import build_image_content_block

            ctx_block = build_image_content_block(
                image_base64=context_image_data["base64"],
                mime_type=context_image_data["mime_type"],
                provider=extractor.provider,
                detail=context_image_data.get("detail"),
                supports_image_detail=extractor.caps.supports_image_detail,
            )
            user_content.append({"type": "text", "text": "Context image:"})
            user_content.append(ctx_block)
        else:
            logger.warning(
                "Context image injection not yet supported for "
                f"provider '{extractor.provider}'. Skipping."
            )
    user_content.extend(user_blocks)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _normalize_structured_schema(
    json_schema: dict[str, Any] | None, caps: Any
) -> dict[str, Any] | None:
    """
    Coerce a wrapped or bare JSON schema into the structured-output form.

    Accepts both the wrapped ``{"name", "schema", "strict"}`` shape and a
    bare JSON Schema. Returns None when no schema is supplied or the model
    does not support structured outputs.
    """
    if not json_schema or not caps.supports_structured_outputs:
        return None
    if "schema" in json_schema and isinstance(json_schema.get("schema"), dict):
        return {
            "name": json_schema.get("name", "TranscriptionSchema"),
            "schema": json_schema.get("schema", {}),
            "strict": bool(json_schema.get("strict", True)),
        }
    return {
        "name": "TranscriptionSchema",
        "schema": json_schema,
        "strict": True,
    }


def _pack_result(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the public response fields from a raw LLM result."""
    response_data = result.get("response_data", {})
    return {
        "output_text": result.get("output_text", ""),
        "response_data": response_data,
        "request_metadata": result.get("request_metadata", {}),
        "usage": response_data.get("usage", {}),
    }


async def process_text_chunk(
    text_chunk: str,
    extractor: LLMExtractor,
    system_message: str | None = None,
    json_schema: dict | None = None,
    enable_cache_control: bool = False,
    context_image_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Process a text chunk using LangChain with the configured provider.

    :param text_chunk: The text to process.
    :param extractor: An instance of LLMExtractor.
    :param system_message: Optional system message.
    :param json_schema: Optional JSON schema for response formatting.
    :param context_image_data: Optional context image dict with 'base64',
        'mime_type', and 'detail' keys. Injected into the user message
        before the text chunk (OpenAI only).
    :return: Dictionary containing the model output text, raw response payload,
        and request metadata used for the call.
    :raises Exception: If the API call fails after all LangChain retries.
    """
    user_blocks: list[dict[str, Any]] = [{"type": "input_text", "text": text_chunk}]
    messages = _build_messages(
        system_message,
        user_blocks,
        extractor=extractor,
        enable_cache_control=enable_cache_control,
        context_image_data=context_image_data,
    )
    structured_schema = _normalize_structured_schema(json_schema, extractor.caps)

    # LangChain handles retries internally via max_retries; token tracking is
    # handled via usage_metadata on the response.
    result = await extractor.llm.ainvoke_with_structured_output(
        messages=messages,
        json_schema=structured_schema,
    )
    return _pack_result(result)


async def process_image_chunk(
    image_base64: str,
    mime_type: str,
    extractor: LLMExtractor,
    system_message: str | None = None,
    json_schema: dict | None = None,
    image_detail: str | None = None,
    user_instruction: str = (
        "Extract structured data from this image according to the schema."
    ),
    enable_cache_control: bool = False,
    context_image_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Process a single image using LangChain with the configured provider.

    Constructs multimodal messages with the image content block and sends
    them through the same LLM pipeline as text chunks.

    :param image_base64: Base64-encoded image data.
    :param mime_type: MIME type of the image (e.g., 'image/jpeg').
    :param extractor: An instance of LLMExtractor.
    :param system_message: Optional system message.
    :param json_schema: Optional JSON schema for response formatting.
    :param image_detail: Image detail level ('low', 'high', 'auto', 'original').
    :param user_instruction: Text instruction accompanying the image.
    :param context_image_data: Optional context image dict with 'base64',
        'mime_type', and 'detail' keys. Injected into the user message
        before the main image (OpenAI only).
    :return: Dictionary containing the model output text, raw response payload,
        and request metadata used for the call.
    """
    from modules.images.message_builder import build_image_content_block

    # Build provider-specific image content block
    image_block = build_image_content_block(
        image_base64=image_base64,
        mime_type=mime_type,
        provider=extractor.provider,
        detail=image_detail,
        supports_image_detail=extractor.caps.supports_image_detail,
    )
    user_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": user_instruction},
        image_block,
    ]
    messages = _build_messages(
        system_message,
        user_blocks,
        extractor=extractor,
        enable_cache_control=enable_cache_control,
        context_image_data=context_image_data,
    )
    structured_schema = _normalize_structured_schema(json_schema, extractor.caps)

    result = await extractor.llm.ainvoke_with_structured_output(
        messages=messages,
        json_schema=structured_schema,
    )
    return _pack_result(result)


async def process_text_chunk_with_provider(
    text_chunk: str,
    system_message: str,
    json_schema: dict | None = None,
    model: str | None = None,
    provider: ProviderType | None = None,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        model = model_config.get("extraction_model", {}).get("name", "")

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
        prompt_path=Path("prompts/text_extraction_prompt.txt"),
        model=model,
        provider=detected_provider,
    ) as extractor:
        return await process_text_chunk(
            text_chunk=text_chunk,
            extractor=extractor,
            system_message=system_message,
            json_schema=json_schema,
        )
