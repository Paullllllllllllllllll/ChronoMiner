"""
ChronoMiner LLM Module.

Provides LLM provider abstractions, model capabilities detection, and batch processing.
"""

from modules.llm.langchain_provider import (
    LangChainLLM,
    LLMProvider,
    ProviderConfig,
    get_default_provider,
)
from modules.llm.model_capabilities import (
    Capabilities,
    detect_capabilities,
    detect_provider,
)
from modules.llm.openai_utils import (
    LLMExtractor,
    open_extractor,
    process_text_chunk,
)
from modules.llm.prompt_utils import (
    load_prompt_template,
    render_prompt_with_schema,
)

__all__ = [
    "LangChainLLM",
    "LLMProvider",
    "ProviderConfig",
    "get_default_provider",
    "Capabilities",
    "detect_capabilities",
    "detect_provider",
    "LLMExtractor",
    "open_extractor",
    "process_text_chunk",
    "load_prompt_template",
    "render_prompt_with_schema",
]
