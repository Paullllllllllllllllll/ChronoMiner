"""
ChronoMiner LLM Module.

Provides the LLM provider abstraction (LangChain-backed), extraction
primitives (payload builder, response parser), prompt template helpers,
and the structured-output schema formatter. Capability detection lives in
:mod:`modules.config.capabilities`; batch backends live in
:mod:`modules.batch`.
"""

# Capabilities are re-exported here for backward compatibility; the
# canonical home is modules.config.capabilities.
from modules.config.capabilities import (
    Capabilities,
    detect_capabilities,
    detect_provider,
)
from modules.llm.langchain_provider import (
    LangChainLLM,
    LLMProvider,
    ProviderConfig,
    get_default_provider,
)
from modules.llm.openai_utils import (
    LLMExtractor,
    open_extractor,
    process_text_chunk,
)
from modules.llm.payload_builder import PayloadBuilder
from modules.llm.prompt_utils import (
    load_prompt_template,
    render_prompt_with_schema,
)
from modules.llm.response_parser import ResponseParser
from modules.llm.schema_utils import build_structured_text_format

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
    "PayloadBuilder",
    "ResponseParser",
    "load_prompt_template",
    "render_prompt_with_schema",
    "build_structured_text_format",
]
