"""
ChronoMiner LLM Module.

Provides the LLM provider abstraction (LangChain-backed), prompt template
helpers, and the structured-output schema formatter. Capability detection
lives in :mod:`modules.config.capabilities`; batch backends live in
:mod:`modules.batch`.
"""

# Capabilities are re-exported here for backward compatibility; the
# canonical home is modules.config.capabilities.
from modules.config.capabilities import (
    Capabilities,
    detect_capabilities,
    detect_provider,
)
from modules.llm.http_timeouts import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_WRITE_TIMEOUT,
    build_httpx_timeout,
)
from modules.llm.langchain_provider import (
    LangChainLLM,
    ProviderConfig,
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
from modules.llm.schema_utils import build_structured_text_format
from modules.llm.transient_errors import (
    ChunkTimeoutError,
    is_connection_error,
    is_timeout_error,
    resolve_chunk_timeout,
)

__all__ = [
    "LangChainLLM",
    "ProviderConfig",
    "Capabilities",
    "detect_capabilities",
    "detect_provider",
    "LLMExtractor",
    "open_extractor",
    "process_text_chunk",
    "load_prompt_template",
    "render_prompt_with_schema",
    "build_structured_text_format",
    "build_httpx_timeout",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_WRITE_TIMEOUT",
    "DEFAULT_POOL_TIMEOUT",
    "is_timeout_error",
    "is_connection_error",
    "ChunkTimeoutError",
    "resolve_chunk_timeout",
]
