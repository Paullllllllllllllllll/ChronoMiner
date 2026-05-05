"""
ChronoMiner Configuration Module.

Provides YAML configuration loading and validation, plus schema-registry
and context-resolution primitives that are compile-time environment
facts (not runtime orchestration).
"""

from modules.config.capabilities import (
    Capabilities,
    detect_capabilities,
    detect_provider,
    disabled_params_for_capabilities,
    disabled_params_for_model,
)
from modules.config.context import (
    resolve_context_for_extraction,
    resolve_context_for_readjustment,
)
from modules.config.loader import (
    ConfigLoader,
    clear_config_cache,
    get_config_loader,
)
from modules.config.manager import (
    ConfigManager,
    ConfigValidationError,
)
from modules.config.schema_manager import SchemaManager

__all__ = [
    "ConfigLoader",
    "get_config_loader",
    "clear_config_cache",
    "ConfigManager",
    "ConfigValidationError",
    "SchemaManager",
    "Capabilities",
    "detect_capabilities",
    "detect_provider",
    "disabled_params_for_capabilities",
    "disabled_params_for_model",
    "resolve_context_for_extraction",
    "resolve_context_for_readjustment",
]
