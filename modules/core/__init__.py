"""
ChronoMiner Core Module.

Provides core utilities for text processing, data conversion, and chunking.
"""

from importlib import import_module

__all__ = [
    "BaseConverter",
    "CSVConverter",
    "DocumentConverter",
    "ChunkingService",
    "SchemaManager",
    "TextProcessor",
    "extract_entries_from_json",
    "setup_logger",
    "ensure_path_safe",
]

_LAZY_EXPORTS = {
    "BaseConverter": ("modules.core.converter_base", "BaseConverter"),
    "CSVConverter": ("modules.core.data_processing", "CSVConverter"),
    "DocumentConverter": ("modules.core.text_processing", "DocumentConverter"),
    "ChunkingService": ("modules.core.chunking_service", "ChunkingService"),
    "SchemaManager": ("modules.core.schema_manager", "SchemaManager"),
    "TextProcessor": ("modules.core.text_utils", "TextProcessor"),
    "extract_entries_from_json": ("modules.core.json_utils", "extract_entries_from_json"),
    "setup_logger": ("modules.core.logger", "setup_logger"),
    "ensure_path_safe": ("modules.core.path_utils", "ensure_path_safe"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
