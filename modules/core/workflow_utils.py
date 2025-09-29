from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from modules.config.loader import ConfigLoader
from modules.core.context_manager import ContextManager
from modules.core.schema_manager import SchemaManager


_TEXT_EXTENSIONS = {".txt"}
_LINE_RANGE_SUFFIXES = {"_line_ranges.txt", "_line_range.txt"}
_CONTEXT_SUFFIX = "_context.txt"


def load_core_resources() -> Tuple[
    ConfigLoader,
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
    """Load configuration resources shared across CLI entry points."""
    config_loader = ConfigLoader()
    config_loader.load_configs()

    paths_config = config_loader.get_paths_config()
    model_config = config_loader.get_model_config()
    chunking_and_context_config = config_loader.get_chunking_and_context_config() or {}
    schemas_paths = config_loader.get_schemas_paths()

    return (
        config_loader,
        paths_config,
        model_config,
        chunking_and_context_config,
        schemas_paths,
    )


def load_schema_manager(*, ensure_available: bool = True) -> SchemaManager:
    """Instantiate and populate a ``SchemaManager`` instance."""
    schema_manager = SchemaManager()
    schema_manager.load_schemas()

    if ensure_available and not schema_manager.get_available_schemas():
        raise RuntimeError("No schemas found in the schemas directory")

    return schema_manager


def prepare_context_manager(context_settings: Dict[str, Any]) -> Optional[ContextManager]:
    """Provision a ``ContextManager`` when default additional context is requested."""
    use_additional = context_settings.get("use_additional_context", False)
    use_default = context_settings.get("use_default_context", False)

    if use_additional and use_default:
        context_manager = ContextManager()
        context_manager.load_additional_context()
        return context_manager

    return None


def filter_text_files(paths: Iterable[Path]) -> List[Path]:
    """Filter an iterable of paths down to eligible text files."""
    filtered: List[Path] = []
    for candidate in paths:
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in _TEXT_EXTENSIONS:
            continue
        name = candidate.name
        if name.endswith(_CONTEXT_SUFFIX):
            continue
        if any(name.endswith(suffix) for suffix in _LINE_RANGE_SUFFIXES):
            continue
        filtered.append(candidate)
    return filtered


def collect_text_files(root: Path) -> List[Path]:
    """Collect eligible text files from a root path, skipping auxiliary files."""
    if root.is_file():
        return filter_text_files([root])

    return filter_text_files(sorted(root.rglob("*")))
