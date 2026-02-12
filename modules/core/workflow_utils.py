from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from modules.config.loader import ConfigLoader, get_config_loader
from modules.core.logger import setup_logger
from modules.core.schema_manager import SchemaManager

logger = setup_logger(__name__)

_TEXT_EXTENSIONS = {".txt"}
_LINE_RANGE_SUFFIXES = {"_line_ranges.txt", "_line_range.txt"}
_CONTEXT_SUFFIXES = {"_extraction.txt", "_line_ranges.txt", "_context.txt"}


def load_core_resources() -> Tuple[
    ConfigLoader,
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
    """Load configuration resources shared across CLI entry points."""
    config_loader = get_config_loader()

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




def filter_text_files(paths: Iterable[Path]) -> List[Path]:
    """Filter an iterable of paths down to eligible text files."""
    filtered: List[Path] = []
    for candidate in paths:
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in _TEXT_EXTENSIONS:
            continue
        name = candidate.name
        if any(name.endswith(suffix) for suffix in _CONTEXT_SUFFIXES):
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


def validate_schema_paths(
    schema_name: str,
    schemas_paths: Dict[str, Any],
    ui: Optional[Any] = None,
) -> bool:
    """
    Validate that a schema has input/output paths configured in paths_config.yaml.
    
    Args:
        schema_name: The selected schema name.
        schemas_paths: The schemas_paths dict from paths_config.yaml.
        ui: Optional UserInterface for formatted output (interactive mode).
    
    Returns:
        True if paths are configured, False otherwise.
    """
    def _report_error(msg: str) -> bool:
        logger.error(msg)
        if ui:
            ui.print_error(msg)
        else:
            print(f"[ERROR] {msg}")
        return False

    if schema_name not in schemas_paths:
        return _report_error(
            f"Schema '{schema_name}' has no path configuration in config/paths_config.yaml. "
            f"Please add an entry under 'schemas_paths' with 'input' and 'output' paths."
        )

    schema_config = schemas_paths[schema_name]
    for key in ("input", "output"):
        if not schema_config.get(key):
            return _report_error(
                f"Schema '{schema_name}' has no '{key}' path configured in config/paths_config.yaml."
            )

    return True
