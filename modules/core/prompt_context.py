from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BASIC_CONTEXT_DIR = _PROJECT_ROOT / "basic_context"


def _normalize_directory(path: Optional[Path]) -> Path:
    """Normalize a directory path, returning default if None."""
    if path is None:
        return _BASIC_CONTEXT_DIR
    return Path(path).resolve()


def _read_text_file(file_path: Path) -> Optional[str]:
    """
    Read text from a file with proper error handling.

    Parameters
    ----------
    file_path : Path
        Path to the text file to read.

    Returns
    -------
    Optional[str]
        The file content if successful, None otherwise.
    """
    try:
        text = file_path.read_text(encoding="utf-8").strip()
        return text if text else None
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read context file %s: %s", file_path, exc)
        return None


@lru_cache(maxsize=32)
def load_basic_context(
    basic_context_dir: Optional[str] = None,
    schema_name: Optional[str] = None
) -> str:
    """
    Load basic context files for a specific schema or all schemas.

    Parameters
    ----------
    basic_context_dir : Optional[str]
        Optional directory containing ``*.txt`` files with reusable context.
    schema_name : Optional[str]
        Optional schema name to load context for. If provided, only loads
        the context file matching this schema (e.g., "BibliographicEntries.txt").
        If None, loads all context files (legacy behavior).

    Returns
    -------
    str
        The aggregated basic context separated by blank lines. Empty string
        when no files are present or schema-specific file not found.
    """
    directory = _normalize_directory(Path(basic_context_dir) if basic_context_dir else None)
    if not directory.exists():
        logger.info("Basic context directory not found: %s", directory)
        return ""

    snippets = []
    
    # If schema_name is provided, load only that specific context file
    if schema_name:
        context_file = directory / f"{schema_name}.txt"
        if context_file.exists():
            snippet = _read_text_file(context_file)
            if snippet:
                snippets.append(snippet)
                logger.info("Loaded basic context for schema: %s", schema_name)
            else:
                logger.warning("Basic context file exists but is empty: %s", context_file)
        else:
            logger.warning("No basic context file found for schema: %s", schema_name)
    else:
        # Legacy behavior: load all context files
        logger.warning(
            "Loading all basic context files (no schema_name provided). "
            "This may inject irrelevant context into prompts."
        )
        for context_file in sorted(directory.glob("*.txt")):
            snippet = _read_text_file(context_file)
            if snippet:
                snippets.append(snippet)

    return "\n\n".join(snippets)


def load_file_specific_context(file_path: Path) -> Optional[str]:
    """
    Return context from ``<stem>_context.txt`` next to ``file_path`` if
    present.

    Parameters
    ----------
    file_path : Path
        The input file path.

    Returns
    -------
    Optional[str]
        File-specific context if found, None otherwise.
    """
    context_file = file_path.with_name(f"{file_path.stem}_context.txt")
    if not context_file.exists():
        return None
    return _read_text_file(context_file)


def combine_contexts(*parts: Optional[str]) -> Optional[str]:
    """
    Combine multiple context strings into one.

    Parameters
    ----------
    *parts : Optional[str]
        Variable number of context strings to combine.

    Returns
    -------
    Optional[str]
        Combined context separated by blank lines, or None if all parts
        are empty.
    """
    snippets = [part.strip() for part in parts if part and part.strip()]
    if not snippets:
        return None
    return "\n\n".join(snippets)


def apply_context_placeholders(
    text: str,
    *,
    basic_context: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> str:
    """
    Replace context placeholders in *text* with provided snippets.

    Parameters
    ----------
    text : str
        The template text containing placeholders.
    basic_context : Optional[str]
        Basic context to replace {{BASIC_CONTEXT}} placeholder.
    additional_context : Optional[str]
        Additional context to replace {{ADDITIONAL_CONTEXT}} placeholder.

    Returns
    -------
    str
        Text with placeholders replaced.
    """
    basic_value = (basic_context or "").strip() or "Empty (no basic context)"
    text = text.replace("{{BASIC_CONTEXT}}", basic_value)

    additional_value = (additional_context or "").strip() or "Empty (no additional context)"
    text = text.replace("{{ADDITIONAL_CONTEXT}}", additional_value)
    return text


def resolve_additional_context(
    schema_name: str,
    *,
    context_settings: Optional[Dict[str, object]],
    context_manager=None,
    text_file: Optional[Path] = None,
    context_sources: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    Determine the additional context string for a schema/file pair.

    Parameters
    ----------
    schema_name : str
        Name of the schema being processed.
    context_settings : Optional[Dict[str, object]]
        Dictionary with context configuration flags.
    context_manager : Optional
        ContextManager instance for loading default context.
    text_file : Optional[Path]
        Input file path for file-specific context lookup.
    context_sources : Optional[Sequence[str]]
        List of explicit context file paths to load.

    Returns
    -------
    Optional[str]
        The resolved additional context string, or None if no context
        is configured.
    """
    if not context_settings or not context_settings.get("use_additional_context", False):
        return None

    use_default = bool(context_settings.get("use_default_context", False))

    default_context: Optional[str] = None
    if use_default and context_manager is not None:
        try:
            default_context = context_manager.get_additional_context(schema_name)
        except Exception as exc:
            logger.warning("Failed to fetch default additional context for %s: %s", schema_name, exc)

    if use_default:
        if default_context is None:
            logger.info("Default additional context requested for schema '%s' but not found", schema_name)
        return default_context

    if text_file is not None:
        file_context = load_file_specific_context(text_file)
        if file_context:
            return file_context

    if context_sources:
        snippets = []
        for source in context_sources:
            context_file = Path(source)
            snippet = _read_text_file(context_file)
            if snippet:
                snippets.append(snippet)
        return combine_contexts(*snippets)

    return None
