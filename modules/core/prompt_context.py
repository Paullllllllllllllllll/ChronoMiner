from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BASIC_CONTEXT_DIR = _PROJECT_ROOT / "basic_context"


def _normalize_directory(path: Optional[Path]) -> Path:
    if path is None:
        return _BASIC_CONTEXT_DIR
    return Path(path).resolve()


def _read_text_file(file_path: Path) -> Optional[str]:
    try:
        text = file_path.read_text(encoding="utf-8").strip()
        return text if text else None
    except Exception as exc:  # pragma: no cover - filesystem guard
        logger.warning("Failed to read context file %s: %s", file_path, exc)
        return None


@lru_cache(maxsize=4)
def load_basic_context(basic_context_dir: Optional[str] = None) -> str:
    """Load and concatenate all basic context files.

    Parameters
    ----------
    basic_context_dir:
        Optional directory containing ``*.txt`` files with reusable context.

    Returns
    -------
    str
        The aggregated basic context separated by blank lines. Empty string when
        no files are present.
    """
    directory = _normalize_directory(Path(basic_context_dir) if basic_context_dir else None)
    if not directory.exists():
        logger.info("Basic context directory not found: %s", directory)
        return ""

    snippets = []
    for context_file in sorted(directory.glob("*.txt")):
        snippet = _read_text_file(context_file)
        if snippet:
            snippets.append(snippet)

    return "\n\n".join(snippets)


def load_file_specific_context(file_path: Path) -> Optional[str]:
    """Return context from ``<stem>_context.txt`` next to ``file_path`` if present."""
    context_file = file_path.with_name(f"{file_path.stem}_context.txt")
    if not context_file.exists():
        return None
    return _read_text_file(context_file)


def combine_contexts(*parts: Optional[str]) -> Optional[str]:
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
    """Replace context placeholders in *text* with provided snippets."""
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
    """Determine the additional context string for a schema/file pair."""

    if not context_settings or not context_settings.get("use_additional_context", False):
        return None

    use_default = bool(context_settings.get("use_default_context", False))

    default_context: Optional[str] = None
    if use_default and context_manager is not None:
        try:
            default_context = context_manager.get_additional_context(schema_name)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to fetch default additional context for %s: %s", schema_name, exc)

    if use_default:
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
