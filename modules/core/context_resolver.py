"""Context resolution utilities for ChronoMiner.

This module provides hierarchical context resolution for extraction and
line-range-readjustment tasks, using filename-suffix-based matching across
three resolution levels.

Context Resolution Hierarchy (most specific wins):
1. File-specific:   {input_stem}_{suffix}.txt   next to the input file
2. Folder-specific: {parent_folder}_{suffix}.txt next to the input's parent folder
3. General fallback: context/{suffix}.txt        in the project root

Suffixes per task type:
- Extraction:              extract_context
- Line-range readjustment: adjust_context
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONTEXT_DIR = _PROJECT_ROOT / "context"

DEFAULT_CONTEXT_SIZE_THRESHOLD = 4000

ContextTask = Literal["extract_context", "adjust_context"]


def _resolve_context(
    suffix: str,
    text_file: Optional[Path] = None,
    context_dir: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Generic hierarchical context resolution.

    Searches for context in this order:
    1. File-specific:   {input_stem}_{suffix}.txt   in the same directory as *text_file*
    2. Folder-specific: {parent_folder}_{suffix}.txt in the grandparent directory
    3. General fallback: context/{suffix}.txt        in the project context directory

    Parameters
    ----------
    suffix : str
        Context-file suffix without leading underscore (e.g. ``"extract_context"``).
    text_file : Optional[Path]
        Path to the input text file (enables file- and folder-specific lookup).
    context_dir : Optional[Path]
        Override for the project-level context directory (defaults to
        ``PROJECT_ROOT/context``).
    size_threshold : int
        Character-count threshold for a size warning.

    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        ``(content, resolved_path)`` or ``(None, None)`` when nothing is found.
    """
    effective_context_dir = context_dir or _CONTEXT_DIR
    filename_suffix = f"_{suffix}.txt"

    # 1. File-specific context
    if text_file is not None:
        text_file = Path(text_file).resolve()
        file_specific = text_file.with_name(f"{text_file.stem}{filename_suffix}")
        if file_specific.exists():
            content = _read_and_validate_context(file_specific, size_threshold)
            if content:
                logger.info(f"Using file-specific context: {file_specific}")
                return content, file_specific

        # 2. Folder-specific context
        parent_folder = text_file.parent
        if parent_folder.parent.exists():
            folder_specific = parent_folder.parent / f"{parent_folder.name}{filename_suffix}"
            if folder_specific.exists():
                content = _read_and_validate_context(folder_specific, size_threshold)
                if content:
                    logger.info(f"Using folder-specific context: {folder_specific}")
                    return content, folder_specific

    # 3. General fallback
    general_fallback = effective_context_dir / f"{suffix}.txt"
    if general_fallback.exists():
        content = _read_and_validate_context(general_fallback, size_threshold)
        if content:
            logger.info(f"Using general context: {general_fallback}")
            return content, general_fallback

    logger.debug(f"No {suffix} context found")
    return None, None


def resolve_context_for_extraction(
    text_file: Optional[Path] = None,
    context_dir: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Resolve extraction context using hierarchical fallback.

    Parameters
    ----------
    text_file : Optional[Path]
        Path to the input text file (for file/folder-specific context).
    context_dir : Optional[Path]
        Override for the project-level context directory.
    size_threshold : int
        Character-count threshold for a size warning.

    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        ``(content, resolved_path)`` or ``(None, None)`` when nothing is found.
    """
    return _resolve_context("extract_context", text_file, context_dir, size_threshold)


def resolve_context_for_readjustment(
    text_file: Optional[Path] = None,
    context_dir: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Resolve line-range-readjustment context using hierarchical fallback.

    Parameters
    ----------
    text_file : Optional[Path]
        Path to the input text file (for file/folder-specific context).
    context_dir : Optional[Path]
        Override for the project-level context directory.
    size_threshold : int
        Character-count threshold for a size warning.

    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        ``(content, resolved_path)`` or ``(None, None)`` when nothing is found.
    """
    return _resolve_context("adjust_context", text_file, context_dir, size_threshold)


def _read_and_validate_context(
    context_path: Path,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Optional[str]:
    """Read and validate a context file.
    
    Parameters
    ----------
    context_path : Path
        Path to the context file
    size_threshold : int
        Character count threshold for size warning
        
    Returns
    -------
    Optional[str]
        The context content, or None if file is empty or unreadable
    """
    try:
        content = context_path.read_text(encoding="utf-8").strip()
        
        if not content:
            logger.debug(f"Context file is empty: {context_path}")
            return None
        
        if len(content) > size_threshold:
            logger.warning(
                f"Context file '{context_path.name}' is large ({len(content):,} chars). "
                f"Consider reducing to under {size_threshold:,} chars for optimal performance."
            )
        
        return content
        
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(f"Failed to read context file {context_path}: {exc}")
        return None
