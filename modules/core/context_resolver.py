"""Context resolution utilities for ChronoMiner.

This module provides hierarchical context resolution for extraction and line-range-readjustment tasks,
supporting file-specific, folder-specific, and schema/boundary-type-specific context files.

Context Resolution Hierarchy:
1. File-specific: <filename>_<suffix>.txt next to the input file
2. Folder-specific: <foldername>_<suffix>.txt in the parent directory
3. Type-specific: context/<subdir>/<type_name>.txt
4. Global fallback: context/<subdir>/general.txt

Where <suffix> and <subdir> are:
- For extraction: "extraction" 
- For line range readjustment: "line_ranges"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONTEXT_DIR = _PROJECT_ROOT / "context"

DEFAULT_CONTEXT_SIZE_THRESHOLD = 4000

ContextType = Literal["extraction", "line_ranges"]


def _resolve_context(
    context_type: ContextType,
    type_name: str,
    text_file: Optional[Path] = None,
    global_context_dir: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Generic hierarchical context resolution.
    
    Searches for context in this order:
    1. File-specific: <filename>_<context_type>.txt in the same directory as text_file
    2. Folder-specific: <parent_folder_name>_<context_type>.txt in grandparent directory
    3. Type-specific: context/<context_type>/<type_name>.txt
    4. Global fallback: context/<context_type>/general.txt
    
    Parameters
    ----------
    context_type : ContextType
        The type of context ("extraction" or "line_ranges")
    type_name : str
        Name of the schema or boundary type
    text_file : Optional[Path]
        Path to the input text file (for file/folder-specific context)
    global_context_dir : Optional[Path]
        Override for the global context directory (defaults to PROJECT_ROOT/context)
    size_threshold : int
        Character count threshold for size warning
        
    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        A tuple of (context_content, resolved_path) or (None, None) if no context found
    """
    context_dir = global_context_dir or _CONTEXT_DIR
    suffix = f"_{context_type}.txt"
    
    # 1. File-specific context
    if text_file is not None:
        text_file = Path(text_file).resolve()
        file_specific = text_file.with_name(f"{text_file.stem}{suffix}")
        if file_specific.exists():
            content = _read_and_validate_context(file_specific, size_threshold)
            if content:
                logger.info(f"Using file-specific {context_type} context: {file_specific}")
                return content, file_specific
        
        # 2. Folder-specific context
        parent_folder = text_file.parent
        if parent_folder.parent.exists():
            folder_specific = parent_folder.parent / f"{parent_folder.name}{suffix}"
            if folder_specific.exists():
                content = _read_and_validate_context(folder_specific, size_threshold)
                if content:
                    logger.info(f"Using folder-specific {context_type} context: {folder_specific}")
                    return content, folder_specific
    
    # 3. Type-specific context
    type_specific = context_dir / context_type / f"{type_name}.txt"
    if type_specific.exists():
        content = _read_and_validate_context(type_specific, size_threshold)
        if content:
            logger.info(f"Using type-specific {context_type} context: {type_specific}")
            return content, type_specific
    
    # 4. Global fallback
    global_fallback = context_dir / context_type / "general.txt"
    if global_fallback.exists():
        content = _read_and_validate_context(global_fallback, size_threshold)
        if content:
            logger.info(f"Using global {context_type} context: {global_fallback}")
            return content, global_fallback
    
    logger.debug(f"No {context_type} context found for '{type_name}'")
    return None, None


def resolve_context_for_extraction(
    schema_name: str,
    text_file: Optional[Path] = None,
    global_context_dir: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Resolve extraction context using hierarchical fallback.
    
    Parameters
    ----------
    schema_name : str
        Name of the extraction schema
    text_file : Optional[Path]
        Path to the input text file (for file/folder-specific context)
    global_context_dir : Optional[Path]
        Override for the global context directory (defaults to PROJECT_ROOT/context)
    size_threshold : int
        Character count threshold for size warning
        
    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        A tuple of (context_content, resolved_path) or (None, None) if no context found
    """
    return _resolve_context("extraction", schema_name, text_file, global_context_dir, size_threshold)


def resolve_context_for_readjustment(
    boundary_type: str,
    text_file: Optional[Path] = None,
    global_context_dir: Optional[Path] = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> Tuple[Optional[str], Optional[Path]]:
    """Resolve line-range-readjustment context using hierarchical fallback.
    
    Parameters
    ----------
    boundary_type : str
        Type of semantic boundary to detect (typically schema name)
    text_file : Optional[Path]
        Path to the input text file (for file/folder-specific context)
    global_context_dir : Optional[Path]
        Override for the global context directory (defaults to PROJECT_ROOT/context)
    size_threshold : int
        Character count threshold for size warning
        
    Returns
    -------
    Tuple[Optional[str], Optional[Path]]
        A tuple of (context_content, resolved_path) or (None, None) if no context found
    """
    return _resolve_context("line_ranges", boundary_type, text_file, global_context_dir, size_threshold)


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
