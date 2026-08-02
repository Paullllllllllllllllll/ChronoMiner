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

import hashlib
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONTEXT_DIR = _PROJECT_ROOT / "context"

DEFAULT_CONTEXT_SIZE_THRESHOLD = 5000

ContextTask = Literal["extract_context", "adjust_context"]


def _resolve_context(
    suffix: str,
    text_file: Path | None = None,
    context_dir: Path | None = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> tuple[str | None, Path | None]:
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
            folder_specific = (
                parent_folder.parent / f"{parent_folder.name}{filename_suffix}"
            )
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
    text_file: Path | None = None,
    context_dir: Path | None = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> tuple[str | None, Path | None]:
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
    text_file: Path | None = None,
    context_dir: Path | None = None,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> tuple[str | None, Path | None]:
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


_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
    ".webp",
)


def resolve_context_image_for_extraction(
    text_file: Path | None = None,
    context_dir: Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Resolve a context image using the same hierarchy as text context.

    Searches for image files at three levels (most specific wins):
    1. File-specific:   {input_stem}_extract_context.{ext}
    2. Folder-specific: {parent_folder}_extract_context.{ext}
    3. General fallback: context/extract_context.{ext}

    Extensions are tried in priority order at each level.

    Parameters
    ----------
    text_file : Optional[Path]
        Path to the input file (enables file- and folder-specific lookup).
    context_dir : Optional[Path]
        Override for the project-level context directory.

    Returns
    -------
    Tuple[Optional[Path], Optional[Path]]
        ``(image_path, image_path)`` or ``(None, None)`` when nothing found.
    """
    effective_context_dir = context_dir or _CONTEXT_DIR
    suffix = "extract_context"

    if text_file is not None:
        text_file = Path(text_file).resolve()

        # Level 1: file-specific
        for ext in _IMAGE_EXTENSIONS:
            candidate = text_file.with_name(f"{text_file.stem}_{suffix}{ext}")
            if candidate.exists():
                logger.info(f"Using file-specific context image: {candidate}")
                return candidate, candidate

        # Level 2: folder-specific
        parent_folder = text_file.parent
        if parent_folder.parent.exists():
            for ext in _IMAGE_EXTENSIONS:
                candidate = parent_folder.parent / f"{parent_folder.name}_{suffix}{ext}"
                if candidate.exists():
                    logger.info(f"Using folder-specific context image: {candidate}")
                    return candidate, candidate

    # Level 3: general fallback
    for ext in _IMAGE_EXTENSIONS:
        candidate = effective_context_dir / f"{suffix}{ext}"
        if candidate.exists():
            logger.info(f"Using general context image: {candidate}")
            return candidate, candidate

    logger.debug("No context image found")
    return None, None


def _read_and_validate_context(
    context_path: Path,
    size_threshold: int = DEFAULT_CONTEXT_SIZE_THRESHOLD,
) -> str | None:
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
                f"Context file '{context_path.name}' is large "
                f"({len(content):,} chars). Consider reducing to under "
                f"{size_threshold:,} chars for optimal performance."
            )

        return content

    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(f"Failed to read context file {context_path}: {exc}")
        return None


# Sentinel recorded when a run resolved to no context at all. It is distinct
# from a missing header field, which means "produced before context hashing
# existed" and is treated as a wildcard by the resume checks.
NO_CONTEXT_HASH = "none"


def compute_context_hash(content: str | None) -> str:
    """Return a stable fingerprint of a resolved context string.

    Hashes the *resolved* context string -- the exact object injected into the
    prompt -- rather than a file path or file bytes. Two consequences follow:

    - All three resolution levels (file, folder, general fallback) are covered
      automatically, and the hash is path-independent: the same content
      promoted from a file-level to a folder-level context yields the same
      hash, and moving a campaign directory does not invalidate artifacts.
    - This is not a file-bytes fingerprint. Edits that cannot change the
      prompt (trailing whitespace, a BOM, CRLF/LF line endings) do not change
      the hash, so they do not force a re-run.

    ``None`` (no context resolved anywhere) maps to the literal sentinel
    ``"none"``, which distinguishes "resolved to no context" from a legacy
    header that simply lacks the field.

    Parameters
    ----------
    content : Optional[str]
        The resolved, stripped context string, or ``None``.

    Returns
    -------
    str
        SHA-256 hex digest of the UTF-8 encoded content, or ``"none"``.
    """
    if content is None:
        return NO_CONTEXT_HASH
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
