"""Context resolution for prompts.

Tiers, most specific first:

1. explicit: text or a file path given for the run
2. file sidecar: ``{name}{suffix}{ext}`` beside the input, where ``name`` is the
   stem of a file input or the full name of a directory input
3. folder sidecar: ``{parent}{suffix}{ext}`` beside the input's parent folder
4. default: text or a file path from the settings

Sidecars that are empty or unreadable fall through to the next tier. There is no
implicit fallback file.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SIZE_THRESHOLD = 4000
NO_CONTEXT_HASH = "none"
TEXT_EXTENSIONS: tuple[str, ...] = (".txt",)
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
    ".webp",
)


class ContextError(ValueError):
    """An explicit or default context file is missing, empty or unreadable."""


class ContextSource(StrEnum):
    """The tier a context came from."""

    EXPLICIT = "explicit"
    FILE = "file"
    FOLDER = "folder"
    DEFAULT = "default"


@dataclass(frozen=True)
class ResolvedContext:
    """A resolved context and where it came from.

    ``text`` is stripped file content, or explicit text as given. ``path`` is
    None for context given as text.
    """

    text: str
    source: ContextSource
    path: Path | None = None
    oversized: bool = False

    @property
    def hash(self) -> str:
        """Fingerprint of ``text``; see :func:`compute_context_hash`."""
        return compute_context_hash(self.text)


def sidecar_candidates(
    input_path: Path,
    suffix: str,
    extensions: Sequence[str] = TEXT_EXTENSIONS,
    *,
    folder_sidecar_for_directories: bool = True,
) -> list[tuple[ContextSource, Path]]:
    """Return the sidecar paths to try for *input_path*, most specific first.

    *suffix* includes its leading separator, for example ``"_notes_context"``.
    With ``folder_sidecar_for_directories=False`` a directory input only has
    its own sidecar.
    """
    path = Path(input_path).resolve()
    is_dir = path.is_dir()
    base = path.name if is_dir else path.stem
    candidates = [
        (ContextSource.FILE, path.with_name(f"{base}{suffix}{ext}"))
        for ext in extensions
    ]
    parent = path.parent
    if (not is_dir or folder_sidecar_for_directories) and parent != parent.parent:
        candidates.extend(
            (ContextSource.FOLDER, parent.parent / f"{parent.name}{suffix}{ext}")
            for ext in extensions
        )
    return candidates


def resolve_context(
    input_path: Path | None,
    suffix: str,
    *,
    explicit: str | Path | None = None,
    default: str | Path | None = None,
    use_sidecars: bool = True,
    folder_sidecar_for_directories: bool = True,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
) -> ResolvedContext | None:
    """Resolve the context for one input through the four tiers.

    A ``str`` for *explicit* or *default* is context text; a ``Path`` is a
    file to read, and a missing, empty or unreadable file raises
    :class:`ContextError`. Empty text counts as not given.
    """
    if explicit is not None:
        given = _given(explicit, ContextSource.EXPLICIT, size_threshold)
        if given is not None:
            return given
    if use_sidecars and input_path is not None:
        for source, candidate in sidecar_candidates(
            input_path,
            suffix,
            folder_sidecar_for_directories=folder_sidecar_for_directories,
        ):
            if not candidate.is_file():
                continue
            text = read_context_file(candidate)
            if text:
                logger.info("Using %s context: %s", source.value, candidate)
                return _resolved(text, source, candidate, size_threshold)
    if default is not None:
        return _given(default, ContextSource.DEFAULT, size_threshold)
    return None


def resolve_context_image(
    input_path: Path | None,
    suffix: str,
    *,
    explicit: Path | None = None,
    default: Path | None = None,
    use_sidecars: bool = True,
    folder_sidecar_for_directories: bool = True,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> tuple[ContextSource, Path] | None:
    """Resolve a context image through the same tiers as text context.

    Extensions are tried in the given order at each tier. An explicit or
    default image that does not exist or has another extension raises
    :class:`ContextError`.
    """
    if explicit is not None:
        return ContextSource.EXPLICIT, _checked_image(explicit, extensions)
    if use_sidecars and input_path is not None:
        for source, candidate in sidecar_candidates(
            input_path,
            suffix,
            extensions,
            folder_sidecar_for_directories=folder_sidecar_for_directories,
        ):
            if candidate.is_file():
                logger.info("Using %s context image: %s", source.value, candidate)
                return source, candidate
    if default is not None:
        return ContextSource.DEFAULT, _checked_image(default, extensions)
    return None


def read_context_file(path: Path) -> str | None:
    """Return the stripped text of *path*, or None when empty or unreadable."""
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read context file %s: %s", path, exc)
        return None
    return text or None


def format_context_for_prompt(context: str) -> str:
    """Join the non-blank lines of *context* with ``", "`` for a prompt."""
    lines = [line.strip() for line in context.strip().split("\n") if line.strip()]
    return ", ".join(lines) if lines else context.strip()


def compute_context_hash(content: str | None) -> str:
    """Return the SHA-256 hex digest of a resolved context string.

    The digest covers the string injected into the prompt, not the file
    bytes or path, so moving or re-encoding a context file keeps the hash.
    None maps to :data:`NO_CONTEXT_HASH`.
    """
    if content is None:
        return NO_CONTEXT_HASH
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _given(
    value: str | Path, source: ContextSource, size_threshold: int
) -> ResolvedContext | None:
    if isinstance(value, Path):
        if not value.is_file():
            raise ContextError(f"Context file not found: {value}")
        text = read_context_file(value)
        if text is None:
            raise ContextError(f"Context file is empty or unreadable: {value}")
        return _resolved(text, source, value, size_threshold)
    if not value.strip():
        return None
    return _resolved(value, source, None, size_threshold)


def _resolved(
    text: str, source: ContextSource, path: Path | None, size_threshold: int
) -> ResolvedContext:
    oversized = len(text) > size_threshold
    if oversized:
        label = path.name if path is not None else f"{source.value} context"
        logger.warning(
            "Context '%s' is large (%s chars). Consider reducing it to under %s chars.",
            label,
            f"{len(text):,}",
            f"{size_threshold:,}",
        )
    return ResolvedContext(text=text, source=source, path=path, oversized=oversized)


def _checked_image(path: Path, extensions: Sequence[str]) -> Path:
    if not path.is_file():
        raise ContextError(f"Context image not found: {path}")
    if path.suffix.lower() not in extensions:
        raise ContextError(f"Unsupported context image format: {path}")
    return path
