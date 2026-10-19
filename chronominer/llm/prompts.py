"""Prompt inputs of extraction and line-range adjustment.

The extraction prompt depends on the chunk kind: a text chunk takes
``text_extraction_prompt.txt``, an image page ``image_extraction_prompt.txt``.
Both receive the context and the selected schema as compact JSON. Enforced
answers use the schema under the name :data:`EXTRACTION_SCHEMA_NAME`. The
boundary prompt of ``--chunking adjust`` and ``ranges adjust`` names the
semantic unit and carries its answer schema.

Context text and the context image are resolved per input through
:mod:`chronominer.common.context` with the suffix of the role. The context
is the ``--context`` given (topics as text, or a path to a text file), else
the input's sidecar (``{stem}_extract_context.txt`` or
``{stem}_adjust_context.txt`` beside it, then ``{folder}_extract_context.txt``
beside its folder), else the ``--fallback-context`` (text or a path). Context
images take the ``_extract_context`` suffix with an image extension, else the
project's ``context/extract_context`` image.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from chronominer.common.context import (
    IMAGE_EXTENSIONS,
    ContextError,
    resolve_context,
    resolve_context_image,
)
from chronominer.common.provider_requests import (
    Content,
    cache_control,
    system_blocks,
)
from chronominer.common.structured import schema_body
from chronominer.llm.prompt_utils import (
    load_prompt_template,
    prepare_prompt_with_context,
)
from chronominer.spec import CONTEXT_NONE, ContextChoice, ContextText

if TYPE_CHECKING:
    from chronominer.settings import PromptCaching

__all__ = [
    "BOUNDARY_PROMPT_PATH",
    "BOUNDARY_TYPE_TOKEN",
    "CONTEXT_DIR",
    "CONTEXT_IMAGE_LABEL",
    "CONTEXT_SIZE_THRESHOLD",
    "CONTEXT_SUFFIX",
    "CONTEXT_SUFFIXES",
    "EXTRACTION_SCHEMA_NAME",
    "IMAGE_CONTEXT_SUFFIX",
    "IMAGE_INSTRUCTION",
    "IMAGE_PROMPT_PATH",
    "PROMPTS_DIR",
    "TEXT_PROMPT_PATH",
    "ChunkKind",
    "ContextError",
    "ContextRole",
    "boundary_prompt",
    "extraction_prompt",
    "extraction_schema",
    "item_context",
    "item_context_image",
    "system_content",
]

logger = logging.getLogger(__name__)

type ChunkKind = Literal["text", "image"]
type ContextRole = Literal["extract", "adjust"]

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _PACKAGE_DIR.parent
PROMPTS_DIR = _PACKAGE_DIR / "resources" / "prompts"
TEXT_PROMPT_PATH = PROMPTS_DIR / "text_extraction_prompt.txt"
IMAGE_PROMPT_PATH = PROMPTS_DIR / "image_extraction_prompt.txt"
BOUNDARY_PROMPT_PATH = PROMPTS_DIR / "semantic_boundary_prompt.txt"
_PROMPT_PATHS: Mapping[ChunkKind, Path] = {
    "text": TEXT_PROMPT_PATH,
    "image": IMAGE_PROMPT_PATH,
}

CONTEXT_DIR = PROJECT_ROOT / "context"
"""The project context folder, where an automatic context image is last looked
up."""

CONTEXT_SUFFIXES: Mapping[ContextRole, str] = {
    "extract": "_extract_context",
    "adjust": "_adjust_context",
}
CONTEXT_SUFFIX = CONTEXT_SUFFIXES["extract"]
IMAGE_CONTEXT_SUFFIX = CONTEXT_SUFFIXES["extract"]
CONTEXT_SIZE_THRESHOLD = 5000
"""Context longer than this many characters is logged as large."""

EXTRACTION_SCHEMA_NAME = "ExtractionSchema"
IMAGE_INSTRUCTION = "Extract structured data from this image according to the schema."
"""The text block sent before each page image."""
CONTEXT_IMAGE_LABEL = "Context image:"
"""The text block sent before the context image."""
BOUNDARY_TYPE_TOKEN = "{{BOUNDARY_TYPE}}"


def extraction_prompt(
    kind: ChunkKind, schema: Mapping[str, Any], *, context: str | None
) -> str:
    """Return the system prompt of a *kind* chunk.

    *schema* is a schema file (``{name, schema, ...}``) or a bare JSON
    schema; its body is injected as compact JSON.

    Raises:
        FileNotFoundError: The prompt template is missing.
    """
    template = load_prompt_template(_PROMPT_PATHS[kind])
    return prepare_prompt_with_context(template, schema_body(schema), context)


def extraction_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema of enforced extraction answers.

    The body of *schema* goes under :data:`EXTRACTION_SCHEMA_NAME`, strict.

    Raises:
        ValueError: *schema* holds no JSON schema.
    """
    body = schema_body(schema)
    if body is None:
        raise ValueError("The extraction schema is empty")
    return {"name": EXTRACTION_SCHEMA_NAME, "schema": body, "strict": True}


def boundary_prompt(
    boundary_type: str,
    schema: Mapping[str, Any],
    context: str | None,
    template: Path | None = None,
) -> str:
    """Return the system prompt of a line-range adjustment call.

    *boundary_type* names the semantic unit (the extraction schema's name);
    *schema* is the answer schema, injected as compact JSON.

    Raises:
        FileNotFoundError: The prompt template is missing.
    """
    text = load_prompt_template(template or BOUNDARY_PROMPT_PATH)
    rendered = prepare_prompt_with_context(text, schema_body(schema), context)
    return rendered.replace(BOUNDARY_TYPE_TOKEN, boundary_type)


def system_content(
    prompt: str, *, provider: str, model: str, caching: PromptCaching
) -> str | Content:
    """Return the system message content of *prompt* for *provider*.

    With caching on, Anthropic system prompts carry a cache breakpoint (with
    ``ttl`` 1h when set), and so do Claude models through OpenRouter when
    ``openrouter_cache_control`` is on; every other prompt is the plain text
    (:func:`~chronominer.common.provider_requests.cache_control`).
    """
    return system_blocks(prompt, cache_control(provider, model, caching))


def _explicit(choice: ContextChoice) -> Path | None:
    if isinstance(choice, Path):
        return choice.expanduser().resolve()
    return None


def text_or_file(value: str | Path | None) -> str | Path | None:
    """Return *value* as a path when it names an existing file, else as text.

    A :class:`~pathlib.Path` stays a path, so a missing file is reported. Text
    that cannot be a path (an unknown ``~user``, a name too long) stays text.
    """
    if value is None or isinstance(value, Path):
        return value
    try:
        candidate = Path(value).expanduser()
        is_file = candidate.is_file()
    except (OSError, RuntimeError):
        return value
    return candidate.resolve() if is_file else value


def _default_image() -> Path | None:
    name = IMAGE_CONTEXT_SUFFIX.lstrip("_")
    for ext in IMAGE_EXTENSIONS:
        candidate = CONTEXT_DIR / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    return None


def item_context(
    item: Path,
    choice: ContextText | Path,
    role: ContextRole = "extract",
    *,
    fallback: ContextText | Path = None,
) -> str | None:
    """Return the context text of *item* for the ``--context`` *choice*.

    *choice* and *fallback* are topics as text or a path: a path to an
    existing file is read, anything else is context text. ``none`` sends no
    context; the automatic lookup (None) takes the input's sidecar of
    *role*, then its folder's, then *fallback*.

    Raises:
        ContextError: A given file is empty or unreadable.
    """
    if choice == CONTEXT_NONE:
        return None
    explicit = text_or_file(choice)
    resolved = resolve_context(
        item,
        CONTEXT_SUFFIXES[role],
        explicit=explicit,
        default=None if explicit is not None else text_or_file(fallback),
        use_sidecars=explicit is None,
        size_threshold=CONTEXT_SIZE_THRESHOLD,
    )
    return resolved.text if resolved is not None else None


def item_context_image(item: Path, choice: ContextChoice) -> Path | None:
    """Return the context image of *item* for the ``--context-image`` *choice*.

    The lookup follows :func:`item_context` with image files.

    Raises:
        ContextError: The given image is missing or not a supported image.
    """
    if choice == CONTEXT_NONE:
        return None
    explicit = _explicit(choice)
    resolved = resolve_context_image(
        item,
        IMAGE_CONTEXT_SUFFIX,
        explicit=explicit,
        default=None if explicit is not None else _default_image(),
        use_sidecars=explicit is None,
        extensions=IMAGE_EXTENSIONS,
    )
    return resolved[1] if resolved is not None else None
