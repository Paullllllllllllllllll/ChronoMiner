"""Prompt template rendering utilities.

Provides functions for rendering prompt templates with schema information
for extraction tasks. The schema is written as compact JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _closing_brace(text: str, start: int) -> int:
    """Return the index of the brace closing the one at *start*, or -1."""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def render_prompt_with_schema(prompt_text: str, schema_obj: dict[str, Any]) -> str:
    """Return the prompt with the schema written in as compact JSON.

    A ``{{TRANSCRIPTION_SCHEMA}}`` token is replaced by the schema. Otherwise,
    after a ``The JSON schema:`` marker, the first balanced ``{...}`` block is
    replaced, or the schema is appended when there is none. Without token or
    marker, a ``The JSON schema:`` section with the schema is appended.
    """
    try:
        schema_str = json.dumps(schema_obj, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        schema_str = str(schema_obj)

    token = "{{TRANSCRIPTION_SCHEMA}}"
    if token in prompt_text:
        return prompt_text.replace(token, schema_str)

    marker = "The JSON schema:"
    if marker in prompt_text:
        idx = prompt_text.find(marker)
        start_brace = prompt_text.find("{", idx)
        if start_brace != -1:
            end_brace = _closing_brace(prompt_text, start_brace)
            if end_brace != -1 and end_brace > start_brace:
                return (
                    prompt_text[:start_brace]
                    + schema_str
                    + prompt_text[end_brace + 1 :]
                )
        return prompt_text + "\n" + schema_str

    return prompt_text + "\n\nThe JSON schema:\n" + schema_str


def inject_additional_context(prompt_text: str, context: str | None) -> str:
    """Return the prompt with *context* in place of its ``{{CONTEXT}}`` marker.

    Without context (None or blank), the marker and its one-line label (a line
    ending in a colon directly above it) are removed. A prompt without the
    marker is returned unchanged.
    """
    marker = "{{CONTEXT}}"
    if marker not in prompt_text:
        return prompt_text

    context_text = context.strip() if context else ""

    if context_text:
        return prompt_text.replace(marker, context_text)
    else:
        patterns = [
            r"(?m)^[^\n]*:[ \t]*\n\s*\{\{CONTEXT\}\}\s*\n?",
        ]

        result = prompt_text
        for pattern in patterns:
            result = re.sub(pattern, "", result)

        result = result.replace(marker, "")
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result


def prepare_prompt_with_context(
    prompt_text: str,
    schema_obj: dict[str, Any] | None = None,
    context: str | None = None,
) -> str:
    """Return the prompt with the schema and the context written in.

    Applies :func:`render_prompt_with_schema` (skipped when *schema_obj* is
    None or empty), then :func:`inject_additional_context`.
    """
    result = prompt_text

    if schema_obj:
        result = render_prompt_with_schema(result, schema_obj)

    result = inject_additional_context(result, context or "")

    return result


def load_prompt_template(path: Path) -> str:
    """Return the stripped text of the prompt template at *path*.

    Raises:
        FileNotFoundError: The file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    return path.read_text(encoding="utf-8").strip()
