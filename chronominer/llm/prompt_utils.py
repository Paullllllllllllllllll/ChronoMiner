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
    """
    Inject a JSON schema into a prompt using one of three strategies:
    - If the token "{{TRANSCRIPTION_SCHEMA}}" exists, replace it with a
      compact schema.
    - If the marker "The JSON schema:" exists, replace any parsable JSON object
      that follows the marker with the new schema; if no parsable block is found,
      append the schema after the marker.
    - If neither token nor marker exists, append a new section "The JSON schema:"
      followed by the schema.
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
    """
    Inject additional context into a prompt using the {{CONTEXT}} marker.

    If the marker exists and context is provided, the marker is replaced with
    the context. If the marker exists but context is empty/None, the marker
    and its one-line label (any line ending in a colon directly above it) are
    removed to save tokens and avoid confusing the model. If the marker does
    not exist, the prompt is returned unchanged (fail-safe behavior).

    Parameters
    ----------
    prompt_text : str
        The prompt template text containing the marker
    context : str | None
        The additional context to inject

    Returns
    -------
    str
        The prompt with context injected or section removed
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
    """
    Prepare a complete prompt by rendering schema and injecting context.

    This is a convenience function that combines render_prompt_with_schema
    and inject_additional_context into a single call.

    Parameters
    ----------
    prompt_text : str
        The raw prompt template text
    schema_obj : Optional[Dict[str, Any]]
        The JSON schema to inject. If None, schema injection is skipped.
    context : Optional[str]
        The additional context to inject. If None/empty, the context section is removed.

    Returns
    -------
    str
        The fully prepared prompt with schema and context applied
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
