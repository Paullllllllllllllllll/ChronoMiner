from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union


def render_prompt_with_schema(
    prompt_text: str,
    schema_obj: Dict[str, Any],
    schema_name: str | None = None,
    inject_schema: bool = True,
    context: Optional[str] = None,
) -> str:
    """Inject schema metadata and unified context into a system prompt.
    
    Parameters
    ----------
    prompt_text : str
        The prompt template text
    schema_obj : Dict[str, Any]
        The JSON schema to inject
    schema_name : str | None
        Optional schema name for {{SCHEMA_NAME}} placeholder
    inject_schema : bool
        Whether to inject the schema
    context : Optional[str]
        Unified context to inject (replaces both basic and additional context)
    
    Returns
    -------
    str
        The rendered prompt with schema and context injected
    """
    schema_name_token = "{{SCHEMA_NAME}}"
    if schema_name_token in prompt_text:
        prompt_text = prompt_text.replace(schema_name_token, schema_name or "")

    # Handle unified context placeholder
    context_placeholder = "{{CONTEXT}}"
    if context_placeholder in prompt_text:
        if context and context.strip():
            # Replace with actual context
            prompt_text = prompt_text.replace(context_placeholder, context.strip())
        else:
            # Remove entire context section to save tokens
            # Pattern: "Context:\n{{CONTEXT}}\n"
            prompt_text = re.sub(r"Context:\s*\n\s*\{\{CONTEXT\}\}\s*\n?", "", prompt_text)
            # Fallback: just remove the placeholder
            prompt_text = prompt_text.replace(context_placeholder, "")

    # Handle schema injection
    schema_placeholder = "{{TRANSCRIPTION_SCHEMA}}"
    if not inject_schema or not schema_obj:
        if schema_placeholder in prompt_text:
            return prompt_text.replace(schema_placeholder, "")
        return prompt_text

    try:
        schema_str = json.dumps(
            schema_obj,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except Exception:
        schema_str = str(schema_obj)

    if schema_placeholder in prompt_text:
        return prompt_text.replace(schema_placeholder, schema_str)

    marker = "The JSON schema:"
    if marker in prompt_text:
        idx = prompt_text.find(marker)
        start_brace = prompt_text.find("{", idx)
        if start_brace != -1:
            end_brace = prompt_text.rfind("}")
            if end_brace != -1 and end_brace > start_brace:
                return prompt_text[:start_brace] + schema_str + prompt_text[end_brace + 1 :]
        return prompt_text + "\n" + schema_str

    return prompt_text + "\n\nThe JSON schema:\n" + schema_str


def load_prompt_template(prompt_path: Path) -> str:
    """Load and return the stripped prompt template text."""

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")

    with prompt_path.open("r", encoding="utf-8") as prompt_file:
        return prompt_file.read().strip()
