from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from modules.core.prompt_context import apply_context_placeholders


def render_prompt_with_schema(
    prompt_text: str,
    schema_obj: Dict[str, Any],
    schema_name: str | None = None,
    inject_schema: bool = True,
    additional_context: Optional[str] = None,
    basic_context: Optional[str] = None,
) -> str:
    """Inject schema metadata and optional schema name into a system prompt."""

    schema_name_token = "{{SCHEMA_NAME}}"
    if schema_name_token in prompt_text:
        prompt_text = prompt_text.replace(schema_name_token, schema_name or "")

    prompt_text = apply_context_placeholders(
        prompt_text,
        basic_context=basic_context,
        additional_context=additional_context,
    )

    schema_placeholder = "{{TRANSCRIPTION_SCHEMA}}"
    if not inject_schema or not schema_obj:
        if schema_placeholder in prompt_text:
            return prompt_text.replace(schema_placeholder, "")
        return prompt_text

    try:
        schema_str = json.dumps(schema_obj, indent=2, ensure_ascii=False)
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
