from __future__ import annotations

import json
from typing import Any, Dict


def render_prompt_with_schema(prompt_text: str, schema_obj: Dict[str, Any]) -> str:
    """Inject a JSON schema into a system prompt using flexible heuristics."""
    try:
        schema_str = json.dumps(schema_obj, indent=2, ensure_ascii=False)
    except Exception:
        schema_str = str(schema_obj)

    token = "{{TRANSCRIPTION_SCHEMA}}"
    if token in prompt_text:
        return prompt_text.replace(token, schema_str)

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
