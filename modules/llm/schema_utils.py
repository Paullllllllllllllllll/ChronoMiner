# modules/llm/schema_utils.py

"""
Shared schema utilities for LLM payload construction.
"""

from typing import Any, Dict, Optional


def _build_structured_text_format(
    schema_obj: Dict[str, Any],
    default_name: str = "ExtractionSchema",
    default_strict: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Build the Responses API `text.format` object for Structured Outputs.

    Args:
        schema_obj: Schema object with optional "name", "schema", "strict" keys,
                    or a bare JSON Schema dict.
        default_name: Default schema name if not provided.
        default_strict: Default strict mode setting.

    Returns:
        Dict with shape {"type": "json_schema", "name": ..., "schema": ..., "strict": ...}
        or None if the provided schema is not usable.
    """
    if not isinstance(schema_obj, dict) or not schema_obj:
        return None

    # Unwrap schema: accept either wrapper dict or bare JSON Schema
    if "schema" in schema_obj and isinstance(schema_obj["schema"], dict):
        name = schema_obj.get("name") or default_name
        schema = schema_obj.get("schema") or {}
        strict = bool(schema_obj.get("strict", default_strict))
    else:
        name = default_name
        schema = schema_obj
        strict = default_strict

    if not schema:
        return None

    return {
        "type": "json_schema",
        "name": str(name),
        "schema": schema,
        "strict": strict,
    }
