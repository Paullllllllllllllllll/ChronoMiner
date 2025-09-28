from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _unwrap_schema(
    schema_obj: Dict[str, Any],
    default_name: str = "TranscriptionSchema",
    default_strict: bool = True,
) -> Tuple[str, Dict[str, Any], bool]:
    """
    Normalize a provided schema object into (name, schema, strict).

    Accepts either:
      - A wrapper dict with keys {"name", "schema", "strict"}, or
      - A bare JSON Schema dict.

    Returns:
      (name, schema, strict)
    """
    if not isinstance(schema_obj, dict) or not schema_obj:
        return default_name, {}, default_strict

    if "schema" in schema_obj and isinstance(schema_obj["schema"], dict):
        name_val = schema_obj.get("name") or default_name
        schema_val = schema_obj.get("schema") or {}
        strict_val = bool(schema_obj.get("strict", default_strict))
        return str(name_val), schema_val, strict_val

    # Bare JSON Schema object
    return default_name, schema_obj, default_strict


def build_structured_text_format(
    schema_obj: Dict[str, Any],
    default_name: str = "TranscriptionSchema",
    default_strict: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Build the Responses API `text.format` object for Structured Outputs.

    Returns:
      dict with shape:
        {
          "type": "json_schema",
          "name": <name>,
          "schema": <json schema dict>,
          "strict": <bool>
        }
      or None if the provided schema is not usable.
    """
    name, schema, strict = _unwrap_schema(schema_obj, default_name, default_strict)
    if not isinstance(schema, dict) or not schema:
        return None

    return {
        "type": "json_schema",
        "name": name,
        "schema": schema,
        "strict": bool(strict),
    }
