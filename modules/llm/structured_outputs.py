"""
Structured output utilities for JSON schema formatting.

DEPRECATED: This module is largely deprecated as LangChain now handles
structured outputs internally via with_structured_output().

The functions here are kept for backward compatibility with existing code
that uses the build_structured_text_format() function. New code should
use LangChain's native structured output support instead:

    # LangChain native approach (recommended):
    llm.with_structured_output(schema_def, method="json_schema")
    
    # Or use response_format binding:
    llm.bind(response_format={"type": "json_schema", "json_schema": {...}})

LangChain handles:
- Schema validation
- Provider-specific formatting (OpenAI, Anthropic, Google)
- Automatic retry on schema validation errors
- Strict mode enforcement

These utilities remain for:
- Legacy compatibility with existing batch processing code
- Custom schema wrapping requirements
"""

from __future__ import annotations

import warnings
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
    
    .. deprecated::
        This function is deprecated. Use LangChain's native structured output
        support instead via ``llm.with_structured_output(schema)``. This
        function is kept for backward compatibility with batch processing code.

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
    warnings.warn(
        "build_structured_text_format() is deprecated. Use LangChain's "
        "with_structured_output() method instead for new code.",
        DeprecationWarning,
        stacklevel=2
    )
    name, schema, strict = _unwrap_schema(schema_obj, default_name, default_strict)
    if not isinstance(schema, dict) or not schema:
        return None

    return {
        "type": "json_schema",
        "name": name,
        "schema": schema,
        "strict": bool(strict),
    }
