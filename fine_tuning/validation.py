from __future__ import annotations

from typing import Any, Dict, Iterable


def validate_top_level_output(output: Dict[str, Any], schema_wrapper: Dict[str, Any]) -> None:
    if not isinstance(output, dict):
        raise ValueError(f"Output must be a JSON object (got {type(output).__name__})")

    root_schema = schema_wrapper.get("schema")
    if not isinstance(root_schema, dict):
        return

    required: Iterable[str] = root_schema.get("required") or []
    if not isinstance(required, list):
        required = []

    for key in required:
        if key not in output:
            raise ValueError(f"Missing required top-level key: {key}")

    properties = root_schema.get("properties")
    additional_properties = root_schema.get("additionalProperties", True)
    if additional_properties is False and isinstance(properties, dict):
        allowed = set(properties.keys())
        extra = sorted(set(output.keys()) - allowed)
        if extra:
            raise ValueError(f"Unexpected top-level key(s): {', '.join(extra)}")

    if "contains_no_content_of_requested_type" in output:
        val = output.get("contains_no_content_of_requested_type")
        if not isinstance(val, bool):
            raise ValueError(
                "'contains_no_content_of_requested_type' must be a boolean "
                f"(got {type(val).__name__})"
            )

    if "entries" in output:
        entries = output.get("entries")
        if entries is not None and not isinstance(entries, list):
            raise ValueError(f"'entries' must be an array or null (got {type(entries).__name__})")
