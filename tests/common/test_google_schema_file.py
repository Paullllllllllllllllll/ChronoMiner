"""A Gemini batch part whose requests carry a response schema goes as a file.

The Batch API ignores the schema of an inline request; the uploaded JSONL
file enforces it (live probe of round 5).
"""

from __future__ import annotations

from typing import Any

from chronominer.common.batch_google import file_request, goes_inline


def _entry(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": "p1",
        "request": {
            "model": "gemini-3.1-flash-lite",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "config": config,
        },
    }


def test_a_part_without_a_schema_goes_inline() -> None:
    assert goes_inline([_entry({"max_output_tokens": 64})])


def test_a_part_with_a_schema_goes_as_a_file() -> None:
    schema = {"type": "object", "properties": {"word": {"type": "string"}}}
    entry = _entry(
        {
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            "thinking_config": {"thinking_level": "low"},
        }
    )

    assert not goes_inline([entry])
    rest = file_request(entry["request"])
    assert rest["generation_config"]["response_json_schema"] == schema
    assert rest["generation_config"]["thinking_config"] == {"thinking_level": "low"}
