"""Repo-wide invariants for the shipped JSON schemas in ``schemas/``.

Both rules encode constraints of constrained decoding rather than of JSON
Schema itself, so a violation is silent until a provider rejects the request
(or, worse, until a field turns out to be unreachable at generation time):

1. OpenAI strict structured outputs require ``required`` to list EVERY key of
   an object's ``properties``. Optionality is expressed by making the value
   nullable, not by omitting it from ``required``.
2. A nullable enum must contain ``null`` among its ``enum`` values, otherwise
   the "use null if unknown" instruction in the field description can never
   be followed under constrained decoding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"
SCHEMA_FILES = sorted(SCHEMAS_DIR.glob("*.json"))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk(node: Any, path: str) -> list[tuple[str, Any]]:
    """Yield ``(json_path, node)`` for every dict in the schema tree."""
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        found.append((path or "$", node))
        for key, value in node.items():
            found.extend(_walk(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_walk(value, f"{path}[{index}]"))
    return found


def test_schema_files_present() -> None:
    assert SCHEMA_FILES, "no schemas found — the invariant tests would be vacuous"


@pytest.mark.unit
@pytest.mark.parametrize("schema_path", SCHEMA_FILES, ids=lambda p: p.name)
def test_strict_schema_requires_every_property(schema_path: Path) -> None:
    document = _load(schema_path)
    if not document.get("strict", False):
        pytest.skip(f"{schema_path.name} is not a strict schema")

    violations: list[str] = []
    for json_path, node in _walk(document, ""):
        properties = node.get("properties")
        if node.get("type") != "object" or not isinstance(properties, dict):
            continue
        required = set(node.get("required", []) or [])
        missing = sorted(set(properties) - required)
        if missing:
            violations.append(f"{json_path}: missing from 'required': {missing}")

    assert not violations, (
        f"{schema_path.name} violates OpenAI strict structured outputs:\n"
        + "\n".join(violations)
    )


@pytest.mark.unit
@pytest.mark.parametrize("schema_path", SCHEMA_FILES, ids=lambda p: p.name)
def test_nullable_enums_include_null(schema_path: Path) -> None:
    document = _load(schema_path)

    violations: list[str] = []
    for json_path, node in _walk(document, ""):
        enum = node.get("enum")
        type_ = node.get("type")
        if not isinstance(enum, list) or not isinstance(type_, list):
            continue
        if "null" in type_ and None not in enum:
            violations.append(f"{json_path}: type allows null but enum does not")

    assert not violations, (
        f"{schema_path.name} declares unreachable null values:\n"
        + "\n".join(violations)
    )
