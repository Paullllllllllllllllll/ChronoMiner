"""Golden JSON files of request and image-setting matrices.

:func:`jsonable` turns request kwargs, batch bodies and image settings into
JSON that keeps what a byte comparison of the requests would see: key order,
the difference between ``1`` and ``1.0`` and between a tuple and a list. Long
strings (prompts, schemas rendered into prompts) become a digest with their
length. :func:`check_matrix` compares a case mapping with
``golden/<name>.json`` beside the calling test and rewrites the file when
``CM_UPDATE_GOLDEN=1``.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

UPDATE_ENV = "CM_UPDATE_GOLDEN"
LONG_TEXT = 200
DIGESTED_KEYS = frozenset({"schema", "response_json_schema"})
"""Keys whose values are recorded as a digest: schemas sent with every body."""


def digest(value: Any) -> str:
    """Return a short digest of *value*'s canonical JSON."""
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def jsonable(value: Any, key: str | None = None) -> Any:
    """Return *value* as golden JSON; see the module docstring."""
    if key in DIGESTED_KEYS and isinstance(value, Mapping):
        return {"<digest>": digest(jsonable(value))}
    if isinstance(value, httpx.Timeout):
        return {
            "<httpx.Timeout>": {
                "connect": value.connect,
                "read": value.read,
                "write": value.write,
                "pool": value.pool,
            }
        }
    if isinstance(value, Mapping):
        return {str(k): jsonable(item, str(k)) for k, item in value.items()}
    if isinstance(value, tuple):
        return {"<tuple>": [jsonable(item) for item in value]}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, str) and len(value) > LONG_TEXT:
        return f"<text {len(value)} chars {digest(value)}>"
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return f"<{type(value).__name__}>"


def _dumps(cases: Mapping[str, Any]) -> str:
    """Return *cases* as a JSON object with one case per line."""
    lines = [
        f"{json.dumps(case)}: {json.dumps(value, ensure_ascii=False)}"
        for case, value in cases.items()
    ]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def check_matrix(
    test_file: str,
    name: str,
    cases: Mapping[str, Any],
    consume: Callable[[Path], object],
) -> None:
    """Compare *cases* with ``golden/<name>.json`` beside *test_file*.

    With ``CM_UPDATE_GOLDEN=1`` the file is rewritten and *consume* (the
    write guard's) receives its folder.
    """
    folder = Path(test_file).resolve().parent / "golden"
    golden = folder / f"{name}.json"
    actual = _dumps({case: jsonable(value) for case, value in cases.items()})
    if os.environ.get(UPDATE_ENV) == "1":
        folder.mkdir(parents=True, exist_ok=True)
        golden.write_text(actual, encoding="utf-8", newline="\n")
        consume(folder)
        return
    if not golden.is_file():
        raise AssertionError(f"No golden {golden.name}; run with {UPDATE_ENV}=1.")
    expected = golden.read_text(encoding="utf-8")
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=f"golden/{name}.json",
                tofile="actual",
                n=8,
            )
        )
        raise AssertionError(f"{name} changed (set {UPDATE_ENV}=1 to accept):\n{diff}")
