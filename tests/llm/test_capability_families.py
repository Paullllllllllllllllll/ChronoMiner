"""The capability table reproduces the recorded capability of every model id.

``data/capability_expectations.json`` records the shared table of
``common/capability_table.py`` (regenerated when the tables of the three
tools merged and the vendor facts were settled); this tool adds no audio
flags. Ids are matched after a ``provider:`` or ``models/`` prefix is
removed. The rows hold one id per family, a few dated and dash-spelled
variants, the prefix-override ids, OpenRouter ``vendor/model`` ids and
unknown ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chronominer.llm.capabilities import (
    detect_capabilities,
    detect_provider,
)

_DATA = Path(__file__).parent / "data" / "capability_expectations.json"
_ROWS: list[dict[str, Any]] = json.loads(_DATA.read_text(encoding="utf-8"))


def _recorded(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


@pytest.mark.parametrize("row", _ROWS, ids=[repr(row["model"]) for row in _ROWS])
def test_family_table_matches_recorded_capabilities(row: dict[str, Any]) -> None:
    expected = dict(row)
    model = expected.pop("model")
    provider = expected.pop("provider")
    extras = expected.pop("extras")
    caps = detect_capabilities(model)

    assert detect_provider(model) == provider
    assert caps.model_name == model
    assert {name: _recorded(getattr(caps, name)) for name in expected} == expected
    assert dict(caps.extras) == extras


def test_expectations_cover_every_kind_of_id() -> None:
    models = [row["model"] for row in _ROWS]

    assert any(m.startswith("openai:") for m in models)
    assert any(m.startswith("models/gemini") for m in models)
    assert sum("/" in m for m in models) > 10
    assert any(row["family"] == "unknown" for row in _ROWS)
    assert not any(set(row["extras"]) - {"image_original_max_side_px"} for row in _ROWS)


def test_unknown_model_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        detect_capabilities("never-seen-family-model-x9")
        detect_capabilities("never-seen-family-model-x9")

    hits = [r for r in caplog.records if "never-seen-family-model-x9" in r.message]
    assert len(hits) == 1
