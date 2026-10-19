"""The eight example schemas shipped in ``schemas/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"
EXPECTED = {
    "price_list_entries": "PriceListEntries",
    "wage_records": "WageRecords",
    "trade_records": "TradeRecords",
    "probate_inventories": "ProbateInventories",
    "census_households": "CensusHouseholds",
    "parish_register_events": "ParishRegisterEvents",
    "bibliographic_catalogue_entries": "BibliographicCatalogueEntries",
    "correspondence_metadata": "CorrespondenceMetadata",
}
ADMINISTRATIVE_FIELDS = {"source_page", "notes"}


def _load(stem: str) -> dict[str, Any]:
    path = SCHEMAS_DIR / f"{stem}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _objects(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            found.append(node)
        for value in node.values():
            found.extend(_objects(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_objects(value))
    return found


@pytest.mark.unit
@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_example_schema_header(stem: str) -> None:
    document = _load(stem)
    assert document["name"] == EXPECTED[stem]
    assert document["schema_version"] == "1.0"
    assert document["type"] == "json_schema"
    assert document["strict"] is True
    description = document.get("description")
    assert isinstance(description, str) and description.strip()


@pytest.mark.unit
@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_example_schema_entry_fields(stem: str) -> None:
    root = _load(stem)["schema"]
    assert set(root["properties"]) == {
        "contains_no_content_of_requested_type",
        "entries",
    }
    assert root["properties"]["contains_no_content_of_requested_type"]["type"] == (
        "boolean"
    )
    item = root["properties"]["entries"]["items"]
    fields = set(item["properties"])
    assert fields >= ADMINISTRATIVE_FIELDS
    core = fields - ADMINISTRATIVE_FIELDS
    assert 5 <= len(core) <= 10, sorted(core)
    for name, spec in item["properties"].items():
        assert spec.get("description", "").strip(), name


@pytest.mark.unit
@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_example_schema_satisfies_strict_rules(stem: str) -> None:
    violations: list[str] = []
    for node in _objects(_load(stem)["schema"]):
        properties = node.get("properties", {})
        if node.get("additionalProperties") is not False:
            violations.append(f"additionalProperties not false: {sorted(properties)}")
        if set(node.get("required", [])) != set(properties):
            violations.append(f"required differs from properties: {sorted(properties)}")
    assert not violations, "\n".join(violations)


@pytest.mark.unit
def test_nested_structures_follow_the_source_type() -> None:
    probate = _load("probate_inventories")["schema"]["properties"]["entries"]["items"]
    assert probate["properties"]["items"]["type"] == "array"
    assert set(probate["properties"]["items"]["items"]["properties"]) == {
        "description",
        "quantity",
        "appraised_value",
    }
    parish = _load("parish_register_events")["schema"]["properties"]["entries"]["items"]
    persons = parish["properties"]["persons"]["items"]["properties"]
    assert set(persons) == {"name", "role"}
