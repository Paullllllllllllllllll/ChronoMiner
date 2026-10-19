"""Schema resolution: a path, a file name in schemas/, or a schema's name field."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronominer.spec import (
    OPTIONS,
    SCHEMAS_DIR,
    SpecError,
    resolve_schema,
    schema_names,
)

EXAMPLES = {
    "bibliographic_catalogue_entries": "BibliographicCatalogueEntries",
    "census_households": "CensusHouseholds",
    "correspondence_metadata": "CorrespondenceMetadata",
    "parish_register_events": "ParishRegisterEvents",
    "price_list_entries": "PriceListEntries",
    "probate_inventories": "ProbateInventories",
    "trade_records": "TradeRecords",
    "wage_records": "WageRecords",
}


def _schema(folder: Path, file_name: str, name: str | None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {"schema": {"type": "object"}}
    if name is not None:
        data["name"] = name
    path = folder / file_name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_the_schema_folder_is_the_repository_root_folder() -> None:
    assert Path(__file__).resolve().parents[2] / "schemas" == SCHEMAS_DIR


@pytest.mark.parametrize(("stem", "name"), sorted(EXAMPLES.items()))
def test_the_eight_examples_resolve_by_their_name_field(stem: str, name: str) -> None:
    resolved = resolve_schema(name)
    assert resolved.name == name
    assert resolved.path == SCHEMAS_DIR / f"{stem}.json"
    assert resolved.schema["name"] == name
    assert resolve_schema(stem).path == resolved.path


def test_the_examples_are_listed_by_name() -> None:
    assert set(EXAMPLES.values()) <= set(schema_names())
    suggest = OPTIONS["schema"].choice_provider
    assert suggest is not None
    assert set(EXAMPLES.values()) <= set(suggest())


def test_an_existing_path_wins(tmp_path: Path) -> None:
    folder = tmp_path / "schemas"
    _schema(folder, "ledger.json", "Ledger")
    own = _schema(tmp_path / "elsewhere", "ledger.json", "OwnLedger")

    resolved = resolve_schema(str(own), folder)

    assert (resolved.name, resolved.path) == ("OwnLedger", own)


def test_a_file_name_in_the_folder_resolves_with_or_without_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "schemas"
    path = _schema(folder, "ledger.json", "Ledger")
    elsewhere = tmp_path / "work"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert resolve_schema("ledger", folder).path == path
    assert resolve_schema("ledger.json", folder).path == path


def test_a_file_name_comes_before_a_name_field(tmp_path: Path) -> None:
    folder = tmp_path / "schemas"
    by_file = _schema(folder, "Ledger.json", "Accounts")
    _schema(folder, "other.json", "Ledger")

    assert resolve_schema("Ledger", folder).path == by_file


def test_the_name_field_resolves_a_schema_of_another_file_name(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "schemas"
    path = _schema(folder, "guide_light.json", "TravelGuidesLight")

    resolved = resolve_schema("TravelGuidesLight", folder)

    assert (resolved.name, resolved.path) == ("TravelGuidesLight", path)


def test_a_schema_without_a_name_field_is_named_by_its_stem(tmp_path: Path) -> None:
    folder = tmp_path / "schemas"
    _schema(folder, "plain.json", None)
    assert resolve_schema("plain", folder).name == "plain"
    assert schema_names(folder) == ("plain",)


def test_an_unknown_name_lists_the_available_schemas(tmp_path: Path) -> None:
    folder = tmp_path / "schemas"
    _schema(folder, "a.json", "Alpha")
    _schema(folder, "b.json", "Beta")
    (folder / "broken.json").write_text("{", encoding="utf-8")

    with pytest.raises(SpecError, match="available: Alpha, Beta"):
        resolve_schema("Gamma", folder)


def test_a_broken_schema_file_is_an_error(tmp_path: Path) -> None:
    folder = tmp_path / "schemas"
    folder.mkdir()
    (folder / "broken.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(SpecError, match="JSON object"):
        resolve_schema("broken", folder)


def test_a_path_into_a_subfolder_is_not_a_folder_name(tmp_path: Path) -> None:
    folder = tmp_path / "schemas"
    _schema(folder / "sub", "x.json", "X")
    with pytest.raises(SpecError, match="unknown schema"):
        resolve_schema("sub/x", folder)
