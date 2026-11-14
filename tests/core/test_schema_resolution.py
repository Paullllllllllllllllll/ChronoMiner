"""Schema resolution: a path, or a file name or name field in the schema folders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronominer.spec import (
    OPTIONS,
    SCHEMAS_DIR,
    AmbiguousSchemaError,
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

    resolved = resolve_schema(str(own), bundled=folder)

    assert (resolved.name, resolved.path) == ("OwnLedger", own)


def test_a_file_name_in_the_folder_resolves_with_or_without_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "schemas"
    path = _schema(folder, "ledger.json", "Ledger")
    elsewhere = tmp_path / "work"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert resolve_schema("ledger", bundled=folder).path == path
    assert resolve_schema("ledger.json", bundled=folder).path == path


def test_a_file_name_comes_before_a_name_field(tmp_path: Path) -> None:
    folder = tmp_path / "schemas"
    by_file = _schema(folder, "Ledger.json", "Accounts")
    _schema(folder, "other.json", "Ledger")

    assert resolve_schema("Ledger", bundled=folder).path == by_file


def test_the_name_field_resolves_a_schema_of_another_file_name(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "schemas"
    path = _schema(folder, "guide_light.json", "TravelGuidesLight")

    resolved = resolve_schema("TravelGuidesLight", bundled=folder)

    assert (resolved.name, resolved.path) == ("TravelGuidesLight", path)


def test_a_schema_without_a_name_field_is_named_by_its_stem(tmp_path: Path) -> None:
    folder = tmp_path / "schemas"
    _schema(folder, "plain.json", None)
    assert resolve_schema("plain", bundled=folder).name == "plain"
    assert schema_names(bundled=folder) == ("plain",)


def test_an_unknown_name_lists_the_available_schemas(tmp_path: Path) -> None:
    folder = tmp_path / "schemas"
    _schema(folder, "a.json", "Alpha")
    _schema(folder, "b.json", "Beta")
    (folder / "broken.json").write_text("{", encoding="utf-8")

    with pytest.raises(SpecError, match="available: Alpha, Beta"):
        resolve_schema("Gamma", bundled=folder)


def test_a_broken_schema_file_is_an_error(tmp_path: Path) -> None:
    folder = tmp_path / "schemas"
    folder.mkdir()
    (folder / "broken.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(SpecError, match="JSON object"):
        resolve_schema("broken", bundled=folder)


def test_a_path_into_a_subfolder_is_not_a_folder_name(tmp_path: Path) -> None:
    folder = tmp_path / "schemas"
    _schema(folder / "sub", "x.json", "X")
    with pytest.raises(SpecError, match="unknown schema"):
        resolve_schema("sub/x", bundled=folder)


def test_a_user_folder_is_searched_before_the_bundled_folder(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    _schema(bundled, "ledger.json", "Ledger")
    user = tmp_path / "project"
    own = _schema(user, "tally.json", "Ledger")

    assert resolve_schema("Ledger", [user], bundled=bundled).path == own
    assert schema_names([user], bundled=bundled) == ("Ledger",)


def test_a_user_file_shadows_the_bundled_file_of_its_file_name(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    _schema(bundled, "accounts.json", "Ledger")
    _schema(bundled, "wages.json", "Wages")
    user = tmp_path / "project"
    own = _schema(user, "accounts.json", "Books")

    assert resolve_schema("accounts", [user], bundled=bundled).path == own
    with pytest.raises(SpecError, match="unknown schema"):
        resolve_schema("Ledger", [user], bundled=bundled)
    assert schema_names([user], bundled=bundled) == ("Books", "Wages")


@pytest.mark.parametrize(
    ("second_file", "second_name"),
    [("ledger.json", "Other"), ("tally.json", "ledger")],
    ids=["file-name", "name-field"],
)
def test_a_name_in_two_user_folders_is_ambiguous(
    tmp_path: Path, second_file: str, second_name: str
) -> None:
    first = _schema(tmp_path / "one", "ledger.json", "Ledger")
    second = _schema(tmp_path / "two", second_file, second_name)

    with pytest.raises(AmbiguousSchemaError) as caught:
        resolve_schema("ledger", [first.parent, second.parent], bundled=tmp_path)

    assert isinstance(caught.value, SpecError)
    assert caught.value.paths == (first, second)
    assert str(first) in str(caught.value)
    assert str(second) in str(caught.value)


def test_a_folder_named_twice_is_searched_once(tmp_path: Path) -> None:
    path = _schema(tmp_path / "project", "ledger.json", "Ledger")
    folders = [path.parent, tmp_path / "project" / ".." / "project"]

    assert resolve_schema("Ledger", folders, bundled=tmp_path / "none").path == path


def test_a_file_path_wins_over_the_user_folders(tmp_path: Path) -> None:
    _schema(tmp_path / "one", "ledger.json", "Ledger")
    _schema(tmp_path / "two", "ledger.json", "Ledger")
    own = _schema(tmp_path / "elsewhere", "ledger.json", "Own")
    folders = [tmp_path / "one", tmp_path / "two"]

    assert resolve_schema(str(own), folders).path == own


def test_an_unknown_name_names_every_folder_searched(tmp_path: Path) -> None:
    user = tmp_path / "project"
    _schema(user, "a.json", "Alpha")
    bundled = tmp_path / "bundled"
    _schema(bundled, "b.json", "Beta")

    with pytest.raises(SpecError) as caught:
        resolve_schema("Gamma", [user], bundled=bundled)

    message = str(caught.value)
    assert f"{user}, {bundled}" in message
    assert "available: Alpha, Beta" in message
