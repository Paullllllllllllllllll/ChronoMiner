"""``classify entries``: units, leases, states and the schema filter."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from chronominer import __main__ as entry
from chronominer.classify import plan_classify, run_classify
from chronominer.output.store import database_path
from chronominer.run import ConfigurationError
from chronominer.settings import Settings
from chronominer.spec import resolve_classify_entries
from tests.classify.conftest import (
    SCHEMA,
    Chunk,
    Classify,
    DecisionServer,
    classifier_file,
    seed,
    view,
    write_text,
)
from tests.cli.conftest import json_lines

BREAD = {"item": "bread", "unit": "lb", "price": 4.0, "notes": "wheaten"}
BEEF = {"item": "beef", "unit": "lb", "price": 6.0}
SALT = {"item": "salt", "unit": "bushel", "price": 2.0}
CHUNKS = (Chunk(1, 20, entries=(BREAD, BEEF)), Chunk(21, 40, entries=(SALT,)))


def test_the_entries_of_a_finished_document_are_classified_under_its_doc_id(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "in" / "prices.txt")
    out = tmp_path / "out"
    doc_id = seed(out, text, CHUNKS)

    code, summary = classify(
        "entries", "--classifier", "commodity_group", "--input", out
    )

    assert code == 0, summary
    assert (summary["unit"], summary["items_classified"]) == ("entry", 1)
    assert summary["units_asked"] == 3
    found = view(out)
    assert [(row["locator"], row["ordinal"]) for row in found] == [
        ("1:0", 1),
        ("1:1", 1),
        ("2:0", 2),
    ]
    assert {row["doc_id"] for row in found} == {doc_id}
    assert {row["unit"] for row in found} == {"entry"}
    assert len(server.requests) == 2  # entries are never grouped across chunks
    context, label, first = server.parts(0)[:3]
    assert json.loads(context["text"]) == {
        "document": "prices",
        "schema": "PriceListEntries",
    }
    assert label["text"] == "Entry 1:"
    assert json.loads(first["text"]) == {
        "item": "bread",
        "unit": "lb",
        "notes": "wheaten",
    }
    second = json.loads(server.parts(0)[4]["text"])
    assert second == {"item": "beef", "unit": "lb", "notes": None}


def test_a_document_leased_by_another_run_is_blocked(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    doc_id = seed(tmp_path, text, CHUNKS)
    with contextlib.closing(sqlite3.connect(database_path(tmp_path))) as connection:
        connection.execute(
            "UPDATE documents SET lease_run_id = 'killed-run', lease_until = ? "
            "WHERE doc_id = ?",
            (time.time() + 600, doc_id),
        )
        connection.commit()

    code, summary = classify(
        "entries", "--classifier", "commodity_group", "--input", tmp_path
    )

    assert code == 1
    assert summary["items_blocked"] == 1
    assert server.requests == []


def test_an_unfinished_document_is_skipped(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    seed(tmp_path, text, (CHUNKS[0], Chunk(21, 40, status="pending")))

    code, summary = classify(
        "entries", "--classifier", "commodity_group", "--input", tmp_path
    )

    assert code == 0
    assert (summary["items_skipped"], summary["units_asked"]) == (1, 0)
    assert server.requests == []


def test_a_document_with_a_failed_chunk_is_finished(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    seed(tmp_path, text, (CHUNKS[0], Chunk(21, 40, status="failed")))

    code, summary = classify(
        "entries", "--classifier", "commodity_group", "--input", tmp_path
    )

    assert code == 0
    assert summary["units_asked"] == 2


def test_no_document_of_the_schema_exits_2(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    seed(tmp_path, text, CHUNKS)

    code, summary = classify(
        "entries",
        "--classifier",
        "occupation_sector",
        "--input",
        database_path(tmp_path),
    )

    assert code == 2
    assert summary["items_total"] == 0


@pytest.mark.parametrize(
    ("change", "state"),
    [
        ("UPDATE documents SET input_sha256 = 'other'", "refused"),
        ("UPDATE units SET status = 'pending' WHERE idx = 1", "skipped"),
        (
            "UPDATE document_meta SET metadata = "
            "json_set(metadata, '$.schema_name', 'WageRecords')",
            "skipped",
        ),
        ("UPDATE documents SET name = 'renamed'", "skipped"),
        ("DELETE FROM documents", "skipped"),
    ],
    ids=["input", "status", "schema", "renamed", "dropped"],
)
def test_a_document_changed_after_planning_is_checked_again_under_the_lease(
    tmp_path: Path, server: DecisionServer, change: str, state: str
) -> None:
    text = write_text(tmp_path / "prices.txt")
    seed(tmp_path, text, CHUNKS)
    flags = {"input": tmp_path, "classifier": "commodity_group"}
    plan = plan_classify(resolve_classify_entries(flags).spec, Settings())
    with contextlib.closing(sqlite3.connect(database_path(tmp_path))) as connection:
        connection.execute(change)
        connection.execute(
            "UPDATE documents SET status = CASE WHEN EXISTS (SELECT 1 FROM units "
            "WHERE units.doc_id = documents.doc_id AND status = 'pending') "
            "THEN 'partial' ELSE status END"
        )
        connection.commit()

    result = asyncio.run(run_classify(plan))

    assert [target.state for target in result.targets] == [state]
    assert server.requests == []


def _classify_entries(
    capsys: pytest.CaptureFixture[str], *argv: str | Path
) -> tuple[int, dict[str, Any], str]:
    """Run ``classify entries --json``; return the exit code, summary and stderr."""
    code = entry.main(["classify", "entries", *(str(arg) for arg in argv), "--json"])
    captured = capsys.readouterr()
    (summary,) = json_lines(captured.out)
    return code, summary, captured.err


PROJECT_SCHEMA = "ProjectPriceEntries"
"""An invented schema name that only a user schema folder holds."""


def _project_folders(tmp_path: Path, fields: list[str]) -> tuple[Path, Path]:
    """Write a user schema folder and an entry classifier naming its schema."""
    schemas = tmp_path / "project" / "schemas"
    schemas.mkdir(parents=True)
    body = {**SCHEMA.schema, "name": PROJECT_SCHEMA}
    (schemas / "project_prices.json").write_text(json.dumps(body), encoding="utf-8")
    classifiers = tmp_path / "project" / "classifiers"
    classifier_file(
        classifiers,
        {
            "name": "project_group",
            "version": 1,
            "unit": "entry",
            "schema": PROJECT_SCHEMA,
            "fields": fields,
            "questions": [
                {"name": "bread", "type": "predicate", "instructions": "Bread?"}
            ],
        },
    )
    return schemas, classifiers


def _seed_project(tmp_path: Path) -> Path:
    text = write_text(tmp_path / "in" / "prices.txt")
    out = tmp_path / "out"
    seed(out, text, CHUNKS)
    with contextlib.closing(sqlite3.connect(database_path(out))) as connection:
        connection.execute(
            "UPDATE document_meta SET metadata = "
            "json_set(metadata, '$.schema_name', ?)",
            (PROJECT_SCHEMA,),
        )
        connection.commit()
    return out


def test_an_entry_classifier_finds_its_schema_in_a_schemas_dir(
    tmp_path: Path, server: DecisionServer, capsys: pytest.CaptureFixture[str]
) -> None:
    schemas, classifiers = _project_folders(tmp_path, ["item", "unit"])
    out = _seed_project(tmp_path)

    code, summary, err = _classify_entries(
        capsys,
        "--classifier",
        "project_group",
        "--classifiers-dir",
        classifiers,
        "--schemas-dir",
        schemas,
        "--input",
        out,
    )

    assert code == 0, summary
    assert (summary["classifier"], summary["units_asked"]) == ("project_group", 3)


def test_fields_the_user_folder_schema_lacks_are_refused(
    tmp_path: Path, server: DecisionServer, capsys: pytest.CaptureFixture[str]
) -> None:
    schemas, classifiers = _project_folders(tmp_path, ["item", "vintage"])
    out = _seed_project(tmp_path)

    code, summary, err = _classify_entries(
        capsys,
        "--classifier",
        "project_group",
        "--classifiers-dir",
        classifiers,
        "--schemas-dir",
        schemas,
        "--input",
        out,
    )

    assert code == 2
    assert "vintage" in err
    assert server.requests == []


def test_the_settings_folders_serve_classify_entries(
    tmp_path: Path, server: DecisionServer
) -> None:
    schemas, classifiers = _project_folders(tmp_path, ["item", "vintage"])
    out = _seed_project(tmp_path)
    settings = Settings(schemas_dirs=(schemas,), classifiers_dirs=(classifiers,))
    flags = {"input": out, "classifier": "project_group"}

    with pytest.raises(ConfigurationError, match="lacks: vintage"):
        plan_classify(resolve_classify_entries(flags).spec, settings)


def test_a_schema_name_in_two_user_folders_is_an_error(
    tmp_path: Path, server: DecisionServer, capsys: pytest.CaptureFixture[str]
) -> None:
    schemas, classifiers = _project_folders(tmp_path, ["item"])
    other = tmp_path / "other"
    other.mkdir()
    (other / "copy.json").write_text(
        (schemas / "project_prices.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    out = _seed_project(tmp_path)

    code, summary, err = _classify_entries(
        capsys,
        "--classifier",
        "project_group",
        "--classifiers-dir",
        classifiers,
        "--schemas-dir",
        schemas,
        "--schemas-dir",
        other,
        "--input",
        out,
    )

    assert code == 2
    assert "ambiguous" in err
    assert str(other / "copy.json") in err
    assert server.requests == []
