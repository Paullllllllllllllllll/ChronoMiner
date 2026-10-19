"""Characterization of the rendered outputs of the extract command.

The ``_output.json`` shape (``_chronominer_metadata`` and ``records`` with
``chunk_index``, ``custom_id``, ``response.output_text`` and ``usage``), the
generic CSV of the flat and the nested fixture schema (nested objects as
dotted columns, lists joined), a run without ``--formats csv``, and a
failed chunk as the output records it. Each
case runs twice in fresh roots; both snapshots must match before the run is
compared with ``golden/outputs/<case>/``.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.testing import WriteGuard
from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.conftest import CaseRoot
from tests.characterization.fakes import (
    USAGE,
    FakeLLM,
    Reply,
    permanent_error,
    record_unit,
)
from tests.characterization.inputs import make_text_file
from tests.characterization.stages import single_case_runner, stage

pytestmark = pytest.mark.integration

CASE_PREFIX = "outputs"
RECORDS = 12

Build = Callable[[Path], RunResult]

NESTED_PAYLOAD: dict[str, Any] = {
    "contains_no_content_of_requested_type": False,
    "entries": [
        {
            "name": "Granary of Basel",
            "place": {"town": "Basel", "region": None},
            "tags": ["grain", "storage", "town"],
            "source_page": "12",
            "notes": None,
        },
        {
            "name": "Mill at Liestal",
            "place": {"town": "Liestal", "region": "Basel-Land"},
            "tags": [],
            "source_page": None,
            "notes": "Rebuilt after the flood.",
        },
    ],
}


@pytest.fixture
def output_case(
    fake_llm: FakeLLM, case_root: CaseRoot, write_guard: WriteGuard
) -> Callable[[str, Build], RunResult]:
    """Return ``run(case, build)``: run twice, compare, check the golden files."""
    runner = single_case_runner(
        CASE_PREFIX, case_root, write_guard, before_each=fake_llm.units.clear
    )
    return lambda case, build: runner(case, lambda root: stage(build(root))).result


def _extract(
    llm: FakeLLM,
    root: Path,
    *,
    schema: str = "FlatEntries",
    records: int = RECORDS,
    csv_output: bool = True,
) -> RunResult:
    ledger = make_text_file(root / "docs" / "ledger.txt", records)
    config: Mapping[str, Mapping[str, Any]] = {
        "paths_config": {"schemas_paths": {schema: {"csv_output": csv_output}}}
    }
    return run_tool(
        "extract",
        llm=llm,
        root=root,
        config=config,
        schema=schema,
        input=ledger,
        output="out",
    )


def _output(result: RunResult) -> dict[str, Any]:
    path = result.output_json(result.root / "docs" / "ledger.txt")
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(result: RunResult) -> list[dict[str, str]]:
    path = result.output_json(result.root / "docs" / "ledger.txt").with_suffix(".csv")
    text = path.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def test_output_json_shape(
    fake_llm: FakeLLM, output_case: Callable[[str, Build], RunResult]
) -> None:
    result = output_case("output_json_shape", lambda root: _extract(fake_llm, root))

    assert result.exit_code == 0, result.stdout + result.stderr
    data = _output(result)
    assert set(data) == {"_chronominer_metadata", "records"}
    metadata = data["_chronominer_metadata"]
    assert metadata["schema_name"] == "FlatEntries"
    assert metadata["total_chunks"] == len(data["records"])
    for index, record in enumerate(data["records"], 1):
        assert record["chunk_index"] == index
        assert record["custom_id"] == f"ledger-chunk-{index}"
        assert isinstance(json.loads(record["response"]["output_text"]), dict)
        assert record["response"]["usage"] == {
            **USAGE,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }


def test_csv_flat_schema(
    fake_llm: FakeLLM, output_case: Callable[[str, Build], RunResult]
) -> None:
    result = output_case("csv_flat", lambda root: _extract(fake_llm, root))

    assert result.exit_code == 0, result.stdout + result.stderr
    rows = _csv_rows(result)
    assert len(rows) == len(_output(result)["records"])
    assert list(rows[0]) == [
        "record_id",
        "item",
        "quantity",
        "price",
        "date",
        "source_page",
        "notes",
    ]


def test_csv_nested_schema(
    fake_llm: FakeLLM, output_case: Callable[[str, Build], RunResult]
) -> None:
    def build(root: Path) -> RunResult:
        fake_llm.script(record_unit(0), payload=NESTED_PAYLOAD)
        return _extract(fake_llm, root, schema="NestedEntries", records=3)

    result = output_case("csv_nested", build)

    assert result.exit_code == 0, result.stdout + result.stderr
    rows = _csv_rows(result)
    assert len(rows) == 2
    assert list(rows[0]) == [
        "name",
        "place.town",
        "place.region",
        "tags",
        "source_page",
        "notes",
    ]
    assert rows[0]["place.town"] == "Basel"
    assert rows[0]["tags"] == "grain; storage; town"
    assert (rows[1]["tags"], rows[1]["place.region"]) == ("", "Basel-Land")


def test_csv_output_disabled(
    fake_llm: FakeLLM, output_case: Callable[[str, Build], RunResult]
) -> None:
    result = output_case(
        "csv_disabled",
        lambda root: _extract(fake_llm, root, records=3, csv_output=False),
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert not any(path.suffix == ".csv" for path in result.created)


def test_failed_chunk_in_the_output(
    fake_llm: FakeLLM, output_case: Callable[[str, Build], RunResult]
) -> None:
    def build(root: Path) -> RunResult:
        fake_llm.script(record_unit(3), always=Reply(permanent_error()))
        return _extract(fake_llm, root)

    result = output_case("failed_chunk", build)

    assert result.exit_code == 1, result.stdout + result.stderr
    assert result.summary is not None
    assert (result.summary["items_complete"], result.summary["items_failed"]) == (
        0,
        1,
    )
    data = _output(result)
    metadata = data["_chronominer_metadata"]
    assert metadata.get("partial") is True
    assert metadata.get("failed_chunks") == [2]
    assert [record["chunk_index"] for record in data["records"]] == [1, 3, 4]
