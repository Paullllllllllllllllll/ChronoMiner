"""Resume matrix of the run core on the store, through ``plan()`` and ``run()``.

Run on Windows with (``<repo>`` is the repository root):

    uv --directory <repo> run pytest tests/core/test_run_store.py -q

A scripted chat model answers every chunk (``tests.llm.helpers``), so no
request leaves the machine; ``script.requests`` counts the model calls. The
oracles read the store through ``mode=ro`` connections.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

import chronominer.run as core
from chronominer import __main__ as entry
from chronominer.common.contract import ExitCode
from chronominer.common.sqlite import read, read_rows
from chronominer.common.store import UnitKey
from chronominer.output.store import DATABASE_NAME, open_store
from chronominer.spec import to_argv
from tests.conftest import PROJECT_ROOT, make_settings, make_spec
from tests.llm import helpers
from tests.llm.helpers import Script, StatusError
from tests.recording_events import RecordingEvents

ANSWER: dict[str, Any] = {
    "contains_no_content_of_requested_type": False,
    "entries": [
        {
            "person": "John Carter",
            "occupation": "mason",
            "wage": 2.5,
            "period": "per day",
            "currency": "shillings",
            "date": "1791",
            "employer": None,
            "source_page": None,
            "notes": None,
        }
    ],
}
OTHER_MODEL = "gpt-5.6-terra"


def _pdf(path: Path, pages: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=200, height=300)
        page.insert_text((50, 100), f"Page {index + 1}")
    document.save(path)
    document.close()
    return path


def _text(path: Path, lines: int = 5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{n}. John Carter, mason, 2s 6d" for n in range(lines))
    path.write_text(body + "\n", encoding="utf-8")
    return path


class Scenario:
    """One PDF, one output folder and the scripted model of a test."""

    def __init__(self, tmp_path: Path, script: Script, pages: int = 3) -> None:
        self.pdf = _pdf(tmp_path / "in" / "doc.pdf", pages)
        self.out = tmp_path / "out"
        self.script = script
        self.events = RecordingEvents()

    @property
    def db(self) -> Path:
        return self.out / DATABASE_NAME

    def plan(self, **flags: Any) -> core.RunPlan:
        flags.setdefault("concurrency", 1)
        spec = make_spec(self.pdf, output=self.out, **flags)
        return core.plan(spec, make_settings(rate_limits=((100_000, 1),)))

    def run(self, **flags: Any) -> core.RunResult:
        self.events = RecordingEvents()
        return asyncio.run(core.run(self.plan(**flags), self.events))

    @property
    def calls(self) -> int:
        return len(self.script.requests)

    def units(self) -> list[dict[str, Any]]:
        return read_rows(self.db, "units", order_by=["idx"])

    def statuses(self) -> list[str]:
        return [row["status"] for row in self.units()]

    def document_status(self) -> str:
        rows = read(
            self.db,
            lambda connection: connection.execute(
                "SELECT status FROM documents"
            ).fetchall(),
        )
        ((status,),) = rows
        return str(status)

    def entries(self) -> int:
        (count,) = read(
            self.db,
            lambda connection: connection.execute(
                "SELECT count(*) FROM entries"
            ).fetchone(),
        )
        return int(count)

    def notes(self) -> list[str]:
        return [row["message"] for row in read_rows(self.db, "notes")]

    def output(self) -> Path:
        return self.out / "doc_output.json"

    def records(self) -> list[dict[str, Any]]:
        data = json.loads(self.output().read_text(encoding="utf-8"))
        return list(data["records"])


@pytest.fixture(autouse=True)
def openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable of the default provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> Script:
    """Answer every chunk with the same entry."""
    return helpers.install(monkeypatch, [ANSWER])


@pytest.fixture
def scenario(tmp_path: Path, script: Script) -> Scenario:
    return Scenario(tmp_path, script)


@pytest.fixture
def six(tmp_path: Path, script: Script) -> Scenario:
    return Scenario(tmp_path, script, pages=6)


def test_a_fresh_run_writes_one_unit_per_chunk_and_renders(
    scenario: Scenario,
) -> None:
    result = scenario.run()

    assert (result.complete, result.failed) == (1, 0)
    assert scenario.calls == 3
    rows = scenario.units()
    assert [(row["kind"], row["idx"], row["status"]) for row in rows] == [
        ("chunk", 0, "done"),
        ("chunk", 1, "done"),
        ("chunk", 2, "done"),
    ]
    assert [row["label"] for row in rows] == [f"page_000{n}" for n in (1, 2, 3)]
    assert [row["source_index"] for row in rows] == [0, 1, 2]
    assert len(read_rows(scenario.db, "responses")) == 3
    assert scenario.entries() == 3
    assert [record["chunk_index"] for record in scenario.records()] == [1, 2, 3]
    assert result.items[0].outputs == (str(scenario.output().resolve()),)
    assert result.databases == (str(scenario.db.resolve()),)
    (run_row,) = read_rows(scenario.db, "runs")
    assert (run_row["command"], run_row["exit_code"]) == ("extract", 0)
    kinds = {row["kind"] for row in read_rows(scenario.db, "artifacts")}
    assert {"prompt", "schema"} <= kinds
    assert scenario.document_status() == "complete"


def test_a_second_run_skips_the_item_without_model_calls(
    scenario: Scenario,
) -> None:
    scenario.run()
    before = scenario.output().read_bytes()

    run_plan = scenario.plan()
    result = scenario.run()

    assert [item.name for item in run_plan.skipped] == ["doc.pdf"]
    assert run_plan.to_process == ()
    assert not run_plan.render_only
    assert scenario.calls == 3
    assert result.items == ()
    assert scenario.output().read_bytes() == before


def test_a_pages_run_leaves_the_document_partial(six: Scenario) -> None:
    six.run(pages="3-4")

    assert six.calls == 2
    assert six.statuses() == ["pending", "pending", "done", "done"] + ["pending"] * 2
    assert six.document_status() == "partial"
    assert [record["chunk_index"] for record in six.records()] == [3, 4]

    run_plan = six.plan()
    assert [(item.name, item.state) for item in run_plan.to_process] == [
        ("doc.pdf", "partial")
    ]
    six.run()

    assert six.calls == 6
    assert six.statuses() == ["done"] * 6
    assert six.document_status() == "complete"
    assert len(six.records()) == 6


def test_a_failed_chunk_is_retried_by_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = helpers.install(monkeypatch, [ANSWER, StatusError(400), ANSWER])
    scenario = Scenario(tmp_path, script)

    first = scenario.run()

    assert (first.complete, first.failed) == (0, 1)
    assert scenario.statuses() == ["done", "failed", "done"]
    (failure,) = first.items[0].failures
    assert (failure.kind, failure.idx) == ("chunk", 1)
    assert failure.error_type is not None
    assert [record["chunk_index"] for record in scenario.records()] == [1, 3]
    calls = scenario.calls

    retry = scenario.plan()
    assert [(item.name, item.state) for item in retry.to_process] == [
        ("doc.pdf", "partial")
    ]
    assert retry.to_process[0].resume.indices == (1,)
    second = scenario.run()

    assert (second.complete, second.failed) == (1, 0)
    assert scenario.calls == calls + 1
    assert scenario.statuses() == ["done"] * 3
    assert scenario.entries() == 3
    assert [record["chunk_index"] for record in scenario.records()] == [1, 2, 3]


def test_an_output_without_a_store_row_is_skipped_as_unstored(
    scenario: Scenario,
) -> None:
    output = scenario.plan().to_process[0].output
    output.parent.mkdir(parents=True)
    output.write_text('{"records": []}\n', encoding="utf-8")

    run_plan = scenario.plan()
    result = scenario.run()

    assert run_plan.to_process == ()
    assert [item.name for item in run_plan.unstored] == ["doc.pdf"]
    assert run_plan.unstored[0].state == "unstored"
    assert "--force" in run_plan.unstored[0].reason
    assert result.items == ()
    assert result.unstored == ("doc.pdf",)
    assert scenario.calls == 0
    assert output.read_text(encoding="utf-8") == '{"records": []}\n'
    assert any("Skipping 'doc.pdf'" in w for w in scenario.events.of("warning"))


def test_a_dry_run_lists_the_items_left_out_with_their_reasons(
    scenario: Scenario, capsys: pytest.CaptureFixture[str]
) -> None:
    output = scenario.plan().to_process[0].output
    output.parent.mkdir(parents=True)
    output.write_text('{"records": []}\n', encoding="utf-8")
    spec = make_spec(scenario.pdf, output=scenario.out, dry_run=True)

    code = entry.main(["extract", *to_argv(spec), "--json"])

    (line,) = capsys.readouterr().out.splitlines()
    summary = json.loads(line)
    assert code == 0
    assert summary["to_process"] == []
    assert [entry["name"] for entry in summary["unstored"]] == ["doc.pdf"]
    assert "--force" in summary["unstored"][0]["reason"]
    assert (summary["blocked"], summary["refused"]) == ([], [])
    assert scenario.calls == 0


def test_force_extracts_an_unstored_item_again(scenario: Scenario) -> None:
    output = scenario.plan().to_process[0].output
    output.parent.mkdir(parents=True)
    output.write_text('{"records": []}\n', encoding="utf-8")

    result = scenario.run(force=True)

    assert result.complete == 1
    assert scenario.calls == 3
    assert len(scenario.records()) == 3


def test_force_resets_the_units_and_keeps_the_responses(scenario: Scenario) -> None:
    scenario.run()

    result = scenario.run(force=True)

    assert result.complete == 1
    assert scenario.calls == 6
    assert scenario.statuses() == ["done"] * 3
    assert len(read_rows(scenario.db, "responses")) == 6
    assert scenario.entries() == 3


def test_force_on_a_shorter_pdf_drops_the_pages_it_lost(scenario: Scenario) -> None:
    scenario.run()
    _pdf(scenario.pdf, 2)

    result = scenario.run(force=True)

    assert result.complete == 1
    assert scenario.statuses() == ["done"] * 2
    assert scenario.document_status() == "complete"
    assert scenario.entries() == 2
    assert len(read_rows(scenario.db, "responses")) == 5
    assert [record["chunk_index"] for record in scenario.records()] == [1, 2]


def test_a_text_chunk_without_a_recorded_range_resumes(
    tmp_path: Path, script: Script
) -> None:
    text = _text(tmp_path / "in" / "ledger.txt", lines=40)
    out = tmp_path / "out"
    spec = make_spec(text, output=out, chunk_tokens=60, concurrency=1)
    settings = make_settings(rate_limits=((100_000, 1),))
    asyncio.run(core.run(core.plan(spec, settings)))
    calls = len(script.requests)
    assert calls > 1
    connection = sqlite3.connect(out / DATABASE_NAME)
    with connection:
        connection.execute(
            "UPDATE units SET status = 'pending', source_index = NULL, "
            "content = NULL WHERE idx = (SELECT max(idx) FROM units)"
        )
    connection.close()

    resumed = asyncio.run(core.run(core.plan(spec, settings)))

    assert (resumed.complete, resumed.failed) == (1, 0)
    assert len(script.requests) == calls + 1


def test_a_deleted_output_of_a_complete_document_is_rendered_again(
    scenario: Scenario,
) -> None:
    scenario.run()
    output = scenario.output()
    before = output.read_bytes()
    output.unlink()

    run_plan = scenario.plan()
    result = scenario.run()

    assert [item.name for item in run_plan.render_only] == ["doc.pdf"]
    assert run_plan.to_process == ()
    assert result.items == ()
    assert scenario.calls == 3
    assert output.read_bytes() == before


def _versionless_database(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            "CREATE TABLE chunks (document TEXT, idx INTEGER, text TEXT);"
            "INSERT INTO chunks VALUES ('doc', 0, 'x');"
        )
        connection.commit()
    finally:
        connection.close()
    return path.read_bytes()


def test_a_versionless_database_is_refused_with_exit_2_as_not_a_store(
    scenario: Scenario, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _versionless_database(scenario.db)

    with pytest.raises(core.StoreRefusedError) as raised:
        scenario.plan()

    assert isinstance(raised.value, core.ConfigurationError)
    assert raised.value.exit_code == ExitCode.USAGE
    assert "is not a chronominer store" in str(raised.value)
    assert "migrate" not in str(raised.value)
    spec = make_spec(scenario.pdf, output=scenario.out)
    assert entry.main(["extract", *to_argv(spec), "--json"]) == 2
    captured = capsys.readouterr()
    (line,) = captured.out.splitlines()
    assert json.loads(line)["items_total"] == 0
    assert "is not a chronominer store" in captured.err
    assert scenario.db.read_bytes() == before
    assert scenario.calls == 0


def test_a_newer_schema_is_refused_with_exit_2(scenario: Scenario) -> None:
    scenario.run()
    connection = sqlite3.connect(scenario.db)
    try:
        connection.execute("UPDATE meta SET version = 99 WHERE component = 'core'")
        connection.commit()
    finally:
        connection.close()
    before = scenario.db.read_bytes()

    with pytest.raises(core.StoreRefusedError, match="newer") as raised:
        scenario.plan()

    assert raised.value.exit_code == ExitCode.USAGE
    assert scenario.db.read_bytes() == before


def test_a_model_switch_resumes_with_a_warning_and_a_note(six: Scenario) -> None:
    six.run(pages="1-2")

    run_plan = six.plan(model=OTHER_MODEL)
    six.run(model=OTHER_MODEL)

    assert [item.state for item in run_plan.to_process] == ["partial"]
    warnings = [w for w in six.events.of("warning") if OTHER_MODEL in w]
    assert warnings and "gpt-6-luna" in warnings[0]
    assert any(OTHER_MODEL in note for note in six.notes())
    assert [row["model"] for row in six.units()] == ["gpt-6-luna"] * 2 + [
        OTHER_MODEL
    ] * 4
    assert six.statuses() == ["done"] * 6


def test_a_model_switch_that_changes_the_detail_resumes(six: Scenario) -> None:
    six.run(pages="1-2", model="gpt-5.4")

    run_plan = six.plan(model="gpt-4o")
    six.run(model="gpt-4o")

    assert [item.state for item in run_plan.to_process] == ["partial"]
    assert run_plan.refused == ()
    assert [row["model"] for row in six.units()] == ["gpt-5.4"] * 2 + ["gpt-4o"] * 4
    assert six.statuses() == ["done"] * 6


def test_a_changed_user_image_setting_refuses_the_item(six: Scenario) -> None:
    six.run(pages="1-2")

    run_plan = six.plan(jpeg_quality=70)
    result = six.run(jpeg_quality=70)

    assert run_plan.to_process == ()
    assert [item.name for item in run_plan.refused] == ["doc.pdf"]
    assert "jpeg_quality" in run_plan.refused[0].reason
    assert [(item.name, item.success) for item in result.items] == [("doc.pdf", False)]
    assert six.calls == 2
    assert six.statuses() == ["done"] * 2 + ["pending"] * 4


def test_a_changed_schema_refuses_a_partial_item_unless_forced(
    six: Scenario, tmp_path: Path
) -> None:
    schema = tmp_path / "wages.json"
    text = (PROJECT_ROOT / "schemas" / "wage_records.json").read_text(encoding="utf-8")
    schema.write_text(text, encoding="utf-8")
    six.run(pages="1-2", schema=str(schema))
    schema.write_text(text.replace("Payrolls", "Pay lists"), encoding="utf-8")

    run_plan = six.plan(schema=str(schema))
    result = six.run(schema=str(schema))

    assert run_plan.to_process == ()
    assert [item.name for item in run_plan.refused] == ["doc.pdf"]
    assert "schema changed" in run_plan.refused[0].reason
    assert "--force" in run_plan.refused[0].reason
    assert [(item.name, item.success) for item in result.items] == [("doc.pdf", False)]
    assert six.calls == 2

    forced = six.run(schema=str(schema), force=True)

    assert (forced.complete, forced.failed) == (1, 0)
    assert six.statuses() == ["done"] * 6
    assert six.plan(schema=str(schema)).skipped[0].name == "doc.pdf"


def test_a_changed_schema_resumes_an_item_without_done_chunks(
    six: Scenario, tmp_path: Path
) -> None:
    schema = tmp_path / "wages.json"
    text = (PROJECT_ROOT / "schemas" / "wage_records.json").read_text(encoding="utf-8")
    schema.write_text(text, encoding="utf-8")
    six.run(pages="1-2", schema=str(schema))
    connection = sqlite3.connect(six.db)
    with connection:
        connection.execute("UPDATE units SET status = 'pending', content = NULL")
    connection.close()
    schema.write_text(text.replace("Payrolls", "Pay lists"), encoding="utf-8")

    run_plan = six.plan(schema=str(schema))

    assert [item.state for item in run_plan.to_process] == ["partial"]


def test_another_input_of_the_same_document_is_refused_unless_forced(
    tmp_path: Path, script: Script
) -> None:
    out = tmp_path / "out"
    first = _pdf(tmp_path / "a" / "doc.pdf", 2)
    second = _pdf(tmp_path / "b" / "doc.pdf", 3)
    settings = make_settings(rate_limits=((100_000, 1),))
    spec = make_spec(second, output=out, concurrency=1)
    asyncio.run(
        core.run(core.plan(make_spec(first, output=out, concurrency=1), settings))
    )

    run_plan = core.plan(spec, settings)
    result = asyncio.run(core.run(run_plan))

    assert (run_plan.to_process, run_plan.skipped) == ((), ())
    assert [item.name for item in run_plan.refused] == ["doc.pdf"]
    reason = run_plan.refused[0].reason
    assert str(first.resolve()) in reason
    assert "--force" in reason
    assert [(item.name, item.success) for item in result.items] == [("doc.pdf", False)]
    assert len(script.requests) == 2

    forced = asyncio.run(
        core.run(core.plan(make_spec(second, output=out, force=True), settings))
    )

    assert (forced.complete, forced.failed) == (1, 0)
    assert len(script.requests) == 5
    assert [item.name for item in core.plan(spec, settings).skipped] == ["doc.pdf"]


def test_an_edited_input_is_refused_unless_forced(scenario: Scenario) -> None:
    scenario.run()
    _pdf(scenario.pdf, 2)

    run_plan = scenario.plan()

    assert [item.name for item in run_plan.refused] == ["doc.pdf"]
    assert "changed since 'doc' was extracted" in run_plan.refused[0].reason
    assert scenario.run(force=True).complete == 1


def _lease_elsewhere(db: Path) -> None:
    connection = sqlite3.connect(db)
    try:
        connection.execute(
            "UPDATE documents SET lease_run_id = 'other-run', lease_until = ?, "
            "lease_generation = lease_generation + 1",
            (time.time() + 600,),
        )
        connection.commit()
    finally:
        connection.close()


def test_a_document_leased_by_a_live_holder_is_skipped_with_a_note(
    six: Scenario,
) -> None:
    six.run(pages="1-2")
    _lease_elsewhere(six.db)

    run_plan = six.plan()
    result = six.run()

    assert run_plan.to_process == ()
    assert [item.name for item in run_plan.blocked] == ["doc.pdf"]
    assert "leased by run other-run until " in run_plan.blocked[0].reason
    assert result.items == () and result.blocked == ("doc.pdf",)
    assert result.blocked_reasons == (run_plan.blocked[0].reason,)
    assert result.leased == ("doc.pdf",) and result.exit_code == 1
    assert six.calls == 2
    assert any("Skipping 'doc.pdf'" in note for note in six.notes())


def test_a_lease_taken_after_planning_skips_the_item_at_run_time(
    six: Scenario,
) -> None:
    six.run(pages="1-2")
    run_plan = six.plan()
    _lease_elsewhere(six.db)

    events = RecordingEvents()
    result = asyncio.run(core.run(run_plan, events))

    assert [item.name for item in run_plan.to_process] == ["doc.pdf"]
    assert result.items == () and result.blocked == ("doc.pdf",)
    (reason,) = result.blocked_reasons
    assert "leased by run other-run until " in reason
    assert result.leased == ("doc.pdf",) and result.exit_code == 1
    assert six.calls == 2
    assert any("another run holds" in note for note in six.notes())
    assert events.of("item_started") == []


def test_a_complete_item_leased_before_its_render_counts_as_blocked(
    scenario: Scenario,
) -> None:
    scenario.run()
    output = scenario.output()
    output.unlink()
    run_plan = scenario.plan()
    _lease_elsewhere(scenario.db)

    result = asyncio.run(core.run(run_plan))

    assert [item.name for item in run_plan.render_only] == ["doc.pdf"]
    assert result.skipped == () and result.blocked == ("doc.pdf",)
    assert result.leased == ("doc.pdf",) and result.exit_code == 1
    assert not output.exists()


def _submitting_elsewhere(root: Path, name: str, idxs: tuple[int, ...]) -> None:
    """Leave a submission ``submitting`` without a batch id, as a killed run does."""

    async def leave() -> None:
        store = open_store(root)
        await store.open()
        try:
            await store.begin_run("extract")
            doc_id = await store.ensure_document(name)
            lease = await store.acquire(doc_id)
            assert lease is not None
            await store.open_submission(
                provider="openai",
                model="gpt-5.6-luna",
                units=[UnitKey(doc_id, "chunk", idx) for idx in idxs],
            )
            await store.release(lease)
            await store.end_run(1)
        finally:
            await store.close()

    asyncio.run(leave())


def test_a_submission_left_submitting_is_marked_unresolved_at_start(
    six: Scenario,
) -> None:
    six.run(pages="1-2")
    _submitting_elsewhere(six.out, six.plan().to_process[0].document, (2, 3))

    run_plan = six.plan()
    result = six.run()

    assert [item.name for item in run_plan.blocked] == ["doc.pdf"]
    assert result.items == () and result.blocked == ("doc.pdf",)
    # Chunks waiting for a batch are submitted work, not a failure.
    assert result.leased == () and result.exit_code == 0
    (submission,) = read_rows(six.db, "submissions")
    assert (submission["status"], submission["batch_id"]) == ("unresolved", None)
    assert any("unresolved" in warning for warning in six.events.of("warning"))
    assert six.calls == 2
    assert six.statuses() == ["done"] * 2 + ["pending"] * 4

    six.run()
    assert six.calls == 2
    assert six.statuses() == ["done"] * 2 + ["pending"] * 4


def _files(folder: Path) -> Iterator[Path]:
    return (path for path in folder.rglob("*") if path.is_file())


@pytest.mark.parametrize("layout", ["beside", "flat", "mirror"])
def test_a_run_leaves_only_rendered_files_and_the_database(
    tmp_path: Path, script: Script, layout: str
) -> None:
    source = _text(tmp_path / "in" / "sub" / "ledger.txt")
    flags: dict[str, Any] = {"concurrency": 1, "formats": ("csv",), "recursive": True}
    if layout != "beside":
        flags.update(output=tmp_path / "out", layout=layout)
    spec = make_spec(tmp_path / "in", **flags)

    result = asyncio.run(
        core.run(core.plan(spec, make_settings(rate_limits=((100_000, 1),))))
    )

    assert result.complete == 1
    root = source.parent if layout == "beside" else tmp_path / "out"
    names = sorted(path.relative_to(root).as_posix() for path in _files(root))
    rendered = [name for name in names if name.endswith(("_output.json", ".csv"))]
    assert {Path(name).name for name in rendered} == {
        "ledger_output.json",
        "ledger_output.csv",
    }
    if layout == "mirror":
        assert all(name.startswith("sub/") for name in rendered)
    others = sorted(set(names) - set(rendered))
    expected = [DATABASE_NAME]
    if layout == "beside":
        expected.append("ledger.txt")
    assert others == expected
    for folder in (tmp_path / "in", root):
        assert not list(folder.rglob("*.jsonl"))
        assert not list(folder.rglob("*_batch_submission_debug.json"))
        assert not list(folder.rglob("*_line_ranges.txt"))
