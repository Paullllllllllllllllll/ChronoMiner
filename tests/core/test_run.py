"""Core entry points: discover, plan with typed errors, run with events.

A scripted chat model (``tests.llm.helpers``) answers every chunk, so no
request leaves the machine.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

import chronominer.llm.caller as caller_module
import chronominer.run as core
from chronominer import __main__ as entry
from chronominer.common.contract import ExitCode
from chronominer.common.testing import FakeRequest
from chronominer.events import ItemFinished, ItemStarted
from chronominer.output.store import DATABASE_NAME, ItemState
from chronominer.settings import Endpoint
from chronominer.spec import to_argv
from tests.conftest import make_settings, make_spec
from tests.llm import helpers
from tests.llm.helpers import StatusError
from tests.recording_events import RecordingEvents

MakeFile = Callable[..., Path]
KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY")
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


@pytest.fixture
def make_pdf(tmp_path: Path) -> MakeFile:
    """Return a factory for PDFs under ``tmp_path``."""

    def make(name: str = "doc.pdf", pages: int = 1) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        document = fitz.open()
        for index in range(pages):
            page = document.new_page(width=200, height=300)
            page.insert_text((50, 100), f"Page {index + 1}")
        document.save(path)
        document.close()
        return path

    return make


@pytest.fixture
def make_text(tmp_path: Path) -> MakeFile:
    """Return a factory for payroll text files under ``tmp_path``."""

    def make(name: str = "ledger.txt", lines: int = 5) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(f"{n}. John Carter, mason, 2s 6d" for n in range(lines))
        path.write_text(body + "\n", encoding="utf-8")
        return path

    return make


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variables of the providers."""
    for name in KEYS:
        monkeypatch.setenv(name, "test-key")


def _plan(input_path: Path, **flags: Any) -> core.RunPlan:
    return core.plan(make_spec(input_path, **flags), make_settings())


def _settings() -> Any:
    return make_settings(rate_limits=((100_000, 1),))


def test_discover_finds_the_files_of_the_type(
    tmp_path: Path, make_text: MakeFile, make_pdf: MakeFile
) -> None:
    make_text("in/a.txt")
    make_text("in/sub/b.md")
    make_pdf("in/scan.pdf")

    flat = core.discover(make_spec(tmp_path / "in", type="text"))
    deep = core.discover(make_spec(tmp_path / "in", type="text", recursive=True))

    assert [item.name for item in flat] == ["a.txt"]
    assert sorted(item.name for item in deep) == ["a.txt", "b.md"]
    assert all(item.type == "text" and item.path.is_absolute() for item in deep)


def test_discover_in_auto_mode_types_every_item(
    tmp_path: Path, make_text: MakeFile, make_pdf: MakeFile
) -> None:
    make_text("in/a.txt")
    make_pdf("in/b.pdf")
    (tmp_path / "in" / "c.png").write_bytes(b"png")
    (tmp_path / "in" / "a_extract_context.txt").write_text("x", encoding="utf-8")

    items = core.discover(make_spec(tmp_path / "in"))

    assert [(item.type, item.name) for item in items] == [
        ("text", "a.txt"),
        ("pdf", "b.pdf"),
        ("image", "c.png"),
    ]


def test_a_folder_of_several_items_needs_a_selection(
    tmp_path: Path, make_text: MakeFile
) -> None:
    make_text("in/a.txt")
    make_text("in/b.txt")

    with pytest.raises(core.InputError, match="holds 2 items") as raised:
        core.discover(make_spec(tmp_path / "in", all=False))

    assert raised.value.exit_code == ExitCode.USAGE
    assert "--all, --select EXPR" in str(raised.value)
    make_text("one/a.txt")
    assert len(core.discover(make_spec(tmp_path / "one", all=False))) == 1


@pytest.mark.parametrize(
    ("expr", "names"),
    [("2", ["b.txt"]), ("1,3", ["a.txt", "c.txt"]), ("2-3", ["b.txt", "c.txt"])],
)
def test_select_keeps_items_by_number_or_range(
    tmp_path: Path, make_text: MakeFile, expr: str, names: list[str]
) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        make_text(f"in/{name}")

    items = core.discover(make_spec(tmp_path / "in", select=expr))

    assert [item.name for item in items] == names


@pytest.mark.usefixtures("keys")
def test_select_searches_names_and_reports_unmatched_numbers(
    tmp_path: Path, make_text: MakeFile
) -> None:
    for name in ("ledger_1791.txt", "ledger_1792.txt", "notes.txt"):
        make_text(f"in/{name}")

    searched = core.discover(make_spec(tmp_path / "in", select="ledger"))
    numbered = _plan(tmp_path / "in", select="1,9", dry_run=True)

    assert [item.name for item in searched] == ["ledger_1791.txt", "ledger_1792.txt"]
    assert [item.name for item in numbered.to_process] == ["ledger_1791.txt"]
    assert any("'9'" in warning for warning in numbered.warnings)
    with pytest.raises(core.NoItemsError, match="matches none of the 3 item"):
        core.discover(make_spec(tmp_path / "in", select="inventory"))


def _missing_input(tmp_path: Path, make_text: MakeFile) -> tuple[Path, dict[str, Any]]:
    return tmp_path / "missing.txt", {}


def _empty_folder(tmp_path: Path, make_text: MakeFile) -> tuple[Path, dict[str, Any]]:
    (tmp_path / "in").mkdir()
    return tmp_path / "in", {}


def _wrong_type(tmp_path: Path, make_text: MakeFile) -> tuple[Path, dict[str, Any]]:
    return make_text(), {"type": "pdf"}


def _sidecar_input(tmp_path: Path, make_text: MakeFile) -> tuple[Path, dict[str, Any]]:
    return make_text("ledger_line_ranges.txt"), {}


def _output_is_a_file(
    tmp_path: Path, make_text: MakeFile
) -> tuple[Path, dict[str, Any]]:
    (tmp_path / "out").write_text("x", encoding="utf-8")
    return make_text(), {"output": tmp_path / "out"}


def _unknown_schema(tmp_path: Path, make_text: MakeFile) -> tuple[Path, dict[str, Any]]:
    return make_text(), {"schema": "no_such_schema"}


def _missing_context(
    tmp_path: Path, make_text: MakeFile
) -> tuple[Path, dict[str, Any]]:
    return make_text(), {"context": tmp_path / "nope.txt"}


@pytest.mark.usefixtures("keys")
@pytest.mark.parametrize(
    ("setup", "error", "message"),
    [
        (_missing_input, core.InputError, "Input path does not exist"),
        (_empty_folder, core.NoItemsError, "No text, image or PDF files found"),
        (_wrong_type, core.InputError, "--type pdf does not fit"),
        (_sidecar_input, core.InputError, "is a sidecar file"),
        (_output_is_a_file, core.InputError, "not a directory"),
        (_unknown_schema, core.ConfigurationError, "unknown schema 'no_such_schema'"),
        (_missing_context, core.ConfigurationError, "nope.txt"),
    ],
)
def test_plan_errors_carry_exit_code_2(
    tmp_path: Path,
    make_text: MakeFile,
    setup: Callable[[Path, MakeFile], tuple[Path, dict[str, Any]]],
    error: type[core.PlanError],
    message: str,
) -> None:
    input_path, flags = setup(tmp_path, make_text)

    with pytest.raises(error) as raised:
        _plan(input_path, **flags)

    assert raised.value.exit_code == ExitCode.USAGE
    assert message in str(raised.value)


CONFIG_ERRORS: dict[str, tuple[dict[str, Any], tuple[str, ...], str]] = {
    "missing key": ({}, (), "OPENAI_API_KEY"),
    "batch on openrouter": (
        {"model": "openai/gpt-5.6-luna", "provider": "openrouter", "batch": True},
        KEYS,
        r"batch API \(openai, anthropic, google\), not openrouter",
    ),
    "non-vision model on image input": (
        {"model": "gpt-7-foo", "type": "pdf"},
        KEYS,
        "does not support image inputs",
    ),
    "unknown schema": ({"schema": "NoSuchSchema"}, KEYS, "unknown schema"),
}


@pytest.mark.parametrize("case", list(CONFIG_ERRORS))
def test_configuration_errors_stop_the_plan_before_the_output_folder(
    tmp_path: Path,
    make_text: MakeFile,
    make_pdf: MakeFile,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    flags, keys, message = CONFIG_ERRORS[case]
    for name in keys:
        monkeypatch.setenv(name, "test-key")
    source = make_pdf() if flags.get("type") == "pdf" else make_text()
    out = tmp_path / "out"

    with pytest.raises(core.ConfigurationError, match=message) as raised:
        _plan(source, output=out, **flags)

    assert raised.value.exit_code == ExitCode.USAGE
    assert not out.exists()
    assert not list(tmp_path.rglob(DATABASE_NAME))


KEYED_RUNS: dict[str, tuple[str, dict[str, Any]]] = {
    "extract": ("OPENAI_API_KEY", {}),
    "batch": ("ANTHROPIC_API_KEY", {"batch": True, "model": "claude-sonnet-5-5"}),
    "google": ("GOOGLE_API_KEY", {"model": "gemini-3.5-flash"}),
}


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("case", list(KEYED_RUNS))
def test_an_unset_key_variable_exits_2_but_a_dry_run_only_warns(
    tmp_path: Path,
    make_text: MakeFile,
    capsys: pytest.CaptureFixture[str],
    case: str,
    dry_run: bool,
) -> None:
    variable, flags = KEYED_RUNS[case]
    out = tmp_path / "out"
    spec = make_spec(make_text(), output=out, **flags)

    argv = ["extract", *to_argv(spec), "--json"]
    if dry_run:
        argv.append("--dry-run")

    assert entry.main(argv) == (0 if dry_run else 2)

    captured = capsys.readouterr()
    (line,) = captured.out.splitlines()
    assert json.loads(line)["items_total"] == 1
    assert variable in captured.err
    assert not out.exists()


def test_with_the_key_set_the_plan_passes(
    make_text: MakeFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    for variable, flags in KEYED_RUNS.values():
        monkeypatch.setenv(variable, "test-key")
        run_plan = _plan(make_text(), **flags)
        assert len(run_plan.to_process) == 1


def test_a_dry_run_plans_without_keys_and_names_the_missing_variable(
    make_text: MakeFile,
) -> None:
    run_plan = _plan(make_text(), dry_run=True)

    assert len(run_plan.to_process) == 1
    (missing,) = run_plan.missing_keys
    assert missing.variable == "OPENAI_API_KEY"
    assert any("OPENAI_API_KEY" in warning for warning in run_plan.warnings)
    settings = make_settings()
    spec = make_spec(make_text(), dry_run=True)
    assert core.plan(spec, settings, require_keys=False).missing_keys
    with pytest.raises(core.ConfigurationError, match="OPENAI_API_KEY"):
        core.plan(spec, settings, require_keys=True)


def test_a_named_endpoint_takes_its_own_key_variable(
    make_text: MakeFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(
        endpoints={"local": Endpoint("local", "http://127.0.0.1:1/v1", "LOCAL_KEY")}
    )
    spec = make_spec(make_text(), endpoint="local", model="m")
    with pytest.raises(core.ConfigurationError, match="LOCAL_KEY"):
        core.plan(spec, settings)
    monkeypatch.setenv("LOCAL_KEY", "k")
    assert core.plan(spec, settings).to_process


def test_adjust_chunking_needs_a_chat_key_beside_the_batch_key(
    make_text: MakeFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronominer import ext

    def variables(provider: str, endpoint: Any = None, **kwargs: Any) -> Any:
        return ("BATCH_KEY",) if kwargs["kind"] == "batch" else ("CHAT_KEY",)

    monkeypatch.setattr(ext, "key_variables", variables)
    monkeypatch.setenv("BATCH_KEY", "k")
    text = make_text()

    assert _plan(text, batch=True).to_process
    with pytest.raises(core.ConfigurationError, match="CHAT_KEY.*--chunking adjust"):
        _plan(text, batch=True, chunking="adjust")


@pytest.mark.usefixtures("keys")
def test_a_text_only_run_needs_no_vision_model(make_text: MakeFile) -> None:
    assert len(_plan(make_text(), model="gpt-7-foo").to_process) == 1


@pytest.mark.usefixtures("keys")
def test_a_dry_run_plan_classifies_resume_and_writes_nothing(
    tmp_path: Path, make_text: MakeFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    helpers.install(monkeypatch, [ANSWER])
    make_text("in/a.txt")
    make_text("in/b.txt")
    out = tmp_path / "out"
    spec = make_spec(tmp_path / "in", files=("a.txt",), output=out)
    asyncio.run(core.run(core.plan(spec, _settings())))
    before = sorted(path.name for path in out.rglob("*"))

    run_plan = _plan(tmp_path / "in", output=out, dry_run=True)

    assert [item.name for item in run_plan.skipped] == ["a.txt"]
    assert [(item.name, item.state) for item in run_plan.to_process] == [
        ("b.txt", ItemState.NONE.value)
    ]
    assert sorted(path.name for path in out.rglob("*")) == before
    with pytest.raises(ValueError, match="dry-run"):
        asyncio.run(core.run(run_plan))


@pytest.mark.usefixtures("keys")
def test_a_run_reports_its_items_and_writes_every_format(
    tmp_path: Path,
    make_text: MakeFile,
    make_pdf: MakeFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers.install(monkeypatch, [ANSWER])
    make_text("in/a.txt")
    make_pdf("in/b.pdf", pages=2)
    events = RecordingEvents()
    spec = make_spec(tmp_path / "in", output=tmp_path / "out", formats=("csv",))

    result = asyncio.run(core.run(core.plan(spec, _settings()), events))

    assert (result.complete, result.failed) == (2, 0)
    started = events.of("item_started")
    finished = events.of("item_finished")
    assert [(e.name, e.kind, e.index, e.total) for e in started] == [
        ("a.txt", "text", 1, 2),
        ("b.pdf", "pdf", 2, 2),
    ]
    assert all(isinstance(e, ItemStarted) for e in started)
    assert all(isinstance(e, ItemFinished) and e.success for e in finished)
    assert [e.chunk for e in events.of("chunk_done")] == [0, 0, 1]
    assert sorted(Path(path).name for path in result.outputs) == [
        "a_output.csv",
        "a_output.json",
        "b_output.csv",
        "b_output.json",
        DATABASE_NAME,
    ]
    database = str((tmp_path / "out" / DATABASE_NAME).resolve())
    assert result.outputs[-1] == database
    assert result.databases == (database,)
    assert all(Path(path).exists() for path in result.outputs)
    assert result.usage["extract"].total_tokens == 360


@pytest.mark.usefixtures("keys")
def test_a_failing_item_is_reported_and_the_run_continues(
    tmp_path: Path, make_text: MakeFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    def answer(request: FakeRequest) -> Any:
        text = " ".join(str(message.content) for message in request.messages)
        return StatusError(400) if "Ann Baker" in text else ANSWER

    helpers.install(monkeypatch, [answer])
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.txt").write_text("1. Ann Baker, spinner\n", encoding="utf-8")
    make_text("in/b.txt")
    events = RecordingEvents()
    spec = make_spec(tmp_path / "in", output=tmp_path / "out")

    result = asyncio.run(core.run(core.plan(spec, _settings()), events))

    assert [(item.name, item.success) for item in result.items] == [
        ("a.txt", False),
        ("b.txt", True),
    ]
    assert result.items[0].error
    failed = events.of("item_finished")[0]
    assert (failed.success, failed.error) == (False, result.items[0].error)
    assert result.items[0].outputs == (
        str((tmp_path / "out" / "a_output.json").resolve()),
    )


@pytest.mark.usefixtures("keys")
def test_a_client_that_cannot_be_built_stops_the_run_before_any_write(
    tmp_path: Path, make_text: MakeFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("model cannot be reached")

    monkeypatch.setattr(caller_module, "build_chat_model", refuse)
    out = tmp_path / "out"
    run_plan = _plan(make_text(), output=out)

    with pytest.raises(core.ConfigurationError, match="cannot be reached"):
        asyncio.run(core.run(run_plan))

    assert not out.exists()


@pytest.mark.usefixtures("keys")
def test_a_run_with_nothing_to_process_builds_no_client(
    tmp_path: Path, make_text: MakeFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    helpers.install(monkeypatch, [ANSWER])
    out = tmp_path / "out"
    text = make_text()
    asyncio.run(core.run(_plan(text, output=out)))
    (out / "ledger_output.json").unlink()
    opened: list[bool] = []

    async def refuse(self: Any) -> None:
        opened.append(True)
        raise ValueError("no client")

    monkeypatch.setattr(core.Extractor, "open", refuse)
    run_plan = _plan(text, output=out)

    result = asyncio.run(core.run(run_plan))

    assert (run_plan.to_process, result.items) == ((), ())
    assert (result.skipped, result.failed) == (("ledger.txt",), 0)
    assert [item.name for item in run_plan.render_only] == ["ledger.txt"]
    assert (out / "ledger_output.json").exists()
    assert not opened
