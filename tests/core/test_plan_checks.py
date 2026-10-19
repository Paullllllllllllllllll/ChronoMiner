"""Plan-time checks: batch API, image input, endpoints and the output path.

Each refused plan exits 2 before any output folder or database exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

import chronominer.run as core
from chronominer import __main__ as entry
from chronominer.common.contract import ExitCode
from chronominer.output.store import DATABASE_NAME
from chronominer.settings import Endpoint
from tests.cli.conftest import json_lines
from tests.conftest import make_settings, make_spec

KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY")
SCHEMA = "price_list_entries"


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dummy keys, so only the check under test can fail the plan."""
    for name in (*KEYS, "LOCAL_KEY"):
        monkeypatch.setenv(name, "test-key")


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    """An input folder with one born-digital PDF."""
    folder = tmp_path / "in"
    folder.mkdir()
    document = fitz.open()
    document.new_page(width=200, height=300).insert_text((50, 100), "Page 1")
    document.save(folder / "x.pdf")
    document.close()
    return folder


@pytest.fixture
def texts(tmp_path: Path) -> Path:
    """An input folder with one text file."""
    folder = tmp_path / "texts"
    folder.mkdir()
    (folder / "list.txt").write_text("Bread, 2 lb, 4 d\n", encoding="utf-8")
    return folder


def _settings_file(tmp_path: Path, *, vision: bool) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(
        "endpoints:\n"
        "  local:\n"
        "    base_url: http://127.0.0.1:8000/v1\n"
        "    api_key_env: LOCAL_KEY\n"
        f"    supports_vision: {'true' if vision else 'false'}\n",
        encoding="utf-8",
    )
    return path


def _extract(source: Path, out: Path, *extra: str) -> list[str]:
    return [
        "extract",
        "--schema",
        SCHEMA,
        "--input",
        str(source),
        "--output",
        str(out),
        *extra,
        "--dry-run",
        "--json",
    ]


def _result(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    (payload,) = json_lines(captured.out)
    return payload, captured.err


def _nothing_written(out: Path, *sources: Path) -> None:
    assert not out.exists()
    for source in sources:
        assert not (source / DATABASE_NAME).exists()


def test_batch_on_a_provider_without_a_batch_api_exits_2(
    tmp_path: Path, texts: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    argv = _extract(
        texts,
        out,
        "--provider",
        "openrouter",
        "--model",
        "openai/gpt-5.6-luna",
        "--batch",
    )

    assert entry.main(argv) == 2

    payload, err = _result(capsys)
    assert payload["items_total"] == 1
    assert "batch API (openai, anthropic, google), not openrouter" in err
    _nothing_written(out, texts)


@pytest.mark.parametrize(
    ("model", "code"), [("gpt-7-foo", ExitCode.USAGE), ("gpt-5.6-luna", ExitCode.OK)]
)
def test_the_model_must_accept_images_for_pdf_input(
    tmp_path: Path,
    docs: Path,
    capsys: pytest.CaptureFixture[str],
    model: str,
    code: ExitCode,
) -> None:
    out = tmp_path / "out"

    assert entry.main(_extract(docs, out, "--model", model)) == code

    payload, err = _result(capsys)
    assert payload["items_total"] == 1
    assert ("does not support image inputs" in err) is (code == ExitCode.USAGE)
    _nothing_written(out, docs)


def test_text_input_needs_no_image_model(texts: Path) -> None:
    spec = make_spec(texts, schema=SCHEMA, model="gpt-7-foo", dry_run=True)

    assert len(core.plan(spec, make_settings()).to_process) == 1


@pytest.mark.parametrize(("vision", "code"), [(False, 2), (True, 0)])
def test_a_custom_endpoint_declares_its_image_input(
    tmp_path: Path,
    docs: Path,
    capsys: pytest.CaptureFixture[str],
    vision: bool,
    code: int,
) -> None:
    settings = _settings_file(tmp_path, vision=vision)
    out = tmp_path / "out"
    argv = _extract(
        docs,
        out,
        "--endpoint",
        "local",
        "--model",
        "my-model",
        "--settings",
        str(settings),
    )

    assert entry.main(argv) == code

    _payload, err = _result(capsys)
    assert ("'my-model' does not support image inputs" in err) is (not vision)
    _nothing_written(out, docs)


def test_a_missing_endpoint_keeps_its_settings_error(docs: Path) -> None:
    spec = make_spec(docs, schema=SCHEMA, endpoint="local", model="m", dry_run=True)
    endpoint = Endpoint("other", "http://127.0.0.1:8000/v1", "LOCAL_KEY")

    with pytest.raises(core.ConfigurationError, match="unknown endpoint 'local'"):
        core.plan(spec, make_settings(endpoints={"other": endpoint}))


def test_a_missing_key_is_a_dry_run_warning_naming_the_variable(
    tmp_path: Path,
    texts: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    out = tmp_path / "out"

    assert entry.main(_extract(texts, out, "--model", "claude-sonnet-5-5")) == 0

    payload, err = _result(capsys)
    assert payload["items_total"] == 1
    assert "[WARNING]" in err and "ANTHROPIC_API_KEY" in err
    _nothing_written(out, texts)


def test_an_output_path_that_is_a_file_exits_2(
    tmp_path: Path, texts: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    out.write_text("not a folder\n", encoding="utf-8")

    assert entry.main(_extract(texts, out)) == 2

    _payload, err = _result(capsys)
    assert "not a directory" in err
    assert out.read_text(encoding="utf-8") == "not a folder\n"


@pytest.mark.parametrize("dry_run", [False, True])
def test_an_unknown_schema_exits_2_before_any_output_folder(
    tmp_path: Path, texts: Path, capsys: pytest.CaptureFixture[str], dry_run: bool
) -> None:
    out = tmp_path / "out"
    argv = _extract(texts, out)
    argv[argv.index(SCHEMA)] = "no_such_schema"
    if not dry_run:
        argv.remove("--dry-run")
    before = sorted(texts.rglob("*"))

    assert entry.main(argv) == 2

    _payload, err = _result(capsys)
    assert "no_such_schema" in err
    _nothing_written(out, texts)
    assert sorted(texts.rglob("*")) == before
