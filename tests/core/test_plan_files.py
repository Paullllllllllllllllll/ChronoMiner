"""Names given with --files are checked when the run is planned."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

import chronominer.run as core
from chronominer import __main__ as entry
from chronominer.spec import to_argv
from tests.conftest import make_settings, make_spec


def _pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    document.new_page(width=200, height=300).insert_text((50, 100), "Page 1")
    document.save(path)
    document.close()


@pytest.fixture(autouse=True)
def openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable of the default provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    """An input folder with two PDFs, a text file and its line-ranges sidecar."""
    folder = tmp_path / "in"
    _pdf(folder / "a.pdf")
    _pdf(folder / "b.pdf")
    (folder / "ledger.txt").write_text("1. John Carter\n", encoding="utf-8")
    (folder / "ledger_line_ranges.txt").write_text("(1, 1)\n", encoding="utf-8")
    return folder


CASES: dict[str, tuple[dict[str, Any], tuple[str, ...], str]] = {
    "missing": (
        {"type": "pdf"},
        ("a.pdf", "missing.pdf", "gone.pdf"),
        "'missing.pdf', 'gone.pdf'",
    ),
    "wrong type": (
        {"type": "text"},
        ("ledger.txt", "a.pdf"),
        "not inputs of --type text: 'a.pdf'",
    ),
    "sidecar": ({}, ("ledger_line_ranges.txt",), "'ledger_line_ranges.txt'"),
}


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("case", list(CASES))
def test_the_cli_exits_2_on_unusable_names(
    tmp_path: Path,
    docs: Path,
    capsys: pytest.CaptureFixture[str],
    case: str,
    dry_run: bool,
) -> None:
    flags, files, named = CASES[case]
    out = tmp_path / "out"
    spec = make_spec(docs, files=files, output=out, **flags)
    argv = ["extract", *to_argv(spec), "--json"]
    if dry_run:
        argv.append("--dry-run")
    before = sorted(docs.rglob("*"))

    assert entry.main(argv) == 2

    assert named in capsys.readouterr().err
    assert not out.exists()
    assert sorted(docs.rglob("*")) == before


def test_existing_names_select_their_items(docs: Path) -> None:
    pdfs = core.plan(
        make_spec(docs, type="pdf", files=("b.pdf",), dry_run=True),
        make_settings(),
    )
    mixed = core.plan(
        make_spec(docs, files=("ledger.txt", "a.pdf"), dry_run=True),
        make_settings(),
    )

    assert [item.name for item in pdfs.to_process] == ["b.pdf"]
    assert [(item.name, item.type) for item in mixed.to_process] == [
        ("ledger.txt", "text"),
        ("a.pdf", "pdf"),
    ]


def test_inputs_sharing_one_output_exit_2(tmp_path: Path, docs: Path) -> None:
    _pdf(docs / "sub" / "a.pdf")
    (docs / "B.txt").write_text("1. Jane Carter\n", encoding="utf-8")
    out = tmp_path / "out"
    flat = make_spec(docs, type="pdf", recursive=True, output=out)
    beside = make_spec(docs, files=("b.pdf", "B.txt"))

    for spec in (flat, beside):
        with pytest.raises(core.InputError, match="share the output"):
            core.plan(spec, make_settings())
    assert not out.exists()
