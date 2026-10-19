"""The golden normalizer, the update mode and ``CM_GOLDEN_RENDERED_ONLY=1``.

The mode tests write a golden case into ``tmp_path``, change one file of the
run's snapshot and check it with and without the mode.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.characterization import golden

CASE = "mode/case"
OUTPUT_JSON = "files/out/ledger_output.json"
OUTPUT_CSV = "files/out/ledger_output.csv"
LINE_RANGES = "files/docs/ledger_line_ranges.txt"
SNAPSHOT = {
    OUTPUT_JSON: '{\n  "records": []\n}\n',
    OUTPUT_CSV: "<utf-8 bom>record_id,item\nRecord 001.,wheat\n",
    LINE_RANGES: "(1, 12)\n(13, 24)\n",
    "files/out/temp_jsonl/ledger_temp.jsonl": '[\n  {\n    "chunk_index": 1\n  }\n]\n',
    "files/out/ledger_output.sqlite.dump.txt": "## records\n",
    "files/out/ledger_output.docx.bytes.txt": "<binary file>\n",
    "stdout.txt": "[INFO] Processing 1 file(s)\n",
    "summary.json": json.dumps({"exit_code": 0, "json_line": {"files": 1}}),
    "requests.json": "{}\n",
}

Change = Callable[[dict[str, str]], None]


@pytest.fixture
def golden_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write ``SNAPSHOT`` as the golden case below ``tmp_path``."""
    monkeypatch.setattr(golden, "GOLDEN_DIR", tmp_path)
    monkeypatch.delenv(golden.UPDATE_ENV, raising=False)
    monkeypatch.delenv(golden.RENDERED_ONLY_ENV, raising=False)
    for name, content in SNAPSHOT.items():
        target = tmp_path / CASE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    return tmp_path


def _set(name: str, content: str) -> Change:
    def change(files: dict[str, str]) -> None:
        files[name] = content

    return change


def _drop(name: str) -> Change:
    def change(files: dict[str, str]) -> None:
        del files[name]

    return change


def _check(change: Change) -> None:
    actual = dict(SNAPSHOT)
    change(actual)
    golden.check_files(CASE, actual)


IGNORED = [
    _set("files/out/temp_jsonl/ledger_temp.jsonl", "[]\n"),
    _drop("files/out/temp_jsonl/ledger_temp.jsonl"),
    _set("files/out/ledger_output.sqlite.dump.txt", "## units\n"),
    _set("files/out/ledger_output.docx.bytes.txt", "<other>\n"),
    _set("stdout.txt", "[INFO] Processing 2 file(s)\n"),
    _set("requests.json", '{"calls": []}\n'),
    _set("files/out/batch_submission_debug.json", "{}\n"),
    _set("summary.json", json.dumps({"exit_code": 0, "json_line": None})),
]
CHECKED = [
    _set(OUTPUT_JSON, '{\n  "records": [1]\n}\n'),
    _set(OUTPUT_CSV, "<utf-8 bom>record_id,item\n"),
    _set(LINE_RANGES, "(1, 24)\n"),
    _drop(OUTPUT_JSON),
    _set("files/out/extra_output.json", "{}\n"),
    _set("summary.json", json.dumps({"exit_code": 1, "json_line": {}})),
]


@pytest.mark.parametrize("change", IGNORED)
def test_the_mode_ignores_files_other_than_the_views(
    change: Change, golden_case: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(AssertionError, match="Golden mismatch"):
        _check(change)

    monkeypatch.setenv(golden.RENDERED_ONLY_ENV, "1")
    _check(change)


@pytest.mark.parametrize("change", CHECKED)
def test_the_mode_fails_on_a_changed_view_or_exit_code(
    change: Change, golden_case: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(golden.RENDERED_ONLY_ENV, "1")
    with pytest.raises(AssertionError, match="Golden mismatch"):
        _check(change)


def test_the_mode_passes_an_unchanged_snapshot(
    golden_case: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(golden.RENDERED_ONLY_ENV, "1")
    golden.check_files(CASE, dict(SNAPSHOT))


def test_the_view_keeps_rendered_files_and_exit_codes() -> None:
    view = golden.rendered_view(SNAPSHOT)
    assert sorted(view) == [LINE_RANGES, OUTPUT_CSV, OUTPUT_JSON, "summary.json"]
    assert json.loads(view["summary.json"]) == {"exit_code": 0}


def test_update_mode_rewrites_the_case(
    golden_case: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[Path] = []
    actual = {**SNAPSHOT, LINE_RANGES: "(1, 24)\n"}
    del actual["stdout.txt"]

    monkeypatch.setenv(golden.UPDATE_ENV, "1")
    golden.check_files(CASE, actual, on_write=written.append)

    assert written == [golden_case]
    assert not (golden_case / CASE / "stdout.txt").exists()
    monkeypatch.delenv(golden.UPDATE_ENV)
    golden.check_files(CASE, actual)


def test_a_missing_case_asks_for_the_update_mode(golden_case: Path) -> None:
    with pytest.raises(AssertionError, match=golden.UPDATE_ENV):
        golden.check_files("mode/absent", dict(SNAPSHOT))


def test_the_normalizer_replaces_run_specific_values(tmp_path: Path) -> None:
    root = tmp_path / "case"
    norm = golden.Normalizer(
        {root: "<tmp>", tmp_path: "<test_tmp>"},
        replacements={"batch_fake0001": "<openai-batch-1>"},
    )
    windows = str(root / "out" / "ledger_output.json")
    escaped = json.dumps({"path": windows})
    text = (
        f"{windows} {escaped} batch_fake0001 2026-01-01T10:20:30.5+00:00 "
        "0f8fad5b-d9cb-469f-a165-70867728950e "
        "manifest_9b5ad71b2ce5302211f9c61530b329a4.json "
        "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
    )

    normalized = norm.text(text)

    assert normalized == (
        '<tmp>/out/ledger_output.json {"path": "<tmp>/out/ledger_output.json"} '
        "<openai-batch-1> <datetime> <uuid> manifest_<hex>.json <data-url>"
    )
    data = norm.data(
        {
            "timestamp": 1767225600,
            "image_sha256": "ab" * 32,
            "byte_size": 2048,
            "pillow_version": "12.3.0",
            "model": "gpt-4o",
        }
    )
    assert data == {
        "timestamp": "<timestamp>",
        "image_sha256": "<sha256>",
        "byte_size": "<byte_size>",
        "pillow_version": "<version>",
        "model": "gpt-4o",
    }


def test_file_values_parse_json_text_and_digest_long_text() -> None:
    long_text = "x" * (golden.LONG_FILE_TEXT + 1)
    value = golden._file_value(
        {"output_text": '{"entries": [{"item": "wheat"}]}', "prompt": long_text}
    )

    assert value["output_text"] == {"entries": [{"item": "wheat"}]}
    assert value["prompt"].startswith(f"<text: {len(long_text)} chars ")
