"""Regression tests for the CLI exit-code contract (0/1/2/130).

Covers fixes 1-5 from the exit-code audit:
    1. line_range_readjuster._run_cli_mode exits 1 on failures/stopped files.
    2. check_batches.run_cli exits 2 on schema/input/config configuration errors.
    3. generate_line_ranges.run_cli exits 1 when any file fails to process.
    4. repair_extractions.run_cli exits 2 when a filter matches nothing.
    5. cancel_batches.run_cli exits 1 on a failed cancellation and exits 2
       when --force is not supplied.

Every test invokes ``run_cli``/``_run_cli_mode`` directly with a constructed
``Namespace`` (or monkeypatches the underlying async worker), so no network
call or real LLM/batch-provider call is ever made.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Fix 1: line_range_readjuster.py
# ---------------------------------------------------------------------------


class _FakeSchemaManager:
    def get_available_schemas(self) -> dict[str, object]:
        return {"TestSchema": object()}


@pytest.mark.asyncio
async def test_readjuster_cli_mode_exits_1_on_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import main.line_range_readjuster as lrr

    text_file = tmp_path / "sample.txt"
    text_file.write_text("line one\nline two\n", encoding="utf-8")

    monkeypatch.setattr(lrr, "validate_schema_paths", lambda *a, **kw: True)
    monkeypatch.setattr(
        lrr,
        "_adjust_files",
        AsyncMock(
            return_value=(
                [],  # successes
                [],  # skipped_no_ranges
                [],  # skipped_already_adjusted
                [(text_file, RuntimeError("boom"))],  # failures
                [],  # stopped
            )
        ),
    )

    args = Namespace(
        path=tmp_path,
        schema="TestSchema",
        context_window=None,
        prompt_path=None,
        resume=False,
        force=False,
        first_n_chunks=None,
        last_n_chunks=None,
        model=None,
        reasoning_effort=None,
        max_output_tokens=None,
        temperature=None,
        top_p=None,
    )

    with pytest.raises(SystemExit) as exc:
        await lrr._run_cli_mode(
            args=args,
            schema_manager=_FakeSchemaManager(),
            schemas_paths={"TestSchema": {"input": str(tmp_path)}},
            model_config={},
            chunking_config={},
            matching_config={},
            retry_config={},
            default_context_window=6,
        )

    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_readjuster_cli_mode_exits_1_on_stopped_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user-declined token-limit wait counts as partial completion (exit 1)."""
    import main.line_range_readjuster as lrr

    text_file = tmp_path / "sample.txt"
    text_file.write_text("line one\nline two\n", encoding="utf-8")

    monkeypatch.setattr(lrr, "validate_schema_paths", lambda *a, **kw: True)
    monkeypatch.setattr(
        lrr,
        "_adjust_files",
        AsyncMock(return_value=([], [], [], [], [text_file])),
    )

    args = Namespace(
        path=tmp_path,
        schema="TestSchema",
        context_window=None,
        prompt_path=None,
        resume=False,
        force=False,
        first_n_chunks=None,
        last_n_chunks=None,
        model=None,
        reasoning_effort=None,
        max_output_tokens=None,
        temperature=None,
        top_p=None,
    )

    with pytest.raises(SystemExit) as exc:
        await lrr._run_cli_mode(
            args=args,
            schema_manager=_FakeSchemaManager(),
            schemas_paths={"TestSchema": {"input": str(tmp_path)}},
            model_config={},
            chunking_config={},
            matching_config={},
            retry_config={},
            default_context_window=6,
        )

    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_readjuster_cli_mode_clean_run_does_not_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run with only successes/skips must not raise SystemExit at all."""
    import main.line_range_readjuster as lrr

    text_file = tmp_path / "sample.txt"
    text_file.write_text("line one\nline two\n", encoding="utf-8")

    monkeypatch.setattr(lrr, "validate_schema_paths", lambda *a, **kw: True)
    monkeypatch.setattr(
        lrr,
        "_adjust_files",
        AsyncMock(return_value=([(text_file, text_file)], [], [], [], [])),
    )

    args = Namespace(
        path=tmp_path,
        schema="TestSchema",
        context_window=None,
        prompt_path=None,
        resume=False,
        force=False,
        first_n_chunks=None,
        last_n_chunks=None,
        model=None,
        reasoning_effort=None,
        max_output_tokens=None,
        temperature=None,
        top_p=None,
    )

    # Should complete without raising SystemExit.
    await lrr._run_cli_mode(
        args=args,
        schema_manager=_FakeSchemaManager(),
        schemas_paths={"TestSchema": {"input": str(tmp_path)}},
        model_config={},
        chunking_config={},
        matching_config={},
        retry_config={},
        default_context_window=6,
    )

    out = capsys.readouterr().out
    assert "Successful adjustments: 1" in out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path_key, schema, schemas_paths, valid_paths",
    [
        ("missing_path", "TestSchema", {"TestSchema": {"input": "."}}, True),
        ("tmp", None, {"TestSchema": {"input": "."}}, True),
        ("tmp", "NoSuchSchema", {"TestSchema": {"input": "."}}, True),
        ("tmp", "TestSchema", {}, False),
    ],
    ids=["no-path", "no-schema", "unknown-schema", "schema-without-paths"],
)
async def test_readjuster_cli_usage_errors_exit_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_key: str,
    schema: str | None,
    schemas_paths: dict[str, object],
    valid_paths: bool,
) -> None:
    """Usage/configuration errors follow the documented contract: exit 2."""
    import main.line_range_readjuster as lrr

    (tmp_path / "sample.txt").write_text("line one\n", encoding="utf-8")

    monkeypatch.setattr(lrr, "validate_schema_paths", lambda *a, **kw: valid_paths)
    monkeypatch.setattr(
        lrr, "_adjust_files", AsyncMock(return_value=([], [], [], [], []))
    )

    args = Namespace(
        path=None if path_key == "missing_path" else tmp_path,
        schema=schema,
        context_window=None,
        prompt_path=None,
        resume=False,
        force=False,
        first_n_chunks=None,
        last_n_chunks=None,
        model=None,
        reasoning_effort=None,
        max_output_tokens=None,
        temperature=None,
        top_p=None,
    )

    with pytest.raises(SystemExit) as exc:
        await lrr._run_cli_mode(
            args=args,
            schema_manager=_FakeSchemaManager(),
            schemas_paths=schemas_paths,
            model_config={},
            chunking_config={},
            matching_config={},
            retry_config={},
            default_context_window=6,
        )

    assert exc.value.code == 2


def test_readjuster_parser_accepts_input_alias(tmp_path: Path) -> None:
    """--input is an accepted alias of --path (README examples use --input)."""
    import sys as _sys

    import main.line_range_readjuster as lrr

    old_argv = _sys.argv
    try:
        _sys.argv = ["line_range_readjuster.py", "--input", str(tmp_path)]
        args = lrr.parse_arguments()
    finally:
        _sys.argv = old_argv

    assert args.path == tmp_path


def test_readjuster_keyboard_interrupt_exits_130(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import main.line_range_readjuster as lrr

    def _raise_keyboard_interrupt(coro: object) -> None:
        coro.close()  # type: ignore[attr-defined]
        raise KeyboardInterrupt

    monkeypatch.setattr(lrr.asyncio, "run", _raise_keyboard_interrupt)

    with pytest.raises(SystemExit) as exc:
        lrr.main()

    assert exc.value.code == 130
    assert "[STOPPED]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Fix 2: check_batches.py
# ---------------------------------------------------------------------------


def _make_check_batches_script(monkeypatch: pytest.MonkeyPatch, repo_info_list):
    from main.check_batches import CheckBatchesScript

    script = CheckBatchesScript()
    monkeypatch.setattr(script, "_load_batch_config", lambda: None)
    script.repo_info_list = repo_info_list
    script.processing_settings = {}
    return script


def test_check_batches_exits_2_when_schema_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _make_check_batches_script(
        monkeypatch, [("OtherSchema", Path("some/dir"), {})]
    )
    args = Namespace(schema="Missing", input=None, verbose=False, json_summary=False)

    with pytest.raises(SystemExit) as exc:
        script.run_cli(args)

    assert exc.value.code == 2


def test_check_batches_exits_2_when_input_path_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _make_check_batches_script(
        monkeypatch, [("TestSchema", Path("some/dir"), {})]
    )
    args = Namespace(
        schema=None,
        input="Z:/definitely/does/not/exist/anywhere",
        verbose=False,
        json_summary=False,
    )

    with pytest.raises(SystemExit) as exc:
        script.run_cli(args)

    assert exc.value.code == 2


def test_check_batches_exits_2_when_no_schema_configuration_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _make_check_batches_script(monkeypatch, [])
    args = Namespace(
        schema=None, input=str(tmp_path), verbose=False, json_summary=False
    )

    with pytest.raises(SystemExit) as exc:
        script.run_cli(args)

    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Fix 3: generate_line_ranges.py
# ---------------------------------------------------------------------------


def test_generate_line_ranges_exits_1_on_file_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from main.generate_line_ranges import GenerateLineRangesScript

    text_file = tmp_path / "sample.txt"
    text_file.write_text("hello world\n", encoding="utf-8")

    script = GenerateLineRangesScript()
    script.model_config = {"extraction_model": {"name": "gpt-4o"}}
    script.chunking_and_context_config = {"chunking": {"default_tokens_per_chunk": 100}}

    # Force one failure regardless of the real generation logic.
    monkeypatch.setattr(script, "_process_files", lambda *a, **kw: (0, 1))

    args = Namespace(
        tokens=None,
        input=str(tmp_path),
        verbose=False,
        first_n_chunks=None,
        last_n_chunks=None,
    )

    with pytest.raises(SystemExit) as exc:
        script.run_cli(args)

    assert exc.value.code == 1


def test_generate_line_ranges_no_exit_when_all_succeed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from main.generate_line_ranges import GenerateLineRangesScript

    text_file = tmp_path / "sample.txt"
    text_file.write_text("hello world\n", encoding="utf-8")

    script = GenerateLineRangesScript()
    script.model_config = {"extraction_model": {"name": "gpt-4o"}}
    script.chunking_and_context_config = {"chunking": {"default_tokens_per_chunk": 100}}

    monkeypatch.setattr(script, "_process_files", lambda *a, **kw: (1, 0))

    args = Namespace(
        tokens=None,
        input=str(tmp_path),
        verbose=False,
        first_n_chunks=None,
        last_n_chunks=None,
    )

    # Should complete without raising SystemExit.
    script.run_cli(args)


def test_generate_line_ranges_collects_md_files(tmp_path: Path) -> None:
    """Fix 12: CLI file collection must include .md alongside .txt."""
    from main.cli_args import get_files_from_path

    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "a_line_ranges.txt").write_text("x", encoding="utf-8")

    exclude_patterns = ["*_line_ranges.txt", "*_context.txt", "*_output.txt"]
    seen: dict[Path, None] = {}
    for pattern in ("*.txt", "*.md"):
        for found in get_files_from_path(
            tmp_path, pattern=pattern, exclude_patterns=exclude_patterns
        ):
            seen[found] = None
    files = sorted(seen)

    assert (tmp_path / "a.txt") in files
    assert (tmp_path / "b.md") in files
    assert (tmp_path / "a_line_ranges.txt") not in files


def test_default_excludes_cover_legacy_singular_sidecar(tmp_path: Path) -> None:
    """The legacy singular '_line_range.txt' sidecar (recognized by the
    readjuster and excluded by interactive discovery) must be excluded by
    CLI collection too."""
    from main.cli_args import DEFAULT_EXCLUDE_PATTERNS, get_files_from_path

    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "a_line_range.txt").write_text("x", encoding="utf-8")
    (tmp_path / "a_line_ranges.txt").write_text("x", encoding="utf-8")

    files = get_files_from_path(
        tmp_path, pattern="*.txt", exclude_patterns=list(DEFAULT_EXCLUDE_PATTERNS)
    )

    assert (tmp_path / "a.txt") in files
    assert (tmp_path / "a_line_range.txt") not in files
    assert (tmp_path / "a_line_ranges.txt") not in files


def test_generate_line_ranges_cli_uses_shared_exclusion_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_cli kept a local exclusion list that omitted the legacy singular
    sidecar, so '*_line_range.txt' was re-ingested as an input file."""
    from main.generate_line_ranges import GenerateLineRangesScript

    (tmp_path / "doc.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / "doc.md").write_text("hello\n", encoding="utf-8")
    (tmp_path / "doc_line_range.txt").write_text("(1, 5)\n", encoding="utf-8")
    (tmp_path / "doc_line_ranges.txt").write_text("(1, 5)\n", encoding="utf-8")

    script = GenerateLineRangesScript()
    script.model_config = {"extraction_model": {"name": "gpt-4o"}}
    script.chunking_and_context_config = {"chunking": {"default_tokens_per_chunk": 100}}

    seen_files: list[Path] = []

    def _capture(files, *a, **kw):
        seen_files.extend(files)
        return (len(files), 0)

    monkeypatch.setattr(script, "_process_files", _capture)

    script.run_cli(
        Namespace(
            tokens=None,
            input=str(tmp_path),
            verbose=False,
            first_n_chunks=None,
            last_n_chunks=None,
        )
    )

    names = {f.name for f in seen_files}
    assert names == {"doc.txt", "doc.md"}


# ---------------------------------------------------------------------------
# process_text_files.py --json count semantics
# ---------------------------------------------------------------------------


async def _run_process_cli(
    *,
    per_file_status: dict[str, str],
    tmp_path: Path,
    config_loader: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[dict[str, Any], str, int | None]:
    """Run the CLI path with a stubbed FileProcessor; no API calls are made."""
    import main.process_text_files as ptf
    from main.cli_args import create_process_parser

    input_root = tmp_path / "input"
    input_root.mkdir()
    for stem in per_file_status:
        (input_root / f"{stem}.txt").write_text("text\n", encoding="utf-8")
    out_root = tmp_path / "out"
    out_root.mkdir()

    class _SchemaManager:
        @staticmethod
        def get_available_schemas() -> dict[str, dict[str, Any]]:
            return {"TestSchema": {"type": "object"}}

    monkeypatch.setattr(ptf, "load_schema_manager", lambda: _SchemaManager())
    monkeypatch.setattr(ptf, "validate_schema_paths", lambda *a, **k: True)

    async def _fake_process_file(self: Any, *, file_path: Path, **_kw: Any) -> str:
        return per_file_status[file_path.stem]

    monkeypatch.setattr(ptf.FileProcessor, "process_file", _fake_process_file)

    args = create_process_parser().parse_args(
        [
            "--schema",
            "TestSchema",
            "--input",
            str(input_root),
            "--json",
            "--non-interactive",
        ]
    )

    exit_code: int | None = None
    try:
        await ptf._run_cli_mode(
            args,
            config_loader,
            {"general": {"allow_relative_paths": True}},
            {"extraction_model": {"name": "gpt-4o"}},
            {"chunking": {"default_tokens_per_chunk": 10}, "context": {}},
            {"TestSchema": {"output": str(out_root)}},
        )
    except SystemExit as exc:  # pragma: no cover - depends on the scenario
        exit_code = int(exc.code or 0)

    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    return payload, out, exit_code


@pytest.mark.asyncio
async def test_cli_json_counts_are_disjoint(
    tmp_path: Path,
    config_loader: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A skipped file was counted both as 'complete' and as 'skipped', so the
    buckets overcounted the file total and the summary line claimed a
    completion that never happened."""
    payload, out, exit_code = await _run_process_cli(
        per_file_status={"a": "complete", "b": "skipped", "c": "skipped"},
        tmp_path=tmp_path,
        config_loader=config_loader,
        monkeypatch=monkeypatch,
        capsys=capsys,
    )

    assert payload["files"] == 3
    assert payload["complete"] == 1
    assert payload["skipped"] == 2
    assert payload["partial"] == 0
    assert payload["failed"] == 0
    assert (
        payload["complete"]
        + payload["partial"]
        + payload["failed"]
        + payload["skipped"]
        == payload["files"]
    )
    assert "1 complete" in out
    # Skips still count as success for the exit-code contract.
    assert exit_code is None


@pytest.mark.asyncio
async def test_cli_exit_1_on_partial_with_disjoint_counts(
    tmp_path: Path,
    config_loader: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload, _out, exit_code = await _run_process_cli(
        per_file_status={"a": "skipped", "b": "partial"},
        tmp_path=tmp_path,
        config_loader=config_loader,
        monkeypatch=monkeypatch,
        capsys=capsys,
    )

    assert payload["complete"] == 0
    assert payload["skipped"] == 1
    assert payload["partial"] == 1
    assert exit_code == 1


# ---------------------------------------------------------------------------
# Fix 4: repair_extractions.py
# ---------------------------------------------------------------------------


def _make_repair_script(monkeypatch: pytest.MonkeyPatch, candidates):
    import main.repair_extractions as repair_mod
    from main.repair_extractions import RepairExtractionsScript

    script = RepairExtractionsScript()
    monkeypatch.setattr(script, "_load_repair_config", lambda: None)
    monkeypatch.setattr(
        repair_mod, "_discover_candidate_temp_files", lambda *a, **kw: candidates
    )
    script.repo_info_list = []
    script.processing_settings = {}
    return script


def test_repair_extractions_exits_2_when_schema_filter_matches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        {
            "schema_name": "OtherSchema",
            "schema_config": {},
            "temp_file": Path("x_temp.jsonl"),
            "temp_files": [Path("x_temp.jsonl")],
            "identifier": "x",
            "final_json": Path("x_output.json"),
            "responses_count": 0,
            "tracking_count": 0,
            "has_final": False,
            "tracking": [],
            "responses": [],
            "custom_id_map": None,
            "order_map": None,
        }
    ]
    script = _make_repair_script(monkeypatch, candidates)
    args = Namespace(schema="Missing", files=None, force=True, verbose=False)

    with pytest.raises(SystemExit) as exc:
        script.run_cli(args)

    assert exc.value.code == 2


def test_repair_extractions_exits_2_when_files_filter_matches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        {
            "schema_name": "TestSchema",
            "schema_config": {},
            "temp_file": Path("x_temp.jsonl"),
            "temp_files": [Path("x_temp.jsonl")],
            "identifier": "x",
            "final_json": Path("x_output.json"),
            "responses_count": 0,
            "tracking_count": 0,
            "has_final": False,
            "tracking": [],
            "responses": [],
            "custom_id_map": None,
            "order_map": None,
        }
    ]
    script = _make_repair_script(monkeypatch, candidates)
    args = Namespace(
        schema=None, files=["nonexistent_temp.jsonl"], force=True, verbose=False
    )

    with pytest.raises(SystemExit) as exc:
        script.run_cli(args)

    assert exc.value.code == 2


def test_repair_temp_file_reports_status_based_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix 6: an early-bail repair must be counted as 'skipped', not a success."""
    import main.repair_extractions as repair_mod

    class _Ui:
        def print_subsection_header(self, *a: object, **kw: object) -> None:
            pass

        def print_warning(self, *a: object, **kw: object) -> None:
            pass

    candidate = {
        "schema_name": "TestSchema",
        "schema_config": {},
        "temp_file": Path("x_temp.jsonl"),
        "temp_files": [Path("x_temp.jsonl")],
        "identifier": "x",
        "tracking": [],  # no tracking entries -> early bail
        "responses": [],
        "custom_id_map": None,
        "order_map": None,
    }

    status = repair_mod._repair_temp_file(candidate, {}, _Ui())  # type: ignore[arg-type]
    assert status == "skipped"


def test_repair_confirm_keyboard_interrupt_exits_130(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl+C at the CLI confirm prompt must exit 130, not report success."""
    candidates = [
        {
            "schema_name": "TestSchema",
            "schema_config": {},
            "temp_file": Path("x_temp.jsonl"),
            "temp_files": [Path("x_temp.jsonl")],
            "identifier": "x",
            "final_json": Path("x_output.json"),
            "responses_count": 0,
            "tracking_count": 0,
            "has_final": False,
            "tracking": [],
            "responses": [],
            "custom_id_map": None,
            "order_map": None,
        }
    ]
    script = _make_repair_script(monkeypatch, candidates)
    args = Namespace(schema=None, files=None, force=False, verbose=False)

    def _raise_interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_interrupt)

    with pytest.raises(SystemExit) as exc:
        script.run_cli(args)

    assert exc.value.code == 130


# ---------------------------------------------------------------------------
# Fix 5: cancel_batches.py
# ---------------------------------------------------------------------------


def _make_cancel_batches_script(monkeypatch: pytest.MonkeyPatch):
    from main.cancel_batches import CancelBatchesScript

    script = CancelBatchesScript()
    monkeypatch.setattr(script, "_load_root_folders", lambda: None)
    return script


def test_cancel_batches_exits_2_without_force(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _make_cancel_batches_script(monkeypatch)
    monkeypatch.setattr(
        script, "_get_cancellable_batches", lambda: [({"batch_id": "b1"}, object())]
    )

    args = Namespace(force=False, verbose=False)

    with pytest.raises(SystemExit) as exc:
        script.run_cli(args)

    assert exc.value.code == 2


def test_cancel_batches_exits_1_on_failed_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _make_cancel_batches_script(monkeypatch)
    monkeypatch.setattr(
        script, "_get_cancellable_batches", lambda: [({"batch_id": "b1"}, object())]
    )
    monkeypatch.setattr(script, "_cancel_batches", lambda *a, **kw: (0, 1))

    args = Namespace(force=True, verbose=False)

    with pytest.raises(SystemExit) as exc:
        script.run_cli(args)

    assert exc.value.code == 1


def test_cancel_batches_no_exit_when_all_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _make_cancel_batches_script(monkeypatch)
    monkeypatch.setattr(
        script, "_get_cancellable_batches", lambda: [({"batch_id": "b1"}, object())]
    )
    monkeypatch.setattr(script, "_cancel_batches", lambda *a, **kw: (1, 0))

    args = Namespace(force=True, verbose=False)

    # Should complete without raising SystemExit.
    script.run_cli(args)


def _cancel_script_with_unreadable_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Build a script whose only tracked batch raises on status lookup."""
    import main.cancel_batches as cancel_mod

    script = cancel_mod.CancelBatchesScript()
    monkeypatch.setattr(script, "_load_root_folders", lambda: None)
    monkeypatch.setattr(
        cancel_mod,
        "_scan_for_batch_tracking",
        lambda folders: [{"batch_id": "b1", "provider": "openai"}],
    )

    def _boom(_provider: str):
        raise RuntimeError("no API key")

    monkeypatch.setattr(cancel_mod, "get_batch_backend", _boom)
    return script


def test_cancel_batches_exits_1_when_all_status_lookups_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A batch whose status cannot be read may still be running: reporting
    'No batches require cancellation' and exiting 0 is a false all-clear."""
    script = _cancel_script_with_unreadable_batch(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc:
        script.run_cli(Namespace(force=True, verbose=False))

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "No batches require cancellation" not in out
    assert "Could not determine the status" in out


def test_cancel_batches_interactive_warns_instead_of_all_clear(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from unittest.mock import MagicMock

    script = _cancel_script_with_unreadable_batch(monkeypatch, tmp_path)
    script.ui = MagicMock()

    script.run_interactive()

    assert script.ui.print_warning.called
    infos = [str(call.args[0]) for call in script.ui.print_info.call_args_list]
    assert not any(msg.startswith("No batches require cancellation") for msg in infos)


def test_cancel_batches_status_lookup_failures_reset_between_scans(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _cancel_script_with_unreadable_batch(monkeypatch, tmp_path)

    assert script._get_cancellable_batches() == []
    assert script.status_lookup_failures == 1
    assert script._get_cancellable_batches() == []
    assert script.status_lookup_failures == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
