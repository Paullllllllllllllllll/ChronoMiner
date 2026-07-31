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

from argparse import Namespace
from pathlib import Path
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
