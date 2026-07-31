"""Regression tests for CLI/UI robustness fixes.

Covers offline hardening fixes:

* The interactive line-range prompt (``ChunkHandler.adjust_line_ranges``)
  degrades gracefully when stdin is unavailable instead of crashing with an
  ``EOFError``.
* ``check_and_wait_for_token_limit`` lets ``asyncio.CancelledError`` propagate
  (so a mid-wait Ctrl+C is not misreported as a token-budget stop).
* ``generate_line_ranges``' interactive file selection survives absolute-path
  and ``../`` input, excludes every auxiliary sidecar, and re-prompts instead
  of killing the process on a typo or an empty folder.
* ``line_range_readjuster._prompt_int`` warns instead of silently clamping a
  non-positive value.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modules.infra.chunking import ChunkHandler, TextProcessor
from modules.infra.token_tracker import check_and_wait_for_token_limit


@pytest.mark.unit
def test_adjust_line_ranges_survives_missing_stdin():
    """EOFError from input() is caught: remaining chunks keep their defaults."""
    handler = ChunkHandler(
        model_name="gpt-4o",
        default_tokens_per_chunk=100,
        text_processor=TextProcessor(),
    )
    with patch("builtins.input", side_effect=EOFError):
        result = handler.adjust_line_ranges(
            initial_ranges=[(1, 5)],
            original_start_line=1,
            total_processed_lines=10,
        )
    # No crash; the current default range is kept and the remainder is filled
    # in without further prompting.
    assert result == [(1, 5), (6, 10)]


@pytest.mark.unit
def test_adjust_line_ranges_survives_keyboard_interrupt():
    handler = ChunkHandler(
        model_name="gpt-4o",
        default_tokens_per_chunk=100,
        text_processor=TextProcessor(),
    )
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        result = handler.adjust_line_ranges(
            initial_ranges=[(1, 5)],
            original_start_line=1,
            total_processed_lines=10,
        )
    assert result == [(1, 5), (6, 10)]


@pytest.mark.asyncio
async def test_check_and_wait_propagates_cancellation():
    """A cancellation during the wait must propagate, not return False."""
    fake_tracker = SimpleNamespace(
        enabled=True,
        is_limit_reached=lambda: True,
        get_stats=lambda: {"tokens_used_today": 100, "daily_limit": 100},
        get_reset_time=lambda: datetime.now(UTC) + timedelta(minutes=5),
        get_seconds_until_reset=lambda: 300,
        describe_pool_block=lambda: None,
    )
    with (
        patch(
            "modules.infra.token_tracker.get_token_tracker",
            return_value=fake_tracker,
        ),
        patch("asyncio.sleep", side_effect=asyncio.CancelledError),
        pytest.raises(asyncio.CancelledError),
    ):
        await check_and_wait_for_token_limit(ui=None, logger=None)


# ---------------------------------------------------------------------------
# generate_line_ranges: interactive single-file selection hardening
# ---------------------------------------------------------------------------


def _script_with_ui(answers: list[str]) -> tuple[object, MagicMock]:
    """Build a GenerateLineRangesScript wired to a scripted mock UI."""
    from main.generate_line_ranges import GenerateLineRangesScript

    script = GenerateLineRangesScript.__new__(GenerateLineRangesScript)
    mock_ui = MagicMock()
    mock_ui.get_input.side_effect = answers
    script.ui = mock_ui  # type: ignore[attr-defined]
    return script, mock_ui


@pytest.mark.unit
def test_single_file_absolute_path_does_not_crash(tmp_path: Path) -> None:
    """Python 3.13 raises NotImplementedError for non-relative glob patterns;
    the wizard must catch it and re-prompt rather than die."""
    raw = tmp_path / "input"
    raw.mkdir()
    (raw / "document.txt").write_text("content", encoding="utf-8")

    # Absolute path first, then a bare Enter to back out.
    script, mock_ui = _script_with_ui([str(raw / "document.txt"), ""])
    result = script._select_single_file(raw)  # type: ignore[attr-defined]

    assert result is None
    infos = [str(call.args[0]) for call in mock_ui.print_info.call_args_list]
    assert any("relative to the input directory" in m for m in infos)


@pytest.mark.unit
def test_single_file_cannot_escape_input_directory(tmp_path: Path) -> None:
    """A '../' pattern resolves outside the configured directory and must be
    filtered out (rglob happily returns such matches)."""
    raw = tmp_path / "input"
    raw.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("classified", encoding="utf-8")

    script, mock_ui = _script_with_ui(["../outside/secret.txt", ""])
    result = script._select_single_file(raw)  # type: ignore[attr-defined]

    assert result is None
    assert mock_ui.print_error.called


@pytest.mark.unit
def test_single_file_excludes_singular_line_range_sidecar(tmp_path: Path) -> None:
    raw = tmp_path / "input"
    raw.mkdir()
    (raw / "document_line_range.txt").write_text("(1, 5)", encoding="utf-8")

    script, mock_ui = _script_with_ui(["document_line_range.txt", ""])
    result = script._select_single_file(raw)  # type: ignore[attr-defined]

    assert result is None
    assert mock_ui.print_error.called


@pytest.mark.unit
def test_single_file_typo_reprompts_instead_of_exiting(tmp_path: Path) -> None:
    raw = tmp_path / "input"
    raw.mkdir()
    (raw / "document.txt").write_text("content", encoding="utf-8")

    # A typo, then the correct name: the second attempt must succeed.
    script, _ui = _script_with_ui(["documnet.txt", "document.txt"])
    result = script._select_single_file(raw)  # type: ignore[attr-defined]

    assert result is not None
    assert [p.name for p in result] == ["document.txt"]


@pytest.mark.unit
def test_empty_folder_reprompts_instead_of_exiting(tmp_path: Path) -> None:
    """An empty input folder must not kill the process (select_input_source
    re-prompts in the same situation)."""
    from main.generate_line_ranges import GenerateLineRangesScript

    raw = tmp_path / "input"
    raw.mkdir()

    script = GenerateLineRangesScript.__new__(GenerateLineRangesScript)
    mock_ui = MagicMock()
    script.ui = mock_ui  # type: ignore[attr-defined]

    files = script._select_folder_files(raw)  # type: ignore[attr-defined]

    assert files == []
    assert mock_ui.print_error.called


@pytest.mark.unit
def test_folder_selection_excludes_all_auxiliary_sidecars(tmp_path: Path) -> None:
    raw = tmp_path / "input"
    raw.mkdir()
    (raw / "document.txt").write_text("content", encoding="utf-8")
    for sidecar in (
        "document_line_ranges.txt",
        "document_line_range.txt",
        "document_extract_context.txt",
        "document_output.txt",
    ):
        (raw / sidecar).write_text("x", encoding="utf-8")

    from main.generate_line_ranges import GenerateLineRangesScript

    script = GenerateLineRangesScript.__new__(GenerateLineRangesScript)
    script.ui = MagicMock()  # type: ignore[attr-defined]

    files = script._select_folder_files(raw)  # type: ignore[attr-defined]

    assert [p.name for p in files] == ["document.txt"]


# ---------------------------------------------------------------------------
# line_range_readjuster: non-positive input is reported, not silently clamped
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prompt_int_warns_when_clamping_ui_path() -> None:
    from main.line_range_readjuster import _prompt_int

    ui = MagicMock()
    ui.get_input.return_value = "0"

    assert _prompt_int(ui, "How many?", 5) == 1
    warnings = [str(call.args[0]) for call in ui.print_warning.call_args_list]
    assert any("at least 1" in w for w in warnings)


@pytest.mark.unit
def test_prompt_int_accepts_positive_without_warning() -> None:
    from main.line_range_readjuster import _prompt_int

    ui = MagicMock()
    ui.get_input.return_value = "7"

    assert _prompt_int(ui, "How many?", 5) == 7
    assert not ui.print_warning.called


@pytest.mark.unit
def test_prompt_int_warns_when_clamping_stdin_path(capsys) -> None:
    from main.line_range_readjuster import _prompt_int

    with patch("builtins.input", return_value="-5"):
        assert _prompt_int(None, "How many?", 5) == 1
    assert "at least 1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# repair_extractions: the shared parser validates bounds at parse time
# ---------------------------------------------------------------------------


def _repair_script_with_selection(monkeypatch, selection: str, candidate_count: int):
    import main.repair_extractions as repair

    script = repair.RepairExtractionsScript.__new__(repair.RepairExtractionsScript)
    mock_ui = MagicMock()
    mock_ui.get_input.return_value = selection
    mock_ui.confirm.return_value = False  # never proceed to real repairs
    script.ui = mock_ui  # type: ignore[attr-defined]
    script.repo_info_list = []  # type: ignore[attr-defined]

    monkeypatch.setattr(script, "_load_repair_config", lambda: None)
    monkeypatch.setattr(
        repair,
        "_discover_candidate_temp_files",
        lambda _repos, _ui: [
            {
                "temp_file": Path(f"temp_{i}.jsonl"),
                "has_final": False,
                "responses_count": 1,
                "tracking_count": 1,
                "schema_name": "TestSchema",
            }
            for i in range(candidate_count)
        ],
    )
    return script, mock_ui


@pytest.mark.unit
def test_repair_selection_out_of_range_is_rejected(monkeypatch) -> None:
    script, mock_ui = _repair_script_with_selection(monkeypatch, "5", 2)

    with pytest.raises(SystemExit) as exc:
        script.run_interactive()

    assert exc.value.code == 1
    assert mock_ui.print_error.called


@pytest.mark.unit
def test_repair_selection_in_range_reaches_confirmation(monkeypatch) -> None:
    script, mock_ui = _repair_script_with_selection(monkeypatch, "1,2", 2)

    script.run_interactive()

    # Parsing succeeded, so the user was asked to confirm two repairs.
    assert mock_ui.confirm.called
    assert "2 file(s)" in str(mock_ui.confirm.call_args.args[0])
