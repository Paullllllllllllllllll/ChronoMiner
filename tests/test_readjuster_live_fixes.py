"""Regression tests for live-testing bugs CM-7, CM-8, and CM-9 (July 2026).

CM-7: range deletions use a dedicated, stricter certainty gate
      (retry.delete_certainty_threshold, default 85) and every accepted
      deletion is logged at WARNING with the dropped line span.
CM-8: sliced adjustment runs (--first-n-chunks) compose with --resume via a
      fingerprint chain: the post-write fingerprint of _line_ranges.txt is
      recorded in the temp JSONL header, and resume accepts the artifact
      when the file matches either the pre-run or the post-write fingerprint.
CM-9: INFO records from plain logging.getLogger(__name__) module loggers
      (context resolution, readjuster) reach the shared file/console
      handlers via top-level namespace configuration, without duplication.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from modules.infra.jsonl import (
    compute_ranges_fingerprint,
    read_jsonl_header,
)
from modules.line_ranges.readjuster import (
    BoundaryDecision,
    LineRangeReadjuster,
    RangeResult,
)

# ---------------------------------------------------------------------------
# Shared helpers (mirroring tests/test_readjuster_persistence.py conventions)
# ---------------------------------------------------------------------------


class _async_noop_context:
    """Async context manager that yields a dummy extractor."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> MagicMock:
        return MagicMock()

    async def __aexit__(self, *exc: Any) -> None:
        pass


def _make_readjuster(
    model_name: str = "gpt-4o",
    context_window: int = 3,
    retry_config: dict[str, Any] | None = None,
) -> LineRangeReadjuster:
    """Create a readjuster without needing real API keys or prompt files."""
    with (
        patch(
            "modules.line_ranges.readjuster.load_prompt_template",
            return_value="fake prompt",
        ),
        patch(
            "modules.line_ranges.readjuster.detect_capabilities",
            return_value=MagicMock(supports_prompt_caching=False),
        ),
    ):
        return LineRangeReadjuster(
            {"extraction_model": {"name": model_name}},
            context_window=context_window,
            retry_config=retry_config,
        )


def _fake_range_result(
    index: int,
    original: tuple[int, int],
    adjusted: tuple[int, int],
    delete: bool = False,
) -> RangeResult:
    return RangeResult(
        range_index=index,
        original_range=original,
        adjusted_range=adjusted,
        should_delete=delete,
        decision=BoundaryDecision(
            contains_no_semantic_boundary=delete,
            needs_more_context=False,
            certainty=90,
            semantic_marker="marker" if not delete else None,
        ),
        attempts=[],
        total_llm_calls=1,
    )


def _no_content_payload(certainty: int) -> dict[str, Any]:
    return {
        "contains_no_semantic_boundary": True,
        "needs_more_context": False,
        "boundary_already_on_target": False,
        "certainty": certainty,
        "semantic_marker": "",
    }


# ---------------------------------------------------------------------------
# CM-7: dedicated delete certainty gate
# ---------------------------------------------------------------------------


class TestCM7DeleteCertaintyGate:
    """Deletions require retry.delete_certainty_threshold (default 85)."""

    RAW_LINES = [f"Line {i}\n" for i in range(1, 21)]

    @pytest.mark.unit
    def test_default_is_85(self) -> None:
        readjuster = _make_readjuster(retry_config={"certainty_threshold": 70})
        assert readjuster.delete_certainty_threshold == 85
        assert readjuster.certainty_threshold == 70

    @pytest.mark.unit
    def test_configurable_via_retry_config(self) -> None:
        readjuster = _make_readjuster(retry_config={"delete_certainty_threshold": 92})
        assert readjuster.delete_certainty_threshold == 92

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_deletion_rejected_below_85_even_above_global_threshold(
        self,
    ) -> None:
        """A no-content verdict at certainty 80 (above the global 70) must
        NOT delete the range."""
        readjuster = _make_readjuster(retry_config={"certainty_threshold": 70})

        async def mock_run_model(**_kwargs: Any) -> dict[str, Any]:
            return _no_content_payload(80)

        with patch.object(readjuster, "_run_model", side_effect=mock_run_model):
            should_delete, reanchored, attempts = await readjuster._verify_no_content(
                extractor=MagicMock(),
                raw_lines=self.RAW_LINES,
                original_range=(5, 12),
                range_index=1,
                boundary_type="BibliographicEntries",
                context=None,
            )

        assert should_delete is False
        assert reanchored is None
        assert len(attempts) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_deletion_accepted_at_or_above_85_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A no-content verdict at certainty 90 deletes, and the acceptance
        is logged at WARNING with the dropped line span."""
        readjuster = _make_readjuster(retry_config={"certainty_threshold": 70})

        async def mock_run_model(**_kwargs: Any) -> dict[str, Any]:
            return _no_content_payload(90)

        with (
            patch.object(readjuster, "_run_model", side_effect=mock_run_model),
            caplog.at_level(logging.WARNING, logger="modules.line_ranges.readjuster"),
        ):
            should_delete, _, _ = await readjuster._verify_no_content(
                extractor=MagicMock(),
                raw_lines=self.RAW_LINES,
                original_range=(5, 12),
                range_index=1,
                boundary_type="BibliographicEntries",
                context=None,
            )

        assert should_delete is True
        warning_messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]
        assert any(
            "DELETION accepted" in msg and "5-12" in msg and "90" in msg
            for msg in warning_messages
        ), f"expected deletion WARNING with span and certainty: {warning_messages}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_boundary_moves_unaffected_between_70_and_84(self) -> None:
        """A boundary move at certainty 75 is still accepted (only the
        delete path uses the stricter gate)."""
        readjuster = _make_readjuster(retry_config={"certainty_threshold": 70})

        raw_lines = [
            "filler alpha\n",
            "filler beta\n",
            "filler gamma\n",
            "1864. Host and Guest, a book about dinners.\n",
            "continuation of the entry\n",
            "more continuation\n",
            "filler delta\n",
            "filler epsilon\n",
            "filler zeta\n",
            "filler eta\n",
        ]

        async def mock_run_model(**_kwargs: Any) -> dict[str, Any]:
            return {
                "contains_no_semantic_boundary": False,
                "needs_more_context": False,
                "boundary_already_on_target": False,
                "certainty": 75,
                "semantic_marker": "Host and Guest",
            }

        with patch.object(readjuster, "_run_model", side_effect=mock_run_model):
            result = await readjuster._process_single_range(
                extractor=MagicMock(),
                raw_lines=raw_lines,
                original_range=(5, 10),
                range_index=1,
                boundary_type="BibliographicEntries",
                context=None,
            )

        assert result.should_delete is False
        assert result.adjusted_range == (4, 10)
        assert not any(
            attempt["decision_type"] == "low_certainty" for attempt in result.attempts
        )


# ---------------------------------------------------------------------------
# CM-8: sliced runs compose with --resume via the fingerprint chain
# ---------------------------------------------------------------------------


class TestCM8SlicedResumeFingerprintChain:
    """A sliced run's rewrite of _line_ranges.txt must not invalidate its
    own temp JSONL on the follow-up resume run."""

    @staticmethod
    def _setup(tmp_path: Path) -> tuple[Path, Path]:
        text_file = tmp_path / "sample.txt"
        text_file.write_text(
            "\n".join(f"Line {i}" for i in range(1, 31)) + "\n",
            encoding="utf-8",
        )
        lr_file = tmp_path / "sample_line_ranges.txt"
        lr_file.write_text("(1, 10)\n(11, 20)\n(21, 30)\n", encoding="utf-8")
        return text_file, lr_file

    @staticmethod
    def _run_patches(readjuster: LineRangeReadjuster, mock_process: Any) -> Any:
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(
            patch.object(readjuster, "_process_single_range", side_effect=mock_process)
        )
        mock_provider = stack.enter_context(
            patch("modules.line_ranges.readjuster.ProviderConfig")
        )
        mock_provider._detect_provider.return_value = "openai"
        mock_provider._get_api_key.return_value = "fake-key"
        stack.enter_context(
            patch(
                "modules.line_ranges.readjuster.open_extractor",
                new_callable=lambda: _async_noop_context,
            )
        )
        stack.enter_context(
            patch(
                "modules.line_ranges.readjuster.resolve_context_for_readjustment",
                return_value=(None, None),
            )
        )
        return stack

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sliced_run_records_post_write_fingerprint(
        self, tmp_path: Path
    ) -> None:
        text_file, lr_file = self._setup(tmp_path)
        readjuster = _make_readjuster()

        async def mock_process(**kwargs: Any) -> RangeResult:
            idx = kwargs["range_index"]
            start, end = kwargs["original_range"]
            return _fake_range_result(idx, (start, end), (start + 2, end))

        with self._run_patches(readjuster, mock_process):
            await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=lr_file,
                boundary_type="TestSchema",
                first_n_chunks=2,
            )

        header = read_jsonl_header(tmp_path / "sample_line_ranges_adjust_temp.jsonl")
        assert header is not None
        # Unfinalized (sliced), but the post-write fingerprint is recorded
        # and matches the rewritten ranges file.
        assert header.get("completed_at") is None
        assert header.get("post_write_ranges_fingerprint") == (
            compute_ranges_fingerprint(lr_file)
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sliced_then_unsliced_resume_processes_only_remainder(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """After --first-n-chunks 2, an unsliced resume run re-adjusts only
        range 3; the sliced work is reused, not paid for twice. The resume
        must NOT emit drift warnings for the recognized own output."""
        text_file, lr_file = self._setup(tmp_path)
        readjuster = _make_readjuster()
        processed: list[int] = []

        async def mock_process(**kwargs: Any) -> RangeResult:
            idx = kwargs["range_index"]
            processed.append(idx)
            start, end = kwargs["original_range"]
            return _fake_range_result(idx, (start, end), (start + 2, end))

        with self._run_patches(readjuster, mock_process):
            await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=lr_file,
                boundary_type="TestSchema",
                first_n_chunks=2,
            )
        assert processed == [1, 2]
        # The sliced run rewrote the file in place with adjusted ranges 1-2.
        assert lr_file.read_text(encoding="utf-8") == "(3, 10)\n(13, 20)\n(21, 30)\n"

        with (
            self._run_patches(readjuster, mock_process),
            caplog.at_level(logging.WARNING, logger="modules.line_ranges.readjuster"),
        ):
            await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=lr_file,
                boundary_type="TestSchema",
            )

        # Only the remainder was processed; ranges 1-2 were not re-adjusted.
        assert processed == [1, 2, 3]
        # Recognized own output is healthy: no drift warning may be emitted.
        drift_warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.WARNING
            and "stale JSONL leak" in record.getMessage()
        ]
        assert drift_warnings == []
        # The final file keeps the sliced adjustments and adds range 3's.
        assert lr_file.read_text(encoding="utf-8") == "(3, 10)\n(13, 20)\n(23, 30)\n"
        # The now-complete artifact is finalized.
        header = read_jsonl_header(tmp_path / "sample_line_ranges_adjust_temp.jsonl")
        assert header is not None
        assert header.get("completed_at") is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_external_regeneration_still_invalidates(
        self, tmp_path: Path
    ) -> None:
        """A ranges file regenerated externally after the sliced run matches
        neither fingerprint, so the stale JSONL is discarded and all ranges
        are re-adjusted (v1.19 stale-rejection semantics preserved)."""
        text_file, lr_file = self._setup(tmp_path)
        readjuster = _make_readjuster()
        processed: list[int] = []

        async def mock_process(**kwargs: Any) -> RangeResult:
            idx = kwargs["range_index"]
            processed.append(idx)
            return _fake_range_result(
                idx, kwargs["original_range"], kwargs["original_range"]
            )

        with self._run_patches(readjuster, mock_process):
            await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=lr_file,
                boundary_type="TestSchema",
                first_n_chunks=2,
            )
        assert processed == [1, 2]

        # External regeneration (e.g. different tokens_per_chunk).
        lr_file.write_text("(1, 15)\n(16, 30)\n", encoding="utf-8")

        with self._run_patches(readjuster, mock_process):
            await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=lr_file,
                boundary_type="TestSchema",
            )
        # Both new ranges were processed from scratch.
        assert processed == [1, 2, 1, 2]

    @pytest.mark.unit
    def test_stale_deletion_not_applied_to_mismatched_range(
        self, tmp_path: Path
    ) -> None:
        """_rebuild_ranges_from_jsonl never applies a recorded deletion to a
        range that differs from the recorded original (index-drift guard)."""
        temp_jsonl = tmp_path / "x_adjust_temp.jsonl"
        record = _fake_range_result(1, (1, 10), (1, 10), delete=True).to_jsonl_record(
            "x"
        )
        temp_jsonl.write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        adjusted, deleted = LineRangeReadjuster._rebuild_ranges_from_jsonl(
            temp_jsonl,
            [(5, 14)],  # current range at index 1 differs
        )
        assert deleted == []
        assert adjusted == [(5, 14)]

        # Matching original: the deletion applies as recorded.
        adjusted, deleted = LineRangeReadjuster._rebuild_ranges_from_jsonl(
            temp_jsonl, [(1, 10)]
        )
        assert deleted == [0]
        assert adjusted == []

    @pytest.mark.unit
    def test_recognized_own_output_no_warning_genuine_drift_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A current range equal to the record's adjusted value is recognized
        own output (no warning); a range matching neither the original nor
        the adjusted value is genuine drift (warning)."""
        temp_jsonl = tmp_path / "x_adjust_temp.jsonl"
        record = _fake_range_result(1, (1, 10), (3, 10)).to_jsonl_record("x")
        temp_jsonl.write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # Current range equals the recorded ADJUSTED value: own output.
        with caplog.at_level(logging.DEBUG, logger="modules.line_ranges.readjuster"):
            adjusted, deleted = LineRangeReadjuster._rebuild_ranges_from_jsonl(
                temp_jsonl, [(3, 10)]
            )
        assert adjusted == [(3, 10)]
        assert deleted == []
        assert not any(
            record.levelno >= logging.WARNING for record in caplog.records
        ), "recognized own output must not emit a WARNING"
        assert any(
            record.levelno == logging.DEBUG and "own output" in record.getMessage()
            for record in caplog.records
        )

        # Current range matches NEITHER original nor adjusted: genuine drift.
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="modules.line_ranges.readjuster"):
            LineRangeReadjuster._rebuild_ranges_from_jsonl(temp_jsonl, [(5, 14)])
        assert any(
            "stale JSONL leak" in record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ), "genuine drift must still emit the WARNING"


# ---------------------------------------------------------------------------
# CM-9: module-logger INFO records reach the shared handlers
# ---------------------------------------------------------------------------


class TestCM9ModuleLoggerVisibility:
    """Plain logging.getLogger(__name__) module loggers must be visible."""

    @pytest.mark.unit
    def test_modules_namespace_gets_file_and_console_handlers(self) -> None:
        from modules.infra.logger import setup_logger

        setup_logger("cm9_probe_top_level")
        modules_logger = logging.getLogger("modules")
        assert modules_logger.level == logging.INFO
        assert any(
            isinstance(handler, logging.FileHandler)
            for handler in modules_logger.handlers
        )
        assert any(
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            and handler.level == logging.WARNING
            for handler in modules_logger.handlers
        )

    @pytest.mark.unit
    def test_context_and_readjuster_loggers_emit_info(self) -> None:
        """The exact loggers behind the invisible live INFO lines are now
        enabled for INFO through the configured 'modules' namespace."""
        from modules.infra.logger import setup_logger

        setup_logger("cm9_probe_top_level")
        assert logging.getLogger("modules.config.context").isEnabledFor(logging.INFO)
        assert logging.getLogger("modules.line_ranges.readjuster").isEnabledFor(
            logging.INFO
        )

    @pytest.mark.unit
    def test_dotted_setup_logger_has_no_own_handlers(self) -> None:
        """setup_logger with a dotted name relies on the namespace handlers,
        so each record is emitted exactly once (no duplication)."""
        from modules.infra.logger import setup_logger

        logger = setup_logger("modules.infra.cm9_probe_child")
        assert logger.handlers == []
        assert logger.propagate is True

    @pytest.mark.unit
    def test_module_info_record_propagates(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from modules.infra.logger import setup_logger

        setup_logger("cm9_probe_top_level")
        with caplog.at_level(logging.INFO, logger="modules.config.context"):
            logging.getLogger("modules.config.context").info(
                "Using file-specific context: cm9-probe"
            )
        assert any(
            "Using file-specific context: cm9-probe" in record.getMessage()
            for record in caplog.records
        )
