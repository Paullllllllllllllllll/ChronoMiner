"""Tests for JSONL persistence, RangeResult, and range-level resume in the readjuster."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.infra.jsonl import (
    build_jsonl_header,
    compute_ranges_fingerprint,
    read_jsonl_records,
)
from modules.line_ranges.readjuster import (
    BoundaryDecision,
    LineRangeReadjuster,
    RangeResult,
)


# ---------------------------------------------------------------------------
# RangeResult
# ---------------------------------------------------------------------------

class TestRangeResult:
    def test_to_jsonl_record_envelope(self) -> None:
        decision = BoundaryDecision(
            contains_no_semantic_boundary=False,
            needs_more_context=False,
            certainty=92,
            semantic_marker="Chapter III",
        )
        result = RangeResult(
            range_index=3,
            original_range=(10, 50),
            adjusted_range=(12, 50),
            should_delete=False,
            decision=decision,
            attempts=[
                {
                    "window": [4, 16],
                    "window_index": 0,
                    "decision_type": "marker_found",
                    "certainty": 92,
                    "semantic_marker": "Chapter III",
                    "marker_matched": True,
                }
            ],
            total_llm_calls=1,
        )
        record = result.to_jsonl_record("my_document")

        assert record["custom_id"] == "my_document-range-3"
        body = record["response"]["body"]
        assert body["original_range"] == [10, 50]
        assert body["adjusted_range"] == [12, 50]
        assert body["should_delete"] is False
        assert body["decision"]["certainty"] == 92
        assert body["decision"]["semantic_marker"] == "Chapter III"
        assert len(body["attempts"]) == 1
        assert body["total_llm_calls"] == 1

    def test_to_jsonl_record_deletion(self) -> None:
        decision = BoundaryDecision(
            contains_no_semantic_boundary=True,
            needs_more_context=False,
            certainty=95,
        )
        result = RangeResult(
            range_index=7,
            original_range=(100, 150),
            adjusted_range=(100, 150),
            should_delete=True,
            decision=decision,
            attempts=[],
            total_llm_calls=3,
        )
        record = result.to_jsonl_record("doc")
        assert record["response"]["body"]["should_delete"] is True

    def test_to_jsonl_record_boundary_on_target(self) -> None:
        decision = BoundaryDecision(
            contains_no_semantic_boundary=False,
            needs_more_context=False,
            certainty=88,
            boundary_already_on_target=True,
        )
        result = RangeResult(
            range_index=2,
            original_range=(20, 60),
            adjusted_range=(20, 60),
            should_delete=False,
            decision=decision,
            attempts=[
                {
                    "window": [14, 26],
                    "window_index": 0,
                    "decision_type": "already_on_target",
                    "certainty": 88,
                    "semantic_marker": "",
                    "marker_matched": False,
                }
            ],
            total_llm_calls=1,
        )
        record = result.to_jsonl_record("doc")
        body = record["response"]["body"]
        assert body["adjusted_range"] == [20, 60]
        assert body["should_delete"] is False
        assert body["decision"]["boundary_already_on_target"] is True
        assert body["total_llm_calls"] == 1


class TestBoundaryDecisionRoundTrip:
    def test_round_trip_on_target_true(self) -> None:
        original = BoundaryDecision(
            contains_no_semantic_boundary=False,
            needs_more_context=False,
            certainty=85,
            boundary_already_on_target=True,
        )
        d = original.to_dict()
        restored = BoundaryDecision.from_payload(d)
        assert restored.boundary_already_on_target is True
        assert restored.certainty == 85
        assert restored.contains_no_semantic_boundary is False
        assert restored.needs_more_context is False

    def test_round_trip_on_target_false(self) -> None:
        original = BoundaryDecision(
            contains_no_semantic_boundary=False,
            needs_more_context=False,
            certainty=90,
            semantic_marker="Chapter I",
        )
        d = original.to_dict()
        restored = BoundaryDecision.from_payload(d)
        assert restored.boundary_already_on_target is False

    def test_from_payload_missing_field_defaults_false(self) -> None:
        payload = {
            "contains_no_semantic_boundary": False,
            "needs_more_context": False,
            "certainty": 80,
            "semantic_marker": "Test",
        }
        decision = BoundaryDecision.from_payload(payload)
        assert decision.boundary_already_on_target is False

    def test_round_trip_via_json(self) -> None:
        """Verify the record survives JSON serialization and deserialization."""
        decision = BoundaryDecision(
            contains_no_semantic_boundary=False,
            needs_more_context=False,
            certainty=80,
            semantic_marker="test",
        )
        result = RangeResult(
            range_index=1,
            original_range=(1, 10),
            adjusted_range=(3, 10),
            should_delete=False,
            decision=decision,
            attempts=[],
            total_llm_calls=1,
        )
        record = result.to_jsonl_record("stem")
        serialized = json.dumps(record)
        deserialized = json.loads(serialized)
        assert deserialized["custom_id"] == "stem-range-1"
        assert deserialized["response"]["body"]["adjusted_range"] == [3, 10]


# ---------------------------------------------------------------------------
# BoundaryDecision.to_dict
# ---------------------------------------------------------------------------

class TestBoundaryDecisionToDict:
    def test_serialization(self) -> None:
        d = BoundaryDecision(
            contains_no_semantic_boundary=False,
            needs_more_context=True,
            certainty=55,
            semantic_marker=None,
        )
        out = d.to_dict()
        assert out == {
            "contains_no_semantic_boundary": False,
            "needs_more_context": True,
            "boundary_already_on_target": False,
            "certainty": 55,
            "semantic_marker": None,
        }


# ---------------------------------------------------------------------------
# Readjuster temp JSONL persistence
# ---------------------------------------------------------------------------

def _make_readjuster(
    model_name: str = "gpt-4o",
    context_window: int = 3,
) -> LineRangeReadjuster:
    """Create a readjuster without needing real API keys or prompt files."""
    with patch(
        "modules.line_ranges.readjuster.load_prompt_template",
        return_value="fake prompt",
    ), patch(
        "modules.line_ranges.readjuster.detect_capabilities",
        return_value=MagicMock(supports_prompt_caching=False),
    ):
        return LineRangeReadjuster(
            {"extraction_model": {"name": model_name}},
            context_window=context_window,
        )


def _fake_range_result(
    index: int,
    original: Tuple[int, int],
    adjusted: Tuple[int, int],
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
        attempts=[{
            "window": [1, 10],
            "window_index": 0,
            "decision_type": "marker_found" if not delete else "no_semantic_boundary",
            "certainty": 90,
            "semantic_marker": "marker" if not delete else None,
            "marker_matched": not delete,
        }],
        total_llm_calls=1,
    )


class TestReadjusterTempJsonl:
    """Verify that the readjuster writes a temp JSONL with per-range records."""

    @pytest.mark.asyncio
    async def test_temp_jsonl_written(self, tmp_path: Path) -> None:
        """After adjustment, a temp JSONL should exist with one record per range."""
        # Prepare text and line ranges files
        text_file = tmp_path / "sample.txt"
        text_file.write_text(
            "\n".join(f"Line {i}" for i in range(1, 21)) + "\n",
            encoding="utf-8",
        )
        lr_file = tmp_path / "sample_line_ranges.txt"
        lr_file.write_text("(1, 10)\n(11, 20)\n", encoding="utf-8")

        readjuster = _make_readjuster()

        fake_results = [
            _fake_range_result(1, (1, 10), (3, 10)),
            _fake_range_result(2, (11, 20), (11, 20)),
        ]
        call_count = 0

        async def mock_process_range(**kwargs: Any) -> RangeResult:
            nonlocal call_count
            result = fake_results[call_count]
            call_count += 1
            return result

        with patch.object(
            readjuster, "_process_single_range", side_effect=mock_process_range
        ), patch(
            "modules.line_ranges.readjuster.ProviderConfig"
        ) as mock_provider, patch(
            "modules.line_ranges.readjuster.open_extractor",
            new_callable=lambda: _async_noop_context,
        ), patch(
            "modules.line_ranges.readjuster.resolve_context_for_readjustment",
            return_value=(None, None),
        ):
            mock_provider._detect_provider.return_value = "openai"
            mock_provider._get_api_key.return_value = "fake-key"

            await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=lr_file,
                boundary_type="TestSchema",
            )

        temp_jsonl = tmp_path / "sample_line_ranges_adjust_temp.jsonl"
        assert temp_jsonl.exists(), "Temp JSONL should have been created"

        records = list(read_jsonl_records(temp_jsonl))
        # Header record + 2 range records
        assert len(records) == 3
        assert "jsonl_header" in records[0]
        assert records[1]["custom_id"] == "sample_line_ranges-range-1"
        assert records[2]["custom_id"] == "sample_line_ranges-range-2"


class TestReadjusterResume:
    """Verify range-level resume from a partial temp JSONL."""

    @pytest.mark.asyncio
    async def test_resumes_from_partial(self, tmp_path: Path) -> None:
        """If 2 of 3 ranges are already in temp JSONL, only range 3 is processed."""
        text_file = tmp_path / "sample.txt"
        text_file.write_text(
            "\n".join(f"Line {i}" for i in range(1, 31)) + "\n",
            encoding="utf-8",
        )
        lr_file = tmp_path / "sample_line_ranges.txt"
        lr_file.write_text("(1, 10)\n(11, 20)\n(21, 30)\n", encoding="utf-8")

        # Pre-populate temp JSONL with a valid header + ranges 1 and 2
        temp_jsonl = tmp_path / "sample_line_ranges_adjust_temp.jsonl"
        fingerprint = compute_ranges_fingerprint(lr_file)
        # Header must match readjuster defaults: model="gpt-4o",
        # context_window=3, matching_config={}, retry_config={}.
        header = build_jsonl_header(
            ranges_fingerprint=fingerprint,
            total_ranges=3,
            boundary_type="TestSchema",
            model_name="gpt-4o",
            context_window=3,
            matching_config={},
            retry_config={},
        )
        existing_records = [
            header,
            _fake_range_result(1, (1, 10), (3, 10)).to_jsonl_record("sample_line_ranges"),
            _fake_range_result(2, (11, 20), (13, 20)).to_jsonl_record("sample_line_ranges"),
        ]
        temp_jsonl.write_text(
            "\n".join(json.dumps(r) for r in existing_records) + "\n",
            encoding="utf-8",
        )

        readjuster = _make_readjuster()

        # Track which range indices are processed
        processed_indices: List[int] = []

        async def mock_process_range(**kwargs: Any) -> RangeResult:
            idx = kwargs["range_index"]
            processed_indices.append(idx)
            return _fake_range_result(idx, kwargs["original_range"], (22, 30))

        with patch.object(
            readjuster, "_process_single_range", side_effect=mock_process_range
        ), patch(
            "modules.line_ranges.readjuster.ProviderConfig"
        ) as mock_provider, patch(
            "modules.line_ranges.readjuster.open_extractor",
            new_callable=lambda: _async_noop_context,
        ), patch(
            "modules.line_ranges.readjuster.resolve_context_for_readjustment",
            return_value=(None, None),
        ):
            mock_provider._detect_provider.return_value = "openai"
            mock_provider._get_api_key.return_value = "fake-key"

            await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=lr_file,
                boundary_type="TestSchema",
            )

        # Only range 3 should have been processed
        assert processed_indices == [3]

        # Temp JSONL should now have header + all 3 range records
        records = list(read_jsonl_records(temp_jsonl))
        assert len(records) == 4


class TestReadjusterForceFresh:
    """Verify that force_fresh discards stale temp JSONL."""

    @pytest.mark.asyncio
    async def test_force_fresh_discards_stale_jsonl(self, tmp_path: Path) -> None:
        """With force_fresh=True, a pre-existing temp JSONL is deleted and all ranges are reprocessed."""
        text_file = tmp_path / "sample.txt"
        text_file.write_text(
            "\n".join(f"Line {i}" for i in range(1, 21)) + "\n",
            encoding="utf-8",
        )
        lr_file = tmp_path / "sample_line_ranges.txt"
        lr_file.write_text("(1, 10)\n(11, 20)\n", encoding="utf-8")

        # Pre-populate temp JSONL with both ranges (simulating a completed prior run)
        temp_jsonl = tmp_path / "sample_line_ranges_adjust_temp.jsonl"
        stale_records = [
            _fake_range_result(1, (1, 10), (3, 10)).to_jsonl_record("sample_line_ranges"),
            _fake_range_result(2, (11, 20), (13, 20)).to_jsonl_record("sample_line_ranges"),
        ]
        temp_jsonl.write_text(
            "\n".join(json.dumps(r) for r in stale_records) + "\n",
            encoding="utf-8",
        )

        readjuster = _make_readjuster()
        processed_indices: List[int] = []

        async def mock_process_range(**kwargs: Any) -> RangeResult:
            idx = kwargs["range_index"]
            processed_indices.append(idx)
            return _fake_range_result(idx, kwargs["original_range"], kwargs["original_range"])

        with patch.object(
            readjuster, "_process_single_range", side_effect=mock_process_range
        ), patch(
            "modules.line_ranges.readjuster.ProviderConfig"
        ) as mock_provider, patch(
            "modules.line_ranges.readjuster.open_extractor",
            new_callable=lambda: _async_noop_context,
        ), patch(
            "modules.line_ranges.readjuster.resolve_context_for_readjustment",
            return_value=(None, None),
        ):
            mock_provider._detect_provider.return_value = "openai"
            mock_provider._get_api_key.return_value = "fake-key"

            await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=lr_file,
                boundary_type="TestSchema",
                force_fresh=True,
            )

        # Both ranges should have been reprocessed despite the stale temp JSONL
        assert processed_indices == [1, 2]


class TestReadjusterCleanup:
    """Verify temp JSONL removal when retain_temp_jsonl=False."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_temp(self, tmp_path: Path) -> None:
        text_file = tmp_path / "sample.txt"
        text_file.write_text("Line 1\nLine 2\n", encoding="utf-8")
        lr_file = tmp_path / "sample_line_ranges.txt"
        lr_file.write_text("(1, 2)\n", encoding="utf-8")

        readjuster = _make_readjuster()

        async def mock_process_range(**kwargs: Any) -> RangeResult:
            return _fake_range_result(1, (1, 2), (1, 2))

        with patch.object(
            readjuster, "_process_single_range", side_effect=mock_process_range
        ), patch(
            "modules.line_ranges.readjuster.ProviderConfig"
        ) as mock_provider, patch(
            "modules.line_ranges.readjuster.open_extractor",
            new_callable=lambda: _async_noop_context,
        ), patch(
            "modules.line_ranges.readjuster.resolve_context_for_readjustment",
            return_value=(None, None),
        ):
            mock_provider._detect_provider.return_value = "openai"
            mock_provider._get_api_key.return_value = "fake-key"

            await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=lr_file,
                boundary_type="TestSchema",
                retain_temp_jsonl=False,
            )

        temp_jsonl = tmp_path / "sample_line_ranges_adjust_temp.jsonl"
        assert not temp_jsonl.exists(), "Temp JSONL should have been cleaned up"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _async_noop_context:
    """Async context manager that yields a dummy extractor."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> MagicMock:
        return MagicMock()

    async def __aexit__(self, *exc: Any) -> None:
        pass
