"""Readjuster per-key budget: stamped reservations and mid-wait key switch.

The range-level gate reserved without a key stamp, so only the combined
daily cap could block a readjustment run: a key whose own pool was spent
kept being billed. Reservations now carry (provider, key_env, model), and
when a budget wait ends on a different active key the extractor is rebuilt
on that key instead of continuing on the exhausted one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from modules.infra.jsonl import read_jsonl_records
from modules.infra.shared_ledger import BucketKey
from modules.infra.token_tracker import DailyTokenTracker
from modules.line_ranges.readjuster import (
    BoundaryDecision,
    LineRangeReadjuster,
    RangeResult,
)

pytestmark = pytest.mark.unit

_MODEL = "gpt-4o"


class _RecordingExtractorContext:
    """Async context manager standing in for ``open_extractor``.

    Records the ``api_key`` of every instantiation so a test can assert
    that a key switch rebuilt the extractor on the new key.
    """

    opened: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).opened.append(str(kwargs.get("api_key")))

    async def __aenter__(self) -> MagicMock:
        return MagicMock()

    async def __aexit__(self, *exc: Any) -> None:
        pass


def _make_readjuster() -> LineRangeReadjuster:
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
            {"extraction_model": {"name": _MODEL}}, context_window=3
        )


def _fake_range_result(index: int, rng: tuple[int, int]) -> RangeResult:
    return RangeResult(
        range_index=index,
        original_range=rng,
        adjusted_range=rng,
        should_delete=False,
        decision=BoundaryDecision(
            contains_no_semantic_boundary=False,
            needs_more_context=False,
            certainty=95,
            boundary_already_on_target=True,
            semantic_marker=None,
        ),
        attempts=[],
        total_llm_calls=1,
    )


@pytest.mark.asyncio
async def test_per_key_block_switches_key_and_rebuilds_extractor(
    tmp_path: Path,
) -> None:
    text_file = tmp_path / "s.txt"
    text_file.write_text(
        "\n".join(f"Line {i}" for i in range(1, 31)) + "\n", encoding="utf-8"
    )
    lr_file = tmp_path / "s_line_ranges.txt"
    lr_file.write_text("(1, 10)\n(11, 20)\n(21, 30)\n", encoding="utf-8")

    # Per-key pool cap of 100 tokens on the "small" pool; the combined cap is
    # far away, so only the per-key gate can block. Two 60-token ranges spend
    # the first key's pool; the third range must move to the second key.
    tracker = DailyTokenTracker(
        daily_limit=10_000_000,
        enabled=True,
        state_file=tmp_path / "tok.json",
        chunk_estimate_seed=10,
        estimate_smoothing=0.3,
        pool_caps={("openai", "small"): 100},
        provider_pools={"openai": {"small": (_MODEL,)}},
    )
    active = {"env": "OPENAI_API_KEY"}
    processed: list[int] = []
    waits: list[dict[str, Any]] = []
    _RecordingExtractorContext.opened = []

    async def mock_process_range(**kwargs: Any) -> RangeResult:
        processed.append(kwargs["range_index"])
        tracker.add_tokens(60, provider="openai", key_env=active["env"], model=_MODEL)
        return _fake_range_result(kwargs["range_index"], kwargs["original_range"])

    async def fake_wait(**kwargs: Any) -> bool:
        # Stand-in for the auto key switch: the wait ends on a fresh key.
        waits.append(kwargs)
        active["env"] = "OPENAI_API_KEY_2"
        tracker.rebind_active_key_env("openai", active["env"])
        return True

    readjuster = _make_readjuster()
    with (
        patch.object(
            readjuster, "_process_single_range", side_effect=mock_process_range
        ),
        patch("modules.line_ranges.readjuster.ProviderConfig") as mock_provider,
        patch(
            "modules.line_ranges.readjuster.open_extractor",
            new_callable=lambda: _RecordingExtractorContext,
        ),
        patch(
            "modules.line_ranges.readjuster.resolve_context_for_readjustment",
            return_value=(None, None),
        ),
        patch("modules.line_ranges.readjuster.get_token_tracker", return_value=tracker),
        patch(
            "modules.line_ranges.readjuster.check_and_wait_for_token_limit",
            side_effect=fake_wait,
        ),
    ):
        mock_provider._detect_provider.return_value = "openai"
        mock_provider._get_api_key.side_effect = lambda _p: "secret-" + active["env"]
        mock_provider.resolve_key_env_var.side_effect = lambda _p: active["env"]
        await readjuster.ensure_adjusted_line_ranges(
            text_file=text_file,
            line_ranges_file=lr_file,
            boundary_type="TestSchema",
        )

    # The per-key gate blocked range 3 on the first key, the wait was
    # reservation-aware, and the run finished on the second key.
    assert processed == [1, 2, 3]
    assert len(waits) == 1
    assert waits[0].get("reservation_aware") is True
    # The extractor was rebuilt on the new key rather than reused.
    assert _RecordingExtractorContext.opened == [
        "secret-OPENAI_API_KEY",
        "secret-OPENAI_API_KEY_2",
    ]
    # Usage landed per key, so the per-key caps saw it.
    used = tracker._bucket_used_today
    assert used[BucketKey("openai", "OPENAI_API_KEY", "small")] == 120
    assert used[BucketKey("openai", "OPENAI_API_KEY_2", "small")] == 60
    # No reservation leaked on either bucket.
    assert tracker._tokens_reserved == {}
    records = list(read_jsonl_records(tmp_path / "s_line_ranges_adjust_temp.jsonl"))
    assert len(records) == 4  # header + 3 ranges
