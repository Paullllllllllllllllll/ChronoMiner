"""Efforts above high reach Gemini 3 and the Claude models with thinking budgets.

Run on Windows with:

    uv --directory <repo> run pytest tests/common/test_request_options.py -q
"""

from __future__ import annotations

import pytest

from chronominer.common.capabilities import resolve_capabilities
from chronominer.common.capability_table import TABLE
from chronominer.common.request_options import (
    anthropic_thinking,
    google_thinking_level,
)


@pytest.mark.parametrize(
    ("model", "effort", "level"),
    [
        ("gemini-3-flash", "xhigh", "high"),
        ("gemini-3.1-flash-lite", "xhigh", "high"),
        ("gemini-3.1-flash-lite", "minimal", "minimal"),
        ("gemini-3.1-pro", "xhigh", "high"),
    ],
)
def test_gemini_3_takes_a_thinking_level_for_every_effort(
    model: str, effort: str, level: str
) -> None:
    caps = resolve_capabilities(model, TABLE)

    assert google_thinking_level(effort, caps) == level


@pytest.mark.parametrize(
    ("model", "limit", "budgets"),
    [
        ("claude-sonnet-4-5", 64000, (16384, 24576, 32768)),
        ("claude-opus-4-1", 32000, (16384, 24576, 31744)),
    ],
)
def test_claude_budgets_grow_above_high_within_the_output_limit(
    model: str, limit: int, budgets: tuple[int, int, int]
) -> None:
    caps = resolve_capabilities(model, TABLE)
    sent = tuple(
        (anthropic_thinking(effort, caps, limit) or {}).get("budget_tokens")
        for effort in ("high", "xhigh", "max")
    )

    assert sent == budgets
