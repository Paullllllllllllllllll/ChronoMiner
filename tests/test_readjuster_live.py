"""Live API tests for the line-range readjuster.

Opt-in only: these tests make real OpenAI calls. They run when BOTH
``OPENAI_API_KEY`` is set and ``CHRONOMINER_LIVE_TESTS=1``; otherwise the
whole module is skipped, so the offline suite stays offline.

    CHRONOMINER_LIVE_TESTS=1 uv run pytest tests/test_readjuster_live.py -m live -s

Model: gpt-5.4-mini (free daily small-model allowance) with low reasoning
effort and low text verbosity; ``service_tier`` is forced to ``default``
(non-flex) regardless of the local concurrency config. Expected cost is a
few dozen small calls per test.
"""

import functools
import os
from pathlib import Path

import pytest

from modules.infra.jsonl import read_jsonl_header
from modules.infra.token_tracker import get_token_tracker
from modules.line_ranges.readjuster import LineRangeReadjuster
from modules.llm.langchain_provider import LLMProvider
from modules.llm.openai_utils import open_extractor as _real_open_extractor

pytestmark = [
    pytest.mark.live,
    # One event loop for the whole module: langchain-openai shares a cached
    # async httpx client process-wide, which binds to the first event loop it
    # is used on; per-test loops would fail with 'Event loop is closed'.
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    ),
    pytest.mark.skipif(
        os.environ.get("CHRONOMINER_LIVE_TESTS") != "1",
        reason="live tests are opt-in: set CHRONOMINER_LIVE_TESTS=1",
    ),
]

_MODEL_CONFIG = {
    "extraction_model": {
        "name": "gpt-5.4-mini",
        "max_output_tokens": 4096,
        "reasoning": {"effort": "low"},
        "text": {"verbosity": "low"},
    }
}

# Non-flex processing regardless of the local concurrency_config.yaml.
_CONCURRENCY_OVERRIDE = {
    "concurrency": {"extraction": {"service_tier": "default"}}
}

_RETRY_CONFIG = {
    "certainty_threshold": 70,
    "max_low_certainty_retries": 3,
    "max_marker_mismatch_retries": 2,
    "max_context_expansion_attempts": 3,
    "delete_ranges_with_no_content": True,
    "max_gap_between_ranges": 2,
}

_ADJUST_CONTEXT = (
    "The source is a 19th-century German cookbook. Each recipe is one"
    " semantic unit. A recipe BEGINS with a numbered title line of the form"
    " '3. Gebratene Forelle.' — a number, a period, the dish name, and a"
    " final period. Chunk boundaries must sit exactly on such title lines."
    " Page decorations, page numbers, and asterisk rules are not recipe"
    " content."
)

_RECIPE_NAMES = [
    "Ochsenschwanzsuppe",
    "Gebratene Forelle",
    "Gefuellte Kalbsbrust",
    "Erbsenpuree mit Speck",
    "Boehmische Knoedel",
    "Geduenstetes Rindfleisch",
    "Karpfen in schwarzer Sosse",
    "Apfelstrudel nach Wiener Art",
    "Gebackene Quitten",
    "Punschtorte",
]

_FILLER_BLOCK = [
    "*   *   *",
    "— 47 —",
    "Verlag der Buchhandlung J. G. Ritter, Wien.",
    "*   *   *",
    "— 48 —",
    "*   *   *",
]


def _recipe_body(name: str) -> list[str]:
    return [
        f"Man nehme zu diesem Gerichte ({name}) zwei Pfund gute Zuthaten.",
        "Hierauf schneide man Zwiebeln und Wurzelwerk recht klein.",
        "Alles zusammen in einem irdenen Topfe mit Wasser zustellen.",
        "Mit Salz und einem Loeffel Pfeffer nach Belieben wuerzen.",
        "Eine gute Stunde auf gelindem Feuer langsam kochen lassen.",
        "Zuletzt heiss anrichten und sogleich zu Tische geben.",
    ]


def _build_cookbook() -> tuple[list[str], list[int], tuple[int, int]]:
    """Return (lines, recipe header line numbers, filler span), 1-indexed."""
    lines: list[str] = []
    header_lines: list[int] = []
    filler_span = (0, 0)
    for i, name in enumerate(_RECIPE_NAMES, 1):
        if i == 6:
            filler_start = len(lines) + 1
            lines.extend(_FILLER_BLOCK)
            filler_span = (filler_start, len(lines))
        header_lines.append(len(lines) + 1)
        lines.append(f"{i}. {name}.")
        lines.extend(_recipe_body(name))
    return lines, header_lines, filler_span


def _write_env(tmp_path: Path, ranges: list[tuple[int, int]]) -> tuple[Path, Path]:
    lines, _, _ = _build_cookbook()
    text_file = tmp_path / "kochbuch.txt"
    text_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "kochbuch_adjust_context.txt").write_text(
        _ADJUST_CONTEXT, encoding="utf-8"
    )
    lr_file = tmp_path / "kochbuch_line_ranges.txt"
    lr_file.write_text(
        "".join(f"({s}, {e})\n" for s, e in ranges), encoding="utf-8"
    )
    return text_file, lr_file


def _make_live_readjuster() -> LineRangeReadjuster:
    return LineRangeReadjuster(
        _MODEL_CONFIG,
        context_window=6,
        retry_config=_RETRY_CONFIG,
    )


@pytest.fixture(autouse=True)
def _fresh_llm_client_cache():
    """Drop cached LLMProvider instances between tests (defense-in-depth
    against cross-test client reuse; the module-scoped event loop above is
    the primary safeguard)."""
    LLMProvider.clear_cache()
    yield
    LLMProvider.clear_cache()


@pytest.fixture(autouse=True)
def _force_default_service_tier(monkeypatch: pytest.MonkeyPatch):
    """Route the readjuster's extractor through a non-flex service tier."""
    wrapped = functools.partial(
        _real_open_extractor, concurrency_config_override=_CONCURRENCY_OVERRIDE
    )
    monkeypatch.setattr(
        "modules.line_ranges.readjuster.open_extractor", wrapped
    )


@pytest.fixture(autouse=True)
def _report_token_usage():
    tracker = get_token_tracker()
    before = tracker.get_stats().get("tokens_used_today", 0)
    yield
    after = tracker.get_stats().get("tokens_used_today", 0)
    print(f"\n[live] tokens recorded by tracker this test: {after - before:,}")


def _assert_sane(
    adjusted: list[tuple[int, int]], total_lines: int
) -> None:
    """Hard structural invariants that must hold regardless of model output."""
    for start, end in adjusted:
        assert 1 <= start <= end <= total_lines, f"bad range ({start}, {end})"
    for (_, prev_end), (next_start, _) in zip(
        adjusted, adjusted[1:], strict=False
    ):
        assert prev_end < next_start, "adjusted ranges overlap"


async def test_live_boundary_alignment(tmp_path: Path) -> None:
    """Misaligned mid-recipe boundaries are pulled onto recipe title lines."""
    lines, header_lines, _ = _build_cookbook()

    # Every range starts 3 lines INTO a recipe; the correct boundary (the
    # recipe title line) is 3 lines above each start.
    starts = [1] + [h + 3 for h in header_lines[1:]]
    ranges = [
        (s, next_s - 1) for s, next_s in zip(starts, starts[1:], strict=False)
    ] + [(starts[-1], len(lines))]

    text_file, lr_file = _write_env(tmp_path, ranges)
    readjuster = _make_live_readjuster()

    adjusted = await readjuster.ensure_adjusted_line_ranges(
        text_file=text_file,
        line_ranges_file=lr_file,
        boundary_type="HistoricalRecipes",
    )

    _assert_sane(adjusted, len(lines))

    # Completed run: header finalized with the post-write fingerprint.
    header = read_jsonl_header(tmp_path / "kochbuch_line_ranges_adjust_temp.jsonl")
    assert header is not None
    assert header.get("completed_at") is not None
    assert header.get("final_ranges_fingerprint") is not None

    # Soft quality gate: at least half of the boundaries land exactly on a
    # recipe title line (the first range legitimately stays at line 1).
    header_set = set(header_lines)
    scored = [start for start, _ in adjusted[1:]]
    hits = sum(1 for start in scored if start in header_set)
    print(
        f"[live] boundary alignment: {hits}/{len(scored)} adjusted starts on"
        f" recipe title lines; adjusted={adjusted}"
    )
    assert hits >= len(scored) / 2, (
        f"only {hits}/{len(scored)} boundaries aligned: {adjusted}"
    )


async def test_live_deletion_safety(tmp_path: Path) -> None:
    """A pure-filler range may be deleted, but no recipe range may be."""
    lines, header_lines, filler_span = _build_cookbook()

    # Ranges aligned with recipes, plus one range covering only the filler
    # block (page decorations between recipes 5 and 6).
    ranges: list[tuple[int, int]] = []
    for i, h in enumerate(header_lines):
        end = (
            header_lines[i + 1] - 1
            if i + 1 < len(header_lines)
            else len(lines)
        )
        if filler_span[0] > 0 and h < filler_span[0] <= end:
            ranges.append((h, filler_span[0] - 1))
            ranges.append(filler_span)
        else:
            ranges.append((h, end))

    text_file, lr_file = _write_env(tmp_path, ranges)
    readjuster = _make_live_readjuster()

    adjusted = await readjuster.ensure_adjusted_line_ranges(
        text_file=text_file,
        line_ranges_file=lr_file,
        boundary_type="HistoricalRecipes",
    )

    _assert_sane(adjusted, len(lines))

    # Hard safety property: every recipe title line is still covered by the
    # union of the adjusted ranges — deleting a content-bearing range would
    # break this.
    uncovered = [
        h
        for h in header_lines
        if not any(start <= h <= end for start, end in adjusted)
    ]
    assert not uncovered, f"recipe title lines lost from coverage: {uncovered}"

    filler_deleted = not any(
        start <= filler_span[0] <= end for start, end in adjusted
    )
    print(
        f"[live] deletion safety: filler range deleted={filler_deleted}"
        f" (allowed either way); adjusted={adjusted}"
    )
