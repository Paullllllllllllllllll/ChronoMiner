"""Page-type profiles, the no-text set, the title-page rule and the settings block."""

from __future__ import annotations

import pytest

from chronominer.common.options import Condition
from chronominer.common.page_types import (
    DEFAULT_PROFILE,
    NO_TEXT,
    PROFILES,
    no_text_probability,
    page_type_question,
    page_types,
    title_page_index,
)
from chronominer.common.settings_loader import SettingsError
from chronominer.common.settings_schema import Decisions, Reader, read_decisions
from chronominer.common.standard_options import decision_options

CULINARY_ONLY = {"recipe_text", "menu_or_bill_of_fare", "carving_or_table_service"}


def test_general_is_the_default_and_culinary_adds_three_types() -> None:
    assert DEFAULT_PROFILE == "general"
    general = {t.value for t in page_types("general")}
    culinary = {t.value for t in page_types("culinary")}
    assert len(general) == len(page_types("general")) == 18
    assert culinary - general == CULINARY_ONLY
    assert general <= culinary


@pytest.mark.parametrize("profile", PROFILES)
def test_every_profile_holds_the_no_text_types_and_a_catch_all(profile) -> None:
    values = {t.value for t in page_types(profile)}
    assert values >= NO_TEXT
    assert {"other", "title_page", "library_mark"} <= values
    assert "library_mark" not in NO_TEXT


def test_an_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown page-type profile"):
        page_types("newspaper")  # type: ignore[arg-type]


def test_the_question_offers_the_profile_types() -> None:
    question = page_type_question("page_type__01", "image 1")
    assert question.instructions == "What kind of page is image 1?"
    assert set(question.choices) == {t.value for t in page_types()}


def test_no_text_probability_sums_the_no_text_types() -> None:
    probs = {"blank": 0.5, "binding": 0.3, "scan_target": 0.15, "library_mark": 0.05}
    assert no_text_probability(probs) == pytest.approx(0.95)


def test_the_title_page_is_the_most_probable_leading_page_above_the_threshold() -> None:
    pages = [{"binding": 1.0}, {"title_page": 0.6}, {"title_page": 0.9}]
    assert title_page_index(pages) == 2
    assert title_page_index([{"title_page": 0.4}, {"blank": 1.0}]) is None
    assert title_page_index([]) is None
    late = [{"blank": 1.0}] * 10 + [{"title_page": 1.0}]
    assert title_page_index(late) is None


def test_decision_rows_apply_only_while_decisions_are_on() -> None:
    rows = {row.name: row for row in decision_options("what apply does")}
    assert list(rows) == [
        "decisions",
        "decision_model",
        "decision_provider",
        "decision_endpoint",
    ]
    assert rows["decisions"].default == "off"
    assert rows["decisions"].target == "decision.mode"
    active = Condition("decisions", frozenset({"shadow", "apply"}))
    assert all(rows[n].condition == active for n in list(rows)[1:])
    assert all(row.fallback is None for row in rows.values())


def test_the_decisions_block_reads_thresholds_by_fingerprint() -> None:
    raw = {"profile": "culinary", "pages_per_request": 12, "thresholds": {"ab12": 0.95}}
    block = read_decisions(Reader("settings"), raw, Decisions())
    assert (block.profile, block.pages_per_request) == ("culinary", 12)
    assert dict(block.thresholds) == {"ab12": 0.95}
    assert read_decisions(Reader("settings"), None, Decisions()) == Decisions()


@pytest.mark.parametrize(
    "raw",
    [
        {"profile": "newspaper"},
        {"thresholds": {"ab12": 1.5}},
        {"thresholds": {"ab12": None}},
        {"thresholds": {"ab12": 0}},
        {"unknown": 1},
    ],
)
def test_the_decisions_block_rejects_bad_values(raw) -> None:
    with pytest.raises(SettingsError):
        read_decisions(Reader("settings"), raw, Decisions())
