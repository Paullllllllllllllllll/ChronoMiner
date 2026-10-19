"""Entries from model answers in chronominer.extract.response_parsing.

Without structured outputs a model may answer in prose, wrap its JSON in
fences, concatenate objects or bend the schema; each shape yields its entries
or none, never an exception. The helpers cover JSON isolation, fence
stripping and bounded-time salvage of pathological input.
"""

from __future__ import annotations

import json
import time

import pytest

from chronominer.extract.response_parsing import (
    _salvage_last_json_object,
    _strip_code_fences,
    _try_parse_json,
    answer_outcome,
    extract_json_text,
    normalize_llm_text,
    parse_answer,
    parse_extraction,
)


def _answer(*entries: dict[str, object], no_content: bool = False) -> str:
    return json.dumps(
        {"contains_no_content_of_requested_type": no_content, "entries": list(entries)}
    )


class TestExtractJsonText:
    """The JSON isolation the converters always used."""

    @pytest.mark.unit
    def test_pure_json_passes_through_stripped(self) -> None:
        assert extract_json_text('  \n {"entries": []}  \n') == '{"entries": []}'

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text",
        [
            'Here is the data:\n```json\n{"entries": [{"id": 1}]}\n```\n',
            '```\n{"entries": [{"id": 1}]}\n```',
            'Here is the extracted data: {"entries": [{"id": 1}]}',
        ],
        ids=["fence_with_preamble", "bare_fence", "preamble_text"],
    )
    def test_json_is_isolated_from_its_wrapping(self, text: str) -> None:
        result = extract_json_text(text)
        assert result is not None
        assert json.loads(result)["entries"] == [{"id": 1}]

    @pytest.mark.unit
    def test_braces_inside_strings(self) -> None:
        obj = {"entries": [{"text": 'a {nested} value "with quotes"'}]}
        result = extract_json_text(f"Result: {json.dumps(obj)} and more")
        assert result is not None
        assert json.loads(result) == obj

    @pytest.mark.unit
    @pytest.mark.parametrize("text", ["No JSON here at all", "", "   "])
    def test_no_json(self, text: str) -> None:
        assert extract_json_text(text) is None


class TestParseExtraction:
    """The entries of an answer in every shape a model sends."""

    @pytest.mark.unit
    def test_pure_json(self) -> None:
        assert parse_extraction(_answer({"id": 1}, {"id": 2})) == [
            {"id": 1},
            {"id": 2},
        ]

    @pytest.mark.unit
    def test_no_content_flag_has_no_entries(self) -> None:
        assert parse_extraction(_answer({"id": 1}, no_content=True)) == []

    @pytest.mark.unit
    def test_no_content_flag_in_a_fence_with_prose(self) -> None:
        text = f"The page is blank.\n```json\n{_answer(no_content=True)}\n```"
        assert parse_extraction(text) == []
        assert answer_outcome(text) == "no_content"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "not json",
            '{"entries": [{"id": 1}, ',
            '{"other": []}',
            '{"entries": null}',
            '[{"id": 1}]',
        ],
    )
    def test_answers_without_entries(self, text: str) -> None:
        assert parse_extraction(text) == []

    @pytest.mark.unit
    def test_entries_that_are_not_objects_are_dropped(self) -> None:
        text = json.dumps({"entries": [{"id": 1}, None, "loose", 3, {"id": 2}]})
        assert parse_extraction(text) == [{"id": 1}, {"id": 2}]

    @pytest.mark.unit
    def test_salvages_object_after_invalid_fenced_block(self) -> None:
        text = f"```json\n{{not json}}\n```\n{_answer({'id': 'kept'})}"
        assert parse_extraction(text) == [{"id": "kept"}]

    @pytest.mark.unit
    def test_concatenated_objects_keep_the_last_complete_one(self) -> None:
        text = _answer({"id": "first"}) + _answer({"id": "second"})
        assert parse_extraction(text) == [{"id": "second"}]


class TestAnswerOutcome:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("text", "outcome"),
        [
            (_answer({"id": 1}), "ok"),
            (_answer(), "ok"),
            (_answer(no_content=True), "no_content"),
            ("Sorry, I cannot read this page.", "unparsed"),
            ('{"entries": [', "unparsed"),
        ],
    )
    def test_outcome(self, text: str, outcome: str) -> None:
        assert answer_outcome(text) == outcome

    @pytest.mark.unit
    def test_parse_answer_returns_the_object(self) -> None:
        assert parse_answer(f"Result:\n{_answer({'id': 1})}") == {
            "contains_no_content_of_requested_type": False,
            "entries": [{"id": 1}],
        }
        assert parse_answer(None) is None


class TestSalvageLastJsonObject:
    """_salvage_last_json_object and _try_parse_json on pathological input.

    Many unbalanced or deeply nested braces degrade to None in bounded time
    instead of backtracking quadratically or raising RecursionError.
    """

    @pytest.mark.unit
    def test_salvages_last_of_concatenated_objects(self) -> None:
        text = '{"entries": "first"}{"entries": "second"}'
        obj = _salvage_last_json_object(text)
        assert obj is not None
        assert obj["entries"] == "second"

    @pytest.mark.unit
    def test_salvages_trailing_object_after_prose(self) -> None:
        text = 'garbage {oops not json} tail {"entries": "ok"}'
        obj = _salvage_last_json_object(text)
        assert obj is not None
        assert obj["entries"] == "ok"

    @pytest.mark.unit
    @pytest.mark.parametrize("text", ["", "{" * 5000])
    def test_nothing_to_salvage_returns_none(self, text: str) -> None:
        assert _salvage_last_json_object(text) is None

    @pytest.mark.unit
    def test_try_parse_json_deeply_nested_returns_none(self) -> None:
        """A deeply nested valid-JSON string must not raise RecursionError."""
        depth = 6000
        deep = '{"a":' * depth + "1" + "}" * depth
        result = _try_parse_json(deep)
        assert result is None or isinstance(result, dict)

    @pytest.mark.unit
    def test_thousands_of_unbalanced_braces_degrade_fast(self) -> None:
        text = '{"a":' * 6000 + "}"
        start = time.perf_counter()
        result = _salvage_last_json_object(text)
        elapsed = time.perf_counter() - start
        assert result is None
        assert elapsed < 5.0

    @pytest.mark.unit
    def test_pathological_answer_has_no_entries(self) -> None:
        assert parse_extraction("{" * 6000) == []


class TestStripCodeFences:
    """Tests for _strip_code_fences helper."""

    @pytest.mark.unit
    @pytest.mark.parametrize("opening", ["```json", "```"])
    def test_leading_code_fence_stripped(self, opening: str) -> None:
        text = opening + '\n{"entries": []}\n```'
        assert _strip_code_fences(text) == '{"entries": []}'

    @pytest.mark.unit
    def test_no_fence_returns_unchanged(self) -> None:
        text = '{"entries": []}'
        assert _strip_code_fences(text) == text

    @pytest.mark.unit
    def test_multiple_fences_returns_last(self) -> None:
        text = '```\nthinking...\n```\n```json\n{"entries": ["real"]}\n```'
        assert _strip_code_fences(text) == '{"entries": ["real"]}'

    @pytest.mark.unit
    def test_fence_with_leading_prose_passes_through(self) -> None:
        text = 'Here is the result:\n```json\n{"entries": []}\n```\nDone!'
        assert _strip_code_fences(text) == text

    @pytest.mark.unit
    def test_fully_wrapped_json_with_inner_fence_keeps_inner(self) -> None:
        text = (
            "```json\n"
            '{"entries": [{"notes": "Here is code:\n'
            "```python\n"
            "x = 1\n"
            "```\n"
            'done"}]}\n'
            "```"
        )
        result = _strip_code_fences(text)
        assert result.startswith("{")
        assert result.endswith("}")
        assert "```python" in result
        assert "x = 1" in result


class TestNormalizeLlmText:
    """Tests for normalize_llm_text helper."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text",
        [
            '{"entries": ["a"]}',
            '```json\n{"entries": ["a"]}\n```',
            'Here is the result:\n```json\n{"entries": ["a"]}\n```\nDone!',
        ],
        ids=["clean", "fenced", "prose_wrapped_fence"],
    )
    def test_json_is_normalized_to_its_object(self, text: str) -> None:
        assert json.loads(normalize_llm_text(text))["entries"] == ["a"]

    @pytest.mark.unit
    def test_preamble_postamble(self) -> None:
        text = 'Here is the data:\n{"entries": ["a"]}\nI hope this helps!'
        result = normalize_llm_text(text)
        assert result.startswith("{")
        assert json.loads(result)["entries"] == ["a"]

    @pytest.mark.unit
    def test_plain_text_unchanged(self) -> None:
        text = "The text on this page reads: Lorem ipsum"
        assert normalize_llm_text(text) == text

    @pytest.mark.unit
    @pytest.mark.parametrize("text", ["", "   \n\t  "])
    def test_blank_text_normalizes_to_empty(self, text: str) -> None:
        assert normalize_llm_text(text) == ""
