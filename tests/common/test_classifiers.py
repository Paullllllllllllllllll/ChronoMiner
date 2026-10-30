"""Classifier files: validation, resolution, question conversion and hashing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.classifiers import (
    DEFAULT_FIRST_PAGES,
    Selection,
    load_classifier,
    parse_classifier,
    resolve_classifier,
)
from chronominer.common.decisions import Choice, Predicate, Score
from chronominer.common.page_types import page_types


def _raw(**extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "front",
        "version": 1,
        "unit": "page",
        "questions": [
            {"name": "kind", "type": "page_types", "profile": "culinary"},
            {"name": "title", "type": "predicate", "instructions": "A title page."},
            {
                "name": "script",
                "type": "choice",
                "instructions": "Which script?",
                "choices": {"roman": "Roman type.", "blackletter": None},
            },
            {
                "name": "legible",
                "type": "score",
                "instructions": "How legible?",
                "levels": [{"label": "no"}, {"label": "yes", "description": "Clear."}],
            },
        ],
    }
    body.update(extra)
    return body


def test_a_valid_file_parses_with_defaults() -> None:
    classifier = parse_classifier(_raw())

    assert (classifier.name, classifier.unit, classifier.per_request) == (
        "front",
        "page",
        1,
    )
    assert classifier.description == ""
    assert [q.kind for q in classifier.questions] == [
        "choice",
        "predicate",
        "choice",
        "score",
    ]
    assert classifier.selection is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"colour": "red"}, "unknown key 'colour'"),
        ({"unit": "line"}, "unit must be one of"),
        ({"version": 0}, "version must be an integer of at least 1"),
        ({"version": True}, "version must be an integer"),
        ({"name": "a/b"}, "name must be letters"),
        ({"per_request": 0}, "per_request must be an integer"),
        ({"questions": []}, "questions must be a nonempty list"),
        ({"schema": "recipes"}, "schema applies only to unit entry"),
        ({"selection": {}}, "selection applies only to unit document"),
        ({"description": 3}, "description must be a text"),
    ],
)
def test_invalid_top_level_keys_are_named(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_classifier(_raw(**change), "front.json")


@pytest.mark.parametrize(
    ("question", "message"),
    [
        ({"name": "Title", "type": "predicate"}, r"questions\[0\]\.name must be"),
        ({"name": "a__b", "type": "predicate"}, "must not contain '__'"),
        ({"name": "t", "type": "guess"}, "type must be one of"),
        ({"name": "t", "type": "predicate"}, "instructions must be a nonempty text"),
        (
            {"name": "t", "type": "predicate", "instructions": "x", "levels": []},
            "question 't': unknown key 'levels'",
        ),
        (
            {"name": "t", "type": "choice", "instructions": "x", "choices": {}},
            "choices must be a nonempty object",
        ),
        (
            {
                "name": "t",
                "type": "choice",
                "instructions": "x",
                "choices": {"a": None},
            },
            "choices needs at least two values",
        ),
        (
            {
                "name": "t",
                "type": "score",
                "instructions": "x",
                "levels": [{"label": "a"}, {"label": "a"}],
            },
            "labels must be unique",
        ),
        ({"name": "t", "type": "page_types", "profile": "legal"}, "profile must be"),
        (
            {"name": "t", "type": "page_types", "instructions": "x"},
            "unknown key 'instructions'",
        ),
    ],
)
def test_invalid_questions_are_named(question: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_classifier(_raw(questions=[question]))


def test_question_names_must_not_repeat() -> None:
    twice = [{"name": "t", "type": "predicate", "instructions": "x"}] * 2
    with pytest.raises(ValueError, match="question name 't' repeats"):
        parse_classifier(_raw(questions=twice))


def test_page_types_need_a_page_or_chunk_unit() -> None:
    with pytest.raises(ValueError, match="page_types needs unit page or chunk"):
        parse_classifier(
            _raw(unit="document", questions=[{"name": "k", "type": "page_types"}])
        )


def test_entry_and_document_units_take_their_own_keys() -> None:
    entry = parse_classifier(
        _raw(
            unit="entry",
            schema="recipes",
            fields=["title", "ingredients"],
            questions=[{"name": "sweet", "type": "predicate", "instructions": "x"}],
        )
    )
    assert (entry.schema, entry.fields) == ("recipes", ("title", "ingredients"))
    with pytest.raises(ValueError, match="unit entry needs fields"):
        parse_classifier(
            _raw(unit="entry", schema="recipes", questions=entry.as_dict()["questions"])
        )

    document = parse_classifier(
        _raw(
            unit="document",
            questions=[{"name": "cookbook", "type": "predicate", "instructions": "x"}],
        )
    )
    assert document.selection == Selection(DEFAULT_FIRST_PAGES, 20_000)
    narrowed = parse_classifier(
        _raw(
            unit="document",
            selection={"first_pages": 4},
            questions=[{"name": "cookbook", "type": "predicate", "instructions": "x"}],
        )
    )
    assert narrowed.selection == Selection(4, 20_000)
    with pytest.raises(ValueError, match="selection: unknown key 'pages'"):
        parse_classifier(
            _raw(
                unit="document",
                selection={"pages": 4},
                questions=[{"name": "c", "type": "predicate", "instructions": "x"}],
            )
        )


def test_a_single_unit_keeps_names_and_instructions() -> None:
    kind, title, script, legible = parse_classifier(_raw()).questions_for(None)

    assert isinstance(kind, Choice)
    assert kind.name == "kind"
    assert kind.instructions == "What kind of page is this page?"
    assert set(kind.choices) == {t.value for t in page_types("culinary")}
    assert title == Predicate("title", "A title page.")
    assert isinstance(script, Choice)
    assert dict(script.choices) == {"roman": "Roman type.", "blackletter": None}
    assert isinstance(legible, Score)
    assert [level.label for level in legible.levels] == ["no", "yes"]


def test_grouped_units_get_suffixed_names_and_their_subject() -> None:
    classifier = parse_classifier(_raw(per_request=5))
    kind, title, *_ = classifier.questions_for(3)

    assert classifier.grouped
    assert kind.name == "kind__03"
    assert kind.instructions == "What kind of page is page 3?"
    assert title == Predicate("title__03", "About page 3: A title page.")


def test_the_hash_covers_the_normalized_classifier() -> None:
    plain = parse_classifier(_raw())
    explicit = parse_classifier(_raw(per_request=1, description=""))
    other = parse_classifier(_raw(per_request=2))

    assert plain.sha256 == explicit.sha256
    assert plain.sha256 != other.sha256
    assert len(plain.sha256) == 64
    assert plain.as_dict()["questions"][0]["taxonomy"]


def test_reordered_choices_change_the_hash() -> None:
    raw = _raw()
    reordered = _raw()
    reordered["questions"][2]["choices"] = {
        "blackletter": None,
        "roman": "Roman type.",
    }

    assert parse_classifier(raw).sha256 != parse_classifier(reordered).sha256


def test_load_names_the_file_in_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json: not valid JSON"):
        load_classifier(path)
    path.write_text(json.dumps(_raw(unit="line")), encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.json: unit must be"):
        load_classifier(path)


def test_resolution_takes_a_path_a_stem_then_a_declared_name(tmp_path: Path) -> None:
    folder = tmp_path / "classifiers"
    folder.mkdir()
    (folder / "front.json").write_text(json.dumps(_raw()), encoding="utf-8")
    (folder / "other.json").write_text(
        json.dumps(_raw(name="leaves")), encoding="utf-8"
    )
    (folder / "broken.json").write_text("{", encoding="utf-8")
    outside = tmp_path / "mine.json"
    outside.write_text(json.dumps(_raw(name="mine")), encoding="utf-8")

    assert resolve_classifier(str(outside), folder) == outside.resolve()
    assert resolve_classifier("front", folder) == (folder / "front.json").resolve()
    assert resolve_classifier("leaves", folder) == (folder / "other.json").resolve()
    with pytest.raises(FileNotFoundError, match="Available: front, leaves"):
        resolve_classifier("missing", folder)
    with pytest.raises(FileNotFoundError, match="Available: none"):
        resolve_classifier("missing", None)
