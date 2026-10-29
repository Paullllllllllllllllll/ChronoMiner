"""Decision-model dialects: request bodies, typed answers, grouping, fingerprints.

The response shapes are those recorded from OpenAI ``/v1/decisions`` and
OpenRouter ``/api/alpha/decisions`` in October 2026.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from chronominer.common.decision_client import DecisionClient
from chronominer.common.decisions import (
    MAX_QUESTIONS,
    Choice,
    ChoiceAnswer,
    Image,
    Json,
    Level,
    MalformedAnswer,
    Predicate,
    PredicateAnswer,
    RefusalAnswer,
    Score,
    ScoreAnswer,
    Text,
    dialect_for,
    group_indices,
    parse_response,
    request_body,
    request_fingerprint,
)

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
QUESTIONS = (
    Predicate("title", "The image is a title page."),
    Choice(
        "kind",
        "What kind of page is it?",
        {"title_page": "A title page.", "other": None},
    ),
    Score("legible", "How legible is it?", (Level("no", "Unreadable."), Level("yes"))),
)
PARTS = (Json({"book": {"title": "Kochbuch"}}), Text("Image 1:"), Image(JPEG))

OPENAI_RESPONSE = {
    "id": "dec_1",
    "model": "gpt-6-luna",
    "answers": [
        {"type": "predicate", "name": "title", "probability": 0.9},
        {
            "type": "choice",
            "name": "kind",
            "choice": "title_page",
            "probabilities": [
                {"value": "title_page", "probability": 1.0},
                {"value": "other", "probability": 0.0},
            ],
            "confidence": 1.0,
        },
        {
            "type": "score",
            "name": "legible",
            "score": 0.08,
            "probabilities": [
                {"value": 0, "label": "no", "probability": 0.92},
                {"value": 1, "label": "yes", "probability": 0.08},
            ],
            "confidence": 0.92,
        },
    ],
    "usage": {
        "input_tokens": 2077,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1640},
        "output_tokens": 0,
    },
}
OPENROUTER_RESPONSE = {
    "model": "typesafe/jev-1.14",
    "answers": {
        "title": {"type": "noul", "noul": 0.59},
        "kind": {
            "type": "choice",
            "choice": "other",
            "probabilities": {"title_page": 0.01, "other": 0.99},
            "confidence": 0.99,
        },
        "legible": {
            "type": "score",
            "score": 0.99,
            "probabilities": {"0": 0.01, "1": 0.99},
            "confidence": 0.98,
        },
    },
    "usage": {"input_tokens": 25207, "output_tokens": 1231, "cost": 0.001058694},
}


def test_openai_body_lists_named_questions_and_input_parts() -> None:
    body = request_body("openai", "gpt-6-luna", PARTS, QUESTIONS)
    content = body["input"][0]["content"]
    assert [p["type"] for p in content] == ["input_json", "input_text", "input_image"]
    assert content[2]["image_url"].startswith("data:image/jpeg;base64,")
    kinds = [(q["type"], q["name"]) for q in body["questions"]]
    assert kinds == [("predicate", "title"), ("choice", "kind"), ("score", "legible")]
    assert body["questions"][1]["choices"] == [
        {"value": "title_page", "description": "A title page."},
        {"value": "other"},
    ]
    assert body["questions"][2]["levels"] == [
        {"label": "no", "description": "Unreadable."},
        {"label": "yes"},
    ]


def test_openrouter_body_maps_names_to_noul_choice_and_score() -> None:
    body = request_body("openrouter", "typesafe/jev-1.14", PARTS, QUESTIONS)
    assert [p["type"] for p in body["state"]] == ["text", "text", "image_url"]
    assert json.loads(body["state"][0]["text"]) == {"book": {"title": "Kochbuch"}}
    questions = body["questions"]
    assert questions["title"] == {
        "type": "noul",
        "instructions": "The image is a title page.",
    }
    assert questions["kind"]["criteria"] == {
        "title_page": "A title page.",
        "other": "other",
    }
    assert questions["legible"]["criteria"][1] == {"label": "yes"}


@pytest.mark.parametrize(
    ("dialect", "response"),
    [("openai", OPENAI_RESPONSE), ("openrouter", OPENROUTER_RESPONSE)],
)
def test_both_dialects_parse_into_the_same_answer_types(dialect, response) -> None:
    result = parse_response(dialect, response, QUESTIONS)
    assert isinstance(result.answers["title"], PredicateAnswer)
    kind = result.answers["kind"]
    assert isinstance(kind, ChoiceAnswer)
    assert set(kind.probabilities) == {"title_page", "other"}
    legible = result.answers["legible"]
    assert isinstance(legible, ScoreAnswer)
    assert set(legible.probabilities) == {"no", "yes"}


def test_usage_keeps_cache_writes_and_cost_apart() -> None:
    openai = parse_response("openai", OPENAI_RESPONSE, QUESTIONS).usage
    assert (openai.input_tokens, openai.cache_write_tokens, openai.cost) == (
        2077,
        1640,
        None,
    )
    router = parse_response("openrouter", OPENROUTER_RESPONSE, QUESTIONS).usage
    assert router.cost == pytest.approx(0.001058694)


def test_a_refusal_is_an_answer() -> None:
    response = {"answers": [{"type": "refusal", "name": "title"}]}
    result = parse_response("openai", response, QUESTIONS[:1])
    assert result.answers["title"] == RefusalAnswer("title")


@pytest.mark.parametrize(
    "answers",
    [
        {},
        {"title": {"type": "noul", "noul": 1.5}},
        {"title": {"type": "choice", "choice": "x"}},
    ],
)
def test_answers_that_do_not_fit_raise(answers) -> None:
    with pytest.raises(MalformedAnswer):
        parse_response("openrouter", {"answers": answers}, QUESTIONS[:1])


def test_an_unknown_choice_raises() -> None:
    answers = {"kind": {"type": "choice", "choice": "binding", "probabilities": {}}}
    with pytest.raises(MalformedAnswer, match="unknown choice"):
        parse_response("openrouter", {"answers": answers}, QUESTIONS[1:2])


@pytest.mark.parametrize("probabilities", [None, {}, {"title_page": 0.8, "other": 0.8}])
def test_a_choice_needs_a_probability_distribution(probabilities) -> None:
    entry = {"type": "choice", "choice": "other", "probabilities": probabilities}
    with pytest.raises(MalformedAnswer):
        parse_response("openrouter", {"answers": {"kind": entry}}, QUESTIONS[1:2])


def test_openai_score_labels_stay_labels_even_when_numeric() -> None:
    question = Score("n", "How many?", (Level("1"), Level("2")))
    entry = {
        "type": "score",
        "name": "n",
        "score": 0.3,
        "probabilities": [
            {"label": "1", "probability": 0.7},
            {"label": "2", "probability": 0.3},
        ],
    }
    answer = parse_response("openai", {"answers": [entry]}, (question,)).answers["n"]
    assert isinstance(answer, ScoreAnswer)
    assert dict(answer.probabilities) == {"1": 0.7, "2": 0.3}


def test_an_error_envelope_names_its_message() -> None:
    with pytest.raises(MalformedAnswer, match="bad key"):
        parse_response("openai", {"error": {"message": "bad key"}}, QUESTIONS)


def test_requests_are_validated() -> None:
    with pytest.raises(ValueError, match="at least one question"):
        request_body("openai", "m", PARTS, ())
    with pytest.raises(ValueError, match="unique"):
        request_body("openai", "m", PARTS, (QUESTIONS[0], QUESTIONS[0]))
    with pytest.raises(ValueError, match="not a JPEG"):
        request_body("openai", "m", (Image(b"not an image"),), QUESTIONS)
    with pytest.raises(ValueError, match="at least two"):
        Choice("c", "?", {"only": None})


def test_dialect_follows_the_provider_else_the_model_id() -> None:
    assert dialect_for("gpt-6-luna") == "openai"
    assert dialect_for("typesafe/jev-1.14") == "openrouter"
    assert dialect_for("typesafe/jev-1.14", "openai") == "openai"
    with pytest.raises(ValueError):
        dialect_for("m", "anthropic")


def test_groups_respect_the_page_and_question_limits() -> None:
    assert group_indices(45, per_group=20) == [
        range(0, 20),
        range(20, 40),
        range(40, 45),
    ]
    groups = group_indices(500, per_group=500, questions_per_item=2)
    assert all(len(g) * 2 <= MAX_QUESTIONS for g in groups)
    assert sum(len(g) for g in groups) == 500


def test_fingerprints_are_stable_and_sensitive() -> None:
    a = request_fingerprint(model="m", pages=20, size=768)
    assert a == request_fingerprint(size=768, pages=20, model="m")
    assert a != request_fingerprint(model="m", pages=12, size=768)


def test_the_client_posts_the_body_and_parses_the_answers() -> None:
    async def run() -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=OPENROUTER_RESPONSE)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = DecisionClient(http, dialect="openrouter", api_key="sk-test")
            result = await client.decide("typesafe/jev-1.14", PARTS, QUESTIONS)
        assert str(seen[0].url) == "https://openrouter.ai/api/alpha/decisions"
        assert seen[0].headers["authorization"] == "Bearer sk-test"
        assert json.loads(seen[0].content)["model"] == "typesafe/jev-1.14"
        assert result.model == "typesafe/jev-1.14"
        assert "sk-test" not in repr(client)

    asyncio.run(run())


def test_the_client_lets_status_errors_through_for_the_retry_ladder() -> None:
    async def run() -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(429, json={}))
        async with httpx.AsyncClient(transport=transport) as http:
            client = DecisionClient(
                http, dialect="openai", api_key="k", endpoint="https://x.test/d"
            )
            with pytest.raises(httpx.HTTPStatusError):
                await client.decide("gpt-6-luna", PARTS, QUESTIONS)

    asyncio.run(run())


def test_a_non_json_response_is_malformed() -> None:
    async def run() -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text="<html>")
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = DecisionClient(http, dialect="openai", api_key="k")
            with pytest.raises(MalformedAnswer, match="not JSON"):
                await client.decide("gpt-6-luna", PARTS, QUESTIONS)

    asyncio.run(run())


def test_an_echoed_key_is_masked_in_parse_errors() -> None:
    async def run() -> None:
        echo = {"error": {"message": "bad key sk-secret-1"}}
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=echo))
        async with httpx.AsyncClient(transport=transport) as http:
            client = DecisionClient(http, dialect="openai", api_key="sk-secret-1")
            with pytest.raises(MalformedAnswer) as caught:
                await client.decide("gpt-6-luna", PARTS, QUESTIONS)
            assert "sk-secret-1" not in str(caught.value)

    asyncio.run(run())
