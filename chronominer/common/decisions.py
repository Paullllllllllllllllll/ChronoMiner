"""Decision models: typed questions, two request dialects, typed answers.

A decision model answers questions about text and images with probabilities
instead of generated text. Three question types exist: a :class:`Predicate`
(the probability that a statement holds), a :class:`Choice` (one of fixed
values, with the probability of each) and a :class:`Score` (an expected
level on an ordered scale). The type names follow
``langchain_openai.decisions``.

Two dialects build the request body and read the answers:

- ``openai``: ``POST /v1/decisions`` (``input`` messages with
  ``input_text``, ``input_json`` and ``input_image`` parts; ``questions`` a
  list of named questions; answers a list).
- ``openrouter``: ``POST /api/alpha/decisions`` (``state`` parts of types
  ``text`` and ``image_url``; ``questions`` a mapping of names; predicates are
  ``noul`` questions and answers; choices and score levels are
  ``criteria``).

The module does no transport: a caller sends :func:`request_body` with its
own client and passes the decoded response to :func:`parse_response`. It
imports only the standard library, so ChronoDownloader vendors it unchanged.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal

__all__ = [
    "DIALECTS",
    "ENDPOINTS",
    "MAX_QUESTIONS",
    "MODES",
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "DecisionResult",
    "DecisionUsage",
    "Dialect",
    "Image",
    "Json",
    "Level",
    "MalformedAnswer",
    "Mode",
    "Part",
    "Predicate",
    "PredicateAnswer",
    "Question",
    "RefusalAnswer",
    "Score",
    "ScoreAnswer",
    "Text",
    "dialect_for",
    "group_indices",
    "parse_response",
    "request_body",
    "request_fingerprint",
]

type Dialect = Literal["openai", "openrouter"]
type Mode = Literal["off", "shadow", "apply"]

DIALECTS: Final[tuple[Dialect, ...]] = ("openai", "openrouter")
MODES: Final[tuple[Mode, ...]] = ("off", "shadow", "apply")
"""``off`` asks nothing; ``shadow`` records answers only; ``apply`` acts on them."""

ENDPOINTS: Final[Mapping[Dialect, str]] = MappingProxyType(
    {
        "openai": "https://api.openai.com/v1/decisions",
        "openrouter": "https://openrouter.ai/api/alpha/decisions",
    }
)

MAX_QUESTIONS: Final = 400
"""The most questions OpenAI accepts in one request."""

_IMAGE_TYPES: Final = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


class MalformedAnswer(ValueError):
    """A response does not answer the questions asked.

    ``usage`` holds the token counts the response reported, if it was a
    decoded response: a billed response counts although it is discarded.
    """

    def __init__(self, message: str, usage: DecisionUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class Predicate:
    """The probability that the statement in ``instructions`` holds."""

    name: str
    instructions: str


@dataclass(frozen=True)
class Choice:
    """One of ``choices``, a mapping of value to description (None: none)."""

    name: str
    instructions: str
    choices: Mapping[str, str | None]

    def __post_init__(self) -> None:
        if len(self.choices) < 2:
            raise ValueError(f"choice {self.name!r} needs at least two values")
        object.__setattr__(self, "choices", MappingProxyType(dict(self.choices)))


@dataclass(frozen=True)
class Level:
    """One level of a :class:`Score`, lowest first."""

    label: str
    description: str | None = None


@dataclass(frozen=True)
class Score:
    """An expected level on ``levels``, ordered lowest to highest."""

    name: str
    instructions: str
    levels: tuple[Level, ...]

    def __post_init__(self) -> None:
        if not 2 <= len(self.levels) <= 10:
            raise ValueError(f"score {self.name!r} needs 2 to 10 levels")


type Question = Predicate | Choice | Score


@dataclass(frozen=True)
class Text:
    """A text part of the shared input."""

    text: str


@dataclass(frozen=True)
class Json:
    """A JSON part of the shared input (sent as text under ``openrouter``)."""

    data: Any


@dataclass(frozen=True)
class Image:
    """An image part: JPEG, PNG, GIF or WebP bytes."""

    data: bytes = field(repr=False)

    @property
    def media_type(self) -> str:
        """Return the media type sniffed from the leading bytes.

        Raises:
            ValueError: The bytes are no JPEG, PNG, GIF or WebP image.
        """
        for magic, media_type in _IMAGE_TYPES:
            if self.data.startswith(magic):
                return media_type
        if self.data[:4] == b"RIFF" and self.data[8:12] == b"WEBP":
            return "image/webp"
        raise ValueError("image part is not a JPEG, PNG, GIF or WebP image")

    def data_url(self) -> str:
        """Return the image as a base64 data URL."""
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.media_type};base64,{encoded}"


type Part = Text | Json | Image


@dataclass(frozen=True)
class PredicateAnswer:
    """The probability of a :class:`Predicate`."""

    name: str
    probability: float


@dataclass(frozen=True)
class ChoiceAnswer:
    """The chosen value of a :class:`Choice` with every value's probability."""

    name: str
    choice: str
    probabilities: Mapping[str, float]
    confidence: float | None = None


@dataclass(frozen=True)
class ScoreAnswer:
    """The expected level index of a :class:`Score`, probabilities by label."""

    name: str
    score: float
    probabilities: Mapping[str, float]
    confidence: float | None = None


@dataclass(frozen=True)
class RefusalAnswer:
    """The model declined the question."""

    name: str
    reason: str | None = None


type Answer = PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer


@dataclass(frozen=True)
class DecisionUsage:
    """Token counts of one request; zero where the dialect reports none.

    ``input_tokens`` includes ``cached_tokens`` (read from the cache) and
    ``cache_write_tokens`` (written to it); ``cost`` is the billed amount in
    USD where the endpoint reports one (OpenRouter).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float | None = None


@dataclass(frozen=True)
class DecisionResult:
    """The answers of one request by question name, with model and usage."""

    answers: Mapping[str, Answer]
    model: str
    usage: DecisionUsage
    request_id: str | None = None


def dialect_for(model: str, provider: str | None = None) -> Dialect:
    """Return the dialect of *provider*, else of *model* (``vendor/model``: OpenRouter).

    Raises:
        ValueError: *provider* is no dialect.
    """
    if provider is not None:
        if provider not in DIALECTS:
            raise ValueError(f"unknown decision provider {provider!r}")
        return provider
    return "openrouter" if "/" in model else "openai"


def _check_questions(questions: Sequence[Question]) -> None:
    names = [q.name for q in questions]
    if not names:
        raise ValueError("a decision request needs at least one question")
    if len(names) > MAX_QUESTIONS:
        raise ValueError(f"{len(names)} questions exceed the limit of {MAX_QUESTIONS}")
    if len(set(names)) != len(names):
        raise ValueError("question names must be unique")


def _openai_question(question: Question) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": question.name,
        "instructions": question.instructions,
    }
    if isinstance(question, Predicate):
        body["type"] = "predicate"
    elif isinstance(question, Choice):
        body["type"] = "choice"
        body["choices"] = [
            {"value": value} if text is None else {"value": value, "description": text}
            for value, text in question.choices.items()
        ]
    else:
        body["type"] = "score"
        body["levels"] = [_level(level) for level in question.levels]
    return body


def _openrouter_question(question: Question) -> dict[str, Any]:
    body: dict[str, Any] = {"instructions": question.instructions}
    if isinstance(question, Predicate):
        body["type"] = "noul"
    elif isinstance(question, Choice):
        body["type"] = "choice"
        body["criteria"] = {
            value: text or value for value, text in question.choices.items()
        }
    else:
        body["type"] = "score"
        body["criteria"] = [_level(level) for level in question.levels]
    return body


def _level(level: Level) -> dict[str, str]:
    if level.description is None:
        return {"label": level.label}
    return {"label": level.label, "description": level.description}


def _openai_part(part: Part) -> dict[str, Any]:
    if isinstance(part, Text):
        return {"type": "input_text", "text": part.text}
    if isinstance(part, Json):
        return {"type": "input_json", "data": part.data}
    return {"type": "input_image", "image_url": part.data_url()}


def _openrouter_part(part: Part) -> dict[str, Any]:
    if isinstance(part, Text):
        return {"type": "text", "text": part.text}
    if isinstance(part, Json):
        return {"type": "text", "text": json.dumps(part.data, ensure_ascii=False)}
    return {"type": "image_url", "image_url": {"url": part.data_url()}}


def request_body(
    dialect: Dialect, model: str, parts: Sequence[Part], questions: Sequence[Question]
) -> dict[str, Any]:
    """Return the request body asking *questions* about *parts*.

    Raises:
        ValueError: No question, too many, duplicate names, no part, or an
            image of an unknown type.
    """
    _check_questions(questions)
    if not parts:
        raise ValueError("a decision request needs at least one input part")
    if dialect == "openai":
        content = [_openai_part(part) for part in parts]
        return {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "questions": [_openai_question(q) for q in questions],
        }
    return {
        "model": model,
        "state": [_openrouter_part(part) for part in parts],
        "questions": {q.name: _openrouter_question(q) for q in questions},
    }


def _probability(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise MalformedAnswer(f"{where}: expected a probability, got {value!r}")
    return float(value)


def _confidence(entry: Mapping[str, Any], where: str) -> float | None:
    value = entry.get("confidence")
    return None if value is None else _probability(value, f"{where}.confidence")


def _probabilities(raw: Any, where: str) -> dict[str, float]:
    """Read probabilities as a mapping, or a list of ``{value, probability}``."""
    if isinstance(raw, Mapping):
        items = [(str(key), value) for key, value in raw.items()]
    elif isinstance(raw, list):
        items = []
        for item in raw:
            if not isinstance(item, Mapping) or "probability" not in item:
                raise MalformedAnswer(f"{where}: malformed probability entry")
            key = item.get("label", item.get("value"))
            items.append((str(key), item["probability"]))
    else:
        raise MalformedAnswer(f"{where}: expected probabilities")
    probabilities = {key: _probability(value, f"{where}.{key}") for key, value in items}
    if sum(probabilities.values()) > 1.0 + _SUM_TOLERANCE:
        raise MalformedAnswer(f"{where}: probabilities sum to more than 1")
    return probabilities


_SUM_TOLERANCE: Final = 0.01


def _answer(dialect: Dialect, question: Question, entry: Any) -> Answer:
    where = question.name
    if not isinstance(entry, Mapping):
        raise MalformedAnswer(f"{where}: no answer")
    kind = entry.get("type")
    if kind == "refusal":
        reason = entry.get("reason", entry.get("refusal"))
        return RefusalAnswer(where, reason if isinstance(reason, str) else None)
    if isinstance(question, Predicate):
        if kind not in ("predicate", "noul"):
            raise MalformedAnswer(f"{where}: expected a predicate answer, got {kind!r}")
        value = entry.get("probability", entry.get("noul"))
        return PredicateAnswer(where, _probability(value, where))
    if isinstance(question, Choice):
        if kind != "choice":
            raise MalformedAnswer(f"{where}: expected a choice answer, got {kind!r}")
        choice = entry.get("choice")
        if choice not in question.choices:
            raise MalformedAnswer(f"{where}: unknown choice {choice!r}")
        probabilities = _probabilities(entry.get("probabilities"), where)
        if not probabilities:
            raise MalformedAnswer(f"{where}: no probabilities")
        if not set(probabilities) <= set(question.choices):
            raise MalformedAnswer(f"{where}: probabilities name unknown values")
        return ChoiceAnswer(
            where,
            str(choice),
            MappingProxyType(probabilities),
            _confidence(entry, where),
        )
    if kind != "score":
        raise MalformedAnswer(f"{where}: expected a score answer, got {kind!r}")
    return _score_answer(dialect, question, entry)


def _score_answer(
    dialect: Dialect, question: Score, entry: Mapping[str, Any]
) -> ScoreAnswer:
    where = question.name
    score = entry.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, int | float)
        or not math.isfinite(score)
        or not 0 <= score <= len(question.levels) - 1
    ):
        raise MalformedAnswer(
            f"{where}: expected a score from 0 to {len(question.levels) - 1}, "
            f"got {score!r}"
        )
    raw = _probabilities(entry.get("probabilities", {}), where)
    labels = [level.label for level in question.levels]
    if dialect == "openrouter":  # keyed by level index; OpenAI keys by label
        by_label = {
            labels[int(key)] if key.isdigit() and int(key) < len(labels) else key: v
            for key, v in raw.items()
        }
    else:
        by_label = raw
    if not set(by_label) <= set(labels):
        raise MalformedAnswer(f"{where}: probabilities name unknown levels")
    return ScoreAnswer(
        where, float(score), MappingProxyType(by_label), _confidence(entry, where)
    )


def _integer(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _usage(raw: Any) -> DecisionUsage:
    if not isinstance(raw, Mapping):
        return DecisionUsage()
    details = raw.get("input_tokens_details")
    details = details if isinstance(details, Mapping) else {}
    cost = raw.get("cost")
    return DecisionUsage(
        input_tokens=_integer(raw, "input_tokens"),
        output_tokens=_integer(raw, "output_tokens"),
        cached_tokens=_integer(details, "cached_tokens"),
        cache_write_tokens=_integer(details, "cache_write_tokens"),
        cost=float(cost)
        if isinstance(cost, int | float) and not isinstance(cost, bool)
        else None,
    )


def parse_response(
    dialect: Dialect, payload: Any, questions: Sequence[Question]
) -> DecisionResult:
    """Return the typed answers of a decoded response to *questions*.

    Raises:
        MalformedAnswer: The response holds no answers, misses a question, or
            an answer does not fit its question; it carries the response's
            usage when the response is a JSON object.
    """
    if not isinstance(payload, Mapping):
        raise MalformedAnswer("the response is not a JSON object")
    usage = _usage(payload.get("usage"))
    try:
        answers = _answers(dialect, payload, questions)
    except MalformedAnswer as exc:
        raise MalformedAnswer(str(exc), usage) from None
    model = payload.get("model")
    request_id = payload.get("id")
    return DecisionResult(
        MappingProxyType(answers),
        model if isinstance(model, str) else "",
        usage,
        request_id if isinstance(request_id, str) else None,
    )


def _answers(
    dialect: Dialect, payload: Mapping[str, Any], questions: Sequence[Question]
) -> dict[str, Answer]:
    raw = payload.get("answers")
    if dialect == "openai" and isinstance(raw, list):
        raw = {entry.get("name"): entry for entry in raw if isinstance(entry, Mapping)}
    if not isinstance(raw, Mapping):
        error = payload.get("error")
        message = error.get("message") if isinstance(error, Mapping) else None
        detail = f": {message[:200]}" if isinstance(message, str) else ""
        raise MalformedAnswer(f"the response holds no answers{detail}")
    return {q.name: _answer(dialect, q, raw.get(q.name)) for q in questions}


def group_indices(
    count: int, *, per_group: int, questions_per_item: int = 1
) -> list[range]:
    """Split ``range(count)`` into consecutive groups for one request each.

    A group holds at most *per_group* items and at most :data:`MAX_QUESTIONS`
    questions at *questions_per_item* per item.

    Raises:
        ValueError: *per_group* or *questions_per_item* is below 1, or one
            item alone exceeds the question limit.
    """
    if per_group < 1 or questions_per_item < 1:
        raise ValueError("per_group and questions_per_item must be at least 1")
    if questions_per_item > MAX_QUESTIONS:
        raise ValueError("one item asks more questions than a request may hold")
    size = min(per_group, MAX_QUESTIONS // questions_per_item)
    return [range(start, min(start + size, count)) for start in range(0, count, size)]


def request_fingerprint(**fields: Any) -> str:
    """Return a short stable hash of the request settings in *fields*.

    A threshold validated on one request applies only to requests with the
    same fingerprint: dialect, model, question and taxonomy version, image
    size, grouping and context fields all change the answers.
    """
    text = json.dumps(fields, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
