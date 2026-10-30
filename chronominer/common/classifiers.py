"""Classifier files: the questions a decision model answers about each unit.

A classifier is a JSON file::

    {
      "name": "front_matter",
      "version": 1,
      "unit": "page",
      "description": "Find the title page and the blank leaves.",
      "per_request": 10,
      "questions": [
        {"name": "kind", "type": "page_types", "profile": "culinary"},
        {"name": "title", "type": "predicate",
         "instructions": "The page is the book's title page."},
        {"name": "script", "type": "choice", "instructions": "Which script?",
         "choices": {"roman": "Roman type.", "blackletter": null}},
        {"name": "legible", "type": "score", "instructions": "How legible?",
         "levels": [{"label": "no"}, {"label": "yes", "description": "Clear."}]}
      ]
    }

``unit`` is what one answer is about: ``page`` (a page image of a PDF or an
image folder), ``chunk`` (an extraction chunk: a text line range, an image or
a PDF page), ``entry`` (an extracted entry) or ``document`` (an item judged
from a selection of its pages or chunks in one request). A question is a
``predicate`` (the probability that ``instructions`` holds), a ``choice`` (one
of ``choices``, value to description or null), a ``score`` (an expected level
of ``levels``, lowest first) or ``page_types`` (the shared page-type choice of
``profile``, see :mod:`.page_types`; units ``page`` and ``chunk`` only).

``per_request`` (default 1) is the most units one request asks about. With
more than one, every question of unit *n* in a request is sent as
``<name>__<nn>`` and its instructions name the unit (``About page 3: ...``).
Unit ``entry`` needs ``schema`` (the extraction schema's ``name``) and
``fields`` (the entry fields sent); unit ``document`` takes ``selection``
(``first_pages``, default 10, for page images; ``max_tokens``, default
20,000, for the leading text chunks).

:attr:`Classifier.sha256` hashes the normalized classifier, defaults and the
page-type taxonomy version included, so it changes whenever the questions
sent change. The module imports only the standard library and the shared
decision modules.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, cast

from .decisions import MAX_QUESTIONS, Choice, Level, Predicate, Question, Score
from .page_types import DEFAULT_PROFILE, PROFILES, VERSION, Profile, page_type_question

__all__ = [
    "DEFAULT_FIRST_PAGES",
    "DEFAULT_MAX_TOKENS",
    "QUESTION_TYPES",
    "UNITS",
    "Classifier",
    "ClassifierQuestion",
    "Kind",
    "QuestionType",
    "Selection",
    "UnitKind",
    "load_classifier",
    "parse_classifier",
    "resolve_classifier",
    "slot_name",
]

type UnitKind = Literal["page", "chunk", "entry", "document"]
type QuestionType = Literal["predicate", "choice", "score", "page_types"]
type Kind = Literal["predicate", "choice", "score"]
"""The stored kind of an answer; ``page_types`` questions are choices."""

UNITS: Final[tuple[UnitKind, ...]] = ("page", "chunk", "entry", "document")
QUESTION_TYPES: Final[tuple[QuestionType, ...]] = (
    "predicate",
    "choice",
    "score",
    "page_types",
)
DEFAULT_FIRST_PAGES: Final = 10
DEFAULT_MAX_TOKENS: Final = 20_000

_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_QUESTION_NAME: Final = re.compile(r"[a-z][a-z0-9_]*")
_TOP_KEYS: Final = frozenset(
    {
        "name",
        "version",
        "unit",
        "description",
        "questions",
        "per_request",
        "schema",
        "fields",
        "selection",
    }
)
_UNIT_KEYS: Final[Mapping[str, UnitKind]] = MappingProxyType(
    {"schema": "entry", "fields": "entry", "selection": "document"}
)
_QUESTION_KEYS: Final[Mapping[QuestionType, frozenset[str]]] = MappingProxyType(
    {
        "predicate": frozenset({"name", "type", "instructions"}),
        "choice": frozenset({"name", "type", "instructions", "choices"}),
        "score": frozenset({"name", "type", "instructions", "levels"}),
        "page_types": frozenset({"name", "type", "profile"}),
    }
)
_PAGE_TYPE_UNITS: Final = frozenset({"page", "chunk"})
_SELECTION_KEYS: Final = frozenset({"first_pages", "max_tokens"})


def slot_name(name: str, slot: int | None) -> str:
    """Return the name question *name* is sent as for unit *slot* of a request."""
    return name if slot is None else f"{name}__{slot:02d}"


@dataclass(frozen=True)
class Selection:
    """The part of a document a ``document`` classifier sends."""

    first_pages: int = DEFAULT_FIRST_PAGES
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass(frozen=True)
class ClassifierQuestion:
    """One question of a classifier file, as validated."""

    name: str
    type: QuestionType
    instructions: str | None = None
    choices: Mapping[str, str | None] | None = None
    levels: tuple[Level, ...] | None = None
    profile: Profile | None = None

    @property
    def kind(self) -> Kind:
        """Return the stored kind of the answers."""
        return "choice" if self.type == "page_types" else self.type

    def question(self, slot: int | None, subject: str) -> Question:
        """Return the decision question for unit *slot*, named by *subject*.

        *subject* names the unit in the request, such as ``page 3``; a
        question sent about a single unit keeps its instructions.
        """
        name = slot_name(self.name, slot)
        if self.type == "page_types":
            return page_type_question(name, subject, self.profile or DEFAULT_PROFILE)
        text = cast(str, self.instructions)
        if slot is not None:
            text = f"About {subject}: {text}"
        if self.type == "predicate":
            return Predicate(name, text)
        if self.type == "choice":
            return Choice(name, text, cast(Mapping[str, str | None], self.choices))
        return Score(name, text, cast(tuple[Level, ...], self.levels))

    def as_dict(self) -> dict[str, Any]:
        """Return the normalized question, as hashed."""
        body: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.type == "page_types":
            body["profile"] = self.profile
            body["taxonomy"] = VERSION
            return body
        body["instructions"] = self.instructions
        if self.choices is not None:
            body["choices"] = dict(self.choices)
        if self.levels is not None:
            body["levels"] = [
                {"label": level.label, "description": level.description}
                for level in self.levels
            ]
        return body


@dataclass(frozen=True)
class Classifier:
    """A validated classifier file."""

    name: str
    version: int
    unit: UnitKind
    description: str
    questions: tuple[ClassifierQuestion, ...]
    per_request: int = 1
    schema: str | None = None
    fields: tuple[str, ...] | None = None
    selection: Selection | None = None

    @property
    def grouped(self) -> bool:
        """Whether requests name their units (``per_request`` above 1)."""
        return self.per_request > 1

    def subject(self, slot: int | None) -> str:
        """Return how the questions name unit *slot* (``page 3``, ``this page``)."""
        return f"this {self.unit}" if slot is None else f"{self.unit} {slot}"

    def questions_for(self, slot: int | None) -> tuple[Question, ...]:
        """Return the decision questions about unit *slot* (None: the only unit)."""
        subject = self.subject(slot)
        return tuple(q.question(slot, subject) for q in self.questions)

    def as_dict(self) -> dict[str, Any]:
        """Return the normalized classifier, defaults filled in."""
        body: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "unit": self.unit,
            "description": self.description,
            "per_request": self.per_request,
            "questions": [q.as_dict() for q in self.questions],
        }
        if self.schema is not None:
            body["schema"] = self.schema
        if self.fields is not None:
            body["fields"] = list(self.fields)
        if self.selection is not None:
            body["selection"] = {
                "first_pages": self.selection.first_pages,
                "max_tokens": self.selection.max_tokens,
            }
        return body

    @property
    def sha256(self) -> str:
        """Return the SHA-256 of the normalized classifier as compact JSON.

        Keys keep their order (choices in the order sent), so reordering the
        choices of a question changes the hash.
        """
        text = json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(raw: Any, where: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{where} must be a nonempty text")
    return raw


def _count(raw: Any, where: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(f"{where} must be an integer of at least 1, got {raw!r}")
    return raw


def _keys(raw: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        names = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"{where}: unknown key {names}")


def _object(raw: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    return raw


def _choices(raw: Any, where: str) -> Mapping[str, str | None]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{where}.choices must be a nonempty object")
    if len(raw) < 2:
        raise ValueError(f"{where}.choices needs at least two values")
    for value, description in raw.items():
        _text(value, f"{where}.choices key")
        if description is not None:
            _text(description, f"{where}.choices[{value!r}]")
    return MappingProxyType(dict(raw))


def _levels(raw: Any, where: str) -> tuple[Level, ...]:
    if not isinstance(raw, list) or not 2 <= len(raw) <= 10:
        raise ValueError(f"{where}.levels must be a list of 2 to 10 levels")
    levels: list[Level] = []
    for number, entry in enumerate(raw):
        at = f"{where}.levels[{number}]"
        level = _object(entry, at)
        _keys(level, frozenset({"label", "description"}), at)
        description = level.get("description")
        levels.append(
            Level(
                _text(level.get("label"), f"{at}.label"),
                None
                if description is None
                else _text(description, f"{at}.description"),
            )
        )
    labels = [level.label for level in levels]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{where}.levels: labels must be unique")
    return tuple(levels)


def _question(raw: Any, number: int, unit: UnitKind, source: str) -> ClassifierQuestion:
    where = f"{source}: questions[{number}]"
    entry = _object(raw, where)
    name = entry.get("name")
    if not isinstance(name, str) or not _QUESTION_NAME.fullmatch(name):
        raise ValueError(
            f"{where}.name must be lowercase letters, digits and underscores, "
            f"starting with a letter; got {name!r}"
        )
    if "__" in name:
        raise ValueError(f"{where}.name {name!r} must not contain '__'")
    where = f"{source}: question {name!r}"
    kind = entry.get("type")
    if kind not in QUESTION_TYPES:
        raise ValueError(
            f"{where}: type must be one of {', '.join(QUESTION_TYPES)}, got {kind!r}"
        )
    kind = cast(QuestionType, kind)
    _keys(entry, _QUESTION_KEYS[kind], where)
    if kind == "page_types":
        if unit not in _PAGE_TYPE_UNITS:
            raise ValueError(f"{where}: page_types needs unit page or chunk")
        profile = entry.get("profile", DEFAULT_PROFILE)
        if profile not in PROFILES:
            raise ValueError(
                f"{where}: profile must be one of {', '.join(PROFILES)}, "
                f"got {profile!r}"
            )
        return ClassifierQuestion(name, kind, profile=cast(Profile, profile))
    instructions = _text(entry.get("instructions"), f"{where}.instructions")
    return ClassifierQuestion(
        name,
        kind,
        instructions,
        choices=_choices(entry.get("choices"), where) if kind == "choice" else None,
        levels=_levels(entry.get("levels"), where) if kind == "score" else None,
    )


def _questions(raw: Any, unit: UnitKind, source: str) -> tuple[ClassifierQuestion, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source}: questions must be a nonempty list")
    if len(raw) > MAX_QUESTIONS:
        raise ValueError(
            f"{source}: {len(raw)} questions exceed the limit of {MAX_QUESTIONS}"
        )
    questions = tuple(
        _question(entry, number, unit, source) for number, entry in enumerate(raw)
    )
    seen: set[str] = set()
    for question in questions:
        if question.name in seen:
            raise ValueError(f"{source}: question name {question.name!r} repeats")
        seen.add(question.name)
    return questions


def _fields(raw: Any, source: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source}: fields must be a nonempty list of field names")
    fields = tuple(_text(field, f"{source}: fields entry") for field in raw)
    if len(set(fields)) != len(fields):
        raise ValueError(f"{source}: fields must not repeat")
    return fields


def _selection(raw: Any, source: str) -> Selection:
    if raw is None:
        return Selection()
    where = f"{source}: selection"
    entry = _object(raw, where)
    _keys(entry, _SELECTION_KEYS, where)
    return Selection(
        first_pages=_count(
            entry.get("first_pages", DEFAULT_FIRST_PAGES), f"{where}.first_pages"
        ),
        max_tokens=_count(
            entry.get("max_tokens", DEFAULT_MAX_TOKENS), f"{where}.max_tokens"
        ),
    )


def _unit_keys(raw: Mapping[str, Any], unit: UnitKind, source: str) -> None:
    for key, owner in _UNIT_KEYS.items():
        if key in raw and unit != owner:
            raise ValueError(f"{source}: {key} applies only to unit {owner}")


def parse_classifier(raw: Any, source: str = "classifier") -> Classifier:
    """Return the classifier of a decoded JSON object.

    *source* names the file in the error messages.

    Raises:
        ValueError: The object is no valid classifier; the message names the
            key at fault.
    """
    body = _object(raw, source)
    _keys(body, _TOP_KEYS, source)
    name = body.get("name")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(
            f"{source}: name must be letters, digits, '_', '.' or '-', got {name!r}"
        )
    unit = body.get("unit")
    if unit not in UNITS:
        raise ValueError(
            f"{source}: unit must be one of {', '.join(UNITS)}, got {unit!r}"
        )
    unit = cast(UnitKind, unit)
    _unit_keys(body, unit, source)
    description = body.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"{source}: description must be a text")
    entry = unit == "entry"
    if entry and "schema" not in body:
        raise ValueError(f"{source}: unit entry needs schema")
    if entry and "fields" not in body:
        raise ValueError(f"{source}: unit entry needs fields")
    return Classifier(
        name=name,
        version=_count(body.get("version"), f"{source}: version"),
        unit=unit,
        description=description,
        questions=_questions(body.get("questions"), unit, source),
        per_request=_count(body.get("per_request", 1), f"{source}: per_request"),
        schema=_text(body["schema"], f"{source}: schema") if entry else None,
        fields=_fields(body["fields"], source) if entry else None,
        selection=(
            _selection(body.get("selection"), source) if unit == "document" else None
        ),
    )


def load_classifier(path: Path | str) -> Classifier:
    """Return the classifier in the JSON file at *path*.

    Raises:
        OSError: The file cannot be read.
        ValueError: The file is no JSON or no valid classifier.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    return parse_classifier(raw, str(path))


def _declared_name(path: Path) -> str | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = raw.get("name") if isinstance(raw, Mapping) else None
    return name if isinstance(name, str) else None


def resolve_classifier(value: str, folder: Path | str | None) -> Path:
    """Return the classifier file *value* names, like ``--schema``.

    *value* is an existing file path, else the stem of ``<value>.json`` in
    *folder*, else the ``name`` of a JSON file in *folder*. Files in *folder*
    that do not parse are passed over.

    Raises:
        FileNotFoundError: Nothing matches; the message lists the names found.
    """
    candidate = Path(value).expanduser()
    try:
        if candidate.is_file():
            return candidate.resolve()
    except (OSError, ValueError):
        pass
    names: Sequence[str] = ()
    if folder is not None and Path(folder).is_dir():
        folder = Path(folder)
        stem = folder / f"{value}.json"
        if stem.is_file():
            return stem.resolve()
        found = {path: _declared_name(path) for path in sorted(folder.glob("*.json"))}
        for path, name in found.items():
            if name == value:
                return path.resolve()
        names = sorted(name for name in found.values() if name is not None)
    available = ", ".join(names) or "none"
    raise FileNotFoundError(
        f"Classifier {value!r} not found as a file, as {value}.json or as a "
        f"classifier name in {folder}. Available: {available}"
    )
