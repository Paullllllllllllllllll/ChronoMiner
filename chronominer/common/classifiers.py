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
sent change.

:func:`find_named_file` is the lookup a tool's ``--classifier`` and
``--schema`` share: an existing file path, else the user folders (the
``--classifiers-dir`` or ``--schemas-dir`` folders, then the settings
folders), else the tool's bundled folder. A file in a user folder shadows the
bundled file of the same file name; a name found in two user folders raises
:class:`AmbiguousNameError`. The module imports only the standard library and
the shared decision modules.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    "AmbiguousNameError",
    "Classifier",
    "ClassifierQuestion",
    "Kind",
    "QuestionType",
    "Selection",
    "UnitKind",
    "declared_name",
    "find_named_file",
    "load_classifier",
    "named_files",
    "parse_classifier",
    "resolve_classifier",
    "search_folders",
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


class AmbiguousNameError(ValueError):
    """A name matches files in more than one user folder.

    *paths* holds the matching file of each folder, in search order. Callers
    that pass over a name they cannot resolve still raise this error.
    """

    def __init__(self, kind: str, value: str, paths: Sequence[Path]) -> None:
        self.kind = kind
        self.value = value
        self.paths = tuple(paths)
        files = " and ".join(str(path) for path in self.paths)
        super().__init__(
            f"{kind} {value!r} is ambiguous: it matches {files}, each in another "
            "user folder; pass the file path, or keep only one of the files"
        )


type NameReader = Callable[[Path], str | None]
"""Returns the name a JSON file declares, None when it declares none."""


def declared_name(path: Path) -> str | None:
    """Return the ``name`` field of the JSON object in *path*, if it has one."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = raw.get("name") if isinstance(raw, Mapping) else None
    return name if isinstance(name, str) and name.strip() else None


def _key(path: Path) -> str:
    try:
        resolved = path.expanduser().resolve()
    except (OSError, RuntimeError):
        resolved = path
    return os.path.normcase(str(resolved))


def _distinct(folders: Iterable[Path | str]) -> list[Path]:
    """Return *folders* as paths, each folder at its first place only."""
    seen: set[str] = set()
    kept: list[Path] = []
    for folder in folders:
        path = Path(folder).expanduser()
        key = _key(path)
        if key not in seen:
            seen.add(key)
            kept.append(path)
    return kept


def search_folders(
    folders: Iterable[Path | str], bundled: Path | str | None = None
) -> tuple[Path, ...]:
    """Return the folders a lookup searches: *folders*, then *bundled*.

    A folder named twice is searched at its first place only.
    """
    return tuple(_distinct([*folders, *([] if bundled is None else [bundled])]))


def _files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


def _file_names(value: str) -> list[str]:
    """Return the file names *value* can mean inside a folder."""
    names = [f"{value}.json"]
    if value.lower().endswith(".json"):
        names.insert(0, value)
    return [name for name in names if Path(name).name == name]


def _match(
    folder: Path,
    value: str,
    declared: NameReader,
    shadowed: frozenset[str] = frozenset(),
) -> Path | None:
    """Return the file of *folder* that *value* names: by file name, else by name."""
    if not folder.is_dir():
        return None
    for name in _file_names(value):
        candidate = folder / name
        if os.path.normcase(name) not in shadowed and candidate.is_file():
            return candidate
    for path in _files(folder):
        if os.path.normcase(path.name) not in shadowed and declared(path) == value:
            return path
    return None


def _shadowed(user: Sequence[Path]) -> frozenset[str]:
    return frozenset(
        os.path.normcase(path.name) for folder in user for path in _files(folder)
    )


def find_named_file(
    value: str,
    folders: Iterable[Path | str] = (),
    bundled: Path | str | None = None,
    *,
    kind: str = "file",
    declared: NameReader = declared_name,
) -> Path | None:
    """Return the JSON file *value* names, None when nothing matches.

    *value* is, in this order: an existing file path; in the user *folders*,
    a file ``<value>.json`` (or *value* itself when it ends in ``.json``) or a
    file whose name (*declared*) is *value*; the same in *bundled*, leaving
    out the files whose file name a user folder holds. Within one folder a
    file name wins over a declared name, and the first file in name order
    wins. *kind* names the file type in the error message.

    Raises:
        AmbiguousNameError: Two user folders each hold a file *value* names.
    """
    text = value.strip()
    if not text:
        return None
    given = Path(text).expanduser()
    try:
        if given.is_file():
            return given
    except (OSError, ValueError):
        pass
    user = _distinct(folders)
    found = [
        match
        for folder in user
        if (match := _match(folder, text, declared)) is not None
    ]
    if len(found) > 1:
        raise AmbiguousNameError(kind, text, found)
    if found:
        return found[0]
    if bundled is None or _key(Path(bundled)) in {_key(folder) for folder in user}:
        return None
    return _match(Path(bundled), text, declared, _shadowed(user))


def named_files(
    folders: Iterable[Path | str] = (), bundled: Path | str | None = None
) -> list[Path]:
    """Return the JSON files a lookup can find, in search order.

    These are the files of the user *folders*, then those of *bundled* whose
    file name no user folder holds.
    """
    user = _distinct(folders)
    files = [path for folder in user for path in _files(folder)]
    if bundled is None or _key(Path(bundled)) in {_key(folder) for folder in user}:
        return files
    shadowed = _shadowed(user)
    return files + [
        path
        for path in _files(Path(bundled))
        if os.path.normcase(path.name) not in shadowed
    ]


def resolve_classifier(
    value: str, folder: Path | str | None, folders: Iterable[Path | str] = ()
) -> Path:
    """Return the classifier file *value* names, like ``--schema``.

    *value* is an existing file path, else a classifier of the user
    *folders*, else one of the bundled *folder*, each found by file name
    without ``.json`` or by its ``name`` (:func:`find_named_file`). Files
    that do not parse are passed over.

    Raises:
        AmbiguousNameError: Two user folders hold a classifier of that name.
        FileNotFoundError: Nothing matches; the message names the folders
            searched and the classifiers found.
    """
    folders = tuple(folders)
    found = find_named_file(value, folders, folder, kind="classifier")
    if found is not None:
        return found.resolve()
    names = sorted(
        {
            name
            for path in named_files(folders, folder)
            if (name := declared_name(path)) is not None
        }
    )
    available = ", ".join(names) or "none"
    searched = ", ".join(str(path) for path in search_folders(folders, folder))
    raise FileNotFoundError(
        f"Classifier {value!r} not found as a file, as {value}.json or as a "
        f"classifier name in {searched or 'no folder'}. Available: {available}"
    )
