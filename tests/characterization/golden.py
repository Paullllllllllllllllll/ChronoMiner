"""Normalized golden snapshots of a characterization run.

``snapshot`` turns a ``RunResult`` into text files: every file the run
created (``_output.json`` and other ``.json`` parsed, ``.csv``,
``_line_ranges.txt`` and other text as text, ``.sqlite`` as a dump of its
tables in key order and the views of ``VIEWS``, other files as a
placeholder), ``summary.json`` with the exit code and the JSON summary
line, and ``requests.json`` with the provider-class constructions, the
per-call request digests, the batch API records and, once per case, the full
text of every long prompt. JSON text inside JSON values (a model's
``output_text``) is parsed; other strings in files longer than
``LONG_FILE_TEXT`` become their length and digest, since ``requests.json``
records the prompts sent.

Temporary paths, the repository path, datetimes, durations, versions, UUIDs
and hex ids, batch and file ids (the run's ``replacements``), file hashes and
the hashes and byte sizes of encoded image payloads become placeholders,
since encoder builds change the payload bytes. The store's run and
submission ids become ``<run-N>`` and ``<submission-N>`` in the order the
store recorded them, and lease expiry, lease generation and latencies
placeholders. ``check_golden`` compares the snapshot with ``golden/<case>/``
and fails with a unified diff, or rewrites the case when
``CM_UPDATE_GOLDEN=1``. With ``CM_GOLDEN_RENDERED_ONLY=1`` only the rendered
outputs (``_output.json``, CSV and text files) and the exit codes are
compared (:func:`rendered_view`).
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from chronominer.common.sqlite import read
from chronominer.output.store import DATABASE_NAME, VIEWS
from tests.characterization.adapter import RunResult

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
UPDATE_ENV = "CM_UPDATE_GOLDEN"
RENDERED_ONLY_ENV = "CM_GOLDEN_RENDERED_ONLY"
REPO_ROOT = Path(__file__).resolve().parents[2]

LONG_TEXT = 120
"""Text blocks longer than this are recorded once per case under ``prompts``."""

LONG_FILE_TEXT = 400
"""Strings in created files longer than this are reduced to length and digest."""

_DATETIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_PLACEHOLDER_PATH_RE = re.compile(r"<(?:tmp|test_tmp|repo)>[^\s\"'<>|*?\]]*")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
_HEX_ID_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")
_DATA_URL_RE = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+")

KEY_PLACEHOLDERS: dict[str, str] = {
    "source_sha256": "<sha256>",
    "sha256": "<sha256>",
    "image_sha256": "<sha256>",
    "byte_size": "<byte_size>",
    "pymupdf_version": "<version>",
    "pillow_version": "<version>",
    "tool_version": "<version>",
    "timestamp": "<timestamp>",
    "created_at": "<timestamp>",
    "elapsed": "<duration>",
    "elapsed_s": "<duration>",
    "elapsed_seconds": "<duration>",
    "duration_s": "<duration>",
    "duration_seconds": "<duration>",
    "processing_time": "<duration>",
    "latency": "<duration>",
}
"""JSON keys whose values vary between identical runs or encoder builds."""

COLUMN_PLACEHOLDERS: dict[str, str] = {
    "hash": "<sha256>",
    "input_sha256": "<sha256>",
    "settings_fingerprint": "<sha256>",
    "tool_version": "<version>",
    "lease_until": "<lease_until>",
    "lease_generation": "<generation>",
    "latency": "<duration>",
}
"""SQLite columns whose values vary between identical runs."""

_ROWID_ORDER = frozenset({"runs", "artifacts", "submissions", "submission_requests"})
"""Tables keyed by run-specific values; their rows follow insertion order."""

_ORDER_BY: dict[str, tuple[str, ...]] = {
    "documents": ("name",),
    "notes": ("doc_id", "note_id"),
}
"""Tables dumped in another order than their primary key."""

_NOT_RENDERED = (".dump.txt", ".bytes.txt")
"""Suffixes of snapshot files that stand for something other than a view."""

_RENDERED_SUFFIXES = (".txt", ".md", ".csv", "_output.json")
"""Snapshot files that are the tool's rendered outputs."""

_CONSOLE_FILES = frozenset({"stdout.txt", "stderr.txt"})
"""Snapshot files that record console output, not a view."""


class Normalizer:
    """Replace run-specific values with stable placeholders.

    *roots* maps directories to placeholders; every spelling of a directory
    (as given, resolved, either slash, JSON-escaped) is replaced, longest
    first. *replacements* are extra literal replacements, applied first, and
    *key_placeholders* extra JSON keys whose values become placeholders.
    """

    def __init__(
        self,
        roots: Mapping[Path, str],
        replacements: Mapping[str, str] | None = None,
        key_placeholders: Mapping[str, str] | None = None,
    ) -> None:
        forms: dict[str, str] = {}
        for directory, placeholder in roots.items():
            for candidate in (directory, directory.resolve()):
                text = str(candidate)
                for form in (
                    text,
                    text.replace("\\", "/"),
                    text.replace("/", "\\"),
                    text.replace("\\", "\\\\"),
                ):
                    forms.setdefault(form, placeholder)
        self._roots = sorted(forms.items(), key=lambda item: len(item[0]), reverse=True)
        self._replacements = sorted(
            dict(replacements or {}).items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        self._keys = {**KEY_PLACEHOLDERS, **dict(key_placeholders or {})}

    def text(self, value: str) -> str:
        """Normalize paths, ids, payloads and datetimes in free text."""
        for literal, placeholder in self._replacements:
            value = value.replace(literal, placeholder)
        for form, placeholder in self._roots:
            value = value.replace(form, placeholder)
        value = _PLACEHOLDER_PATH_RE.sub(
            lambda m: m.group(0).replace("\\\\", "/").replace("\\", "/"), value
        )
        value = _DATA_URL_RE.sub("<data-url>", value)
        value = _UUID_RE.sub("<uuid>", value)
        value = _HEX_ID_RE.sub("<hex>", value)
        return _DATETIME_RE.sub("<datetime>", value)

    def data(self, value: Any, key: str | None = None) -> Any:
        """Normalize a parsed JSON value recursively."""
        if key in self._keys and value is not None:
            return self._keys[key]
        if isinstance(value, dict):
            return {
                self.text(k) if isinstance(k, str) else k: self.data(v, k)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self.data(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value


def normalizer_for(
    result: RunResult,
    replacements: Mapping[str, str] | None = None,
    key_placeholders: Mapping[str, str] | None = None,
) -> Normalizer:
    """Return the normalizer for *result*: case root, test tmp dir, repository.

    The run and submission ids of the stores below the case root and the
    run's folders (:func:`store_ids`) and the batch and file ids of the run's
    batch service are replaced first, then *replacements*.
    """
    roots = {
        result.root: "<tmp>",
        result.config_dir.parent: "<test_tmp>",
        REPO_ROOT: "<repo>",
    }
    dirs = [result.root, *result.scan_dirs]
    literals = {
        **store_ids(dirs),
        **result.replacements,
        **dict(replacements or {}),
    }
    return Normalizer(roots, literals, key_placeholders)


def _databases(dirs: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for directory in dirs:
        if directory.is_dir():
            found.update(path.resolve() for path in directory.rglob(DATABASE_NAME))
    return sorted(found)


def _recorded_ids(path: Path) -> tuple[list[str], list[str]]:
    def queries(connection: sqlite3.Connection) -> tuple[list[str], list[str]]:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        ids: list[list[str]] = []
        for table, column in (("runs", "run_id"), ("submissions", "submission_id")):
            if table not in tables:
                ids.append([])
                continue
            rows = connection.execute(f"SELECT {column} FROM {table} ORDER BY rowid")
            ids.append([str(row[0]) for row in rows])
        return ids[0], ids[1]

    return read(path, queries)


def store_ids(dirs: Sequence[Path]) -> dict[str, str]:
    """Return literal replacements for the run and submission ids under *dirs*.

    Every store found below *dirs* (in path order) contributes its runs and
    submissions in the order they were recorded: a run id becomes
    ``<run-N>``, a submission id ``<submission-N>``, and the eight-character
    prefix that scopes the submission's custom ids ``<submission-N>-``.
    """
    runs: dict[str, str] = {}
    submissions: dict[str, str] = {}
    for path in _databases(dirs):
        try:
            run_ids, submission_ids = _recorded_ids(path)
        except (OSError, sqlite3.Error):
            continue
        for run_id in run_ids:
            runs.setdefault(run_id, f"<run-{len(runs) + 1}>")
        for submission_id in submission_ids:
            submissions.setdefault(
                submission_id, f"<submission-{len(submissions) + 1}>"
            )
    prefixes = {
        f"{submission_id[:8]}-": f"{placeholder}-"
        for submission_id, placeholder in submissions.items()
    }
    return {**runs, **submissions, **prefixes}


def _dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _decoded(value: Any) -> Any:
    """Return a value as JSON would show it: JSON text parsed, bytes decoded."""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary: {len(value)} bytes>"
    if isinstance(value, str) and value.lstrip()[:1] in ("[", "{"):
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(value)
    return value


def _file_value(value: Any) -> Any:
    """Return a file's JSON value with JSON text parsed and long text digested."""
    if isinstance(value, dict):
        return {key: _file_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_file_value(item) for item in value]
    if isinstance(value, str):
        decoded = _decoded(value)
        if not isinstance(decoded, str):
            return _file_value(decoded)
        if len(value) > LONG_FILE_TEXT:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
            return f"<text: {len(value)} chars {digest}>"
    return value


def _shortened(value: Any) -> Any:
    """Return *value* with every string longer than ``LONG_TEXT`` as its length."""
    if isinstance(value, dict):
        return {key: _shortened(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shortened(item) for item in value]
    if isinstance(value, str) and len(value) > LONG_TEXT:
        return f"<text: {len(value)} chars>"
    return value


def _column_value(column: str, value: Any, norm: Normalizer) -> Any:
    if column in COLUMN_PLACEHOLDERS and value is not None:
        return COLUMN_PLACEHOLDERS[column]
    return norm.data(_file_value(_decoded(value)), column)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _order(connection: sqlite3.Connection, table: str) -> str:
    """Return the ORDER BY clause of *table*: its key, or its insertion order."""
    if table in _ROWID_ORDER:
        return "rowid"
    if table in _ORDER_BY:
        return ", ".join(_quote(column) for column in _ORDER_BY[table])
    columns = connection.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
    keyed = sorted((column[5], column[1]) for column in columns if column[5])
    return ", ".join(_quote(name) for _, name in keyed) or "rowid"


def _select(connection: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _response_ranks(rows: list[dict[str, Any]]) -> dict[Any, int]:
    """Return each response id's rank when attempts are ordered by unit.

    Attempts of different units interleave under concurrency; the ranks
    follow document, unit kind, index and, within a unit, the attempt order.
    """
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("doc_id") if row.get("doc_id") is not None else -1,
            str(row.get("kind") or ""),
            row.get("idx") if row.get("idx") is not None else -1,
            row["response_id"],
        ),
    )
    rows[:] = ordered
    return {row["response_id"]: rank for rank, row in enumerate(ordered, 1)}


def _store_rows(connection: sqlite3.Connection) -> list[tuple[str, list[dict]]]:
    names = connection.execute(
        "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'view') "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    tables = [name for kind, name in names if kind == "table"]
    views = {name for kind, name in names if kind == "view"}
    dumped = [
        (
            table,
            _select(
                connection,
                f"SELECT * FROM {_quote(table)} ORDER BY {_order(connection, table)}",
            ),
        )
        for table in tables
    ]
    dumped.extend(
        (f"view {view}", _select(connection, f"SELECT * FROM {_quote(view)}"))
        for view in VIEWS
        if view in views
    )
    return dumped


def dump_sqlite(path: Path, norm: Normalizer) -> str:
    """Return every table and view of a store, rows in key order, normalized.

    Rows follow the table's primary key, the insertion order for tables keyed
    by run-specific ids, or the order of ``_ORDER_BY``; responses follow
    their unit, and their ids become ranks in that order (the units' links
    to them too). The documented views (``VIEWS``) follow the tables.
    JSON text and raw bodies are parsed so that their paths and ids
    normalize too. Prompt texts of ``artifacts`` and the strings of a
    response's request parameters longer than ``LONG_TEXT`` are reduced to
    their length, as ``requests.json`` records the prompts and schemas sent.
    """
    dumped = read(path, _store_rows)
    tables = dict(dumped)
    ranks = _response_ranks(tables["responses"]) if "responses" in tables else {}
    parts: list[str] = []
    for table, rows in dumped:
        parts.append(f"## {table}\n")
        for row in rows:
            values = dict(row)
            if values.get("response_id") in ranks:
                values["response_id"] = ranks[values["response_id"]]
            text = values.get("text")
            if table == "artifacts" and isinstance(text, str) and len(text) > LONG_TEXT:
                values["text"] = f"<text: {len(text)} chars>"
            if table == "responses":
                values["request"] = _shortened(_decoded(values.get("request")))
            parts.append(
                _dumps(
                    {key: _column_value(key, val, norm) for key, val in values.items()}
                )
            )
    return "".join(parts)


def _file_entry(path: Path, norm: Normalizer) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".sqlite":
        return ".dump.txt", dump_sqlite(path, norm)
    if suffix in (".txt", ".md", ".csv"):
        text = path.read_bytes().decode("utf-8", errors="replace")
        if text.startswith("﻿"):
            text = "<utf-8 bom>" + text[1:]
        return "", norm.text(text.replace("\r\n", "\n"))
    if suffix == ".json":
        text = path.read_text(encoding="utf-8")
        try:
            return "", _dumps(norm.data(_file_value(json.loads(text))))
        except json.JSONDecodeError:
            return "", norm.text(text)
    return ".bytes.txt", "<binary file>\n"


def _prompt_refs(calls: list[dict[str, Any]], prompts: dict[str, str]) -> None:
    ids = {text: prompt_id for prompt_id, text in prompts.items()}
    for call in calls:
        for message in call.get("messages", []):
            for index, block in enumerate(message.get("blocks", [])):
                text = block.get("text")
                if not isinstance(text, str) or len(text) <= LONG_TEXT:
                    continue
                prompt_id = ids.get(text)
                if prompt_id is None:
                    prompt_id = f"prompt-{len(ids) + 1}"
                    ids[text] = prompt_id
                    prompts[prompt_id] = text
                message["blocks"][index] = {
                    "type": block.get("type", "text"),
                    "prompt": prompt_id,
                    "chars": len(text),
                }


def _requests(result: RunResult, norm: Normalizer) -> dict[str, Any]:
    """Return the request record with long prompts stored once by id."""
    prompts: dict[str, str] = {}
    calls = norm.data(result.calls)
    _prompt_refs(calls, prompts)
    models = sorted(
        norm.data(result.models),
        key=lambda m: json.dumps(m, sort_keys=True, default=str),
    )
    record: dict[str, Any] = {"models": models, "calls": calls, "prompts": prompts}
    if result.batch is not None:
        record["batch"] = norm.data(result.batch)
    return record


def snapshot(
    result: RunResult,
    *,
    replacements: Mapping[str, str] | None = None,
    key_placeholders: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the normalized golden files of one run, keyed by relative name.

    *replacements* (literal text to placeholder) and *key_placeholders*
    (JSON key to placeholder) extend the normalization.
    """
    norm = normalizer_for(result, replacements, key_placeholders)
    files: dict[str, str] = {}
    for path in result.created:
        relative = norm.text(str(path)).removeprefix("<tmp>/")
        extra, content = _file_entry(path, norm)
        files[f"files/{relative}{extra}"] = content
    files["summary.json"] = _dumps(
        {"exit_code": result.exit_code, "json_line": norm.data(result.summary)}
    )
    files["requests.json"] = _dumps(_requests(result, norm))
    return files


def _read_case(case_dir: Path) -> dict[str, str]:
    if not case_dir.is_dir():
        return {}
    return {
        path.relative_to(case_dir).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(case_dir.rglob("*"))
        if path.is_file()
    }


def _write_case(case_dir: Path, files: Mapping[str, str]) -> None:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    for name, content in files.items():
        target = case_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")


def _is_rendered(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return (
        base.endswith(_RENDERED_SUFFIXES)
        and not base.endswith(_NOT_RENDERED)
        and base not in _CONSOLE_FILES
    )


def _exit_code(content: str) -> str:
    try:
        summary = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(summary, dict):
        return _dumps({"exit_code": summary.get("exit_code")})
    return content


def rendered_view(files: Mapping[str, str]) -> dict[str, str]:
    """Return the rendered outputs and the exit codes of a snapshot.

    The rendered outputs are ``_output.json``, CSV and text files such as
    ``_line_ranges.txt``. Database dumps, placeholders of binary files,
    captured stdout, other JSON files and the request records are left out;
    each ``summary.json`` is reduced to its exit code.
    """
    view = {name: content for name, content in files.items() if _is_rendered(name)}
    for name, content in files.items():
        if name.rsplit("/", 1)[-1] == "summary.json":
            view[name] = _exit_code(content)
    return view


def compare(expected: Mapping[str, str], actual: Mapping[str, str]) -> list[str]:
    """Return human-readable differences between two snapshots."""
    problems: list[str] = []
    for name in sorted(set(expected) | set(actual)):
        if name not in actual:
            problems.append(f"missing from run: {name}")
        elif name not in expected:
            problems.append(f"not in golden: {name}")
        elif expected[name] != actual[name]:
            diff = difflib.unified_diff(
                expected[name].splitlines(keepends=True),
                actual[name].splitlines(keepends=True),
                fromfile=f"golden/{name}",
                tofile=f"run/{name}",
            )
            problems.append("".join(diff))
    return problems


def check_files(
    case: str,
    actual: Mapping[str, str],
    *,
    on_write: Callable[[Path], object] | None = None,
) -> None:
    """Compare text files keyed by relative name with ``golden/<case>/``.

    With ``CM_UPDATE_GOLDEN=1`` the case is rewritten instead and *on_write*
    receives the golden directory (the conftest clears the write guard's
    entries for it). With ``CM_GOLDEN_RENDERED_ONLY=1`` only the
    :func:`rendered_view` of both sides is compared.
    """
    case_dir = GOLDEN_DIR / case
    if os.environ.get(UPDATE_ENV) == "1":
        _write_case(case_dir, actual)
        if on_write is not None:
            on_write(GOLDEN_DIR)
        return
    expected = _read_case(case_dir)
    if not expected:
        raise AssertionError(
            f"No golden files for case {case!r}; run with {UPDATE_ENV}=1 to create."
        )
    if os.environ.get(RENDERED_ONLY_ENV) == "1":
        expected, actual = rendered_view(expected), rendered_view(actual)
    problems = compare(expected, actual)
    if problems:
        raise AssertionError(
            f"Golden mismatch for case {case!r} (set {UPDATE_ENV}=1 to accept):\n"
            + "\n".join(problems)
        )


def check_golden(
    case: str,
    result: RunResult,
    *,
    replacements: Mapping[str, str] | None = None,
    key_placeholders: Mapping[str, str] | None = None,
    on_write: Callable[[Path], object] | None = None,
) -> None:
    """Compare a run with ``golden/<case>/``; rewrite it when updating."""
    actual = snapshot(
        result, replacements=replacements, key_placeholders=key_placeholders
    )
    check_files(case, actual, on_write=on_write)
