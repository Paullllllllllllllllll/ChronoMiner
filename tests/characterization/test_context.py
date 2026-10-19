"""Characterization of extraction context: text context and context images.

Each case runs ``extract`` through ``run_tool``, twice in fresh roots, and
compares the second run with ``golden/context/<case>/``. Text context
resolves in this order: ``--context PATH`` or topics as text (manual), the
file sidecar ``{stem}_extract_context.txt``, the folder sidecar
``{folder}_extract_context.txt`` beside the input folder, then
``--fallback-context``. The project context folder, which the
characterization conftest points at ``tests/characterization/context``,
serves context images only. ``--context none`` sends no context.

``--context-image`` is a switch: it resolves an image with
the same hierarchy (``{stem}_extract_context.png`` and so on) and sends it as
a ``Context image:`` text block and an image block before the input, for
OpenAI and OpenRouter only. Context-image cases use an image input, so the
page image stays the last image of each request.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.characterization.adapter import RunResult, run_tool
from tests.characterization.conftest import CONTEXT_DIR, CaseRoot, GoldenCheck
from tests.characterization.fakes import FakeLLM, page_unit
from tests.characterization.golden import compare, snapshot
from tests.characterization.inputs import (
    make_context_image,
    make_image,
    make_text_file,
)

pytestmark = pytest.mark.integration

RECORDS = 6
MANUAL_CONTEXT = "Manual context: the grain ledger of a provincial granary."
FILE_CONTEXT = "File context: deliveries recorded by the granary clerk."
FOLDER_CONTEXT = "Folder context: ledgers kept by the town council."
PROJECT_CONTEXT = "Project context: a fixture image folder of scanned ledgers."
FALLBACK_CONTEXT = "Fallback context: grain deliveries, one per line."
TOPICS = "grain deliveries, 1790s"
CONTEXT_LABEL = "Context image:"
DEFAULT_CONTEXT = (CONTEXT_DIR / "extract_context.txt").read_text(encoding="utf-8")

Setup = Callable[[Path], dict[str, Any]]
Check = Callable[[RunResult], None]
UseContext = Callable[[Path], None]


@dataclass(frozen=True)
class ContextCase:
    """One context case.

    ``setup`` writes context files into the case root and returns more
    ``run_tool`` flags; ``project`` names a project context folder (inside
    the test's tmp dir) that replaces the default one.
    """

    name: str
    source: str
    setup: Setup | None = None
    flags: Mapping[str, Any] = field(default_factory=dict)
    project: bool = False
    check: Check | None = None


def _text(root: Path) -> Path:
    return make_text_file(root / "docs" / "ledger.txt", RECORDS)


def _image(root: Path) -> Path:
    return make_image(root / "scans" / "page.png", 0)


SOURCES: dict[str, Callable[[Path], Path]] = {"text": _text, "image": _image}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _system_text(call: Mapping[str, Any]) -> str:
    message = call["messages"][0]
    assert message["role"] == "system"
    return "".join(block.get("text", "") for block in message["blocks"])


def _human_blocks(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    message = call["messages"][-1]
    assert message["role"] == "human"
    return list(message["blocks"])


def _manual(root: Path) -> dict[str, Any]:
    _write(root / "docs" / "ledger_extract_context.txt", FILE_CONTEXT)
    return {"context": _write(root / "ctx" / "manual.txt", MANUAL_CONTEXT)}


def _file_sidecar(root: Path) -> dict[str, Any]:
    _write(root / "docs" / "ledger_extract_context.txt", FILE_CONTEXT)
    _write(root / "docs_extract_context.txt", FOLDER_CONTEXT)
    return {}


def _folder_sidecar(root: Path) -> dict[str, Any]:
    _write(root / "docs_extract_context.txt", FOLDER_CONTEXT)
    return {}


def _none(root: Path) -> dict[str, Any]:
    _write(root / "docs" / "ledger_extract_context.txt", FILE_CONTEXT)
    return {"context": "none"}


def _image_sidecar(root: Path) -> dict[str, Any]:
    make_context_image(root / "scans" / "page_extract_context.png")
    return {"context_image": True}


def _image_switch(root: Path) -> dict[str, Any]:
    return {"context_image": True}


def _image_sidecar_switch_off(root: Path) -> dict[str, Any]:
    make_context_image(root / "scans" / "page_extract_context.png")
    return {}


def _image_sidecar_context_none(root: Path) -> dict[str, Any]:
    make_context_image(root / "scans" / "page_extract_context.png")
    return {"context_image": True, "context": "none"}


def _image_sidecar_anthropic(root: Path) -> dict[str, Any]:
    make_context_image(root / "scans" / "page_extract_context.png")
    return {"context_image": True, "provider": "anthropic"}


def _context_check(expected: str, absent: tuple[str, ...] = ()) -> Check:
    def check(result: RunResult) -> None:
        for call in result.calls:
            system = _system_text(call)
            assert expected in system
            assert all(text not in system for text in absent)

    return check


def _check_no_context(result: RunResult) -> None:
    for call in result.calls:
        system = _system_text(call)
        assert FILE_CONTEXT not in system
        assert DEFAULT_CONTEXT.splitlines()[0] not in system


def _with_context_image(result: RunResult) -> None:
    for call in result.calls:
        blocks = _human_blocks(call)
        assert [block["type"] for block in blocks] == [
            "text",
            "image_url",
            "text",
            "image_url",
        ]
        assert blocks[0]["text"] == CONTEXT_LABEL
        assert call["unit"] == page_unit(0)


def _without_context_image(result: RunResult) -> None:
    for call in result.calls:
        blocks = _human_blocks(call)
        assert CONTEXT_LABEL not in [block.get("text") for block in blocks]
        assert len([b for b in blocks if b["type"] in ("image", "image_url")]) == 1


def _check_project_image(result: RunResult) -> None:
    _with_context_image(result)
    for call in result.calls:
        assert PROJECT_CONTEXT not in _system_text(call)


def _fallback(root: Path) -> dict[str, Any]:
    return {"fallback_context": FALLBACK_CONTEXT}


def _fallback_beside_a_sidecar(root: Path) -> dict[str, Any]:
    _write(root / "docs" / "ledger_extract_context.txt", FILE_CONTEXT)
    return {"fallback_context": FALLBACK_CONTEXT}


def _topics(root: Path) -> dict[str, Any]:
    _write(root / "docs" / "ledger_extract_context.txt", FILE_CONTEXT)
    return {"context": TOPICS}


CASES: tuple[ContextCase, ...] = (
    ContextCase(
        "text_manual_path",
        "text",
        setup=_manual,
        check=_context_check(MANUAL_CONTEXT, absent=(FILE_CONTEXT,)),
    ),
    ContextCase(
        "text_file_sidecar",
        "text",
        setup=_file_sidecar,
        check=_context_check(FILE_CONTEXT, absent=(FOLDER_CONTEXT,)),
    ),
    ContextCase(
        "text_folder_sidecar",
        "text",
        setup=_folder_sidecar,
        check=_context_check(FOLDER_CONTEXT),
    ),
    ContextCase(
        "text_fallback_context",
        "text",
        setup=_fallback,
        check=_context_check(
            FALLBACK_CONTEXT, absent=(DEFAULT_CONTEXT.splitlines()[0],)
        ),
    ),
    ContextCase(
        "text_sidecar_over_fallback",
        "text",
        setup=_fallback_beside_a_sidecar,
        check=_context_check(FILE_CONTEXT, absent=(FALLBACK_CONTEXT,)),
    ),
    ContextCase(
        "text_topics",
        "text",
        setup=_topics,
        check=_context_check(TOPICS, absent=(FILE_CONTEXT,)),
    ),
    ContextCase("text_context_none", "text", setup=_none, check=_check_no_context),
    ContextCase(
        "image_context_image_sidecar",
        "image",
        setup=_image_sidecar,
        check=_with_context_image,
    ),
    ContextCase(
        "image_context_image_project",
        "image",
        setup=_image_switch,
        project=True,
        check=_check_project_image,
    ),
    ContextCase(
        "image_context_image_missing",
        "image",
        setup=_image_switch,
        check=_without_context_image,
    ),
    ContextCase(
        "image_context_image_switch_off",
        "image",
        setup=_image_sidecar_switch_off,
        check=_without_context_image,
    ),
    ContextCase(
        "image_context_image_context_none",
        "image",
        setup=_image_sidecar_context_none,
        check=_without_context_image,
    ),
    ContextCase(
        "image_context_image_anthropic",
        "image",
        setup=_image_sidecar_anthropic,
        check=_without_context_image,
    ),
)


def _project_folder(base: Path) -> Path:
    """Write a project context folder with a text context and a context image."""
    folder = base / "project_context"
    if not folder.is_dir():
        _write(folder / "extract_context.txt", PROJECT_CONTEXT)
        make_context_image(folder / "extract_context.png")
    return folder


def _run(case: ContextCase, llm: FakeLLM, root: Path) -> RunResult:
    source = SOURCES[case.source](root)
    extra = case.setup(root) if case.setup is not None else {}
    return run_tool(
        "extract",
        llm=llm,
        root=root,
        input=source,
        output=root / "out",
        **{**dict(case.flags), **extra},
    )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_context_case(
    case: ContextCase,
    fake_llm: FakeLLM,
    case_root: CaseRoot,
    golden: GoldenCheck,
    project_context: UseContext,
    tmp_path: Path,
) -> None:
    if case.project:
        project_context(_project_folder(tmp_path))
    first = snapshot(_run(case, fake_llm, case_root("run1")))
    result = _run(case, fake_llm, case_root("run2"))
    problems = compare(first, snapshot(result))
    assert not problems, "two identical runs differ:\n" + "\n".join(problems)

    assert result.exit_code == 0, result.stdout + result.stderr
    assert result.calls
    if case.check is not None:
        case.check(result)
    golden(f"context/{case.name}", result)


def test_case_names_are_unique() -> None:
    names = [case.name for case in CASES]
    assert len(names) == len(set(names))
