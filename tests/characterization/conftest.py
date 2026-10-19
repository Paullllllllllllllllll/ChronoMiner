"""Fixtures for the characterization tests: fakes, goldens, contexts, case roots.

``fake_llm`` (autouse) replaces the provider classes with ``FakeLLM``
stand-ins; ``default_context`` (autouse) makes ``context/`` beside this file
the project context folder; ``fake_batch`` installs the batch fakes;
``golden`` returns ``check(case, result, ...)`` for case names
``<area>/<name>``; ``case_root`` makes a fresh root per run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import pytest

from chronominer.common.testing import WriteGuard
from tests.characterization.adapter import RunResult
from tests.characterization.fakes import PROVIDER_CLASSES, FakeLLM
from tests.characterization.fakes_batch import FakeBatchService
from tests.characterization.golden import check_golden
from tests.characterization.seams import replace

GoldenCheck = Callable[..., None]
CaseRoot = Callable[..., Path]
"""``make(name="case")`` returned by the ``case_root`` fixture."""

CONTEXT_DIR = Path(__file__).parent / "context"
"""The project context folder of every run: neutral default contexts.

Runs read it instead of the repository's ``context/`` folder, so the goldens
do not depend on the default context a checkout carries.
"""


@pytest.fixture(autouse=True)
def default_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make :data:`CONTEXT_DIR` the project context folder of the run."""
    import chronominer.llm.prompts as prompts_module

    monkeypatch.setattr(prompts_module, "CONTEXT_DIR", CONTEXT_DIR)


@pytest.fixture
def project_context(monkeypatch: pytest.MonkeyPatch) -> Callable[[Path], None]:
    """Return ``use(folder)``, which makes *folder* the project context folder.

    The project context folder holds the default context of every run
    (``llm.prompts.CONTEXT_DIR``).
    """
    import chronominer.llm.prompts as prompts_module

    def use(folder: Path) -> None:
        monkeypatch.setattr(prompts_module, "CONTEXT_DIR", folder)

    return use


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Replace the provider classes with ``FakeLLM`` stand-ins and return it."""
    controller = FakeLLM()
    for module_name, name in PROVIDER_CLASSES:
        replace(monkeypatch, module_name, name, controller.constructor(name))
    return controller


@pytest.fixture
def fake_batch(
    fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> Iterator[FakeBatchService]:
    """Install the batch fakes of OpenAI, Anthropic and Google; return the service."""
    service = FakeBatchService(fake_llm)
    service.install(monkeypatch)
    yield service
    service.reset()


@pytest.fixture
def golden(write_guard: WriteGuard) -> GoldenCheck:
    """Return ``check(case, result, *, replacements=None, key_placeholders=None)``.

    The check compares a run with ``golden/<case>/`` (see ``golden.py``).
    """

    def check(
        case: str,
        result: RunResult,
        *,
        replacements: Mapping[str, str] | None = None,
        key_placeholders: Mapping[str, str] | None = None,
    ) -> None:
        check_golden(
            case,
            result,
            replacements=replacements,
            key_placeholders=key_placeholders,
            on_write=write_guard.consume,
        )

    return check


@pytest.fixture
def case_root(tmp_path: Path) -> CaseRoot:
    """Return ``make(name="case")``, which creates and returns ``tmp_path/name``.

    Each run of a case gets its own root, so files of one run never count as
    created by the next.
    """

    def make(name: str = "case") -> Path:
        root = tmp_path / name
        root.mkdir()
        return root

    return make
