"""Stages of characterization cases and the runner that checks them twice.

A case builds its inputs in a fresh root and runs one or more commands; each
command is a stage with its golden files. ``run_twice`` runs a case in the
roots ``run1`` and ``run2`` and fails when the two snapshots differ, so a
golden never records output that changes between identical runs.
``case_runner`` then compares every stage with
``golden/<prefix>/<case>/<stage>/``, ``single_case_runner`` a one-stage case
with ``golden/<prefix>/<case>/``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol

from chronominer.common.testing import WriteGuard
from tests.characterization.adapter import RunResult
from tests.characterization.golden import check_files, compare, snapshot


class Snapshotted(Protocol):
    """Anything that carries the golden files of one stage."""

    @property
    def files(self) -> dict[str, str]: ...


CaseRootMaker = Callable[[str], Path]


@dataclasses.dataclass
class Stage:
    """One ``run_tool`` call of a case and its golden files."""

    result: RunResult
    files: dict[str, str]


Build = Callable[[Path], dict[str, Stage]]
RunCase = Callable[[str, Build], dict[str, Stage]]
RunSingle = Callable[[str, Callable[[Path], Stage]], Stage]


def current_files(
    result: RunResult, paths: Iterable[Path], folder: str
) -> dict[str, str]:
    """Return the snapshot of the existing *paths* under ``<folder>/``.

    The files are read now, so a stage can record files that an earlier
    stage or the case itself created and this stage rewrote.
    """
    kept = sorted(path for path in paths if path.exists())
    if not kept:
        return {}
    current = snapshot(dataclasses.replace(result, created=kept))
    return {
        f"{folder}/" + name.removeprefix("files/"): content
        for name, content in current.items()
        if name.startswith("files/")
    }


def stage(result: RunResult, *, outputs: Iterable[Path] = ()) -> Stage:
    """Snapshot *result* and add the current content of *outputs*."""
    files = snapshot(result)
    files.update(current_files(result, outputs, "outputs"))
    return Stage(result, files)


def run_twice[S: Snapshotted](
    case_root: CaseRootMaker,
    build: Callable[[Path], Mapping[str, S]],
    *,
    before_each: Callable[[], object] | None = None,
) -> dict[str, S]:
    """Run *build* in two fresh roots and return the stages of the second.

    *before_each* runs before each build, for example to clear unit scripts
    the build sets. Fails when the snapshots of the two runs differ.
    """
    runs: list[Mapping[str, S]] = []
    for name in ("run1", "run2"):
        if before_each is not None:
            before_each()
        runs.append(build(case_root(name)))
    first, second = runs
    problems = [
        f"[{name}] {problem}"
        for name in sorted(set(first) | set(second))
        for problem in compare(
            first[name].files if name in first else {},
            second[name].files if name in second else {},
        )
    ]
    assert not problems, "two identical runs differ:\n" + "\n".join(problems)
    return dict(second)


def check_stages(
    prefix: str, stages: Mapping[str, Snapshotted], guard: WriteGuard
) -> None:
    """Compare every stage with ``golden/<prefix>/<stage>/``."""
    for name, current in stages.items():
        check_files(f"{prefix}/{name}", current.files, on_write=guard.consume)


def case_runner(
    prefix: str,
    case_root: CaseRootMaker,
    guard: WriteGuard,
    *,
    before_each: Callable[[], object] | None = None,
) -> RunCase:
    """Return ``run(case, build)``: run twice, then check every stage."""

    def run(case: str, build: Build) -> dict[str, Stage]:
        stages = run_twice(case_root, build, before_each=before_each)
        check_stages(f"{prefix}/{case}", stages, guard)
        return stages

    return run


def single_case_runner(
    prefix: str,
    case_root: CaseRootMaker,
    guard: WriteGuard,
    *,
    before_each: Callable[[], object] | None = None,
) -> RunSingle:
    """Return ``run(case, build)`` for one-stage cases checked at the case level."""

    def run(case: str, build: Callable[[Path], Stage]) -> Stage:
        stages = run_twice(
            case_root, lambda root: {"run": build(root)}, before_each=before_each
        )
        check_files(f"{prefix}/{case}", stages["run"].files, on_write=guard.consume)
        return stages["run"]

    return run


__all__ = [
    "Build",
    "RunCase",
    "RunSingle",
    "Snapshotted",
    "Stage",
    "case_runner",
    "check_stages",
    "current_files",
    "run_twice",
    "single_case_runner",
    "stage",
]
