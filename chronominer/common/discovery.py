"""Input discovery shared by the tools: one tree walk, one order, one name rule.

:func:`scan` lists the files below an input folder with one ``os.scandir``
call per folder. It never enters :data:`SKIPPED_FOLDERS` or a folder the
caller's *skip* names, follows no directory link, and returns the files in
the natural order of their paths relative to the input folder
(:func:`natural_key`). A run processes its items in that order and
``--select`` numbers them in it (:func:`ordered`, :func:`relative_name`).

What an item is stays the tool's decision: a tool filters the files by
extension, or groups page images by their folder with
:func:`image_folders`. :func:`resolve_named` turns the names of ``--files``,
paths relative to the input folder, into paths.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "SKIPPED_FOLDERS",
    "DiscoveryError",
    "ImageFolders",
    "duplicated",
    "image_folders",
    "natural_key",
    "ordered",
    "relative_name",
    "resolve_named",
    "scan",
]

SKIPPED_FOLDERS: Final = frozenset({".mabillon"})
"""Folders no scan enters: Mabillon's working folders below an output root."""

_DIGITS = re.compile(r"(\d+)")


class DiscoveryError(ValueError):
    """The input folder cannot serve the items asked of it."""


def natural_key(text: str) -> tuple[str | int, ...]:
    """Return a sort key that compares digit runs by value and letters without case.

    ``page_2`` sorts before ``page_10``. Applied to a relative path, the key
    covers the whole path, folders included.
    """
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in _DIGITS.split(text)
    )


def relative_name(path: Path, root: Path) -> str:
    """Return *path* below the input *root* with forward slashes.

    This is the name ``--select`` searches and ``--files`` takes. A path
    outside *root*, and *root* itself when it is the one input, gives the
    path's own name.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.name
    return relative.as_posix() if relative.parts else path.name


def ordered[T](items: Iterable[T], root: Path, path: Callable[[T], Path]) -> list[T]:
    """Return *items* in the natural order of their paths below *root*.

    Names with equal keys (``01`` and ``1``) keep the order of their
    spelling, so the result does not depend on the order found.
    """

    def key(item: T) -> tuple[tuple[str | int, ...], str]:
        name = relative_name(path(item), root)
        return natural_key(name), name

    return sorted(items, key=key)


def scan(
    root: Path,
    *,
    depth: int | None = 0,
    skip: Callable[[str], bool] | None = None,
) -> list[Path]:
    """Return the files in *root* and in its folders down to *depth* levels.

    *depth* 0 lists the files directly in *root*, 1 also those of its direct
    subfolders, None every level. *skip* receives the path of a folder
    relative to *root* (forward slashes) and returns True to leave the
    folder and everything below it out; :data:`SKIPPED_FOLDERS` are always
    left out. Directory links and junctions are not entered. A subfolder
    that cannot be listed is left out; the files come in the order of
    :func:`ordered`.

    Raises:
        OSError: *root* cannot be listed.
    """
    found: list[tuple[str, Path]] = []
    pending: list[tuple[Path, str, int]] = [(root, "", 0)]
    while pending:
        folder, prefix, level = pending.pop()
        try:
            with os.scandir(folder) as entries:
                listing = list(entries)
        except OSError:
            if level == 0:
                raise
            continue
        for entry in listing:
            if entry.is_dir(follow_symlinks=False) and not entry.is_junction():
                if depth is not None and level >= depth:
                    continue
                name = f"{prefix}{entry.name}"
                if entry.name in SKIPPED_FOLDERS or (skip is not None and skip(name)):
                    continue
                pending.append((Path(entry.path), f"{name}/", level + 1))
            elif entry.is_file():
                found.append((f"{prefix}{entry.name}", Path(entry.path)))
    found.sort(key=lambda named: (natural_key(named[0]), named[0]))
    return [path for _, path in found]


@dataclass(frozen=True)
class ImageFolders:
    """The image folders among scanned files.

    *folders* maps each folder that directly holds a page image to its
    images, in the order given. *nested* lists every image folder that lies
    below another one, with that folder: a parent's pages do not include a
    subfolder's images, so a tool leaves the nested folder out and says so.
    """

    folders: Mapping[Path, tuple[Path, ...]]
    nested: tuple[tuple[Path, Path], ...] = ()


def image_folders(files: Iterable[Path], extensions: Collection[str]) -> ImageFolders:
    """Group the page images among *files* by the folder that holds them.

    *extensions* are the lower-case suffixes of page images, dot included.
    """
    grouped: dict[Path, list[Path]] = {}
    for file in files:
        if file.suffix.lower() in extensions:
            grouped.setdefault(file.parent, []).append(file)
    kept: dict[Path, tuple[Path, ...]] = {}
    nested: list[tuple[Path, Path]] = []
    for folder, images in grouped.items():
        above = next((parent for parent in folder.parents if parent in grouped), None)
        if above is None:
            kept[folder] = tuple(images)
        else:
            nested.append((folder, above))
    return ImageFolders(kept, tuple(nested))


def duplicated[T](items: Iterable[T], key: Callable[[T], object]) -> list[tuple[T, T]]:
    """Return each item whose *key* an earlier item has, with that earlier item.

    A tool refuses a run in which two inputs would write one output file or
    take one document name; *key* returns that path or name.
    """
    first: dict[object, T] = {}
    pairs: list[tuple[T, T]] = []
    for item in items:
        earlier = first.setdefault(key(item), item)
        if earlier is not item:
            pairs.append((item, earlier))
    return pairs


def resolve_named(
    root: Path,
    names: Sequence[str],
    *,
    present: Callable[[Path], bool] = Path.exists,
    missing: str = "do not exist",
) -> list[Path]:
    """Return the paths the ``--files`` *names* give below the input *root*.

    A name is a path relative to *root*, with forward or back slashes.
    ``.`` and ``..`` segments are resolved in the name itself, so two
    spellings of one path give one path; a name that leads out of *root*
    fits no item. *present* tells whether a path can be an item (a file, or
    a folder for image folders); *missing* words the error for the names
    that cannot. The paths come in the order of their first name.

    Raises:
        DiscoveryError: Naming every name that is outside *root* or not
            *present*.
    """
    paths: list[Path] = []
    absent: list[str] = []
    for name in names:
        relative = os.path.normpath(name.replace("\\", "/"))
        # A root or a drive alone (``\name``, ``C:name``) is not absolute on
        # Windows, yet joining it to the root replaces the root.
        outside = bool(Path(relative).anchor) or relative.split(os.sep)[0] == ".."
        path = root if relative == "." else root / relative
        if outside or not present(path):
            absent.append(name)
        elif path not in paths:
            paths.append(path)
    if absent:
        listed = ", ".join(repr(name) for name in absent)
        raise DiscoveryError(f"--files names that {missing} in {root}: {listed}")
    return paths
