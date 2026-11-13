"""Input discovery and type detection; output paths and Windows-safe names.

:func:`discover_items` finds the text (.txt, .md), image and PDF files a run
processes; each file is one item. A folder in auto mode yields every kind
(a mixed folder). Files the tool writes or reads beside an input
(:func:`is_sidecar`) are never inputs. :func:`item_paths` derives where an
item writes: beside the input, flat in the output folder, or in the mirrored
input tree.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from chronominer.common.discovery import (
    DiscoveryError,
    ordered,
    relative_name,
    resolve_named,
    scan,
)
from chronominer.llm.media import SUPPORTED_IMAGE_EXTENSIONS
from chronominer.output.store import DATABASE_NAME, LINE_RANGES_SUFFIX

__all__ = [
    "DATABASE_NAME",
    "EXTENSIONS",
    "HASH_LENGTH",
    "INPUT_TYPES",
    "MAX_SAFE_NAME_LENGTH",
    "MIN_READABLE_LENGTH",
    "WINDOWS_MAX_PATH",
    "DiscoveryError",
    "ItemPaths",
    "OutputMode",
    "SourceItem",
    "create_safe_directory_name",
    "create_safe_filename",
    "detect_input_type",
    "discover_items",
    "is_sidecar",
    "item_paths",
    "mirror_output_path",
    "output_names",
    "prepare_item_paths",
    "source_type",
]

INPUT_TYPES = ("text", "image", "pdf")

EXTENSIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "text": (".txt", ".md"),
        "image": tuple(sorted(SUPPORTED_IMAGE_EXTENSIONS)),
        "pdf": (".pdf",),
    }
)
"""File extensions of each input type."""

OUTPUT_SUFFIX = "_output"

_SIDECAR_STEMS = ("_extract_context", "_adjust_context")
_SIDECAR_ENDINGS = (
    LINE_RANGES_SUFFIX,
    f"{OUTPUT_SUFFIX}.json",
    f"{OUTPUT_SUFFIX}.csv",
)
_OUTPUT_FOLDER = "output"
_OUTPUTS_FOLDER_SUFFIX = "_outputs"


@dataclass(frozen=True)
class SourceItem:
    """One input file and its type: text, image or pdf."""

    path: Path
    type: str

    @property
    def name(self) -> str:
        """The item's file name."""
        return self.path.name


def source_type(path: Path) -> str | None:
    """Return the input type of *path* by its suffix, or None."""
    suffix = path.suffix.lower()
    for kind, extensions in EXTENSIONS.items():
        if suffix in extensions:
            return kind
    return None


def is_sidecar(path: Path) -> bool:
    """Whether *path* is a file the tool writes or reads beside an input.

    Sidecars are the context files ``{stem}_extract_context.*`` and
    ``{stem}_adjust_context.*``, ``{stem}_line_ranges.txt``, the outputs
    ``{stem}_output.json`` and ``.csv``, and the database and its journal.
    """
    name = path.name.lower()
    if name == DATABASE_NAME or name.startswith(DATABASE_NAME + "-"):
        return True
    if Path(name).stem.endswith(_SIDECAR_STEMS):
        return True
    return name.endswith(_SIDECAR_ENDINGS)


def _output_folder(name: str) -> bool:
    """Whether *name*, a folder below the input, is a top-level output folder."""
    if "/" in name:
        return False
    top = name.lower()
    return top == _OUTPUT_FOLDER or top.endswith(_OUTPUTS_FOLDER_SUFFIX)


def _walk(folder: Path, recursive: bool) -> list[Path]:
    """Return the input files in *folder*, its subfolders with *recursive*.

    The files come in the natural order of their paths below *folder*.
    """
    files = scan(folder, depth=None if recursive else 0, skip=_output_folder)
    return [path for path in files if not is_sidecar(path)]


_TYPE_ITEMS: Mapping[str, str] = MappingProxyType({"images": "image"})
"""The item type of each ``--type`` value that differs from it."""


def _accepts(kind: str) -> Callable[[str | None], bool]:
    if kind == "auto":
        return lambda found: found is not None
    wanted = _TYPE_ITEMS.get(kind, kind)
    return lambda found: found == wanted


def discover_items(
    input_path: Path,
    kind: str = "auto",
    *,
    files: Sequence[str] = (),
    recursive: bool = False,
) -> list[SourceItem]:
    """Return the items of *kind* (auto, text, images or pdf) at *input_path*.

    *kind* is a ``--type`` value; the item types are text, image and pdf,
    and ``image`` also names the images.

    A file input is the one item; its type comes from its suffix, and a file
    of no known type is text. A folder yields the named *files* inside it,
    else its input files (all subfolders with *recursive*, except top-level
    ``output`` and ``*_outputs`` folders and Mabillon's ``.mabillon``
    working folders); either way in natural order of their relative paths.
    Sidecars are never items.

    Raises:
        DiscoveryError: A file input or a named file whose type is not
            *kind*, a sidecar named as input, a named file missing from or
            outside the input folder, or an input that does not exist.
    """
    accepts = _accepts(kind)
    if input_path.is_file():
        if is_sidecar(input_path):
            raise DiscoveryError(f"{input_path.name} is a sidecar file, not an input")
        detected = source_type(input_path) or "text"
        if not accepts(detected):
            raise DiscoveryError(
                f"--type {kind} does not fit {input_path.name} ({detected})"
            )
        return [SourceItem(input_path, detected)]
    if not input_path.is_dir():
        raise DiscoveryError(f"input not found: {input_path}")
    if files:
        return _named(input_path, files, accepts, kind)
    items: list[SourceItem] = []
    for path in _walk(input_path, recursive):
        found = source_type(path)
        if found is not None and accepts(found):
            items.append(SourceItem(path, found))
    return items


def _named(
    folder: Path,
    files: Sequence[str],
    accepts: Callable[[str | None], bool],
    kind: str,
) -> list[SourceItem]:
    """Return the items of the named *files* in *folder*.

    A name is a path relative to *folder*, with forward or back slashes.
    Two names of one file give one item.

    Raises:
        DiscoveryError: Naming the files that lie outside *folder*, are
            missing, sidecars or not of *kind*.
    """
    names = [name.replace("\\", "/") for name in files]
    outside = [
        given
        for given, name in zip(files, names, strict=True)
        if not Path(os.path.normpath(folder / name)).is_relative_to(folder)
    ]
    if outside:
        listed = ", ".join(repr(name) for name in outside)
        raise DiscoveryError(f"--files names paths outside {folder}: {listed}")
    items: list[SourceItem] = []
    unfit: list[str] = []
    for path in resolve_named(folder, names, present=Path.is_file):
        found = source_type(path)
        if is_sidecar(path) or not accepts(found):
            unfit.append(relative_name(path, folder))
        else:
            items.append(SourceItem(path, found or "text"))
    if unfit:
        listed = ", ".join(repr(name) for name in unfit)
        wanted = "inputs" if kind == "auto" else f"inputs of --type {kind}"
        raise DiscoveryError(f"--files names that are not {wanted}: {listed}")
    return ordered(items, folder, lambda item: item.path)


def detect_input_type(items: Sequence[SourceItem]) -> str | None:
    """Return the type shared by *items*, ``mixed``, or None without items."""
    kinds = {item.type for item in items}
    if not kinds:
        return None
    return kinds.pop() if len(kinds) == 1 else "mixed"


WINDOWS_MAX_PATH = 260
MAX_SAFE_NAME_LENGTH = 80
HASH_LENGTH = 8
MIN_READABLE_LENGTH = 20


class OutputMode(StrEnum):
    """Output directory layout mode."""

    FLAT = "flat"
    MIRROR = "mirror"


def mirror_output_path(item: Path, input_root: Path, output_root: Path) -> Path:
    """Return the mirrored output folder of the file *item*.

    Replicates the folder of *item* relative to *input_root* (a folder, or
    the single input file) under *output_root*; an item outside the root
    writes into *output_root* itself.
    """
    root = input_root if input_root.is_dir() else input_root.parent
    try:
        relative = item.parent.relative_to(root)
    except ValueError:
        relative = Path()
    return output_root / relative


def _compute_name_hash(name: str) -> str:
    """Return an 8-character SHA-256 hex digest of *name*."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def _truncate_name(name: str, max_length: int) -> str:
    """Truncate *name* to *max_length*, stripping trailing punctuation."""
    if len(name) <= max_length:
        return name
    return name[:max_length].rstrip(".-_ ")


def create_safe_directory_name(original_name: str, suffix: str = "") -> str:
    """Create a safe directory name that won't exceed Windows path limits."""
    name_hash = _compute_name_hash(original_name)

    reserved_length = len(suffix) + HASH_LENGTH + 1  # +1 for the dash
    available_length = MAX_SAFE_NAME_LENGTH - reserved_length

    truncated = _truncate_name(original_name, available_length)
    safe_name = f"{truncated}-{name_hash}{suffix}"

    return safe_name


def create_safe_filename(
    original_name: str,
    extension: str,
    parent_path: Path | None = None,
) -> str:
    """Create a safe filename that won't exceed Windows path limits."""
    name_hash = _compute_name_hash(original_name)

    if parent_path is not None:
        parent_len = len(str(parent_path)) + 1  # +1 for path separator
        max_filename_len = WINDOWS_MAX_PATH - parent_len - 10  # -10 safety margin
        max_filename_len = max(
            MIN_READABLE_LENGTH + HASH_LENGTH + len(extension) + 1,
            min(max_filename_len, MAX_SAFE_NAME_LENGTH),
        )
    else:
        max_filename_len = MAX_SAFE_NAME_LENGTH

    reserved_length = len(extension) + HASH_LENGTH + 1  # +1 for the dash
    available_length = max_filename_len - reserved_length

    if len(original_name) > available_length:
        truncated = _truncate_name(
            original_name, max(MIN_READABLE_LENGTH, available_length)
        )
        safe_name = f"{truncated}-{name_hash}{extension}"
    else:
        safe_name = f"{original_name}{extension}"

    return safe_name


def output_names(stem: str) -> tuple[str, str]:
    """Return the names of an item's JSON and CSV outputs.

    The names keep the stem whole (``{stem}_output.json``), as the readers of
    the outputs expect.
    """
    return f"{stem}{OUTPUT_SUFFIX}.json", f"{stem}{OUTPUT_SUFFIX}.csv"


@dataclass(frozen=True)
class ItemPaths:
    """Where one input item writes.

    *folder* holds the item's outputs and the database of that output
    folder; *output* is ``{stem}_output.json`` and *csv* ``{stem}_output.csv``.
    """

    folder: Path
    output: Path
    csv: Path


def item_paths(
    item: Path,
    *,
    output_dir: Path | None,
    layout: str,
    input_root: Path | None,
) -> ItemPaths:
    """Return the paths of *item*; nothing is created.

    *output_dir* None writes beside the input. The flat layout writes every
    output into *output_dir*; the mirror layout writes into the input tree
    replicated below *output_dir* (an item outside *input_root*, or a
    single-file input, writes into *output_dir* itself).
    """
    if output_dir is None:
        folder = item.parent
    elif layout == OutputMode.MIRROR and input_root is not None:
        folder = mirror_output_path(item, input_root, output_dir)
    else:
        folder = output_dir
    json_name, csv_name = output_names(item.stem)
    return ItemPaths(folder=folder, output=folder / json_name, csv=folder / csv_name)


def prepare_item_paths(
    item: Path,
    *,
    output_dir: Path | None,
    layout: str,
    input_root: Path | None,
) -> ItemPaths:
    """Return :func:`item_paths` after creating the folder of the output.

    No file is created.
    """
    paths = item_paths(
        item, output_dir=output_dir, layout=layout, input_root=input_root
    )
    paths.folder.mkdir(parents=True, exist_ok=True)
    return paths
