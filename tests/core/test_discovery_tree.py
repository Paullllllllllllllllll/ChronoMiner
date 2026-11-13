"""Discovery on a nested input tree: the items, their order, names and outputs.

The tree mixes input types, sidecars, the tool's output folders, Mabillon's
working folders and names whose natural order differs from their plain
order. Document names and output paths are what existing stores resume by.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

import chronominer.run as core
from chronominer.spec import resolve_ranges_generate
from chronominer.ui.session import take_inventory
from tests.conftest import make_settings, make_spec

TREE = (
    "a.txt",
    "B.txt",
    "p10.png",
    "p2.png",
    "p1.png",
    "p01.png",
    "ledger.pdf",
    "notes.md",
    "x.doc",
    "doc_output.json",
    "doc_output.csv",
    "doc_line_ranges.txt",
    "doc_extract_context.txt",
    "doc_adjust_context.txt",
    "chronominer.sqlite-journal",
    "vol2/p1.png",
    "vol10/p1.png",
    "vol2/deep/z.txt",
    "vol10/deep/9.md",
    "output/o.txt",
    "Run_Outputs/r.txt",
    "vol2/output/kept.txt",
    ".mabillon/doc/m.txt",
    "vol10/.mabillon/w/p.png",
    ".hidden/h.txt",
)

TOP = [
    "a.txt",
    "B.txt",
    "ledger.pdf",
    "notes.md",
    "p01.png",
    "p1.png",
    "p2.png",
    "p10.png",
]

DEEP = [
    ".hidden/h.txt",
    "a.txt",
    "B.txt",
    "ledger.pdf",
    "notes.md",
    "p01.png",
    "p1.png",
    "p2.png",
    "p10.png",
    "vol2/deep/z.txt",
    "vol2/output/kept.txt",
    "vol2/p1.png",
    "vol10/deep/9.md",
    "vol10/p1.png",
]

DEEP_TEXT = [
    ".hidden/h.txt",
    "a.txt",
    "B.txt",
    "notes.md",
    "vol2/deep/z.txt",
    "vol2/output/kept.txt",
    "vol10/deep/9.md",
]


@pytest.fixture(autouse=True)
def openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable of the default provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """The input folder of :data:`TREE`, every file one byte long."""
    root = tmp_path / "in"
    for name in TREE:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return root


def _relative(paths: Any, root: Path) -> list[str]:
    return [Path(getattr(p, "path", p)).relative_to(root).as_posix() for p in paths]


def _found(root: Path, **flags: Any) -> list[str]:
    spec = make_spec(root, **flags)
    return _relative(core.discover_inputs(spec.input, root)[0], root)


@pytest.mark.parametrize(
    ("kind", "recursive", "expected"),
    [
        ("auto", False, TOP),
        ("text", False, ["a.txt", "B.txt", "notes.md"]),
        ("images", False, ["p01.png", "p1.png", "p2.png", "p10.png"]),
        ("pdf", False, ["ledger.pdf"]),
        ("auto", True, DEEP),
        ("text", True, DEEP_TEXT),
        (
            "images",
            True,
            ["p01.png", "p1.png", "p2.png", "p10.png", "vol2/p1.png", "vol10/p1.png"],
        ),
        ("pdf", True, ["ledger.pdf"]),
    ],
)
def test_the_items_found_and_their_order(
    tree: Path, kind: str, recursive: bool, expected: list[str]
) -> None:
    assert _found(tree, type=kind, recursive=recursive) == expected


@pytest.mark.parametrize("kind", ["auto", "text", "images", "pdf"])
def test_recursive_reads_every_level_and_the_default_one(
    tmp_path: Path, kind: str
) -> None:
    root = tmp_path / "in"
    deep = ["1/2/3/4/a.txt", "1/2/3/4/b.png", "1/2/3/4/c.pdf"]
    for name in ["top.txt", "top.png", "top.pdf", *deep]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"x")
    wanted = {"auto": "", "text": ".txt", "images": ".png", "pdf": ".pdf"}[kind]

    flat = _found(root, type=kind)
    every = _found(root, type=kind, recursive=True)

    assert all("/" not in name for name in flat)
    assert all(name.endswith(wanted) for name in flat)
    assert [name for name in every if "/" in name] == [
        name for name in deep if name.endswith(wanted)
    ]


def _located(tree: Path, **flags: Any) -> list[tuple[str, str]]:
    run_plan = core.plan(make_spec(tree, dry_run=True, **flags), make_settings())
    base = tree.parent
    return [
        (item.document, item.output.relative_to(base).as_posix())
        for item in run_plan.to_process
    ]


def test_document_names_and_outputs_flat(tree: Path) -> None:
    located = _located(tree, output=tree.parent / "flat", layout="flat")
    assert located == [
        (Path(name).stem, f"flat/{Path(name).stem}_output.json") for name in TOP
    ]


def test_document_names_and_outputs_mirrored(tree: Path) -> None:
    located = _located(
        tree, recursive=True, output=tree.parent / "out", layout="mirror"
    )
    assert located == [
        (name.rsplit(".", 1)[0], f"out/{name.rsplit('.', 1)[0]}_output.json")
        for name in DEEP
    ]


def test_document_names_and_outputs_beside_the_inputs(tree: Path) -> None:
    located = _located(tree, recursive=True)
    assert located == [
        (Path(name).stem, f"in/{name.rsplit('.', 1)[0]}_output.json") for name in DEEP
    ]


def test_select_numbers_and_searches_the_natural_order(tree: Path) -> None:
    assert _found(tree, recursive=True, select="3,11") == [
        "B.txt",
        "vol2/output/kept.txt",
    ]
    assert _found(tree, recursive=True, select="vol10") == [
        "vol10/deep/9.md",
        "vol10/p1.png",
    ]


def test_select_searches_the_name_of_a_file_input(tree: Path) -> None:
    file = tree / "notes.md"
    spec = make_spec(file, select="notes")
    assert core.discover_inputs(spec.input, file)[0][0].path == file


def test_ranges_commands_find_the_text_inputs(tree: Path) -> None:
    spec = resolve_ranges_generate({"input": tree, "recursive": True}).spec
    files = core.plan_ranges(spec, make_settings()).files
    assert _relative(files, tree) == DEEP_TEXT
    flat = resolve_ranges_generate({"input": tree}).spec
    assert _relative(core.plan_ranges(flat, make_settings()).files, tree) == [
        "a.txt",
        "B.txt",
        "notes.md",
    ]


def test_the_wizard_lists_the_items_by_type(tree: Path) -> None:
    items = take_inventory(tree, recursive=True).items
    assert _relative(items, tree) == [
        *DEEP_TEXT,
        "ledger.pdf",
        "p01.png",
        "p1.png",
        "p2.png",
        "p10.png",
        "vol2/p1.png",
        "vol10/p1.png",
    ]


def _link(target: Path, link: Path) -> None:
    """Create a directory link: a junction on Windows, else a symbolic link."""
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def test_a_directory_link_is_not_entered(tmp_path: Path) -> None:
    root = tmp_path / "in"
    root.mkdir()
    (root / "a.txt").write_bytes(b"x")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "l.txt").write_bytes(b"x")
    _link(elsewhere, root / "linked")

    assert _found(root, recursive=True) == ["a.txt"]
