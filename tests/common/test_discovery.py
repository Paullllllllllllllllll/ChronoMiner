"""Shared input discovery: the tree walk, the order and the --files names."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chronominer.common.discovery import (
    SKIPPED_FOLDERS,
    DiscoveryError,
    duplicated,
    image_folders,
    natural_key,
    ordered,
    relative_name,
    resolve_named,
    scan,
)

IMAGES = frozenset({".jpg", ".png"})


def _tree(root: Path, *files: str) -> None:
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def _names(paths: list[Path], root: Path) -> list[str]:
    return [relative_name(path, root) for path in paths]


def test_natural_key_compares_numbers_by_value_and_ignores_case() -> None:
    names = ["page_10.txt", "Page_2.txt", "page_1.txt", "b/1.txt", "a/12.txt"]
    assert sorted(names, key=natural_key) == [
        "a/12.txt",
        "b/1.txt",
        "page_1.txt",
        "Page_2.txt",
        "page_10.txt",
    ]


def test_scan_lists_only_the_files_of_the_folder_by_default(tmp_path: Path) -> None:
    _tree(tmp_path, "b.pdf", "a.pdf", "sub/c.pdf")
    assert _names(scan(tmp_path), tmp_path) == ["a.pdf", "b.pdf"]


def test_scan_depth_limits_the_levels_and_none_takes_all(tmp_path: Path) -> None:
    _tree(tmp_path, "a.pdf", "one/b.pdf", "one/two/c.pdf")
    assert _names(scan(tmp_path, depth=1), tmp_path) == ["a.pdf", "one/b.pdf"]
    assert _names(scan(tmp_path, depth=None), tmp_path) == [
        "a.pdf",
        "one/b.pdf",
        "one/two/c.pdf",
    ]


def test_scan_orders_by_the_relative_path_across_folders(tmp_path: Path) -> None:
    _tree(tmp_path, "vol_10/p.txt", "vol_2/p.txt", "z.txt", "Vol_3/p.txt")
    assert _names(scan(tmp_path, depth=None), tmp_path) == [
        "vol_2/p.txt",
        "Vol_3/p.txt",
        "vol_10/p.txt",
        "z.txt",
    ]


def test_scan_returns_the_files_as_ordered_sorts_them(tmp_path: Path) -> None:
    _tree(tmp_path, "1.txt", "01.txt", "a/01/p.txt", "a/1/p.txt", "A/2.txt", "b.txt")
    found = scan(tmp_path, depth=None)
    assert len(found) >= 5
    assert found == ordered(reversed(found), tmp_path, lambda path: path)


def test_scan_never_enters_skipped_folders(tmp_path: Path) -> None:
    skipped = next(iter(SKIPPED_FOLDERS))
    _tree(tmp_path, "a.txt", f"{skipped}/work.txt", f"sub/{skipped}/work.txt")
    assert _names(scan(tmp_path, depth=None), tmp_path) == ["a.txt"]


def test_scan_leaves_out_what_skip_names_with_everything_below(tmp_path: Path) -> None:
    _tree(tmp_path, "a.txt", "output/b.txt", "output/deep/c.txt", "sub/output/d.txt")
    seen: list[str] = []

    def skip(folder: str) -> bool:
        seen.append(folder)
        return folder == "output"

    found = scan(tmp_path, depth=None, skip=skip)

    assert _names(found, tmp_path) == ["a.txt", "sub/output/d.txt"]
    assert "output/deep" not in seen
    assert "sub/output" in seen


def test_scan_does_not_enter_a_linked_folder(tmp_path: Path) -> None:
    _tree(tmp_path, "real/a.txt")
    link = tmp_path / "root" / "link"
    link.parent.mkdir()
    try:
        link.symlink_to(tmp_path / "real", target_is_directory=True)
    except OSError:
        pytest.skip("directory links need a privilege on this system")
    _tree(tmp_path, "root/b.txt")
    assert _names(scan(tmp_path / "root", depth=None), tmp_path / "root") == ["b.txt"]


def test_scan_raises_for_a_root_that_cannot_be_listed(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        scan(tmp_path / "missing")


def test_ordered_breaks_ties_of_equal_keys_by_spelling(tmp_path: Path) -> None:
    paths = [tmp_path / "1.txt", tmp_path / "01.txt"]
    assert ordered(paths, tmp_path, lambda path: path) == [
        tmp_path / "01.txt",
        tmp_path / "1.txt",
    ]
    assert ordered(reversed(paths), tmp_path, lambda path: path) == [
        tmp_path / "01.txt",
        tmp_path / "1.txt",
    ]


def test_relative_name_falls_back_to_the_name(tmp_path: Path) -> None:
    assert relative_name(tmp_path / "a" / "b.pdf", tmp_path) == "a/b.pdf"
    assert relative_name(tmp_path, tmp_path) == tmp_path.name
    assert relative_name(Path("elsewhere") / "c.pdf", tmp_path) == "c.pdf"


def test_image_folders_groups_images_and_reports_nested_folders(
    tmp_path: Path,
) -> None:
    _tree(
        tmp_path,
        "book/p1.jpg",
        "book/p2.PNG",
        "book/notes.txt",
        "book/thumbs/t1.jpg",
        "shelf/other/p1.jpg",
        "loose.pdf",
    )
    found = image_folders(scan(tmp_path, depth=None), IMAGES)

    assert {relative_name(folder, tmp_path) for folder in found.folders} == {
        "book",
        "shelf/other",
    }
    assert [image.name for image in found.folders[tmp_path / "book"]] == [
        "p1.jpg",
        "p2.PNG",
    ]
    assert found.nested == ((tmp_path / "book" / "thumbs", tmp_path / "book"),)


def test_duplicated_pairs_an_item_with_the_first_of_its_key() -> None:
    inputs = [
        Path("a/doc.pdf"),
        Path("b/doc.pdf"),
        Path("b/other.pdf"),
        Path("doc.pdf"),
    ]

    pairs = duplicated(inputs, key=lambda path: path.name)

    assert pairs == [(inputs[1], inputs[0]), (inputs[3], inputs[0])]
    assert duplicated(inputs, key=str) == []


def test_resolve_named_joins_relative_names(tmp_path: Path) -> None:
    _tree(tmp_path, "a.pdf", "sub/b.pdf")
    assert resolve_named(tmp_path, ["sub/b.pdf", "a.pdf"], present=Path.is_file) == [
        tmp_path / "sub" / "b.pdf",
        tmp_path / "a.pdf",
    ]


def test_resolve_named_gives_one_path_for_two_spellings(tmp_path: Path) -> None:
    _tree(tmp_path, "a.pdf", "sub/b.pdf")
    names = ["sub\\b.pdf", "./a.pdf", "sub/../a.pdf", "sub/./b.pdf"]

    assert resolve_named(tmp_path, names, present=Path.is_file) == [
        tmp_path / "sub" / "b.pdf",
        tmp_path / "a.pdf",
    ]


def test_resolve_named_refuses_names_that_lead_out_of_the_root(tmp_path: Path) -> None:
    root = tmp_path / "in"
    _tree(root, "a.pdf")
    _tree(tmp_path, "out.pdf")
    outside = ["../out.pdf", "sub/../../out.pdf", str(tmp_path / "out.pdf")]

    with pytest.raises(DiscoveryError) as raised:
        resolve_named(root, ["a.pdf", *outside], present=Path.is_file)

    assert str(raised.value).endswith(", ".join(repr(name) for name in outside))


def test_resolve_named_refuses_a_name_with_a_root_or_a_drive(tmp_path: Path) -> None:
    rooted = ["/out.pdf", "\\out.pdf"]
    if os.name == "nt":
        rooted.append(f"{tmp_path.drive}out.pdf")

    with pytest.raises(DiscoveryError) as raised:
        resolve_named(tmp_path, ["a.pdf", *rooted], present=lambda path: True)

    assert str(raised.value).endswith(", ".join(repr(name) for name in rooted))


def test_resolve_named_names_every_missing_file(tmp_path: Path) -> None:
    _tree(tmp_path, "a.pdf")
    (tmp_path / "folder").mkdir()
    with pytest.raises(DiscoveryError, match=r"'b\.pdf', 'folder'") as raised:
        resolve_named(tmp_path, ["a.pdf", "b.pdf", "folder"], present=Path.is_file)
    assert "do not exist" in str(raised.value)
    with pytest.raises(DiscoveryError, match="are not folders"):
        resolve_named(
            tmp_path, ["a.pdf"], present=Path.is_dir, missing="are not folders"
        )
