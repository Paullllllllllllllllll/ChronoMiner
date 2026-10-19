"""Input discovery: input types, mixed folders, sidecar rules and --files."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronominer.discovery import (
    DiscoveryError,
    SourceItem,
    detect_input_type,
    discover_items,
    is_sidecar,
    natural_sort_key,
    source_type,
)
from chronominer.spec import parse


def _touch(folder: Path, *names: str) -> None:
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _names(items: list[SourceItem]) -> list[str]:
    return [item.name for item in items]


@pytest.mark.unit
class TestTypes:
    @pytest.mark.parametrize(
        ("name", "kind"),
        [
            ("a.txt", "text"),
            ("a.MD", "text"),
            ("a.pdf", "pdf"),
            ("a.PNG", "image"),
            ("a.jpeg", "image"),
            ("a.tif", "image"),
            ("a.webp", "image"),
            ("a.docx", None),
            ("a.json", None),
        ],
    )
    def test_the_suffix_names_the_type(self, name: str, kind: str | None) -> None:
        assert source_type(Path(name)) == kind

    def test_an_empty_folder_has_no_items(self, temp_dir: Path) -> None:
        assert discover_items(temp_dir) == []
        assert detect_input_type([]) is None

    def test_a_mixed_folder_yields_every_type_in_natural_order(
        self, temp_dir: Path
    ) -> None:
        _touch(
            temp_dir, "p10.png", "p2.png", "ledger.pdf", "notes.md", "a.txt", "x.doc"
        )

        items = discover_items(temp_dir)

        assert _names(items) == ["a.txt", "ledger.pdf", "notes.md", "p2.png", "p10.png"]
        assert [item.type for item in items] == [
            "text",
            "pdf",
            "text",
            "image",
            "image",
        ]
        assert detect_input_type(items) == "mixed"

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("text", ["a.txt", "notes.md"]),
            ("image", ["p2.png"]),
            ("pdf", ["ledger.pdf"]),
        ],
    )
    def test_an_input_type_keeps_only_its_files(
        self, temp_dir: Path, kind: str, expected: list[str]
    ) -> None:
        _touch(temp_dir, "p2.png", "ledger.pdf", "notes.md", "a.txt")

        items = discover_items(temp_dir, kind)

        assert _names(items) == expected
        assert detect_input_type(items) == kind

    def test_a_single_file_is_the_one_item(self, temp_dir: Path) -> None:
        _touch(temp_dir, "scan.jpg", "list.csv")
        assert discover_items(temp_dir / "scan.jpg") == [
            SourceItem(temp_dir / "scan.jpg", "image")
        ]
        assert discover_items(temp_dir / "list.csv", "text")[0].type == "text"

    def test_a_single_file_of_another_type_is_an_error(self, temp_dir: Path) -> None:
        _touch(temp_dir, "scan.jpg")
        with pytest.raises(DiscoveryError, match="--type pdf"):
            discover_items(temp_dir / "scan.jpg", "pdf")

    def test_a_missing_input_is_an_error(self, temp_dir: Path) -> None:
        with pytest.raises(DiscoveryError, match="not found"):
            discover_items(temp_dir / "nothing")


@pytest.mark.unit
class TestRecursion:
    def test_subfolders_need_recursive(self, temp_dir: Path) -> None:
        _touch(temp_dir, "top.txt", "vol1/p1.png", "vol1/deep/p2.pdf")

        assert _names(discover_items(temp_dir)) == ["top.txt"]
        found = discover_items(temp_dir, recursive=True)
        assert [item.path.relative_to(temp_dir).as_posix() for item in found] == [
            "top.txt",
            "vol1/deep/p2.pdf",
            "vol1/p1.png",
        ]

    def test_top_level_output_folders_are_skipped(self, temp_dir: Path) -> None:
        _touch(
            temp_dir,
            "a.txt",
            "output/b.txt",
            "run_outputs/c.txt",
            "vol/output/d.txt",
        )
        found = discover_items(temp_dir, recursive=True)
        assert _names(found) == ["a.txt", "d.txt"]


SIDECARS = [
    "doc_extract_context.txt",
    "doc_extract_context.png",
    "doc_extract_context.jpg",
    "vol1_extract_context.txt",
    "doc_adjust_context.txt",
    "doc_line_ranges.txt",
    "doc_output.json",
    "doc_output.csv",
    "chronominer.sqlite",
    "chronominer.sqlite-journal",
]


@pytest.mark.unit
class TestSidecars:
    @pytest.mark.parametrize("name", SIDECARS)
    def test_tool_files_are_sidecars(self, name: str) -> None:
        assert is_sidecar(Path(name))

    @pytest.mark.parametrize(
        "name", ["doc.txt", "context.txt", "output.json", "doc_lines.txt", "p1.png"]
    )
    def test_inputs_are_not_sidecars(self, name: str) -> None:
        assert not is_sidecar(Path(name))

    def test_sidecars_are_never_items(self, temp_dir: Path) -> None:
        _touch(temp_dir, "doc.txt", "doc.png", *SIDECARS)
        assert _names(discover_items(temp_dir)) == ["doc.png", "doc.txt"]

    def test_a_sidecar_given_as_input_is_an_error(self, temp_dir: Path) -> None:
        _touch(temp_dir, "doc_line_ranges.txt")
        with pytest.raises(DiscoveryError, match="sidecar"):
            discover_items(temp_dir / "doc_line_ranges.txt")


@pytest.mark.unit
class TestNamedFiles:
    def test_files_select_inputs_of_the_folder(self, temp_dir: Path) -> None:
        _touch(temp_dir, "a.txt", "b.pdf", "c.png")
        found = discover_items(temp_dir, files=("c.png", "a.txt"))
        assert found == [
            SourceItem(temp_dir / "c.png", "image"),
            SourceItem(temp_dir / "a.txt", "text"),
        ]

    def test_missing_files_are_named(self, temp_dir: Path) -> None:
        _touch(temp_dir, "a.txt")
        with pytest.raises(DiscoveryError, match="'b.pdf'"):
            discover_items(temp_dir, files=("a.txt", "b.pdf"))

    def test_files_of_another_type_or_sidecars_are_named(self, temp_dir: Path) -> None:
        _touch(temp_dir, "a.txt", "a_output.json", "c.png")
        with pytest.raises(DiscoveryError, match="not inputs of --type text: 'c.png'"):
            discover_items(temp_dir, "text", files=("a.txt", "c.png"))
        with pytest.raises(DiscoveryError, match="'a_output.json'"):
            discover_items(temp_dir, files=("a_output.json",))

    def test_repeated_files_flags_accumulate_into_one_selection(
        self, temp_dir: Path
    ) -> None:
        _touch(temp_dir, "a.txt", "b.pdf", "c.png")
        spec = parse(
            [
                "--input",
                str(temp_dir),
                "--schema",
                "WageRecords",
                "--files",
                "a.txt",
                "--files",
                "c.png,a.txt",
            ]
        )

        found = discover_items(
            spec.input.path,
            spec.input.type,
            files=spec.input.files,
            recursive=spec.input.recursive,
        )

        assert _names(found) == ["a.txt", "c.png"]


@pytest.mark.unit
def test_natural_sort_orders_digit_runs_by_value() -> None:
    names = ["page_10.png", "Page_2.png", "page_1.png"]
    assert sorted(names, key=natural_sort_key) == [
        "page_1.png",
        "Page_2.png",
        "page_10.png",
    ]
