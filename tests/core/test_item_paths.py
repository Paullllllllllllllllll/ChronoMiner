"""Item output paths: beside the input, the flat folder and the mirror layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronominer.discovery import item_paths, output_names, prepare_item_paths


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "in" / "sub" / "doc.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"%PDF")
    return path


@pytest.mark.unit
def test_the_flat_layout_writes_every_output_into_the_output_folder(
    tmp_path: Path, pdf: Path
) -> None:
    out = tmp_path / "out"
    paths = item_paths(pdf, output_dir=out, layout="flat", input_root=tmp_path / "in")

    assert paths.folder == out
    assert paths.output == out / "doc_output.json"
    assert paths.csv == out / "doc_output.csv"


@pytest.mark.unit
def test_beside_the_input_the_output_sits_next_to_it(pdf: Path) -> None:
    paths = item_paths(pdf, output_dir=None, layout="mirror", input_root=pdf.parent)

    assert paths.folder == pdf.parent
    assert paths.output == pdf.parent / "doc_output.json"


@pytest.mark.unit
def test_the_mirror_layout_replicates_the_input_tree(tmp_path: Path, pdf: Path) -> None:
    out = tmp_path / "out"
    paths = item_paths(pdf, output_dir=out, layout="mirror", input_root=tmp_path / "in")

    assert paths.folder == out / "sub"
    assert paths.output.name == "doc_output.json"


@pytest.mark.unit
def test_a_single_file_input_mirrors_into_the_output_folder(
    tmp_path: Path, pdf: Path
) -> None:
    out = tmp_path / "out"
    paths = item_paths(pdf, output_dir=out, layout="mirror", input_root=pdf)
    assert paths.folder == out


@pytest.mark.unit
def test_an_item_outside_the_input_root_mirrors_into_the_output_folder(
    tmp_path: Path, pdf: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    out = tmp_path / "out"
    paths = item_paths(pdf, output_dir=out, layout="mirror", input_root=other)
    assert paths.folder == out


@pytest.mark.unit
def test_long_stems_stay_whole_in_output_names() -> None:
    stem = "Annual Price Register 1931 " * 6
    assert output_names(stem) == (f"{stem}_output.json", f"{stem}_output.csv")


@pytest.mark.unit
def test_prepare_creates_only_the_output_folder(tmp_path: Path, pdf: Path) -> None:
    paths = prepare_item_paths(
        pdf, output_dir=tmp_path / "out", layout="mirror", input_root=tmp_path / "in"
    )
    assert paths.folder.is_dir()
    assert list(paths.folder.iterdir()) == []


@pytest.mark.unit
@pytest.mark.parametrize("layout", ["flat", "mirror"])
def test_prepare_writes_no_file_in_any_layout(
    tmp_path: Path, pdf: Path, layout: str
) -> None:
    out = tmp_path / "out"
    prepare_item_paths(pdf, output_dir=out, layout=layout, input_root=pdf.parent)
    assert [path for path in out.rglob("*") if path.is_file()] == []


@pytest.mark.unit
def test_beside_the_input_prepare_creates_nothing(pdf: Path) -> None:
    before = sorted(pdf.parent.iterdir())
    prepare_item_paths(pdf, output_dir=None, layout="flat", input_root=pdf.parent)
    assert sorted(pdf.parent.iterdir()) == before
