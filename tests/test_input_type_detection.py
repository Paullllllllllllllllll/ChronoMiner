"""Tests for input type detection in modules/cli/args_parser.py."""

from pathlib import Path

from main.cli_args import detect_input_type, get_files_from_path


class TestDetectInputType:
    def test_text_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        assert detect_input_type(f) == "text"

    def test_png_file(self, tmp_path):
        f = tmp_path / "page.png"
        f.write_bytes(b"dummy")
        assert detect_input_type(f) == "image"

    def test_jpg_file(self, tmp_path):
        f = tmp_path / "page.jpg"
        f.write_bytes(b"dummy")
        assert detect_input_type(f) == "image"

    def test_pdf_file(self, tmp_path):
        f = tmp_path / "document.pdf"
        f.write_bytes(b"dummy")
        assert detect_input_type(f) == "pdf"

    def test_directory_with_images(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"x")
        (tmp_path / "b.jpg").write_bytes(b"x")
        assert detect_input_type(tmp_path) == "image"

    def test_directory_with_pdfs(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"x")
        assert detect_input_type(tmp_path) == "pdf"

    def test_directory_with_text(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        assert detect_input_type(tmp_path) == "text"

    def test_directory_mixed(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.png").write_bytes(b"x")
        assert detect_input_type(tmp_path) == "mixed"

    def test_directory_mixed_with_pdf(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.pdf").write_bytes(b"x")
        assert detect_input_type(tmp_path) == "mixed"

    def test_empty_directory(self, tmp_path):
        assert detect_input_type(tmp_path) == "text"

    def test_nonexistent_path(self, tmp_path):
        fake = tmp_path / "nonexistent"
        assert detect_input_type(fake) == "text"

    def test_tiff_file(self, tmp_path):
        f = tmp_path / "page.tiff"
        f.write_bytes(b"dummy")
        assert detect_input_type(f) == "image"

    def test_webp_file(self, tmp_path):
        f = tmp_path / "page.webp"
        f.write_bytes(b"dummy")
        assert detect_input_type(f) == "image"


class TestGetFilesFromPathVisual:
    def test_image_input_type_collects_images(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"x")
        (tmp_path / "b.jpg").write_bytes(b"x")
        (tmp_path / "c.txt").write_text("text")
        files = get_files_from_path(tmp_path, input_type="image")
        extensions = {f.suffix.lower() for f in files}
        assert ".txt" not in extensions
        assert ".png" in extensions
        assert ".jpg" in extensions

    def test_pdf_input_type_collects_pdfs(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"x")
        (tmp_path / "b.txt").write_text("text")
        files = get_files_from_path(tmp_path, input_type="pdf")
        assert len(files) == 1
        assert files[0].suffix == ".pdf"

    def test_mixed_input_type_collects_all(self, tmp_path):
        (tmp_path / "a.txt").write_text("text")
        (tmp_path / "b.png").write_bytes(b"x")
        (tmp_path / "c.pdf").write_bytes(b"x")
        files = get_files_from_path(tmp_path, input_type="mixed")
        extensions = {f.suffix.lower() for f in files}
        assert ".txt" in extensions
        assert ".png" in extensions
        assert ".pdf" in extensions

    def test_default_text_behavior_unchanged(self, tmp_path):
        (tmp_path / "a.txt").write_text("text")
        (tmp_path / "b.png").write_bytes(b"x")
        files = get_files_from_path(tmp_path, pattern="*.txt")
        assert len(files) == 1
        assert files[0].suffix == ".txt"

    def test_single_image_file(self, tmp_path):
        f = tmp_path / "image.png"
        f.write_bytes(b"x")
        files = get_files_from_path(f, input_type="image")
        assert len(files) == 1

    def test_image_files_sorted(self, tmp_path):
        (tmp_path / "c.png").write_bytes(b"x")
        (tmp_path / "a.png").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"x")
        files = get_files_from_path(tmp_path, input_type="image")
        names = [f.name for f in files]
        assert names == ["a.png", "b.png", "c.png"]

    def test_excludes_output_dirs(self, tmp_path):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        (out_dir / "a.png").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"x")
        files = get_files_from_path(tmp_path, input_type="image")
        names = [f.name for f in files]
        assert "a.png" not in names
        assert "b.png" in names
