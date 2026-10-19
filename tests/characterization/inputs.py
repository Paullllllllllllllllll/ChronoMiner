"""Deterministic text, PDF, image-folder and mixed inputs for the characterization.

Text inputs are ledgers of numbered records; each record opens with a heading
line ``Record NNN.`` (``record_heading``), which ``records_in`` finds again in
a chunk, so the fakes match a text request to its records by content.
``write_ranges`` and ``read_ranges`` handle a ledger's ``_line_ranges.txt``.

Visual inputs carry a decodable page marker: every page is filled with one
gray level from ``PALETTE``; the level encodes the page index. The marker
survives JPEG and PNG encoding, grayscale conversion, resizing and white
box-fit padding, because ``decode_marker`` takes the median of the non-white
pixels and picks the nearest palette level.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

PALETTE: tuple[int, ...] = tuple(24 + 18 * step for step in range(12))
"""Gray levels for page indices 0-11; padding white (255) stays out of range."""

_PADDING_CUTOFF = 245

_PIL_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".webp": "WEBP",
}

RECORD_RE = re.compile(r"\bRecord (\d{3})\.")
"""Matches a record heading; group 1 is the 1-based record number."""

_ITEMS = (
    "wheat, best quality",
    "rye flour",
    "barley malt",
    "salted butter",
    "tallow candles",
    "sea salt",
    "dried peas",
    "oats",
)
_MONTHS = ("January", "March", "May", "July", "September", "November")


def record_heading(index: int) -> str:
    """Return the heading line of record *index* (0-based)."""
    return f"Record {index + 1:03d}."


def record_lines(index: int) -> list[str]:
    """Return the lines of record *index* (0-based), heading first."""
    item = _ITEMS[index % len(_ITEMS)]
    month = _MONTHS[index % len(_MONTHS)]
    return [
        record_heading(index),
        f"Item: {item}, delivered to the town granary",
        f"Quantity: {index + 2} bushels by the town measure",
        f"Price: {index % 7 + 3} shillings {index % 11} pence the bushel",
        f"Date: {index % 27 + 1} {month} {1701 + index}",
    ]


def ledger_text(records: int = 12) -> str:
    """Return a ledger of *records* records separated by blank lines."""
    blocks = ["\n".join(record_lines(index)) for index in range(records)]
    return "\n\n".join(blocks) + "\n"


def make_text_file(path: Path, records: int = 12) -> Path:
    """Write a ledger of *records* records to *path* (UTF-8, LF line ends)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ledger_text(records), encoding="utf-8", newline="\n")
    return path


def records_in(text: str) -> list[int]:
    """Return the 0-based indices of the record headings in *text*, in order."""
    return [int(match.group(1)) - 1 for match in RECORD_RE.finditer(text)]


def heading_lines(path: Path) -> list[int]:
    """Return the 1-based line numbers of the record headings in a text file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [number for number, line in enumerate(lines, 1) if RECORD_RE.search(line)]


def ranges_path(text_file: Path) -> Path:
    """Return the ``_line_ranges.txt`` path of *text_file*."""
    return text_file.with_name(f"{text_file.stem}_line_ranges.txt")


def write_ranges(text_file: Path, pairs: Iterable[tuple[int, int]]) -> Path:
    """Write *pairs* as the ``_line_ranges.txt`` of *text_file*; return its path."""
    path = ranges_path(text_file)
    path.write_text(
        "".join(f"({start}, {end})\n" for start, end in pairs),
        encoding="utf-8",
        newline="\n",
    )
    return path


def read_ranges(path: Path) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` pairs of a ``_line_ranges.txt`` file."""
    pairs: list[tuple[int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            start, end = line.strip().strip("()").split(",")
            pairs.append((int(start), int(end)))
    return pairs


def marker_gray(index: int) -> int:
    """Return the gray level that marks page *index* (0-based)."""
    if not 0 <= index < len(PALETTE):
        raise ValueError(f"page index {index} outside the marker palette")
    return PALETTE[index]


def decode_marker(image: Image.Image) -> int | None:
    """Return the page index encoded in *image*, or None for an unmarked image."""
    gray = image.convert("L")
    gray.thumbnail((256, 256))
    histogram = gray.histogram()[:_PADDING_CUTOFF]
    total = sum(histogram)
    if total == 0:
        return None
    running = 0
    median = 0
    for level, count in enumerate(histogram):
        running += count
        if running * 2 >= total:
            median = level
            break
    return min(range(len(PALETTE)), key=lambda idx: abs(PALETTE[idx] - median))


def page_text(index: int) -> str:
    """Return the text layer of born-digital page *index* (0-based)."""
    return f"Born-digital page {index + 1}: grain prices in year {1601 + index}."


def _page_image(index: int, size: tuple[int, int], mode: str = "RGB") -> Image.Image:
    level = marker_gray(index)
    color: int | tuple[int, ...] = level if mode == "L" else (level, level, level)
    return Image.new(mode, size, color=color)


def _save_pdf(doc: fitz.Document, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.set_metadata({})
    doc.save(str(path), garbage=4, deflate=True, no_new_id=True)
    return path


def make_born_digital_pdf(
    path: Path,
    pages: int = 3,
    *,
    page_size: tuple[float, float] = (300.0, 400.0),
    texts: Sequence[str] | None = None,
) -> Path:
    """Write a searchable PDF: a marker fill and one line of text per page.

    The fill is vector-drawn, so native DPI resolution treats each page as a
    page without a scan. Page *i* carries ``texts[i]`` (``page_text(i)`` by
    default) as its text layer.
    """
    doc = fitz.open()
    try:
        width, height = page_size
        for index in range(pages):
            page = doc.new_page(width=width, height=height)
            shade = marker_gray(index) / 255
            page.draw_rect(page.rect, color=None, fill=(shade, shade, shade))
            text = texts[index] if texts is not None else page_text(index)
            page.insert_text((20, 40), text, fontsize=9, color=(1, 1, 1))
        return _save_pdf(doc, path)
    finally:
        doc.close()


def make_scanned_pdf(
    path: Path,
    pages: int = 3,
    *,
    page_size: tuple[float, float] = (200.0, 300.0),
    scan_size: tuple[int, int] = (500, 750),
) -> Path:
    """Write a PDF without a text layer: one page-spanning scan per page.

    Each page holds one grayscale marker image of *scan_size* pixels that
    covers the whole page, so native DPI resolution reads the scan density.
    """
    doc = fitz.open()
    try:
        width, height = page_size
        for index in range(pages):
            page = doc.new_page(width=width, height=height)
            buffer = io.BytesIO()
            _page_image(index, scan_size, mode="L").save(buffer, format="PNG")
            page.insert_image(page.rect, stream=buffer.getvalue())
        return _save_pdf(doc, path)
    finally:
        doc.close()


def _save_image(image: Image.Image, target: Path) -> None:
    pil_format = _PIL_FORMATS[target.suffix.lower()]
    if pil_format == "JPEG":
        image.save(target, format=pil_format, quality=95)
    elif pil_format == "WEBP":
        image.save(target, format=pil_format, lossless=True)
    else:
        image.save(target, format=pil_format)


def make_image(
    path: Path, index: int = 0, *, size: tuple[int, int] = (400, 600)
) -> Path:
    """Write one image carrying the marker of page *index* to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_image(_page_image(index, size), path)
    return path


def make_image_folder(
    folder: Path,
    pages: int = 3,
    *,
    extensions: Sequence[str] = (".png", ".jpg"),
    size: tuple[int, int] = (400, 600),
    stem: str = "page",
) -> Path:
    """Write an image folder with one marked image per page.

    Page *i* is named ``<stem>_<i+1:03d><ext>``, cycling through *extensions*,
    so one folder mixes PNG and JPEG pages by default.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(pages):
        extension = extensions[index % len(extensions)].lower()
        _save_image(
            _page_image(index, size), folder / f"{stem}_{index + 1:03d}{extension}"
        )
    return folder


def make_named_images(
    folder: Path,
    names: Sequence[str],
    *,
    size: tuple[int, int] = (400, 600),
) -> Path:
    """Write one marked image per name; ``names[i]`` carries page marker *i*.

    The format follows each name's extension, so a caller fixes both the file
    names (for example unpadded numbers) and the page each file holds.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        _save_image(_page_image(index, size), folder / name)
    return folder


def make_context_image(path: Path, *, size: tuple[int, int] = (300, 200)) -> Path:
    """Write a context image: white with a black frame, no page fill.

    The fakes match a request by its last image, so this image, which the
    tool sends before the page, never decides the match.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=(255, 255, 255))
    for x in range(size[0]):
        for y in (0, 1, size[1] - 2, size[1] - 1):
            image.putpixel((x, y), (0, 0, 0))
    _save_image(image, path)
    return path


def make_mixed_folder(folder: Path, *, records: int = 6) -> Path:
    """Write a folder with one ledger, one image and one scanned PDF.

    The ledger is ``ledger.txt``, the PDF ``book.pdf`` with two pages
    (markers 0 and 1) and the image ``scan.png`` with marker 2, so every
    visual unit of the folder is distinct.
    """
    folder.mkdir(parents=True, exist_ok=True)
    make_text_file(folder / "ledger.txt", records)
    make_image(folder / "scan.png", 2)
    make_scanned_pdf(folder / "book.pdf", 2)
    return folder
