"""Fixtures of the console script tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

MakePdf = Callable[..., Path]
MakeText = Callable[..., Path]

TEXT = "".join(f"Entry {index}: bread, 2 lb, 4 d\n" for index in range(1, 41))
"""Forty lines of a price list."""


@pytest.fixture
def make_pdf(tmp_path: Path) -> MakePdf:
    """Return a factory that writes a born-digital PDF under ``tmp_path``."""

    def make(name: str = "doc.pdf", pages: int = 1) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        document = fitz.open()
        for index in range(pages):
            page = document.new_page(width=200, height=300)
            page.insert_text((50, 100), f"Page {index + 1}")
        document.save(path)
        document.close()
        return path

    return make


@pytest.fixture
def make_text(tmp_path: Path) -> MakeText:
    """Return a factory that writes a text input under ``tmp_path``."""

    def make(name: str = "prices.txt", text: str = TEXT) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    return make


def json_lines(stdout: str) -> list[dict[str, Any]]:
    """Return every JSON line of *stdout*."""
    return [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
