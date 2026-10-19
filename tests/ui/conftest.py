"""Fixtures of the ui tests: a settings file, sessions over it, input files."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from chronominer.settings import load
from chronominer.ui.session import Session
from tests.characterization.inputs import (
    make_image_folder,
    make_mixed_folder,
    make_scanned_pdf,
    make_text_file,
    write_ranges,
)

ENDPOINT = "local"
ENDPOINT_URL = "http://localhost:8000/v1"
ENDPOINT_KEY = "LOCAL_API_KEY"

DUMMY_KEYS: Mapping[str, str] = {
    "OPENAI_API_KEY": "test-openai",
    "ANTHROPIC_API_KEY": "test-anthropic",
    "GOOGLE_API_KEY": "test-google",
    "OPENROUTER_API_KEY": "test-openrouter",
    ENDPOINT_KEY: "test-local",
}
"""Dummy values for every provider key variable and the endpoint's."""

SettingsWriter = Callable[..., Path]
SessionMaker = Callable[..., Session]


@pytest.fixture
def write_settings(tmp_path: Path) -> SettingsWriter:
    """Return a writer of a settings file with the endpoint ``local``.

    ``defaults`` becomes the file's ``defaults:`` block when given.
    """

    def write(
        defaults: Mapping[str, Any] | None = None, name: str = "settings.yaml"
    ) -> Path:
        data: dict[str, Any] = {
            "endpoints": {
                ENDPOINT: {
                    "base_url": ENDPOINT_URL,
                    "api_key_env": ENDPOINT_KEY,
                    "supports_vision": True,
                    "supports_schema": True,
                }
            }
        }
        if defaults is not None:
            data["defaults"] = dict(defaults)
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    return write


@pytest.fixture
def make_session(write_settings: SettingsWriter) -> SessionMaker:
    """Return a maker of a session over a written settings file.

    The session's environment holds :data:`DUMMY_KEYS`.
    """

    def make(defaults: Mapping[str, Any] | None = None) -> Session:
        path = write_settings(defaults)
        return Session(load(path), path, dict(DUMMY_KEYS))

    return make


@pytest.fixture
def session(make_session: SessionMaker) -> Session:
    """A session over a settings file without defaults."""
    return make_session()


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    """A ledger of twelve records without line ranges."""
    return make_text_file(tmp_path / "inputs" / "ledger.txt")


@pytest.fixture
def text_with_ranges(tmp_path: Path) -> Path:
    """A ledger of twelve records with a ``_line_ranges.txt`` beside it."""
    path = make_text_file(tmp_path / "ranged" / "ledger.txt")
    write_ranges(path, [(1, 29), (31, 59)])
    return path


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """A three-page scanned PDF."""
    return make_scanned_pdf(tmp_path / "inputs" / "book.pdf", pages=3)


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    """A folder of three page images."""
    return make_image_folder(tmp_path / "inputs" / "scans", pages=3)


@pytest.fixture
def mixed_folder(tmp_path: Path) -> Path:
    """A folder with a ledger, an image and a two-page scanned PDF."""
    return make_mixed_folder(tmp_path / "mixed")
