"""Prompt inputs of chronominer.llm.prompts.

The extraction prompt is chosen by chunk kind and carries the schema body as
compact JSON; enforced answers use the schema under ``ExtractionSchema``; the
boundary prompt names the semantic unit; context is topics as text or a
file, or resolves through the sidecar, folder and fallback tiers of each
role.
"""

from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

import chronominer.llm.prompts as prompts
from chronominer.llm.prompts import (
    TEXT_PROMPT_PATH,
    ContextError,
    boundary_prompt,
    extraction_prompt,
    extraction_schema,
    item_context,
    item_context_image,
)
from chronominer.spec import CONTEXT_NONE

BODY = {
    "type": "object",
    "properties": {"entries": {"type": "array", "items": {"type": "string"}}},
    "required": ["entries"],
    "additionalProperties": False,
}
SCHEMA_FILE = {"name": "PriceListEntries", "strict": True, "schema": BODY}
COMPACT = json.dumps(BODY, separators=(",", ":"))


@pytest.fixture
def context_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the project context folder at an empty temporary folder."""
    folder = tmp_path / "project_context"
    folder.mkdir()
    monkeypatch.setattr(prompts, "CONTEXT_DIR", folder)
    return folder


@pytest.mark.parametrize(
    ("kind", "opening"),
    [
        ("text", "provided input text"),
        ("image", "advanced image analysis"),
    ],
)
def test_the_chunk_kind_selects_the_template(kind: str, opening: str) -> None:
    prompt = extraction_prompt(
        kind,  # type: ignore[arg-type]
        SCHEMA_FILE,
        context="One entry per line.",
    )

    assert opening in prompt.splitlines()[0]
    assert f"The JSON schema:\n{COMPACT}" in prompt
    assert prompt.endswith("One entry per line.")


def test_the_text_prompt_matches_the_recorded_assembly() -> None:
    template = TEXT_PROMPT_PATH.read_text(encoding="utf-8").strip()
    expected = template.replace("{{TRANSCRIPTION_SCHEMA}}", COMPACT).replace(
        "{{CONTEXT}}", "Rules."
    )

    prompt = extraction_prompt("text", SCHEMA_FILE, context=" Rules.\n")

    assert prompt == expected


def test_a_prompt_without_context_drops_the_rules_label() -> None:
    prompt = extraction_prompt("text", SCHEMA_FILE, context=None)

    assert "{{CONTEXT}}" not in prompt
    assert "Rules for this corpus and work:" not in prompt
    assert prompt.endswith(COMPACT + "\n\n")


def test_a_bare_schema_is_injected_as_given() -> None:
    prompt = extraction_prompt("text", BODY, context=None)

    assert COMPACT in prompt


def test_the_enforced_schema_takes_the_extraction_name() -> None:
    assert extraction_schema(SCHEMA_FILE) == {
        "name": "ExtractionSchema",
        "schema": BODY,
        "strict": True,
    }
    assert extraction_schema(BODY)["schema"] == BODY


def test_an_empty_schema_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        extraction_schema({"name": "x", "schema": {}})


def test_the_boundary_prompt_names_the_unit_and_carries_its_schema() -> None:
    answer = {"type": "object", "properties": {"certainty": {"type": "integer"}}}

    prompt = boundary_prompt("FlatEntries", answer, "A record begins at a heading.")

    assert "(schema name: FlatEntries)" in prompt
    assert "{{BOUNDARY_TYPE}}" not in prompt
    assert json.dumps(answer, separators=(",", ":")) in prompt
    assert prompt.endswith(
        "Rules for this corpus and work:\nA record begins at a heading."
    )


def test_the_boundary_prompt_without_context_drops_the_rules_label() -> None:
    prompt = boundary_prompt("FlatEntries", {"type": "object"}, None)

    assert "Rules for this corpus and work:" not in prompt
    assert "{{CONTEXT}}" not in prompt


class TestContext:
    """The sidecar, folder and fallback tiers of the extract and adjust roles."""

    def test_the_file_sidecar_wins(self, tmp_path: Path, context_dir: Path) -> None:
        folder = tmp_path / "registers"
        folder.mkdir()
        item = folder / "book.txt"
        item.write_text("x", encoding="utf-8")
        (folder / "book_extract_context.txt").write_text("file", encoding="utf-8")
        (tmp_path / "registers_extract_context.txt").write_text("dir", encoding="utf-8")
        (context_dir / "extract_context.txt").write_text("project", encoding="utf-8")

        assert item_context(item, None) == "file"

    def test_the_folder_sidecar_follows(
        self, tmp_path: Path, context_dir: Path
    ) -> None:
        folder = tmp_path / "registers"
        folder.mkdir()
        item = folder / "book.txt"
        item.write_text("x", encoding="utf-8")
        (tmp_path / "registers_extract_context.txt").write_text("dir", encoding="utf-8")

        assert item_context(item, None) == "dir"

    def test_the_fallback_is_the_last_tier(
        self, tmp_path: Path, context_dir: Path
    ) -> None:
        item = tmp_path / "book.txt"
        item.write_text("x", encoding="utf-8")
        (context_dir / "extract_context.txt").write_text("project", encoding="utf-8")
        rules = tmp_path / "rules.txt"
        rules.write_text("rules", encoding="utf-8")

        assert item_context(item, None) is None
        assert item_context(item, None, fallback="wage ledgers") == "wage ledgers"
        assert item_context(item, None, "adjust", fallback=str(rules)) == "rules"
        (tmp_path / "book_extract_context.txt").write_text("e", encoding="utf-8")
        assert item_context(item, None, fallback="wage ledgers") == "e"

    def test_topics_as_text_skip_the_sidecars(
        self, tmp_path: Path, context_dir: Path
    ) -> None:
        item = tmp_path / "book.txt"
        (tmp_path / "book_extract_context.txt").write_text("e", encoding="utf-8")
        given = tmp_path / "manual.txt"
        given.write_text("manual", encoding="utf-8")

        assert item_context(item, "wage ledgers, 1790s") == "wage ledgers, 1790s"
        assert item_context(item, str(given)) == "manual"

    def test_text_that_cannot_be_a_path_stays_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unknown_user, too_long = "~1850 Paris", "wage ledgers " * 30 + "1790s"
        expanduser, is_file = Path.expanduser, Path.is_file

        def fake_expanduser(path: Path) -> Path:
            if str(path) == unknown_user:
                raise RuntimeError("Could not determine home directory.")
            return expanduser(path)

        def fake_is_file(path: Path) -> bool:
            if str(path) == too_long:
                raise OSError(errno.ENAMETOOLONG, "File name too long")
            return is_file(path)

        monkeypatch.setattr(Path, "expanduser", fake_expanduser)
        monkeypatch.setattr(Path, "is_file", fake_is_file)

        assert prompts.text_or_file(unknown_user) == unknown_user
        assert prompts.text_or_file(too_long) == too_long

    def test_the_adjust_role_reads_its_own_sidecar(
        self, tmp_path: Path, context_dir: Path
    ) -> None:
        item = tmp_path / "book.txt"
        item.write_text("x", encoding="utf-8")
        (tmp_path / "book_extract_context.txt").write_text("e", encoding="utf-8")
        (tmp_path / "book_adjust_context.txt").write_text("a", encoding="utf-8")

        assert item_context(item, None, "adjust") == "a"
        assert item_context(item, None) == "e"

    def test_none_sends_no_context(self, tmp_path: Path, context_dir: Path) -> None:
        item = tmp_path / "book.txt"
        (tmp_path / "book_extract_context.txt").write_text("e", encoding="utf-8")

        assert item_context(item, CONTEXT_NONE) is None
        assert item_context(tmp_path / "other.txt", None) is None

    def test_an_explicit_file_skips_the_sidecars(
        self, tmp_path: Path, context_dir: Path
    ) -> None:
        item = tmp_path / "book.txt"
        (tmp_path / "book_extract_context.txt").write_text("e", encoding="utf-8")
        given = tmp_path / "manual.txt"
        given.write_text("  manual  \n", encoding="utf-8")

        assert item_context(item, given) == "manual"
        with pytest.raises(ContextError):
            item_context(item, tmp_path / "missing.txt")

    def test_context_images_take_the_extract_suffix(
        self, tmp_path: Path, context_dir: Path
    ) -> None:
        item = tmp_path / "page.pdf"
        assert item_context_image(item, None) is None

        default = context_dir / "extract_context.jpg"
        default.write_bytes(b"jpg")
        assert item_context_image(item, None) == default

        sidecar = tmp_path / "page_extract_context.png"
        sidecar.write_bytes(b"png")
        assert item_context_image(item, None) == sidecar.resolve()
        assert item_context_image(item, CONTEXT_NONE) is None
