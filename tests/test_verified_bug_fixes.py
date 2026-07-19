"""Regression tests for a batch of verified bug fixes.

Covers converter registry-key coverage, StructuredSummaries / Bibliographic
schema-key alignment, converter crash-surface hardening, P-mode transparency
flattening, the Responses-API capability flag, the line_range_readjuster CLI
contract, schema_manager guarding, prompt schema-marker replacement, and
provider detection for the Google-native "models/gemini-..." form.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from modules.conversion.csv_converter import CSVConverter
from modules.conversion.document_converter import DocumentConverter


def _write_json(path: Path, entries: list) -> Path:
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# FIX 1 — converter registry keys reach the shipped schema names
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recipes_v3_key_reaches_dedicated_converter(tmp_path: Path) -> None:
    """The lowercased shipped name resolves to the dedicated recipe converter."""
    entry = {
        "title_original": "Tarte",
        "recipe_type": "Pastry",
        "ingredients": [
            {
                "name_modern_english": "sugar",
                "quantity_original": "2 oz",
                "ingredient_luxury_signal_rating_1_7": 4,
            }
        ],
    }
    json_file = _write_json(tmp_path / "in.json", [entry])
    out = tmp_path / "out.csv"
    CSVConverter("HistoricalRecipesEntriesProductionV3").convert_to_csv(json_file, out)

    header = out.read_text(encoding="utf-8").splitlines()[0]
    # A dedicated-converter-only column (absent from json_normalize output).
    assert "ingredient_luxury_signal_ratings" in header


@pytest.mark.unit
def test_michelin_light_key_reaches_dedicated_converter(tmp_path: Path) -> None:
    """MichelinGuidesLight resolves to a converter reading the Light shape."""
    entry = {
        "establishment_name": "Chez Test",
        "location": {"city_or_town": "Lyon", "neighbourhood_or_area": None},
        "awards": {"stars": 2, "restaurant_class": 3},
        "cuisine": {
            "cuisine_origin": ["french"],
            "culinary_style": ["bistronomy"],
            "specialties": ["quenelle"],
        },
    }
    json_file = _write_json(tmp_path / "in.json", [entry])
    out = tmp_path / "out.csv"
    CSVConverter("MichelinGuidesLight").convert_to_csv(json_file, out)

    text = out.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert "cuisine_origin" in header
    assert "restaurant_class" in header
    assert "Chez Test" in text
    assert "french" in text


# ---------------------------------------------------------------------------
# FIX 2 — StructuredSummaries reads page_number / references
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_structured_summaries_txt_reads_schema_keys(tmp_path: Path) -> None:
    entry = {
        "page_number": {"page_number_integer": 42, "contains_no_page_number": False},
        "contains_no_semantic_content": False,
        "bullet_points": ["First point", "Second point"],
        "references": ["Doe, J. (2020). A work."],
    }
    json_file = _write_json(tmp_path / "in.json", [entry])
    out = tmp_path / "out.txt"
    DocumentConverter("StructuredSummaries").convert_to_txt(json_file, out)

    text = out.read_text(encoding="utf-8")
    assert "Page 42" in text
    assert "Page Unknown" not in text
    assert "First point" in text
    assert "Doe, J. (2020). A work." in text


@pytest.mark.unit
def test_structured_summaries_csv_and_docx_consistent(tmp_path: Path) -> None:
    entry = {
        "page_number": {"page_number_integer": 7, "contains_no_page_number": False},
        "bullet_points": ["Alpha"],
        "references": ["Ref One"],
    }
    json_file = _write_json(tmp_path / "in.json", [entry])
    csv_out = tmp_path / "out.csv"
    CSVConverter("StructuredSummaries").convert_to_csv(json_file, csv_out)
    text = csv_out.read_text(encoding="utf-8")
    assert "page_number" in text.splitlines()[0]
    assert "7" in text


# ---------------------------------------------------------------------------
# FIX 3 — Bibliographic CSV/TXT align with the current schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bibliographic_csv_reads_publication_locations(tmp_path: Path) -> None:
    entry = {
        "full_title": "Le Cuisinier",
        "short_title": "Cuisinier",
        "main_author": "La Varenne",
        "edition_info": [
            {
                "year": 1651,
                "edition_number": 1,
                "publication_locations": [
                    {
                        "original_place": "Paris",
                        "modern_place": "Paris",
                        "original_region": "France",
                        "modern_region": "France",
                    }
                ],
                "contributors": [{"name": "La Varenne", "role": "Author"}],
            }
        ],
    }
    json_file = _write_json(tmp_path / "in.json", [entry])
    out = tmp_path / "out.csv"
    CSVConverter("BibliographicEntries").convert_to_csv(json_file, out)

    text = out.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert "publication_places" in header
    # Retired columns must not be emitted.
    assert "publication_cities" not in header
    assert "library_name" not in header
    assert "Paris" in text
    assert "La Varenne (Author)" in text


@pytest.mark.unit
def test_bibliographic_txt_reads_schema_keys(tmp_path: Path) -> None:
    entry = {
        "full_title": "Another Book",
        "main_author": "Anon",
        "edition_info": [
            {
                "year": 1700,
                "edition_number": 2,
                "publication_locations": [
                    {"original_place": "Lyon", "modern_place": "Lyon"}
                ],
                "contributors": [{"name": "Ed", "role": "Editor"}],
            }
        ],
    }
    json_file = _write_json(tmp_path / "in.json", [entry])
    out = tmp_path / "out.txt"
    DocumentConverter("BibliographicEntries").convert_to_txt(json_file, out)

    text = out.read_text(encoding="utf-8")
    assert "Another Book" in text
    assert "Main Author: Anon" in text
    assert "Lyon" in text
    assert "Ed (Editor)" in text


# ---------------------------------------------------------------------------
# FIX 10 — converter crash surface hardening
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_csv_converter_error_degrades_to_fallback(tmp_path: Path) -> None:
    """A converter that raises falls back to json_normalize, not an abort."""
    json_file = _write_json(tmp_path / "in.json", [{"a": 1}])
    out = tmp_path / "out.csv"

    converter = CSVConverter("BibliographicEntries")

    def _boom(_entries: list) -> object:
        raise RuntimeError("hostile output")

    with patch.object(converter, "get_converter", return_value=_boom):
        converter.convert_to_csv(json_file, out)

    assert out.exists()
    assert "a" in out.read_text(encoding="utf-8").splitlines()[0]


@pytest.mark.unit
def test_michelin_legacy_styles_with_none_elements(tmp_path: Path) -> None:
    """Legacy Michelin CSV converter tolerates None elements in list fields."""
    entry = {
        "establishment_name": "Old Guide",
        "cuisine": {"styles": ["French", None], "specialties": [None, "Duck"]},
    }
    json_file = _write_json(tmp_path / "in.json", [entry])
    out = tmp_path / "out.csv"
    CSVConverter("MichelinGuides").convert_to_csv(json_file, out)

    text = out.read_text(encoding="utf-8")
    assert "French" in text
    assert "Duck" in text


@pytest.mark.unit
def test_brazilian_converter_skips_non_dict_entries(tmp_path: Path) -> None:
    entries = [
        "not a dict",
        {"surname": "Silva", "first_name": "Joao"},
    ]
    json_file = _write_json(tmp_path / "in.json", entries)
    out = tmp_path / "out.csv"
    CSVConverter("BrazilianMilitaryRecords").convert_to_csv(json_file, out)

    text = out.read_text(encoding="utf-8")
    assert "Silva" in text


# ---------------------------------------------------------------------------
# FIX 4 — P-mode transparency flattening
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_handle_transparency_palette_mode() -> None:
    from PIL import Image

    from modules.images.llm_preprocess import ImageProcessor

    img = Image.new("P", (4, 4))
    img.info["transparency"] = 0  # palette-index transparency

    processor = ImageProcessor(provider="openai", image_config={})
    result = processor.handle_transparency(img)

    assert result.mode == "RGB"
    assert result.size == (4, 4)


# ---------------------------------------------------------------------------
# FIX 5 — Responses-API capability flag is enforced
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gpt5_sets_use_responses_api() -> None:
    from modules.llm.langchain_provider import LangChainLLM, ProviderConfig

    config = ProviderConfig(provider="openai", model="gpt-5", api_key="test-key")
    chat_model = LangChainLLM(config)._create_chat_model()
    assert chat_model.use_responses_api is True


@pytest.mark.unit
def test_gpt4o_does_not_set_use_responses_api() -> None:
    from modules.llm.langchain_provider import LangChainLLM, ProviderConfig

    config = ProviderConfig(provider="openai", model="gpt-4o", api_key="test-key")
    chat_model = LangChainLLM(config)._create_chat_model()
    assert not chat_model.use_responses_api


# ---------------------------------------------------------------------------
# FIX 6 — line_range_readjuster accepts the mode-override flags
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_readjuster_parse_arguments_accepts_non_interactive() -> None:
    import main.line_range_readjuster as lrr

    argv = ["line_range_readjuster.py", "--path", "data/", "--non-interactive"]
    with patch.object(sys, "argv", argv):
        args = lrr.parse_arguments()

    assert args.non_interactive is True
    assert str(args.path) == "data"


@pytest.mark.unit
def test_readjuster_parse_arguments_accepts_interactive() -> None:
    import main.line_range_readjuster as lrr

    with patch.object(sys, "argv", ["line_range_readjuster.py", "--interactive"]):
        args = lrr.parse_arguments()

    assert args.interactive is True


# ---------------------------------------------------------------------------
# FIX 7 — schema_manager guards a missing developer_messages/ directory
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_dev_messages_missing_dir(tmp_path: Path) -> None:
    from modules.config.schema_manager import SchemaManager

    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()
    mgr = SchemaManager(
        schemas_dir=schemas_dir, dev_messages_dir=tmp_path / "does_not_exist"
    )
    # Must not raise FileNotFoundError.
    mgr.load_dev_messages()
    assert mgr.dev_messages == {}


# ---------------------------------------------------------------------------
# FIX 8 — legacy schema-marker replacement preserves trailing prose
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_prompt_preserves_prose_after_schema() -> None:
    from modules.llm.prompt_utils import render_prompt_with_schema

    prompt = (
        "Intro.\n"
        "The JSON schema:\n"
        '{"old": "placeholder"}\n'
        "Then a closing note with a brace }.\n"
    )
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    rendered = render_prompt_with_schema(prompt, schema, inject_schema=True)

    assert "Then a closing note with a brace }." in rendered
    assert '"old"' not in rendered
    assert '"properties"' in rendered


# ---------------------------------------------------------------------------
# FIX 9 — detect_provider routes the Google-native "models/gemini-..." form
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_detect_provider_models_gemini_is_google() -> None:
    from modules.config.capabilities.detection import detect_provider

    assert detect_provider("models/gemini-2.5-flash") == "google"
    assert detect_provider("models/gemma-3-27b-it") == "google"


@pytest.mark.unit
def test_detect_provider_openrouter_slash_still_openrouter() -> None:
    from modules.config.capabilities.detection import detect_provider

    assert detect_provider("google/gemini-2.5-flash") == "openrouter"
    assert detect_provider("gemini-2.5-flash") == "google"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
