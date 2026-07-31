# modules/conversion/csv_converter.py

"""CSV conversion utilities for JSON data transformation."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from modules.conversion.base import BaseConverter, resolve_field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level extractor helpers for declarative field specs.
# Each takes an *entry* dict and returns a formatted value.
# ---------------------------------------------------------------------------


def _join_list(entry: dict, key: str, sep: str = ", ") -> str:
    """Join a simple list field into a string."""
    vals = entry.get(key) or []
    if isinstance(vals, list):
        return sep.join(str(v) for v in vals if v is not None)
    return ""


def _join_dicts(entry: dict, key: str, fmt: Callable, sep: str = "; ") -> str:
    """Join a list-of-dicts field by applying *fmt* to each element.

    A null or non-list value (schema-valid for optional list fields) yields
    an empty string rather than raising. Elements that format to nothing
    (all-null objects) are dropped instead of emitting empty separators.
    """
    items = entry.get(key) or []
    if not isinstance(items, list):
        return ""
    cells = (fmt(item) for item in items if isinstance(item, dict))
    return sep.join(cell for cell in cells if cell)


def _event_cell(event: dict) -> str:
    """Render a v3.0 place event, omitting empty parentheses and colons."""
    event_type = event.get("event_type") or ""
    year = event.get("year")
    year_str = "" if year is None or year == "" else str(year)
    head = (
        f"{event_type} ({year_str})"
        if event_type and year_str
        else (event_type or year_str)
    )
    description = event.get("description") or ""
    if head and description:
        return f"{head}: {description}"
    return head or description


def _contributor_cell(contributor: dict, name_keys: tuple[str, ...]) -> str:
    """Render a contributor as ``name (role)``, omitting missing parts."""
    name = ""
    for key in name_keys:
        name = contributor.get(key) or ""
        if name:
            break
    role = contributor.get("role") or ""
    if name and role:
        return f"{name} ({role})"
    return str(name or role)


def _nested(entry: dict, key: str) -> dict:
    """Return a nested object field as a dict, tolerating null/non-dict."""
    value = entry.get(key)
    return value if isinstance(value, dict) else {}


class CSVConverter(BaseConverter):
    """
    Converts JSON-extracted data to CSV format.

    Inherits from BaseConverter for shared entry extraction and utility methods.
    """

    def convert(self, json_file: Path, output_file: Path) -> None:
        """Convert JSON to CSV format."""
        self.convert_to_csv(json_file, output_file)

    def convert_to_csv(self, json_file: Path, output_csv: Path) -> None:
        """Convert JSON entries to a CSV file.

        An empty entry list still produces a file so that the CSV, DOCX and
        TXT paths agree: header-only when the schema has a declarative field
        spec, otherwise an empty file.
        """
        entries = self.get_entries(json_file)
        if not entries:
            logger.warning("No entries found for CSV conversion.")

        converters = {
            "bibliographicentries": self._convert_bibliographic_entries_to_df,
            "structuredsummaries": self._convert_structured_summaries_to_df,
            "historicaladdressbookentries": (
                self._convert_historicaladdressbookentries_to_df
            ),
            "brazilianmilitaryrecords": self._convert_brazilianoccupationrecords_to_df,
            "culinarypersonsentries": self._convert_culinary_persons_to_df,
            "culinaryplacesentries": self._convert_culinary_places_to_df,
            "culinaryworksentries": self._convert_culinary_works_to_df,
            "culinaryentitiesentries": self._convert_culinary_entities_to_df,
            "historicalrecipesentriesproduction": (
                self._convert_historical_recipes_production_to_df
            ),
            "historicalrecipesentriesproductionv3": (
                self._convert_historical_recipes_production_to_df
            ),
            "michelinguideslight": self._convert_michelin_guides_light_to_df,
            "cookbookmetadataentries": self._convert_cookbook_metadata_to_df,
        }
        converter = self.get_converter(converters)
        # Run the converter inside the guard so a single hostile element degrades
        # to the json_normalize fallback rather than aborting the whole file.
        try:
            if converter:
                df = converter(entries)
            else:
                df = pd.json_normalize(entries, sep="_")
        except Exception as e:
            logger.warning(
                f"Converter for schema '{self.schema_name}' failed ({e}); "
                "falling back to json_normalize."
            )
            df = pd.json_normalize(entries, sep="_")
        try:
            # Nullable integers otherwise become float64 and render as "1651.0";
            # convert_dtypes() maps them to Int64 so the CSV shows "1651"/empty.
            df = df.convert_dtypes()
            # utf-8-sig so Excel recognises the encoding of non-ASCII cells.
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            logger.info(f"CSV file generated at {output_csv}")
        except Exception as e:
            logger.error(f"Error saving CSV file {output_csv}: {e}")

    # ------------------------------------------------------------------
    # Generic spec-driven converter
    # ------------------------------------------------------------------

    def _spec_to_df(
        self,
        entries: list[Any],
        field_specs: list[tuple],
        *,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Convert entries to DataFrame using declarative field specs.

        Each spec is ``(column_name, extractor, default)``:

        - If *extractor* is callable, ``extractor(entry)`` is called
          (the *default* element is ignored).
        - Otherwise ``resolve_field(entry, extractor, default)`` is used.
        """
        if normalize:
            entries = self._normalize_entries(entries)
        rows: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            row: dict[str, Any] = {}
            for col, extractor, default in field_specs:
                if callable(extractor):
                    row[col] = extractor(entry)
                else:
                    row[col] = resolve_field(entry, extractor, default)
            rows.append(row)
        return pd.DataFrame(rows, columns=[col for col, _, _ in field_specs])

    # ------------------------------------------------------------------
    # Declarative field specs for simple / medium schemas
    # ------------------------------------------------------------------

    _ADDRESSBOOK_CSV_FIELDS: list[tuple] = [
        ("last_name", "last_name", None),
        ("first_name", "first_name", None),
        ("street", "address.street", None),
        ("street_number", "address.street_number", None),
        ("occupation", "occupation", None),
        ("section", "section", None),
        ("honorific", "honorific", None),
        ("additional_notes", "additional_notes", None),
    ]

    _STRUCTURED_SUMMARIES_CSV_FIELDS: list[tuple] = [
        (
            "page_number",
            lambda e: (
                (e.get("page_number") or {}).get("page_number_integer")
                if isinstance(e.get("page_number", {}), dict)
                else None
            ),
            None,
        ),
        (
            "contains_no_page_number",
            lambda e: (
                (e.get("page_number") or {}).get("contains_no_page_number", False)
                if isinstance(e.get("page_number", {}), dict)
                else False
            ),
            None,
        ),
        ("contains_no_semantic_content", "contains_no_semantic_content", False),
        ("bullet_points", lambda e: _join_list(e, "bullet_points", "; "), None),
        ("references", lambda e: _join_list(e, "references", "; "), None),
    ]

    # Shared field keys for Brazilian occupation/military records CSV columns.
    # Each tuple is (column_name, entry_key, default).
    _BRAZILIAN_CSV_FIELDS: list[tuple] = [
        ("surname", "surname", ""),
        ("first_name", "first_name", ""),
        ("record_header", "record_header", ""),
        ("location", "location", ""),
        ("height", "height", ""),
        ("skin_color", "skin_color", ""),
        ("hair_color", "hair_color", ""),
        ("hair_texture", "hair_texture", ""),
        ("beard", "beard", ""),
        ("mustache", "mustache", ""),
        ("assignatura", "assignatura", ""),
        ("reservista", "reservista", ""),
        ("eyes", "eyes", ""),
        ("mouth", "mouth", ""),
        ("face", "face", ""),
        ("nose", "nose", ""),
        ("marks", "marks", ""),
        ("father", "father", ""),
        ("mother", "mother", ""),
        ("birth_date", "birth_date", ""),
        ("birth_place", "birth_place", ""),
        ("municipality", "municipality", ""),
        ("profession", "profession", ""),
        ("civil_status", "civil_status", ""),
        ("vaccinated", "vaccinated", ""),
        ("can_read", "can_read", ""),
        ("can_write", "can_write", ""),
        ("can_count", "can_count", ""),
        ("swimming", "swimming", ""),
        ("cyclist", "cyclist", ""),
        ("motorcyclist", "motorcyclist", ""),
        ("driver", "driver", ""),
        ("chauffeur", "chauffeur", ""),
        ("telegraphist", "telegraphist", ""),
        ("telephonist", "telephonist", ""),
        ("residence", "residence", ""),
        ("observations", "observations", ""),
    ]

    # CulinaryPersonsEntries (schema v3.0) — nested names/timeframe/lifespan/
    # geography plus the unified associations list.
    _CULINARY_PERSONS_CSV_FIELDS: list[tuple] = [
        ("name_original", "names.original", None),
        ("name_modern_english", "names.modern_english", None),
        ("short_notes", "short_notes", None),
        ("historical_importance", "historical_importance", None),
        ("gender", "gender", None),
        ("roles", lambda e: _join_list(e, "roles"), None),
        ("timeframe_start_year", lambda e: BaseConverter._extract_period(e)[0], None),
        ("timeframe_end_year", lambda e: BaseConverter._extract_period(e)[1], None),
        ("timeframe_notation", lambda e: BaseConverter._extract_period(e)[2], None),
        ("birth_year", lambda e: _nested(e, "lifespan").get("birth_year"), None),
        ("death_year", lambda e: _nested(e, "lifespan").get("death_year"), None),
        ("city_original", "geography.city_original", None),
        ("city_modern", "geography.city_modern", None),
        ("country_original", "geography.country_original", None),
        ("country_modern", "geography.country_modern", None),
        (
            "associations",
            lambda e: BaseConverter._format_links(e.get("associations")),
            None,
        ),
    ]

    # CulinaryPlacesEntries (schema v3.0).
    _CULINARY_PLACES_CSV_FIELDS: list[tuple] = [
        ("name_original", "names.original", None),
        ("name_modern_english", "names.modern_english", None),
        ("short_notes", "short_notes", None),
        ("historical_importance", "historical_importance", None),
        ("place_type", "place_type", None),
        (
            "roles_in_culinary_ecosystem",
            lambda e: _join_list(e, "roles_in_culinary_ecosystem"),
            None,
        ),
        ("timeframe_start_year", lambda e: BaseConverter._extract_period(e)[0], None),
        ("timeframe_end_year", lambda e: BaseConverter._extract_period(e)[1], None),
        ("timeframe_notation", lambda e: BaseConverter._extract_period(e)[2], None),
        ("city_original", "geography.city_original", None),
        ("city_modern", "geography.city_modern", None),
        ("country_original", "geography.country_original", None),
        ("country_modern", "geography.country_modern", None),
        ("events", lambda e: _join_dicts(e, "events", _event_cell), None),
        (
            "associations",
            lambda e: BaseConverter._format_links(e.get("associations")),
            None,
        ),
    ]

    # CulinaryWorksEntries (schema v3.0) — nested titles/timeframe/geography.
    _CULINARY_WORKS_CSV_FIELDS: list[tuple] = [
        ("title_original", "titles.original", None),
        ("title_modern_english", "titles.modern_english", None),
        ("title_short", "titles.short", None),
        ("short_notes", "short_notes", None),
        ("historical_importance", "historical_importance", None),
        ("genre", "genre", None),
        ("culinary_focus", lambda e: _join_list(e, "culinary_focus"), None),
        ("languages", lambda e: _join_list(e, "languages"), None),
        ("edition_years", lambda e: _join_list(e, "edition_years"), None),
        ("timeframe_start_year", lambda e: BaseConverter._extract_period(e)[0], None),
        ("timeframe_end_year", lambda e: BaseConverter._extract_period(e)[1], None),
        ("timeframe_notation", lambda e: BaseConverter._extract_period(e)[2], None),
        ("city_original", "geography.city_original", None),
        ("city_modern", "geography.city_modern", None),
        ("country_original", "geography.country_original", None),
        ("country_modern", "geography.country_modern", None),
        (
            "contributors",
            lambda e: _join_dicts(
                e,
                "contributors",
                lambda c: _contributor_cell(
                    c, ("name_original", "name_modern_english")
                ),
            ),
            None,
        ),
        (
            "associations",
            lambda e: BaseConverter._format_links(e.get("associations")),
            None,
        ),
    ]

    # ------------------------------------------------------------------
    # Spec-driven converter wrappers
    # ------------------------------------------------------------------

    def _convert_historicaladdressbookentries_to_df(
        self, entries: list[Any]
    ) -> pd.DataFrame:
        return self._spec_to_df(entries, self._ADDRESSBOOK_CSV_FIELDS)

    def _convert_structured_summaries_to_df(self, entries: list[Any]) -> pd.DataFrame:
        return self._spec_to_df(
            entries,
            self._STRUCTURED_SUMMARIES_CSV_FIELDS,
            normalize=False,
        )

    def _convert_culinary_persons_to_df(self, entries: list[Any]) -> pd.DataFrame:
        return self._spec_to_df(entries, self._CULINARY_PERSONS_CSV_FIELDS)

    def _convert_culinary_places_to_df(self, entries: list[Any]) -> pd.DataFrame:
        return self._spec_to_df(entries, self._CULINARY_PLACES_CSV_FIELDS)

    def _convert_culinary_works_to_df(self, entries: list[Any]) -> pd.DataFrame:
        return self._spec_to_df(entries, self._CULINARY_WORKS_CSV_FIELDS)

    # ------------------------------------------------------------------
    # Brazilian records (already spec-driven)
    # ------------------------------------------------------------------

    def _convert_brazilianoccupationrecords_to_df(
        self, entries: list[Any]
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            row = {
                col: resolve_field(entry, key, default)
                for col, key, default in self._BRAZILIAN_CSV_FIELDS
            }
            row["officials"] = self._format_officials(entry)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Complex schema converters (kept as specialized methods)
    # ------------------------------------------------------------------

    def _convert_bibliographic_entries_to_df(self, entries: list[Any]) -> pd.DataFrame:
        """
        Converts bibliographic entries to a pandas DataFrame (schema v4.4).

        Creates one row per edition with all entry-level data repeated. Reads
        the current schema keys only; columns for retired fields (library
        location, culinary_focus, publishers, dimensions) are dropped rather
        than emitted empty.

        :param entries: List of bibliographic entry dictionaries
        :return: pandas DataFrame with normalized bibliographic data
        """
        rows: list[dict[str, Any]] = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            entry_row: dict[str, Any] = {
                "full_title": entry.get("full_title", ""),
                "short_title": entry.get("short_title", ""),
                "main_author": entry.get("main_author", ""),
                "institutional_main_author": entry.get("institutional_main_author"),
                "short_note": entry.get("short_note"),
                "library_abbreviation": entry.get("library_abbreviation"),
                "volumes_overview": entry.get("volumes_overview"),
                "volume_numbers": self.join_list(entry.get("volume_numbers")),
            }

            edition_info = entry.get("edition_info", [])
            if not isinstance(edition_info, list) or not edition_info:
                rows.append(dict(entry_row))
                continue

            for edition in edition_info:
                if not isinstance(edition, dict):
                    continue

                # Publication locations: prefer modern equivalents.
                pub_locations = edition.get("publication_locations") or []
                places = [
                    loc.get("modern_place") or loc.get("original_place")
                    for loc in pub_locations
                    if isinstance(loc, dict)
                    and (loc.get("modern_place") or loc.get("original_place"))
                ]
                regions = [
                    loc.get("modern_region") or loc.get("original_region")
                    for loc in pub_locations
                    if isinstance(loc, dict)
                    and (loc.get("modern_region") or loc.get("original_region"))
                ]

                contributors = edition.get("contributors") or []
                contributor_strs = [
                    cell
                    for cell in (
                        _contributor_cell(c, ("name",))
                        for c in contributors
                        if isinstance(c, dict)
                    )
                    if cell
                ]

                price_info = edition.get("price_information") or {}
                price_str = ""
                if isinstance(price_info, dict) and (
                    price_info.get("price") is not None or price_info.get("currency")
                ):
                    price_str = (
                        f"{self.safe_str(price_info.get('price'))} "
                        f"{self.safe_str(price_info.get('currency'))}"
                    ).strip()

                edition_row = dict(entry_row)
                edition_row.update(
                    {
                        "edition_year": edition.get("year"),
                        "edition_number": edition.get("edition_number"),
                        "edition_volume_numbers": self.join_list(
                            edition.get("volume_numbers")
                        ),
                        "publication_places": ", ".join(str(p) for p in places),
                        "publication_regions": ", ".join(str(r) for r in regions),
                        "contributors": "; ".join(contributor_strs),
                        "edition_category": edition.get("edition_category"),
                        "language": edition.get("language"),
                        "translated_from": edition.get("translated_from"),
                        "format": edition.get("format"),
                        "pages": edition.get("pages"),
                        "has_illustrations": edition.get("has_illustrations"),
                        "is_manuscript": edition.get("is_manuscript"),
                        "price": price_str,
                    }
                )
                rows.append(edition_row)

        return pd.DataFrame(rows)

    def _convert_culinary_entities_to_df(self, entries: list[Any]) -> pd.DataFrame:
        """Flatten unified culinary entities entries (schema v3.0) into tabular rows."""
        rows: list[dict[str, Any]] = []
        profile_keys = {
            "Person": "person_entry",
            "Place": "place_entry",
            "Work": "work_entry",
        }

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            entry_type = entry.get("entry_type", "")
            profile_key = profile_keys.get(str(entry_type))
            profile = entry.get(profile_key, {}) if profile_key else {}
            if not isinstance(profile, dict):
                profile = {}

            names = _nested(profile, "names")
            timeframe = _nested(profile, "timeframe")
            geography = _nested(profile, "geography")

            row: dict[str, Any] = {
                "entry_type": entry_type,
                "names_original": names.get("original"),
                "names_modern_english": names.get("modern_english"),
                "importance": profile.get("importance"),
                "summary": profile.get("summary"),
                "timeframe_start_year": timeframe.get("start_year"),
                "timeframe_end_year": timeframe.get("end_year"),
                "timeframe_notation": timeframe.get("notation"),
                "geography_primary_location": geography.get("primary_location"),
                "geography_additional_context": geography.get("additional_context"),
                "topical_focus": self.join_list(profile.get("topical_focus")),
                "language_contexts": self.join_list(profile.get("language_contexts")),
                # Person-specific defaults
                "person_roles": None,
                # Place-specific defaults
                "place_roles_in_culinary_ecosystem": None,
                "place_associated_products": None,
                "place_notable_establishments": None,
                "place_notes": None,
                # Work-specific defaults
                "work_short_title": None,
                "work_genre": None,
            }

            if entry_type == "Person":
                row["person_roles"] = self.join_list(profile.get("roles"))

            elif entry_type == "Place":
                row.update(
                    {
                        "place_roles_in_culinary_ecosystem": self.join_list(
                            profile.get("roles_in_culinary_ecosystem")
                        ),
                        "place_associated_products": self.join_list(
                            profile.get("associated_products")
                        ),
                        "place_notable_establishments": self.join_list(
                            profile.get("notable_establishments")
                        ),
                        "place_notes": profile.get("place_notes"),
                    }
                )

            elif entry_type == "Work":
                row.update(
                    {
                        "work_short_title": profile.get("short_title"),
                        "work_genre": profile.get("genre"),
                    }
                )

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    # ------------------------------------------------------------------
    # MichelinGuidesLight (schema 3.4-light)
    # ------------------------------------------------------------------

    def _convert_michelin_guides_light_to_df(self, entries: list[Any]) -> pd.DataFrame:
        """Convert MichelinGuidesLight entries to DataFrame (schema 3.4-light).

        Reads the current Light schema shape (location, address, awards with
        hotel/restaurant class, cuisine_origin/culinary_style arrays, pricing,
        rooms, inspector_note, entry_is_fragment).
        """
        entries = self._normalize_entries(entries)
        rows: list[dict[str, Any]] = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            location = entry.get("location", {}) or {}
            address = entry.get("address", {}) or {}
            awards = entry.get("awards", {}) or {}
            cuisine = entry.get("cuisine", {}) or {}
            pricing = entry.get("pricing", {}) or {}
            rooms = entry.get("rooms", {}) or {}

            rows.append(
                {
                    "establishment_name": entry.get("establishment_name"),
                    "city_or_town": location.get("city_or_town"),
                    "neighbourhood_or_area": location.get("neighbourhood_or_area"),
                    "street": address.get("street"),
                    "house_number": address.get("house_number"),
                    "postal_code": address.get("postal_code"),
                    "stars": awards.get("stars"),
                    "bib_gourmand": awards.get("bib_gourmand"),
                    "michelin_plate": awards.get("michelin_plate"),
                    "pleasant_marker": awards.get("pleasant_marker"),
                    "hotel_class": awards.get("hotel_class"),
                    "restaurant_class": awards.get("restaurant_class"),
                    "cuisine_origin": self.join_list(cuisine.get("cuisine_origin")),
                    "culinary_style": self.join_list(cuisine.get("culinary_style")),
                    "specialties": self.join_list(cuisine.get("specialties")),
                    "currency": pricing.get("currency"),
                    "menu_price_min": pricing.get("menu_price_min"),
                    "menu_price_max": pricing.get("menu_price_max"),
                    "a_la_carte_price_min": pricing.get("a_la_carte_price_min"),
                    "a_la_carte_price_max": pricing.get("a_la_carte_price_max"),
                    "lunch_menu_price": pricing.get("lunch_menu_price"),
                    "room_count": rooms.get("room_count"),
                    "room_price_min": rooms.get("room_price_min"),
                    "room_price_max": rooms.get("room_price_max"),
                    "accepts_credit_cards": entry.get("accepts_credit_cards"),
                    "inspector_note": entry.get("inspector_note"),
                    "entry_is_fragment": entry.get("entry_is_fragment"),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # HistoricalRecipesEntriesProduction (schema v3.0)
    # ------------------------------------------------------------------

    def _convert_historical_recipes_production_to_df(
        self, entries: list[Any]
    ) -> pd.DataFrame:
        """
        Converts HistoricalRecipesEntriesProduction entries to DataFrame (schema v3.0).

        Extends the base recipe converter with per-ingredient rating columns
        (luxury signal, trade distance, novelty, stated origin), per-method
        complexity rating and per-utensil specialization/modernity ratings,
        each serialised as a semicolon-separated list in source order. Also
        exposes the culinary_style, intertextuality, geographic_signals,
        economic_signals and religious_signals analytical fields.
        """
        entries = self._normalize_entries(entries)
        rows: list[dict[str, Any]] = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            # Base textual fields
            recipe_text_orig = entry.get("recipe_text_original")
            recipe_text_modern = entry.get("recipe_text_modern_english")
            title_orig = entry.get("title_original")
            title_modern = entry.get("title_modern_english")
            recipe_type = entry.get("recipe_type")

            # Ingredients — production schema uses quantity_original
            # (no standardized fields)
            ingredients = entry.get("ingredients") or []
            ingredients_list: list[str] = []
            luxury_ratings: list[str] = []
            trade_distance_ratings: list[str] = []
            novelty_ratings: list[str] = []
            ingredient_origins: list[str] = []
            for ing in ingredients:
                if not isinstance(ing, dict):
                    continue
                name = ing.get("name_modern_english") or ing.get("name_original") or ""
                qty = ing.get("quantity_original") or ""
                ing_str = f"{name} ({qty})".strip() if qty else name
                ingredients_list.append(ing_str)
                luxury_ratings.append(
                    str(ing.get("ingredient_luxury_signal_rating_1_7") or "")
                )
                trade_distance_ratings.append(
                    str(ing.get("ingredient_trade_distance_rating_1_7") or "")
                )
                novelty_ratings.append(
                    str(ing.get("ingredient_novelty_rating_1_7") or "")
                )
                ingredient_origins.append(
                    str(ing.get("origin_explicitly_stated") or "")
                )

            ingredients_str = "; ".join(ingredients_list)
            luxury_ratings_str = "; ".join(luxury_ratings)
            trade_distance_ratings_str = "; ".join(trade_distance_ratings)
            novelty_ratings_str = "; ".join(novelty_ratings)
            ingredient_origins_str = "; ".join(ingredient_origins)

            # Cooking methods
            methods = entry.get("cooking_methods") or []
            methods_list: list[str] = []
            complexity_ratings: list[str] = []
            for m in methods:
                if not isinstance(m, dict):
                    continue
                methods_list.append(
                    m.get("method_modern_english") or m.get("method_original") or ""
                )
                complexity_ratings.append(
                    str(m.get("method_complexity_rating_1_7") or "")
                )
            methods_str = ", ".join(methods_list)
            complexity_ratings_str = "; ".join(complexity_ratings)

            # Utensils with per-utensil ratings, kept index-parallel
            utensils = entry.get("utensils_equipment") or []
            utensils_list: list[str] = []
            utensil_specialization: list[str] = []
            utensil_modernity: list[str] = []
            for u in utensils:
                if not isinstance(u, dict):
                    continue
                utensils_list.append(
                    u.get("utensil_modern_english") or u.get("utensil_original") or ""
                )
                utensil_specialization.append(
                    str(u.get("utensil_specialization_rating_1_7") or "")
                )
                utensil_modernity.append(
                    str(u.get("utensil_modernity_rating_1_7") or "")
                )
            utensils_str = "; ".join(utensils_list)
            utensil_specialization_str = "; ".join(utensil_specialization)
            utensil_modernity_str = "; ".join(utensil_modernity)

            # Timing/yield — production schema stores these in a nested object
            timing_yield = entry.get("timing_yield", {}) or {}
            yield_str = timing_yield.get("yield_original") or ""
            prep_time_str = timing_yield.get("preparation_time_original") or ""
            cook_time_str = timing_yield.get("cooking_time_original") or ""

            # Ingredient category boolean flags
            categories = entry.get("ingredient_categories", {})
            if not isinstance(categories, dict):
                categories = {}

            # Culinary style analytical fields
            culinary_style = entry.get("culinary_style", {}) or {}
            modernity_rating = culinary_style.get("modernity_rating_1_7")
            innovation_markers = self.join_list(
                culinary_style.get("innovation_markers_observed"), "; "
            )
            archaism_markers = self.join_list(
                culinary_style.get("archaism_markers_observed"), "; "
            )

            # Intertextuality analytical fields
            inter = entry.get("intertextuality", {}) or {}

            # Geographic / economic / religious signal blocks
            geo = entry.get("geographic_signals", {}) or {}
            place_refs = geo.get("place_references") or []
            place_refs_str = "; ".join(
                (
                    f"{ref.get('place_name_original') or ''}"
                    f" ({ref.get('reference_function') or ''})"
                )
                for ref in place_refs
                if isinstance(ref, dict)
            )
            econ = entry.get("economic_signals", {}) or {}
            relig = entry.get("religious_signals", {}) or {}

            row: dict[str, Any] = {
                "recipe_text_original": recipe_text_orig,
                "recipe_text_modern_english": recipe_text_modern,
                "title_original": title_orig,
                "title_modern_english": title_modern,
                "recipe_type": recipe_type,
                "ingredients": ingredients_str,
                "ingredient_luxury_signal_ratings": luxury_ratings_str,
                "ingredient_trade_distance_ratings": trade_distance_ratings_str,
                "ingredient_novelty_ratings": novelty_ratings_str,
                "ingredient_origins_explicitly_stated": ingredient_origins_str,
                "cooking_methods": methods_str,
                "method_complexity_ratings": complexity_ratings_str,
                "utensils_equipment": utensils_str,
                "utensil_specialization_ratings": utensil_specialization_str,
                "utensil_modernity_ratings": utensil_modernity_str,
                "yield": yield_str,
                "preparation_time": prep_time_str,
                "cooking_time": cook_time_str,
                "contains_meat": categories.get("contains_meat", False),
                "contains_poultry": categories.get("contains_poultry", False),
                "contains_fish_seafood": categories.get("contains_fish_seafood", False),
                "contains_dairy": categories.get("contains_dairy", False),
                "contains_eggs": categories.get("contains_eggs", False),
                "contains_butter": categories.get("contains_butter", False),
                "contains_olive_oil": categories.get("contains_olive_oil", False),
                "contains_lard_animal_fat": categories.get(
                    "contains_lard_animal_fat", False
                ),
                "contains_alcohol": categories.get("contains_alcohol", False),
                "contains_refined_sugar": categories.get(
                    "contains_refined_sugar", False
                ),
                "contains_honey": categories.get("contains_honey", False),
                "contains_other_sweeteners": categories.get(
                    "contains_other_sweeteners", False
                ),
                "contains_foreign_spices": categories.get(
                    "contains_foreign_spices", False
                ),
                "contains_luxury_ingredients": categories.get(
                    "contains_luxury_ingredients", False
                ),
                "modernity_rating_1_7": modernity_rating,
                "innovation_markers_observed": innovation_markers,
                "archaism_markers_observed": archaism_markers,
                "explicit_source_attribution": inter.get("explicit_source_attribution"),
                "explicit_foreign_style_reference": inter.get(
                    "explicit_foreign_style_reference"
                ),
                "self_positioning_temporal": inter.get("self_positioning_temporal"),
                "tradition_claim_present": inter.get("tradition_claim_present"),
                "authenticity_claim_present": inter.get("authenticity_claim_present"),
                "national_identity_claim_present": inter.get(
                    "national_identity_claim_present"
                ),
                "anti_foreign_sentiment_present": inter.get(
                    "anti_foreign_sentiment_present"
                ),
                "place_references": place_refs_str,
                "economic_framing_detected": self.join_list(
                    econ.get("economic_framing_detected"), "; "
                ),
                "luxury_intensity_rating_1_7": econ.get("luxury_intensity_rating_1_7"),
                "occasion_type": self.join_list(econ.get("occasion_type"), "; "),
                "fasting_context_indicated": relig.get("fasting_context_indicated"),
                "meat_day_context_indicated": relig.get("meat_day_context_indicated"),
                "confessional_hint": relig.get("confessional_hint"),
                "moral_virtue_framing_present": relig.get(
                    "moral_virtue_framing_present"
                ),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # CookbookMetadataEntries (schema v1.0)
    # ------------------------------------------------------------------

    _COOKBOOK_METADATA_CSV_FIELDS: list[tuple] = [
        ("title", "title", None),
        ("author", "author", None),
        ("year", "year", None),
        ("edition", "edition", None),
        ("content", "content", None),
        ("notes", "notes", None),
        ("library", "library", None),
        ("digitizer", "digitizer", None),
        ("misc", "misc", None),
    ]

    def _convert_cookbook_metadata_to_df(self, entries: list[Any]) -> pd.DataFrame:
        return self._spec_to_df(entries, self._COOKBOOK_METADATA_CSV_FIELDS)
