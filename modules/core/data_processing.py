# modules/core/data_processing.py

"""CSV conversion utilities for JSON data transformation."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd

from modules.core.converter_base import BaseConverter, resolve_field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level extractor helpers for declarative field specs.
# Each takes an *entry* dict and returns a formatted value.
# ---------------------------------------------------------------------------

def _join_list(entry: dict, key: str, sep: str = ", ") -> str:
    """Join a simple list field into a string."""
    vals = entry.get(key, [])
    if isinstance(vals, list):
        return sep.join(str(v) for v in vals if v is not None)
    return ""


def _join_dicts(
    entry: dict, key: str, fmt: Callable, sep: str = "; "
) -> str:
    """Join a list-of-dicts field by applying *fmt* to each element."""
    items = entry.get(key, [])
    return sep.join(fmt(item) for item in items if isinstance(item, dict))


class CSVConverter(BaseConverter):
    """
    Converts JSON-extracted data to CSV format.

    Inherits from BaseConverter for shared entry extraction and utility methods.
    """

    def convert(self, json_file: Path, output_file: Path) -> None:
        """Convert JSON to CSV format."""
        self.convert_to_csv(json_file, output_file)

    def convert_to_csv(self, json_file: Path, output_csv: Path) -> None:
        """Convert JSON entries to a CSV file."""
        entries = self.get_entries(json_file)
        if not entries:
            logger.warning("No entries found for CSV conversion.")
            return

        converters = {
            "bibliographicentries": self._convert_bibliographic_entries_to_df,
            "structuredsummaries": self._convert_structured_summaries_to_df,
            "historicaladdressbookentries": self._convert_historicaladdressbookentries_to_df,
            "brazilianoccupationrecords": self._convert_brazilianoccupationrecords_to_df,
            "brazilianmilitaryrecords": self._convert_brazilianoccupationrecords_to_df,
            "culinarypersonsentries": self._convert_culinary_persons_to_df,
            "culinaryplacesentries": self._convert_culinary_places_to_df,
            "culinaryworksentries": self._convert_culinary_works_to_df,
            "culinaryentitiesentries": self._convert_culinary_entities_to_df,
            "historicalrecipesentries": self._convert_historical_recipes_to_df,
            "michelinguides": self._convert_michelin_guides_to_df
        }
        converter = self.get_converter(converters)
        if converter:
            df = converter(entries)
        else:
            df = pd.json_normalize(entries, sep='_')
        try:
            df.to_csv(output_csv, index=False)
            logger.info(f"CSV file generated at {output_csv}")
        except Exception as e:
            logger.error(f"Error saving CSV file {output_csv}: {e}")

    # ------------------------------------------------------------------
    # Generic spec-driven converter
    # ------------------------------------------------------------------

    def _spec_to_df(
        self,
        entries: List[Any],
        field_specs: List[tuple],
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
        rows: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            row: Dict[str, Any] = {}
            for col, extractor, default in field_specs:
                if callable(extractor):
                    row[col] = extractor(entry)
                else:
                    row[col] = resolve_field(entry, extractor, default)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Declarative field specs for simple / medium schemas
    # ------------------------------------------------------------------

    _ADDRESSBOOK_CSV_FIELDS: List[tuple] = [
        ("last_name", "last_name", None),
        ("first_name", "first_name", None),
        ("street", "address.street", None),
        ("street_number", "address.street_number", None),
        ("occupation", "occupation", None),
        ("section", "section", None),
        ("honorific", "honorific", None),
        ("additional_notes", "additional_notes", None),
    ]

    _STRUCTURED_SUMMARIES_CSV_FIELDS: List[tuple] = [
        ("page_number", lambda e: (
            (e.get("page_number") or {}).get("page_number_integer")
            if isinstance(e.get("page_number", {}), dict) else None
        ), None),
        ("contains_no_page_number", lambda e: (
            (e.get("page_number") or {}).get(
                "contains_no_page_number", False)
            if isinstance(e.get("page_number", {}), dict) else False
        ), None),
        ("contains_no_semantic_content",
         "contains_no_semantic_content", False),
        ("bullet_points",
         lambda e: _join_list(e, "bullet_points", "; "), None),
        ("references",
         lambda e: _join_list(e, "references", "; "), None),
    ]

    # Shared field keys for Brazilian occupation/military records CSV columns.
    # Each tuple is (column_name, entry_key, default).
    _BRAZILIAN_CSV_FIELDS: List[tuple] = [
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

    _CULINARY_PERSONS_CSV_FIELDS: List[tuple] = [
        ("canonical_name_original", "canonical_name_original", None),
        ("canonical_name_modern_english",
         "canonical_name_modern_english", None),
        ("gender", "gender", None),
        ("roles", lambda e: _join_list(e, "roles"), None),
        ("period_start_year",
         lambda e: BaseConverter._extract_period(e)[0], None),
        ("period_end_year",
         lambda e: BaseConverter._extract_period(e)[1], None),
        ("period_notation",
         lambda e: BaseConverter._extract_period(e)[2], None),
        ("name_variants", lambda e: _join_dicts(
            e, "name_variants",
            lambda v: (
                f"{v.get('name_original', '')} "
                f"({v.get('name_modern_english', '')})"
            ),
        ), None),
        ("associated_places", lambda e: _join_dicts(
            e, "associated_places",
            lambda p: (
                f"{p.get('place_original', '')} - "
                f"{p.get('association_type', '')}"
            ),
        ), None),
        ("associated_works", lambda e: _join_dicts(
            e, "associated_works",
            lambda w: (
                f"{w.get('title_original', '')} "
                f"({w.get('role', '')})"
            ),
        ), None),
        ("notes", "notes", None),
        ("sources", lambda e: _join_dicts(
            e, "sources",
            lambda s: (
                f"{s.get('author', '')} - "
                f"{s.get('title', '')} ({s.get('year', '')})"
            ),
        ), None),
        ("links",
         lambda e: BaseConverter._format_links(
             e.get("links", [])), None),
    ]

    _CULINARY_PLACES_CSV_FIELDS: List[tuple] = [
        ("name_original", "name_original", None),
        ("name_modern_english", "name_modern_english", None),
        ("place_type", "place_type", None),
        ("country_modern", "country_modern", None),
        ("period_start_year",
         lambda e: BaseConverter._extract_period(e)[0], None),
        ("period_end_year",
         lambda e: BaseConverter._extract_period(e)[1], None),
        ("period_notation",
         lambda e: BaseConverter._extract_period(e)[2], None),
        ("roles_in_culinary_ecosystem",
         lambda e: _join_list(
             e, "roles_in_culinary_ecosystem"), None),
        ("associated_products",
         lambda e: _join_list(e, "associated_products"), None),
        ("notable_establishments",
         lambda e: _join_list(e, "notable_establishments"), None),
        ("associated_people", lambda e: _join_dicts(
            e, "associated_people",
            lambda p: (
                f"{p.get('name_original', '')} - "
                f"{p.get('association_type', '')}"
            ),
        ), None),
        ("notes", "notes", None),
        ("links",
         lambda e: BaseConverter._format_links(
             e.get("links", [])), None),
    ]

    _CULINARY_WORKS_CSV_FIELDS: List[tuple] = [
        ("title_original", "title_original", None),
        ("title_modern_english", "title_modern_english", None),
        ("short_title", "short_title", None),
        ("description", "description", None),
        ("genre", "genre", None),
        ("culinary_focus",
         lambda e: _join_list(e, "culinary_focus"), None),
        ("languages", lambda e: _join_list(e, "languages"), None),
        ("contributors", lambda e: _join_dicts(
            e, "contributors",
            lambda c: (
                f"{c.get('name_original', '')} "
                f"({c.get('role', '')})"
            ),
        ), None),
        ("edition_years",
         lambda e: _join_list(e, "edition_years"), None),
        ("publication_places", lambda e: _join_dicts(
            e, "publication_places",
            lambda p: (
                f"{p.get('name_original', '')} "
                f"({p.get('name_modern_english', '')})"
            ),
        ), None),
        ("associated_places", lambda e: _join_dicts(
            e, "associated_places",
            lambda p: (
                f"{p.get('name_original', '')} - "
                f"{p.get('association_type', '')}"
            ),
        ), None),
        ("associated_persons", lambda e: _join_dicts(
            e, "associated_persons",
            lambda p: (
                f"{p.get('name_original', '')} - "
                f"{p.get('association_type', '')}"
            ),
        ), None),
        ("notes", "notes", None),
        ("links",
         lambda e: BaseConverter._format_links(
             e.get("links", [])), None),
    ]

    # ------------------------------------------------------------------
    # Spec-driven converter wrappers
    # ------------------------------------------------------------------

    def _convert_historicaladdressbookentries_to_df(
        self, entries: List[Any]
    ) -> pd.DataFrame:
        return self._spec_to_df(entries, self._ADDRESSBOOK_CSV_FIELDS)

    def _convert_structured_summaries_to_df(
        self, entries: List[Any]
    ) -> pd.DataFrame:
        return self._spec_to_df(
            entries, self._STRUCTURED_SUMMARIES_CSV_FIELDS,
            normalize=False,
        )

    def _convert_culinary_persons_to_df(
        self, entries: List[Any]
    ) -> pd.DataFrame:
        return self._spec_to_df(
            entries, self._CULINARY_PERSONS_CSV_FIELDS)

    def _convert_culinary_places_to_df(
        self, entries: List[Any]
    ) -> pd.DataFrame:
        return self._spec_to_df(
            entries, self._CULINARY_PLACES_CSV_FIELDS)

    def _convert_culinary_works_to_df(
        self, entries: List[Any]
    ) -> pd.DataFrame:
        return self._spec_to_df(
            entries, self._CULINARY_WORKS_CSV_FIELDS)

    # ------------------------------------------------------------------
    # Brazilian records (already spec-driven)
    # ------------------------------------------------------------------

    def _convert_brazilianoccupationrecords_to_df(
        self, entries: List[Any]
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for entry in entries:
            row = {col: resolve_field(entry, key, default)
                   for col, key, default in self._BRAZILIAN_CSV_FIELDS}
            row["officials"] = self._format_officials(entry)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Complex schema converters (kept as specialized methods)
    # ------------------------------------------------------------------

    def _convert_bibliographic_entries_to_df(
        self, entries: List[Any]
    ) -> pd.DataFrame:
        """
        Converts bibliographic entries to a pandas DataFrame according to schema version 3.3.
        Creates one row per edition with all entry-level data repeated.

        :param entries: List of bibliographic entry dictionaries
        :return: pandas DataFrame with normalized bibliographic data
        """
        rows = []

        for entry in entries:
            # Extract primary entry data
            full_title = entry.get("full_title", "")
            short_title = entry.get("short_title", "")
            main_author = entry.get("main_author", "")

            # Extract library location
            library_location = entry.get("library_location")
            library_name = None
            library_city = None
            if library_location and isinstance(library_location, dict):
                library_name = library_location.get("library_name")
                library_city = library_location.get("library_city")

            culinary_focus = entry.get("culinary_focus", [])
            culinary_focus_str = ", ".join(culinary_focus) if isinstance(
                culinary_focus, list) else str(culinary_focus)

            # Process edition info
            edition_info = entry.get("edition_info", [])
            if not edition_info or not isinstance(edition_info, list):
                edition_info = []

            # If no editions, create a single row with entry data
            if not edition_info:
                rows.append({
                    "full_title": full_title,
                    "short_title": short_title,
                    "main_author": main_author,
                    "library_name": library_name,
                    "library_city": library_city,
                    "culinary_focus": culinary_focus_str
                })
                continue

            # Create one row per edition
            for edition in edition_info:
                # Extract publication locations
                pub_locations = edition.get("publication_locations", [])
                cities = [loc.get("city") for loc in pub_locations if loc.get("city")]
                countries = [loc.get("country") for loc in pub_locations if loc.get("country")]

                # Extract contributors with roles
                contributors = edition.get("contributors", [])
                contributor_strs = [f"{c.get('name', '')} ({c.get('role', '')})"
                                   for c in contributors if isinstance(c, dict)]

                # Extract publishers
                publishers = edition.get("publishers", [])
                publishers_str = ", ".join(publishers) if isinstance(publishers, list) else ""

                # Extract price info
                price_info = edition.get("price_information")
                price_str = ""
                if price_info and isinstance(price_info, dict):
                    price_str = f"{price_info.get('price', '')} {price_info.get('currency', '')}"

                edition_row = {
                    # Entry-level fields
                    "full_title": full_title,
                    "short_title": short_title,
                    "main_author": main_author,
                    "library_name": library_name,
                    "library_city": library_city,
                    "culinary_focus": culinary_focus_str,

                    # Edition-level fields
                    "edition_year": edition.get("year"),
                    "edition_number": edition.get("edition_number"),
                    "publication_cities": ", ".join(cities),
                    "publication_countries": ", ".join(countries),
                    "contributors": "; ".join(contributor_strs),
                    "publishers": publishers_str,
                    "edition_category": edition.get("edition_category"),
                    "short_note": edition.get("short_note"),
                    "language": edition.get("language"),
                    "translated_from": edition.get("translated_from"),
                    "format": edition.get("format"),
                    "pages": edition.get("pages"),
                    "has_illustrations": edition.get("has_illustrations"),
                    "dimensions": edition.get("dimensions"),
                    "price": price_str
                }
                rows.append(edition_row)

        df = pd.DataFrame(rows)
        return df

    def _convert_culinary_entities_to_df(self, entries: List[Any]) -> pd.DataFrame:
        """Flatten unified culinary entities entries (schema v3.0) into tabular rows."""
        rows: List[Dict[str, Any]] = []
        profile_keys = {
            "Person": "person_entry",
            "Place": "place_entry",
            "Work": "work_entry"
        }

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            entry_type = entry.get("entry_type")
            profile_key = profile_keys.get(entry_type)
            profile = entry.get(profile_key, {}) if profile_key else {}
            if not isinstance(profile, dict):
                profile = {}

            names = profile.get("names", {}) or {}
            timeframe = profile.get("timeframe", {}) or {}
            topical_focus = profile.get("topical_focus")
            language_contexts = profile.get("language_contexts")
            associations = profile.get("associations")

            row: Dict[str, Any] = {
                "entry_type": entry_type,
                "names_original": names.get("original"),
                "names_modern_english": names.get("modern_english"),
                "entity_summary": profile.get("entity_summary"),
                "timeframe_start_year": timeframe.get("start_year"),
                "timeframe_end_year": timeframe.get("end_year"),
                "timeframe_notation": timeframe.get("notation"),
                "topical_focus": self.join_list(topical_focus),
                "language_contexts": self.join_list(language_contexts),
                "associations": self.format_associations(associations),
                "notes": profile.get("notes"),
                # Person-specific defaults
                "person_gender": None,
                "person_roles": None,
                "person_name_variants": None,
                "person_biographical_notes": None,
                # Place-specific defaults
                "place_type": None,
                "place_country_modern": None,
                "place_roles_in_culinary_ecosystem": None,
                "place_associated_products": None,
                "place_notable_establishments": None,
                "place_notes": None,
                # Work-specific defaults
                "work_short_title": None,
                "work_description": None,
                "work_genre": None,
                "work_edition_years": None,
                "work_material_format": None,
                "work_material_has_illustrations": None,
                "work_material_page_count": None,
                "work_material_notes": None
            }

            if entry_type == "Person":
                row.update({
                    "person_gender": profile.get("gender"),
                    "person_roles": self.join_list(profile.get("roles")),
                    "person_name_variants": self.format_name_variants(profile.get("name_variants")),
                    "person_biographical_notes": profile.get("biographical_notes")
                })

            elif entry_type == "Place":
                row.update({
                    "place_type": profile.get("place_type"),
                    "place_country_modern": profile.get("country_modern"),
                    "place_roles_in_culinary_ecosystem": self.join_list(profile.get("roles_in_culinary_ecosystem")),
                    "place_associated_products": self.join_list(profile.get("associated_products")),
                    "place_notable_establishments": self.join_list(profile.get("notable_establishments")),
                    "place_notes": profile.get("place_notes")
                })

            elif entry_type == "Work":
                material_features = profile.get("material_features", {}) or {}
                row.update({
                    "work_short_title": profile.get("short_title"),
                    "work_description": profile.get("description"),
                    "work_genre": profile.get("genre"),
                    "work_edition_years": self.join_list(profile.get("edition_years")),
                    "work_material_format": material_features.get("format"),
                    "work_material_has_illustrations": material_features.get("has_illustrations"),
                    "work_material_page_count": material_features.get("page_count"),
                    "work_material_notes": material_features.get("notes")
                })

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def _convert_historical_recipes_to_df(self, entries: List[Any]) -> pd.DataFrame:
        """
        Converts historical recipes entries to DataFrame according to schema v2.2.
        Handles deeply nested structures for ingredients, cooking methods, utensils, and more.
        """
        entries = self._normalize_entries(entries)
        rows = []
        for entry in entries:
            # Extract basic fields
            recipe_text_orig = entry.get("recipe_text_original")
            recipe_text_modern = entry.get("recipe_text_modern_english")
            title_orig = entry.get("title_original")
            title_modern = entry.get("title_modern_english")
            recipe_type = entry.get("recipe_type")

            # Extract ingredients
            ingredients = entry.get("ingredients", [])
            ingredients_list = []
            for ing in ingredients:
                if isinstance(ing, dict):
                    name = ing.get("name_modern_english") or ing.get("name_original") or ""
                    qty_val = ing.get("quantity_standardized_value")
                    qty_unit = ing.get("quantity_standardized_unit")
                    qty_str = f"{qty_val} {qty_unit}" if qty_val and qty_unit else ""
                    prep = ing.get("preparation_note_modern_english") or ""
                    ing_str = f"{name} ({qty_str}) {prep}".strip()
                    ingredients_list.append(ing_str)
            ingredients_str = "; ".join(ingredients_list)

            # Extract cooking methods
            methods = entry.get("cooking_methods", [])
            methods_str = ", ".join([m.get("method_modern_english") or m.get("method_original") or ""
                                     for m in methods if isinstance(m, dict)])

            # Extract utensils
            utensils = entry.get("utensils_equipment", [])
            utensils_str = ", ".join([u.get("utensil_modern_english") or u.get("utensil_original") or ""
                                      for u in utensils if isinstance(u, dict)])

            # Extract yield and times via shared helper
            yield_str = self._extract_first_measurement(entry, "yield")
            prep_time_str = self._extract_first_measurement(
                entry, "preparation_time")
            cook_time_str = self._extract_first_measurement(
                entry, "cooking_time")

            # Extract ingredient categories
            categories = entry.get("ingredient_categories", {})
            if not isinstance(categories, dict):
                categories = {}

            row = {
                "recipe_text_original": recipe_text_orig,
                "recipe_text_modern_english": recipe_text_modern,
                "title_original": title_orig,
                "title_modern_english": title_modern,
                "recipe_type": recipe_type,
                "ingredients": ingredients_str,
                "cooking_methods": methods_str,
                "utensils_equipment": utensils_str,
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
                "contains_lard_animal_fat": categories.get("contains_lard_animal_fat", False),
                "contains_alcohol": categories.get("contains_alcohol", False),
                "contains_refined_sugar": categories.get("contains_refined_sugar", False),
                "contains_honey": categories.get("contains_honey", False),
                "contains_other_sweeteners": categories.get("contains_other_sweeteners", False),
                "contains_foreign_spices": categories.get("contains_foreign_spices", False),
                "contains_luxury_ingredients": categories.get("contains_luxury_ingredients", False)
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def _convert_michelin_guides_to_df(self, entries: List[Any]) -> pd.DataFrame:
        """
        Converts Michelin Guide entries to DataFrame according to schema v1.1.
        Handles deeply nested structures for location, address, awards, cuisine, pricing, amenities, etc.
        """
        entries = self._normalize_entries(entries)
        rows = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            # Basic info
            establishment_name = entry.get("establishment_name")
            raw_entry_text = entry.get("raw_entry_text")

            # Location
            location = entry.get("location", {}) or {}
            city = location.get("city_or_town")
            neighbourhood = location.get("neighbourhood_or_area")

            # Address
            address = entry.get("address", {}) or {}
            street = address.get("street")
            house_number = address.get("house_number")
            postal_code = address.get("postal_code")

            # Contact
            contact = entry.get("contact", {}) or {}
            telephone = contact.get("telephone")
            fax = contact.get("fax")
            website = contact.get("website")
            email = contact.get("email")

            # Map reference
            map_ref = entry.get("map_reference", {}) or {}
            plan_grid = map_ref.get("plan_grid")

            # Awards
            awards = entry.get("awards", {}) or {}
            stars = awards.get("stars")
            bib_gourmand = awards.get("bib_gourmand")
            michelin_plate = awards.get("michelin_plate")
            green_star = awards.get("green_star")
            new_in_guide = awards.get("new_in_guide")
            pleasant_marker = awards.get("pleasant_marker")
            comfort_covers = awards.get("comfort_covers")

            # Cuisine
            cuisine = entry.get("cuisine", {}) or {}
            styles = cuisine.get("styles", [])
            styles_str = ", ".join(styles) if isinstance(styles, list) and styles else ""
            specialties = cuisine.get("specialties", [])
            specialties_str = ", ".join(specialties) if isinstance(specialties, list) and specialties else ""
            chef = cuisine.get("chef")
            keywords = cuisine.get("keywords", [])
            keywords_str = ", ".join(keywords) if isinstance(keywords, list) and keywords else ""

            # Opening
            opening = entry.get("opening", {}) or {}
            lunch_hours = opening.get("lunch_hours")
            dinner_hours = opening.get("dinner_hours")
            days_closed = opening.get("days_closed", [])
            days_closed_str = ", ".join(days_closed) if isinstance(days_closed, list) and days_closed else ""
            annual_closure = opening.get("annual_closure")
            open_for_breakfast = opening.get("open_for_breakfast")

            # Pricing
            pricing = entry.get("pricing", {}) or {}
            currency = pricing.get("currency")
            menu_price_min = pricing.get("menu_price_min")
            menu_price_max = pricing.get("menu_price_max")
            a_la_carte_min = pricing.get("a_la_carte_price_min")
            a_la_carte_max = pricing.get("a_la_carte_price_max")
            lunch_menu_price = pricing.get("lunch_menu_price")
            price_note = pricing.get("price_note")
            set_menus = pricing.get("set_menus", [])
            set_menus_str = "; ".join([
                f"{m.get('label', '')}: {m.get('price_min', '')}-{m.get('price_max', '')}"
                for m in set_menus if isinstance(m, dict)
            ]) if isinstance(set_menus, list) and set_menus else ""

            # Amenities
            amenities = entry.get("amenities", {}) or {}

            # Rooms
            rooms = entry.get("rooms", {}) or {}
            room_count = rooms.get("room_count")
            room_price_min = rooms.get("room_price_min")
            room_price_max = rooms.get("room_price_max")
            room_currency = rooms.get("room_currency")
            breakfast_available = rooms.get("breakfast_available")

            # Payments
            payments = entry.get("payments", {}) or {}

            row = {
                "establishment_name": establishment_name,
                "city_or_town": city,
                "neighbourhood_or_area": neighbourhood,
                "street": street,
                "house_number": house_number,
                "postal_code": postal_code,
                "telephone": telephone,
                "fax": fax,
                "website": website,
                "email": email,
                "plan_grid": plan_grid,
                "stars": stars,
                "bib_gourmand": bib_gourmand,
                "michelin_plate": michelin_plate,
                "green_star": green_star,
                "new_in_guide": new_in_guide,
                "pleasant_marker": pleasant_marker,
                "comfort_covers": comfort_covers,
                "cuisine_styles": styles_str,
                "specialties": specialties_str,
                "chef": chef,
                "cuisine_keywords": keywords_str,
                "lunch_hours": lunch_hours,
                "dinner_hours": dinner_hours,
                "days_closed": days_closed_str,
                "annual_closure": annual_closure,
                "open_for_breakfast": open_for_breakfast,
                "currency": currency,
                "menu_price_min": menu_price_min,
                "menu_price_max": menu_price_max,
                "a_la_carte_price_min": a_la_carte_min,
                "a_la_carte_price_max": a_la_carte_max,
                "lunch_menu_price": lunch_menu_price,
                "set_menus": set_menus_str,
                "price_note": price_note,
                "wheelchair_access": amenities.get("wheelchair_access"),
                "air_conditioning": amenities.get("air_conditioning"),
                "terrace": amenities.get("terrace"),
                "garden_or_park": amenities.get("garden_or_park"),
                "outside_dining": amenities.get("outside_dining"),
                "great_view": amenities.get("great_view"),
                "peaceful": amenities.get("peaceful"),
                "notable_wine_list": amenities.get("notable_wine_list"),
                "private_dining_room": amenities.get("private_dining_room"),
                "parking": amenities.get("parking"),
                "valet_parking": amenities.get("valet_parking"),
                "has_rooms": amenities.get("has_rooms"),
                "room_count": room_count,
                "room_price_min": room_price_min,
                "room_price_max": room_price_max,
                "room_currency": room_currency,
                "breakfast_available": breakfast_available,
                "credit_cards_accepted": payments.get("credit_cards_accepted"),
                "accept_visa": payments.get("accept_visa"),
                "accept_mastercard": payments.get("accept_mastercard"),
                "accept_amex": payments.get("accept_amex"),
                "raw_entry_text": raw_entry_text
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df
