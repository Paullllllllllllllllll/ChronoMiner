# modules/core/data_processing.py

"""CSV conversion utilities for JSON data transformation."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from modules.core.converter_base import BaseConverter, resolve_field

logger = logging.getLogger(__name__)


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

    def _convert_structured_summaries_to_df(self,
                                            entries: List[Any]) -> pd.DataFrame:
        """
        Converts structured summaries to DataFrame according to schema v4.0.
        Properly handles nested page_number object.
        """
        rows = []
        for entry in entries:
            # Extract nested page_number object
            page_num_obj = entry.get("page_number", {})
            page_num = page_num_obj.get("page_number_integer") if isinstance(page_num_obj, dict) else None
            no_page_num = page_num_obj.get("contains_no_page_number", False) if isinstance(page_num_obj, dict) else False
            
            # Extract other fields
            no_content = entry.get("contains_no_semantic_content", False)
            bullet_points = entry.get("bullet_points", [])
            references = entry.get("references", [])
            
            row = {
                "page_number": page_num,
                "contains_no_page_number": no_page_num,
                "contains_no_semantic_content": no_content,
                "bullet_points": "; ".join(bullet_points) if isinstance(bullet_points, list) else "",
                "references": "; ".join(references) if isinstance(references, list) else ""
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df

    def _convert_historicaladdressbookentries_to_df(
        self, entries: List[Any]
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            address_data = entry.get("address") or {}
            if not isinstance(address_data, dict):
                address_data = {}

            row: Dict[str, Any] = {
                "last_name": entry.get("last_name"),
                "first_name": entry.get("first_name"),
                "street": address_data.get("street"),
                "street_number": address_data.get("street_number"),
                "occupation": entry.get("occupation"),
                "section": entry.get("section"),
                "honorific": entry.get("honorific"),
                "additional_notes": entry.get("additional_notes"),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

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

    @staticmethod
    def _format_officials(entry: dict) -> str:
        officials = entry.get("officials", [])
        if officials is None:
            return ""
        return "; ".join(
            f"{o.get('position', '')}: {o.get('signature', '')}" for o in officials
        )

    def _convert_brazilianoccupationrecords_to_df(self, entries: List[Any]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for entry in entries:
            row = {col: resolve_field(entry, key, default)
                   for col, key, default in self._BRAZILIAN_CSV_FIELDS}
            row["officials"] = self._format_officials(entry)
            rows.append(row)
        return pd.DataFrame(rows)

    def _convert_culinary_persons_to_df(self, entries: List[Any]) -> pd.DataFrame:
        """
        Converts culinary persons entries to DataFrame according to schema v2.0.
        Handles nested arrays for name_variants, associated_places, works, sources, and links.
        """
        # Filter out None entries
        if entries is None:
            entries = []
        entries = [entry for entry in entries if entry is not None]
        rows = []
        for entry in entries:
            # Extract basic fields
            canonical_orig = entry.get("canonical_name_original")
            canonical_modern = entry.get("canonical_name_modern_english")
            gender = entry.get("gender")
            roles = entry.get("roles", [])
            notes = entry.get("notes")
            
            # Extract period
            period = entry.get("period", {})
            period_start = period.get("start_year") if isinstance(period, dict) else None
            period_end = period.get("end_year") if isinstance(period, dict) else None
            period_notation = period.get("notation") if isinstance(period, dict) else None
            
            # Extract and format arrays
            name_variants = entry.get("name_variants", [])
            name_variants_str = "; ".join([f"{v.get('name_original', '')} ({v.get('name_modern_english', '')})" 
                                           for v in name_variants if isinstance(v, dict)])
            
            associated_places = entry.get("associated_places", [])
            places_str = "; ".join([f"{p.get('place_original', '')} - {p.get('association_type', '')}" 
                                    for p in associated_places if isinstance(p, dict)])
            
            associated_works = entry.get("associated_works", [])
            works_str = "; ".join([f"{w.get('title_original', '')} ({w.get('role', '')})" 
                                   for w in associated_works if isinstance(w, dict)])
            
            sources = entry.get("sources", [])
            sources_str = "; ".join([f"{s.get('author', '')} - {s.get('title', '')} ({s.get('year', '')})" 
                                     for s in sources if isinstance(s, dict)])
            
            links = entry.get("links", [])
            links_str = "; ".join([f"{l.get('entity_type', '')}: {l.get('entity_label', '')} - {l.get('relationship', '')}" 
                                   for l in links if isinstance(l, dict)])
            
            row = {
                "canonical_name_original": canonical_orig,
                "canonical_name_modern_english": canonical_modern,
                "gender": gender,
                "roles": ", ".join(roles) if isinstance(roles, list) else "",
                "period_start_year": period_start,
                "period_end_year": period_end,
                "period_notation": period_notation,
                "name_variants": name_variants_str,
                "associated_places": places_str,
                "associated_works": works_str,
                "notes": notes,
                "sources": sources_str,
                "links": links_str
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df

    def _convert_culinary_places_to_df(self, entries: List[Any]) -> pd.DataFrame:
        """
        Converts culinary places entries to DataFrame according to schema v2.0.
        Handles nested arrays for roles, products, establishments, people, and links.
        """
        # Filter out None entries
        if entries is None:
            entries = []
        entries = [entry for entry in entries if entry is not None]
        rows = []
        for entry in entries:
            # Extract basic fields
            name_orig = entry.get("name_original")
            name_modern = entry.get("name_modern_english")
            place_type = entry.get("place_type")
            country_modern = entry.get("country_modern")
            notes = entry.get("notes")
            
            # Extract period
            period = entry.get("period", {})
            period_start = period.get("start_year") if isinstance(period, dict) else None
            period_end = period.get("end_year") if isinstance(period, dict) else None
            period_notation = period.get("notation") if isinstance(period, dict) else None
            
            # Extract and format arrays
            roles = entry.get("roles_in_culinary_ecosystem", [])
            roles_str = ", ".join(roles) if isinstance(roles, list) else ""
            
            products = entry.get("associated_products", [])
            products_str = ", ".join(products) if isinstance(products, list) else ""
            
            establishments = entry.get("notable_establishments", [])
            establishments_str = ", ".join(establishments) if isinstance(establishments, list) else ""
            
            people = entry.get("associated_people", [])
            people_str = "; ".join([f"{p.get('name_original', '')} - {p.get('association_type', '')}" 
                                    for p in people if isinstance(p, dict)])
            
            links = entry.get("links", [])
            links_str = "; ".join([f"{l.get('entity_type', '')}: {l.get('entity_label', '')} - {l.get('relationship', '')}" 
                                   for l in links if isinstance(l, dict)])
            
            row = {
                "name_original": name_orig,
                "name_modern_english": name_modern,
                "place_type": place_type,
                "country_modern": country_modern,
                "period_start_year": period_start,
                "period_end_year": period_end,
                "period_notation": period_notation,
                "roles_in_culinary_ecosystem": roles_str,
                "associated_products": products_str,
                "notable_establishments": establishments_str,
                "associated_people": people_str,
                "notes": notes,
                "links": links_str
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df

    def _convert_culinary_works_to_df(self, entries: List[Any]) -> pd.DataFrame:
        """
        Converts culinary works entries to DataFrame according to schema v2.0.
        Handles nested arrays for culinary_focus, languages, contributors, edition_years, places, and links.
        """
        # Filter out None entries
        if entries is None:
            entries = []
        entries = [entry for entry in entries if entry is not None]
        rows = []
        for entry in entries:
            # Extract basic fields
            title_orig = entry.get("title_original")
            title_modern = entry.get("title_modern_english")
            short_title = entry.get("short_title")
            description = entry.get("description")
            genre = entry.get("genre")
            notes = entry.get("notes")
            
            # Extract and format arrays
            culinary_focus = entry.get("culinary_focus", [])
            culinary_focus_str = ", ".join(culinary_focus) if isinstance(culinary_focus, list) else ""
            
            languages = entry.get("languages", [])
            languages_str = ", ".join(languages) if isinstance(languages, list) else ""
            
            contributors = entry.get("contributors", [])
            contributors_str = "; ".join([f"{c.get('name_original', '')} ({c.get('role', '')})" 
                                          for c in contributors if isinstance(c, dict)])
            
            edition_years = entry.get("edition_years", [])
            edition_years_str = ", ".join([str(y) for y in edition_years if y is not None]) if isinstance(edition_years, list) else ""
            
            pub_places = entry.get("publication_places", [])
            pub_places_str = "; ".join([f"{p.get('name_original', '')} ({p.get('name_modern_english', '')})" 
                                        for p in pub_places if isinstance(p, dict)])
            
            assoc_places = entry.get("associated_places", [])
            assoc_places_str = "; ".join([f"{p.get('name_original', '')} - {p.get('association_type', '')}" 
                                          for p in assoc_places if isinstance(p, dict)])
            
            assoc_persons = entry.get("associated_persons", [])
            assoc_persons_str = "; ".join([f"{p.get('name_original', '')} - {p.get('association_type', '')}" 
                                           for p in assoc_persons if isinstance(p, dict)])
            
            links = entry.get("links", [])
            links_str = "; ".join([f"{l.get('entity_type', '')}: {l.get('entity_label', '')} - {l.get('relationship', '')}" 
                                   for l in links if isinstance(l, dict)])
            
            row = {
                "title_original": title_orig,
                "title_modern_english": title_modern,
                "short_title": short_title,
                "description": description,
                "genre": genre,
                "culinary_focus": culinary_focus_str,
                "languages": languages_str,
                "contributors": contributors_str,
                "edition_years": edition_years_str,
                "publication_places": pub_places_str,
                "associated_places": assoc_places_str,
                "associated_persons": assoc_persons_str,
                "notes": notes,
                "links": links_str
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df

    def _convert_historical_recipes_to_df(self, entries: List[Any]) -> pd.DataFrame:
        """
        Converts historical recipes entries to DataFrame according to schema v2.2.
        Handles deeply nested structures for ingredients, cooking methods, utensils, and more.
        """
        # Filter out None entries
        if entries is None:
            entries = []
        entries = [entry for entry in entries if entry is not None]
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
                    name = ing.get("name_modern_english", ing.get("name_original", ""))
                    qty_val = ing.get("quantity_standardized_value")
                    qty_unit = ing.get("quantity_standardized_unit")
                    qty_str = f"{qty_val} {qty_unit}" if qty_val and qty_unit else ""
                    prep = ing.get("preparation_note_modern_english", "")
                    ing_str = f"{name} ({qty_str}) {prep}".strip()
                    ingredients_list.append(ing_str)
            ingredients_str = "; ".join(ingredients_list)
            
            # Extract cooking methods
            methods = entry.get("cooking_methods", [])
            methods_str = ", ".join([m.get("method_modern_english", m.get("method_original", "")) 
                                     for m in methods if isinstance(m, dict)])
            
            # Extract utensils
            utensils = entry.get("utensils_equipment", [])
            utensils_str = ", ".join([u.get("utensil_modern_english", u.get("utensil_original", "")) 
                                      for u in utensils if isinstance(u, dict)])
            
            # Extract yield
            yields = entry.get("yield", [])
            yield_str = ""
            if yields and isinstance(yields, list) and len(yields) > 0:
                y = yields[0]
                if isinstance(y, dict):
                    val = y.get("value_modern_english")
                    unit = y.get("unit_modern_english")
                    if val and unit:
                        yield_str = f"{val} {unit}"
            
            # Extract times
            prep_times = entry.get("preparation_time", [])
            prep_time_str = ""
            if prep_times and isinstance(prep_times, list) and len(prep_times) > 0:
                t = prep_times[0]
                if isinstance(t, dict):
                    val = t.get("value_modern_english")
                    unit = t.get("unit_modern_english")
                    if val and unit:
                        prep_time_str = f"{val} {unit}"
            
            cook_times = entry.get("cooking_time", [])
            cook_time_str = ""
            if cook_times and isinstance(cook_times, list) and len(cook_times) > 0:
                t = cook_times[0]
                if isinstance(t, dict):
                    val = t.get("value_modern_english")
                    unit = t.get("unit_modern_english")
                    if val and unit:
                        cook_time_str = f"{val} {unit}"
            
            # Extract ingredient categories
            categories = entry.get("ingredient_categories", {})
            
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
                "contains_meat": categories.get("contains_meat", False) if isinstance(categories, dict) else False,
                "contains_poultry": categories.get("contains_poultry", False) if isinstance(categories, dict) else False,
                "contains_fish_seafood": categories.get("contains_fish_seafood", False) if isinstance(categories, dict) else False,
                "contains_dairy": categories.get("contains_dairy", False) if isinstance(categories, dict) else False,
                "contains_eggs": categories.get("contains_eggs", False) if isinstance(categories, dict) else False,
                "contains_butter": categories.get("contains_butter", False) if isinstance(categories, dict) else False,
                "contains_olive_oil": categories.get("contains_olive_oil", False) if isinstance(categories, dict) else False,
                "contains_lard_animal_fat": categories.get("contains_lard_animal_fat", False) if isinstance(categories, dict) else False,
                "contains_alcohol": categories.get("contains_alcohol", False) if isinstance(categories, dict) else False,
                "contains_refined_sugar": categories.get("contains_refined_sugar", False) if isinstance(categories, dict) else False,
                "contains_honey": categories.get("contains_honey", False) if isinstance(categories, dict) else False,
                "contains_other_sweeteners": categories.get("contains_other_sweeteners", False) if isinstance(categories, dict) else False,
                "contains_foreign_spices": categories.get("contains_foreign_spices", False) if isinstance(categories, dict) else False,
                "contains_luxury_ingredients": categories.get("contains_luxury_ingredients", False) if isinstance(categories, dict) else False
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df

    def _convert_michelin_guides_to_df(self, entries: List[Any]) -> pd.DataFrame:
        """
        Converts Michelin Guide entries to DataFrame according to schema v1.1.
        Handles deeply nested structures for location, address, awards, cuisine, pricing, amenities, etc.
        """
        if entries is None:
            entries = []
        entries = [entry for entry in entries if entry is not None]
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
