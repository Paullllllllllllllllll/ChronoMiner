# data_processing.py

from pathlib import Path
import json
from modules.core.json_utils import extract_entries_from_json
import pandas as pd
from typing import Any, List, Dict

class CSVConverter:
    """
    A converter class to transform JSON extracted data into CSV format.
    """
    def __init__(self, schema_name: str) -> None:
        self.schema_name: str = schema_name.lower()

    def extract_entries(self, json_file: Path) -> List[Any]:
        """
        Extract entries from JSON file.

        :param json_file: Path to the JSON file
        :return: List of entries extracted from the JSON file
        """
        return extract_entries_from_json(json_file)

    def convert_to_csv(self, json_file: Path, output_csv: Path) -> None:
        entries: List[Any] = self.extract_entries(json_file)
        if not entries:
            print("No entries found for CSV conversion.")
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
            "historicalrecipesentries": self._convert_historical_recipes_to_df
        }
        key = self.schema_name.lower()
        if key in converters:
            df = converters[key](entries)
        else:
            df = pd.json_normalize(entries, sep='_')
        try:
            df.to_csv(output_csv, index=False)
            print(f"CSV file generated at {output_csv}")
        except Exception as e:
            print(f"Error saving CSV file {output_csv}: {e}")

    def _convert_bibliographic_entries_to_df(self, entries: List[
        Any]) -> pd.DataFrame:
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

    def _convert_historicaladdressbookentries_to_df(self, entries: List[
        Any]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for entry in entries:
            address = entry.get("address", {})
            street = address.get("street", "Unknown")
            street_number = address.get("street_number", "*0*")
            row: Dict[str, Any] = {
                "last_name": entry.get("last_name", ""),
                "first_name": entry.get("first_name", ""),
                "street": street,
                "street_number": street_number,
                "occupation": entry.get("occupation", ""),
                "section": entry.get("section"),
                "honorific": entry.get("honorific"),
                "additional_notes": entry.get("additional_notes")
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def _convert_brazilianoccupationrecords_to_df(self, entries: List[Any]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for entry in entries:
            officials = entry.get("officials", [])
            if officials is None:
                officials_str = ""
            else:
                officials_str = "; ".join(
                    [f"{o.get('position', '')}: {o.get('signature', '')}" for o in officials]
                )
            row = {
                "surname": entry.get("surname", ""),
                "first_name": entry.get("first_name", ""),
                "record_header": entry.get("record_header", ""),
                "location": entry.get("location", ""),
                "height": entry.get("height", ""),
                "skin_color": entry.get("skin_color", ""),
                "hair_color": entry.get("hair_color", ""),
                "hair_texture": entry.get("hair_texture", ""),
                "beard": entry.get("beard", ""),
                "mustache": entry.get("mustache", ""),
                "assignatura": entry.get("assignatura", ""),
                "reservista": entry.get("reservista", ""),
                "eyes": entry.get("eyes", ""),
                "mouth": entry.get("mouth", ""),
                "face": entry.get("face", ""),
                "nose": entry.get("nose", ""),
                "marks": entry.get("marks", ""),
                "officials": officials_str,
                "father": entry.get("father", ""),
                "mother": entry.get("mother", ""),
                "birth_date": entry.get("birth_date", ""),
                "birth_place": entry.get("birth_place", ""),
                "municipality": entry.get("municipality", ""),
                "profession": entry.get("profession", ""),
                "civil_status": entry.get("civil_status", ""),
                "vaccinated": entry.get("vaccinated", ""),
                "can_read": entry.get("can_read", ""),
                "can_write": entry.get("can_write", ""),
                "can_count": entry.get("can_count", ""),
                "swimming": entry.get("swimming", ""),
                "cyclist": entry.get("cyclist", ""),
                "motorcyclist": entry.get("motorcyclist", ""),
                "driver": entry.get("driver", ""),
                "chauffeur": entry.get("chauffeur", ""),
                "telegraphist": entry.get("telegraphist", ""),
                "telephonist": entry.get("telephonist", ""),
                "residence": entry.get("residence", ""),
                "observations": entry.get("observations", "")
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def _convert_culinary_persons_to_df(self, entries: List[Any]) -> pd.DataFrame:
        """
        Converts culinary persons entries to DataFrame according to schema v2.0.
        Handles nested arrays for name_variants, associated_places, works, sources, and links.
        """
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
