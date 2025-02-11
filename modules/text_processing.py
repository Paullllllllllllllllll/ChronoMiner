# modules/text_processing.py

from pathlib import Path
import json
from docx import Document
from typing import Any, List

class DocumentConverter:
    """
    Converts JSON-extracted data to DOCX or TXT documents.
    """
    def __init__(self, schema_name: str) -> None:
        self.schema_name: str = schema_name.lower()

    def extract_entries(self, json_file: Path) -> List[Any]:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON file {json_file}: {e}")
            return []
        entries: List[Any] = []
        if isinstance(data, dict):
            if "entries" in data:
                entries = data["entries"]
            elif "responses" in data:
                for resp in data["responses"]:
                    try:
                        body = resp.get("body", {})
                        choices = body.get("choices", [])
                        if choices:
                            message = choices[0].get("message", {})
                            content = message.get("content", "")
                            content_json = json.loads(content)
                            if isinstance(content_json, dict) and "entries" in content_json:
                                entries.extend(content_json["entries"])
                    except Exception as e:
                        print(f"Error processing response: {e}")
        elif isinstance(data, list):
            for item in data:
                response = item.get("response")
                if isinstance(response, str):
                    try:
                        response_obj = json.loads(response)
                        if isinstance(response_obj, dict) and "entries" in response_obj:
                            entries.extend(response_obj["entries"])
                    except Exception as e:
                        print(f"Error parsing response string: {e}")
                elif isinstance(response, dict) and "entries" in response:
                    entries.extend(response["entries"])
        return entries

    def convert_to_docx(self, json_file: Path, output_file: Path) -> None:
        entries: List[Any] = self.extract_entries(json_file)
        document: Document = Document()
        document.add_heading(json_file.stem, 0)
        converters = {
            "structuredsummaries": self._convert_structured_summaries_to_docx,
            "bibliographicentries": self._convert_bibliographic_entries_to_docx,
            "recipes": self._convert_recipes_to_docx,
            "historicaladdressbookentries": self._convert_historicaladdressbookentries_to_docx,
            "brazilianoccupationrecords": self._convert_brazilianoccupationrecords_to_docx
        }
        key = self.schema_name.lower()
        if key in converters:
            converters[key](entries, document)
        else:
            for entry in entries:
                document.add_paragraph(str(entry))
        try:
            document.save(output_file)
            print(f"DOCX file generated at {output_file}")
        except Exception as e:
            print(f"Error saving DOCX file {output_file}: {e}")

    def convert_to_txt(self, json_file: Path, output_file: Path) -> None:
        entries: List[Any] = self.extract_entries(json_file)
        lines: List[str] = []
        converters = {
            "structuredsummaries": self._convert_structured_summaries_to_txt,
            "bibliographicentries": self._convert_bibliographic_entries_to_txt,
            "recipes": self._convert_recipes_to_txt,
            "historicaladdressbookentries": self._convert_historicaladdressbookentries_to_txt,
            "brazilianoccupationrecords": self._convert_brazilianoccupationrecords_to_txt
        }
        key = self.schema_name.lower()
        if key in converters:
            lines = converters[key](entries)
        else:
            for entry in entries:
                lines.append(str(entry))
        try:
            with output_file.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"TXT file generated at {output_file}")
        except Exception as e:
            print(f"Error saving TXT file {output_file}: {e}")

    # --- Schema-Specific DOCX Converters ---
    def _convert_structured_summaries_to_docx(self, entries: List[Any], document: Document) -> None:
        literature_set = set()
        for entry in entries:
            page = entry.get("page")
            bullet_points = entry.get("bullet_points")
            literature = entry.get("literature")
            if page is not None:
                document.add_heading(f"Page {page}", level=1)
            else:
                document.add_heading("Page Unknown", level=1)
            if bullet_points and isinstance(bullet_points, list):
                for bp in bullet_points:
                    if bp:
                        p = document.add_paragraph(style='List Bullet')
                        p.add_run(str(bp))
            else:
                document.add_paragraph("No bullet points available.")
            if literature and isinstance(literature, list):
                for lit in literature:
                    if lit:
                        literature_set.add(lit)
            document.add_paragraph("")
        if literature_set:
            document.add_page_break()
            document.add_heading("Literature", level=1)
            for lit in sorted(literature_set):
                document.add_paragraph(str(lit), style='List Bullet')

    def _convert_bibliographic_entries_to_docx(self, entries: List[Any], document: Document) -> None:
        for entry in entries:
            full_title = entry.get("full_title", "Unknown Title")
            short_title = entry.get("short_title", "")
            bibliography_number = entry.get("bibliography_number", "")
            authors = entry.get("authors", [])
            roles = entry.get("roles", [])
            culinary_focus = entry.get("culinary_focus", [])
            book_format = entry.get("format", "")
            pages = entry.get("pages", "")
            edition_info = entry.get("edition_info", [])
            total_editions = entry.get("total_editions", "")
            document.add_heading(full_title, level=1)
            document.add_paragraph(f"Short Title: {short_title}")
            document.add_paragraph(f"Bibliography Number: {bibliography_number}")
            document.add_paragraph(f"Authors: {', '.join(authors) if authors else 'Anonymous'}")
            document.add_paragraph(f"Roles: {', '.join(roles)}")
            document.add_paragraph(f"Culinary Focus: {', '.join(culinary_focus)}")
            document.add_paragraph(f"Format: {book_format}")
            document.add_paragraph(f"Pages: {pages}")
            document.add_paragraph(f"Total Editions: {total_editions}")
            document.add_heading("Edition Information", level=2)
            for edition in edition_info:
                year = edition.get("year", "Unknown")
                edition_number = edition.get("edition_number", "Unknown")
                location = edition.get("location", {})
                country = location.get("country", "Unknown")
                city = location.get("city", "Unknown")
                ed_roles = edition.get("roles", [])
                short_note = edition.get("short_note", "")
                edition_category = edition.get("edition_category", "")
                language = edition.get("language", "")
                original_language = edition.get("original_language", "")
                translated_from = edition.get("translated_from", "")
                edition_text = (
                    f"Year: {year}, Edition: {edition_number}, "
                    f"Location: {city}, {country}, Roles: {', '.join(ed_roles)}, "
                    f"Note: {short_note}, Category: {edition_category}, "
                    f"Language: {language}, Orig Lang: {original_language}, "
                    f"Translated From: {translated_from}"
                )
                document.add_paragraph(edition_text, style='List Bullet')
            document.add_page_break()

    def _convert_recipes_to_docx(self, entries: List[Any], document: Document) -> None:
        for entry in entries:
            name = entry.get("Name", "Unknown")
            original_text = entry.get("Original_Text", "")
            translated_text = entry.get("Translated_Text", "")
            ingredients = entry.get("Ingredients", [])
            servings = entry.get("Servings", "")
            category = entry.get("Category", "")
            original_language = entry.get("Original_Language", "")
            cooking_methods = entry.get("cooking_methods", [])
            cooking_utensils = entry.get("cooking_utensils", [])
            document.add_heading(name, level=1)
            document.add_paragraph("Original Text:")
            document.add_paragraph(original_text)
            document.add_paragraph("Translated Text:")
            document.add_paragraph(translated_text)
            document.add_paragraph(f"Servings: {servings}")
            document.add_paragraph(f"Category: {category}")
            document.add_paragraph(f"Original Language: {original_language}")
            document.add_heading("Ingredients", level=2)
            for ingredient in ingredients:
                historical_name = ingredient.get("historical_name", "")
                modern_name = ingredient.get("modern_name", "")
                quantity = ingredient.get("quantity", "")
                unit = ingredient.get("unit", "")
                document.add_paragraph(
                    f"{historical_name} ({modern_name}), Qty: {quantity} {unit}",
                    style='List Bullet'
                )
            document.add_heading("Cooking Methods", level=2)
            for method in cooking_methods:
                historical_method = method.get("historical_method", "")
                modern_method = method.get("modern_method", "")
                document.add_paragraph(
                    f"{historical_method} -> {modern_method}",
                    style='List Bullet'
                )
            document.add_heading("Cooking Utensils", level=2)
            for utensil in cooking_utensils:
                historical_utensil = utensil.get("historical_utensil", "")
                modern_utensil = utensil.get("modern_utensil", "")
                document.add_paragraph(
                    f"{historical_utensil} -> {modern_utensil}",
                    style='List Bullet'
                )
            document.add_page_break()

    def _convert_historicaladdressbookentries_to_docx(self, entries: List[Any],
                                                      document: Document) -> None:
        for entry in entries:
            last_name = entry.get("last_name", "Unknown")
            first_name = entry.get("first_name", "Unknown")
            occupation = entry.get("occupation", "Unknown")
            section = entry.get("section")
            honorific = entry.get("honorific")
            additional_notes = entry.get("additional_notes")

            header = f"{last_name}, {first_name} - {occupation}"
            if section:
                header += f" (Section: {section})"
            document.add_heading(header, level=1)

            # Extract and add address information.
            address = entry.get("address", {})
            street = address.get("street", "Unknown")
            street_number = address.get("street_number", "*0*")
            document.add_paragraph(f"Address: {street} {street_number}")

            details = []
            if honorific:
                details.append(f"Honorific: {honorific}")
            if additional_notes:
                details.append(f"Additional Notes: {additional_notes}")
            if details:
                document.add_paragraph("; ".join(details))
            document.add_page_break()

    def _convert_brazilianoccupationrecords_to_docx(self, entries: List[Any], document: Document) -> None:
        for entry in entries:
            header = f"{entry.get('surname', '')}, {entry.get('first_name', '')} - {entry.get('profession', '')}"
            document.add_heading(header, level=1)
            document.add_paragraph(f"Record Header: {entry.get('record_header', '')}")
            document.add_paragraph(f"Location: {entry.get('location', '')}")
            document.add_paragraph(f"Height: {entry.get('height', '')}")
            document.add_paragraph(f"Skin Color: {entry.get('skin_color', '')}")
            document.add_paragraph(f"Hair Color: {entry.get('hair_color', '')}")
            document.add_paragraph(f"Hair Texture: {entry.get('hair_texture', '')}")
            document.add_paragraph(f"Beard: {entry.get('beard', '')}")
            document.add_paragraph(f"Mustache: {entry.get('mustache', '')}")
            document.add_paragraph(f"Assignatura: {entry.get('assignatura', '')}")
            document.add_paragraph(f"Reservista: {entry.get('reservista', '')}")
            document.add_paragraph(f"Eyes: {entry.get('eyes', '')}")
            document.add_paragraph(f"Mouth: {entry.get('mouth', '')}")
            document.add_paragraph(f"Face: {entry.get('face', '')}")
            document.add_paragraph(f"Nose: {entry.get('nose', '')}")
            document.add_paragraph(f"Marks: {entry.get('marks', '')}")
            officials = entry.get("officials", [])
            if officials:
                officials_str = "; ".join(
                    [f"{o.get('position', '')}: {o.get('signature', '')}" for o in officials]
                )
            else:
                officials_str = ""
            document.add_paragraph(f"Officials: {officials_str}")
            document.add_paragraph(f"Father: {entry.get('father', '')}")
            document.add_paragraph(f"Mother: {entry.get('mother', '')}")
            document.add_paragraph(f"Birth Date: {entry.get('birth_date', '')}")
            document.add_paragraph(f"Birth Place: {entry.get('birth_place', '')}")
            document.add_paragraph(f"Municipality: {entry.get('municipality', '')}")
            document.add_paragraph(f"Civil Status: {entry.get('civil_status', '')}")
            document.add_paragraph(f"Vaccinated: {entry.get('vaccinated', '')}")
            document.add_paragraph(f"Can Read: {entry.get('can_read', '')}")
            document.add_paragraph(f"Can Write: {entry.get('can_write', '')}")
            document.add_paragraph(f"Can Count: {entry.get('can_count', '')}")
            document.add_paragraph(f"Swimming: {entry.get('swimming', '')}")
            document.add_paragraph(f"Cyclist: {entry.get('cyclist', '')}")
            document.add_paragraph(f"Motorcyclist: {entry.get('motorcyclist', '')}")
            document.add_paragraph(f"Driver: {entry.get('driver', '')}")
            document.add_paragraph(f"Chauffeur: {entry.get('chauffeur', '')}")
            document.add_paragraph(f"Telegraphist: {entry.get('telegraphist', '')}")
            document.add_paragraph(f"Telephonist: {entry.get('telephonist', '')}")
            document.add_paragraph(f"Residence: {entry.get('residence', '')}")
            document.add_paragraph(f"Observations: {entry.get('observations', '')}")
            document.add_page_break()

    # --- Schema-Specific TXT Converters ---
    def _convert_structured_summaries_to_txt(self, entries: List[Any]) -> List[str]:
        lines: List[str] = []
        literature_set = set()
        for entry in entries:
            page = entry.get("page", "Unknown")
            bullet_points = entry.get("bullet_points", [])
            literature = entry.get("literature", [])
            lines.append(f"Page {page}")
            if bullet_points and isinstance(bullet_points, list):
                for bp in bullet_points:
                    if bp:
                        lines.append(f" - {bp}")
            else:
                lines.append("No bullet points available.")
            if literature and isinstance(literature, list):
                for lit in literature:
                    if lit:
                        literature_set.add(lit)
            lines.append("")
        if literature_set:
            lines.append("Literature:")
            for lit in sorted(literature_set):
                lines.append(f" - {lit}")
        return lines

    def _convert_bibliographic_entries_to_txt(self, entries: List[Any]) -> List[str]:
        lines: List[str] = []
        for entry in entries:
            lines.append(f"Full Title: {entry.get('full_title', 'Unknown Title')}")
            lines.append(f"Short Title: {entry.get('short_title', '')}")
            lines.append(f"Bibliography Number: {entry.get('bibliography_number', '')}")
            authors = entry.get("authors", [])
            lines.append(f"Authors: {', '.join(authors) if authors else 'Anonymous'}")
            lines.append(f"Roles: {', '.join(entry.get('roles', []))}")
            lines.append(f"Culinary Focus: {', '.join(entry.get('culinary_focus', []))}")
            lines.append(f"Format: {entry.get('format', '')}")
            lines.append(f"Pages: {entry.get('pages', '')}")
            lines.append(f"Total Editions: {entry.get('total_editions', '')}")
            lines.append("Edition Information:")
            for edition in entry.get("edition_info", []):
                edition_text = (
                    f"Year: {edition.get('year', 'Unknown')}, "
                    f"Edition: {edition.get('edition_number', 'Unknown')}, "
                    f"Location: {edition.get('location', {}).get('city', 'Unknown')}, "
                    f"{edition.get('location', {}).get('country', 'Unknown')}, "
                    f"Roles: {', '.join(edition.get('roles', []))}, "
                    f"Note: {edition.get('short_note', '')}, "
                    f"Category: {edition.get('edition_category', '')}, "
                    f"Language: {edition.get('language', '')}, "
                    f"Orig Lang: {edition.get('original_language', '')}, "
                    f"Translated From: {edition.get('translated_from', '')}"
                )
                lines.append(f" - {edition_text}")
            lines.append("\n" + "=" * 40 + "\n")
        return lines

    def _convert_recipes_to_txt(self, entries: List[Any]) -> List[str]:
        lines: List[str] = []
        for entry in entries:
            lines.append(f"Recipe Name: {entry.get('Name', 'Unknown')}")
            lines.append("Original Text:")
            lines.append(entry.get("Original_Text", ""))
            lines.append("Translated Text:")
            lines.append(entry.get("Translated_Text", ""))
            lines.append(f"Servings: {entry.get('Servings', '')}")
            lines.append(f"Category: {entry.get('Category', '')}")
            lines.append(f"Original Language: {entry.get('Original_Language', '')}")
            lines.append("Ingredients:")
            for ingredient in entry.get("Ingredients", []):
                lines.append(
                    f" - {ingredient.get('historical_name', '')} "
                    f"({ingredient.get('modern_name', '')}), Qty: {ingredient.get('quantity', '')} {ingredient.get('unit', '')}"
                )
            lines.append("Cooking Methods:")
            for method in entry.get("cooking_methods", []):
                lines.append(
                    f" - {method.get('historical_method', '')} -> {method.get('modern_method', '')}"
                )
            lines.append("Cooking Utensils:")
            for utensil in entry.get("cooking_utensils", []):
                lines.append(
                    f" - {utensil.get('historical_utensil', '')} -> {utensil.get('modern_utensil', '')}"
                )
            lines.append("\n" + "=" * 40 + "\n")
        return lines

    def _convert_historicaladdressbookentries_to_txt(self,
                                                     entries: List[Any]) -> \
    List[str]:
        lines: List[str] = []
        for entry in entries:
            last_name = entry.get("last_name", "Unknown")
            first_name = entry.get("first_name", "Unknown")
            occupation = entry.get("occupation", "Unknown")
            section = entry.get("section")
            honorific = entry.get("honorific")
            additional_notes = entry.get("additional_notes")

            header = f"{last_name}, {first_name} - {occupation}"
            if section:
                header += f" (Section: {section})"
            lines.append(header)

            # Extract and append address information.
            address = entry.get("address", {})
            street = address.get("street", "Unknown")
            street_number = address.get("street_number", "*0*")
            lines.append(f"Address: {street} {street_number}")

            if honorific:
                lines.append(f"Honorific: {honorific}")
            if additional_notes:
                lines.append(f"Additional Notes: {additional_notes}")
            lines.append("")
        return lines

    def _convert_brazilianoccupationrecords_to_txt(self, entries: List[Any]) -> List[str]:
        lines: List[str] = []
        for entry in entries:
            header = f"{entry.get('surname', '')}, {entry.get('first_name', '')} - {entry.get('profession', '')}"
            lines.append(header)
            lines.append(f"Record Header: {entry.get('record_header', '')}")
            lines.append(f"Location: {entry.get('location', '')}")
            lines.append(f"Height: {entry.get('height', '')}")
            lines.append(f"Skin Color: {entry.get('skin_color', '')}")
            lines.append(f"Hair Color: {entry.get('hair_color', '')}")
            lines.append(f"Hair Texture: {entry.get('hair_texture', '')}")
            lines.append(f"Beard: {entry.get('beard', '')}")
            lines.append(f"Mustache: {entry.get('mustache', '')}")
            lines.append(f"Assignatura: {entry.get('assignatura', '')}")
            lines.append(f"Reservista: {entry.get('reservista', '')}")
            lines.append(f"Eyes: {entry.get('eyes', '')}")
            lines.append(f"Mouth: {entry.get('mouth', '')}")
            lines.append(f"Face: {entry.get('face', '')}")
            lines.append(f"Nose: {entry.get('nose', '')}")
            lines.append(f"Marks: {entry.get('marks', '')}")
            officials = entry.get("officials", [])
            if officials:
                officials_str = "; ".join(
                    [f"{o.get('position', '')}: {o.get('signature', '')}" for o in officials]
                )
            else:
                officials_str = ""
            lines.append(f"Officials: {officials_str}")
            lines.append(f"Father: {entry.get('father', '')}")
            lines.append(f"Mother: {entry.get('mother', '')}")
            lines.append(f"Birth Date: {entry.get('birth_date', '')}")
            lines.append(f"Birth Place: {entry.get('birth_place', '')}")
            lines.append(f"Municipality: {entry.get('municipality', '')}")
            lines.append(f"Civil Status: {entry.get('civil_status', '')}")
            lines.append(f"Vaccinated: {entry.get('vaccinated', '')}")
            lines.append(f"Can Read: {entry.get('can_read', '')}")
            lines.append(f"Can Write: {entry.get('can_write', '')}")
            lines.append(f"Can Count: {entry.get('can_count', '')}")
            lines.append(f"Swimming: {entry.get('swimming', '')}")
            lines.append(f"Cyclist: {entry.get('cyclist', '')}")
            lines.append(f"Motorcyclist: {entry.get('motorcyclist', '')}")
            lines.append(f"Driver: {entry.get('driver', '')}")
            lines.append(f"Chauffeur: {entry.get('chauffeur', '')}")
            lines.append(f"Telegraphist: {entry.get('telegraphist', '')}")
            lines.append(f"Telephonist: {entry.get('telephonist', '')}")
            lines.append(f"Residence: {entry.get('residence', '')}")
            lines.append(f"Observations: {entry.get('observations', '')}")
            lines.append("\n" + "=" * 40 + "\n")
        return lines
