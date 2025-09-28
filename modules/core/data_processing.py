# data_processing.py

from pathlib import Path
import json
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
                for resp in data.get("responses", []):
                    if resp is None:
                        continue  # Skip None responses

                    try:
                        if isinstance(resp, str):
                            # Try to parse response string as JSON
                            content_json = json.loads(resp)
                            if isinstance(content_json, dict) and "entries" in content_json:
                                # Filter out None entries
                                valid_entries = [entry for entry in content_json.get("entries", []) if entry is not None]
                                entries.extend(valid_entries)
                        elif isinstance(resp, dict):
                            # Handle batch response structure (Chat Completions and Responses API)
                            body = resp.get("body", {}) if isinstance(resp, dict) else {}
                            content = ""
                            if body:
                                # Chat Completions path
                                choices = body.get("choices", []) if isinstance(body, dict) else []
                                if choices:
                                    message = choices[0].get("message", {}) if choices[0] else {}
                                    content = message.get("content", "") if message else ""
                                else:
                                    # Responses API path: prefer output_text, else concatenate output message texts
                                    if isinstance(body.get("output_text"), str):
                                        content = body.get("output_text", "")
                                    elif isinstance(body.get("output"), list):
                                        parts = []
                                        for item in body.get("output", []):
                                            if isinstance(item, dict) and item.get("type") == "message":
                                                for c in item.get("content", []):
                                                    txt = c.get("text") if isinstance(c, dict) else None
                                                    if isinstance(txt, str):
                                                        parts.append(txt)
                                        content = "".join(parts)
                            if content:
                                try:
                                    content_json = json.loads(content)
                                    if isinstance(content_json, dict) and "entries" in content_json:
                                        # Filter out None entries
                                        valid_entries = [entry for entry in content_json.get("entries", []) if entry is not None]
                                        entries.extend(valid_entries)
                                except json.JSONDecodeError:
                                    print(f"Failed to parse response content as JSON: {content[:100]}...")
                    except Exception as e:
                        print(f"Error processing response: {e}")
        elif isinstance(data, list):
            for item in data:
                if item is None:
                    continue  # Skip None items

                response = item.get("response") if isinstance(item,
                                                              dict) else None
                if response is None:
                    continue  # Skip items with None response

                if isinstance(response, str):
                    try:
                        response_obj = json.loads(response)
                        if isinstance(response_obj,
                                      dict) and "entries" in response_obj:
                            # Filter out None entries
                            valid_entries = [entry for entry in
                                             response_obj.get("entries", []) if
                                             entry is not None]
                            entries.extend(valid_entries)
                    except json.JSONDecodeError:
                        print(
                            f"Error parsing response string as JSON: {response[:100]}...")
                    except Exception as e:
                        print(f"Error processing string response: {e}")
                elif isinstance(response, dict):
                    # Handle dict responses: either already parsed entries or raw API body
                    if "entries" in response:
                        # Filter out None entries
                        valid_entries = [entry for entry in response.get("entries", []) if entry is not None]
                        entries.extend(valid_entries)
                    else:
                        body = response.get("body", {}) if isinstance(response, dict) else {}
                        content = ""
                        if body:
                            choices = body.get("choices", []) if isinstance(body, dict) else []
                            if choices:
                                message = choices[0].get("message", {}) if choices[0] else {}
                                content = message.get("content", "") if message else ""
                            else:
                                if isinstance(body.get("output_text"), str):
                                    content = body.get("output_text", "")
                                elif isinstance(body.get("output"), list):
                                    parts = []
                                    for item in body.get("output", []):
                                        if isinstance(item, dict) and item.get("type") == "message":
                                            for c in item.get("content", []):
                                                txt = c.get("text") if isinstance(c, dict) else None
                                                if isinstance(txt, str):
                                                    parts.append(txt)
                                    content = "".join(parts)
                        if content:
                            try:
                                content_json = json.loads(content)
                                if isinstance(content_json, dict) and "entries" in content_json:
                                    valid_entries = [entry for entry in content_json.get("entries", []) if entry is not None]
                                    entries.extend(valid_entries)
                            except json.JSONDecodeError:
                                print(f"Error parsing response content as JSON: {content[:100]}...")

        return entries

    def convert_to_csv(self, json_file: Path, output_csv: Path) -> None:
        entries: List[Any] = self.extract_entries(json_file)
        if not entries:
            print("No entries found for CSV conversion.")
            return

        converters = {
            "bibliographicentries": self._convert_bibliographic_entries_to_df,
            "structuredsummaries": self._convert_structured_summaries_to_df,
            "historicaladdressbookentries": self._convert_historicaladdressbookentries_to_df,
            "brazilianoccupationrecords": self._convert_brazilianoccupationrecords_to_df
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
        Converts bibliographic entries to a pandas DataFrame according to schema version 2.9.

        :param entries: List of bibliographic entry dictionaries
        :return: pandas DataFrame with normalized bibliographic data
        """
        rows = []

        for entry in entries:
            # Extract primary entry data
            full_title = entry.get("full_title", "")
            short_title = entry.get("short_title", "")
            culinary_focus = entry.get("culinary_focus", [])
            book_format = entry.get("format", None)
            pages = entry.get("pages", None)
            total_editions = entry.get("total_editions", None)

            # Normalize culinary_focus to string
            culinary_focus_str = ", ".join(culinary_focus) if isinstance(
                culinary_focus, list) else str(culinary_focus)

            # Process edition info
            edition_info = entry.get("edition_info", [])
            if not edition_info or not isinstance(edition_info, list):
                edition_info = []

            # Create edition summary
            edition_summaries = []
            for edition in edition_info:
                # Extract edition data
                edition_number = edition.get("edition_number", None)
                year = edition.get("year", None)

                # Extract location data
                location = edition.get("location", {})
                country = location.get("country", None) if location else None
                city = location.get("city", None) if location else None

                # Get contributors
                contributors = edition.get("contributors", [])
                contributors_str = ", ".join(
                    contributors) if contributors and isinstance(contributors,
                                                                 list) else ""

                # Get other edition details
                language = edition.get("language", None)
                short_note = edition.get("short_note", None)
                edition_category = edition.get("edition_category", None)

                # Format edition summary
                summary_parts = []
                if edition_number is not None:
                    summary_parts.append(f"Edition: {edition_number}")
                if year is not None:
                    summary_parts.append(f"Year: {year}")
                if city or country:
                    location_str = f"{city or 'Unknown'}, {country or 'Unknown'}"
                    summary_parts.append(f"Location: {location_str}")
                if language:
                    summary_parts.append(f"Language: {language}")
                if contributors_str:
                    summary_parts.append(f"Contributors: {contributors_str}")
                if edition_category:
                    summary_parts.append(f"Category: {edition_category}")
                if short_note:
                    summary_parts.append(f"Note: {short_note}")

                edition_summaries.append(" | ".join(summary_parts))

            # Create a row for the main entry
            main_row = {
                "full_title": full_title,
                "short_title": short_title,
                "culinary_focus": culinary_focus_str,
                "format": book_format,
                "pages": pages,
                "total_editions": total_editions,
                "edition_info_summary": " || ".join(
                    edition_summaries) if edition_summaries else "",
            }

            # Add the main row
            rows.append(main_row)

            for idx, edition in enumerate(edition_info, 1):
                edition_row = {**main_row}  # Copy the main entry data
                edition_row["edition_number"] = edition.get("edition_number")
                edition_row["year"] = edition.get("year")
                edition_row["city"] = edition.get("location", {}).get("city")
                edition_row["country"] = edition.get("location", {}).get("country")
                edition_row["language"] = edition.get("language")
                edition_row["contributors"] = ", ".join(edition.get("contributors", []))
                edition_row["edition_category"] = edition.get("edition_category")
                edition_row["short_note"] = edition.get("short_note")
                rows.append(edition_row)

        # Create DataFrame from all rows
        df = pd.DataFrame(rows)

        return df

    def _convert_structured_summaries_to_df(self,
                                            entries: List[Any]) -> pd.DataFrame:
        df = pd.json_normalize(entries, sep='_')
        if "bullet_points" in df.columns:
            df["bullet_points_summary"] = df["bullet_points"].apply(
                lambda x: "; ".join(x) if isinstance(x, list) else x
            )
            df.drop(columns=["bullet_points"], inplace=True)
        if "keywords" in df.columns:
            df["keywords_summary"] = df["keywords"].apply(
                lambda x: "; ".join(x) if isinstance(x, list) else x
            )
            df.drop(columns=["keywords"], inplace=True)
        if "literature" in df.columns:
            df["literature_summary"] = df["literature"].apply(
                lambda x: "; ".join(x) if isinstance(x, list) else x
            )
            df.drop(columns=["literature"], inplace=True)
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
