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

    def convert_to_csv(self, json_file: Path, output_csv: Path) -> None:
        entries: List[Any] = self.extract_entries(json_file)
        if not entries:
            print("No entries found for CSV conversion.")
            return

        converters = {
            "bibliographicentries": self._convert_bibliographic_entries_to_df,
            "recipes": self._convert_recipes_to_df,
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

    def _convert_bibliographic_entries_to_df(self, entries: List[Any]) -> pd.DataFrame:
        df = pd.json_normalize(entries, sep='_')
        for col in ['authors', 'roles', 'culinary_focus']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        if 'edition_info' in df.columns:
            def summarize_editions(editions: Any) -> str:
                if not editions or not isinstance(editions, list):
                    return ""
                summaries = []
                for edition in editions:
                    year = edition.get("year", "Unknown")
                    edition_number = edition.get("edition_number", "Unknown")
                    location = edition.get("location", {})
                    city = location.get("city", "Unknown")
                    country = location.get("country", "Unknown")
                    roles = edition.get("roles", [])
                    roles_str = ', '.join(roles) if isinstance(roles, list) else str(roles)
                    short_note = edition.get("short_note", "")
                    edition_category = edition.get("edition_category", "")
                    language = edition.get("language", "")
                    original_language = edition.get("original_language", "")
                    translated_from = edition.get("translated_from", "")
                    summary = (
                        f"Year: {year}, Edition: {edition_number}, "
                        f"Location: {city}, {country}, Roles: {roles_str}, "
                        f"Note: {short_note}, Category: {edition_category}, "
                        f"Language: {language}, Orig Lang: {original_language}, "
                        f"Translated From: {translated_from}"
                    )
                    summaries.append(summary)
                return " | ".join(summaries)
            df["edition_info_summary"] = df["edition_info"].apply(summarize_editions)
            df.drop(columns=["edition_info"], inplace=True)
        return df
    
    def _convert_structured_summaries_to_df(self, entries: List[Any]) -> pd.DataFrame:
        df = pd.json_normalize(entries, sep='_')
        if "bullet_points" in df.columns:
            df["bullet_points_summary"] = df["bullet_points"].apply(
                lambda x: "; ".join(x) if isinstance(x, list) else x
            )
            df.drop(columns=["bullet_points"], inplace=True)
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
