# modules/operations/extraction/response_parser.py

"""
Response parsing and validation.
Separated from schema handlers for better modularity.
"""

import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parses and validates API responses."""

    def __init__(self, schema_name: str):
        """
        Initialize response parser.

        :param schema_name: Name of the schema
        """
        self.schema_name = schema_name

    def parse_response(self, response_str: str) -> Dict[str, Any]:
        """
        Parse response string into dictionary.

        :param response_str: Raw response string
        :return: Parsed response dictionary
        """
        try:
            return json.loads(response_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response for {self.schema_name}: {e}")
            return {"error": f"JSON decode error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error parsing response for {self.schema_name}: {e}")
            return {"error": str(e)}

    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate response structure.

        :param response: Parsed response dictionary
        :return: True if valid, False otherwise
        """
        if "error" in response:
            logger.warning(f"Response contains error: {response['error']}")
            return False
        
        # Basic validation - check for expected structure
        if not isinstance(response, dict):
            logger.warning(f"Response is not a dictionary: {type(response)}")
            return False
        
        return True

    def extract_entries(self, response: Dict[str, Any]) -> list:
        """
        Extract entries from response.

        :param response: Parsed response dictionary
        :return: List of entries
        """
        if not self.validate_response(response):
            return []
        
        # Handle standard schema structure
        if "entries" in response:
            entries = response.get("entries", [])
            if isinstance(entries, list):
                return entries
        
        logger.warning(f"No entries found in response for {self.schema_name}")
        return []
