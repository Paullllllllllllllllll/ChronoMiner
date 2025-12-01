# modules/operations/extraction/payload_builder.py

"""
Payload construction for API requests.
Separated from schema handlers for better modularity.
"""

import logging
from typing import Any, Dict, Optional

from modules.llm.model_capabilities import detect_capabilities

logger = logging.getLogger(__name__)


def _build_structured_text_format(
    schema_obj: Dict[str, Any],
    default_name: str = "TranscriptionSchema",
    default_strict: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Build the Responses API `text.format` object for Structured Outputs.
    
    Args:
        schema_obj: Schema object with optional "name", "schema", "strict" keys,
                    or a bare JSON Schema dict.
        default_name: Default schema name if not provided.
        default_strict: Default strict mode setting.
    
    Returns:
        Dict with shape {"type": "json_schema", "name": ..., "schema": ..., "strict": ...}
        or None if the provided schema is not usable.
    """
    if not isinstance(schema_obj, dict) or not schema_obj:
        return None
    
    # Unwrap schema: accept either wrapper dict or bare JSON Schema
    if "schema" in schema_obj and isinstance(schema_obj["schema"], dict):
        name = schema_obj.get("name") or default_name
        schema = schema_obj.get("schema") or {}
        strict = bool(schema_obj.get("strict", default_strict))
    else:
        name = default_name
        schema = schema_obj
        strict = default_strict
    
    if not schema:
        return None
    
    return {
        "type": "json_schema",
        "name": str(name),
        "schema": schema,
        "strict": strict,
    }


class PayloadBuilder:
    """Builds API request payloads for the Responses API."""

    def __init__(self, schema_name: str):
        """
        Initialize payload builder.

        :param schema_name: Name of the schema
        """
        self.schema_name = schema_name

    def build_payload(
        self,
        text_chunk: str,
        dev_message: str,
        model_config: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build a complete API request payload.

        :param text_chunk: Text chunk to process
        :param dev_message: Developer/system message
        :param model_config: Model configuration
        :param schema: JSON schema
        :return: Complete request payload
        """
        user_message = f"Input text:\n{text_chunk}"
        
        json_schema_payload = self._build_json_schema_payload(
            dev_message, model_config, schema
        )
        
        model_cfg = model_config.get("transcription_model", {})
        model_name = model_cfg.get("name")
        
        # Detect model capabilities
        caps = detect_capabilities(model_name) if model_name else None
        
        # Build structured format
        fmt = _build_structured_text_format(json_schema_payload, self.schema_name, True)
        
        body = {
            "model": model_name,
            "max_output_tokens": model_cfg.get("max_output_tokens", 4096),
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": dev_message}]
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_message}]
                }
            ]
        }
        
        if fmt is not None:
            body.setdefault("text", {})["format"] = fmt
        
        # Add sampler controls only for non-reasoning families
        if caps and caps.supports_sampler_controls:
            for k in ("temperature", "top_p"):
                if k in model_cfg and model_cfg[k] is not None:
                    body[k] = model_cfg[k]
            # Only include penalties if non-zero
            if model_cfg.get("frequency_penalty"):
                body["frequency_penalty"] = model_cfg["frequency_penalty"]
            if model_cfg.get("presence_penalty"):
                body["presence_penalty"] = model_cfg["presence_penalty"]

        request_obj = {
            "custom_id": None,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        return request_obj

    def _build_json_schema_payload(
        self,
        dev_message: str,
        model_config: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build JSON schema payload.

        :param dev_message: Developer message
        :param model_config: Model configuration
        :param schema: JSON schema
        :return: JSON schema payload
        """
        return {
            "name": self.schema_name,
            "schema": schema,
            "strict": True
        }
