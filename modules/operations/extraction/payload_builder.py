# modules/operations/extraction/payload_builder.py

"""
Payload construction for API requests.
Separated from schema handlers for better modularity.
"""

import logging
from typing import Dict, Any, Optional

from modules.llm.structured_outputs import build_structured_text_format
from modules.llm.model_capabilities import detect_capabilities

logger = logging.getLogger(__name__)


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
        fmt = build_structured_text_format(json_schema_payload, self.schema_name, True)
        
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
