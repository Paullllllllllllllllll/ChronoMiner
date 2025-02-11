# modules/openai_utils.py

from pathlib import Path
import aiohttp
from typing import Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from tenacity import retry, wait_exponential, stop_after_attempt

logger = setup_logger(__name__)


class OpenAIExtractor:
    """
    A wrapper for interacting with the OpenAI API for structured data
    extraction tasks.
    """
    def __init__(self, api_key: str, prompt_path: Path, model: str) -> None:
        if not model:
            raise ValueError("Model must be specified.")
        self.api_key: str = api_key
        self.model: str = model
        self.endpoint: str = "https://api.openai.com/v1/chat/completions"

        if self.model == "o3-mini":
            self.prompt_text: str = ""
        else:
            self.prompt_path: Path = prompt_path
            if not self.prompt_path.exists():
                logger.error(f"Prompt file not found: {self.prompt_path}")
                raise FileNotFoundError(f"Prompt file does not exist: {self.prompt_path}")
            try:
                with self.prompt_path.open('r', encoding='utf-8') as prompt_file:
                    self.prompt_text = prompt_file.read().strip()
            except Exception as e:
                logger.error(f"Failed to read prompt: {e}")
                raise

        config_loader = ConfigLoader()
        config_loader.load_configs()
        self.model_config: Dict[str, Any] = config_loader.get_model_config()
        self.temperature: float = self.model_config["extraction_model"]["temperature"]
        self.max_tokens: int = self.model_config["extraction_model"]["max_completion_tokens"]
        self.reasoning_effort: str = self.model_config["extraction_model"]["reasoning_effort"]
        self.session: aiohttp.ClientSession = aiohttp.ClientSession()

    async def close(self) -> None:
        """
        Close the aiohttp session.
        """
        if self.session and not self.session.closed:
            await self.session.close()


@asynccontextmanager
async def open_extractor(api_key: str, prompt_path: Path, model: str) -> AsyncGenerator[OpenAIExtractor, None]:
    """
    Asynchronous context manager for OpenAIExtractor.

    :param api_key: OpenAI API key.
    :param prompt_path: Path to the prompt file.
    :param model: Model name.
    :yield: An instance of OpenAIExtractor.
    """
    extractor = OpenAIExtractor(api_key, prompt_path, model)
    try:
        yield extractor
    finally:
        await extractor.close()


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
async def process_text_chunk(
    text_chunk: str,
    extractor: OpenAIExtractor,
    system_message: Optional[str] = None,
    json_schema: Optional[dict] = None
) -> str:
    """
    Process a text chunk by sending a request to the OpenAI API.

    :param text_chunk: The text to process.
    :param extractor: An instance of OpenAIExtractor.
    :param system_message: Optional system message.
    :param json_schema: Optional JSON schema for response formatting.
    :return: Processed text response.
    :raises Exception: If the API call fails.
    """
    if system_message is None:
        system_message = ""
    if not json_schema:
        json_schema_payload = {
            "name": "FallbackSchema",
            "schema": {
                "type": "object",
                "properties": {
                    "processed_text": {"type": "string"}
                },
                "required": ["processed_text"],
                "additionalProperties": False
            },
            "strict": True
        }
    else:
        name_val = json_schema.get("name")
        schema_val = json_schema.get("schema")
        json_schema_payload = {
            "name": name_val,
            "schema": schema_val,
            "strict": True
        }

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text_chunk}
    ]
    payload: Dict[str, Any] = {
        "model": extractor.model,
        "messages": messages,
        "max_completion_tokens": extractor.max_tokens,
        "reasoning_effort": extractor.reasoning_effort,
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema_payload
        }
    }

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {extractor.api_key}",
        "Content-Type": "application/json"
    }
    async with extractor.session.post(extractor.endpoint, headers=headers, json=payload) as response:
        if response.status != 200:
            error_text: str = await response.text()
            logger.error(f"OpenAI API error for text chunk: {error_text}")
            raise Exception(f"OpenAI API error: {error_text}")

        data: Dict[str, Any] = await response.json()
        return data["choices"][0]["message"]["content"].strip()
