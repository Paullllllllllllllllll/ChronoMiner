# modules/openai_utils.py

from pathlib import Path
import aiohttp
from typing import Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from modules.config_loader import ConfigLoader
from modules.logger import setup_logger
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    wait_random,
)
from modules.structured_outputs import build_structured_text_format
from modules.model_capabilities import detect_capabilities

logger = setup_logger(__name__)


# ---------- Exceptions for retry control ----------


class TransientOpenAIError(Exception):
    """Error category that is safe to retry (429/5xx/timeouts)."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class NonRetryableOpenAIError(Exception):
    """Error category that should not be retried (e.g., 4xx other than 429)."""


def _load_retry_policy() -> tuple[int, float, float, float]:
    """Load retry attempts and wait window from concurrency configuration if present."""
    try:
        cl = ConfigLoader()
        cl.load_configs()
        cc = cl.get_concurrency_config() or {}
        trans_cfg = (cc.get("concurrency", {}) or {}).get("transcription", {}) or {}
        retry_cfg = (trans_cfg.get("retry", {}) or {})
        attempts = int(retry_cfg.get("attempts", 5))
        wait_min = float(retry_cfg.get("wait_min_seconds", 4))
        wait_max = float(retry_cfg.get("wait_max_seconds", 60))
        jitter_max = float(retry_cfg.get("jitter_max_seconds", 1))
        if attempts <= 0:
            attempts = 1
        if wait_min < 0:
            wait_min = 0
        if wait_max < wait_min:
            wait_max = wait_min
        if jitter_max < 0:
            jitter_max = 0
        return attempts, wait_min, wait_max, jitter_max
    except Exception:
        return 5, 4.0, 60.0, 1.0


_RETRY_ATTEMPTS, _RETRY_WAIT_MIN, _RETRY_WAIT_MAX, _RETRY_JITTER_MAX = _load_retry_policy()
_WAIT_BASE = wait_exponential(multiplier=1, min=_RETRY_WAIT_MIN, max=_RETRY_WAIT_MAX) + wait_random(
    0, _RETRY_JITTER_MAX
)


def _wait_with_server_hint_factory(base_wait):
    """Respect server-provided Retry-After if present; otherwise use base wait."""

    def _wait(retry_state):
        try:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            ra = getattr(exc, "retry_after", None)
            if isinstance(ra, (int, float)) and ra and ra > 0:
                return ra
        except Exception:
            pass
        return base_wait(retry_state)

    return _wait


class OpenAIExtractor:
    """
    A wrapper for interacting with the OpenAI Responses API for structured data
    extraction tasks (text-only in this repository).
    """
    def __init__(self, api_key: str, prompt_path: Path, model: str) -> None:
        if not model:
            raise ValueError("Model must be specified.")
        self.api_key: str = api_key
        self.model: str = model
        # Responses API endpoint
        self.endpoint: str = "https://api.openai.com/v1/responses"

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
        self.concurrency_config: Dict[str, Any] = config_loader.get_concurrency_config()
        tm: Dict[str, Any] = self.model_config["transcription_model"]
        # Token budget for Responses API
        self.max_output_tokens: int = int(tm["max_output_tokens"])
        # Classic sampler controls (applied only when supported)
        self.temperature: float = float(tm.get("temperature", 0.0))
        self.top_p: float = float(tm.get("top_p", 1.0))
        self.presence_penalty: float = float(tm.get("presence_penalty", 0.0))
        self.frequency_penalty: float = float(tm.get("frequency_penalty", 0.0))
        # Reasoning / text controls (used for GPT-5 family when supported)
        self.reasoning: Dict[str, Any] = tm.get("reasoning", {"effort": "medium"})
        self.text_params: Dict[str, Any] = tm.get("text", {"verbosity": "medium"})
        # Optional service tier from concurrency config
        try:
            trans_cfg = (self.concurrency_config.get("concurrency", {}) or {}).get("transcription", {}) or {}
        except Exception:
            trans_cfg = {}

        raw_tier = trans_cfg.get("service_tier")
        service_tier_normalized: Optional[str]
        if raw_tier is None:
            service_tier_normalized = None
        else:
            tier_str = str(raw_tier).lower().strip()
            if tier_str in {"auto", "default", "flex", "priority"}:
                service_tier_normalized = tier_str
            else:
                logger.warning("Ignoring unsupported service_tier value '%s' in concurrency_config.yaml", raw_tier)
                service_tier_normalized = None
        self.service_tier: Optional[str] = service_tier_normalized

        # Capabilities gating
        self.caps = detect_capabilities(self.model)

        # Configure aiohttp timeouts and connector pool based on concurrency settings
        try:
            conn_limit = int(trans_cfg.get("concurrency_limit", 100))
            if conn_limit <= 0:
                conn_limit = 100
        except Exception:
            conn_limit = 100

        default_timeouts = {
            "total": 900.0,
            "connect": 120.0,
            "sock_connect": 120.0,
            "sock_read": 600.0,
        }
        if self.service_tier == "flex":
            default_timeouts = {
                "total": 1800.0,
                "connect": 180.0,
                "sock_connect": 180.0,
                "sock_read": 1200.0,
            }
        elif self.service_tier == "priority":
            default_timeouts = {
                "total": 600.0,
                "connect": 90.0,
                "sock_connect": 90.0,
                "sock_read": 420.0,
            }

        timeout_overrides = trans_cfg.get("timeouts", {}) if isinstance(trans_cfg.get("timeouts"), dict) else {}
        total_timeout = float(timeout_overrides.get("total", default_timeouts["total"]))
        connect_timeout = float(timeout_overrides.get("connect", default_timeouts["connect"]))
        sock_connect_timeout = float(timeout_overrides.get("sock_connect", default_timeouts["sock_connect"]))
        sock_read_timeout = float(timeout_overrides.get("sock_read", default_timeouts["sock_read"]))

        client_timeout = aiohttp.ClientTimeout(
            total=total_timeout,
            connect=connect_timeout,
            sock_connect=sock_connect_timeout,
            sock_read=sock_read_timeout,
        )
        connector = aiohttp.TCPConnector(limit=conn_limit, limit_per_host=conn_limit)
        self.session: aiohttp.ClientSession = aiohttp.ClientSession(timeout=client_timeout, connector=connector)

    async def close(self) -> None:
        """
        Close the aiohttp session.
        """
        if self.session and not self.session.closed:
            await self.session.close()


def _collect_output_text(data: Dict[str, Any]) -> str:
    """
    Normalize Responses API output into a single text string.

    Prefers 'output_text' when present; otherwise concatenates message text parts
    from the 'output' list.
    """
    try:
        if isinstance(data, dict) and isinstance(data.get("output_text"), str):
            return data["output_text"].strip()
        parts: list[str] = []
        output = data.get("output") if isinstance(data, dict) else None
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and item.get("type") == "message":
                    for c in item.get("content", []):
                        t = c.get("text") if isinstance(c, dict) else None
                        if isinstance(t, str):
                            parts.append(t)
        return "".join(parts).strip()
    except Exception:
        return ""


async def _post_with_handling(session: aiohttp.ClientSession, endpoint: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    async with session.post(endpoint, headers=headers, json=payload) as resp:
        if resp.status != 200:
            error_text = await resp.text()
            retry_after_val: Optional[float] = None
            if resp.status == 429 or 500 <= resp.status < 600:
                # Respect Retry-After when provided by the server
                ra_hdr = resp.headers.get("Retry-After")
                if ra_hdr is not None:
                    try:
                        retry_after_val = float(ra_hdr)
                    except Exception:
                        try:
                            dt = parsedate_to_datetime(ra_hdr)
                            if dt is not None:
                                delta = (dt - datetime.now(timezone.utc)).total_seconds()
                                if delta and delta > 0:
                                    retry_after_val = delta
                        except Exception:
                            retry_after_val = None
                logger.warning("Transient OpenAI error (%s): %s", resp.status, error_text)
                raise TransientOpenAIError(f"{resp.status}: {error_text}", retry_after=retry_after_val)
            logger.error("Non-retryable OpenAI error (%s): %s", resp.status, error_text)
            raise NonRetryableOpenAIError(f"{resp.status}: {error_text}")
        return await resp.json()


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


@retry(
    wait=_wait_with_server_hint_factory(_WAIT_BASE),
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    retry=(
        retry_if_exception_type(TransientOpenAIError)
        | retry_if_exception_type(aiohttp.ClientError)
        | retry_if_exception_type(asyncio.TimeoutError)
    ),
)
async def process_text_chunk(
    text_chunk: str,
    extractor: OpenAIExtractor,
    system_message: Optional[str] = None,
    json_schema: Optional[dict] = None
) -> str:
    """
    Process a text chunk by sending a request to the OpenAI Responses API.

    :param text_chunk: The text to process.
    :param extractor: An instance of OpenAIExtractor.
    :param system_message: Optional system message.
    :param json_schema: Optional JSON schema for response formatting.
    :return: Processed text response.
    :raises Exception: If the API call fails.
    """
    if system_message is None:
        system_message = ""
    # Build typed input for Responses API
    input_messages = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": text_chunk}],
        },
    ]

    payload: Dict[str, Any] = {
        "model": extractor.model,
        "max_output_tokens": extractor.max_output_tokens,
        "input": input_messages,
    }
    if getattr(extractor, "service_tier", None):
        payload["service_tier"] = extractor.service_tier

    # Structured outputs (text.format) when supported
    if json_schema and extractor.caps.supports_structured_outputs:
        fmt = build_structured_text_format(json_schema, "TranscriptionSchema", True)
        if fmt is not None:
            payload.setdefault("text", {})
            payload["text"]["format"] = fmt

    # GPT-5 public controls when applicable
    if extractor.caps.supports_reasoning_effort:
        payload["reasoning"] = extractor.reasoning
        if (
            isinstance(extractor.text_params, dict)
            and extractor.text_params.get("verbosity") is not None
        ):
            payload.setdefault("text", {})["verbosity"] = extractor.text_params["verbosity"]

    # Sampler controls only for non-reasoning families
    if extractor.caps.supports_sampler_controls:
        payload["temperature"] = extractor.temperature
        payload["top_p"] = extractor.top_p
        # Only include penalties if non-zero to keep payload tidy
        if extractor.frequency_penalty:
            payload["frequency_penalty"] = extractor.frequency_penalty
        if extractor.presence_penalty:
            payload["presence_penalty"] = extractor.presence_penalty

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {extractor.api_key}",
        "Content-Type": "application/json",
    }
    data: Dict[str, Any] = await _post_with_handling(extractor.session, extractor.endpoint, headers, payload)
    return _collect_output_text(data)
