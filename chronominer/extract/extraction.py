"""Chunk extraction: one text chunk or page image in, the response dict out.

:class:`ChunkExtractor` builds each provider's messages (the system prompt of
the chunk kind with the item's context, an optional context image, then the
chunk text or the page image with its detail) and sends them through
:class:`~chronominer.llm.caller.ModelCaller` in the run's response format:
``schema`` has the provider enforce the selected schema, ``json`` puts the
schema in the prompt and checks the answer. The answer becomes the response
dict the unit pipeline records::

    {
        "output_text": str,               # the answer as received
        "usage": {"input_tokens": int, "output_tokens": int,
                  "total_tokens": int, "cached_tokens": int,
                  "reasoning_tokens": int},
        "metadata": {...},                # the provider's response metadata
        "error": str,                     # only for a failed chunk
        "error_type": str,                # a stopped answer's outcome kind
    }

A failed chunk returns the dict with ``error`` set instead of raising; an
answer the provider stopped (a content filter, a refusal, an answer cut at
the output-token limit) fails the chunk. An
answer that fails JSON validation keeps its text: its entries are parsed as
far as they can be (:mod:`~chronominer.extract.response_parsing`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from chronominer.common.provider_requests import (
    Part,
    message_content,
    system_content,
)
from chronominer.common.responses import unwrap_raw_response
from chronominer.common.structured import (
    ResponseFormat,
    StructuredRequest,
    StructuredResult,
    fit_response_format,
)
from chronominer.common.usage import Usage, usage_from_response
from chronominer.extract.images import load_image, resolve_image_settings
from chronominer.extract.response_parsing import INVALID_JSON, NO_OBJECT, parse_answer
from chronominer.llm.caller import CallEnv
from chronominer.llm.options import (
    model_capabilities,
    request_plan,
    response_format_for,
)
from chronominer.llm.prompts import (
    CONTEXT_IMAGE_LABEL,
    IMAGE_INSTRUCTION,
    ChunkKind,
    extraction_prompt,
    extraction_schema,
)
from chronominer.spec import ImageSpec, ModelSpec

__all__ = [
    "CONTEXT_IMAGE_PROVIDERS",
    "ROLE",
    "ChunkExtractor",
    "anthropic_schema",
    "error_response",
    "usage_record",
]

logger = logging.getLogger(__name__)

ROLE = "extract"
"""The usage role of extraction calls."""

CONTEXT_IMAGE_PROVIDERS = frozenset({"openai", "openrouter"})
"""Providers that receive the context image; the others skip it."""

ANTHROPIC_SCHEMA_DESCRIPTION = "Return a JSON object that conforms to this schema."

# Per provider: usage block, input, output and total token keys in
# ``response_metadata`` (total is empty when the provider reports none).
_USAGE_KEYS: Mapping[str, tuple[str, str, str, str]] = {
    "anthropic": ("usage", "input_tokens", "output_tokens", ""),
    "google": (
        "usage_metadata",
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
    ),
}
_OPENAI_USAGE_KEYS = (
    "token_usage",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def error_response(message: str) -> dict[str, Any]:
    """Return the response dict of a chunk that failed before an answer."""
    return {
        "output_text": "",
        "usage": Usage().as_dict(),
        "error": message,
    }


def _int(value: Any) -> int:
    return int(value or 0)


def usage_record(message: Any, provider: str) -> dict[str, int]:
    """Return the token counts of an answer message for the response dict.

    The counts have the shape of :class:`~chronominer.common.usage.Usage`,
    as every stored usage record: LangChain's ``usage_metadata`` first, else
    the provider's own usage block in ``response_metadata``. Cache reads and
    writes count as input; ``cached_tokens`` are the reads.
    """
    usage = usage_from_response(message)
    if usage is None:
        usage = _provider_usage(message, provider)
    return usage.as_dict()


def _provider_usage(message: Any, provider: str) -> Usage:
    """Return the usage of the provider's block in ``response_metadata``."""
    block, input_key, output_key, total_key = _USAGE_KEYS.get(
        provider, _OPENAI_USAGE_KEYS
    )
    metadata = getattr(message, "response_metadata", None)
    usage = metadata.get(block) if isinstance(metadata, dict) else None
    if not isinstance(usage, dict):
        return Usage()
    input_tokens = _int(usage.get(input_key))
    output_tokens = _int(usage.get(output_key))
    total_tokens = _int(usage.get(total_key)) if total_key else 0
    total_tokens = total_tokens or input_tokens + output_tokens
    read = _int(usage.get("cache_read_input_tokens"))
    written = _int(usage.get("cache_creation_input_tokens"))
    cached = read
    details = usage.get("prompt_tokens_details")
    if not cached and isinstance(details, dict):
        cached = _int(details.get("cached_tokens"))
    return Usage(
        input_tokens=input_tokens + read + written,
        output_tokens=output_tokens,
        total_tokens=total_tokens + read + written,
        cached_tokens=cached,
    )


def anthropic_schema(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return *spec* with the title and description Anthropic's schema takes."""
    body = spec.get("schema", spec)
    inner = dict(body) if isinstance(body, Mapping) else {}
    inner.setdefault("title", str(spec.get("name") or "Response"))
    inner.setdefault("description", ANTHROPIC_SCHEMA_DESCRIPTION)
    return {**{k: v for k, v in spec.items() if k != "schema"}, "schema": inner}


def _page_detail(model: ModelSpec, images: ImageSpec | None) -> str | None:
    """Return the detail page images are sized with, None when unresolvable."""
    try:
        cfg, _model_type, _section = resolve_image_settings(
            model.provider, model.model, images
        )
    except ValueError:
        return None
    return str(cfg["resolved_detail"])


class ChunkExtractor:
    """Extract the entries of text chunks and page images with the run's model.

    *schema* is the schema file's JSON and *schema_name* its name; the system
    prompt of each chunk kind carries both and the context text. *images* are
    the run's image options, which size the context image; *context* is the
    context text until :meth:`update_context`.

    Raises:
        FileNotFoundError: A prompt template is missing.
        ValueError: The schema holds no JSON schema.
        OSError: The key variable is unset.
    """

    def __init__(
        self,
        model: ModelSpec,
        env: CallEnv,
        *,
        schema: Mapping[str, Any],
        schema_name: str,
        images: ImageSpec | None = None,
        context: str | None = None,
        role: str = ROLE,
    ) -> None:
        self.provider = model.provider
        self.model_name = model.model
        self.capabilities = model_capabilities(model, env.settings)
        self.plan = request_plan(model, env.settings, self.capabilities)
        self.request_schema = extraction_schema(schema)
        self.response_format: ResponseFormat = fit_response_format(
            model.provider,
            response_format_for(model, self.capabilities),
            self.request_schema,
        )
        self.settings = env.settings
        self.schema = dict(schema)
        self.schema_name = schema_name
        self.images = images
        self.page_detail = _page_detail(model, images)
        self.context = context
        self._prompts: dict[ChunkKind, str] = {}
        for kind in ("text", "image"):
            self._prompts[kind] = self._prompt(kind)
        self._context_image: tuple[str, str, str] | None = None
        self._skipped_context_image = False
        self.caller = env.caller(role, model)

    def _prompt(self, kind: ChunkKind) -> str:
        return extraction_prompt(kind, self.schema, context=self.context)

    def system_prompt(self, kind: ChunkKind) -> str:
        """Return the system prompt of a *kind* chunk."""
        return self._prompts[kind]

    def update_context(self, context: str | None) -> None:
        """Put *context* into the system prompts; None removes the section."""
        if context == self.context:
            return
        self.context = context
        for kind in ("text", "image"):
            self._prompts[kind] = self._prompt(kind)

    def update_context_image(self, path: Path | None) -> None:
        """Send *path* as the context image of later chunks; None sends none.

        The image is preprocessed like a page and sent as JPEG, to OpenAI and
        OpenRouter only.
        """
        self._context_image = None
        if path is None or not path.exists():
            return
        if self.provider not in CONTEXT_IMAGE_PROVIDERS:
            if not self._skipped_context_image:
                logger.warning(
                    "Context images are not sent to provider '%s'; skipping %s.",
                    self.provider,
                    path.name,
                )
                self._skipped_context_image = True
            return
        cfg, model_type, _section = resolve_image_settings(
            self.provider,
            self.model_name,
            self.images,
            self.settings.images.original_max_side_px,
        )
        page = load_image(path, 0, {**cfg, "payload_format": "jpeg"}, model_type)
        if page.render_error is not None:
            raise ValueError(f"Context image {path.name}: {page.render_error}")
        self._context_image = (page.base64, page.mime_type, cfg["resolved_detail"])
        logger.info("Context image loaded: %s", path.name)

    def _system_message(self, kind: ChunkKind) -> SystemMessage:
        return SystemMessage(
            content=system_content(self.plan, self.system_prompt(kind))
        )

    def image_block(
        self, data: str, mime_type: str, detail: str | None
    ) -> dict[str, Any]:
        """Return the provider's content block of one image.

        Anthropic takes the base64 source; Gemini a data block with the media
        resolution its detail maps to; OpenAI and OpenRouter a data URL with
        the detail the model takes
        (:func:`~chronominer.common.provider_requests.message_content`).
        """
        (block,) = message_content(
            self.plan, [Part(data=data, mime=mime_type, detail=detail)]
        )
        assert isinstance(block, dict)
        return block

    def _context_blocks(self) -> list[str | dict[str, Any]]:
        if self._context_image is None:
            return []
        data, mime, detail = self._context_image
        return [
            {"type": "text", "text": CONTEXT_IMAGE_LABEL},
            self.image_block(data, mime, detail),
        ]

    def text_messages(self, text: str) -> list[Any]:
        """Return the system and user messages of one text chunk.

        The chunk is the user message; a context image and its label come
        before it.
        """
        context = self._context_blocks()
        if not context:
            return [self._system_message("text"), HumanMessage(content=text)]
        content = [*context, {"type": "text", "text": text}]
        return [self._system_message("text"), HumanMessage(content=content)]

    def image_messages(
        self, image_base64: str, mime_type: str, *, image_detail: str | None = None
    ) -> list[Any]:
        """Return the system and user messages of one page image.

        The context image and its label come first, then the page instruction
        and the page image.
        """
        content: list[str | dict[str, Any]] = self._context_blocks()
        content.append({"type": "text", "text": IMAGE_INSTRUCTION})
        detail = image_detail or self.page_detail
        content.append(self.image_block(image_base64, mime_type, detail))
        return [self._system_message("image"), HumanMessage(content=content)]

    def request(self) -> StructuredRequest:
        """Return the structured request of every chunk.

        ``schema`` checks only that the answer is a JSON object; ``json``
        also checks the schema's required keys. Invalid answers are asked
        again within ``retry.validation_attempts``.
        """
        schema: Mapping[str, Any] = self.request_schema
        if self.provider == "anthropic":
            schema = anthropic_schema(schema)
        policy = self.settings.retry.policy()
        return StructuredRequest(
            provider=self.provider,
            response_format=self.response_format,
            schema=schema,
            required=() if self.response_format != "json" else None,
            validation_retries=max(0, self.settings.retry.validation_attempts - 1),
            validation_delay=lambda retry: policy.backoff(retry - 1, "validation"),
        )

    async def _call(
        self, messages: list[Any], label: str, *, min_input_tokens: int = 0
    ) -> dict[str, Any]:
        try:
            result = await self.caller.call(
                messages,
                self.request(),
                label=label,
                min_input_tokens=min_input_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - a failed chunk returns an error dict
            logger.error(
                "Extraction failed for %s: %s - %s", label, type(exc).__name__, exc
            )
            return error_response(str(exc))
        return self._response(result)

    async def extract_text(
        self, text: str, *, label: str | None = None
    ) -> dict[str, Any]:
        """Extract one text chunk; a failure returns the dict with ``error``."""
        return await self._call(self.text_messages(text), label or "chunk")

    async def extract_image_from_base64(
        self,
        image_base64: str,
        mime_type: str,
        *,
        label: str | None = None,
        image_detail: str | None = None,
    ) -> dict[str, Any]:
        """Extract one page image; a failure returns the dict with ``error``.

        *image_detail* replaces the detail the pages are sized with in the
        image block.
        """
        if not self.capabilities.supports_vision:
            return error_response(
                f"Model {self.model_name} does not support vision/image inputs."
            )
        messages = self.image_messages(
            image_base64, mime_type, image_detail=image_detail
        )
        return await self._call(
            messages,
            label or "page",
            min_input_tokens=self.settings.retry.min_input_tokens,
        )

    def _response(self, result: StructuredResult) -> dict[str, Any]:
        message = unwrap_raw_response(result.response)
        response: dict[str, Any] = {
            "output_text": result.text,
            "usage": usage_record(message, self.provider),
        }
        metadata = getattr(message, "response_metadata", None)
        if isinstance(metadata, dict) and metadata:
            response["metadata"] = metadata
        if result.stopped:
            error = result.outcome.message
            logger.warning("Chunk answer rejected: %s", error)
            response["output_text"] = ""
            response["error"] = error
            response["error_type"] = result.outcome.kind
        elif result.not_object and parse_answer(result.text) is None:
            error = f"{NO_OBJECT}: {result.error}"
            logger.warning("Chunk answer rejected: %s", error)
            response["output_text"] = ""
            response["error"] = error
            response["error_type"] = INVALID_JSON
        elif result.error and result.text:
            logger.warning("Answer kept despite invalid JSON: %s", result.error)
        return response

    async def close(self) -> None:
        """Close the clients of the caller. Never raises."""
        await self.caller.aclose()
