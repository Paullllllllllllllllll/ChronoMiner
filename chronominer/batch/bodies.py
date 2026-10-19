"""Request bodies of batch extraction, built by the shared request builder.

:func:`build_plan` reads the run-invariant parts once: the system prompts of
text and image chunks with the schema name, the schema and the context
injected, the enforced schema, the response format, the image detail, the
request plan and the context image. :meth:`BodyPlan.body` then turns each
:class:`~chronominer.common.batch.BatchRequest` into the provider-native body
the shared backends wrap, through
:func:`~chronominer.common.provider_requests.batch_body`: Responses API
bodies for OpenAI, Messages API parameters for Anthropic and
``InlinedRequest`` bodies for Google. A request with image data is an image
chunk; any other request sends its text.

The bodies carry what the synchronous calls send, from the same
:class:`~chronominer.common.provider_requests.RequestPlan`
(:func:`~chronominer.llm.options.request_plan`): output limit, reasoning
effort or thinking, samplers, verbosity, service tier, prompt caching and,
for a Gemini page image, the media resolution its detail maps to.
``schema`` enforces the schema under
:data:`~chronominer.llm.prompts.EXTRACTION_SCHEMA_NAME` (OpenAI
``text.format``, Anthropic ``output_config.format`` on the schema the
synchronous request sends, Google ``response_json_schema``), ``json`` sends
it only in the prompt. A text chunk is sent as the user message; an image
chunk follows :data:`~chronominer.llm.prompts.IMAGE_INSTRUCTION`. The
context image is sent to OpenAI only, as the synchronous path sends it:
preprocessed like a page, as JPEG with the model's detail, first under
:data:`~chronominer.llm.prompts.CONTEXT_IMAGE_LABEL`. The format follows the
synchronous rules: without ``--response-format`` a model with structured
output enforces the schema and any other model gets prompted JSON, as does an
Anthropic schema with more union parameters than Anthropic compiles.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronominer.common.batch import BatchRequest, supports_batch
from chronominer.common.provider_requests import Part, RequestPlan, batch_body
from chronominer.common.structured import ResponseFormat, fit_response_format
from chronominer.extract.extraction import anthropic_schema
from chronominer.extract.images import load_image, resolve_image_settings
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
from chronominer.settings import Settings
from chronominer.spec import ImageSpec, ModelSpec

__all__ = [
    "CONTEXT_IMAGE_INSTRUCTION",
    "CONTEXT_IMAGE_PROVIDERS",
    "USER_INSTRUCTION",
    "BodyPlan",
    "build_plan",
    "context_image_payload",
]

logger = logging.getLogger(__name__)

USER_INSTRUCTION = IMAGE_INSTRUCTION
CONTEXT_IMAGE_INSTRUCTION = CONTEXT_IMAGE_LABEL
CONTEXT_IMAGE_PROVIDERS = frozenset({"openai"})
"""Batch providers that receive the context image, as in synchronous calls."""

_CHUNK_KINDS: tuple[ChunkKind, ...] = ("text", "image")


@dataclass(frozen=True)
class BodyPlan:
    """The run-invariant inputs of every request body of one submission.

    *prompts* holds the system prompt of each chunk kind; *schema* is the
    enforced schema (``{name, schema, strict}``), None under prompted JSON;
    *detail* is the image detail of requests that name none (the OpenAI
    detail, or the Gemini media resolution it maps to); *context_image*
    the base64 data, MIME type and detail of the context image.
    """

    plan: RequestPlan
    prompts: Mapping[str, str]
    response_format: ResponseFormat
    schema: Mapping[str, Any] | None = None
    detail: str | None = None
    context_image: tuple[str, str, str | None] | None = None

    @property
    def provider(self) -> str:
        """The provider of the submission."""
        return self.plan.provider

    @property
    def model(self) -> str:
        """The model every body names."""
        return self.plan.model

    def system_prompt(self, request: BatchRequest) -> str:
        """Return the system prompt of *request*'s chunk kind."""
        return self.prompts["image" if request.is_visual else "text"]

    def _enforced(self) -> Mapping[str, Any] | None:
        if self.response_format != "schema" or self.schema is None:
            return None
        if self.plan.provider == "anthropic":
            return anthropic_schema(self.schema)
        return self.schema

    def body(self, request: BatchRequest) -> dict[str, Any]:
        """Return the provider-native body of *request*.

        Raises:
            ValueError: The request carries neither image data nor text.
        """
        if not request.is_visual and not request.text:
            raise ValueError(f"Request {request.custom_id} has no text or image")
        parts: list[Part] = []
        if self.context_image is not None and self.plan.provider == "openai":
            data, mime, context_detail = self.context_image
            parts.append(Part(text=CONTEXT_IMAGE_INSTRUCTION))
            parts.append(Part(data=data, mime=mime, detail=context_detail))
        if request.is_visual:
            parts.append(Part(text=USER_INSTRUCTION))
            parts.append(
                Part(
                    data=request.image_base64,
                    mime=request.mime_type,
                    detail=request.image_detail or self.detail,
                )
            )
        else:
            parts.append(Part(text=request.text))
        return batch_body(
            self.plan, self.system_prompt(request), parts, self._enforced()
        )


def context_image_payload(
    model: ModelSpec,
    path: Path | None,
    images: ImageSpec | None = None,
    original_max_side_px: int | None = None,
) -> tuple[str, str, str | None] | None:
    """Return the base64 data, MIME type and detail of a context image.

    The image is preprocessed like a page and encoded as JPEG with the
    model's detail, for the providers of :data:`CONTEXT_IMAGE_PROVIDERS`;
    the others get none, with a warning. *original_max_side_px* is the
    ``images`` setting of that name.

    Raises:
        ValueError: The image cannot be loaded, or an image setting is invalid.
    """
    if path is None or not path.exists():
        return None
    if model.provider not in CONTEXT_IMAGE_PROVIDERS:
        logger.warning(
            "Context images are not sent to provider '%s'; skipping %s.",
            model.provider,
            path.name,
        )
        return None
    cfg, model_type, _section = resolve_image_settings(
        model.provider, model.model, images, original_max_side_px
    )
    page = load_image(path, 0, {**cfg, "payload_format": "jpeg"}, model_type)
    if page.render_error is not None:
        raise ValueError(f"Context image {path.name}: {page.render_error}")
    return page.base64, page.mime_type, cfg["resolved_detail"]


def build_plan(
    model: ModelSpec,
    settings: Settings,
    *,
    schema: Mapping[str, Any],
    context: str | None = None,
    context_image: Path | None = None,
    detail: str | None = None,
    images: ImageSpec | None = None,
) -> BodyPlan:
    """Return the body plan of a submission for *model*.

    *schema* is the schema file's JSON; *context* is the extraction
    context; *images* are the run's image options, which size the context
    image.

    Raises:
        ValueError: The provider has no batch API, the schema is empty or
            the context image cannot be loaded.
        OSError: A prompt template cannot be read.
    """
    if not supports_batch(model.provider):
        raise ValueError(f"Provider '{model.provider}' has no batch API")
    caps = model_capabilities(model, settings)
    enforced = extraction_schema(schema)
    response_format = fit_response_format(
        model.provider, response_format_for(model, caps), enforced
    )
    prompts: dict[str, str] = {
        kind: extraction_prompt(kind, schema, context=context) for kind in _CHUNK_KINDS
    }
    return BodyPlan(
        plan=request_plan(model, settings, caps),
        prompts=prompts,
        response_format=response_format,
        schema=enforced if response_format == "schema" else None,
        detail=detail,
        context_image=context_image_payload(
            model, context_image, images, settings.images.original_max_side_px
        ),
    )
